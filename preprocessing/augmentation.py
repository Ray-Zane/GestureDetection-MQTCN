"""Deterministic augmentation for already-built continuous skeleton clips.

The feature builder stores semantically different values in one flat vector.  This
module deliberately discovers those values from ``feature_names`` instead of
depending on fixed column offsets.  Augmentation is applied before tensors are
created by the training dataset and never mutates the cache-backed input arrays.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


_POSE_PATTERN = re.compile(r"^local_pose\.joint_(\d+)\.([xyz])$")
_MOTION_PATTERN = re.compile(
    r"^local_motion_lag_(\d+)\.joint_(\d+)\.([xyz])$"
)
_WRIST_MOTION_PATTERN = re.compile(r"^global\.wrist_d([xy])_lag_(\d+)$")
_SCALE_MOTION_PATTERN = re.compile(r"^global\.delta_log_scale_lag_(\d+)$")


def _range_pair(name: str, value: Sequence[float], *, positive: bool) -> Tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly [minimum, maximum]")
    low, high = float(value[0]), float(value[1])
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{name} values must be finite")
    if low > high:
        raise ValueError(f"{name} minimum cannot exceed maximum")
    if positive and low <= 0.0:
        raise ValueError(f"{name} values must be positive")
    return low, high


@dataclass(frozen=True)
class SkeletonAugmentationConfig:
    """Configuration for :class:`SkeletonFeatureAugmenter`.

    ``coordinate_noise`` is the Gaussian standard deviation for local joint and
    global wrist coordinates.  Joint dropout samples one mask per joint trajectory
    (not independently per frame).  ``translation_range`` is sampled independently
    for x and y.  A speed greater than one plays the source clip faster.

    Missing-run length is encoded in the same normalized form as
    :class:`preprocessing.feature_builder.SkeletonFeatureBuilder`, hence
    ``missing_clip_frames`` must match the feature-cache configuration.
    """

    coordinate_noise: float = 0.0
    joint_dropout: float = 0.0
    frame_dropout: float = 0.0
    scale_range: Tuple[float, float] = (1.0, 1.0)
    translation_range: Tuple[float, float] = (0.0, 0.0)
    speed_range: Tuple[float, float] = (1.0, 1.0)
    max_hold_frames: int = 5
    missing_clip_frames: int = 30
    ignore_label: int = -100

    def __post_init__(self) -> None:
        noise = float(self.coordinate_noise)
        if not math.isfinite(noise) or noise < 0.0:
            raise ValueError("coordinate_noise must be finite and non-negative")
        for name in ("joint_dropout", "frame_dropout"):
            probability = float(getattr(self, name))
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        scale = _range_pair("scale_range", self.scale_range, positive=True)
        translation = _range_pair(
            "translation_range", self.translation_range, positive=False
        )
        speed = _range_pair("speed_range", self.speed_range, positive=True)
        if int(self.max_hold_frames) < 0:
            raise ValueError("max_hold_frames must be non-negative")
        if int(self.missing_clip_frames) <= 0:
            raise ValueError("missing_clip_frames must be positive")
        object.__setattr__(self, "coordinate_noise", noise)
        object.__setattr__(self, "joint_dropout", float(self.joint_dropout))
        object.__setattr__(self, "frame_dropout", float(self.frame_dropout))
        object.__setattr__(self, "scale_range", scale)
        object.__setattr__(self, "translation_range", translation)
        object.__setattr__(self, "speed_range", speed)
        object.__setattr__(self, "max_hold_frames", int(self.max_hold_frames))
        object.__setattr__(self, "missing_clip_frames", int(self.missing_clip_frames))
        object.__setattr__(self, "ignore_label", int(self.ignore_label))

    @property
    def enabled(self) -> bool:
        return bool(
            self.coordinate_noise > 0.0
            or self.joint_dropout > 0.0
            or self.frame_dropout > 0.0
            or self.scale_range != (1.0, 1.0)
            or self.translation_range != (0.0, 0.0)
            or self.speed_range != (1.0, 1.0)
        )


@dataclass(frozen=True)
class AugmentedClip:
    """Synchronized augmented arrays plus auditable sampled decisions."""

    features: np.ndarray
    frame_labels: np.ndarray
    start_targets: np.ndarray
    end_targets: np.ndarray
    supervision_mask: np.ndarray
    source_indices: np.ndarray
    dropped_frames: np.ndarray
    dropped_joints: np.ndarray
    speed_factor: float
    scale_factor: float
    translation_xy: Tuple[float, float]

    def as_training_dict(self) -> Mapping[str, np.ndarray]:
        """Return only fields consumed by the continuous training loop."""

        return {
            "features": self.features,
            "frame_labels": self.frame_labels,
            "start_targets": self.start_targets,
            "end_targets": self.end_targets,
            "supervision_mask": self.supervision_mask,
        }


@dataclass(frozen=True)
class _FeatureLayout:
    pose: Mapping[Tuple[int, str], int]
    motion: Mapping[int, Mapping[Tuple[int, str], int]]
    wrist: Mapping[str, int]
    wrist_motion: Mapping[int, Mapping[str, int]]
    log_scale: Optional[int]
    scale_motion: Mapping[int, int]
    quality_valid: int
    quality_missing: int
    dynamic_columns: Tuple[int, ...]

    @property
    def state_columns(self) -> Tuple[int, ...]:
        columns = list(self.pose.values()) + list(self.wrist.values())
        if self.log_scale is not None:
            columns.append(self.log_scale)
        return tuple(sorted(set(columns)))


def _parse_feature_layout(feature_names: Sequence[str]) -> _FeatureLayout:
    names = tuple(str(value) for value in feature_names)
    if not names:
        raise ValueError("feature_names cannot be empty")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")

    pose: Dict[Tuple[int, str], int] = {}
    motion: Dict[int, Dict[Tuple[int, str], int]] = {}
    wrist: Dict[str, int] = {}
    wrist_motion: Dict[int, Dict[str, int]] = {}
    scale_motion: Dict[int, int] = {}
    log_scale = None
    for column, name in enumerate(names):
        match = _POSE_PATTERN.match(name)
        if match:
            pose[(int(match.group(1)), match.group(2))] = column
            continue
        match = _MOTION_PATTERN.match(name)
        if match:
            lag, joint, axis = int(match.group(1)), int(match.group(2)), match.group(3)
            if lag <= 0:
                raise ValueError(f"motion lag must be positive in {name!r}")
            motion.setdefault(lag, {})[(joint, axis)] = column
            continue
        if name in {"global.wrist_x", "global.wrist_y"}:
            wrist[name[-1]] = column
            continue
        match = _WRIST_MOTION_PATTERN.match(name)
        if match:
            axis, lag = match.group(1), int(match.group(2))
            if lag <= 0:
                raise ValueError(f"motion lag must be positive in {name!r}")
            wrist_motion.setdefault(lag, {})[axis] = column
            continue
        if name == "global.log_scale":
            log_scale = column
            continue
        match = _SCALE_MOTION_PATTERN.match(name)
        if match:
            lag = int(match.group(1))
            if lag <= 0:
                raise ValueError(f"motion lag must be positive in {name!r}")
            scale_motion[lag] = column

    if not pose:
        raise ValueError("feature_names do not contain local_pose fields")
    for lag, columns in motion.items():
        missing = sorted(set(columns).difference(pose))
        if missing:
            raise ValueError(
                f"local_motion_lag_{lag} has no matching local_pose fields: {missing}"
            )
    for lag, columns in wrist_motion.items():
        missing = sorted(set(columns).difference(wrist))
        if missing:
            raise ValueError(
                f"global wrist lag {lag} has no matching wrist fields: {missing}"
            )
    try:
        quality_valid = names.index("quality.valid")
        quality_missing = names.index("quality.missing_run_length")
    except ValueError as error:
        raise ValueError(
            "feature_names must contain quality.valid and quality.missing_run_length"
        ) from error
    dynamic = tuple(
        index
        for index, name in enumerate(names)
        if name.startswith("local_pose.")
        or name.startswith("local_motion_")
        or name.startswith("global.")
    )
    return _FeatureLayout(
        pose=pose,
        motion={lag: dict(values) for lag, values in motion.items()},
        wrist=wrist,
        wrist_motion={lag: dict(values) for lag, values in wrist_motion.items()},
        log_scale=log_scale,
        scale_motion=scale_motion,
        quality_valid=quality_valid,
        quality_missing=quality_missing,
        dynamic_columns=dynamic,
    )


def _fixed_or_uniform(
    rng: np.random.Generator, interval: Tuple[float, float]
) -> float:
    low, high = interval
    return low if low == high else float(rng.uniform(low, high))


class SkeletonFeatureAugmenter:
    """Apply synchronized deterministic augmentation to a fixed-length clip.

    The call keeps the temporal length unchanged.  Temporal speed uses
    ``round(output_index * speed_factor)`` (clipped to the source extent), producing
    a monotonically non-decreasing source-index map.  Every dense target and the
    supervision mask use that exact same map.
    """

    _RANDOM_STREAMS = 6

    def __init__(
        self,
        feature_names: Sequence[str],
        config: Optional[SkeletonAugmentationConfig] = None,
    ) -> None:
        self.feature_names = tuple(str(value) for value in feature_names)
        self.config = config or SkeletonAugmentationConfig()
        self.layout = _parse_feature_layout(self.feature_names)
        if self.config.translation_range != (0.0, 0.0) and not {"x", "y"}.issubset(
            self.layout.wrist
        ):
            raise ValueError("translation augmentation requires global.wrist_x/y")

    def __call__(
        self,
        features: np.ndarray,
        frame_labels: np.ndarray,
        start_targets: np.ndarray,
        end_targets: np.ndarray,
        supervision_mask: np.ndarray,
        *,
        seed: int,
    ) -> AugmentedClip:
        return self.augment(
            features,
            frame_labels,
            start_targets,
            end_targets,
            supervision_mask,
            seed=seed,
        )

    def augment(
        self,
        features: np.ndarray,
        frame_labels: np.ndarray,
        start_targets: np.ndarray,
        end_targets: np.ndarray,
        supervision_mask: np.ndarray,
        *,
        seed: int,
    ) -> AugmentedClip:
        arrays = self._validate_inputs(
            features,
            frame_labels,
            start_targets,
            end_targets,
            supervision_mask,
        )
        feature_values, labels, starts, ends, supervision = arrays
        frames = int(feature_values.shape[0])
        identity = np.arange(frames, dtype=np.int64)
        if not self.config.enabled:
            return AugmentedClip(
                features=feature_values.copy(),
                frame_labels=labels.copy(),
                start_targets=starts.copy(),
                end_targets=ends.copy(),
                supervision_mask=supervision.copy(),
                source_indices=identity,
                dropped_frames=np.zeros(frames, dtype=np.bool_),
                dropped_joints=np.empty(0, dtype=np.int64),
                speed_factor=1.0,
                scale_factor=1.0,
                translation_xy=(0.0, 0.0),
            )

        integer_seed = int(seed)
        if integer_seed < 0:
            raise ValueError("seed must be non-negative")
        streams = [
            np.random.default_rng(child)
            for child in np.random.SeedSequence(integer_seed).spawn(
                self._RANDOM_STREAMS
            )
        ]
        speed = _fixed_or_uniform(streams[0], self.config.speed_range)
        scale = _fixed_or_uniform(streams[1], self.config.scale_range)
        translation = (
            _fixed_or_uniform(streams[2], self.config.translation_range),
            _fixed_or_uniform(streams[2], self.config.translation_range),
        )
        source_indices = np.floor(identity.astype(np.float64) * speed + 0.5).astype(
            np.int64
        )
        np.clip(source_indices, 0, max(0, frames - 1), out=source_indices)

        output = feature_values[source_indices].copy()
        labels = labels[source_indices].copy()
        starts = starts[source_indices].copy()
        ends = ends[source_indices].copy()
        supervision = supervision[source_indices].copy()
        self._apply_spatial_transform(
            output,
            scale=scale,
            translation=translation,
            noise_rng=streams[3],
        )
        dropped_joints = self._apply_joint_dropout(output, streams[4])
        dropped_frames, effective, generation = self._apply_frame_policy(
            output, labels, streams[5]
        )
        self._recompute_motion(output, effective, generation)
        if not np.isfinite(output).all():
            raise FloatingPointError("skeleton augmentation produced non-finite features")
        return AugmentedClip(
            features=output,
            frame_labels=labels,
            start_targets=starts,
            end_targets=ends,
            supervision_mask=supervision,
            source_indices=source_indices,
            dropped_frames=dropped_frames,
            dropped_joints=dropped_joints,
            speed_factor=speed,
            scale_factor=scale,
            translation_xy=translation,
        )

    def _validate_inputs(
        self,
        features: np.ndarray,
        frame_labels: np.ndarray,
        start_targets: np.ndarray,
        end_targets: np.ndarray,
        supervision_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feature_values = np.asarray(features)
        if feature_values.ndim != 2:
            raise ValueError(f"features must have shape [T,D], got {feature_values.shape}")
        if feature_values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"feature dimension {feature_values.shape[1]} does not match "
                f"{len(self.feature_names)} feature_names"
            )
        if not np.issubdtype(feature_values.dtype, np.floating):
            raise TypeError("features must use a floating dtype")
        if not np.isfinite(feature_values).all():
            raise ValueError("features contain non-finite values")
        frames = int(feature_values.shape[0])
        values = tuple(
            np.asarray(value)
            for value in (frame_labels, start_targets, end_targets, supervision_mask)
        )
        field_names = (
            "frame_labels",
            "start_targets",
            "end_targets",
            "supervision_mask",
        )
        for name, value in zip(field_names, values):
            if value.shape != (frames,):
                raise ValueError(f"{name} must have shape ({frames},), got {value.shape}")
        if not np.isfinite(values[1]).all() or not np.isfinite(values[2]).all():
            raise ValueError("start_targets/end_targets contain non-finite values")
        return (feature_values, values[0], values[1], values[2], values[3])

    def _apply_spatial_transform(
        self,
        features: np.ndarray,
        *,
        scale: float,
        translation: Tuple[float, float],
        noise_rng: np.random.Generator,
    ) -> None:
        layout = self.layout
        observed = features[:, layout.quality_valid] >= 0.5
        encoded_missing = np.rint(
            features[:, layout.quality_missing]
            * float(self.config.missing_clip_frames)
        ).astype(np.int64)
        state_columns = np.asarray(layout.state_columns, dtype=np.int64)
        has_state = observed.copy()
        if state_columns.size:
            held = (
                (encoded_missing > 0)
                & (encoded_missing <= self.config.max_hold_frames)
                & np.any(np.abs(features[:, state_columns]) > 0.0, axis=1)
            )
            has_state |= held
        rows = np.flatnonzero(has_state)
        pose_columns = np.asarray(tuple(layout.pose.values()), dtype=np.int64)
        if rows.size and pose_columns.size and scale != 1.0:
            features[np.ix_(rows, pose_columns)] *= scale
        for axis, column in layout.wrist.items():
            if rows.size:
                features[rows, column] *= scale
                features[rows, column] += translation[0 if axis == "x" else 1]
        if rows.size and layout.log_scale is not None and scale != 1.0:
            features[rows, layout.log_scale] += math.log(scale)

        if self.config.coordinate_noise > 0.0:
            observed_rows = np.flatnonzero(observed)
            if observed_rows.size and pose_columns.size:
                noise = noise_rng.normal(
                    0.0,
                    self.config.coordinate_noise,
                    size=(observed_rows.size, pose_columns.size),
                )
                features[np.ix_(observed_rows, pose_columns)] += noise.astype(
                    features.dtype, copy=False
                )
            wrist_columns = np.asarray(tuple(layout.wrist.values()), dtype=np.int64)
            if observed_rows.size and wrist_columns.size:
                noise = noise_rng.normal(
                    0.0,
                    self.config.coordinate_noise,
                    size=(observed_rows.size, wrist_columns.size),
                )
                features[np.ix_(observed_rows, wrist_columns)] += noise.astype(
                    features.dtype, copy=False
                )

    def _apply_joint_dropout(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        joints = np.asarray(sorted({key[0] for key in self.layout.pose}), dtype=np.int64)
        if self.config.joint_dropout <= 0.0:
            return np.empty(0, dtype=np.int64)
        dropped = joints[rng.random(joints.size) < self.config.joint_dropout]
        if dropped.size:
            dropped_set = set(int(value) for value in dropped)
            columns = [
                column
                for (joint, _axis), column in self.layout.pose.items()
                if joint in dropped_set
            ]
            features[:, columns] = 0.0
        return dropped

    def _apply_frame_policy(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        layout = self.layout
        frames = int(features.shape[0])
        original_observed = features[:, layout.quality_valid] >= 0.5
        padding = labels == self.config.ignore_label
        draws = rng.random(frames) < self.config.frame_dropout
        dropped = original_observed & draws & ~padding
        observed = original_observed & ~dropped & ~padding
        original_missing = np.rint(
            features[:, layout.quality_missing]
            * float(self.config.missing_clip_frames)
        ).astype(np.int64)

        source_state = features[:, layout.state_columns].copy()
        features[:, layout.dynamic_columns] = 0.0
        effective = np.zeros(frames, dtype=np.bool_)
        generation = np.zeros(frames, dtype=np.int64)
        last_state = None
        run = 0
        current_generation = 0
        reset_active = False
        for frame in range(frames):
            if padding[frame]:
                run = 0
                last_state = None
                if not reset_active:
                    current_generation += 1
                reset_active = True
                features[frame, layout.quality_valid] = 0.0
                features[frame, layout.quality_missing] = 0.0
                generation[frame] = current_generation
                continue
            if observed[frame]:
                run = 0
                reset_active = False
                last_state = source_state[frame].copy()
                features[frame, layout.state_columns] = last_state
                effective[frame] = True
            else:
                if frame == 0 and not dropped[frame] and original_missing[frame] > 0:
                    run = int(original_missing[frame])
                    if (
                        run <= self.config.max_hold_frames
                        and np.any(np.abs(source_state[frame]) > 0.0)
                    ):
                        last_state = source_state[frame].copy()
                else:
                    run += 1
                if last_state is not None and run <= self.config.max_hold_frames:
                    features[frame, layout.state_columns] = last_state
                    effective[frame] = True
                else:
                    last_state = None
                    if not reset_active:
                        current_generation += 1
                        reset_active = True
            features[frame, layout.quality_valid] = float(observed[frame])
            features[frame, layout.quality_missing] = min(
                run, self.config.missing_clip_frames
            ) / float(self.config.missing_clip_frames)
            generation[frame] = current_generation
        return dropped, effective, generation

    def _recompute_motion(
        self,
        features: np.ndarray,
        effective: np.ndarray,
        generation: np.ndarray,
    ) -> None:
        frames = int(features.shape[0])
        for lag, columns in self.layout.motion.items():
            features[:, tuple(columns.values())] = 0.0
            if lag >= frames:
                continue
            compatible = (
                effective[lag:]
                & effective[:-lag]
                & (generation[lag:] == generation[:-lag])
            )
            targets = np.flatnonzero(compatible) + lag
            sources = targets - lag
            for key, column in columns.items():
                pose_column = self.layout.pose[key]
                features[targets, column] = (
                    features[targets, pose_column] - features[sources, pose_column]
                )
        for lag, columns in self.layout.wrist_motion.items():
            features[:, tuple(columns.values())] = 0.0
            if lag >= frames:
                continue
            compatible = (
                effective[lag:]
                & effective[:-lag]
                & (generation[lag:] == generation[:-lag])
            )
            targets = np.flatnonzero(compatible) + lag
            sources = targets - lag
            for axis, column in columns.items():
                wrist_column = self.layout.wrist[axis]
                features[targets, column] = (
                    features[targets, wrist_column] - features[sources, wrist_column]
                )
        for lag, column in self.layout.scale_motion.items():
            features[:, column] = 0.0
            if lag >= frames or self.layout.log_scale is None:
                continue
            compatible = (
                effective[lag:]
                & effective[:-lag]
                & (generation[lag:] == generation[:-lag])
            )
            targets = np.flatnonzero(compatible) + lag
            sources = targets - lag
            features[targets, column] = (
                features[targets, self.layout.log_scale]
                - features[sources, self.layout.log_scale]
            )


__all__ = [
    "AugmentedClip",
    "SkeletonAugmentationConfig",
    "SkeletonFeatureAugmenter",
]
