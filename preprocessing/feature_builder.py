"""Causal MediaPipe hand feature construction for full training videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from datasets.ipn_skeleton import VideoRecord


@dataclass(frozen=True)
class FeatureBuilderConfig:
    preprocessing_profile: str = "p3"
    coordinate_source: str = "image_xyz"
    motion_lags: Tuple[int, ...] = (1,)
    max_hold_frames: int = 5
    missing_clip_frames: int = 30
    scale_epsilon: float = 1.0e-6
    include_handedness: bool = True

    def __post_init__(self) -> None:
        profile = str(self.preprocessing_profile).lower()
        if profile not in {"p0", "p1", "p2", "p3", "p4"}:
            raise ValueError(f"unsupported preprocessing profile {profile!r}")
        object.__setattr__(self, "preprocessing_profile", profile)
        if self.coordinate_source not in {
            "image_xy",
            "image_xyz",
            "world_local_xyz",
        }:
            raise ValueError(f"unsupported coordinate source {self.coordinate_source!r}")
        if not self.motion_lags or any(int(lag) <= 0 for lag in self.motion_lags):
            raise ValueError("motion_lags must contain positive integers")
        if tuple(sorted(set(self.motion_lags))) != tuple(self.motion_lags):
            raise ValueError("motion_lags must be sorted and unique")
        if self.max_hold_frames < 0 or self.missing_clip_frames <= 0:
            raise ValueError("missing-frame limits must be non-negative/positive")
        if self.scale_epsilon <= 0:
            raise ValueError("scale_epsilon must be positive")


@dataclass(frozen=True)
class FeatureBatch:
    features: np.ndarray
    feature_names: Tuple[str, ...]
    observed_valid: np.ndarray
    effective_valid: np.ndarray
    reset_mask: np.ndarray
    missing_run_length: np.ndarray

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[-1])


def _handedness_value(metadata_hand: Optional[str]) -> float:
    value = "" if metadata_hand is None else str(metadata_hand).strip().lower()
    if value.startswith("r"):
        return 1.0
    if value.startswith("l"):
        return -1.0
    return 0.0


class SkeletonFeatureBuilder:
    """Build wrist-normalized pose, causal motion and global hand features."""

    def __init__(self, config: Optional[FeatureBuilderConfig] = None) -> None:
        self.config = config or FeatureBuilderConfig()
        self.feature_names = self._feature_names()

    @property
    def output_dim(self) -> int:
        return len(self.feature_names)

    def _feature_names(self) -> Tuple[str, ...]:
        axes = ("x", "y") if self.config.coordinate_source == "image_xy" else ("x", "y", "z")
        names = [f"local_pose.joint_{joint}.{axis}" for joint in range(21) for axis in axes]
        for lag in self.config.motion_lags:
            names.extend(
                f"local_motion_lag_{lag}.joint_{joint}.{axis}"
                for joint in range(21)
                for axis in axes
            )
        names.extend(("global.wrist_x", "global.wrist_y"))
        for lag in self.config.motion_lags:
            names.extend(
                (f"global.wrist_dx_lag_{lag}", f"global.wrist_dy_lag_{lag}")
            )
        names.append("global.log_scale")
        for lag in self.config.motion_lags:
            names.append(f"global.delta_log_scale_lag_{lag}")
        names.extend(("quality.valid", "quality.missing_run_length"))
        if self.config.include_handedness:
            names.append("quality.metadata_handedness")
        return tuple(names)

    @staticmethod
    def _aspect_correct_image(
        image_landmarks: np.ndarray, *, width: int, height: int
    ) -> np.ndarray:
        output = np.asarray(image_landmarks, dtype=np.float32).copy()
        if output.ndim != 3 or output.shape[1:] != (21, 3):
            raise ValueError(f"expected image landmarks [T,21,3], got {output.shape}")
        if width <= 0 or height <= 0:
            raise ValueError("image width and height must be positive")
        output[:, :, 1] *= float(height) / float(width)
        return output

    @staticmethod
    def _causal_palm_rotation(centered: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Rotate each valid frame from its current palm basis only.

        No temporal statistic or future frame is used. Degenerate frames keep the
        centered coordinates unchanged, which is the explicit P4 fallback.
        """

        output = np.asarray(centered, dtype=np.float32).copy()
        dimensions = int(output.shape[-1])
        epsilon = 1.0e-6
        for frame in np.flatnonzero(valid):
            points = output[frame]
            if dimensions == 2:
                forward = points[9]
                norm = float(np.linalg.norm(forward))
                if not np.isfinite(norm) or norm <= epsilon:
                    continue
                cosine, sine = float(forward[0] / norm), float(forward[1] / norm)
                rotation = np.asarray(
                    [[cosine, sine], [-sine, cosine]], dtype=np.float32
                )
                output[frame] = points @ rotation.T
                continue

            forward = points[9]
            lateral = points[5] - points[17]
            forward_norm = float(np.linalg.norm(forward))
            if not np.isfinite(forward_norm) or forward_norm <= epsilon:
                continue
            y_axis = forward / forward_norm
            lateral = lateral - float(np.dot(lateral, y_axis)) * y_axis
            lateral_norm = float(np.linalg.norm(lateral))
            if not np.isfinite(lateral_norm) or lateral_norm <= epsilon:
                continue
            x_axis = lateral / lateral_norm
            z_axis = np.cross(x_axis, y_axis)
            z_norm = float(np.linalg.norm(z_axis))
            if not np.isfinite(z_norm) or z_norm <= epsilon:
                continue
            z_axis = z_axis / z_norm
            x_axis = np.cross(y_axis, z_axis)
            basis = np.stack((x_axis, y_axis, z_axis), axis=1).astype(np.float32)
            output[frame] = points @ basis
        return output

    def build(
        self,
        *,
        image_landmarks: np.ndarray,
        world_landmarks: np.ndarray,
        valid_mask: np.ndarray,
        width: int,
        height: int,
        metadata_hand: Optional[str] = None,
    ) -> FeatureBatch:
        image = self._aspect_correct_image(
            image_landmarks, width=int(width), height=int(height)
        )
        world = np.asarray(world_landmarks, dtype=np.float32)
        if world.shape != image.shape:
            raise ValueError(
                f"world/image landmark shape mismatch: {world.shape} vs {image.shape}"
            )
        valid = np.asarray(valid_mask, dtype=np.bool_)
        frames = image.shape[0]
        if valid.shape != (frames,):
            raise ValueError(f"valid mask shape {valid.shape} does not match T={frames}")
        finite = np.isfinite(image).all(axis=(1, 2)) & np.isfinite(world).all(axis=(1, 2))
        observed = valid & finite

        if self.config.coordinate_source.startswith("image"):
            local_source = image
        else:
            local_source = world.copy()
        local_dims = 2 if self.config.coordinate_source == "image_xy" else 3
        local_source = local_source[:, :, :local_dims]

        filled_local = np.zeros_like(local_source, dtype=np.float32)
        filled_global = np.zeros_like(image[:, :, :2], dtype=np.float32)
        effective = np.zeros(frames, dtype=np.bool_)
        reset_mask = np.zeros(frames, dtype=np.bool_)
        missing_run = np.zeros(frames, dtype=np.int32)
        generation = np.zeros(frames, dtype=np.int64)
        current_generation = 0
        last_local: Optional[np.ndarray] = None
        last_global: Optional[np.ndarray] = None
        run = 0
        reset_emitted = False

        for frame in range(frames):
            if observed[frame]:
                run = 0
                reset_emitted = False
                last_local = local_source[frame].copy()
                last_global = image[frame, :, :2].copy()
                filled_local[frame] = last_local
                filled_global[frame] = last_global
                effective[frame] = True
            else:
                run += 1
                missing_run[frame] = run
                if last_local is not None and run <= self.config.max_hold_frames:
                    filled_local[frame] = last_local
                    filled_global[frame] = last_global
                    effective[frame] = True
                else:
                    last_local = None
                    last_global = None
                    if not reset_emitted:
                        current_generation += 1
                        reset_mask[frame] = True
                        reset_emitted = True
            generation[frame] = current_generation

        profile = self.config.preprocessing_profile
        center = filled_local[:, 0:1, :]
        scale_1 = np.linalg.norm(
            filled_local[:, 5, :] - filled_local[:, 17, :], axis=-1
        )
        scale_2 = np.linalg.norm(
            filled_local[:, 0, :] - filled_local[:, 9, :], axis=-1
        )
        scale = 0.5 * (scale_1 + scale_2)
        geometry_valid = effective & np.isfinite(scale) & (scale > self.config.scale_epsilon)
        local_pose = np.zeros_like(filled_local, dtype=np.float32)
        if profile == "p0":
            local_pose[effective] = filled_local[effective]
        elif profile == "p1":
            local_pose[effective] = (
                filled_local[effective] - center[effective]
            )
        else:
            centered = filled_local - center
            if profile == "p4":
                centered = self._causal_palm_rotation(centered, geometry_valid)
            local_pose[geometry_valid] = centered[geometry_valid] / scale[
                geometry_valid, None, None
            ]

        wrist = np.zeros((frames, 2), dtype=np.float32)
        if profile in {"p3", "p4"}:
            wrist[effective] = filled_global[effective, 0, :]
        log_scale = np.zeros(frames, dtype=np.float32)
        if profile in {"p3", "p4"}:
            log_scale[geometry_valid] = np.log(
                np.maximum(scale[geometry_valid], self.config.scale_epsilon)
            )

        parts = [local_pose.reshape(frames, -1)]
        wrist_deltas = []
        scale_deltas = []
        for lag in self.config.motion_lags:
            pose_delta = np.zeros_like(local_pose, dtype=np.float32)
            wrist_delta = np.zeros((frames, 2), dtype=np.float32)
            scale_delta = np.zeros(frames, dtype=np.float32)
            if profile in {"p3", "p4"} and lag < frames:
                compatible = (
                    geometry_valid[lag:]
                    & geometry_valid[:-lag]
                    & (generation[lag:] == generation[:-lag])
                )
                target = np.flatnonzero(compatible) + lag
                source = target - lag
                pose_delta[target] = local_pose[target] - local_pose[source]
                wrist_delta[target] = wrist[target] - wrist[source]
                scale_delta[target] = log_scale[target] - log_scale[source]
            parts.append(pose_delta.reshape(frames, -1))
            wrist_deltas.append(wrist_delta)
            scale_deltas.append(scale_delta[:, None])

        parts.append(wrist)
        parts.extend(wrist_deltas)
        parts.append(log_scale[:, None])
        parts.extend(scale_deltas)
        parts.append(observed.astype(np.float32)[:, None])
        normalized_missing = np.minimum(
            missing_run, self.config.missing_clip_frames
        ).astype(np.float32) / float(self.config.missing_clip_frames)
        parts.append(normalized_missing[:, None])
        if self.config.include_handedness:
            handedness = np.full(
                (frames, 1), _handedness_value(metadata_hand), dtype=np.float32
            )
            parts.append(handedness)

        features = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        if features.shape != (frames, self.output_dim):
            raise AssertionError(
                f"feature shape {features.shape} does not match output_dim={self.output_dim}"
            )
        if not np.isfinite(features).all():
            raise FloatingPointError("feature builder produced non-finite values")
        return FeatureBatch(
            features=features,
            feature_names=self.feature_names,
            observed_valid=observed,
            effective_valid=geometry_valid,
            reset_mask=reset_mask,
            missing_run_length=missing_run,
        )

    def build_from_record(self, record: VideoRecord) -> FeatureBatch:
        return self.build(
            image_landmarks=record.image_landmarks,
            world_landmarks=record.world_landmarks,
            valid_mask=record.valid_mask,
            width=record.width,
            height=record.height,
            metadata_hand=record.metadata.get("Hand"),
        )


__all__ = ["FeatureBatch", "FeatureBuilderConfig", "SkeletonFeatureBuilder"]
