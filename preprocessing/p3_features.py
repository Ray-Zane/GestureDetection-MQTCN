"""Incremental P3 features for the final online runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np


def _metadata_handedness(value: Optional[str]) -> float:
    normalized = "" if value is None else str(value).strip().lower()
    if normalized.startswith("r"):
        return 1.0
    if normalized.startswith("l"):
        return -1.0
    return 0.0


@dataclass(frozen=True)
class P3FeatureConfig:
    coordinate_source: str = "image_xyz"
    motion_lags: Tuple[int, ...] = (1,)
    max_hold_frames: int = 5
    missing_clip_frames: int = 30
    scale_epsilon: float = 1.0e-6
    include_handedness: bool = True

    def __post_init__(self) -> None:
        if self.coordinate_source not in {"image_xy", "image_xyz", "world_local_xyz"}:
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
class StreamingFeatureResult:
    features: np.ndarray
    observed_valid: bool
    effective_valid: bool
    held_last: bool
    missing_run_length: int
    reset_required: bool


@dataclass
class _Primitive:
    local_pose: np.ndarray
    wrist: np.ndarray
    log_scale: float
    geometry_valid: bool
    generation: int


@dataclass
class _FeatureState:
    last_local: Optional[np.ndarray] = None
    last_global: Optional[np.ndarray] = None
    missing_run: int = 0
    generation: int = 0
    reset_emitted: bool = False
    history: List[_Primitive] = field(default_factory=list)


class StreamingP3FeatureBuilder:
    """Build one causal feature vector per MediaPipe landmark frame."""

    def __init__(self, config: Optional[P3FeatureConfig] = None) -> None:
        self.config = config or P3FeatureConfig()
        axes = 2 if self.config.coordinate_source == "image_xy" else 3
        self.output_dim = (
            21 * axes * (1 + len(self.config.motion_lags))
            + 2
            + 2 * len(self.config.motion_lags)
            + 1
            + len(self.config.motion_lags)
            + 2
            + int(self.config.include_handedness)
        )
        self._states: Dict[Hashable, _FeatureState] = {}

    @property
    def active_stream_ids(self) -> Tuple[Hashable, ...]:
        return tuple(self._states)

    def reset_state(self, stream_ids: Optional[Sequence[Hashable]] = None) -> None:
        if stream_ids is None:
            self._states.clear()
        else:
            for stream_id in tuple(stream_ids):
                self._states.pop(stream_id, None)

    def step(
        self,
        *,
        image_landmarks: np.ndarray,
        world_landmarks: np.ndarray,
        valid: bool,
        width: int,
        height: int,
        metadata_hand: Optional[str] = None,
        stream_id: Hashable = 0,
    ) -> StreamingFeatureResult:
        image = np.asarray(image_landmarks, dtype=np.float32).copy()
        world = np.asarray(world_landmarks, dtype=np.float32)
        if image.shape != (21, 3) or world.shape != (21, 3):
            raise ValueError("landmarks must have shape [21,3]")
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("image size must be positive")
        image[:, 1] *= float(height) / float(width)
        finite = bool(np.isfinite(image).all() and np.isfinite(world).all())
        observed = bool(valid) and finite
        state = self._states.setdefault(stream_id, _FeatureState())
        source = image if self.config.coordinate_source.startswith("image") else world
        dimensions = 2 if self.config.coordinate_source == "image_xy" else 3
        source = source[:, :dimensions]
        reset_required = False
        held = False

        if observed:
            state.missing_run = 0
            state.reset_emitted = False
            state.last_local = source.copy()
            state.last_global = image[:, :2].copy()
            filled_local = state.last_local
            filled_global = state.last_global
            effective = True
        else:
            state.missing_run += 1
            if state.last_local is not None and state.missing_run <= self.config.max_hold_frames:
                filled_local = state.last_local
                filled_global = state.last_global
                effective = True
                held = True
            else:
                state.last_local = None
                state.last_global = None
                filled_local = np.zeros((21, dimensions), dtype=np.float32)
                filled_global = np.zeros((21, 2), dtype=np.float32)
                effective = False
                if not state.reset_emitted:
                    state.generation += 1
                    state.reset_emitted = True
                    state.history.clear()
                    reset_required = True

        scale_1 = float(np.linalg.norm(filled_local[5] - filled_local[17]))
        scale_2 = float(np.linalg.norm(filled_local[0] - filled_local[9]))
        scale = 0.5 * (scale_1 + scale_2)
        geometry_valid = bool(
            effective and np.isfinite(scale) and scale > self.config.scale_epsilon
        )
        local_pose = np.zeros((21, dimensions), dtype=np.float32)
        wrist = np.zeros(2, dtype=np.float32)
        log_scale = 0.0
        if effective:
            wrist = filled_global[0].astype(np.float32)
        if geometry_valid:
            local_pose = ((filled_local - filled_local[0:1]) / scale).astype(np.float32)
            log_scale = float(np.log(max(scale, self.config.scale_epsilon)))

        parts = [local_pose.reshape(-1)]
        wrist_deltas = []
        scale_deltas = []
        for lag in self.config.motion_lags:
            pose_delta = np.zeros_like(local_pose)
            wrist_delta = np.zeros(2, dtype=np.float32)
            scale_delta = 0.0
            if len(state.history) >= lag:
                previous = state.history[-lag]
                if geometry_valid and previous.geometry_valid and state.generation == previous.generation:
                    pose_delta = local_pose - previous.local_pose
                    wrist_delta = wrist - previous.wrist
                    scale_delta = log_scale - previous.log_scale
            parts.append(pose_delta.reshape(-1))
            wrist_deltas.append(wrist_delta)
            scale_deltas.append(np.asarray([scale_delta], dtype=np.float32))
        parts.append(wrist)
        parts.extend(wrist_deltas)
        parts.append(np.asarray([log_scale], dtype=np.float32))
        parts.extend(scale_deltas)
        parts.append(np.asarray([float(observed)], dtype=np.float32))
        parts.append(
            np.asarray(
                [min(state.missing_run, self.config.missing_clip_frames) / float(self.config.missing_clip_frames)],
                dtype=np.float32,
            )
        )
        if self.config.include_handedness:
            parts.append(np.asarray([_metadata_handedness(metadata_hand)], dtype=np.float32))
        features = np.concatenate(parts).astype(np.float32, copy=False)
        if features.shape != (self.output_dim,) or not np.isfinite(features).all():
            raise FloatingPointError("invalid incremental P3 features")

        state.history.append(
            _Primitive(local_pose.copy(), wrist.copy(), log_scale, geometry_valid, state.generation)
        )
        maximum_lag = max(self.config.motion_lags)
        if len(state.history) > maximum_lag:
            del state.history[:-maximum_lag]
        return StreamingFeatureResult(
            features=features,
            observed_valid=observed,
            effective_valid=geometry_valid,
            held_last=held,
            missing_run_length=state.missing_run,
            reset_required=reset_required,
        )


__all__ = ["P3FeatureConfig", "StreamingFeatureResult", "StreamingP3FeatureBuilder"]
