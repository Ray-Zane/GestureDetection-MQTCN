"""End-to-end GestureDetection-MQTCN runtime with one reset contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from models.gesture_detection_mqtcn import GestureDetectionMQTCN
from preprocessing.p3_features import (
    P3FeatureConfig,
    StreamingFeatureResult,
    StreamingP3FeatureBuilder,
)
from streaming.decoder import DecoderConfig
from streaming.model_runtime import ModelStepResult, StreamingModelRuntime
from streaming.query_tracker import QueryDecoderConfig


@dataclass(frozen=True)
class RealtimeStepResult:
    """Result of one decoded video/camera frame."""

    status: str
    feature: StreamingFeatureResult
    model: Optional[ModelStepResult]
    emitted_events: Sequence[Mapping[str, object]]


class RealtimePipeline:
    """Own P3, stateful TCN, Frame Memory and all event-decoder state.

    A short missing-hand run uses held P3 geometry but marks the Frame Memory
    entry invalid.  Once the configured hold limit is exceeded, every learned
    and decoder state is reset atomically.  Finalized events are archived before
    reset so a temporary tracking outage cannot erase already produced results.
    """

    EVENT_OUTPUTS = ("frame", "query", "fusion")

    def __init__(
        self,
        model: GestureDetectionMQTCN,
        *,
        feature_config: P3FeatureConfig,
        frame_decoder_config: DecoderConfig,
        query_decoder_config: QueryDecoderConfig,
        query_stride: int,
        device: torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.feature_builder = StreamingP3FeatureBuilder(feature_config)
        if self.feature_builder.output_dim != model.baseline.backbone.input_dim:
            raise ValueError(
                "feature/model input mismatch: "
                f"{self.feature_builder.output_dim} != "
                f"{model.baseline.backbone.input_dim}"
            )
        self.runtime = StreamingModelRuntime(
            model.to(self.device).eval(),
            query_stride=int(query_stride),
            frame_decoder_config=frame_decoder_config,
            query_decoder_config=query_decoder_config,
        )
        self._blocked: Dict[Hashable, bool] = {}
        self._archived: Dict[
            Hashable, Dict[str, List[Mapping[str, object]]]
        ] = {}

    def _archive_runtime_events(self, stream_id: Hashable) -> None:
        current = self.runtime.events_for_stream(stream_id)
        archive = self._archived.setdefault(
            stream_id, {name: [] for name in self.EVENT_OUTPUTS}
        )
        for name in self.EVENT_OUTPUTS:
            archive[name].extend(dict(event) for event in current[name])

    def reset_stream(self, stream_id: Hashable, *, next_frame_index: int = 0) -> None:
        """Start a new logical source and discard all prior source results."""

        self.feature_builder.reset_state((stream_id,))
        self.runtime.reset_state((stream_id,), next_frame_index=int(next_frame_index))
        self._blocked.pop(stream_id, None)
        self._archived.pop(stream_id, None)

    def reset_all(self) -> None:
        self.feature_builder.reset_state()
        self.runtime.reset_state()
        self._blocked.clear()
        self._archived.clear()

    def process_skeleton_frame(
        self,
        *,
        image_landmarks: np.ndarray,
        world_landmarks: np.ndarray,
        valid: bool,
        width: int,
        height: int,
        frame_index: int,
        metadata_hand: Optional[str] = None,
        stream_id: Hashable = 0,
    ) -> RealtimeStepResult:
        feature = self.feature_builder.step(
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
            valid=valid,
            width=width,
            height=height,
            metadata_hand=metadata_hand,
            stream_id=stream_id,
        )
        frame = int(frame_index)
        if feature.reset_required:
            self._archive_runtime_events(stream_id)
            self.runtime.reset_state((stream_id,), next_frame_index=frame + 1)
            self._blocked[stream_id] = True
            return RealtimeStepResult("reset_missing", feature, None, ())
        if self._blocked.get(stream_id, False) and not feature.observed_valid:
            return RealtimeStepResult(
                "waiting_for_skeleton", feature, None, ()
            )

        status = "processed"
        if self._blocked.pop(stream_id, False):
            self.runtime.reset_state((stream_id,), next_frame_index=frame)
            status = "processed_after_reset"
        with torch.inference_mode():
            result = self.runtime.step(
                torch.from_numpy(feature.features).to(self.device),
                frame_index=frame,
                stream_id=stream_id,
                memory_valid=bool(feature.observed_valid),
            )
        return RealtimeStepResult(
            status,
            feature,
            result,
            tuple(dict(event) for event in result.fusion_events),
        )

    def finish_stream(
        self, stream_id: Hashable
    ) -> Mapping[str, Tuple[Mapping[str, object], ...]]:
        self.runtime.finish_stream(stream_id)
        return self.events_for_stream(stream_id)

    def events_for_stream(
        self, stream_id: Hashable
    ) -> Mapping[str, Tuple[Mapping[str, object], ...]]:
        current = self.runtime.events_for_stream(stream_id)
        archive = self._archived.get(stream_id, {})
        output = {}
        for name in self.EVENT_OUTPUTS:
            combined = [dict(event) for event in archive.get(name, ())]
            combined.extend(dict(event) for event in current[name])
            output[name] = tuple(
                sorted(
                    combined,
                    key=lambda event: (
                        int(event["start_frame"]),
                        int(event["end_frame_exclusive"]),
                    ),
                )
            )
        return output


__all__ = ["RealtimePipeline", "RealtimeStepResult"]
