"""Stateful online execution for the final GestureDetection-MQTCN model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from models.gesture_detection_mqtcn import GestureDetectionMQTCN
from models.streaming_tcn import StatefulContinuousBaseline
from streaming.decoder import DecoderConfig, StreamingDecoder
from streaming.query_tracker import (
    QueryDecoderConfig,
    QueryEventTracker,
    decode_query_events,
    fuse_frame_and_query_events,
)


@dataclass(frozen=True)
class ModelStepResult:
    frame_outputs: Mapping[str, Tensor]
    frame_events: Sequence[Mapping[str, object]]
    query_events: Sequence[Mapping[str, object]]
    fusion_events: Sequence[Mapping[str, object]]
    query_executed: bool
    prefix_time: int


class StreamingModelRuntime:
    def __init__(
        self,
        model: GestureDetectionMQTCN,
        *,
        query_stride: int,
        frame_decoder_config: DecoderConfig,
        query_decoder_config: QueryDecoderConfig,
    ) -> None:
        self.model = model.eval()
        self.backbone = StatefulContinuousBaseline(model.baseline).eval()
        self.query_stride = int(query_stride)
        if self.query_stride <= 0:
            raise ValueError("query_stride must be positive")
        self.frame_decoder_config = frame_decoder_config
        self.query_decoder_config = query_decoder_config
        self._frame_decoders: Dict[Hashable, StreamingDecoder] = {}
        self._query_trackers: Dict[Hashable, QueryEventTracker] = {}
        self._fusion_trackers: Dict[Hashable, QueryEventTracker] = {}
        self._stream_start: Dict[Hashable, int] = {}
        self._last_frame: Dict[Hashable, int] = {}
        self._last_query_prefix: Dict[Hashable, int] = {}
        self._last_embedding: Dict[Hashable, Tensor] = {}

    @property
    def active_stream_ids(self) -> Tuple[Hashable, ...]:
        identifiers = set(self.backbone.active_stream_ids)
        identifiers.update(self.model.frame_memory.active_stream_ids)
        return tuple(sorted(identifiers, key=str))

    def reset_state(
        self,
        stream_ids: Optional[Sequence[Hashable]] = None,
        *,
        next_frame_index: int = 0,
    ) -> None:
        identifiers = (
            tuple(set(self.active_stream_ids) | set(self._frame_decoders))
            if stream_ids is None
            else tuple(stream_ids)
        )
        self.backbone.reset_state(None if stream_ids is None else identifiers)
        self.model.frame_memory.reset_state(None if stream_ids is None else identifiers)
        for stream_id in identifiers:
            if stream_id in self._frame_decoders:
                self._frame_decoders[stream_id].reset(start_frame=int(next_frame_index))
            if stream_id in self._query_trackers:
                self._query_trackers[stream_id].reset()
            if stream_id in self._fusion_trackers:
                self._fusion_trackers[stream_id].reset()
            self._stream_start.pop(stream_id, None)
            self._last_frame.pop(stream_id, None)
            self._last_query_prefix.pop(stream_id, None)
            self._last_embedding.pop(stream_id, None)
        if stream_ids is None:
            self._frame_decoders.clear()
            self._query_trackers.clear()
            self._fusion_trackers.clear()

    def _ensure_stream(self, stream_id: Hashable, frame_index: int) -> None:
        if stream_id not in self._stream_start:
            self._stream_start[stream_id] = int(frame_index)
            self._last_frame[stream_id] = int(frame_index) - 1
            self._last_query_prefix[stream_id] = int(frame_index)
            decoder = StreamingDecoder(self.frame_decoder_config)
            decoder.reset(start_frame=int(frame_index))
            self._frame_decoders[stream_id] = decoder
            self._query_trackers[stream_id] = QueryEventTracker(self.query_decoder_config)
            self._fusion_trackers[stream_id] = QueryEventTracker(self.query_decoder_config)

    @torch.inference_mode()
    def _query(
        self, stream_id: Hashable, embedding: Tensor, prefix: int
    ) -> Tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]:
        current = embedding.unsqueeze(0)
        frame_tokens, frame_valid = self.model.frame_memory.read_stream(stream_id)
        logits, intervals, _ = self.model.query_decoder(
            current, frame_tokens, frame_valid
        )
        candidates = decode_query_events(
            logits.cpu().numpy(),
            intervals.cpu().numpy(),
            [int(prefix)],
            memory_length=self.model.frame_memory_length,
            config=self.query_decoder_config,
            stream_start=self._stream_start[stream_id],
        )
        emitted_query = []
        for event in candidates:
            accepted = self._query_trackers[stream_id].add(event)
            if accepted is not None:
                emitted_query.append(accepted)
        fusion_candidates = fuse_frame_and_query_events(
            [event.as_dict() for event in self._frame_decoders[stream_id].tracker.events],
            emitted_query,
            config=self.query_decoder_config,
        )
        emitted_fusion = []
        for event in fusion_candidates:
            accepted = self._fusion_trackers[stream_id].add(event)
            if accepted is not None:
                emitted_fusion.append(accepted)
        self._last_query_prefix[stream_id] = int(prefix)
        return emitted_query, emitted_fusion

    @torch.inference_mode()
    def step(
        self,
        frame_features: Tensor,
        *,
        frame_index: int,
        stream_id: Hashable = 0,
        memory_valid: bool = True,
    ) -> ModelStepResult:
        frame = int(frame_index)
        self._ensure_stream(stream_id, frame)
        if frame != self._last_frame[stream_id] + 1:
            raise ValueError("stream frames must be contiguous after reset")
        outputs, embedding = self.backbone.step_with_embedding(
            frame_features, stream_ids=(stream_id,)
        )
        self._last_embedding[stream_id] = embedding.detach()
        self.model.frame_memory.append_stream(
            stream_id, embedding, valid=bool(memory_valid)
        )
        frame_events = self._frame_decoders[stream_id].step(
            torch.softmax(outputs["frame_logits"], dim=-1).cpu().numpy(),
            float(torch.sigmoid(outputs["start_logits"]).item()),
            float(torch.sigmoid(outputs["end_logits"]).item()),
            frame,
        )
        self._last_frame[stream_id] = frame
        prefix = frame + 1
        query_executed = prefix - self._last_query_prefix[stream_id] >= self.query_stride
        query_events: Sequence[Mapping[str, object]] = ()
        fusion_events: Sequence[Mapping[str, object]] = ()
        if query_executed:
            query_events, fusion_events = self._query(stream_id, embedding, prefix)
        return ModelStepResult(
            frame_outputs=outputs,
            frame_events=tuple(event.as_dict() for event in frame_events),
            query_events=tuple(query_events),
            fusion_events=tuple(fusion_events),
            query_executed=query_executed,
            prefix_time=prefix,
        )

    @torch.inference_mode()
    def finish_stream(
        self, stream_id: Hashable
    ) -> Tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]:
        if stream_id not in self._last_frame:
            return (), ()
        self._frame_decoders[stream_id].finish()
        prefix = self._last_frame[stream_id] + 1
        if prefix == self._last_query_prefix[stream_id]:
            return (), ()
        return self._query(stream_id, self._last_embedding[stream_id], prefix)

    def events_for_stream(
        self, stream_id: Hashable
    ) -> Mapping[str, Tuple[Mapping[str, object], ...]]:
        frame_decoder = self._frame_decoders.get(stream_id)
        query_tracker = self._query_trackers.get(stream_id)
        fusion_tracker = self._fusion_trackers.get(stream_id)
        return {
            "frame": tuple(dict(event.as_dict()) for event in frame_decoder.tracker.events)
            if frame_decoder is not None
            else (),
            "query": tuple(dict(event) for event in query_tracker.events)
            if query_tracker is not None
            else (),
            "fusion": tuple(dict(event) for event in fusion_tracker.events)
            if fusion_tracker is not None
            else (),
        }

    def persistent_state_bytes(self) -> int:
        hidden = self.model.hidden_dim
        tcn = sum(
            block.left_padding * hidden * 4
            for block in self.model.baseline.backbone.blocks
        )
        frame_memory = self.model.frame_memory_length * hidden * 4
        validity = self.model.frame_memory_length
        return int(tcn + frame_memory + validity)


__all__ = ["ModelStepResult", "StreamingModelRuntime"]
