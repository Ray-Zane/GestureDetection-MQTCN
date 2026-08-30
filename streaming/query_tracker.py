"""Decode completed-event Query slots and fuse them with frame-level hints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np



def _event_tiou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    start = max(int(first["start_frame"]), int(second["start_frame"]))
    end = min(
        int(first["end_frame_exclusive"]), int(second["end_frame_exclusive"])
    )
    intersection = max(0, end - start)
    union = max(
        int(first["end_frame_exclusive"]), int(second["end_frame_exclusive"])
    ) - min(int(first["start_frame"]), int(second["start_frame"]))
    return float(intersection / union) if union > 0 else 0.0


@dataclass(frozen=True)
class QueryDecoderConfig:
    score_threshold: float = 0.30
    strong_score_threshold: float = 0.55
    min_event_frames: int = 8
    cooldown_frames: int = 6
    dedup_gap_frames: int = 8
    dedup_tiou: float = 0.5
    fusion_min_tiou: float = 0.10
    fusion_max_boundary_gap: int = 32
    fusion_query_weight: float = 0.75


def query_decoder_config_from_mapping(
    payload: Mapping[str, object]
) -> QueryDecoderConfig:
    fields = QueryDecoderConfig.__dataclass_fields__
    return QueryDecoderConfig(**{key: payload[key] for key in fields if key in payload})


class QueryEventTracker:
    """Suppress repeated Query hypotheses without merging adjacent non-overlaps."""

    def __init__(self, config: QueryDecoderConfig) -> None:
        self.config = config
        self.events: List[Mapping[str, Any]] = []

    def reset(self) -> None:
        self.events.clear()

    def add(self, event: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        if int(event["end_frame_exclusive"]) - int(event["start_frame"]) < int(
            self.config.min_event_frames
        ):
            return None
        for index in range(len(self.events) - 1, -1, -1):
            previous = self.events[index]
            if int(previous["class_id"]) != int(event["class_id"]):
                continue
            overlap = _event_tiou(previous, event)
            boundaries_close = max(
                abs(int(previous["start_frame"]) - int(event["start_frame"])),
                abs(
                    int(previous["end_frame_exclusive"])
                    - int(event["end_frame_exclusive"])
                ),
            ) <= int(self.config.dedup_gap_frames)
            if overlap >= float(self.config.dedup_tiou) or boundaries_close:
                if float(event["score"]) > float(previous["score"]):
                    self.events[index] = dict(event)
                    return None
                return None
        accepted = dict(event)
        self.events.append(accepted)
        return accepted


def decode_query_events(
    class_logits: np.ndarray,
    intervals: np.ndarray,
    query_times: Sequence[int],
    *,
    memory_length: int,
    config: QueryDecoderConfig,
    stream_start: int = 0,
) -> List[Mapping[str, Any]]:
    logits = np.asarray(class_logits, dtype=np.float64)
    bounds = np.asarray(intervals, dtype=np.float64)
    times = np.asarray(query_times, dtype=np.int64)
    if logits.ndim != 3 or bounds.shape != (*logits.shape[:2], 2):
        raise ValueError("Query predictions must be [S,Q,C] and [S,Q,2]")
    if times.shape != (logits.shape[0],):
        raise ValueError("query_times must be [S]")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    tracker = QueryEventTracker(config)
    window = int(memory_length)
    for step, prefix in enumerate(times.tolist()):
        candidates = []
        for slot in range(logits.shape[1]):
            label = int(np.argmax(probabilities[step, slot]))
            score = float(probabilities[step, slot, label])
            if label == 0 or score < float(config.score_threshold):
                continue
            conceptual_left = int(prefix) - window
            start = int(round(conceptual_left + bounds[step, slot, 0] * window))
            end = int(round(conceptual_left + bounds[step, slot, 1] * window))
            start = max(int(stream_start), min(start, int(prefix) - 1))
            end = max(start + 1, min(end, int(prefix)))
            candidates.append(
                {
                    "class_id": label - 1,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "score": score,
                    "emitted_at_frame": int(prefix) - 1,
                    "emitted_at_prefix": int(prefix),
                    "source": "query",
                    "query_slot": slot,
                }
            )
        for event in sorted(candidates, key=lambda value: -float(value["score"])):
            tracker.add(event)
    return sorted(tracker.events, key=lambda value: (int(value["start_frame"]), int(value["end_frame_exclusive"])))


def fuse_frame_and_query_events(
    frame_events: Sequence[Mapping[str, Any]],
    query_events: Sequence[Mapping[str, Any]],
    *,
    config: QueryDecoderConfig,
) -> List[Mapping[str, Any]]:
    """Use completed Query as the final gate; frame events provide consistency/bounds."""

    tracker = QueryEventTracker(config)
    query_weight = float(config.fusion_query_weight)
    for query in sorted(query_events, key=lambda value: int(value["emitted_at_prefix"])):
        best = None
        best_overlap = -1.0
        for frame in frame_events:
            if int(frame["class_id"]) != int(query["class_id"]):
                continue
            overlap = _event_tiou(frame, query)
            gap = max(
                abs(int(frame["start_frame"]) - int(query["start_frame"])),
                abs(
                    int(frame["end_frame_exclusive"])
                    - int(query["end_frame_exclusive"])
                ),
            )
            if (
                overlap >= float(config.fusion_min_tiou)
                or gap <= int(config.fusion_max_boundary_gap)
            ) and overlap > best_overlap:
                best = frame
                best_overlap = overlap
        if best is None:
            if float(query["score"]) >= float(config.strong_score_threshold):
                tracker.add({**query, "source": "query_strong"})
            continue
        start = int(
            round(
                query_weight * int(query["start_frame"])
                + (1.0 - query_weight) * int(best["start_frame"])
            )
        )
        end = int(
            round(
                query_weight * int(query["end_frame_exclusive"])
                + (1.0 - query_weight) * int(best["end_frame_exclusive"])
            )
        )
        end = max(start + 1, end)
        tracker.add(
            {
                **query,
                "start_frame": max(0, start),
                "end_frame_exclusive": end,
                "score": float(
                    0.5 * float(query["score"]) + 0.5 * float(best["score"])
                ),
                "source": "fusion",
                "frame_tiou": best_overlap,
            }
        )
    return sorted(tracker.events, key=lambda value: (int(value["start_frame"]), int(value["end_frame_exclusive"])))


__all__ = [
    "QueryDecoderConfig",
    "QueryEventTracker",
    "decode_query_events",
    "fuse_frame_and_query_events",
    "query_decoder_config_from_mapping",
]
