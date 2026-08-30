"""Confirmed-event validation and one-time duplicate suppression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Mapping, Optional


@dataclass(frozen=True)
class ContinuousEvent:
    class_id: int
    start_frame: int
    end_frame_exclusive: int
    score: float
    emitted_at_frame: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.class_id) < 13:
            raise ValueError("class_id must be in [0,12]")
        if int(self.start_frame) < 0 or int(self.end_frame_exclusive) <= int(
            self.start_frame
        ):
            raise ValueError("event interval must be non-empty and non-negative")
        if int(self.emitted_at_frame) < int(self.start_frame):
            raise ValueError("emitted_at_frame cannot precede event start")

    @property
    def duration(self) -> int:
        return int(self.end_frame_exclusive) - int(self.start_frame)

    def as_dict(self) -> Mapping[str, object]:
        return asdict(self)


def temporal_iou(first: ContinuousEvent, second: ContinuousEvent) -> float:
    intersection = max(
        0,
        min(first.end_frame_exclusive, second.end_frame_exclusive)
        - max(first.start_frame, second.start_frame),
    )
    union = max(first.end_frame_exclusive, second.end_frame_exclusive) - min(
        first.start_frame, second.start_frame
    )
    return float(intersection / union) if union > 0 else 0.0


class EventTracker:
    def __init__(
        self,
        *,
        min_event_frames: int,
        cooldown_frames: int,
        dedup_gap_frames: int,
        dedup_tiou: float,
    ) -> None:
        self.min_event_frames = int(min_event_frames)
        self.cooldown_frames = int(cooldown_frames)
        self.dedup_gap_frames = int(dedup_gap_frames)
        self.dedup_tiou = float(dedup_tiou)
        self.events: List[ContinuousEvent] = []

    def reset(self) -> None:
        self.events.clear()

    def add(self, event: ContinuousEvent) -> Optional[ContinuousEvent]:
        if event.duration < self.min_event_frames:
            return None
        for previous in reversed(self.events):
            if event.start_frame - previous.end_frame_exclusive > max(
                self.cooldown_frames, self.dedup_gap_frames
            ):
                break
            if previous.class_id != event.class_id:
                continue
            gap = max(0, event.start_frame - previous.end_frame_exclusive)
            if gap <= self.dedup_gap_frames or temporal_iou(previous, event) >= self.dedup_tiou:
                return None
        self.events.append(event)
        return event


__all__ = ["ContinuousEvent", "EventTracker", "temporal_iou"]
