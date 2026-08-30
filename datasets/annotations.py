"""Annotation contracts shared by training, decoding and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


def inclusive_to_half_open(start: int, end: int) -> Tuple[int, int]:
    """Convert a zero-based inclusive interval to a half-open interval."""

    start = int(start)
    end = int(end)
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if end < start:
        raise ValueError(f"inclusive end {end} precedes start {start}")
    return start, end + 1


def half_open_to_inclusive(start: int, end_exclusive: int) -> Tuple[int, int]:
    """Convert a non-empty zero-based half-open interval to inclusive form."""

    start = int(start)
    end_exclusive = int(end_exclusive)
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if end_exclusive <= start:
        raise ValueError(
            f"end_exclusive {end_exclusive} must be greater than start {start}"
        )
    return start, end_exclusive - 1


def subject_key(video_id: str) -> str:
    """Return the stable IPN subject key shared by its four videos."""

    value = str(video_id)
    if "#" not in value:
        raise ValueError(f"IPN video id has no '#': {value!r}")
    key = value.split("#", 1)[0]
    if not key:
        raise ValueError(f"empty subject key in video id {value!r}")
    return key


@dataclass(frozen=True)
class GestureSegment:
    """An internal zero-based half-open gesture annotation."""

    class_id: int
    class_name: str
    class_code: str
    official_class_id: int
    start_frame: int
    end_frame_exclusive: int

    @classmethod
    def from_manifest(
        cls, payload: Mapping[str, Any], *, num_frames: int
    ) -> "GestureSegment":
        start, end_exclusive = inclusive_to_half_open(
            int(payload["start"]), int(payload["end"])
        )
        if end_exclusive > int(num_frames):
            raise ValueError(
                f"segment [{start},{end_exclusive}) exceeds video length {num_frames}"
            )
        class_id = int(payload["class_id"])
        if not 0 <= class_id < 13:
            raise ValueError(f"class_id must be in [0,12], got {class_id}")
        return cls(
            class_id=class_id,
            class_name=str(payload["class_name"]),
            class_code=str(payload.get("class_code", "")),
            official_class_id=int(payload.get("official_class_id", class_id + 2)),
            start_frame=start,
            end_frame_exclusive=end_exclusive,
        )

    @property
    def duration(self) -> int:
        return self.end_frame_exclusive - self.start_frame

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "class_code": self.class_code,
            "official_class_id": self.official_class_id,
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
        }

