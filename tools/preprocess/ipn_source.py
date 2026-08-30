"""Shared parsers for the official IPN Hand annotation bundle."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"}


class IPNSourceError(RuntimeError):
    """Raised when official source files are missing or ambiguous."""


@dataclass(frozen=True)
class Annotation:
    video: str
    label: str
    class_id: int
    start: int
    end: int
    frames: int


def normalized_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def clean_video_id(value: str) -> str:
    value = value.strip()
    suffix = Path(value).suffix.lower()
    return Path(value).stem if suffix in VIDEO_SUFFIXES else value


def direct_file_map(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {entry.name.lower(): entry for entry in directory.iterdir() if entry.is_file()}


def find_named_file(files: Dict[str, Path], *names: str) -> Optional[Path]:
    for name in names:
        match = files.get(name.lower())
        if match is not None:
            return match
    return None


def find_annotations_dir(data_root: Path, explicit: Optional[Path]) -> Path:
    candidates = [explicit] if explicit is not None else [data_root / "annotations", data_root]
    for candidate in candidates:
        if candidate is not None and (candidate / "Annot_List.txt").is_file():
            return candidate.resolve()
    locations = ", ".join(str(path) for path in candidates if path is not None)
    raise IPNSourceError(f"Annot_List.txt was not found in: {locations}")


def _annotation_from_row(
    row: Sequence[str], path: Path, line_number: int
) -> Annotation:
    if len(row) != 6:
        raise IPNSourceError(
            f"Expected 6 annotation fields at {path}:{line_number}, got {len(row)}"
        )
    video, label = row[0].strip(), row[1].strip()
    try:
        class_id, start, end, frames = (int(value.strip()) for value in row[2:])
    except ValueError as error:
        raise IPNSourceError(f"Invalid integer at {path}:{line_number}: {row}") from error
    if not video or not label:
        raise IPNSourceError(f"Missing video or label at {path}:{line_number}")
    if start < 1 or end < start or frames != end - start + 1:
        raise IPNSourceError(
            f"Invalid one-based inclusive interval at {path}:{line_number}: {row}"
        )
    return Annotation(clean_video_id(video), label, class_id, start, end, frames)


def read_annotations(path: Path) -> List[Annotation]:
    annotations: List[Annotation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        for line_number, row in enumerate(reader, start=1):
            if not row or not any(value.strip() for value in row):
                continue
            if normalized_name(row[0]) in {"video", "videoname"}:
                continue
            annotations.append(_annotation_from_row(row, path, line_number))
    if not annotations:
        raise IPNSourceError(f"No annotations found in {path}")
    return annotations


def read_video_list(path: Optional[Path]) -> Dict[str, Optional[int]]:
    if path is None:
        return {}
    result: Dict[str, Optional[int]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            values = raw_line.replace(",", " ").split()
            if not values or normalized_name(values[0]) in {"video", "videoname"}:
                continue
            video_id = clean_video_id(values[0])
            try:
                frame_count = int(values[1]) if len(values) >= 2 else None
            except ValueError as error:
                raise IPNSourceError(
                    f"Invalid frame count at {path}:{line_number}"
                ) from error
            if video_id in result:
                raise IPNSourceError(f"Duplicate video {video_id!r} in {path}")
            result[video_id] = frame_count
    return result


def load_official_frame_counts(annotations_dir: Path) -> Dict[str, int]:
    files = direct_file_map(annotations_dir)
    paths = (
        find_named_file(files, "Video_TrainList.txt"),
        find_named_file(files, "Video_TestList.txt"),
    )
    counts: Dict[str, int] = {}
    for path in paths:
        for video_id, frame_count in read_video_list(path).items():
            if frame_count is not None:
                counts[video_id] = frame_count
    return counts


__all__ = [
    "Annotation",
    "IPNSourceError",
    "clean_video_id",
    "direct_file_map",
    "find_annotations_dir",
    "find_named_file",
    "load_official_frame_counts",
    "read_annotations",
    "read_video_list",
]
