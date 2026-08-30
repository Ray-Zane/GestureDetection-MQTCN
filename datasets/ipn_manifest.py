"""Minimal IPN Hand manifest reader used by the final video demo."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

from datasets.annotations import GestureSegment, subject_key


IPN_CLASS_NAMES: Tuple[str, ...] = (
    "Pointing with one finger",
    "Pointing with two fingers",
    "Click with one finger",
    "Click with two fingers",
    "Throw up",
    "Throw down",
    "Throw left",
    "Throw right",
    "Open twice",
    "Double click with one finger",
    "Double click with two fingers",
    "Zoom in",
    "Zoom out",
)


@dataclass(frozen=True)
class VideoManifest:
    video_id: str
    split: str
    video_path: Path
    fps: float
    num_frames: int
    subject: str
    annotations: Tuple[GestureSegment, ...]
    metadata: Mapping[str, str]


def _read_json_list(path: Path) -> List[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise TypeError(f"manifest must be a list: {path}")
    return payload


def _load_metadata(path: Path) -> Dict[str, Mapping[str, str]]:
    if not path.is_file():
        return {}
    output: Dict[str, Mapping[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            video_id = str(row.get("Video Name", "")).strip()
            if video_id:
                output[video_id] = {
                    str(key): str(value).strip() for key, value in row.items()
                }
    return output


def load_manifest(
    project_root: Union[str, Path], split: str
) -> Tuple[VideoManifest, ...]:
    split = str(split).lower()
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train or test, got {split!r}")
    root = Path(project_root).resolve()
    manifest_path = root / "data" / "manifests" / f"{split}.json"
    metadata = _load_metadata(root / "data" / "raw" / "annotations" / "metadata.csv")
    entries = []
    seen = set()
    for raw in _read_json_list(manifest_path):
        video_id = str(raw["video_id"])
        if video_id in seen:
            raise ValueError(f"duplicate video_id in {manifest_path}: {video_id}")
        seen.add(video_id)
        entry_split = str(raw["split"]).lower()
        if entry_split != split:
            raise ValueError(f"{video_id}: split={entry_split!r}, expected {split!r}")
        num_frames = int(raw["num_frames"])
        annotations = tuple(
            GestureSegment.from_manifest(segment, num_frames=num_frames)
            for segment in raw.get("segments", ())
        )
        relative_video = Path(str(raw["video_path"]).replace("\\", "/"))
        entries.append(
            VideoManifest(
                video_id=video_id,
                split=split,
                video_path=(root / relative_video).resolve(),
                fps=float(raw["fps"]),
                num_frames=num_frames,
                subject=subject_key(video_id),
                annotations=annotations,
                metadata=metadata.get(video_id, {}),
            )
        )
    return tuple(entries)


__all__ = ["IPN_CLASS_NAMES", "VideoManifest", "load_manifest"]
