"""Read full IPN Hand skeleton streams without copying source data."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .annotations import GestureSegment, subject_key


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

REQUIRED_NPZ_FIELDS: Tuple[str, ...] = (
    "image_landmarks",
    "world_landmarks",
    "valid_mask",
    "handedness",
    "handedness_score",
    "frame_index",
    "timestamp_ms",
    "fps",
    "width",
    "height",
    "video_id",
    "split",
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


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    split: str
    subject: str
    video_path: Path
    skeleton_path: Path
    image_landmarks: np.ndarray
    world_landmarks: np.ndarray
    valid_mask: np.ndarray
    handedness: np.ndarray
    handedness_score: np.ndarray
    frame_index: np.ndarray
    timestamp_ms: np.ndarray
    fps: float
    width: int
    height: int
    annotations: Tuple[GestureSegment, ...]
    metadata: Mapping[str, str]

    @property
    def num_frames(self) -> int:
        return int(self.valid_mask.shape[0])


def _read_json_list(path: Path) -> List[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise TypeError(f"manifest must be a list: {path}")
    return payload


def load_metadata(path: Path) -> Dict[str, Mapping[str, str]]:
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
    metadata = load_metadata(root / "data" / "raw" / "annotations" / "metadata.csv")
    entries: List[VideoManifest] = []
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


def verify_subject_disjoint(
    train: Sequence[VideoManifest], test: Sequence[VideoManifest]
) -> Mapping[str, Any]:
    train_subjects = sorted({entry.subject for entry in train})
    test_subjects = sorted({entry.subject for entry in test})
    overlap = sorted(set(train_subjects).intersection(test_subjects))
    return {
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "overlap": overlap,
        "is_disjoint": not overlap,
    }


class IPNVideoDataset(Sequence[VideoRecord]):
    """Lazy full-video adapter over manifest entries and raw NPZ archives."""

    def __init__(
        self,
        project_root: Union[str, Path],
        split: str,
        *,
        entries: Optional[Sequence[VideoManifest]] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.split = str(split).lower()
        self.entries = tuple(entries or load_manifest(self.project_root, self.split))
        if any(entry.split != self.split for entry in self.entries):
            raise ValueError("dataset entries contain a mismatched split")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> VideoRecord:
        entry = self.entries[index]
        skeleton_path = (
            self.project_root
            / "data"
            / "skeleton_raw"
            / self.split
            / f"{entry.video_id}.npz"
        )
        if not skeleton_path.is_file():
            raise FileNotFoundError(f"missing skeleton archive: {skeleton_path}")
        with np.load(skeleton_path, allow_pickle=False) as archive:
            missing = sorted(set(REQUIRED_NPZ_FIELDS).difference(archive.files))
            if missing:
                raise KeyError(f"{skeleton_path} missing fields: {missing}")
            image = np.asarray(archive["image_landmarks"], dtype=np.float32).copy()
            world = np.asarray(archive["world_landmarks"], dtype=np.float32).copy()
            valid = np.asarray(archive["valid_mask"], dtype=np.bool_).copy()
            handedness = np.asarray(archive["handedness"]).astype("U8", copy=True)
            handedness_score = np.asarray(
                archive["handedness_score"], dtype=np.float32
            ).copy()
            frame_index = np.asarray(archive["frame_index"], dtype=np.int64).copy()
            timestamp_ms = np.asarray(archive["timestamp_ms"], dtype=np.float64).copy()
            fps = float(np.asarray(archive["fps"]).item())
            width = int(np.asarray(archive["width"]).item())
            height = int(np.asarray(archive["height"]).item())
            archive_video_id = str(np.asarray(archive["video_id"]).item())
            archive_split = str(np.asarray(archive["split"]).item()).lower()
        self._validate_arrays(
            entry,
            image=image,
            world=world,
            valid=valid,
            handedness=handedness,
            handedness_score=handedness_score,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            fps=fps,
            width=width,
            height=height,
            archive_video_id=archive_video_id,
            archive_split=archive_split,
        )
        return VideoRecord(
            video_id=entry.video_id,
            split=entry.split,
            subject=entry.subject,
            video_path=entry.video_path,
            skeleton_path=skeleton_path.resolve(),
            image_landmarks=image,
            world_landmarks=world,
            valid_mask=valid,
            handedness=handedness,
            handedness_score=handedness_score,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            fps=fps,
            width=width,
            height=height,
            annotations=entry.annotations,
            metadata=entry.metadata,
        )

    @staticmethod
    def _validate_arrays(
        entry: VideoManifest,
        *,
        image: np.ndarray,
        world: np.ndarray,
        valid: np.ndarray,
        handedness: np.ndarray,
        handedness_score: np.ndarray,
        frame_index: np.ndarray,
        timestamp_ms: np.ndarray,
        fps: float,
        width: int,
        height: int,
        archive_video_id: str,
        archive_split: str,
    ) -> None:
        frames = entry.num_frames
        if image.shape != (frames, 21, 3):
            raise ValueError(f"{entry.video_id}: invalid image shape {image.shape}")
        if world.shape != (frames, 21, 3):
            raise ValueError(f"{entry.video_id}: invalid world shape {world.shape}")
        for name, array in (
            ("valid_mask", valid),
            ("handedness", handedness),
            ("handedness_score", handedness_score),
            ("frame_index", frame_index),
            ("timestamp_ms", timestamp_ms),
        ):
            if array.shape != (frames,):
                raise ValueError(f"{entry.video_id}: {name} shape is {array.shape}")
        if archive_video_id != entry.video_id or archive_split != entry.split:
            raise ValueError(
                f"{entry.video_id}: NPZ identity mismatch "
                f"({archive_video_id!r}, {archive_split!r})"
            )
        if not np.array_equal(frame_index, np.arange(frames, dtype=np.int64)):
            raise ValueError(f"{entry.video_id}: frame_index is not 0..T-1")
        if not np.isfinite(image[valid]).all() or not np.isfinite(world[valid]).all():
            raise FloatingPointError(f"{entry.video_id}: non-finite valid landmarks")
        if not np.isfinite(handedness_score).all() or not np.isfinite(timestamp_ms).all():
            raise FloatingPointError(f"{entry.video_id}: non-finite metadata arrays")
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"{entry.video_id}: invalid fps/size ({fps}, {width}, {height})"
            )
        if abs(fps - entry.fps) > 1e-3:
            raise ValueError(f"{entry.video_id}: manifest/NPZ fps mismatch")

    def iter_records(self) -> Iterator[VideoRecord]:
        for index in range(len(self)):
            yield self[index]


__all__ = [
    "IPN_CLASS_NAMES",
    "IPNVideoDataset",
    "VideoManifest",
    "VideoRecord",
    "load_manifest",
    "load_metadata",
    "verify_subject_disjoint",
]
