"""Continuous-video targets, feature cache and stratified causal clips."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.annotations import GestureSegment
from datasets.ipn_skeleton import IPNVideoDataset, load_manifest
from preprocessing.feature_builder import FeatureBuilderConfig, SkeletonFeatureBuilder
from preprocessing.augmentation import SkeletonAugmentationConfig, SkeletonFeatureAugmenter
from utils.io import atomic_json, file_sha256


@dataclass(frozen=True)
class FrameTargets:
    """Dense continuous targets; class 0 is Background, gestures are 1..13."""

    frame_labels: np.ndarray
    start_targets: np.ndarray
    end_targets: np.ndarray


def build_frame_targets(
    num_frames: int,
    annotations: Sequence[GestureSegment],
    *,
    boundary_radius: int = 2,
) -> FrameTargets:
    frames = int(num_frames)
    radius = int(boundary_radius)
    if frames <= 0:
        raise ValueError("num_frames must be positive")
    if radius < 0:
        raise ValueError("boundary_radius must be non-negative")
    labels = np.zeros(frames, dtype=np.int64)
    starts = np.zeros(frames, dtype=np.float32)
    ends = np.zeros(frames, dtype=np.float32)
    for segment in annotations:
        start = int(segment.start_frame)
        end = int(segment.end_frame_exclusive)
        if not 0 <= start < end <= frames:
            raise ValueError(f"segment [{start},{end}) is outside T={frames}")
        if np.any(labels[start:end] != 0):
            raise ValueError(f"overlapping annotations at [{start},{end})")
        labels[start:end] = int(segment.class_id) + 1
        end_center = end - 1
        starts[max(0, start - radius) : min(frames, start + radius + 1)] = 1.0
        ends[
            max(0, end_center - radius) : min(frames, end_center + radius + 1)
        ] = 1.0
    return FrameTargets(labels, starts, ends)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def continuous_cache_signature(
    project_root: Union[str, Path],
    feature_config: FeatureBuilderConfig,
    *,
    boundary_radius: int,
) -> str:
    root = Path(project_root).resolve()
    metadata_path = root / "data/raw/annotations/metadata.csv"
    if feature_config.include_handedness and not metadata_path.is_file():
        raise FileNotFoundError(
            "P3 handedness features require data/raw/annotations/metadata.csv"
        )
    payload = {
        "schema_version": 1,
        "feature_builder": {
            "preprocessing_profile": feature_config.preprocessing_profile,
            "coordinate_source": feature_config.coordinate_source,
            "motion_lags": list(feature_config.motion_lags),
            "max_hold_frames": feature_config.max_hold_frames,
            "missing_clip_frames": feature_config.missing_clip_frames,
            "include_handedness": feature_config.include_handedness,
        },
        "boundary_radius": int(boundary_radius),
        "train_manifest_sha256": file_sha256(root / "data/manifests/train.json"),
        "test_manifest_sha256": file_sha256(root / "data/manifests/test.json"),
        "metadata_sha256": (
            file_sha256(metadata_path) if feature_config.include_handedness else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _video_cache_paths(directory: Path, video_id: str) -> Mapping[str, Path]:
    video_dir = directory / video_id
    return {
        "directory": video_dir,
        "features": video_dir / "features.npy",
        "frame_labels": video_dir / "frame_labels.npy",
        "start_targets": video_dir / "start_targets.npy",
        "end_targets": video_dir / "end_targets.npy",
    }


def prepare_continuous_cache(
    *,
    project_root: Union[str, Path],
    cache_root: Union[str, Path],
    split: str,
    builder: SkeletonFeatureBuilder,
    boundary_radius: int,
    signature: str,
) -> Mapping[str, Any]:
    """Build immutable per-video NPY files, or reuse a complete matching cache."""

    root = Path(project_root).resolve()
    cache = Path(cache_root).resolve()
    split_dir = cache / str(split)
    index_path = split_dir / "index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("cache_signature") != signature:
            raise ValueError(f"stale continuous cache: {index_path}")
        missing = []
        for item in payload.get("videos", []):
            paths = _video_cache_paths(split_dir, str(item["video_id"]))
            missing.extend(
                str(path)
                for key, path in paths.items()
                if key != "directory" and not path.is_file()
            )
        if missing:
            raise FileNotFoundError(f"partial continuous cache: {missing[:3]}")
        return payload
    if split_dir.exists() and any(split_dir.iterdir()):
        raise FileExistsError(f"partial continuous cache without index: {split_dir}")

    manifests = load_manifest(root, split)
    dataset = IPNVideoDataset(root, split, entries=manifests)
    videos: List[Mapping[str, Any]] = []
    class_counts = np.zeros(14, dtype=np.int64)
    start_positive = 0
    end_positive = 0
    for record in dataset.iter_records():
        feature_batch = builder.build_from_record(record)
        targets = build_frame_targets(
            record.num_frames, record.annotations, boundary_radius=boundary_radius
        )
        paths = _video_cache_paths(split_dir, record.video_id)
        atomic_npy(paths["features"], feature_batch.features)
        atomic_npy(paths["frame_labels"], targets.frame_labels)
        atomic_npy(paths["start_targets"], targets.start_targets)
        atomic_npy(paths["end_targets"], targets.end_targets)
        class_counts += np.bincount(targets.frame_labels, minlength=14)
        start_positive += int(targets.start_targets.sum())
        end_positive += int(targets.end_targets.sum())
        videos.append(
            {
                "video_id": record.video_id,
                "split": record.split,
                "subject": record.subject,
                "num_frames": record.num_frames,
                "fps": record.fps,
                "width": record.width,
                "height": record.height,
                "annotations": [item.as_dict() for item in record.annotations],
            }
        )
        print(
            f"cache {split}: {len(videos):03d}/{len(dataset):03d} {record.video_id}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "cache_signature": signature,
        "split": split,
        "feature_dim": builder.output_dim,
        "feature_names": list(builder.feature_names),
        "total_frames": int(class_counts.sum()),
        "frame_class_counts": class_counts.tolist(),
        "start_positive_frames": start_positive,
        "end_positive_frames": end_positive,
        "videos": videos,
    }
    atomic_json(index_path, payload)
    return payload


class ContinuousCache:
    """Read-only memory-mapped continuous feature cache."""

    def __init__(self, cache_root: Union[str, Path], split: str) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.split = str(split)
        self.split_dir = self.cache_root / self.split
        self.index = json.loads(
            (self.split_dir / "index.json").read_text(encoding="utf-8")
        )
        self.videos: Tuple[Mapping[str, Any], ...] = tuple(self.index["videos"])
        self.feature_dim = int(self.index["feature_dim"])
        self._arrays: Dict[Tuple[int, str], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.videos)

    def array(self, video_index: int, name: str) -> np.ndarray:
        key = (int(video_index), str(name))
        if key not in self._arrays:
            video_id = str(self.videos[video_index]["video_id"])
            path = _video_cache_paths(self.split_dir, video_id)[name]
            self._arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._arrays[key]

    def video_arrays(self, video_index: int) -> Mapping[str, np.ndarray]:
        return {
            name: self.array(video_index, name)
            for name in ("features", "frame_labels", "start_targets", "end_targets")
        }


class StratifiedContinuousClipDataset(Dataset):
    """Deterministic per-epoch clips balanced across continuous contexts."""

    CATEGORIES = ("dynamic", "pointing", "boundary", "background")

    def __init__(
        self,
        cache: ContinuousCache,
        *,
        clip_length: int,
        supervised_length: int,
        samples_per_epoch: int,
        seed: int,
        center_jitter: int = 32,
        missing_augmentation_probability: float = 0.0,
        missing_augmentation_max_frames: int = 12,
        skeleton_augmentation_probability: float = 0.0,
        skeleton_augmentation_config: Optional[SkeletonAugmentationConfig] = None,
    ) -> None:
        self.cache = cache
        self.clip_length = int(clip_length)
        self.supervised_length = int(supervised_length)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.center_jitter = int(center_jitter)
        self.missing_probability = float(missing_augmentation_probability)
        self.missing_max_frames = int(missing_augmentation_max_frames)
        self.skeleton_augmentation_probability = float(
            skeleton_augmentation_probability
        )
        if not 0.0 <= self.skeleton_augmentation_probability <= 1.0:
            raise ValueError("skeleton_augmentation_probability must be in [0,1]")
        self.skeleton_augmenter = (
            SkeletonFeatureAugmenter(
                tuple(str(value) for value in cache.index["feature_names"]),
                skeleton_augmentation_config,
            )
            if skeleton_augmentation_config is not None
            and skeleton_augmentation_config.enabled
            else None
        )
        if not 0 < self.supervised_length <= self.clip_length:
            raise ValueError("supervised_length must be in [1, clip_length]")
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self._candidates = self._build_candidates()
        self._plan: List[Tuple[int, int, str, int]] = []
        self.set_epoch(0)

    def _build_candidates(self) -> Mapping[str, List[Tuple[int, int]]]:
        candidates: Dict[str, List[Tuple[int, int]]] = {
            category: [] for category in self.CATEGORIES
        }
        for video_index, item in enumerate(self.cache.videos):
            for raw in item["annotations"]:
                start = int(raw["start_frame"])
                end = int(raw["end_frame_exclusive"])
                center = (start + end - 1) // 2
                class_id = int(raw["class_id"])
                category = "pointing" if class_id < 2 else "dynamic"
                candidates[category].append((video_index, center))
                candidates["boundary"].append((video_index, start))
                candidates["boundary"].append((video_index, end - 1))
            labels = self.cache.array(video_index, "frame_labels")
            background = np.flatnonzero(np.asarray(labels) == 0)
            for center in background[:: max(1, background.size // 256 or 1)]:
                candidates["background"].append((video_index, int(center)))
        empty = [key for key, values in candidates.items() if not values]
        if empty:
            raise ValueError(f"empty stratified candidate categories: {empty}")
        return candidates

    @property
    def category_counts(self) -> Mapping[str, int]:
        return {key: len(value) for key, value in self._candidates.items()}

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + 1009 * int(epoch))
        plan: List[Tuple[int, int, str, int]] = []
        for index in range(self.samples_per_epoch):
            category = self.CATEGORIES[index % len(self.CATEGORIES)]
            values = self._candidates[category]
            video_index, center = values[int(rng.integers(0, len(values)))]
            jitter = int(rng.integers(-self.center_jitter, self.center_jitter + 1))
            augmentation_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            plan.append((video_index, center + jitter, category, augmentation_seed))
        rng.shuffle(plan)
        self._plan = plan

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        video_index, center, category, augmentation_seed = self._plan[index]
        item = self.cache.videos[video_index]
        frames = int(item["num_frames"])
        context = self.clip_length - self.supervised_length
        clip_start = center - context - self.supervised_length // 2
        clip_start = min(max(0, clip_start), max(0, frames - self.clip_length))
        clip_end = min(frames, clip_start + self.clip_length)
        length = clip_end - clip_start
        arrays = self.cache.video_arrays(video_index)
        features = np.zeros((self.clip_length, self.cache.feature_dim), np.float32)
        labels = np.full(self.clip_length, -100, np.int64)
        starts = np.zeros(self.clip_length, np.float32)
        ends = np.zeros(self.clip_length, np.float32)
        features[:length] = np.asarray(arrays["features"][clip_start:clip_end])
        labels[:length] = np.asarray(arrays["frame_labels"][clip_start:clip_end])
        starts[:length] = np.asarray(arrays["start_targets"][clip_start:clip_end])
        ends[:length] = np.asarray(arrays["end_targets"][clip_start:clip_end])
        supervision = np.zeros(self.clip_length, np.bool_)
        supervision[context : min(self.clip_length, length)] = True

        rng = np.random.default_rng(augmentation_seed)
        if self.missing_probability > 0 and rng.random() < self.missing_probability:
            span = int(rng.integers(1, self.missing_max_frames + 1))
            high = max(1, length - span + 1)
            start = int(rng.integers(0, high))
            features[start : start + span] = 0.0

        if (
            self.skeleton_augmenter is not None
            and rng.random() < self.skeleton_augmentation_probability
        ):
            augmented = self.skeleton_augmenter(
                features,
                labels,
                starts,
                ends,
                supervision,
                seed=augmentation_seed,
            )
            features = augmented.features
            labels = augmented.frame_labels
            starts = augmented.start_targets
            ends = augmented.end_targets
            supervision = augmented.supervision_mask

        return {
            "features": torch.from_numpy(features),
            "frame_labels": torch.from_numpy(labels),
            "start_targets": torch.from_numpy(starts),
            "end_targets": torch.from_numpy(ends),
            "supervision_mask": torch.from_numpy(supervision),
            "video_index": video_index,
            "clip_start": clip_start,
            "category": category,
        }


__all__ = [
    "ContinuousCache",
    "FrameTargets",
    "StratifiedContinuousClipDataset",
    "build_frame_targets",
    "continuous_cache_signature",
    "prepare_continuous_cache",
]
