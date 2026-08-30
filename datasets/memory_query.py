"""Frozen-backbone embedding cache and full-video sequential Query batches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from datasets.continuous import ContinuousCache, atomic_npy
from models.baseline import ContinuousBaseline
from models.query.targets import build_completed_event_targets, query_times_for_sequence
from utils.io import atomic_json


ARRAY_NAMES = (
    "encoded",
    "frame_logits",
    "start_logits",
    "end_logits",
    "memory_valid",
)


def encoded_cache_signature(
    continuous_signature: str, checkpoint_sha256: str
) -> str:
    payload = {
        "schema_version": 1,
        "continuous_signature": str(continuous_signature),
        "checkpoint_sha256": str(checkpoint_sha256).lower(),
        "precision": "strict_fp32_tf32_disabled",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _paths(root: Path, split: str, video_id: str) -> Mapping[str, Path]:
    directory = root / split / video_id
    return {name: directory / f"{name}.npy" for name in ARRAY_NAMES}


@torch.inference_mode()
def prepare_encoded_cache(
    *,
    cache_root: Union[str, Path],
    source: ContinuousCache,
    baseline: ContinuousBaseline,
    checkpoint_sha256: str,
    device: torch.device,
    video_batch_size: int,
) -> Mapping[str, Any]:
    root = Path(cache_root).resolve()
    split_root = root / source.split
    index_path = split_root / "index.json"
    signature = encoded_cache_signature(
        str(source.index["cache_signature"]), checkpoint_sha256
    )
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("cache_signature") != signature:
            raise ValueError(f"stale MQ encoded cache: {index_path}")
        missing = [
            str(path)
            for item in payload["videos"]
            for path in _paths(root, source.split, str(item["video_id"])).values()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"partial MQ encoded cache: {missing[:3]}")
        return payload
    if split_root.exists() and any(split_root.iterdir()):
        raise FileExistsError(f"partial MQ encoded cache without index: {split_root}")

    baseline.eval()
    valid_column = list(source.index["feature_names"]).index("quality.valid")
    videos: List[Mapping[str, Any]] = []
    size = max(1, int(video_batch_size))
    for group_start in range(0, len(source), size):
        indices = list(range(group_start, min(len(source), group_start + size)))
        sequences = [np.asarray(source.array(index, "features")) for index in indices]
        lengths = [int(value.shape[0]) for value in sequences]
        maximum = max(lengths)
        padded = np.zeros(
            (len(indices), maximum, source.feature_dim), dtype=np.float32
        )
        for item, sequence in enumerate(sequences):
            padded[item, : sequence.shape[0]] = sequence
        features = torch.from_numpy(padded).to(device)
        encoded = baseline.encode(features)
        heads = baseline.head(encoded)
        for item, (video_index, length) in enumerate(zip(indices, lengths)):
            metadata = source.videos[video_index]
            video_id = str(metadata["video_id"])
            paths = _paths(root, source.split, video_id)
            atomic_npy(paths["encoded"], encoded[item, :length].float().cpu().numpy())
            atomic_npy(
                paths["frame_logits"],
                heads["frame_logits"][item, :length].float().cpu().numpy(),
            )
            atomic_npy(
                paths["start_logits"],
                heads["start_logits"][item, :length].float().cpu().numpy(),
            )
            atomic_npy(
                paths["end_logits"],
                heads["end_logits"][item, :length].float().cpu().numpy(),
            )
            atomic_npy(
                paths["memory_valid"],
                (sequences[item][:, valid_column] >= 0.5).astype(np.bool_),
            )
            videos.append(
                {
                    "video_id": video_id,
                    "num_frames": length,
                    "encoded_dim": int(encoded.shape[-1]),
                }
            )
            print(
                f"MQ cache {source.split}: {video_index + 1:03d}/{len(source):03d} {video_id}",
                flush=True,
            )
    payload = {
        "schema_version": 1,
        "cache_signature": signature,
        "continuous_signature": source.index["cache_signature"],
        "checkpoint_sha256": str(checkpoint_sha256).lower(),
        "precision": "strict_fp32_tf32_disabled",
        "split": source.split,
        "encoded_dim": int(baseline.backbone.hidden_dim),
        "videos": videos,
    }
    atomic_json(index_path, payload)
    return payload


class EncodedContinuousCache:
    def __init__(
        self,
        cache_root: Union[str, Path],
        source: ContinuousCache,
    ) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.source = source
        self.split = source.split
        self.index = json.loads(
            (self.cache_root / self.split / "index.json").read_text(encoding="utf-8")
        )
        if [item["video_id"] for item in self.index["videos"]] != [
            item["video_id"] for item in source.videos
        ]:
            raise ValueError("encoded and continuous cache video order differs")
        self._arrays: Dict[Tuple[int, str], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.source)

    @property
    def videos(self) -> Tuple[Mapping[str, Any], ...]:
        return self.source.videos

    def array(self, video_index: int, name: str) -> np.ndarray:
        if name not in ARRAY_NAMES:
            raise KeyError(name)
        key = (int(video_index), str(name))
        if key not in self._arrays:
            video_id = str(self.videos[video_index]["video_id"])
            self._arrays[key] = np.load(
                _paths(self.cache_root, self.split, video_id)[name],
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._arrays[key]


def collate_query_video_batch(
    cache: EncodedContinuousCache,
    video_indices: Sequence[int],
    *,
    query_stride: int,
    memory_length: int,
    num_queries: int,
    device: torch.device,
) -> Mapping[str, Any]:
    indices = [int(value) for value in video_indices]
    if not indices:
        raise ValueError("video_indices cannot be empty")
    lengths = [int(cache.videos[index]["num_frames"]) for index in indices]
    maximum_frames = max(lengths)
    schedules = [query_times_for_sequence(length, query_stride) for length in lengths]
    maximum_steps = max(len(value) for value in schedules)
    hidden = int(cache.index["encoded_dim"])
    encoded = np.zeros((len(indices), maximum_frames, hidden), dtype=np.float32)
    memory_valid = np.zeros((len(indices), maximum_frames), dtype=np.bool_)
    query_times = np.ones((len(indices), maximum_steps), dtype=np.int64)
    previous_times = np.zeros((len(indices), maximum_steps), dtype=np.int64)
    step_valid = np.zeros((len(indices), maximum_steps), dtype=np.bool_)
    target_classes = np.full(
        (len(indices), maximum_steps, num_queries), -1, dtype=np.int64
    )
    target_boundaries = np.zeros(
        (len(indices), maximum_steps, num_queries, 2), dtype=np.float32
    )
    target_valid = np.zeros(
        (len(indices), maximum_steps, num_queries), dtype=np.bool_
    )
    left_censored = np.zeros_like(target_valid)
    boundary_mask = np.zeros(
        (len(indices), maximum_steps, num_queries, 2), dtype=np.bool_
    )
    tiou_valid = np.zeros_like(target_valid)
    event_ids = np.full_like(target_classes, -1)
    global_intervals = np.full(
        (len(indices), maximum_steps, num_queries, 2), -1, dtype=np.int64
    )
    for item, (video_index, length, schedule) in enumerate(
        zip(indices, lengths, schedules)
    ):
        encoded[item, :length] = np.asarray(cache.array(video_index, "encoded"))
        memory_valid[item, :length] = np.asarray(
            cache.array(video_index, "memory_valid")
        )
        targets = build_completed_event_targets(
            cache.videos[video_index]["annotations"],
            schedule,
            memory_length=memory_length,
            max_targets=num_queries,
            previous_query_time=0,
            stream_start=0,
        )
        steps = len(schedule)
        if int(targets.valid.sum()) != len(cache.videos[video_index]["annotations"]):
            raise AssertionError("full-video Query targets do not uniquely cover annotations")
        query_times[item, :steps] = targets.query_times
        previous_times[item, :steps] = targets.previous_query_times
        step_valid[item, :steps] = True
        target_classes[item, :steps] = targets.classes
        target_boundaries[item, :steps] = targets.boundaries
        target_valid[item, :steps] = targets.valid
        left_censored[item, :steps] = targets.left_censored
        boundary_mask[item, :steps] = targets.boundary_mask
        tiou_valid[item, :steps] = targets.tiou_valid
        event_ids[item, :steps] = targets.event_ids
        global_intervals[item, :steps] = targets.global_intervals
    tensor_fields = {
        "encoded": encoded,
        "memory_valid": memory_valid,
        "query_times": query_times,
        "previous_query_times": previous_times,
        "query_step_valid": step_valid,
        "target_classes": target_classes,
        "target_boundaries": target_boundaries,
        "target_valid": target_valid,
        "left_censored": left_censored,
        "boundary_mask": boundary_mask,
        "tiou_valid": tiou_valid,
        "event_ids": event_ids,
        "global_intervals": global_intervals,
    }
    return {
        **{
            key: torch.from_numpy(value).to(device)
            for key, value in tensor_fields.items()
        },
        "video_indices": indices,
        "lengths": lengths,
        "metadata": [cache.videos[index] for index in indices],
    }


__all__ = [
    "EncodedContinuousCache",
    "collate_query_video_batch",
    "encoded_cache_signature",
    "prepare_encoded_cache",
]
