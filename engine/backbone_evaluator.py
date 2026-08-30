"""Chunked full-video backbone inference and continuous metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from datasets.continuous import ContinuousCache
from datasets.ipn_manifest import IPN_CLASS_NAMES
from engine.backbone_trainer import BaselineLoss
from evaluation.classification import classification_metrics
from evaluation.continuous import continuous_event_metrics
from streaming.decoder import DecoderConfig, StreamingDecoder
from utils.io import atomic_json


def infer_video_logits(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    *,
    chunk_size: int,
    amp: bool,
) -> Mapping[str, np.ndarray]:
    """Use RF-1 left context per chunk so output equals a full causal forward pass."""

    frames = int(features.shape[0])
    receptive_field = int(getattr(model, "receptive_field", 1))
    context = max(0, receptive_field - 1)
    outputs: Dict[str, List[np.ndarray]] = {
        "frame_logits": [],
        "start_logits": [],
        "end_logits": [],
    }
    for start in range(0, frames, int(chunk_size)):
        end = min(frames, start + int(chunk_size))
        context_start = max(0, start - context)
        tensor = torch.from_numpy(
            np.asarray(features[context_start:end]).copy()
        ).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            raw = model(tensor)
        offset = start - context_start
        length = end - start
        for key in outputs:
            value = raw[key][0, offset : offset + length].float().cpu().numpy()
            outputs[key].append(value)
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def infer_video_batch_logits(
    model: nn.Module,
    feature_sequences: Sequence[np.ndarray],
    device: torch.device,
    *,
    amp: bool,
) -> Sequence[Mapping[str, np.ndarray]]:
    """Batch variable-length videos; right padding cannot affect causal prefixes."""

    if not feature_sequences:
        return []
    lengths = [int(value.shape[0]) for value in feature_sequences]
    dimension = int(feature_sequences[0].shape[1])
    maximum = max(lengths)
    padded = np.zeros((len(feature_sequences), maximum, dimension), dtype=np.float32)
    for index, sequence in enumerate(feature_sequences):
        if sequence.ndim != 2 or sequence.shape[1] != dimension:
            raise ValueError("batched video features need matching [T,D] dimensions")
        padded[index, : lengths[index]] = np.asarray(sequence)
    tensor = torch.from_numpy(padded).to(device)
    with torch.autocast(device_type=device.type, enabled=amp):
        raw = model(tensor)
    results = []
    for index, length in enumerate(lengths):
        results.append(
            {
                key: value[index, :length].float().cpu().numpy()
                for key, value in raw.items()
            }
        )
    return results


def compact_continuous_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    frame = metrics["frame"]
    events = metrics["events"]
    return {
        "loss_total": metrics["loss_total"],
        "loss_frame_cls": metrics["loss_frame_cls"],
        "loss_start": metrics["loss_start"],
        "loss_end": metrics["loss_end"],
        "frame_accuracy": frame["accuracy"],
        "frame_macro_f1": frame["macro_f1"],
        "levenshtein_accuracy": events["levenshtein_accuracy"],
        "event_f1_at_0_3": events["event_f1"]["0.3"]["f1"],
        "event_f1_at_0_5": events["event_f1"]["0.5"]["f1"],
        "event_f1_at_0_7": events["event_f1"]["0.7"]["f1"],
        "false_positive_per_minute": events["false_positive_per_minute"],
        "start_mae_frames": events["boundary"]["start_mae_frames"],
        "end_mae_frames": events["boundary"]["end_mae_frames"],
        "completion_delay_frames": events["delay"]["completion_delay_frames"],
        "predicted_events": events["predicted_events"],
        "ground_truth_events": events["ground_truth_events"],
    }


@torch.inference_mode()
def evaluate_full_videos(
    model: nn.Module,
    cache: ContinuousCache,
    criterion: BaselineLoss,
    device: torch.device,
    decoder_config: DecoderConfig,
    *,
    chunk_size: int,
    amp: bool,
    video_batch_size: int = 8,
    predictions_dir: Optional[Path] = None,
    limit_videos: Optional[int] = None,
) -> Mapping[str, Any]:
    model.eval()
    target_frames: List[np.ndarray] = []
    predicted_frames: List[np.ndarray] = []
    predictions_by_video: Dict[str, Sequence[Mapping[str, Any]]] = {}
    targets_by_video: Dict[str, Sequence[Mapping[str, Any]]] = {}
    durations: Dict[str, float] = {}
    loss_totals = {key: 0.0 for key in ("loss_total", "loss_frame_cls", "loss_start", "loss_end")}
    total_frames = 0
    number = len(cache) if limit_videos is None else min(len(cache), int(limit_videos))
    batch_size = max(1, int(video_batch_size))
    for group_start in range(0, number, batch_size):
        indices = list(range(group_start, min(number, group_start + batch_size)))
        array_group = [cache.video_arrays(index) for index in indices]
        logits_group = infer_video_batch_logits(
            model,
            [arrays["features"] for arrays in array_group],
            device,
            amp=amp,
        )
        for video_index, arrays, logits in zip(indices, array_group, logits_group):
            metadata = cache.videos[video_index]
            labels = np.asarray(arrays["frame_labels"], dtype=np.int64)
            starts = np.asarray(arrays["start_targets"], dtype=np.float32)
            ends = np.asarray(arrays["end_targets"], dtype=np.float32)
            frames = labels.shape[0]
            loss_batch = {
                "frame_labels": torch.from_numpy(labels.copy()).unsqueeze(0).to(device),
                "start_targets": torch.from_numpy(starts.copy()).unsqueeze(0).to(device),
                "end_targets": torch.from_numpy(ends.copy()).unsqueeze(0).to(device),
                "supervision_mask": torch.ones((1, frames), dtype=torch.bool, device=device),
            }
            output_batch = {
                key: torch.from_numpy(value.copy()).unsqueeze(0).to(device)
                for key, value in logits.items()
            }
            loss = criterion(output_batch, loss_batch)
            for key in loss_totals:
                loss_totals[key] += float(loss[key].item()) * frames
            total_frames += frames

            shifted = logits["frame_logits"] - logits["frame_logits"].max(axis=1, keepdims=True)
            frame_probabilities = np.exp(shifted)
            frame_probabilities /= frame_probabilities.sum(axis=1, keepdims=True)
            start_probabilities = 1.0 / (1.0 + np.exp(-logits["start_logits"]))
            end_probabilities = 1.0 / (1.0 + np.exp(-logits["end_logits"]))
            decoder = StreamingDecoder(decoder_config)
            events = decoder.decode(
                frame_probabilities, start_probabilities, end_probabilities
            )
            video_id = str(metadata["video_id"])
            predictions_by_video[video_id] = events
            targets_by_video[video_id] = list(metadata["annotations"])
            durations[video_id] = frames / float(metadata["fps"])
            target_frames.append(labels)
            predicted_frames.append(frame_probabilities.argmax(axis=1).astype(np.int64))
            if predictions_dir is not None:
                atomic_json(
                    predictions_dir / f"{video_id}.json",
                    {
                        "video_id": video_id,
                        "num_frames": frames,
                        "fps": float(metadata["fps"]),
                        "events": events,
                    },
                )
    if total_frames <= 0:
        raise RuntimeError("evaluation cache has no frames")
    frame_metrics = classification_metrics(
        np.concatenate(target_frames),
        np.concatenate(predicted_frames),
        ("Background",) + tuple(IPN_CLASS_NAMES),
    )
    event_metrics = continuous_event_metrics(
        predictions_by_video, targets_by_video, durations
    )
    return {
        **{key: value / total_frames for key, value in loss_totals.items()},
        "evaluated_videos": number,
        "evaluated_frames": total_frames,
        "frame": frame_metrics,
        "events": event_metrics,
    }


__all__ = [
    "compact_continuous_metrics",
    "evaluate_full_videos",
    "infer_video_batch_logits",
    "infer_video_logits",
]
