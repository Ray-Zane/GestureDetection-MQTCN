"""Full-video final-model evaluation with frame/query/fusion outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from datasets.ipn_manifest import IPN_CLASS_NAMES
from datasets.memory_query import EncodedContinuousCache, collate_query_video_batch
from engine.model_trainer import MemoryQueryLoss
from evaluation.classification import classification_metrics
from evaluation.memory_query import continuous_event_metrics_v2
from models.gesture_detection_mqtcn import GestureDetectionMQTCN
from streaming.decoder import DecoderConfig, StreamingDecoder
from streaming.query_tracker import (
    QueryDecoderConfig,
    decode_query_events,
    fuse_frame_and_query_events,
)
from utils.io import atomic_json


def compact_mqtcn_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    primary = metrics["outputs"]["fusion"]
    f1 = primary["event_f1"]
    return {
        "loss_total": metrics["query_loss"]["loss_total"],
        "loss_query_cls": metrics["query_loss"]["loss_query_cls"],
        "loss_boundary_l1": metrics["query_loss"]["loss_boundary_l1"],
        "loss_tiou": metrics["query_loss"]["loss_tiou"],
        "frame_macro_f1": metrics["frame"]["macro_f1"],
        "levenshtein_accuracy": primary["levenshtein_accuracy"],
        "event_f1_at_0_3": f1["0.3"]["f1"],
        "event_f1_at_0_5": f1["0.5"]["f1"],
        "event_recall_at_0_5": f1["0.5"]["recall"],
        "event_f1_at_0_7": f1["0.7"]["f1"],
        "false_positive_per_minute": primary["false_positive_per_minute"],
        "start_mae_frames": primary["boundary"]["start_mae_frames"],
        "end_mae_frames": primary["boundary"]["end_mae_frames"],
        "completion_delay_frames": primary["delay"]["completion_delay_frames"]["mean"],
        "predicted_events": primary["predicted_events"],
        "ground_truth_events": primary["ground_truth_events"],
    }


@torch.inference_mode()
def evaluate_memory_query(
    model: GestureDetectionMQTCN,
    cache: EncodedContinuousCache,
    criterion: MemoryQueryLoss,
    device: torch.device,
    *,
    query_stride: int,
    video_batch_size: int,
    frame_decoder_config: DecoderConfig,
    query_decoder_config: QueryDecoderConfig,
    query_chunk_steps: int = 32,
    predictions_dir: Optional[Path] = None,
    limit_videos: Optional[int] = None,
) -> Mapping[str, Any]:
    model.eval()
    number = len(cache) if limit_videos is None else min(len(cache), int(limit_videos))
    all_indices = list(range(number))
    target_frames: List[np.ndarray] = []
    predicted_frames: List[np.ndarray] = []
    predictions: Dict[str, Dict[str, Sequence[Mapping[str, Any]]]] = {
        "frame_only": {},
        "query_only": {},
        "fusion": {},
    }
    targets_by_video: Dict[str, Sequence[Mapping[str, Any]]] = {}
    durations: Dict[str, float] = {}
    loss_sums: Dict[str, float] = {}
    count_keys = {
        "matched_queries",
        "unmatched_queries",
        "uncensored_matches",
        "matched_class_correct",
        "unmatched_class_correct",
        "target_steps",
        "empty_steps",
    }
    loss_weight = 0
    query_slots = gesture_slots = 0
    size = max(1, int(video_batch_size))
    for group_start in range(0, number, size):
        group = all_indices[group_start : group_start + size]
        batch = collate_query_video_batch(
            cache,
            group,
            query_stride=query_stride,
            memory_length=model.frame_memory_length,
            num_queries=model.num_queries,
            device=device,
        )
        collected = {key: [] for key in ("pred_logits", "pred_intervals")}
        total_steps = int(batch["query_times"].shape[1])
        chunk = max(1, int(query_chunk_steps))
        for step_start in range(0, total_steps, chunk):
            step_end = min(total_steps, step_start + chunk)
            output = model.query_sequence(
                batch["encoded"],
                batch["memory_valid"],
                batch["query_times"][:, step_start:step_end],
                active_mask=batch["query_step_valid"][:, step_start:step_end],
            )
            for key in collected:
                collected[key].append(output[key])
        raw = {key: torch.cat(value, dim=1) for key, value in collected.items()}
        loss = criterion(raw, batch)
        weight = int(batch["query_step_valid"].sum().item())
        for key, value in loss.items():
            multiplier = 1.0 if key in count_keys else float(max(1, weight))
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.item()) * multiplier
        loss_weight += max(1, weight)
        class_predictions = raw["pred_logits"].argmax(dim=-1)
        valid_slots = batch["query_step_valid"].unsqueeze(-1).expand_as(class_predictions)
        query_slots += int(valid_slots.sum().item())
        gesture_slots += int(((class_predictions != 0) & valid_slots).sum().item())

        for item, video_index in enumerate(group):
            metadata = cache.videos[video_index]
            video_id = str(metadata["video_id"])
            length = int(metadata["num_frames"])
            steps = int(batch["query_step_valid"][item].sum().item())
            frame_logits = np.asarray(cache.array(video_index, "frame_logits"))
            shifted = frame_logits - frame_logits.max(axis=1, keepdims=True)
            frame_probabilities = np.exp(shifted)
            frame_probabilities /= frame_probabilities.sum(axis=1, keepdims=True)
            start_logits = np.asarray(cache.array(video_index, "start_logits"))
            end_logits = np.asarray(cache.array(video_index, "end_logits"))
            frame_events = StreamingDecoder(frame_decoder_config).decode(
                frame_probabilities,
                1.0 / (1.0 + np.exp(-start_logits)),
                1.0 / (1.0 + np.exp(-end_logits)),
            )
            query_events = decode_query_events(
                raw["pred_logits"][item, :steps].float().cpu().numpy(),
                raw["pred_intervals"][item, :steps].float().cpu().numpy(),
                batch["query_times"][item, :steps].cpu().numpy(),
                memory_length=model.frame_memory_length,
                config=query_decoder_config,
            )
            fusion_events = fuse_frame_and_query_events(
                frame_events, query_events, config=query_decoder_config
            )
            predictions["frame_only"][video_id] = frame_events
            predictions["query_only"][video_id] = query_events
            predictions["fusion"][video_id] = fusion_events
            targets_by_video[video_id] = list(metadata["annotations"])
            durations[video_id] = length / float(metadata["fps"])
            labels = np.asarray(cache.source.array(video_index, "frame_labels"))
            target_frames.append(labels)
            predicted_frames.append(frame_probabilities.argmax(axis=1).astype(np.int64))
            if predictions_dir is not None:
                atomic_json(
                    predictions_dir / f"{video_id}.json",
                    {
                        "schema_version": 2,
                        "video_id": video_id,
                        "num_frames": length,
                        "fps": float(metadata["fps"]),
                        "frame_only": frame_events,
                        "query_only": query_events,
                        "fusion": fusion_events,
                    },
                )
    frame_metrics = classification_metrics(
        np.concatenate(target_frames),
        np.concatenate(predicted_frames),
        ("Background",) + tuple(IPN_CLASS_NAMES),
    )
    output_metrics = {
        key: continuous_event_metrics_v2(values, targets_by_video, durations)
        for key, values in predictions.items()
    }
    return {
        "schema_version": 2,
        "evaluated_videos": number,
        "evaluated_frames": int(sum(len(value) for value in target_frames)),
        "frame": frame_metrics,
        "query_loss": {
            key: (value if key in count_keys else value / max(1, loss_weight))
            for key, value in loss_sums.items()
        },
        "query_diagnostics": {
            "slots": query_slots,
            "predicted_gesture_slots": gesture_slots,
            "predicted_gesture_slot_rate": (
                float(gesture_slots / query_slots) if query_slots else 0.0
            ),
        },
        "outputs": output_metrics,
        "predictions_by_video": predictions,
    }


__all__ = ["compact_mqtcn_metrics", "evaluate_memory_query"]
