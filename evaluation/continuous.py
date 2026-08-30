"""Aggregate sequence, event, boundary, false-positive and delay metrics."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from evaluation.event_metrics import match_events, precision_recall_f1
from evaluation.levenshtein import edit_distance


def _ordered_classes(events: Sequence[Mapping[str, Any]]) -> Sequence[int]:
    return [
        int(item["class_id"])
        for item in sorted(events, key=lambda value: int(value["start_frame"]))
    ]


def continuous_event_metrics(
    predictions_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
    targets_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
    duration_seconds_by_video: Mapping[str, float],
    *,
    thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    boundary_match_threshold: float = 0.5,
) -> Mapping[str, Any]:
    video_ids = sorted(targets_by_video)
    if set(predictions_by_video) != set(video_ids):
        raise ValueError("prediction/target video ids do not match")
    total_edit = 0
    total_targets = 0
    for video_id in video_ids:
        predicted = _ordered_classes(predictions_by_video[video_id])
        target = _ordered_classes(targets_by_video[video_id])
        total_edit += edit_distance(predicted, target)
        total_targets += len(target)
    levenshtein_accuracy = (
        float(1.0 - total_edit / total_targets) if total_targets else 0.0
    )

    event_f1: Dict[str, Mapping[str, Any]] = {}
    unmatched_at_boundary = 0
    for threshold in thresholds:
        tp = fp = fn = 0
        for video_id in video_ids:
            matches, unmatched_predictions, unmatched_targets = match_events(
                predictions_by_video[video_id],
                targets_by_video[video_id],
                threshold=float(threshold),
            )
            tp += len(matches)
            fp += len(unmatched_predictions)
            fn += len(unmatched_targets)
        rates = precision_recall_f1(tp, fp, fn)
        event_f1[f"{float(threshold):.1f}"] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            **rates,
        }
        if abs(float(threshold) - float(boundary_match_threshold)) < 1.0e-9:
            unmatched_at_boundary = fp

    start_errors = []
    end_errors = []
    start_latencies = []
    emission_from_start = []
    completion_delays = []
    matched = 0
    missed = 0
    for video_id in video_ids:
        predictions = predictions_by_video[video_id]
        targets = targets_by_video[video_id]
        matches, _, unmatched_targets = match_events(
            predictions, targets, threshold=boundary_match_threshold
        )
        matched += len(matches)
        missed += len(unmatched_targets)
        for match in matches:
            prediction = predictions[match.prediction_index]
            target = targets[match.target_index]
            start_delta = int(prediction["start_frame"]) - int(target["start_frame"])
            end_delta = int(prediction["end_frame_exclusive"]) - int(
                target["end_frame_exclusive"]
            )
            emitted_at = int(
                prediction.get("emitted_at_frame", prediction["end_frame_exclusive"])
            )
            start_errors.append(abs(start_delta))
            end_errors.append(abs(end_delta))
            start_latencies.append(start_delta)
            emission_from_start.append(emitted_at - int(target["start_frame"]))
            completion_delays.append(
                emitted_at - int(target["end_frame_exclusive"])
            )
    total_minutes = sum(float(duration_seconds_by_video[key]) for key in video_ids) / 60.0

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "levenshtein_accuracy": levenshtein_accuracy,
        "edit_distance_total": total_edit,
        "ground_truth_events": total_targets,
        "predicted_events": sum(len(predictions_by_video[key]) for key in video_ids),
        "event_f1": event_f1,
        "false_positive_per_minute": (
            float(unmatched_at_boundary / total_minutes) if total_minutes > 0 else 0.0
        ),
        "duration_minutes": total_minutes,
        "boundary": {
            "match_threshold": float(boundary_match_threshold),
            "matched_events": matched,
            "missed_events": missed,
            "start_mae_frames": mean(start_errors),
            "end_mae_frames": mean(end_errors),
        },
        "delay": {
            "start_trigger_latency_frames": mean(start_latencies),
            "emission_latency_from_start_frames": mean(emission_from_start),
            "completion_delay_frames": mean(completion_delays),
        },
    }


__all__ = ["continuous_event_metrics"]
