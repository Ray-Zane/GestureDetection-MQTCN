"""Versioned continuous event metrics for Memory-Query predictions."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from evaluation.continuous import continuous_event_metrics
from evaluation.event_metrics import match_events
from evaluation.levenshtein import edit_distance


def _ordered(events: Sequence[Mapping[str, Any]]) -> Sequence[int]:
    return [
        int(item["class_id"])
        for item in sorted(events, key=lambda value: int(value["start_frame"]))
    ]


def sequence_edit_operations(
    target: Sequence[int], predicted: Sequence[int]
) -> Mapping[str, int]:
    """Minimum edit decomposition from GT to prediction (insert=extra prediction)."""

    first = list(target)
    second = list(predicted)
    rows, columns = len(first) + 1, len(second) + 1
    table = np.zeros((rows, columns), dtype=np.int32)
    table[:, 0] = np.arange(rows)
    table[0, :] = np.arange(columns)
    operation = np.zeros((rows, columns), dtype=np.int8)
    operation[1:, 0] = 2  # deletion
    operation[0, 1:] = 1  # insertion
    for row in range(1, rows):
        for column in range(1, columns):
            if first[row - 1] == second[column - 1]:
                table[row, column] = table[row - 1, column - 1]
                operation[row, column] = 0
                continue
            choices = (
                (table[row - 1, column - 1] + 1, 3),
                (table[row - 1, column] + 1, 2),
                (table[row, column - 1] + 1, 1),
            )
            value, code = min(choices, key=lambda item: (item[0], -item[1]))
            table[row, column] = value
            operation[row, column] = code
    insertion = deletion = substitution = 0
    row, column = len(first), len(second)
    while row or column:
        code = int(operation[row, column])
        if code == 0:
            row -= 1
            column -= 1
        elif code == 1:
            insertion += 1
            column -= 1
        elif code == 2:
            deletion += 1
            row -= 1
        else:
            substitution += 1
            row -= 1
            column -= 1
    return {
        "insertions": insertion,
        "deletions": deletion,
        "substitutions": substitution,
        "total": insertion + deletion + substitution,
    }


def continuous_event_metrics_v2(
    predictions_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
    targets_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
    duration_seconds_by_video: Mapping[str, float],
) -> Mapping[str, Any]:
    result = dict(
        continuous_event_metrics(
            predictions_by_video, targets_by_video, duration_seconds_by_video
        )
    )
    edits: Dict[str, int] = {
        "insertions": 0,
        "deletions": 0,
        "substitutions": 0,
        "total": 0,
    }
    per_video = {}
    completion = []
    boundary_bias = []
    start_absolute_error = []
    end_absolute_error = []
    scheduling_wait = []
    for video_id in sorted(targets_by_video):
        target_classes = _ordered(targets_by_video[video_id])
        predicted_classes = _ordered(predictions_by_video[video_id])
        operations = sequence_edit_operations(target_classes, predicted_classes)
        for key in edits:
            edits[key] += int(operations[key])
        per_video[video_id] = {
            "ground_truth_events": len(target_classes),
            "predicted_events": len(predicted_classes),
            "edit_distance": int(operations["total"]),
            **operations,
            "levenshtein_accuracy": (
                float(1.0 - operations["total"] / len(target_classes))
                if target_classes
                else (1.0 if not predicted_classes else 0.0)
            ),
        }
        matches, _, _ = match_events(
            predictions_by_video[video_id],
            targets_by_video[video_id],
            threshold=0.5,
        )
        for match in matches:
            prediction = predictions_by_video[video_id][match.prediction_index]
            target = targets_by_video[video_id][match.target_index]
            prefix = int(
                prediction.get(
                    "emitted_at_prefix",
                    int(
                        prediction.get(
                            "emitted_at_frame",
                            int(prediction["end_frame_exclusive"]) - 1,
                        )
                    )
                    + 1,
                )
            )
            target_end = int(target["end_frame_exclusive"])
            completion.append(prefix - target_end)
            start_delta = int(prediction["start_frame"]) - int(target["start_frame"])
            end_delta = int(prediction["end_frame_exclusive"]) - target_end
            boundary_bias.append(start_delta)
            start_absolute_error.append(abs(start_delta))
            end_absolute_error.append(abs(end_delta))
            if "emitted_at_prefix" in prediction:
                scheduling_wait.append(prefix - int(prediction["end_frame_exclusive"]))

    def summary(values: Sequence[float]) -> Mapping[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "samples": 0}
        return {
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "samples": int(array.size),
        }

    legacy_delay = dict(result["delay"])
    result["metric_schema_version"] = 2
    result["levenshtein_accuracy_official_weighted"] = float(
        result["levenshtein_accuracy"]
    )
    result["levenshtein_accuracy_per_video_mean"] = float(
        np.mean([value["levenshtein_accuracy"] for value in per_video.values()])
    ) if per_video else 0.0
    result["sequence_edits"] = edits
    result["per_video"] = per_video
    result["boundary"] = {
        **result["boundary"],
        "start_absolute_error_frames": summary(start_absolute_error),
        "end_absolute_error_frames": summary(end_absolute_error),
    }
    result["delay"] = {
        "start_boundary_bias_frames": summary(boundary_bias),
        "completion_delay_frames": summary(completion),
        "hypothesis_age_frames": summary(scheduling_wait),
        "query_scheduling_wait_frames": summary(scheduling_wait),
        "query_scheduling_wait_deprecated_note": (
            "Deprecated alias: emitted_prefix - predicted_end includes boundary bias; "
            "use hypothesis_age_frames. Pure cadence wait is reported by target audit."
        ),
        "legacy_v1": legacy_delay,
    }
    return result


def paired_bootstrap_levenshtein_delta(
    candidate_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    reference_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    targets: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> Mapping[str, float]:
    video_ids = sorted(targets)
    candidate_edits = np.asarray(
        [
            edit_distance(_ordered(candidate_predictions[key]), _ordered(targets[key]))
            for key in video_ids
        ],
        dtype=np.float64,
    )
    reference_edits = np.asarray(
        [
            edit_distance(_ordered(reference_predictions[key]), _ordered(targets[key]))
            for key in video_ids
        ],
        dtype=np.float64,
    )
    counts = np.asarray([len(targets[key]) for key in video_ids], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        selection = rng.integers(0, len(video_ids), size=len(video_ids))
        denominator = counts[selection].sum()
        candidate = 1.0 - candidate_edits[selection].sum() / denominator
        reference = 1.0 - reference_edits[selection].sum() / denominator
        values[index] = candidate - reference
    alpha = (1.0 - float(confidence)) / 2.0
    point = (
        1.0 - candidate_edits.sum() / counts.sum()
    ) - (1.0 - reference_edits.sum() / counts.sum())
    return {
        "point_estimate": float(point),
        "confidence": float(confidence),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
        "samples": int(samples),
        "seed": int(seed),
    }


__all__ = [
    "continuous_event_metrics_v2",
    "paired_bootstrap_levenshtein_delta",
    "sequence_edit_operations",
]
