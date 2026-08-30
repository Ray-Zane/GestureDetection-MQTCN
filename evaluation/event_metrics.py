"""One-to-one class-aware temporal event matching and F1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple


def event_tiou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    start = max(int(first["start_frame"]), int(second["start_frame"]))
    end = min(
        int(first["end_frame_exclusive"]), int(second["end_frame_exclusive"])
    )
    intersection = max(0, end - start)
    union = max(
        int(first["end_frame_exclusive"]), int(second["end_frame_exclusive"])
    ) - min(int(first["start_frame"]), int(second["start_frame"]))
    return float(intersection / union) if union > 0 else 0.0


@dataclass(frozen=True)
class EventMatch:
    prediction_index: int
    target_index: int
    tiou: float


def match_events(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> Tuple[List[EventMatch], List[int], List[int]]:
    """Greedy maximum-tIoU one-to-one matching restricted to identical classes."""

    edges = []
    for prediction_index, prediction in enumerate(predictions):
        for target_index, target in enumerate(targets):
            if int(prediction["class_id"]) != int(target["class_id"]):
                continue
            overlap = event_tiou(prediction, target)
            if overlap >= float(threshold):
                edges.append((overlap, prediction_index, target_index))
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_predictions = set()
    used_targets = set()
    matches: List[EventMatch] = []
    for overlap, prediction_index, target_index in edges:
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
        matches.append(EventMatch(prediction_index, target_index, float(overlap)))
    unmatched_predictions = [
        index for index in range(len(predictions)) if index not in used_predictions
    ]
    unmatched_targets = [index for index in range(len(targets)) if index not in used_targets]
    return matches, unmatched_predictions, unmatched_targets


def event_f1_counts(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> Mapping[str, int]:
    matches, false_positive, false_negative = match_events(
        predictions, targets, threshold=threshold
    )
    return {
        "true_positive": len(matches),
        "false_positive": len(false_positive),
        "false_negative": len(false_negative),
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> Mapping[str, float]:
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


__all__ = [
    "EventMatch",
    "event_f1_counts",
    "event_tiou",
    "match_events",
    "precision_recall_f1",
]
