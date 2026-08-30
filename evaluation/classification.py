"""Dependency-light multiclass classification metrics."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np


def classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    *,
    probabilities: Optional[np.ndarray] = None,
) -> Mapping[str, Any]:
    target = np.asarray(targets, dtype=np.int64)
    prediction = np.asarray(predictions, dtype=np.int64)
    classes = len(class_names)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("targets and predictions must be matching 1D arrays")
    if target.size and (
        int(target.min()) < 0
        or int(target.max()) >= classes
        or int(prediction.min()) < 0
        or int(prediction.max()) >= classes
    ):
        raise ValueError("classification labels are outside class range")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    if target.size:
        np.add.at(confusion, (target, prediction), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(classes, dtype=np.float64),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(classes, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(classes, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    per_class = [
        {
            "class_id": index,
            "class_name": str(class_names[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(classes)
    ]
    metrics = {
        "samples": int(target.size),
        "accuracy": float(np.mean(target == prediction)) if target.size else 0.0,
        "balanced_accuracy": float(np.mean(recall)) if classes else 0.0,
        "macro_f1": float(np.mean(f1)) if classes else 0.0,
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    if probabilities is not None:
        probability = np.asarray(probabilities, dtype=np.float64)
        if probability.shape != (target.size, classes):
            raise ValueError(
                f"probability shape {probability.shape} != {(target.size, classes)}"
            )
        top_k = min(5, classes)
        top_indices = np.argpartition(probability, -top_k, axis=1)[:, -top_k:]
        metrics["top5_accuracy"] = (
            float(np.mean(np.any(top_indices == target[:, None], axis=1)))
            if target.size
            else 0.0
        )
    return metrics


__all__ = ["classification_metrics"]
