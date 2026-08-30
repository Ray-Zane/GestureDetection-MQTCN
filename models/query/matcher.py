"""Hungarian matching for completed-event set predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class MatchCost:
    class_cost: float = 2.0
    boundary_l1_cost: float = 5.0
    tiou_cost: float = 2.0


def pairwise_tiou(predicted: Tensor, target: Tensor) -> Tensor:
    """Pairwise temporal IoU for normalized ordered ``[start,end]`` intervals."""

    intersection = (
        torch.minimum(predicted[:, None, 1], target[None, :, 1])
        - torch.maximum(predicted[:, None, 0], target[None, :, 0])
    ).clamp_min(0.0)
    union = (
        torch.maximum(predicted[:, None, 1], target[None, :, 1])
        - torch.minimum(predicted[:, None, 0], target[None, :, 0])
    ).clamp_min(1.0e-8)
    return intersection / union


class HungarianMatcher:
    def __init__(self, cost: MatchCost | None = None) -> None:
        self.cost = cost or MatchCost()

    @torch.no_grad()
    def __call__(
        self,
        class_logits: Tensor,
        boundaries: Tensor,
        target_classes: Tensor,
        target_boundaries: Tensor,
        left_censored: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if class_logits.ndim != 2 or boundaries.shape != (class_logits.shape[0], 2):
            raise ValueError("predictions must be class [N,C] and boundary [N,2]")
        targets = int(target_classes.numel())
        device = class_logits.device
        if targets == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty
        if target_boundaries.shape != (targets, 2) or left_censored.shape != (targets,):
            raise ValueError("target tensor shapes do not match")
        probabilities = class_logits.softmax(dim=-1)
        class_cost = -probabilities[:, target_classes.long()]
        absolute = torch.abs(boundaries[:, None, :] - target_boundaries[None, :, :])
        component_mask = torch.ones_like(absolute)
        component_mask[:, left_censored.bool(), 0] = 0.0
        denominator = component_mask.sum(dim=-1).clamp_min(1.0)
        boundary_cost = (absolute * component_mask).sum(dim=-1) / denominator
        # A censored start is unknown, so full-interval tIoU must not sneak the
        # artificial memory-left boundary back into the matching objective.
        overlap_cost = 1.0 - pairwise_tiou(boundaries, target_boundaries)
        overlap_cost[:, left_censored.bool()] = 0.0
        combined = (
            self.cost.class_cost * class_cost
            + self.cost.boundary_l1_cost * boundary_cost
            + self.cost.tiou_cost * overlap_cost
        )
        if targets == 1:
            prediction = torch.argmin(combined[:, 0]).reshape(1)
            return prediction, torch.zeros(1, dtype=torch.long, device=device)
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(combined.detach().cpu().numpy())
        return (
            torch.as_tensor(rows, dtype=torch.long, device=device),
            torch.as_tensor(columns, dtype=torch.long, device=device),
        )


__all__ = ["HungarianMatcher", "MatchCost", "pairwise_tiou"]
