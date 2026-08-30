"""Completed-event matching loss and sequential final-model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from datasets.memory_query import EncodedContinuousCache, collate_query_video_batch
from models.gesture_detection_mqtcn import GestureDetectionMQTCN
from models.query.matcher import HungarianMatcher, MatchCost


@dataclass(frozen=True)
class MemoryQueryLossConfig:
    query_class_weight: float = 2.0
    boundary_l1_weight: float = 5.0
    tiou_weight: float = 2.0
    eos_coef: float = 0.1


def aligned_tiou(predicted: Tensor, target: Tensor) -> Tensor:
    intersection = (
        torch.minimum(predicted[..., 1], target[..., 1])
        - torch.maximum(predicted[..., 0], target[..., 0])
    ).clamp_min(0.0)
    union = (
        torch.maximum(predicted[..., 1], target[..., 1])
        - torch.minimum(predicted[..., 0], target[..., 0])
    ).clamp_min(1.0e-8)
    return intersection / union


class MemoryQueryLoss(nn.Module):
    """DETR-style set loss with positive/negative classes normalized separately."""

    def __init__(
        self,
        *,
        match_cost: Optional[MatchCost] = None,
        config: Optional[MemoryQueryLossConfig] = None,
    ) -> None:
        super().__init__()
        self.matcher = HungarianMatcher(match_cost)
        self.config = config or MemoryQueryLossConfig()

    def _forward_single_target(
        self, logits: Tensor, intervals: Tensor, targets: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        """Vectorized official-data path (audited maximum target count is one)."""

        step_valid = targets["query_step_valid"].bool()
        active_logits = logits[step_valid]
        active_intervals = intervals[step_valid]
        active_target_valid = targets["target_valid"][step_valid].bool()
        active_target_classes = targets["target_classes"][step_valid]
        active_target_boundaries = targets["target_boundaries"][step_valid]
        active_censored = targets["left_censored"][step_valid].bool()
        steps, queries, classes = active_logits.shape
        zero = logits.sum() * 0.0
        if steps == 0:
            count_zero = torch.zeros((), dtype=torch.long, device=logits.device)
            return {
                "loss_total": zero,
                "loss_query_cls": zero,
                "loss_query_cls_matched": zero,
                "loss_query_cls_unmatched": zero,
                "loss_boundary_l1": zero,
                "loss_tiou": zero,
                "matched_queries": count_zero,
                "unmatched_queries": count_zero,
                "uncensored_matches": count_zero,
                "matched_class_correct": count_zero,
                "unmatched_class_correct": count_zero,
                "target_steps": count_zero,
                "empty_steps": count_zero,
            }
        positive_steps = active_target_valid.any(dim=-1)
        positive_indices = torch.nonzero(positive_steps, as_tuple=False).squeeze(1)
        number_positive = positive_indices.numel()
        negative_mask = torch.ones(
            (steps, queries), dtype=torch.bool, device=logits.device
        )
        boundary_loss = zero
        tiou_loss = zero
        uncensored_count = torch.zeros((), dtype=torch.long, device=logits.device)
        if number_positive:
            target_slots = active_target_valid[positive_steps].float().argmax(dim=-1)
            row = torch.arange(number_positive, device=logits.device)
            target_classes = active_target_classes[positive_steps][row, target_slots]
            target_boundaries = active_target_boundaries[positive_steps][row, target_slots]
            censored = active_censored[positive_steps][row, target_slots]
            prediction_logits = active_logits[positive_steps]
            prediction_intervals = active_intervals[positive_steps]
            probabilities = prediction_logits.softmax(dim=-1)
            class_cost = -torch.gather(
                probabilities,
                2,
                target_classes[:, None, None].expand(number_positive, queries, 1),
            ).squeeze(-1)
            absolute = torch.abs(
                prediction_intervals - target_boundaries[:, None, :]
            )
            coordinate_mask = torch.ones_like(absolute)
            coordinate_mask[censored, :, 0] = 0.0
            boundary_cost = (absolute * coordinate_mask).sum(dim=-1) / coordinate_mask.sum(
                dim=-1
            ).clamp_min(1.0)
            target_expanded = target_boundaries[:, None, :].expand_as(
                prediction_intervals
            )
            overlap_cost = 1.0 - aligned_tiou(
                prediction_intervals, target_expanded
            )
            overlap_cost[censored] = 0.0
            cost = (
                self.matcher.cost.class_cost * class_cost
                + self.matcher.cost.boundary_l1_cost * boundary_cost
                + self.matcher.cost.tiou_cost * overlap_cost
            )
            best = cost.argmin(dim=1)
            negative_mask[positive_indices, best] = False
            positive_logits = prediction_logits[row, best]
            positive_class_loss = functional.cross_entropy(
                positive_logits, target_classes, reduction="mean"
            )
            positive_correct = (
                positive_logits.argmax(dim=-1) == target_classes
            ).sum()
            selected_intervals = prediction_intervals[row, best]
            selected_mask = torch.ones_like(selected_intervals)
            selected_mask[censored, 0] = 0.0
            boundary_loss = (
                torch.abs(selected_intervals - target_boundaries) * selected_mask
            ).sum() / max(1, number_positive)
            visible = ~censored
            uncensored_count = visible.sum()
            if torch.any(visible):
                tiou_loss = (
                    1.0
                    - aligned_tiou(
                        selected_intervals[visible], target_boundaries[visible]
                    )
                ).mean()
        else:
            positive_class_loss = zero
            positive_correct = torch.zeros((), dtype=torch.long, device=logits.device)
        negative_logits = active_logits[negative_mask]
        if negative_logits.numel():
            negative_class_loss = functional.cross_entropy(
                negative_logits,
                torch.zeros(
                    negative_logits.shape[0], dtype=torch.long, device=logits.device
                ),
                reduction="mean",
            )
            negative_correct = (negative_logits.argmax(dim=-1) == 0).sum()
        else:
            negative_class_loss = zero
            negative_correct = torch.zeros((), dtype=torch.long, device=logits.device)
        query_class_loss = (
            positive_class_loss + self.config.eos_coef * negative_class_loss
        )
        total = (
            self.config.query_class_weight * query_class_loss
            + self.config.boundary_l1_weight * boundary_loss
            + self.config.tiou_weight * tiou_loss
        )
        matched = torch.as_tensor(number_positive, device=logits.device)
        unmatched = torch.as_tensor(
            int(negative_logits.shape[0]), device=logits.device
        )
        return {
            "loss_total": total,
            "loss_query_cls": query_class_loss,
            "loss_query_cls_matched": positive_class_loss,
            "loss_query_cls_unmatched": negative_class_loss,
            "loss_boundary_l1": boundary_loss,
            "loss_tiou": tiou_loss,
            "matched_queries": matched,
            "unmatched_queries": unmatched,
            "uncensored_matches": uncensored_count,
            "matched_class_correct": positive_correct,
            "unmatched_class_correct": negative_correct,
            "target_steps": matched,
            "empty_steps": torch.as_tensor(
                steps - number_positive, device=logits.device
            ),
        }

    def forward(
        self, outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        logits = outputs["pred_logits"]
        intervals = outputs["pred_intervals"]
        if logits.ndim != 4 or intervals.shape != (*logits.shape[:3], 2):
            raise ValueError("Query predictions must be [B,S,Q,C] and [B,S,Q,2]")
        batch, steps, queries, _ = logits.shape
        step_valid = targets["query_step_valid"].bool()
        if step_valid.shape != (batch, steps):
            raise ValueError("query_step_valid shape mismatch")
        if int(targets["target_valid"].sum(dim=-1).max().item()) <= 1:
            return self._forward_single_target(logits, intervals, targets)
        positive_logits = []
        positive_labels = []
        negative_logits = []
        boundary_sum = logits.sum() * 0.0
        tiou_sum = logits.sum() * 0.0
        matched_count = 0
        uncensored_count = 0
        target_steps = 0
        empty_steps = 0
        for item in range(batch):
            for step in range(steps):
                if not bool(step_valid[item, step].item()):
                    continue
                mask = targets["target_valid"][item, step].bool()
                target_classes = targets["target_classes"][item, step][mask]
                target_boundaries = targets["target_boundaries"][item, step][mask]
                censored = targets["left_censored"][item, step][mask].bool()
                if int(mask.sum().item()) > 0:
                    target_steps += 1
                else:
                    empty_steps += 1
                prediction_indices, target_indices = self.matcher(
                    logits[item, step],
                    intervals[item, step],
                    target_classes,
                    target_boundaries,
                    censored,
                )
                assigned = torch.zeros(queries, dtype=torch.bool, device=logits.device)
                if prediction_indices.numel():
                    assigned[prediction_indices] = True
                    positive_logits.append(logits[item, step, prediction_indices])
                    positive_labels.append(target_classes[target_indices])
                    prediction_intervals = intervals[item, step, prediction_indices]
                    matched_targets = target_boundaries[target_indices]
                    matched_censored = censored[target_indices]
                    coordinate_mask = torch.ones_like(prediction_intervals)
                    coordinate_mask[matched_censored, 0] = 0.0
                    boundary_sum = boundary_sum + (
                        torch.abs(prediction_intervals - matched_targets)
                        * coordinate_mask
                    ).sum()
                    visible = ~matched_censored
                    if torch.any(visible):
                        tiou_sum = tiou_sum + (
                            1.0
                            - aligned_tiou(
                                prediction_intervals[visible], matched_targets[visible]
                            )
                        ).sum()
                        uncensored_count += int(visible.sum().item())
                    matched_count += int(prediction_indices.numel())
                negative_logits.append(logits[item, step, ~assigned])
        zero = logits.sum() * 0.0
        if positive_logits:
            joined_positive = torch.cat(positive_logits, dim=0)
            joined_labels = torch.cat(positive_labels, dim=0)
            positive_class_loss = functional.cross_entropy(
                joined_positive, joined_labels, reduction="mean"
            )
            positive_correct = (
                joined_positive.argmax(dim=-1) == joined_labels
            ).sum()
        else:
            positive_class_loss = zero
            positive_correct = torch.zeros((), device=logits.device, dtype=torch.long)
        if negative_logits:
            joined_negative = torch.cat(negative_logits, dim=0)
            negative_class_loss = functional.cross_entropy(
                joined_negative,
                torch.zeros(
                    joined_negative.shape[0], dtype=torch.long, device=logits.device
                ),
                reduction="mean",
            )
            negative_correct = (joined_negative.argmax(dim=-1) == 0).sum()
            unmatched_count = int(joined_negative.shape[0])
        else:
            negative_class_loss = zero
            negative_correct = torch.zeros((), device=logits.device, dtype=torch.long)
            unmatched_count = 0
        query_class_loss = (
            positive_class_loss + self.config.eos_coef * negative_class_loss
        )
        boundary_loss = boundary_sum / max(1, matched_count)
        tiou_loss = tiou_sum / max(1, uncensored_count)
        total = (
            self.config.query_class_weight * query_class_loss
            + self.config.boundary_l1_weight * boundary_loss
            + self.config.tiou_weight * tiou_loss
        )
        losses = {
            "loss_total": total,
            "loss_query_cls": query_class_loss,
            "loss_query_cls_matched": positive_class_loss,
            "loss_query_cls_unmatched": negative_class_loss,
            "loss_boundary_l1": boundary_loss,
            "loss_tiou": tiou_loss,
            "matched_queries": torch.tensor(matched_count, device=logits.device),
            "unmatched_queries": torch.tensor(unmatched_count, device=logits.device),
            "uncensored_matches": torch.tensor(
                uncensored_count, device=logits.device
            ),
            "matched_class_correct": positive_correct,
            "unmatched_class_correct": negative_correct,
            "target_steps": torch.tensor(target_steps, device=logits.device),
            "empty_steps": torch.tensor(empty_steps, device=logits.device),
        }
        return losses


def _slice_targets(batch: Mapping[str, Any], start: int, end: int) -> Mapping[str, Tensor]:
    keys = (
        "query_step_valid",
        "target_classes",
        "target_boundaries",
        "target_valid",
        "left_censored",
        "boundary_mask",
        "tiou_valid",
        "event_ids",
        "global_intervals",
    )
    return {key: batch[key][:, start:end] for key in keys}


def _float_metrics(losses: Mapping[str, Tensor]) -> Mapping[str, float]:
    return {key: float(value.detach().item()) for key, value in losses.items()}


def train_memory_query_epoch(
    model: GestureDetectionMQTCN,
    cache: EncodedContinuousCache,
    optimizer: torch.optim.Optimizer,
    criterion: MemoryQueryLoss,
    device: torch.device,
    *,
    epoch: int,
    seed: int,
    video_batch_size: int,
    query_chunk_steps: int,
    query_stride: int,
    gradient_clip: float,
    shuffle_video_streams: bool,
    limit_videos: Optional[int] = None,
) -> Mapping[str, float]:
    model.train()
    number = len(cache) if limit_videos is None else min(len(cache), int(limit_videos))
    indices = np.arange(number, dtype=np.int64)
    if shuffle_video_streams:
        np.random.default_rng(int(seed) + 7919 * int(epoch)).shuffle(indices)
    totals: Dict[str, float] = {}
    count_keys = {
        "matched_queries",
        "unmatched_queries",
        "uncensored_matches",
        "matched_class_correct",
        "unmatched_class_correct",
        "target_steps",
        "empty_steps",
    }
    total_weight = 0
    grad_norm_total = 0.0
    optimizer_steps = 0
    size = max(1, int(video_batch_size))
    chunk = max(1, int(query_chunk_steps))
    for group_start in range(0, len(indices), size):
        group = indices[group_start : group_start + size].tolist()
        batch = collate_query_video_batch(
            cache,
            group,
            query_stride=query_stride,
            memory_length=model.frame_memory_length,
            num_queries=model.num_queries,
            device=device,
        )
        steps = int(batch["query_times"].shape[1])
        for step_start in range(0, steps, chunk):
            step_end = min(steps, step_start + chunk)
            output = model.query_sequence(
                batch["encoded"],
                batch["memory_valid"],
                batch["query_times"][:, step_start:step_end],
                active_mask=batch["query_step_valid"][:, step_start:step_end],
            )
            predictions = {
                key: output[key] for key in ("pred_logits", "pred_intervals")
            }
            targets = _slice_targets(batch, step_start, step_end)
            losses = criterion(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            trainable = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(gradient_clip)
            )
            optimizer.step()
            weight = int(targets["query_step_valid"].sum().item())
            values = _float_metrics(losses)
            for key, value in values.items():
                multiplier = 1.0 if key in count_keys else float(max(1, weight))
                totals[key] = totals.get(key, 0.0) + value * multiplier
            total_weight += max(1, weight)
            grad_norm_total += float(grad_norm.item())
            optimizer_steps += 1
    if total_weight <= 0:
        raise RuntimeError("no valid Query steps in training epoch")
    result = {
        key: (value if key in count_keys else value / total_weight)
        for key, value in totals.items()
    }
    result["matched_class_accuracy"] = result.get("matched_class_correct", 0.0) / max(
        1.0, result.get("matched_queries", 0.0)
    )
    result["unmatched_class_accuracy"] = result.get(
        "unmatched_class_correct", 0.0
    ) / max(1.0, result.get("unmatched_queries", 0.0))
    result["grad_norm"] = grad_norm_total / max(1, optimizer_steps)
    result["optimizer_steps"] = float(optimizer_steps)
    result["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    return result


__all__ = [
    "MemoryQueryLoss",
    "MemoryQueryLossConfig",
    "aligned_tiou",
    "train_memory_query_epoch",
]
