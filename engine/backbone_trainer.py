"""Deterministic training primitives for the causal frame backbone."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def frame_class_weights(counts: np.ndarray) -> np.ndarray:
    """Official-Train-only inverse-sqrt weights with bounded rare-class influence."""

    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (14,) or np.any(values <= 0):
        raise ValueError("all 14 frame classes require positive Train counts")
    weights = np.sqrt(np.median(values) / values)
    weights = np.clip(weights, 0.25, 4.0)
    weights /= weights.mean()
    return weights.astype(np.float32)


def boundary_positive_weight(total_frames: int, positive_frames: int) -> float:
    positive = int(positive_frames)
    total = int(total_frames)
    if not 0 < positive < total:
        raise ValueError("boundary positives must be between zero and total")
    return float(np.clip((total - positive) / positive, 1.0, 30.0))


@dataclass(frozen=True)
class BaselineLossConfig:
    frame_weight: float = 1.0
    start_weight: float = 0.5
    end_weight: float = 0.5


class BaselineLoss(nn.Module):
    def __init__(
        self,
        frame_weights: Tensor,
        *,
        start_positive_weight: float,
        end_positive_weight: float,
        config: Optional[BaselineLossConfig] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("frame_weights", frame_weights.detach().float())
        self.register_buffer(
            "start_positive_weight", torch.tensor(float(start_positive_weight))
        )
        self.register_buffer(
            "end_positive_weight", torch.tensor(float(end_positive_weight))
        )
        self.config = config or BaselineLossConfig()

    def forward(
        self, outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        labels = batch["frame_labels"]
        mask = batch["supervision_mask"].bool() & (labels >= 0)
        if not torch.any(mask):
            raise ValueError("batch has no supervised frames")
        frame_loss = functional.cross_entropy(
            outputs["frame_logits"].transpose(1, 2),
            labels,
            weight=self.frame_weights,
            ignore_index=-100,
            reduction="none",
        )[mask].mean()
        start_loss = functional.binary_cross_entropy_with_logits(
            outputs["start_logits"],
            batch["start_targets"],
            pos_weight=self.start_positive_weight,
            reduction="none",
        )[mask].mean()
        end_loss = functional.binary_cross_entropy_with_logits(
            outputs["end_logits"],
            batch["end_targets"],
            pos_weight=self.end_positive_weight,
            reduction="none",
        )[mask].mean()
        total = (
            self.config.frame_weight * frame_loss
            + self.config.start_weight * start_loss
            + self.config.end_weight * end_loss
        )
        return {
            "loss_total": total,
            "loss_frame_cls": frame_loss,
            "loss_start": start_loss,
            "loss_end": end_loss,
            "supervised_frames": mask.sum(),
            "correct_frames": (
                outputs["frame_logits"].argmax(dim=-1)[mask] == labels[mask]
            ).sum(),
        }


def _to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: BaselineLoss,
    device: torch.device,
    *,
    amp: bool,
    gradient_clip: float,
) -> Mapping[str, float]:
    model.train()
    totals = {
        "loss_total": 0.0,
        "loss_frame_cls": 0.0,
        "loss_start": 0.0,
        "loss_end": 0.0,
    }
    supervised_total = 0
    correct_total = 0
    grad_norms = []
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model(batch["features"])
            losses = criterion(outputs, batch)
        if not torch.isfinite(losses["loss_total"]):
            raise FloatingPointError("non-finite training loss")
        scaler.scale(losses["loss_total"]).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        count = int(losses["supervised_frames"].detach().item())
        supervised_total += count
        correct_total += int(losses["correct_frames"].detach().item())
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * count
        grad_norms.append(float(gradient_norm.detach().item()))
    if supervised_total <= 0:
        raise RuntimeError("training loader produced no supervised frames")
    metrics = {key: value / supervised_total for key, value in totals.items()}
    metrics.update(
        {
            "frame_accuracy": correct_total / supervised_total,
            "supervised_frames": supervised_total,
            "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_max": float(np.max(grad_norms)),
            "batches": len(grad_norms),
        }
    )
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError("non-finite aggregated training metric")
    return metrics


__all__ = [
    "BaselineLoss",
    "BaselineLossConfig",
    "boundary_positive_weight",
    "frame_class_weights",
    "seed_everything",
    "train_one_epoch",
]
