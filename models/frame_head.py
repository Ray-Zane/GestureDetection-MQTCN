"""Frame classification and optional start/end boundary heads."""

from __future__ import annotations

from typing import Dict

from torch import Tensor, nn


class FrameBoundaryHead(nn.Module):
    def __init__(
        self, hidden_dim: int, num_classes: int = 14, *, include_boundaries: bool = True
    ) -> None:
        super().__init__()
        self.include_boundaries = bool(include_boundaries)
        self.classifier = nn.Linear(int(hidden_dim), int(num_classes))
        self.start = nn.Linear(int(hidden_dim), 1) if self.include_boundaries else None
        self.end = nn.Linear(int(hidden_dim), 1) if self.include_boundaries else None

    def forward(self, encoded: Tensor) -> Dict[str, Tensor]:
        if not self.include_boundaries:
            zeros = encoded.new_zeros(encoded.shape[:-1])
            return {
                "frame_logits": self.classifier(encoded),
                "start_logits": zeros,
                "end_logits": zeros,
            }
        assert self.start is not None and self.end is not None
        return {
            "frame_logits": self.classifier(encoded),
            "start_logits": self.start(encoded).squeeze(-1),
            "end_logits": self.end(encoded).squeeze(-1),
        }


__all__ = ["FrameBoundaryHead"]
