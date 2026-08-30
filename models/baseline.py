"""Frozen causal TCN backbone embedded in the final model checkpoint."""

from __future__ import annotations

from typing import Mapping, Sequence

from torch import Tensor, nn

from models.causal_tcn import CausalTCN
from models.frame_head import FrameBoundaryHead


class ContinuousBaseline(nn.Module):
    """The final model's causal frame and boundary backbone.

    The ``architecture`` argument remains explicit to make accidental loading of
    an intermediate B0/B1A configuration fail immediately.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        architecture: str = "b1",
        hidden_dim: int = 128,
        num_classes: int = 14,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if str(architecture).lower() != "b1":
            raise ValueError("the final runtime only supports its frozen causal backbone")
        self.architecture = "b1"
        self.backbone = CausalTCN(
            input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.receptive_field = self.backbone.receptive_field
        self.head = FrameBoundaryHead(hidden_dim, num_classes, include_boundaries=True)

    def forward(self, features: Tensor) -> Mapping[str, Tensor]:
        return self.head(self.encode(features))

    def encode(self, features: Tensor) -> Tensor:
        return self.backbone(features)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = ["ContinuousBaseline"]
