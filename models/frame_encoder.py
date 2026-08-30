"""Per-frame projection shared by B0 and B1."""

from __future__ import annotations

from torch import Tensor, nn


class FrameEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected [B,T,{self.input_dim}], got {tuple(features.shape)}"
            )
        return self.network(features)


__all__ = ["FrameEncoder"]
