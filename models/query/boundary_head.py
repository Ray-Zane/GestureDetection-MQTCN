"""Ordered normalized start/end prediction for Event Queries."""

import torch
from torch import Tensor, nn


class QueryBoundaryHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(int(hidden_dim), 2)

    def forward(self, query_states: Tensor) -> Tensor:
        raw = self.projection(query_states)
        # End is an independent branch so a censored start can be masked without
        # indirectly changing end supervision through the parameterization.
        end = torch.sigmoid(raw[..., 1])
        start = end * torch.sigmoid(raw[..., 0])
        return torch.stack((start, end), dim=-1)


__all__ = ["QueryBoundaryHead"]
