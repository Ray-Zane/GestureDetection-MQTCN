"""No-event plus IPN Hand gesture classification for Event Queries."""

from torch import Tensor, nn


class QueryClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int = 14) -> None:
        super().__init__()
        self.projection = nn.Linear(int(hidden_dim), int(num_classes))

    def forward(self, query_states: Tensor) -> Tensor:
        return self.projection(query_states)


__all__ = ["QueryClassificationHead"]
