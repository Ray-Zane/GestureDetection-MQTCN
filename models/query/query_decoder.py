"""Completed-event Query decoder specialized for GestureDetection-MQTCN."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from models.query.boundary_head import QueryBoundaryHead
from models.query.classification_head import QueryClassificationHead


class EventQueryDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_queries: int,
        num_classes: int,
        attention_heads: int,
        decoder_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.learnable_queries = nn.Parameter(
            torch.empty(self.num_queries, self.hidden_dim)
        )
        nn.init.normal_(self.learnable_queries, std=0.02)
        self.current_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(attention_heads),
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(decoder_layers))
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.frame_type = nn.Parameter(torch.zeros(self.hidden_dim))
        # Retained because it is part of the frozen final checkpoint state dict.
        # The production model never reads Semantic Memory.
        self.semantic_type = nn.Parameter(torch.zeros(self.hidden_dim))
        self.null_memory = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.classification = QueryClassificationHead(self.hidden_dim, num_classes)
        self.boundary = QueryBoundaryHead(self.hidden_dim)

    def forward(
        self,
        current_embedding: Tensor,
        frame_memory: Tensor,
        frame_valid: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if current_embedding.ndim != 2 or current_embedding.shape[-1] != self.hidden_dim:
            raise ValueError("current_embedding must be [B,D]")
        if frame_memory.ndim != 3 or frame_memory.shape[0] != current_embedding.shape[0]:
            raise ValueError("frame_memory must be [B,M,D]")
        if frame_valid.shape != frame_memory.shape[:2]:
            raise ValueError("frame_valid must be [B,M]")
        memory = frame_memory + self.frame_type
        valid = frame_valid.bool()
        all_invalid = ~valid.any(dim=1)
        if torch.any(all_invalid):
            memory = memory.clone()
            valid = valid.clone()
            memory[all_invalid, 0:1] = self.null_memory
            valid[all_invalid, 0] = True
        initial = self.learnable_queries.unsqueeze(0) + self.current_projection(
            current_embedding
        ).unsqueeze(1)
        states = self.decoder(initial, memory, memory_key_padding_mask=~valid)
        states = self.output_norm(states)
        return self.classification(states), self.boundary(states), states


__all__ = ["EventQueryDecoder"]
