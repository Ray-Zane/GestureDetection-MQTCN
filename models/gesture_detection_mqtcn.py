"""Final GestureDetection-MQTCN model used by the deployment demo."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Optional

from models.baseline import ContinuousBaseline
from models.memory.frame_memory import FrameMemory
from models.query.query_decoder import EventQueryDecoder


class GestureDetectionMQTCN(nn.Module):
    """Causal TCN, finite Frame Memory and completed-event Query decoder."""

    architecture = "gesture_detection_mqtcn"

    def __init__(
        self,
        baseline: ContinuousBaseline,
        *,
        num_queries: int = 6,
        num_query_classes: int = 14,
        attention_heads: int = 4,
        decoder_layers: int = 1,
        feedforward_dim: int = 256,
        frame_memory_length: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if baseline.architecture != "b1":
            raise ValueError("GestureDetectionMQTCN requires its frozen causal backbone")
        self.baseline = baseline
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.baseline.eval()
        self.hidden_dim = int(self.baseline.backbone.hidden_dim)
        self.num_queries = int(num_queries)
        self.frame_memory_length = int(frame_memory_length)
        self.frame_memory = FrameMemory(self.hidden_dim, self.frame_memory_length)
        self.query_decoder = EventQueryDecoder(
            self.hidden_dim,
            num_queries=self.num_queries,
            num_classes=int(num_query_classes),
            attention_heads=int(attention_heads),
            decoder_layers=int(decoder_layers),
            feedforward_dim=int(feedforward_dim),
            dropout=float(dropout),
        )

    def train(self, mode: bool = True) -> "GestureDetectionMQTCN":
        super().train(mode)
        self.baseline.eval()
        return self

    @property
    def receptive_field(self) -> int:
        return int(self.baseline.receptive_field)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )

    def query_sequence(
        self,
        encoded: Tensor,
        memory_valid: Tensor,
        query_times: Tensor,
        *,
        active_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Evaluate every completed-event Query tick for offline training."""

        if encoded.ndim != 3 or encoded.shape[-1] != self.hidden_dim:
            raise ValueError("encoded must be [B,T,D]")
        if query_times.ndim != 2 or query_times.shape[0] != encoded.shape[0]:
            raise ValueError("query_times must be [B,S]")
        if memory_valid.shape != encoded.shape[:2]:
            raise ValueError("memory_valid must be [B,T]")
        batch, steps = query_times.shape
        frames = int(encoded.shape[1])
        active = (
            torch.ones_like(query_times, dtype=torch.bool)
            if active_mask is None
            else active_mask.bool()
        )
        safe = query_times.clamp(min=1, max=frames)
        current = torch.gather(
            encoded,
            1,
            (safe - 1).unsqueeze(-1).expand(batch, steps, self.hidden_dim),
        )
        frame_tokens, frame_valid = self.frame_memory.read_sequence_many(
            encoded, memory_valid, safe
        )
        frame_valid = frame_valid & active.unsqueeze(-1)
        flattened = batch * steps
        logits, intervals, states = self.query_decoder(
            current.reshape(flattened, self.hidden_dim),
            frame_tokens.reshape(flattened, frame_tokens.shape[2], self.hidden_dim),
            frame_valid.reshape(flattened, frame_valid.shape[2]),
        )
        return {
            "pred_logits": logits.reshape(batch, steps, self.num_queries, -1),
            "pred_intervals": intervals.reshape(batch, steps, self.num_queries, 2),
            "query_states": states.reshape(
                batch, steps, self.num_queries, self.hidden_dim
            ),
            "query_times": query_times,
            "query_step_valid": active,
        }


__all__ = ["GestureDetectionMQTCN"]
