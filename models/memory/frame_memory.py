"""Finite per-stream Frame Memory used by the final runtime."""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn


class FrameMemory(nn.Module):
    def __init__(self, hidden_dim: int, length: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.length = int(length)
        if self.hidden_dim <= 0 or self.length <= 0:
            raise ValueError("hidden_dim and length must be positive")
        self.age_embedding = nn.Embedding(self.length, self.hidden_dim)
        self._stream_states: Dict[Hashable, Tuple[Tensor, Tensor]] = {}

    @property
    def active_stream_ids(self) -> Tuple[Hashable, ...]:
        return tuple(self._stream_states)

    def reset_state(self, stream_ids: Optional[Iterable[Hashable]] = None) -> None:
        if stream_ids is None:
            self._stream_states.clear()
        else:
            for stream_id in tuple(stream_ids):
                self._stream_states.pop(stream_id, None)

    def _add_age(self, frames: Tensor) -> Tensor:
        ages = torch.arange(
            self.length - 1, -1, -1, device=frames.device, dtype=torch.long
        )
        return frames + self.age_embedding(ages).unsqueeze(0)

    def read_sequence(
        self,
        encoded: Tensor,
        valid_mask: Tensor,
        query_times: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Read ``[t-M,t)`` separately for every batch item."""

        if encoded.ndim != 3 or encoded.shape[-1] != self.hidden_dim:
            raise ValueError("encoded must be [B,T,D]")
        if valid_mask.shape != encoded.shape[:2]:
            raise ValueError("valid_mask must be [B,T]")
        batch, frames, _ = encoded.shape
        if query_times.shape != (batch,):
            raise ValueError("query_times must be [B]")
        offsets = torch.arange(self.length, device=encoded.device, dtype=torch.long)
        positions = query_times.long().unsqueeze(1) - self.length + offsets.unsqueeze(0)
        inside = (positions >= 0) & (positions < frames) & (query_times[:, None] > 0)
        safe = positions.clamp(min=0, max=max(0, frames - 1))
        gather_index = safe.unsqueeze(-1).expand(batch, self.length, self.hidden_dim)
        output = torch.gather(encoded, 1, gather_index)
        output_valid = torch.gather(valid_mask.bool(), 1, safe) & inside
        output = output * inside.unsqueeze(-1).to(output.dtype)
        return self._add_age(output), output_valid

    def read_sequence_many(
        self,
        encoded: Tensor,
        valid_mask: Tensor,
        query_times: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Vectorized ``[t-M,t)`` reads for schedules shaped ``[B,S]``."""

        if encoded.ndim != 3 or encoded.shape[-1] != self.hidden_dim:
            raise ValueError("encoded must be [B,T,D]")
        batch, frames, hidden = encoded.shape
        if valid_mask.shape != (batch, frames) or query_times.ndim != 2:
            raise ValueError("valid/query schedule shape mismatch")
        if query_times.shape[0] != batch:
            raise ValueError("query schedule batch mismatch")
        steps = int(query_times.shape[1])
        offsets = torch.arange(self.length, device=encoded.device, dtype=torch.long)
        positions = (
            query_times.long().unsqueeze(-1)
            - self.length
            + offsets.reshape(1, 1, self.length)
        )
        inside = (positions >= 0) & (positions < frames) & (query_times[..., None] > 0)
        safe = positions.clamp(min=0, max=max(0, frames - 1))
        expanded = encoded[:, None].expand(batch, steps, frames, hidden)
        output = torch.gather(
            expanded,
            2,
            safe.unsqueeze(-1).expand(batch, steps, self.length, hidden),
        )
        expanded_valid = valid_mask[:, None].expand(batch, steps, frames)
        output_valid = torch.gather(expanded_valid, 2, safe) & inside
        output = output * inside.unsqueeze(-1).to(output.dtype)
        ages = torch.arange(
            self.length - 1, -1, -1, device=encoded.device, dtype=torch.long
        )
        output = output + self.age_embedding(ages).reshape(
            1, 1, self.length, self.hidden_dim
        )
        return output, output_valid

    def append_stream(
        self, stream_id: Hashable, embedding: Tensor, *, valid: bool = True
    ) -> None:
        value = embedding.detach()
        if value.shape != (self.hidden_dim,):
            raise ValueError(f"embedding must be [{self.hidden_dim}]")
        state = self._stream_states.get(stream_id)
        if state is None:
            frames = value.new_zeros((0, self.hidden_dim))
            mask = torch.zeros(0, dtype=torch.bool, device=value.device)
        else:
            frames, mask = state
            if frames.device != value.device or frames.dtype != value.dtype:
                raise ValueError("stream Frame Memory device/dtype mismatch")
        frames = torch.cat((frames, value.unsqueeze(0)), dim=0)[-self.length :]
        mask = torch.cat(
            (mask, torch.tensor([bool(valid)], device=value.device, dtype=torch.bool))
        )[-self.length :]
        self._stream_states[stream_id] = (frames, mask)

    def read_stream(self, stream_id: Hashable) -> Tuple[Tensor, Tensor]:
        state = self._stream_states.get(stream_id)
        if state is None:
            device = self.age_embedding.weight.device
            dtype = self.age_embedding.weight.dtype
            frames = torch.zeros(0, self.hidden_dim, device=device, dtype=dtype)
            valid = torch.zeros(0, dtype=torch.bool, device=device)
        else:
            frames, valid = state
        padding = self.length - int(frames.shape[0])
        padded = torch.cat(
            (frames.new_zeros((padding, self.hidden_dim)), frames), dim=0
        ).unsqueeze(0)
        padded_valid = torch.cat(
            (torch.zeros(padding, dtype=torch.bool, device=valid.device), valid), dim=0
        ).unsqueeze(0)
        return self._add_age(padded), padded_valid


__all__ = ["FrameMemory"]
