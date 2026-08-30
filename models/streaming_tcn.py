"""Stateful, weight-equivalent step execution for the frozen B1 causal TCN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from models.baseline import ContinuousBaseline


@dataclass(frozen=True)
class StreamingTCNState:
    """Per-stream runtime state; each buffer stores one TCN block's input."""

    buffers: Tuple[Tensor, ...]
    processed_frames: int

    def clone(self) -> "StreamingTCNState":
        return StreamingTCNState(
            tuple(buffer.detach().clone() for buffer in self.buffers),
            int(self.processed_frames),
        )


class StatefulContinuousBaseline(nn.Module):
    """B2 API around B1 modules; runtime state is deliberately not a parameter."""

    def __init__(self, offline_model: ContinuousBaseline) -> None:
        super().__init__()
        if offline_model.architecture != "b1":
            raise ValueError("Stateful execution requires a B1 causal TCN")
        self.offline_model = offline_model
        self._states: Dict[Hashable, StreamingTCNState] = {}

    @property
    def receptive_field(self) -> int:
        return int(self.offline_model.receptive_field)

    @property
    def input_dim(self) -> int:
        return int(self.offline_model.backbone.input_dim)

    @property
    def parameter_count(self) -> int:
        return int(self.offline_model.parameter_count)

    @property
    def active_stream_ids(self) -> Tuple[Hashable, ...]:
        return tuple(self._states)

    def forward(self, features: Tensor) -> Mapping[str, Tensor]:
        return self.offline_model(features)

    def _empty_state(self, device: torch.device, dtype: torch.dtype) -> StreamingTCNState:
        blocks = self.offline_model.backbone.blocks
        buffers = tuple(
            torch.zeros(
                block.left_padding,
                self.offline_model.backbone.hidden_dim,
                device=device,
                dtype=dtype,
            )
            for block in blocks
        )
        return StreamingTCNState(buffers, 0)

    def reset_state(self, stream_ids: Optional[Iterable[Hashable]] = None) -> None:
        """Clear every stream, or atomically clear only the named streams."""

        if stream_ids is None:
            self._states.clear()
            return
        for stream_id in tuple(stream_ids):
            self._states.pop(stream_id, None)

    def get_stream_state(
        self, stream_id: Hashable, *, clone: bool = True
    ) -> Optional[StreamingTCNState]:
        state = self._states.get(stream_id)
        if state is None:
            return None
        return state.clone() if clone else state

    def set_stream_state(self, stream_id: Hashable, state: StreamingTCNState) -> None:
        blocks = self.offline_model.backbone.blocks
        if len(state.buffers) != len(blocks):
            raise ValueError("state buffer count does not match TCN blocks")
        hidden = self.offline_model.backbone.hidden_dim
        for block, buffer in zip(blocks, state.buffers):
            if buffer.shape != (block.left_padding, hidden):
                raise ValueError(
                    f"state buffer {tuple(buffer.shape)} does not match "
                    f"{(block.left_padding, hidden)}"
                )
        self._states[stream_id] = state.clone()

    def _advance_encoded(
        self,
        frame_features: Tensor,
        *,
        stream_ids: Optional[Sequence[Hashable]] = None,
    ) -> Tuple[Tensor, bool]:

        squeeze = frame_features.ndim == 1
        if squeeze:
            frame_features = frame_features.unsqueeze(0)
        if frame_features.ndim != 2 or frame_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected [D] or [B,{self.input_dim}], got {tuple(frame_features.shape)}"
            )
        batch = int(frame_features.shape[0])
        ids: Tuple[Hashable, ...] = (
            tuple(range(batch)) if stream_ids is None else tuple(stream_ids)
        )
        if len(ids) != batch or len(set(ids)) != batch:
            raise ValueError("stream_ids must be unique and match batch size")
        states = []
        for stream_id in ids:
            state = self._states.get(stream_id)
            if state is None:
                state = self._empty_state(frame_features.device, frame_features.dtype)
            elif any(
                buffer.device != frame_features.device or buffer.dtype != frame_features.dtype
                for buffer in state.buffers
            ):
                raise ValueError("runtime state device/dtype mismatch; reset after model.to()")
            states.append(state)

        encoded = self.offline_model.backbone.frame_encoder(
            frame_features.unsqueeze(1)
        )
        updated_by_stream = [[] for _ in range(batch)]
        for layer_index, block in enumerate(self.offline_model.backbone.blocks):
            layer_input = encoded
            buffers = torch.stack(
                [state.buffers[layer_index] for state in states], dim=0
            )
            window = torch.cat((buffers, layer_input), dim=1)
            depthwise = block.depthwise(window.transpose(1, 2))
            output = block.pointwise(depthwise).transpose(1, 2)
            output = block.norm(output)
            output = block.activation(output)
            encoded = layer_input + block.dropout(output)
            next_buffer = window[:, -block.left_padding :, :]
            for item in range(batch):
                updated_by_stream[item].append(next_buffer[item].detach())

        for item, stream_id in enumerate(ids):
            self._states[stream_id] = StreamingTCNState(
                tuple(updated_by_stream[item]),
                states[item].processed_frames + 1,
            )
        squeezed = encoded[:, 0]
        return (squeezed[0] if squeeze else squeezed), squeeze

    def step(
        self,
        frame_features: Tensor,
        *,
        stream_ids: Optional[Sequence[Hashable]] = None,
    ) -> Mapping[str, Tensor]:
        """Advance one frame for one or more independently identified streams."""

        encoded, squeeze = self._advance_encoded(
            frame_features, stream_ids=stream_ids
        )
        batch_encoded = encoded.unsqueeze(0) if squeeze else encoded
        outputs = self.offline_model.head(batch_encoded.unsqueeze(1))
        squeezed = {key: value[:, 0] for key, value in outputs.items()}
        if squeeze:
            return {key: value[0] for key, value in squeezed.items()}
        return squeezed

    def step_with_embedding(
        self,
        frame_features: Tensor,
        *,
        stream_ids: Optional[Sequence[Hashable]] = None,
    ) -> Tuple[Mapping[str, Tensor], Tensor]:
        """Advance once and also expose the frozen 128-D TCN embedding."""

        encoded, squeeze = self._advance_encoded(
            frame_features, stream_ids=stream_ids
        )
        batch_encoded = encoded.unsqueeze(0) if squeeze else encoded
        raw = self.offline_model.head(batch_encoded.unsqueeze(1))
        outputs = {key: value[:, 0] for key, value in raw.items()}
        if squeeze:
            return {key: value[0] for key, value in outputs.items()}, encoded
        return outputs, encoded

    @torch.inference_mode()
    def expected_final_buffers(self, features: Tensor) -> Tuple[Tensor, ...]:
        """Return offline per-layer input tails using the state buffer contract."""

        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("features must be [B,T,D]")
        encoded = self.offline_model.backbone.frame_encoder(features)
        output = []
        for block in self.offline_model.backbone.blocks:
            padding = block.left_padding
            if encoded.shape[1] >= padding:
                tail = encoded[:, -padding:, :]
            else:
                zeros = torch.zeros(
                    encoded.shape[0],
                    padding - encoded.shape[1],
                    encoded.shape[2],
                    device=encoded.device,
                    dtype=encoded.dtype,
                )
                tail = torch.cat((zeros, encoded), dim=1)
            output.append(tail)
            encoded = block(encoded)
        return tuple(output)


__all__ = ["StatefulContinuousBaseline", "StreamingTCNState"]
