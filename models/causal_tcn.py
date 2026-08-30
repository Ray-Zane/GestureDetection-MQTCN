"""Strictly causal depthwise-separable temporal convolution network."""

from __future__ import annotations

from typing import Sequence

import torch.nn.functional as functional
from torch import Tensor, nn

from models.frame_encoder import FrameEncoder


class CausalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size <= 1 or dilation <= 0:
            raise ValueError("kernel_size must be >1 and dilation must be positive")
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError(f"expected [B,T,C], got {tuple(sequence.shape)}")
        residual = sequence
        channels_first = sequence.transpose(1, 2)
        output = self.depthwise(functional.pad(channels_first, (self.left_padding, 0)))
        output = self.pointwise(output).transpose(1, 2)
        output = self.norm(output)
        output = self.activation(output)
        return residual + self.dropout(output)


class CausalTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(value) for value in dilations)
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must contain positive integers")
        self.frame_encoder = FrameEncoder(input_dim, hidden_dim, dropout)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(
                hidden_dim,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in self.dilations
        )

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, features: Tensor) -> Tensor:
        encoded = self.frame_encoder(features)
        for block in self.blocks:
            encoded = block(encoded)
        return encoded


__all__ = ["CausalResidualBlock", "CausalTCN"]
