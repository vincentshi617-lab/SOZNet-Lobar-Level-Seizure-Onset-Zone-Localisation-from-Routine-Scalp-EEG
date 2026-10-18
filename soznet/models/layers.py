"""Shared building blocks: multi-scale conv front-end and factorised attention."""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class MultiScaleConvBlock(nn.Module):
    """Parallel 1D convs over time with kernel sizes {16, 32, 64, 128}.

    Output shape is ``(B, F1 * len(kernel_sizes) * D, T / 2**downsamples)``.
    The single spatial axis (``num_electrodes``) is fused by a depthwise
    convolution, so the output is a channel-free spatiotemporal token tensor.
    """

    def __init__(
        self,
        num_electrodes: int,
        F1: int = 8,
        D: int = 2,
        kernel_sizes: List[int] = (16, 32, 64, 128),
        pool_size: int = 2,
        downsamples: int = 4,
    ) -> None:
        super().__init__()
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.D = D
        self.kernel_sizes = list(kernel_sizes)
        self.pool_size = pool_size
        self.downsamples = downsamples

        self.time_convs = nn.ModuleList(
            [
                nn.Conv2d(1, F1, kernel_size=(1, ks), padding=(0, ks // 2), bias=False)
                for ks in self.kernel_sizes
            ]
        )

        total_f1 = F1 * len(self.kernel_sizes)
        self.bn1 = nn.BatchNorm2d(total_f1)
        self.act1 = nn.ELU(inplace=True)

        self.depthwise = nn.Conv2d(
            total_f1,
            total_f1 * D,
            kernel_size=(num_electrodes, 1),
            groups=total_f1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(total_f1 * D)
        self.act2 = nn.ELU(inplace=True)

        self.pools = nn.ModuleList(
            [
                nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
                for _ in range(downsamples)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [conv(x) for conv in self.time_convs]
        x = torch.cat(branch_outs, dim=1)
        x = self.act1(self.bn1(x))

        x = self.act2(self.bn2(self.depthwise(x)))

        for pool in self.pools:
            x = pool(x)
        x = x.squeeze(2)
        return x


class FactorizedTransformer(nn.Module):
    """Decoupled temporal / channel Transformer encoders.

    Inputs
    ------
    x_time : (B, T, hidden_dim)
    x_chan : (B, H, hidden_dim)

    Output
    ------
    x_fused : (B, H, hidden_dim)
        Sum of channel-encoder output and broadcast of the globally pooled
        temporal token.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_temporal_len: int = 512,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=num_layers
        )

        channel_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.channel_encoder = nn.TransformerEncoder(
            channel_layer, num_layers=num_layers
        )

        self.register_buffer(
            "temporal_pos",
            self._sinusoidal_pos_enc(max_temporal_len, hidden_dim),
        )
        self.channel_pos = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    @staticmethod
    def _sinusoidal_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x_time: torch.Tensor, x_chan: torch.Tensor) -> torch.Tensor:
        T = x_time.size(1)
        x_time = x_time + self.temporal_pos[:, :T, :].to(x_time.device)
        x_time = self.temporal_encoder(x_time)

        x_chan = x_chan + self.channel_pos.to(x_chan.device)
        x_chan = self.channel_encoder(x_chan)

        x_time_global = x_time.mean(dim=1, keepdim=True)
        return x_chan + x_time_global


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (used by M3)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :].to(x.device)
