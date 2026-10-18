"""M1 / SOZNet: multi-scale conv front-end + factorised Transformer."""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import MultiScaleConvBlock, FactorizedTransformer


class EEGNetEnhanced(nn.Module):
    """Factorised spatiotemporal Transformer (paper's M1, ~0.8M params)."""

    def __init__(
        self,
        chunk_size: int,
        num_electrodes: int,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        num_classes: int = 6,
        kernel_1: int = 64,
        kernel_2: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv = MultiScaleConvBlock(
            num_electrodes=num_electrodes,
            F1=F1,
            D=D,
            kernel_sizes=[16, 32, 64, 128],
            pool_size=2,
            downsamples=4,
        )

        self.H = F1 * 4 * D
        self.T_prime = chunk_size // (2 ** 4)
        hidden_dim = 128

        self.time_proj = nn.Linear(self.H, hidden_dim)
        self.chan_proj = nn.Linear(self.T_prime, hidden_dim)

        self.transformer = FactorizedTransformer(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=dropout,
        )

        self.out_proj = nn.Linear(hidden_dim, self.H)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.H, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                  # (B, H, T')

        x_time = x.permute(0, 2, 1)       # (B, T', H)
        x_time = self.time_proj(x_time)   # (B, T', hidden)

        x_chan = x.permute(0, 1, 2)       # (B, H, T')
        x_chan = self.chan_proj(x_chan)   # (B, H, hidden)

        x_fused = self.transformer(x_time, x_chan)

        x_pool = x_fused.mean(dim=1)
        x_pool = self.out_proj(x_pool)
        return self.classifier(x_pool)


def create_m1(
    chunk_size: int,
    num_electrodes: int,
    F1: int = 8,
    F2: int = 16,
    D: int = 2,
    num_classes: int = 6,
    kernel_1: int = 64,
    kernel_2: int = 16,
    dropout: float = 0.5,
) -> EEGNetEnhanced:
    return EEGNetEnhanced(
        chunk_size=chunk_size,
        num_electrodes=num_electrodes,
        F1=F1,
        F2=F2,
        D=D,
        num_classes=num_classes,
        kernel_1=kernel_1,
        kernel_2=kernel_2,
        dropout=dropout,
    )
