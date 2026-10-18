"""M2: single-axis Transformer with learnable-query cross-attention.

Paper's ablation variant: same multi-scale front-end as M1, but replaces
the factorised design with a single temporal Transformer (4 layers, 8 heads)
and a learnable query for pooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import MultiScaleConvBlock


class EEGNetDeepTransformerCrossAttn(nn.Module):
    def __init__(
        self,
        chunk_size: int,
        num_electrodes: int,
        F1: int = 8,
        D: int = 2,
        num_classes: int = 6,
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
        self.T = chunk_size // (2 ** 4)

        self.proj = nn.Linear(self.H, 256)
        self.pos_enc = nn.Parameter(torch.randn(1, self.T, 256) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.global_query = nn.Parameter(torch.randn(1, 1, 256))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=dropout, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)                 # (B, H, T')
        x = x.permute(0, 2, 1)           # (B, T', H)
        x = self.proj(x)                 # (B, T', 256)
        x = x + self.pos_enc
        x = self.transformer(x)          # (B, T', 256)

        query = self.global_query.expand(x.size(0), -1, -1)
        attn_out, _ = self.cross_attn(query, x, x)
        x = attn_out.squeeze(1)
        return self.classifier(x)


def create_m2(
    chunk_size: int = 2048,
    num_electrodes: int = 18,
    F1: int = 8,
    F2: int = 16,
    D: int = 2,
    num_classes: int = 6,
    kernel_1: int = 64,
    kernel_2: int = 16,
    dropout: float = 0.3,
) -> EEGNetDeepTransformerCrossAttn:
    return EEGNetDeepTransformerCrossAttn(
        chunk_size=chunk_size,
        num_electrodes=num_electrodes,
        F1=F1,
        D=D,
        num_classes=num_classes,
        dropout=dropout,
    )
