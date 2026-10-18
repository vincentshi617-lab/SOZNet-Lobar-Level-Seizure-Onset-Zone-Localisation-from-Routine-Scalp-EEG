"""M3: deeper single-axis Transformer (6 layers, 12 heads)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import MultiScaleConvBlock, PositionalEncoding


class EEGNetDeepTransformerV2(nn.Module):
    def __init__(
        self,
        chunk_size: int,
        num_electrodes: int,
        F1: int = 8,
        D: int = 2,
        num_classes: int = 6,
        dropout: float = 0.35,
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

        self.d_model = 384
        self.proj = nn.Linear(self.H, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model, max_len=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=12,
            dim_feedforward=1536,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.attn_pool = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        self.aux_classifier = nn.Sequential(nn.Linear(self.d_model, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)                  # (B, H, T')
        x = x.permute(0, 2, 1)            # (B, T', H)
        x = self.proj(x)                  # (B, T', d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)

        query = self.query.expand(x.size(0), -1, -1)
        x_pooled, _ = self.attn_pool(query, x, x)
        x_feat = x_pooled.squeeze(1)

        logits = self.classifier(x_feat)
        if return_aux:
            return logits, self.aux_classifier(x_feat)
        return logits


def create_m3(
    chunk_size: int = 2048,
    num_electrodes: int = 18,
    F1: int = 8,
    F2: int = 16,
    D: int = 2,
    num_classes: int = 6,
    kernel_1: int = 64,
    kernel_2: int = 16,
    dropout: float = 0.35,
) -> EEGNetDeepTransformerV2:
    return EEGNetDeepTransformerV2(
        chunk_size=chunk_size,
        num_electrodes=num_electrodes,
        F1=F1,
        D=D,
        num_classes=num_classes,
        dropout=dropout,
    )
