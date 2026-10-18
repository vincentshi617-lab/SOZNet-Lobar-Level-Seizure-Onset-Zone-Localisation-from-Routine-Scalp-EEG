"""Multi-label focal loss and hemispheric mutual-exclusivity penalty."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LEFT_REGIONS, RIGHT_REGIONS


class MultilabelFocalLoss(nn.Module):
    """Multi-label focal loss with optional per-class positive weighting."""

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
            else:
                alpha = torch.as_tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def hemispheric_penalty(
    logits: torch.Tensor,
    left_regions: List[int] = None,
    right_regions: List[int] = None,
) -> torch.Tensor:
    """Product of the maximum left- and right-hemisphere probabilities.

    Minimised only when one hemisphere has near-zero probability, which
    enforces the clinical prior of unilateral focal epilepsy.
    """
    left_regions = left_regions or LEFT_REGIONS
    right_regions = right_regions or RIGHT_REGIONS

    probs = torch.sigmoid(logits)
    left_max = probs[:, left_regions].max(dim=1).values
    right_max = probs[:, right_regions].max(dim=1).values
    return (left_max * right_max).mean()
