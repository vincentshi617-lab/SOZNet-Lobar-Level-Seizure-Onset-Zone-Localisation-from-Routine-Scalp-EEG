"""Session-level aggregation and metric computation shared by all entry scripts."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
)

from .config import LEFT_REGIONS, RIGHT_REGIONS, REGION_NAMES, SESSION_PROB_THRESHOLD


def aggregate_sessions(
    labels: np.ndarray,
    probs: np.ndarray,
    patients: List[str],
    sessions: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Average per-window probabilities within each (patient, session)."""
    prob_buckets: "OrderedDict[Tuple[str, str], list]" = OrderedDict()
    label_buckets: Dict[Tuple[str, str], np.ndarray] = {}

    for lb, pr, pat, ses in zip(labels, probs, patients, sessions):
        key = (pat, ses)
        prob_buckets.setdefault(key, []).append(pr)
        label_buckets.setdefault(key, lb)

    keys = list(prob_buckets.keys())
    session_labels = np.stack([label_buckets[k] for k in keys])
    session_probs = np.stack(
        [np.mean(np.stack(prob_buckets[k], axis=0), axis=0) for k in keys]
    )
    return session_labels, session_probs, keys


def apply_hemisphere_constraint(
    session_probs: np.ndarray,
    threshold: float = SESSION_PROB_THRESHOLD,
    left_regions: List[int] = None,
    right_regions: List[int] = None,
) -> np.ndarray:
    """Keep only the higher-probability hemisphere, then threshold."""
    left_regions = left_regions or LEFT_REGIONS
    right_regions = right_regions or RIGHT_REGIONS

    preds = np.zeros_like(session_probs)
    for i in range(session_probs.shape[0]):
        prob = session_probs[i].copy()
        if prob[left_regions].max() > prob[right_regions].max():
            prob[right_regions] = 0
        else:
            prob[left_regions] = 0
        preds[i] = (prob > threshold).astype(float)
    return preds


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Exact Match, Hamming Loss, and per-region P/R/F1."""
    metrics: Dict[str, float] = {
        "exact_match": float(accuracy_score(labels, preds)),
        "hamming_loss": float(hamming_loss(labels, preds)),
    }
    for r, name in enumerate(REGION_NAMES):
        p, rec, f1, _ = precision_recall_fscore_support(
            labels[:, r], preds[:, r], average="binary", zero_division=0
        )
        metrics[f"{name}_precision"] = float(p)
        metrics[f"{name}_recall"] = float(rec)
        metrics[f"{name}_f1"] = float(f1)
    return metrics


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n===== {title} =====")
    print(f"Exact Match: {metrics['exact_match']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    for name in REGION_NAMES:
        print(
            f"{name}: P={metrics[f'{name}_precision']:.3f} "
            f"R={metrics[f'{name}_recall']:.3f} "
            f"F1={metrics[f'{name}_f1']:.3f}"
        )
