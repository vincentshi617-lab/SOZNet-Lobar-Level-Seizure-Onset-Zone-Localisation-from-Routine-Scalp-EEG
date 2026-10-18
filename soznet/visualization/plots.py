"""Static result plots (F1, accuracy, vote histogram, probability curves)."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from ..config import REGION_NAMES


def _safe_close(fig):
    plt.tight_layout()
    plt.close(fig)


def plot_session_f1(labels: np.ndarray, preds: np.ndarray, save_path: str) -> None:
    f1s = []
    for r in range(labels.shape[1]):
        _, _, f, _ = precision_recall_fscore_support(
            labels[:, r], preds[:, r], average="binary", zero_division=0
        )
        f1s.append(f)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(REGION_NAMES, f1s, color="steelblue")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1")
    ax.set_title("Session-level F1 (Ensemble)")
    plt.xticks(rotation=30)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    _safe_close(fig)


def plot_region_accuracy(labels: np.ndarray, preds: np.ndarray, save_path: str) -> None:
    total = len(labels)
    correct = [int(((labels[:, r] == preds[:, r]).sum())) for r in range(labels.shape[1])]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(REGION_NAMES))
    bars = ax.bar(x, correct, color="mediumseagreen", alpha=0.85)
    for bar, c in zip(bars, correct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{c}/{total}\n({c / total * 100:.1f}%)", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_NAMES, rotation=30)
    ax.set_ylim(0, total * 1.15)
    ax.set_ylabel("Number of sessions")
    ax.set_title("Prediction accuracy per brain region")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    _safe_close(fig)


def plot_vote_histogram(all_model_probs: np.ndarray, labels: np.ndarray, save_path: str, threshold: float = 0.3) -> None:
    """``all_model_probs`` has shape ``(M, N_sessions, 6)``."""
    votes = (all_model_probs > threshold).sum(axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for r in range(6):
        ax = axes[r]
        ax.hist(votes[:, r], bins=np.arange(0, 7) - 0.5, rwidth=0.8,
                color="skyblue", edgecolor="black", alpha=0.75)
        ax.set_title(f"{REGION_NAMES[r]} (True pos: {int(labels[:, r].sum())}/{len(labels)})",
                     fontsize=10)
        ax.set_xlabel("Votes (0–5 models)")
        ax.set_ylabel("Sessions")
        ax.set_xticks(range(6))
    fig.suptitle("Model agreement & ground truth context", fontsize=16)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=150)
    _safe_close(fig)


def plot_probability_curves(session_keys, session_probs: np.ndarray, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for i, (pat, ses) in enumerate(session_keys):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.arange(6), session_probs[i], color="steelblue", alpha=0.85)
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(REGION_NAMES, rotation=45)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probability")
        ax.set_title(f"Session {pat} / {ses}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{pat}_{ses}_probs.png"), dpi=150)
        _safe_close(fig)


def plot_training_curves(history: dict, save_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], "o-", label="Train")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, history["train_acc"], "o-", label="Train")
    ax2.plot(epochs, history["val_acc"], "s-", label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    _safe_close(fig)
