"""Integrated Gradients and channel-topography visualisations."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

BIPOLAR_NAMES = [
    "Fp1-F7", "F7-T3", "T3-T5", "T5-O1",
    "Fp2-F8", "F8-T4", "T4-T6", "T6-O2",
    "Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
    "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
    "Fz-Cz", "Cz-Pz",
]

COORDS_10_20 = {
    "Fp1": (-0.3, 0.85), "Fp2": (0.3, 0.85),
    "F7": (-0.7, 0.55), "F3": (-0.35, 0.55), "Fz": (0.0, 0.6),
    "F4": (0.35, 0.55), "F8": (0.7, 0.55),
    "T3": (-0.85, 0.0), "C3": (-0.45, 0.0), "Cz": (0.0, 0.0),
    "C4": (0.45, 0.0), "T4": (0.85, 0.0),
    "T5": (-0.7, -0.55), "P3": (-0.35, -0.55), "Pz": (0.0, -0.6),
    "P4": (0.35, -0.55), "T6": (0.7, -0.55),
    "O1": (-0.3, -0.85), "O2": (0.3, -0.85),
}

BIPOLAR_PAIRS = [
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fz", "Cz"), ("Cz", "Pz"),
]


def _bipolar_coords() -> np.ndarray:
    coords = []
    for a, b in BIPOLAR_PAIRS:
        x1, y1 = COORDS_10_20[a]
        x2, y2 = COORDS_10_20[b]
        coords.append(((x1 + x2) / 2, (y1 + y2) / 2))
    return np.array(coords)


def integrated_gradients(model, input_tensor: torch.Tensor, target_class: int,
                          baseline: torch.Tensor = None, steps: int = 30) -> np.ndarray:
    """Return attribution of shape ``(C, T)`` for a single ``(1, C, T)`` input."""
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    total_grad = torch.zeros_like(input_tensor)
    for alpha in np.linspace(0, 1, steps + 1):
        x = (baseline + alpha * (input_tensor - baseline)).clone().detach()
        x.requires_grad_(True)
        logits = model(x)
        score = logits[0, target_class]
        model.zero_grad()
        score.backward()
        total_grad += x.grad.detach()
    avg = total_grad / (steps + 1)
    attr = (input_tensor - baseline) * avg
    return attr.squeeze(0).cpu().numpy()


def plot_topomap(importance: np.ndarray, save_path: str, title: str = "Channel importance") -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_artist(plt.Circle((0, 0), 0.95, color="black", fill=False, lw=2.5))
    ax.plot([-0.15, 0, 0.15], [0.95, 1.08, 0.95], color="black", lw=2.5)
    coords = _bipolar_coords()
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=importance, cmap="Reds",
                    s=250, edgecolors="black", linewidth=0.8, zorder=5, vmin=0)
    for i, name in enumerate(BIPOLAR_NAMES):
        ax.annotate(name, coords[i], textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=7, weight="bold")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02).set_label("Importance")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_eeg_with_attribution(data: np.ndarray, attribution: np.ndarray,
                              save_path: str, sfreq: int = 256,
                              scale: float = 1.5, cmap: str = "inferno") -> None:
    """``data`` and ``attribution`` are ``(C, T)``."""
    n_ch, n_samples = data.shape
    time = np.arange(n_samples) / sfreq
    spacing = np.max(np.abs(data)) * scale or 1.0
    offsets = np.arange(n_ch)[::-1] * spacing

    attr_abs = np.abs(attribution)
    norm = plt.Normalize(
        vmin=np.percentile(attr_abs, 5),
        vmax=np.percentile(attr_abs, 95) + 1e-9,
    )
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(18, 11), dpi=120)
    for ch in range(n_ch):
        signal = data[ch] + offsets[ch]
        points = np.array([time, signal]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colours = cmap_obj(norm(attr_abs[ch, :-1]))
        ax.add_collection(LineCollection(segments, colors=colours, linewidths=1.1))

    ax.set_yticks(offsets)
    ax.set_yticklabels(BIPOLAR_NAMES, fontsize=9)
    ax.set_xticks(np.arange(0, time[-1], 1.0))
    ax.grid(True, axis="x", color="red", linestyle="-", alpha=0.15)
    ax.set_xlim(0, time[-1])
    ax.set_xlabel("Time (s)")
    ax.set_title("EEG with Integrated Gradients")
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                 ax=ax, aspect=30, pad=0.02).set_label("|Attribution|")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
