"""Session-level smoothing and brain-region animations."""
from __future__ import annotations

import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from ..config import REGION_NAMES


def _smooth_session(items, n_bins: int = 80):
    items = sorted(items, key=lambda x: x[0])
    orders = np.array([it[0] for it in items])
    prob_arr = np.array([it[1] for it in items])

    edges = np.linspace(0, len(items), n_bins + 1, dtype=int)
    sampled_orders, sampled_probs = [], []
    for j in range(n_bins):
        s, e = edges[j], edges[j + 1]
        if s == e:
            mid, p = orders[s], prob_arr[s]
        else:
            mid, p = np.mean(orders[s:e]), prob_arr[s:e].mean(axis=0)
        sampled_orders.append(mid)
        sampled_probs.append(p)
    return np.array(sampled_orders), np.array(sampled_probs)


def _brain_patches(ax):
    left = Polygon([(0.1, 0.2), (0.3, 0.05), (0.45, 0.05), (0.45, 0.95),
                    (0.3, 0.95), (0.1, 0.8)], closed=True,
                   facecolor="none", edgecolor="black", lw=1.5)
    right = Polygon([(0.9, 0.2), (0.7, 0.05), (0.55, 0.05), (0.55, 0.95),
                     (0.7, 0.95), (0.9, 0.8)], closed=True,
                    facecolor="none", edgecolor="black", lw=1.5)
    lf = Polygon([(0.15, 0.7), (0.35, 0.7), (0.4, 0.85), (0.15, 0.85)],
                 closed=True, facecolor="lightgray", edgecolor="gray")
    lt = Polygon([(0.15, 0.4), (0.35, 0.4), (0.4, 0.55), (0.15, 0.55)],
                 closed=True, facecolor="lightgray", edgecolor="gray")
    lo = Polygon([(0.15, 0.1), (0.35, 0.1), (0.4, 0.25), (0.15, 0.25)],
                 closed=True, facecolor="lightgray", edgecolor="gray")
    rf = Polygon([(0.85, 0.7), (0.65, 0.7), (0.6, 0.85), (0.85, 0.85)],
                 closed=True, facecolor="lightgray", edgecolor="gray")
    rt = Polygon([(0.85, 0.4), (0.65, 0.4), (0.6, 0.55), (0.85, 0.55)],
                 closed=True, facecolor="lightgray", edgecolor="gray")
    ro = Polygon([(0.85, 0.1), (0.65, 0.1), (0.6, 0.25), (0.85, 0.25)],
                 closed=True, facecolor="lightgray", edgecolor="gray")
    regions = [lf, rf, lt, rt, lo, ro]
    for patch in [left, right] + regions:
        ax.add_patch(patch)
    for name, (x, y) in zip(REGION_NAMES, [
        (0.23, 0.78), (0.77, 0.78), (0.23, 0.48),
        (0.77, 0.48), (0.23, 0.18), (0.77, 0.18),
    ]):
        ax.text(x, y, name, fontsize=7, ha="center", va="center", weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return regions


def make_session_animations(paths, labels, probs, patients, sessions, output_dir, fps: int = 3) -> None:
    """For each session, save a smoothed CSV, curve, and brain GIF."""
    os.makedirs(output_dir, exist_ok=True)
    groups = defaultdict(list)
    for p, l, pr, pt, sn in zip(paths, labels, probs, patients, sessions):
        m = re.search(r"epoch_(\d+)", os.path.basename(p))
        order = int(m.group(1)) if m else 0
        groups[(pt, sn)].append((order, pr, l))

    for (pt, sn), items in groups.items():
        if not items:
            continue
        sampled_orders, sampled_probs = _smooth_session(items)

        csv_path = os.path.join(output_dir, f"{pt}_{sn}_confidence.csv")
        pd.DataFrame(sampled_probs, columns=[f"prob_{r}" for r in REGION_NAMES]).assign(
            time_index=sampled_orders
        ).to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(6):
            ax.plot(sampled_orders, sampled_probs[:, i], label=REGION_NAMES[i], alpha=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Epoch index")
        ax.set_ylabel("Probability")
        ax.set_title(f"{pt} / {sn} — smoothed probabilities")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{pt}_{sn}_curve.png"), dpi=150)
        plt.close(fig)

        fig, (ax_brain, ax_conf) = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]}
        )
        region_patches = _brain_patches(ax_brain)
        ax_conf.set_xlim(sampled_orders[0], sampled_orders[-1])
        ax_conf.set_ylim(0, 1.05)
        ax_conf.set_xlabel("Epoch index")
        ax_conf.set_ylabel("Confidence (max prob)")
        line_conf, = ax_conf.plot([], [], "r-", lw=2, label="confidence")
        ax_conf.legend(loc="upper right")
        marker = ax_conf.axvline(x=sampled_orders[0], color="gray", linestyle="--", alpha=0.5)
        xs, ys = [], []

        def update(frame):
            prob = sampled_probs[frame]
            for patch, p in zip(region_patches, prob):
                patch.set_facecolor(plt.cm.Blues(p))
            xs.append(sampled_orders[frame])
            ys.append(float(np.max(prob)))
            line_conf.set_data(xs, ys)
            marker.set_xdata([sampled_orders[frame]])
            top = REGION_NAMES[int(np.argmax(prob))]
            ax_brain.set_title(f"{pt}/{sn} time~{sampled_orders[frame]:.0f} | {top} ({np.max(prob):.2f})")
            return marker, line_conf

        ani = matplotlib.animation.FuncAnimation(
            fig, update, frames=len(sampled_probs),
            interval=1000 // max(fps, 1), blit=False,
        )
        ani.save(os.path.join(output_dir, f"{pt}_{sn}_prediction.gif"),
                 writer="pillow", fps=fps)
        plt.close(fig)
        print(f"Saved animation for {pt}/{sn}")
