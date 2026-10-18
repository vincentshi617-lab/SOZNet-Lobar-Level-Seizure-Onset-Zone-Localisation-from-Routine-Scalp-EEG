"""Evaluate saved fused models and optionally generate visualisations.

Example
-------
python -m soznet.evaluate_ensemble --model-dir outputs/ensemble --output-dir outputs/eval
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from .config import (
    ANNOTATION_CSV,
    CHUNK_SIZE,
    DATA_ROOT,
    D,
    DROPOUT,
    F1,
    F2,
    KERNEL_1,
    KERNEL_2,
    NUM_CLASSES,
    NUM_ELECTRODES,
    SEED,
    TEST_RATIO,
    VAL_RATIO,
)
from .data import EEGDataset, load_labels, split_patients
from .engine import evaluate_with_paths
from .models import create_model
from .postprocess import aggregate_sessions, apply_hemisphere_constraint, compute_metrics, print_metrics
from .training import make_loader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["m1", "m2", "m3"], default="m1")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", default="outputs/eval")
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--annotation-csv", default=ANNOTATION_CSV)
    p.add_argument("--num-curves", type=int, default=5)
    p.add_argument("--num-explain-windows", type=int, default=10)
    p.add_argument("--no-vis", action="store_true")
    return p.parse_args()


def arch_kwargs() -> dict:
    return dict(
        chunk_size=CHUNK_SIZE, num_electrodes=NUM_ELECTRODES,
        F1=F1, F2=F2, D=D, num_classes=NUM_CLASSES,
        kernel_1=KERNEL_1, kernel_2=KERNEL_2, dropout=DROPOUT,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_dict = load_labels(args.annotation_csv)
    _, _, test_keys = split_patients(label_dict, TEST_RATIO, VAL_RATIO, SEED)
    test_ds = EEGDataset(
        args.data_root, label_dict, allowed_keys=test_keys, chunk_size=CHUNK_SIZE,
    )
    test_loader = make_loader(test_ds, False)

    paths = sorted(glob.glob(os.path.join(args.model_dir, "fused_*.pth")))
    if not paths:
        raise FileNotFoundError(f"No fused_*.pth found in {args.model_dir}")
    print(f"Found {len(paths)} fused models")

    all_probs, labels, patients, sessions = [], None, None, None
    first_paths = None
    for i, p in enumerate(paths):
        model = create_model(args.model, **arch_kwargs()).to(device)
        model.load_state_dict(torch.load(p, map_location=device))
        model.eval()
        if i == 0:
            first_paths, labels, probs, patients, sessions = evaluate_with_paths(
                model, test_loader, test_ds, device
            )
        else:
            _, _, probs, _, _ = evaluate_with_paths(model, test_loader, test_ds, device)
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    print_metrics("Window-level", compute_metrics(labels, (avg_probs > 0.3).astype(float)))

    s_labels, s_probs, session_keys = aggregate_sessions(labels, avg_probs, patients, sessions)
    s_preds = apply_hemisphere_constraint(s_probs)
    print_metrics("Session-level", compute_metrics(s_labels, s_preds))

    if args.no_vis:
        return

    from .visualization import (
        make_session_animations,
        plot_probability_curves,
        plot_region_accuracy,
        plot_session_f1,
        plot_vote_histogram,
    )

    plot_session_f1(s_labels, s_preds, os.path.join(args.output_dir, "f1_bar.png"))
    plot_region_accuracy(s_labels, s_preds, os.path.join(args.output_dir, "region_accuracy.png"))
    plot_vote_histogram(np.stack(all_probs, axis=0), s_labels,
                        os.path.join(args.output_dir, "vote_hist.png"))
    plot_probability_curves(session_keys, s_probs,
                            os.path.join(args.output_dir, "curves"))
    make_session_animations(first_paths, labels, avg_probs, patients, sessions,
                            os.path.join(args.output_dir, "animations"))
    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
