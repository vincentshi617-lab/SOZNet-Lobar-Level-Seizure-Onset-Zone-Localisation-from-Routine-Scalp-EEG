"""Train a single SOZNet model (M1/M2/M3) and evaluate at window/session level.

Example
-------
python -m soznet.train_single --model m1 --output-dir outputs/single_m1
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from .config import (
    ANNOTATION_CSV,
    CHUNK_SIZE,
    DATA_ROOT,
    D,
    DROPOUT,
    EPOCHS,
    F1,
    F2,
    KERNEL_1,
    KERNEL_2,
    NUM_CLASSES,
    NUM_ELECTRODES,
    OUTPUT_DIR,
    PATIENCE,
    SEED,
    TEST_RATIO,
    VAL_RATIO,
)
from .data import load_labels, split_patients
from .engine import evaluate_with_paths
from .models import create_model
from .postprocess import aggregate_sessions, apply_hemisphere_constraint, compute_metrics, print_metrics
from .training import build_datasets, fit, make_loader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one SOZNet model")
    p.add_argument("--model", choices=["m1", "m2", "m3"], default="m1")
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--annotation-csv", default=ANNOTATION_CSV)
    p.add_argument("--output-dir", default=os.path.join(OUTPUT_DIR, "single"))
    p.add_argument("--checkpoint", default=None,
                   help="Optional checkpoint; if given, skip training and only evaluate")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-aug", action="store_true", help="Disable online augmentation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_dict = load_labels(args.annotation_csv)
    train_keys, val_keys, test_keys = split_patients(
        label_dict,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=args.seed,
    )

    train_ds, val_ds = build_datasets(
        args.data_root, label_dict, train_keys, val_keys,
        use_aug=not args.no_aug, seed=args.seed,
    )
    from .data import EEGDataset
    test_ds = EEGDataset(
        args.data_root, label_dict,
        allowed_keys=test_keys, chunk_size=CHUNK_SIZE,
    )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    arch_kwargs = dict(
        chunk_size=CHUNK_SIZE,
        num_electrodes=NUM_ELECTRODES,
        F1=F1, F2=F2, D=D,
        num_classes=NUM_CLASSES,
        kernel_1=KERNEL_1, kernel_2=KERNEL_2,
        dropout=DROPOUT,
    )
    model = create_model(args.model, **arch_kwargs).to(device)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        best_state, _ = fit(
            model, train_loader, val_loader, device,
            output_dir=args.output_dir,
            name_prefix=f"{args.model}_seed{args.seed}",
            epochs=args.epochs,
            patience=args.patience,
        )
        model.load_state_dict(best_state)

    # Evaluation
    paths, labels, probs, patients, sessions = evaluate_with_paths(
        model, test_loader, test_ds, device
    )
    preds = (probs > 0.3).astype(float)
    print_metrics("Window-level", compute_metrics(labels, preds))

    s_labels, s_probs, _ = aggregate_sessions(labels, probs, patients, sessions)
    s_preds = apply_hemisphere_constraint(s_probs)
    print_metrics("Session-level (hemisphere constraint)", compute_metrics(s_labels, s_preds))

    np.savez_compressed(
        os.path.join(args.output_dir, "test_predictions.npz"),
        labels=labels, probs=probs, paths=np.array(paths),
        patients=np.array(patients), sessions=np.array(sessions),
    )


if __name__ == "__main__":
    main()
