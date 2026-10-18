"""Train a heterogeneous ensemble with model soup and soft voting.

Phase 1: train N base models (different seeds).
Phase 2: for each base model, fine-tune M variants and average weights (soup).
Phase 3: soft-vote all fused models at inference.

Example
-------
python -m soznet.train_ensemble --model m1 --num-models 3 --finetune-m 3
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

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
    FINETUNE_M,
    KERNEL_1,
    KERNEL_2,
    NUM_CLASSES,
    NUM_ELECTRODES,
    OUTPUT_DIR,
    SEED,
    TEST_RATIO,
    VAL_RATIO,
)
from .data import EEGDataset, load_labels, split_patients
from .engine import evaluate_detailed, evaluate_with_paths
from .models import create_model
from .postprocess import aggregate_sessions, apply_hemisphere_constraint, compute_metrics, print_metrics
from .training import (
    average_weights,
    build_datasets,
    finetune_variant,
    fit,
    make_loader,
)
from .train_single import set_seed


def arch_kwargs() -> dict:
    return dict(
        chunk_size=CHUNK_SIZE,
        num_electrodes=NUM_ELECTRODES,
        F1=F1, F2=F2, D=D,
        num_classes=NUM_CLASSES,
        kernel_1=KERNEL_1, kernel_2=KERNEL_2,
        dropout=DROPOUT,
    )


def phase1_train_base_models(
    args, label_dict, train_keys, val_keys,
) -> List[str]:
    print(f"=== Phase 1: training {args.num_models} base models ===")
    base_paths = []
    for i in range(args.num_models):
        seed = args.start_seed + i
        name = f"base_{args.model}_seed{seed}"
        path = os.path.join(args.output_dir, f"{name}.pth")
        base_paths.append(path)
        if os.path.exists(path):
            print(f"[base seed {seed}] found {path}, skipping")
            continue
        set_seed(seed)
        train_ds, val_ds = build_datasets(
            args.data_root, label_dict, train_keys, val_keys,
            use_aug=not args.no_aug, seed=seed,
        )
        model = create_model(args.model, **arch_kwargs()).to(args.device)
        fit(
            model, make_loader(train_ds, True), make_loader(val_ds, False),
            args.device, output_dir=args.output_dir,
            name_prefix=name, epochs=args.epochs, patience=args.patience,
        )
    return base_paths


def phase2_finetune_and_soup(
    args, label_dict, train_keys, base_paths: List[str],
) -> List[Dict[str, torch.Tensor]]:
    print(f"=== Phase 2: fine-tune {args.finetune_m} variants per base model and soup ===")
    fused_states = []
    for base_path in base_paths:
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        seed_tag = base_name.split("seed")[-1]
        try:
            base_seed = int(seed_tag)
        except ValueError:
            base_seed = abs(hash(base_name)) % 10000

        variant_states = []
        for m in range(args.finetune_m):
            v_seed = base_seed * 100 + m
            set_seed(v_seed)
            train_ds, _ = build_datasets(
                args.data_root, label_dict, train_keys, [],
                use_aug=not args.no_aug, seed=v_seed,
            )
            model = create_model(args.model, **arch_kwargs()).to(args.device)
            state = finetune_variant(
                torch.load(base_path, map_location=args.device),
                model, make_loader(train_ds, True), args.device,
            )
            variant_states.append(state)
            print(f"[{base_name}] variant {m} done")

        fused = average_weights(variant_states)
        fused_states.append(fused)
        fused_path = os.path.join(args.output_dir, f"fused_{base_name}.pth")
        torch.save(fused, fused_path)
        print(f"[{base_name}] soup saved to {fused_path}")
    return fused_states


def phase3_soft_vote(
    args, label_dict, test_keys, fused_states: List[Dict[str, torch.Tensor]],
):
    print(f"=== Phase 3: soft-voting {len(fused_states)} fused models ===")
    test_ds = EEGDataset(
        args.data_root, label_dict, allowed_keys=test_keys, chunk_size=CHUNK_SIZE,
    )
    test_loader = make_loader(test_ds, False)

    all_probs, all_labels, all_patients, all_sessions = [], None, None, None
    for i, state in enumerate(fused_states):
        model = create_model(args.model, **arch_kwargs()).to(args.device)
        model.load_state_dict(state)
        model.eval()
        labels, _, probs, patients, sessions = evaluate_detailed(model, test_loader, args.device)
        all_probs.append(probs)
        if all_labels is None:
            all_labels, all_patients, all_sessions = labels, patients, sessions
        print(f"fused model {i + 1}: window EM = {compute_metrics(labels, (probs > 0.3).astype(float))['exact_match']:.4f}")

    avg_probs = np.mean(all_probs, axis=0)
    preds = (avg_probs > 0.3).astype(float)
    print_metrics("Window-level (ensemble)", compute_metrics(all_labels, preds))

    s_labels, s_probs, _ = aggregate_sessions(all_labels, avg_probs, all_patients, all_sessions)
    s_preds = apply_hemisphere_constraint(s_probs)
    print_metrics("Session-level (ensemble)", compute_metrics(s_labels, s_preds))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SOZNet ensemble")
    p.add_argument("--model", choices=["m1", "m2", "m3"], default="m1")
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--annotation-csv", default=ANNOTATION_CSV)
    p.add_argument("--output-dir", default=os.path.join(OUTPUT_DIR, "ensemble"))
    p.add_argument("--num-models", type=int, default=3)
    p.add_argument("--finetune-m", type=int, default=FINETUNE_M)
    p.add_argument("--start-seed", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--no-aug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(SEED)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_dict = load_labels(args.annotation_csv)
    train_keys, val_keys, test_keys = split_patients(
        label_dict, TEST_RATIO, VAL_RATIO, SEED,
    )

    base_paths = phase1_train_base_models(args, label_dict, train_keys, val_keys)
    fused_states = phase2_finetune_and_soup(args, label_dict, train_keys, base_paths)
    phase3_soft_vote(args, label_dict, test_keys, fused_states)


if __name__ == "__main__":
    main()
