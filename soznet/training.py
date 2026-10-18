"""Reusable training utilities for single-model and ensemble scripts."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .config import (
    AUG_NUM_AUGMENTED,
    AUG_SCALE_RANGE,
    AUG_SEIZURE_TARGET,
    BATCH_SIZE,
    CHUNK_SIZE,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    FOCAL_GAMMA,
    HEMI_PENALTY_LAMBDA,
    LEARNING_RATE,
    PATIENCE,
    SAMPLES_PER_SESSION,
    USE_SWAP_AUG,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from .data import (
    EEGDataset,
    build_augmented_samples,
    collate_fn,
)
from .engine import train_one_epoch, validate
from .losses import MultilabelFocalLoss


def build_datasets(
    data_root: str,
    label_dict,
    train_keys,
    val_keys,
    use_aug: bool = True,
    samples_per_session: int = SAMPLES_PER_SESSION,
    num_augmented: int = AUG_NUM_AUGMENTED,
    seed: int = 42,
    chunk_size: int = CHUNK_SIZE,
) -> Tuple[EEGDataset, EEGDataset]:
    """Build train/val datasets. Augmentation is applied only to train."""
    train_original = EEGDataset(
        data_root,
        label_dict,
        samples_per_session=samples_per_session,
        allowed_keys=train_keys,
        chunk_size=chunk_size,
    )
    if use_aug:
        samples = build_augmented_samples(
            base_samples=train_original.samples,
            seizure_target=AUG_SEIZURE_TARGET,
            num_augmented=num_augmented,
            seed=seed,
            use_swap=USE_SWAP_AUG,
            scale_range=AUG_SCALE_RANGE,
        )
        train_ds = EEGDataset(
            data_root, label_dict, samples_list=samples, chunk_size=chunk_size
        )
    else:
        train_ds = train_original

    val_ds = EEGDataset(
        data_root,
        label_dict,
        samples_per_session=samples_per_session,
        allowed_keys=val_keys,
        chunk_size=chunk_size,
    )
    return train_ds, val_ds


def make_loader(dataset, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=shuffle,
    )


def make_optimizer_and_scheduler(
    model,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    epochs: int = 30,
    warmup_epochs: int = WARMUP_EPOCHS,
    eta_min: float = 5e-5,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=eta_min
        )
        scheduler = SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    return optimizer, scheduler


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    output_dir: str,
    name_prefix: str,
    epochs: int = 30,
    patience: int = PATIENCE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    hemi_lambda: float = HEMI_PENALTY_LAMBDA,
    focal_gamma: float = FOCAL_GAMMA,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Train with early stopping. Returns ``(best_state_dict, best_val_acc)``."""
    os.makedirs(output_dir, exist_ok=True)
    criterion = MultilabelFocalLoss(gamma=focal_gamma, alpha=None)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, lr=lr, weight_decay=weight_decay, epochs=epochs
    )

    best_acc = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, hemi_lambda=hemi_lambda
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, hemi_lambda=0.0
        )
        scheduler.step()

        if verbose:
            print(
                f"[{name_prefix}] Ep {epoch:02d}: "
                f"train {train_loss:.4f}/{train_acc:.2f}% | "
                f"val {val_loss:.4f}/{val_acc:.2f}%"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{name_prefix}] early stop at epoch {epoch}")
                break

    torch.save(best_state, os.path.join(output_dir, f"{name_prefix}.pth"))
    print(f"[{name_prefix}] best val acc {best_acc:.2f}%")
    return best_state, best_acc


def finetune_variant(
    base_state: Dict[str, torch.Tensor],
    model,
    train_loader: DataLoader,
    device,
    lr: float = FINETUNE_LR,
    epochs: int = FINETUNE_EPOCHS,
    hemi_lambda: float = HEMI_PENALTY_LAMBDA,
) -> Dict[str, torch.Tensor]:
    """Short fine-tune from a base checkpoint, returning the new state."""
    model.load_state_dict(base_state)
    model.to(device)
    criterion = MultilabelFocalLoss(gamma=FOCAL_GAMMA, alpha=None)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
    )
    for _ in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device, hemi_lambda)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def average_weights(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Element-wise average of state dicts (model soup)."""
    if not states:
        raise ValueError("average_weights received an empty list")
    avg = {k: v.detach().clone() for k, v in states[0].items()}
    for state in states[1:]:
        for k in avg:
            avg[k] += state[k]
    for k in avg:
        avg[k] /= len(states)
    return avg
