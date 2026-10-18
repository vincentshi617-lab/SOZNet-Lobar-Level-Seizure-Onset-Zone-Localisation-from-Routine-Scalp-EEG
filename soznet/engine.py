"""Training / evaluation loops shared by single-model and ensemble scripts."""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .losses import hemispheric_penalty


def train_one_epoch(model, loader, criterion, optimizer, device, hemi_lambda=0.1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loader.dataset.reset_fill_counter()

    for batch in tqdm(loader, desc="Training", leave=False):
        if batch is None:
            continue
        eegs, labels, _, _ = batch
        eegs = eegs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(eegs)
        loss = criterion(logits, labels)
        if hemi_lambda > 0:
            loss = loss + hemi_lambda * hemispheric_penalty(logits)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.3).float()
        total += labels.numel()
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * correct / max(total, 1)
    print(f"Epoch fill stats: {loader.dataset.get_filled_count()} samples were NaN-filled")
    return avg_loss, acc


def validate(model, loader, criterion, device, hemi_lambda=0.0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            if batch is None:
                continue
            eegs, labels, _, _ = batch
            eegs = eegs.to(device)
            labels = labels.to(device)

            logits = model(eegs)
            loss = criterion(logits, labels)
            if hemi_lambda > 0:
                loss = loss + hemi_lambda * hemispheric_penalty(logits)

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.3).float()
            total += labels.numel()
            correct += (preds == labels).sum().item()

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if batch is None:
                continue
            eegs, labels, _, _ = batch
            eegs = eegs.to(device)
            labels = labels.to(device)
            preds = (torch.sigmoid(model(eegs)) > 0.3).float()
            total += labels.numel()
            correct += (preds == labels).sum().item()

    return 100.0 * correct / max(total, 1)


def evaluate_detailed(model, loader, device):
    """Return (labels, preds, probs, patients, sessions) as numpy arrays / lists."""
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    patients_all, sessions_all = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None:
                continue
            eegs, labels, patients, sessions = batch
            eegs = eegs.to(device)
            probs = torch.sigmoid(model(eegs)).cpu()
            preds = (probs > 0.3).float()

            labels_all.append(labels.cpu())
            preds_all.append(preds)
            probs_all.append(probs)
            patients_all.extend(patients)
            sessions_all.extend(sessions)

    return (
        torch.cat(labels_all, dim=0).numpy(),
        torch.cat(preds_all, dim=0).numpy(),
        torch.cat(probs_all, dim=0).numpy(),
        patients_all,
        sessions_all,
    )


def evaluate_with_paths(model, loader, dataset, device):
    """Same as ``evaluate_detailed`` but also returns the sample file paths."""
    model.eval()
    paths_all, labels_all, probs_all, patients_all, sessions_all = [], [], [], [], []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating with paths"):
            if batch is None:
                continue
            eegs, labels, patients, sessions = batch
            eegs = eegs.to(device)
            probs = torch.sigmoid(model(eegs)).cpu()
            batch_size = eegs.size(0)

            for i in range(batch_size):
                fpath = dataset.samples[sample_idx + i][0]
                paths_all.append(fpath)
                labels_all.append(labels[i].cpu().numpy())
                probs_all.append(probs[i].numpy())
                patients_all.append(patients[i])
                sessions_all.append(sessions[i])
            sample_idx += batch_size

    return (
        paths_all,
        np.array(labels_all),
        np.array(probs_all),
        patients_all,
        sessions_all,
    )
