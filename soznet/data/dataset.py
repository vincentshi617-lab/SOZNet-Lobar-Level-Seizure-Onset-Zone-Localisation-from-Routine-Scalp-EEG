"""PyTorch dataset for pre-epoched EEG tensors.

Each ``.pt`` file is expected to be a tensor shaped ``(1, 18, 4096)`` or
``(18, 4096)``. The dataset crops each sample to ``chunk_size`` (default 2048
= 8 s at 256 Hz) and optionally applies an online augmentation described by
``aug_params``.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import CHUNK_SIZE, NUM_ELECTRODES
from .labels import flip_label
from .augment import FLIP_CHANNEL_MAP, add_gaussian_noise, random_scale


def _fill_nan_linear(eeg: torch.Tensor) -> torch.Tensor:
    """Replace NaNs along the time axis by per-channel linear interpolation.

    ``eeg`` is expected to be ``(C, T)`` or ``(1, C, T)``. Channel entirely
    NaN is filled with 0. Boundary NaNs use nearest-neighbour extrapolation.
    """
    orig_shape = eeg.shape
    x = eeg.reshape(-1, orig_shape[-1]).clone()
    arr = x.detach().cpu().numpy()
    mask = np.isnan(arr)
    if not mask.any():
        return eeg

    # all-NaN channels -> 0
    all_nan = mask.all(axis=1)
    if all_nan.any():
        arr[all_nan] = 0.0
        mask = np.isnan(arr)

    t = np.arange(arr.shape[1])
    for c in range(arr.shape[0]):
        m = mask[c]
        if not m.any():
            continue
        arr[c, m] = np.interp(t[m], t[~m], arr[c, ~m])

    filled = torch.from_numpy(arr).to(dtype=eeg.dtype, device=eeg.device)
    return filled.reshape(orig_shape)


def _to_canonical(eeg: torch.Tensor) -> torch.Tensor:
    """Return a ``(1, C, T)`` tensor from common 2D/3D shapes."""
    if eeg.dim() == 2:
        c, t = eeg.shape
        if c == NUM_ELECTRODES:
            return eeg.unsqueeze(0)
        if t == NUM_ELECTRODES:
            return eeg.t().unsqueeze(0)
        raise ValueError(f"Invalid 2D EEG shape: {eeg.shape}")
    if eeg.dim() == 3:
        if eeg.shape[0] != 1:
            raise ValueError(f"Expected batch dim 1, got {eeg.shape}")
        if eeg.shape[1] == NUM_ELECTRODES:
            return eeg
        if eeg.shape[2] == NUM_ELECTRODES:
            return eeg.transpose(1, 2)
        raise ValueError(f"Invalid 3D EEG shape: {eeg.shape}")
    raise ValueError(f"Invalid EEG tensor dimension: {eeg.dim()}")


class EEGDataset(Dataset):
    """Dataset over pre-epoched ``.pt`` files.

    Parameters
    ----------
    data_root : str
        Root directory containing ``patient_*/ses-*/*.pt``.
    label_dict : dict
        ``{(patient_folder, session_folder): [6 labels]}``.
    samples_per_session : int or None
        Maximum number of windows sampled per session.
    allowed_keys : iterable or None
        Restrict to these ``(patient, session)`` keys.
    samples_list : list or None
        Pre-built sample list, used by augmentations. If ``None``, the
        dataset walks ``data_root`` and builds the list itself.
    chunk_size : int
        Output temporal length (default 2048 = 8 s at 256 Hz).
    """

    def __init__(
        self,
        data_root: str,
        label_dict: Dict[Tuple[str, str], List[int]],
        samples_per_session: Optional[int] = None,
        allowed_keys: Optional[set] = None,
        samples_list: Optional[list] = None,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        self.data_root = data_root
        self.label_dict = label_dict
        self.samples_per_session = samples_per_session
        self.allowed_keys = set(allowed_keys) if allowed_keys else None
        self.chunk_size = chunk_size

        if samples_list is not None:
            self.samples = samples_list
        else:
            self.samples = []
            self._collect_samples()

        self.filled_count = 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _collect_samples(self) -> None:
        for patient in os.listdir(self.data_root):
            patient_path = os.path.join(self.data_root, patient)
            if not os.path.isdir(patient_path):
                continue
            for session in os.listdir(patient_path):
                session_path = os.path.join(patient_path, session)
                if not os.path.isdir(session_path):
                    continue
                key = (patient, session)
                if self.allowed_keys and key not in self.allowed_keys:
                    continue
                if key not in self.label_dict:
                    continue
                label = self.label_dict[key]
                files = [f for f in os.listdir(session_path) if f.endswith(".pt")]
                if not files:
                    continue
                if self.samples_per_session and len(files) > self.samples_per_session:
                    files = random.sample(files, self.samples_per_session)
                for fname in files:
                    fpath = os.path.join(session_path, fname)
                    is_seizure = 1 if (
                        "seizure" in fname.lower() or "sz" in fname.lower()
                    ) else 0
                    self.samples.append(
                        (fpath, label, patient, session, is_seizure, None)
                    )
        print(f"Total samples collected: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label_list, patient, session, is_seizure, aug_params = self.samples[idx]

        try:
            eeg = torch.load(fpath, weights_only=False).float()
            eeg = _fill_nan_linear(eeg)
            eeg = _to_canonical(eeg)

            total_len = eeg.shape[-1]
            center_start = (total_len - self.chunk_size) // 2

            if aug_params is None:
                eeg = eeg[:, :, center_start:center_start + self.chunk_size]
            else:
                aug_type = aug_params["type"]
                if aug_type == "noise":
                    eeg = add_gaussian_noise(eeg, aug_params["std"])
                    eeg = eeg[:, :, center_start:center_start + self.chunk_size]
                elif aug_type == "scale":
                    eeg = random_scale(eeg, aug_params["scale_range"])
                    eeg = eeg[:, :, center_start:center_start + self.chunk_size]
                elif aug_type == "crop":
                    start = aug_params["start"]
                    eeg = eeg[:, :, start:start + self.chunk_size]
                elif aug_type == "flip":
                    new_eeg = eeg.clone()
                    for src, dst in FLIP_CHANNEL_MAP.items():
                        new_eeg[:, dst, :] = eeg[:, src, :]
                    eeg = new_eeg
                    label_list = flip_label(label_list)
                    eeg = eeg[:, :, center_start:center_start + self.chunk_size]
                else:
                    eeg = eeg[:, :, center_start:center_start + self.chunk_size]

            label_tensor = torch.tensor(label_list, dtype=torch.float32)
            return eeg, label_tensor, patient, session

        except Exception as exc:  # noqa: BLE001
            print(f"Error loading {fpath}: {exc}")
            dummy_eeg = torch.zeros(1, NUM_ELECTRODES, self.chunk_size)
            dummy_label = torch.full((6,), -1.0)
            return dummy_eeg, dummy_label, patient, session

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def reset_fill_counter(self) -> None:
        self.filled_count = 0

    def get_filled_count(self) -> int:
        return self.filled_count


def collate_fn(batch):
    """Filter failed samples (label == -1) and stack the rest."""
    batch = [b for b in batch if b[1].sum() > -1]
    if not batch:
        return None
    eegs = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    patients = [b[2] for b in batch]
    sessions = [b[3] for b in batch]
    return eegs, labels, patients, sessions
