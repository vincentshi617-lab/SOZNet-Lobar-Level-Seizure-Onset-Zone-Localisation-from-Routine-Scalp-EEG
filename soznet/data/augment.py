"""Online augmentation utilities."""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

# 18-channel bipolar montage index map used for hemispheric swap.
# 0-3: left temporal chain, 4-7: right temporal chain,
# 8-11: left parasagittal, 12-15: right parasagittal,
# 16-17: midline.
FLIP_CHANNEL_MAP = {
    0: 4, 1: 5, 2: 6, 3: 7,
    4: 0, 5: 1, 6: 2, 7: 3,
    8: 12, 9: 13, 10: 14, 11: 15,
    12: 8, 13: 9, 14: 10, 15: 11,
    16: 16, 17: 17,
}


def add_gaussian_noise(eeg: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    return eeg + torch.randn_like(eeg) * std


def random_scale(eeg: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
    scale = torch.empty(1).uniform_(*scale_range).item()
    return eeg * scale


def build_augmented_samples(
    base_samples: list,
    seizure_target: int = 1000,
    num_augmented: int = 40000,
    seed: int = 42,
    use_swap: bool = False,
    noise_std: float = 0.1,
    scale_range: Tuple[float, float] = (0.7, 1.3),
    raw_window_size: int = 4096,
    chunk_size: int = 2048,
) -> list:
    """Create augmented samples for training.

    Each returned sample is a 6-tuple
    ``(fpath, label, patient, session, is_seizure, aug_params)``.
    ``aug_params`` is ``None`` for the original sample and a dict for
    augmented variants.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    seizure_samples = [s for s in base_samples if s[4] == 1]
    n_seizure = len(seizure_samples)
    n_duplicate = max(0, seizure_target - n_seizure)

    duplicated_seizure = []
    if n_duplicate > 0 and n_seizure > 0:
        for sample in rng.choices(seizure_samples, k=n_duplicate):
            duplicated_seizure.append(sample[:5] + (None,))

    augmented = []
    if num_augmented > 0:
        total = len(base_samples)
        if num_augmented <= total:
            indices = rng.sample(range(total), num_augmented)
        else:
            indices = rng.choices(range(total), k=num_augmented)
            print(
                f"Warning: num_augmented ({num_augmented}) > total samples "
                f"({total}), using replacement."
            )

        aug_types = ["noise", "scale", "crop"]
        if use_swap:
            aug_types.append("flip")

        for idx in indices:
            sample = base_samples[idx]
            aug_type = rng.choice(aug_types)
            if aug_type == "noise":
                aug_params = {"type": "noise", "std": noise_std}
            elif aug_type == "scale":
                aug_params = {"type": "scale", "scale_range": scale_range}
            elif aug_type == "crop":
                max_start = raw_window_size - chunk_size
                start = rng.randint(0, max_start)
                aug_params = {"type": "crop", "start": start}
            else:
                aug_params = {"type": "flip"}
            augmented.append(sample[:5] + (aug_params,))

    original = [s[:5] + (None,) for s in base_samples]
    all_samples = original + duplicated_seizure + augmented
    rng.shuffle(all_samples)

    print(f"Original samples: {len(original)} (seizure: {n_seizure})")
    print(f"Duplicated seizure samples: {len(duplicated_seizure)}")
    print(f"Augmented samples: {len(augmented)}")
    print(f"Total training samples: {len(all_samples)}")
    return all_samples
