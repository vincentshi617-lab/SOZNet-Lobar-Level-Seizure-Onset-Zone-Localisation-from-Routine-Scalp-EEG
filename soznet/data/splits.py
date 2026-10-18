"""Patient-level dataset splitting.

All sessions from a given patient are assigned to exactly one split, which
avoids session-level leakage. Stratification is by hemisphere (L/R), using
the patient's majority side if multiple sessions disagree.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import GroupKFold, train_test_split

from ..config import LEFT_REGIONS


def _patient_side(label: List[int]) -> int:
    """1 if any left region is positive, else 0."""
    return 1 if any(label[i] for i in LEFT_REGIONS) else 0


def _group_by_patient(label_dict: Dict[Tuple[str, str], List[int]]):
    patient_to_sessions: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    patient_side_votes: Dict[str, List[int]] = defaultdict(list)

    for key, label in label_dict.items():
        patient, _ = key
        patient_to_sessions[patient].append(key)
        patient_side_votes[patient].append(_patient_side(label))

    patient_side = {
        p: int(sum(votes) >= len(votes) / 2)
        for p, votes in patient_side_votes.items()
    }
    return patient_to_sessions, patient_side


def split_patients(
    label_dict: Dict[Tuple[str, str], List[int]],
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Patient-level 8:1:1 split stratified by hemisphere.

    Returns
    -------
    (train_keys, val_keys, test_keys)
        Each is a list of ``(patient, session)`` keys.
    """
    patient_to_sessions, patient_side = _group_by_patient(label_dict)
    patients = sorted(patient_to_sessions.keys())
    sides = [patient_side[p] for p in patients]

    train_val_patients, test_patients, train_val_sides, _ = train_test_split(
        patients,
        sides,
        test_size=test_ratio,
        stratify=sides,
        random_state=seed,
    )
    val_ratio_adj = val_ratio / (1.0 - test_ratio)
    train_patients, val_patients, _, _ = train_test_split(
        train_val_patients,
        train_val_sides,
        test_size=val_ratio_adj,
        stratify=train_val_sides,
        random_state=seed,
    )

    def _expand(subset: List[str]) -> List[Tuple[str, str]]:
        return [s for p in subset for s in patient_to_sessions[p]]

    train_keys = _expand(train_patients)
    val_keys = _expand(val_patients)
    test_keys = _expand(test_patients)

    def _count_side(keys) -> Tuple[int, int]:
        left = sum(1 for k in keys if patient_side[k[0]] == 1)
        return left, len(keys) - left

    for name, subset, keys in [
        ("Train", train_patients, train_keys),
        ("Val", val_patients, val_keys),
        ("Test", test_patients, test_keys),
    ]:
        left, right = _count_side(keys)
        print(
            f"{name}: {len(subset)} patients, {len(keys)} sessions "
            f"(L sessions: {left}, R sessions: {right})"
        )

    return train_keys, val_keys, test_keys


def split_patients_kfold(
    label_dict: Dict[Tuple[str, str], List[int]],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]]:
    """Patient-level K-fold split (no stratification).

    Note: ``GroupKFold`` is deterministic, so ``seed`` only affects the
    ordering of patient groups. Provided for cross-validation experiments.
    """
    patient_to_sessions, _ = _group_by_patient(label_dict)
    patients = sorted(patient_to_sessions.keys())

    # Light shuffle of patient order, then group split
    import random
    rng = random.Random(seed)
    rng.shuffle(patients)

    groups = [p for p in patients]
    # Build a fake y array so GroupKFold can iterate over groups
    fake_y = [0] * len(groups)
    gkf = GroupKFold(n_splits=n_splits)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(groups, fake_y, groups)
    ):
        train_patients = [groups[i] for i in train_idx]
        val_patients = [groups[i] for i in val_idx]
        train_keys = [s for p in train_patients for s in patient_to_sessions[p]]
        val_keys = [s for p in val_patients for s in patient_to_sessions[p]]
        print(
            f"Fold {fold_idx + 1}: {len(train_patients)} train patients "
            f"/ {len(val_patients)} val patients"
        )
        folds.append((train_keys, val_keys))
    return folds
