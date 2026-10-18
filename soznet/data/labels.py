"""Label loading and multi-label projection utilities."""
from __future__ import annotations

import re
from typing import Dict, Tuple, List

import pandas as pd

from ..config import BLOCK_TO_INDEX, LEFT_REGIONS, RIGHT_REGIONS


def flip_label(label: List[int]) -> List[int]:
    """Swap left/right labels for hemispheric channel swap augmentation."""
    return [label[1], label[0], label[3], label[2], label[5], label[4]]


def parse_blocks_to_multilabel(blocks_str, position: str) -> List[int]:
    """Project the clinical ``Blocks`` string into a 6-D multi-label vector.

    Rules
    -----
    - ``Blocks`` may contain a comma-separated list of integers 1..6.
    - 1 -> LF, 2 -> RF, 3 -> LTP, 4 -> RTP, 5 -> LO, 6 -> RO.
    - If ``Blocks`` is empty or NaN, fall back to the hemisphere given by
      ``position`` (``L`` or ``R``).
    """
    label = [0] * 6

    def _set_hemisphere(side: str) -> None:
        if side.upper() == "L":
            for i in LEFT_REGIONS:
                label[i] = 1
        elif side.upper() == "R":
            for i in RIGHT_REGIONS:
                label[i] = 1

    blocks = ""
    if pd.notna(blocks_str):
        blocks = str(blocks_str).strip()

    if blocks:
        for token in re.findall(r"\d+", blocks):
            region = int(token)
            if region in BLOCK_TO_INDEX:
                label[BLOCK_TO_INDEX[region]] = 1
        if sum(label) == 0:
            _set_hemisphere(position)
    else:
        _set_hemisphere(position)

    return label


def load_labels(csv_path: str) -> Dict[Tuple[str, str], List[int]]:
    """Load the annotation CSV and return ``{(patient_folder, session_folder): label}``.

    The returned keys match the on-disk directory names:
    ``patient_<id>/ses-<id>``.
    """
    df = pd.read_csv(csv_path)
    required = {"patient", "session", "position", "Blocks"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {missing}")

    df = df[list(required)].dropna(subset=["patient", "session", "position"])

    label_dict: Dict[Tuple[str, str], List[int]] = {}
    for _, row in df.iterrows():
        patient_raw = str(row["patient"]).strip()
        try:
            patient_num = str(int(float(patient_raw)))
        except ValueError:
            patient_num = patient_raw
        patient_folder = f"patient_{patient_num}"

        session_raw = str(row["session"]).strip()
        if session_raw.startswith("ses-"):
            session_folder = session_raw
        else:
            session_folder = f"ses-{session_raw}"

        position = str(row["position"]).strip().upper()
        label = parse_blocks_to_multilabel(row["Blocks"], position)
        label_dict[(patient_folder, session_folder)] = label

    region_counts = [0] * 6
    for v in label_dict.values():
        for i in range(6):
            region_counts[i] += v[i]

    print(f"Loaded {len(label_dict)} (patient, session) labels.")
    print(f"Region positive counts: {region_counts}")
    return label_dict
