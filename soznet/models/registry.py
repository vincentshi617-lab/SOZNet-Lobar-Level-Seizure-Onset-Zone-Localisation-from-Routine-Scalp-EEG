"""Factory for M1 / M2 / M3 so callers do not import model modules directly."""
from __future__ import annotations

from typing import Callable, Dict

from .soznet_m1 import create_m1
from .m2_cross_attention import create_m2
from .m3_deep_transformer import create_m3

MODEL_NAMES = ("m1", "m2", "m3")

_BUILDERS: Dict[str, Callable] = {
    "m1": create_m1,
    "m2": create_m2,
    "m3": create_m3,
}


def create_model(name: str, **kwargs):
    """Create a model by name. Accepts ``"m1"``, ``"m2"``, ``"m3"`` (case-insensitive)."""
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")
    return _BUILDERS[key](**kwargs)
