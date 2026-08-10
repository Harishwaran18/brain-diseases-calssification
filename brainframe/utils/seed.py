"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, and (optionally) PyTorch RNGs.

    Returns the seed actually used so callers can log it. If ``seed`` is None the
    value of ``BRAINFRAME_SEED`` is read (default 42).
    """
    if seed is None:
        seed = int(os.environ.get("BRAINFRAME_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch missing
        pass
    return seed
