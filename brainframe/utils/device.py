"""Device selection utilities.

Torch is an optional dependency. These helpers return a lightweight string device name
when torch is unavailable; when torch is present, a proper :class:`torch.device` is
returned. Callers that actually need torch should pass the result to
``torch.device(...)``.
"""

from __future__ import annotations

import os


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def get_device():
    """Return the current default torch device, or the string ``"cpu"`` if no torch."""
    try:
        import torch

        return (
            torch.get_default_device()
            if hasattr(torch, "get_default_device")
            else torch.device("cpu")
        )
    except ImportError:
        return "cpu"


def resolve_device(device: str = "auto"):
    """Resolve a device string.

    ``auto`` selects CUDA if available, else MPS (Apple), else CPU. The environment
    variable ``BRAINFRAME_DEVICE`` overrides the ``device`` argument when set. Returns a
    :class:`torch.device` when torch is installed, otherwise a lower-case string.
    """
    env = os.environ.get("BRAINFRAME_DEVICE", "").strip().lower()
    if env:
        device = env
    device = device.lower()
    try:
        import torch
    except ImportError:
        return device if device != "auto" else "cpu"
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
