"""Shared utilities for IO, logging, seeding, device selection, and metrics."""

from __future__ import annotations

from brainframe.utils.device import get_device, resolve_device
from brainframe.utils.io import ensure_dir, load_json, save_json
from brainframe.utils.logging import get_logger
from brainframe.utils.metrics import dice_score, iou_score
from brainframe.utils.seed import set_seed

__all__ = [
    "dice_score",
    "ensure_dir",
    "get_device",
    "get_logger",
    "iou_score",
    "load_json",
    "resolve_device",
    "save_json",
    "set_seed",
]
