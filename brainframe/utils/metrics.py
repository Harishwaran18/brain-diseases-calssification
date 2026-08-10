"""Evaluation metrics shared across stages."""

from __future__ import annotations

import numpy as np


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection-over-Union for binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(inter / union) if union > 0 else 1.0


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice coefficient for binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    s = pred.sum() + target.sum()
    return float(2.0 * inter / s) if s > 0 else 1.0


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Classification accuracy for 1-D label arrays."""
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    if target.size == 0:
        return 0.0
    return float((pred == target).mean())
