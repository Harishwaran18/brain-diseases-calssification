"""Mask post-processing: largest connected component, hole filling, smoothing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
from scipy.ndimage import label as cc_label

from brainframe.config import LABELS, SegmentationPostprocessConfig
from brainframe.utils.logging import get_logger

log = get_logger("segmentation.postprocess")


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    if mask.sum() == 0:
        return mask
    lbl, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    return (lbl == keep).astype(mask.dtype)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary mask."""
    return binary_fill_holes(mask).astype(mask.dtype)


def postprocess_mask(label_map: np.ndarray, cfg: SegmentationPostprocessConfig) -> np.ndarray:
    """Apply largest-CC, hole filling, and small-region removal to each label."""
    out = np.zeros_like(label_map, dtype=np.int16)
    for name, idx in LABELS.items():
        if name == "background":
            continue
        m = label_map == idx
        if m.sum() == 0:
            continue
        if cfg.fill_holes:
            m = fill_holes(m)
        if cfg.largest_cc:
            m = largest_connected_component(m)
        if cfg.min_voxel_count and int(m.sum()) < cfg.min_voxel_count:
            continue
        out[m] = idx
    if cfg.smoothing_sigma > 0:
        # Smooth the per-label boundary by voting over a slightly blurred label field.
        smooth = np.zeros_like(out, dtype=np.float32)
        for idx in np.unique(out):
            if idx == 0:
                continue
            m = (out == idx).astype(np.float32)
            smooth += idx * gaussian_filter(m, sigma=cfg.smoothing_sigma)
        out = np.rint(smooth).astype(np.int16)
    return out


def postprocess_volume(label_volume: np.ndarray, cfg: SegmentationPostprocessConfig) -> np.ndarray:
    """3D post-processing: largest-CC and hole filling in 3D per label."""
    out = np.zeros_like(label_volume, dtype=np.int16)
    for name, idx in LABELS.items():
        if name == "background":
            continue
        m = label_volume == idx
        if m.sum() == 0:
            continue
        if cfg.fill_holes:
            m = binary_fill_holes(m)
        if cfg.largest_cc:
            lbl, n = cc_label(m)
            if n > 0:
                sizes = np.bincount(lbl.ravel())
                sizes[0] = 0
                m = (lbl == sizes.argmax()).astype(np.int16)
        if cfg.min_voxel_count and int(m.sum()) < cfg.min_voxel_count:
            continue
        out[m > 0] = idx
    return out
