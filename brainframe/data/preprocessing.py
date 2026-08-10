"""Intensity normalization, resampling, and slice extraction."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from brainframe.utils.logging import get_logger

log = get_logger("data.preprocessing")


def normalize_intensity(
    volume: np.ndarray,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
    background: float | None = None,
) -> np.ndarray:
    """Percentile-clipped, zero-mean/unit-variance normalization on non-background voxels.

    A voxel is considered background if it is below ``background`` (default: the 1st
    percentile of intensities). Background voxels are set to the post-norm minimum.
    """
    vol = volume.astype(np.float32)
    if vol.size == 0:
        return vol
    p_lo, p_hi = np.percentile(vol, clip_percentiles)
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    clipped = np.clip(vol, p_lo, p_hi)
    if background is None:
        background = float(p_lo)
    fg = clipped[clipped > background]
    if fg.size == 0:
        return np.zeros_like(clipped, dtype=np.float32)
    mean = float(fg.mean())
    std = float(fg.std()) or 1.0
    norm = (clipped - mean) / std
    norm[clipped <= background] = float(norm.min())
    return norm.astype(np.float32)


def resample_isotropic(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    order: int = 1,
) -> np.ndarray:
    """Resample a volume to ``target_spacing`` (isotropic by default) via spline interp.

    ``order`` follows :func:`scipy.ndimage.map_coordinates` conventions (1=linear, 3=cubic).
    """
    factors = [s / t for s, t in zip(source_spacing, target_spacing, strict=True)]
    if all(abs(f - 1.0) < 1e-6 for f in factors):
        return volume.astype(np.float32, copy=True)
    new_shape = tuple(max(1, int(round(volume.shape[i] * factors[i]))) for i in range(3))
    zoom_factors = tuple(new_shape[i] / volume.shape[i] for i in range(3))
    resampled = zoom(volume.astype(np.float32), zoom_factors, order=order)
    return resampled.astype(np.float32)


def slice_volume(volume: np.ndarray, axis: int = 2, index: int | None = None) -> np.ndarray:
    """Return a 2D slice of ``volume`` along ``axis``.

    If ``index`` is None, the middle slice is returned.
    """
    if volume.ndim == 2:
        return volume.astype(np.float32, copy=True)
    n = volume.shape[axis]
    if index is None:
        index = n // 2
    index = int(np.clip(index, 0, n - 1))
    sl = [slice(None)] * volume.ndim
    sl[axis] = index
    return np.ascontiguousarray(volume[tuple(sl)].astype(np.float32))


def smooth_volume(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing for isotropic volumes."""
    return gaussian_filter(volume.astype(np.float32), sigma=sigma)


def estimate_spacing(affine: np.ndarray) -> tuple[float, float, float]:
    """Return voxel spacing from a 4x4 affine."""
    sp = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return tuple(float(s) for s in sp)  # type: ignore[return-value]


def brain_mask_from_intensity(volume: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """A simple intensity threshold brain mask (non-zero above the given level)."""
    vol = volume.astype(np.float32)
    if threshold is None:
        threshold = (
            float(np.percentile(vol[vol > vol.min()], 5)) if (vol > vol.min()).any() else 0.0
        )
    return (vol > threshold).astype(np.uint8)
