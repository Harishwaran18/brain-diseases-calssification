"""Auto-prompting strategies for SAM (grid points, intensity peaks, gradient bboxes).

No manual interaction required: prompts are derived from the image so segmentation is
genuinely zero-shot / retraining-free.
"""

from __future__ import annotations

import numpy as np


def grid_points(shape: tuple[int, int], spacing: int = 64, max_points: int = 12) -> np.ndarray:
    """Return a grid of foreground prompt points over the image."""
    h, w = shape
    ys = np.arange(spacing // 2, h, spacing)
    xs = np.arange(spacing // 2, w, spacing)
    if len(ys) == 0 or len(xs) == 0:
        ys = np.array([h // 2])
        xs = np.array([w // 2])
    grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)  # (N, 2) as (x, y)
    if len(grid) > max_points:
        idx = np.linspace(0, len(grid) - 1, max_points).astype(int)
        grid = grid[idx]
    return grid.astype(np.float32)


def intensity_peaks(image: np.ndarray, max_points: int = 12) -> np.ndarray:
    """Sample prompt points at local intensity maxima via a max filter heuristic."""
    from scipy.ndimage import maximum_filter

    img = image.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        return grid_points(img.shape, spacing=max(img.shape) // 4, max_points=max_points)
    norm = (img - lo) / (hi - lo)
    mx = maximum_filter(norm, size=max(3, min(img.shape) // 8))
    peaks = (norm == mx) & (norm > np.quantile(norm, 0.85))
    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return grid_points(img.shape, spacing=max(img.shape) // 4, max_points=max_points)
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[idx], ys[idx]
    return np.stack([xs, ys], axis=1).astype(np.float32)


def gradient_bbox(image: np.ndarray) -> np.ndarray:
    """Bounding box from the largest gradient-magnitude region (brain extent)."""
    from scipy.ndimage import sobel

    gx = sobel(image.astype(np.float32), axis=0)
    gy = sobel(image.astype(np.float32), axis=1)
    mag = np.sqrt(gx**2 + gy**2)
    thr = np.quantile(mag, 0.6)
    ys, xs = np.where(mag > thr)
    if len(xs) == 0:
        h, w = image.shape[:2]
        return np.array([0, 0, w, h], dtype=np.float32)
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], dtype=np.float32)


def intensity_bbox(image: np.ndarray) -> np.ndarray:
    """Bounding box of the brain region from intensity thresholding."""
    img = image.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        h, w = img.shape[:2]
        return np.array([0, 0, w, h], dtype=np.float32)
    norm = (img - lo) / (hi - lo)
    mask = norm > np.quantile(norm, 0.5)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = img.shape[:2]
        return np.array([0, 0, w, h], dtype=np.float32)
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], dtype=np.float32)


def build_prompts(
    image: np.ndarray,
    strategy: str = "grid",
    spacing: int = 64,
    max_points: int = 12,
    use_bbox: bool = True,
    bbox_from: str = "intensity",
) -> dict:
    """Construct a SAM prompt dict (point_coords/labels + optional box)."""
    if strategy == "grid":
        points = grid_points(image.shape, spacing=spacing, max_points=max_points)
    elif strategy == "intensity_peaks":
        points = intensity_peaks(image, max_points=max_points)
    else:
        points = grid_points(image.shape, spacing=spacing, max_points=max_points)

    labels = np.ones(len(points), dtype=np.int32)  # all foreground points

    prompt: dict = {"point_coords": points, "point_labels": labels}
    if use_bbox:
        bbox = intensity_bbox(image) if bbox_from == "intensity" else gradient_bbox(image)
        prompt["box"] = bbox
    return prompt
