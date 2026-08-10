"""Stack 2D segmented slices into a 3D voxel label volume.

Spacing-aware interpolation and inter-slice gap filling make the resulting volume
isotropic so that Marching Cubes yields smooth surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, zoom

from brainframe.config import ReconstructionConfig
from brainframe.utils.logging import get_logger

log = get_logger("reconstruction.stacking")


@dataclass
class ReconstructionResult:
    """A reconstructed 3D label volume."""

    label_volume: np.ndarray  # (X, Y, Z) int16
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    labels_present: list[int]


def _label_aware_resize(volume: np.ndarray, target_shape, order: int = 0) -> np.ndarray:
    """Resize a label volume with nearest-neighbour interpolation to avoid label blending."""
    factors = [t / s for t, s in zip(target_shape, volume.shape, strict=True)]
    if all(abs(f - 1.0) < 1e-6 for f in factors):
        return volume
    return zoom(volume.astype(np.int16), factors, order=order)


def fill_gaps(label_volume: np.ndarray, method: str = "morphological") -> np.ndarray:
    """Fill inter-slice gaps by interpolating labels between consecutive slices."""
    if label_volume.shape[2] < 3:
        return label_volume
    out = label_volume.copy()
    for z in range(1, label_volume.shape[2] - 1):
        prev = label_volume[..., z - 1]
        nxt = label_volume[..., z + 1]
        cur = label_volume[..., z]
        if method == "morphological":
            # Where the current slice is empty but neighbours agree, inherit the label.
            agree = (prev == nxt) & (prev > 0) & (cur == 0)
            out[agree, z] = prev[agree]
            # Dilate each label slightly to bridge 1-voxel gaps.
            for lbl in np.unique(out[..., z]):
                if lbl == 0:
                    continue
                m = out[..., z] == lbl
                out[..., z] = np.where(binary_dilation(m), lbl, out[..., z])
        else:  # linear_interp
            from scipy.ndimage import map_coordinates

            coords = np.mgrid[0 : out.shape[0], 0 : out.shape[1]]
            stack = np.stack([prev, nxt], axis=-1).astype(np.float32)
            mid = map_coordinates(stack, [*coords, np.full_like(coords[0], 0.5)], order=1)
            cur_mask = cur == 0
            out[cur_mask, z] = np.rint(mid[cur_mask]).astype(np.int16)
    return out


def stack_slices(
    label_volume: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cfg: ReconstructionConfig | None = None,
) -> ReconstructionResult:
    """Stack an already-3D label volume (slice-stacked) into an isotropic reconstruction.

    The input is the per-slice label volume produced by segmentation. We resample to the
    target spacing, optionally fill inter-slice gaps, and return a ReconstructionResult.
    """
    if cfg is None:
        from brainframe.config import default_config

        cfg = default_config().reconstruction
    vol = label_volume.astype(np.int16)
    target_spacing = cfg.stacking.spacing
    target_shape = tuple(
        max(1, int(round(vol.shape[i] * spacing[i] / target_spacing[i]))) for i in range(3)
    )
    order = 1 if cfg.stacking.interpolation == "linear" else 0
    resampled = _label_aware_resize(vol, target_shape, order=order)
    if cfg.stacking.fill_gaps:
        resampled = fill_gaps(resampled, method=cfg.stacking.gap_method)
    present = [int(v) for v in np.unique(resampled) if v != 0]
    log.info(
        "Reconstructed volume shape %s spacing %s labels %s",
        resampled.shape,
        target_spacing,
        present,
    )
    return ReconstructionResult(
        label_volume=resampled,
        spacing=target_spacing,
        origin=(0.0, 0.0, 0.0),
        labels_present=present,
    )
