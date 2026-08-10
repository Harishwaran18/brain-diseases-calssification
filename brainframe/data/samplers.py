"""2D slice samplers for SAM-style inference over a 3D volume."""

from __future__ import annotations

import numpy as np


class SliceSampler:
    """Yield 2D slices (indices) of a 3D volume along a given axis.

    Parameters
    ----------
    volume : np.ndarray
        3D volume to sample slices from.
    axis : int
        Axis along which to slice (0=sagittal, 1=coronal, 2=axial).
    stride : int
        Yield every ``stride``-th slice (1 = all slices).
    mask : np.ndarray, optional
        Brain mask; slices with fewer than ``min_foreground`` foreground voxels are
        skipped.
    """

    def __init__(
        self,
        volume: np.ndarray,
        axis: int = 2,
        stride: int = 1,
        mask: np.ndarray | None = None,
        min_foreground: int = 0,
    ) -> None:
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
        self.volume = volume
        self.axis = int(axis)
        self.stride = max(1, int(stride))
        self.mask = mask
        self.min_foreground = int(min_foreground)
        self.indices = self._compute_indices()

    def _compute_indices(self) -> list[int]:
        n = self.volume.shape[self.axis]
        cand = list(range(0, n, self.stride))
        if self.mask is None or self.min_foreground <= 0:
            return cand
        kept: list[int] = []
        for i in cand:
            sl = np.take(self.mask, i, axis=self.axis)
            if int((sl > 0).sum()) >= self.min_foreground:
                kept.append(i)
        return kept

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        for i in self.indices:
            yield i, np.take(self.volume, i, axis=self.axis).astype(np.float32, copy=True)

    def get(self, index: int) -> np.ndarray:
        """Return the 2D slice at the given absolute ``index``."""
        return np.take(self.volume, index, axis=self.axis).astype(np.float32, copy=True)

    def all_slices(self) -> np.ndarray:
        """Return a view of all sampled slices stacked along a new leading axis."""
        return np.stack([self.get(i) for i in self.indices], axis=0)
