"""NIfTI / DICOM volume loaders.

We standardise on :mod:`nibabel` for NIfTI I/O. DICOM series are read on a best-effort
basis (sorted slices stacked along the acquisition axis). All loaders return a
:class:`LoadResult` carrying the volume as a NumPy array plus affine and spacing metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from brainframe.utils.logging import get_logger

log = get_logger("data.loaders")


@dataclass
class LoadResult:
    """A loaded medical image volume with metadata."""

    volume: np.ndarray  # (X, Y, Z) float32
    affine: np.ndarray  # (4, 4)
    spacing: tuple[float, float, float]  # voxel spacing (mm)
    origin: tuple[float, float, float]
    source: str
    header: dict[str, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.volume.shape


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    sp = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return tuple(float(s) for s in sp)  # type: ignore[return-value]


def load_nifti(path: str | Path) -> LoadResult:
    """Load a NIfTI file (``.nii`` / ``.nii.gz``) via nibabel."""
    import nibabel as nib

    img = nib.load(str(path))
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    affine = np.asarray(img.affine, dtype=np.float64).copy()
    spacing = _spacing_from_affine(affine)
    header = {"dtype": str(vol.dtype), "shape": list(vol.shape)}
    try:
        header["sform"] = img.get_sform(coded=True).coded.tolist()
    except Exception:  # pragma: no cover - version differences
        pass
    origin = tuple(float(o) for o in affine[:3, 3])
    return LoadResult(
        volume=vol,
        affine=affine,
        spacing=spacing,
        origin=origin,
        source=str(path),
        header=header,
    )


def load_dicom(path: str | Path) -> LoadResult:
    """Load a DICOM series (directory) or single file.

    Uses ``pydicom`` if available; falls back to a helpful error otherwise.
    """
    from pydicom import dcmread

    p = Path(path)
    files = []
    if p.is_dir():
        files = sorted(p.glob("*.dcm"))
        if not files:
            files = sorted(p.glob("*"))
    else:
        files = [p]

    slices: list[Any] = []
    for f in files:
        try:
            slices.append(dcmread(str(f)))
        except Exception as e:  # pragma: no cover - depends on file content
            log.debug("Skipping non-DICOM file %s: %s", f, e)

    if not slices:
        raise FileNotFoundError(f"No readable DICOM files found at {path}")

    slices.sort(
        key=lambda s: float(s.ImagePositionPatient[2]) if "ImagePositionPatient" in s else 0.0
    )
    vol = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=-1)
    spacing = (
        float(slices[0].PixelSpacing[0]) if "PixelSpacing" in slices[0] else 1.0,
        float(slices[0].PixelSpacing[1]) if "PixelSpacing" in slices[0] else 1.0,
        float(slices[0].SliceThickness) if "SliceThickness" in slices[0] else 1.0,
    )
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    return LoadResult(
        volume=vol,
        affine=affine,
        spacing=spacing,
        origin=(0.0, 0.0, 0.0),
        source=str(path),
        header={"shape": list(vol.shape)},
    )


def load_volume(path: str | Path) -> LoadResult:
    """Dispatch loader by extension/path."""
    p = str(path).lower()
    if p.endswith((".nii", ".nii.gz")):
        return load_nifti(path)
    if p.endswith(".dcm") or os.path.isdir(path):
        return load_dicom(path)
    # try NIfTI as a default
    try:
        return load_nifti(path)
    except Exception:
        pass
    raise ValueError(f"Unrecognised image format for path: {path}")


def save_volume(
    volume: np.ndarray,
    path: str | Path,
    affine: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> Path:
    """Save a volume as NIfTI (creating parent dirs)."""
    import nibabel as nib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float64)
        if spacing is not None:
            affine[0, 0] = spacing[0]
            affine[1, 1] = spacing[1]
            affine[2, 2] = spacing[2]
    img = nib.Nifti1Image(volume.astype(np.float32), np.asarray(affine, dtype=np.float64))
    nib.save(img, str(p))
    return p
