"""Anatomically plausible brain phantom generator.

Produces a 3D MRI-like intensity volume that actually *looks like a brain*:
two cerebral hemispheres separated by the interhemispheric fissure, a folded
gyri/sulci cortex, central ventricles, and planted hyper-intense lesions.

The intensity coding matches the heuristic segmenter (gray matter ~0.4,
white matter ~0.7, CSF ~0.15, lesion ~1.0) so segmentation recovers real anatomy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from brainframe.utils.logging import get_logger

log = get_logger("data.brain_phantom")


def _folded_boundary(
    theta: np.ndarray, phi: np.ndarray, base: float, amp: float, freqs: tuple[int, ...] = (3, 5, 7)
) -> np.ndarray:
    """Cortical-fold radius modulation: a sum of low-order sinusoids in (theta, phi)."""
    r = np.ones_like(theta) * base
    for i, m in enumerate(freqs):
        r += amp * np.sin(m * theta + 0.7 * i) * np.cos((m + 1) * phi + 1.1 * i)
    return r


def generate_brain_volume(
    shape: tuple[int, int, int] = (96, 128, 96),
    n_lesions: int = 2,
    lesion_radius: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (intensity_volume, label_volume) of an anatomically plausible brain.

    Axes: 0 = L-R (x), 1 = A-P (y), 2 = I-S (z). The brain is longer A-P.
    """
    rng = np.random.default_rng(seed)
    X, Y, Z = shape
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X].astype(np.float32)

    # Normalize coordinates to a centered [-1, 1] cube with anatomical aspect.
    cx, cy, cz = (X - 1) / 2, (Y - 1) / 2, (Z - 1) / 2
    nx = (xx - cx) / (X / 2)
    ny = (yy - cy) / (Y / 2)
    nz = (zz - cz) / (Z / 2)

    # Spherical coordinates for fold modulation (relative to each hemisphere center).
    def hemi(center_x: float):
        dx = nx - center_x
        # Brain is ellipsoidal: longer A-P, medium L-R, shorter S-I.
        r2 = (dx / 0.78) ** 2 + (ny / 1.05) ** 2 + (nz / 0.85) ** 2
        theta = np.arctan2(nz, np.sqrt(dx**2 + ny**2 + 1e-9))
        phi = np.arctan2(ny, dx + 1e-9)
        boundary = _folded_boundary(theta, phi, base=1.0, amp=0.045)
        return r2 < boundary**2, r2, boundary

    gap = 0.09  # interhemispheric fissure half-width (normalized)
    left_mask, left_r2, left_b = hemi(center_x=-gap)
    right_mask, right_r2, right_b = hemi(center_x=+gap)
    # Carve the fissure down the midline.
    fissure = np.abs(nx) < gap * 0.55
    left_mask &= ~fissure
    right_mask &= ~fissure

    vol = np.zeros(shape[::-1], dtype=np.float32)  # (Z, Y, X) -> transpose later
    labels = np.zeros(shape[::-1], dtype=np.int16)

    from brainframe.config import LABELS

    # Outer cortex shell = gray matter; inner = white matter; central ventricles = CSF.
    for mask, r2, b in [(left_mask, left_r2, left_b), (right_mask, right_r2, right_b)]:
        # White matter: deep inside.
        wm = mask & (r2 < (0.55 * b) ** 2)
        # Gray matter: outer shell.
        gm = mask & (r2 >= (0.55 * b) ** 2) & (r2 < b**2)
        vol[wm] = 0.70
        vol[gm] = 0.40
        labels[wm] = LABELS["white_matter"]
        labels[gm] = LABELS["gray_matter"]

    # Ventricles: small central CSF blobs in each hemisphere.
    for center_x in (-0.22, 0.22):
        v2 = ((nx - center_x) / 0.10) ** 2 + (ny / 0.18) ** 2 + (nz / 0.12) ** 2
        v_mask = v2 < 1.0
        vol[v_mask] = 0.15
        labels[v_mask] = LABELS["csf"]

    # Plant lesions: bright hyper-intense blobs near white-matter / cortex junction.
    for i in range(n_lesions):
        # Random location inside a hemisphere, biased to white matter.
        hemi_cx = rng.choice([-0.30, 0.30])
        lx = hemi_cx + rng.normal(0, 0.10)
        ly = rng.normal(0, 0.35)
        lz = rng.normal(0.15, 0.20)
        r = lesion_radius / max(X, Y, Z) * rng.uniform(1.6, 2.4)
        d2 = ((nx - lx) / 1.0) ** 2 + ((ny - ly) / 1.0) ** 2 + ((nz - lz) / 1.0) ** 2
        les = d2 < r**2
        vol[les] = 1.0
        labels[les] = LABELS["lesion"]
        log.info("Planted lesion %d at (%.2f, %.2f, %.2f) r=%.2f", i, lx, ly, lz, r)

    # Reshape to (X, Y, Z) axis order matching the rest of the codebase.
    vol = np.transpose(vol, (2, 1, 0)).copy()
    labels = np.transpose(labels, (2, 1, 0)).copy()
    return vol, labels


def save_brain_nifti(volume: np.ndarray, path: str | Path, spacing=(2.0, 2.0, 2.0)) -> Path:
    """Save a volume to NIfTI with the given voxel spacing (mm)."""
    import nibabel as nib

    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), str(p))
    log.info("Wrote brain phantom to %s", p)
    return p
