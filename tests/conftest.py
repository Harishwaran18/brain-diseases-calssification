"""Shared pytest fixtures: synthetic NIfTI volumes with planted lesions.

All fixtures are CPU-only and need no network or real data.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def label_indices() -> dict:
    from brainframe.config import LABELS

    return dict(LABELS)


@pytest.fixture
def synthetic_volume() -> np.ndarray:
    """A 3D MRI-like volume with concentric tissue regions and a planted lesion."""
    n = 40
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    vol = np.zeros((n, n, n), dtype=np.float32)
    vol[(xx - 20) ** 2 + (yy - 20) ** 2 + (zz - 20) ** 2 < 28**2] = 0.4  # gray matter
    vol[(xx - 20) ** 2 + (yy - 20) ** 2 + (zz - 20) ** 2 < 20**2] = 0.7  # white matter
    vol[(xx - 20) ** 2 + (yy - 20) ** 2 + (zz - 20) ** 2 < 8**2] = 0.15  # CSF
    vol[(xx - 14) ** 2 + (yy - 14) ** 2 + (zz - 20) ** 2 < 4**2] = 1.0  # lesion (bright)
    return vol


@pytest.fixture
def synthetic_label_volume(label_indices) -> np.ndarray:
    """A 3D label volume with planted lesion (for reconstruction/evaluation tests)."""
    LABELS = label_indices
    n = 40
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    vol = np.zeros((n, n, n), dtype=np.int16)
    vol[(xx - 20) ** 2 + (yy - 20) ** 2 + (zz - 20) ** 2 < 28**2] = LABELS["gray_matter"]
    vol[(xx - 20) ** 2 + (yy - 20) ** 2 + (zz - 20) ** 2 < 22**2] = LABELS["white_matter"]
    vol[(xx - 20) ** 2 + (yy - 20) ** 2 + (zz - 20) ** 2 < 8**2] = LABELS["csf"]
    vol[(xx - 14) ** 2 + (yy - 14) ** 2 + (zz - 20) ** 2 < 4**2] = LABELS["lesion"]
    return vol


@pytest.fixture
def nifti_path(tmp_path, synthetic_volume):
    """Write the synthetic volume to a NIfTI file and return the path."""
    import nibabel as nib

    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = 2.0
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0
    p = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(synthetic_volume, affine), str(p))
    return str(p)


@pytest.fixture
def sphere_label_volume(label_indices) -> np.ndarray:
    """A clean sphere of white_matter with a gray_matter core (analytic checks)."""
    LABELS = label_indices
    n = 40
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    vol = np.zeros((n, n, n), dtype=np.int16)
    r2 = (xx - 20) ** 2 + (yy - 20) ** 2 + (zz - 20) ** 2
    vol[r2 < 10**2] = LABELS["white_matter"]
    vol[r2 < 6**2] = LABELS["gray_matter"]
    return vol


@pytest.fixture
def solid_sphere_label_volume(label_indices) -> np.ndarray:
    """A solid white_matter sphere (no carved core) for analytic volume checks."""
    LABELS = label_indices
    n = 40
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    vol = np.zeros((n, n, n), dtype=np.int16)
    r2 = (xx - 20) ** 2 + (yy - 20) ** 2 + (zz - 20) ** 2
    vol[r2 < 10**2] = LABELS["white_matter"]
    return vol


@pytest.fixture
def default_cfg():
    from brainframe.config import load_config

    return load_config("configs/default.yaml")
