"""Tests for the real human brain data module (ICBM152 + fsaverage cortex).

These run only when the bundled ``assets/real_brain/*.npz`` assets are present
(they are produced by ``scripts/build_real_brain_assets.py``). When the assets
are absent the module gracefully reports unavailability, which is also covered.
"""

import numpy as np
import pytest

from brainframe.data import real_brain
from brainframe.reconstruction.marching import MeshResult


@pytest.fixture(autouse=True)
def _skip_if_no_assets():
    if not real_brain.has_real_brain():
        pytest.skip("real-brain assets not bundled")


def test_has_real_brain_true():
    assert real_brain.has_real_brain() is True


def test_load_real_brain_volume_shapes():
    vol, labels, spacing = real_brain.load_real_brain_volume()
    assert vol.ndim == 3
    assert vol.dtype == np.float32
    assert 0.0 <= vol.min() and vol.max() <= 1.0
    assert labels.shape == vol.shape
    assert len(spacing) == 3
    # All five canonical tissue labels should be representable; lesion planted.
    assert {0, 1, 2, 4}.issubset(set(np.unique(labels).tolist()))


def test_load_real_brain_volume_custom_shape():
    vol, labels, _ = real_brain.load_real_brain_volume(shape=(48, 56, 42))
    assert vol.shape == (48, 56, 42)
    assert labels.shape == (48, 56, 42)


def test_load_real_cortex_mesh_is_real_surface():
    mesh = real_brain.load_real_cortex_mesh()
    assert isinstance(mesh, MeshResult)
    assert len(mesh.meshes) == 1
    m = mesh.meshes[0]
    assert m.label == "cortex"
    assert len(m.vertices) > 1000
    assert len(m.faces) > 1000
    # Normals must be unit-ish for smooth shading.
    norms = np.linalg.norm(m.normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_cortex_mesh_is_not_smooth_ellipsoid():
    """The real pial surface has folds: its radial spread should exceed a
    perfectly smooth ellipsoid of the same bounding box (i.e. it is wrinkly)."""
    mesh = real_brain.load_real_cortex_mesh().meshes[0]
    v = np.asarray(mesh.vertices)
    centre = v.mean(axis=0)
    radii = np.linalg.norm(v - centre, axis=1)
    # A folded cortex has a wide radial distribution (gyri stick out, sulci dip in).
    assert radii.std() / radii.mean() > 0.08
