"""Tests for the data core: loaders, preprocessing, samplers, datasets."""

from __future__ import annotations

import numpy as np


def test_load_nifti_roundtrip(nifti_path, synthetic_volume):
    from brainframe.data.loaders import load_volume, save_volume

    res = load_volume(nifti_path)
    assert res.volume.shape == synthetic_volume.shape
    assert res.affine.shape == (4, 4)
    assert res.spacing == (2.0, 2.0, 2.0)
    np.testing.assert_allclose(res.volume, synthetic_volume, atol=1e-4)
    # save round-trip
    import os

    out = os.path.join(os.path.dirname(nifti_path), "out.nii.gz")
    save_volume(res.volume, out, affine=res.affine)
    res2 = load_volume(out)
    np.testing.assert_allclose(res2.volume, res.volume, atol=1e-4)


def test_normalize_intensity_basic():
    from brainframe.data.preprocessing import normalize_intensity

    vol = np.random.default_rng(0).random((12, 12, 12)).astype(np.float32) * 100
    out = normalize_intensity(vol)
    assert out.shape == vol.shape
    assert out.dtype == np.float32
    # foreground roughly zero-mean
    assert abs(float(out[out > out.min()].mean())) < 2.0


def test_resample_isotropic_shape():
    from brainframe.data.preprocessing import resample_isotropic

    vol = np.ones((10, 10, 10), dtype=np.float32)
    out = resample_isotropic(vol, source_spacing=(2.0, 2.0, 2.0), target_spacing=(1.0, 1.0, 1.0))
    assert out.shape == (20, 20, 20)
    out2 = resample_isotropic(vol, (1, 1, 1), (1, 1, 1))
    assert out2.shape == vol.shape


def test_slice_volume_middle():
    from brainframe.data.preprocessing import slice_volume

    vol = np.arange(8 * 8 * 4).reshape(8, 8, 4).astype(np.float32)
    sl = slice_volume(vol, axis=2)
    assert sl.shape == (8, 8)
    assert np.array_equal(sl, vol[..., 2])


def test_slice_sampler():
    from brainframe.data.samplers import SliceSampler

    vol = np.random.default_rng(1).random((16, 16, 20)).astype(np.float32)
    s = SliceSampler(vol, axis=2, stride=4)
    assert len(s) == 5  # 0,4,8,12,16
    sl = s.get(0)
    assert sl.shape == (16, 16)
    # mask filtering
    mask = np.zeros((16, 16, 20), dtype=np.uint8)
    mask[..., 0] = 1
    s2 = SliceSampler(vol, axis=2, stride=1, mask=mask, min_foreground=1)
    assert s2.indices == [0]


def test_split_indices_deterministic():
    from brainframe.data.datasets import split_indices

    tr, va, te = split_indices(100, seed=42)
    assert len(tr) + len(va) + len(te) == 100
    assert len(tr) == 70
    assert len(va) == 15
    assert len(te) == 15
    # disjoint
    assert set(tr) & set(va) == set()
    assert set(tr) & set(te) == set()
