"""Tests for the 3D reconstruction module."""

from __future__ import annotations

import math

import pytest


def test_stack_slices_isotropic(synthetic_label_volume, default_cfg):
    from brainframe.reconstruction.stacking import stack_slices

    res = stack_slices(synthetic_label_volume, spacing=(1, 1, 1), cfg=default_cfg.reconstruction)
    assert res.label_volume.ndim == 3
    assert res.spacing == (1.0, 1.0, 1.0)
    assert len(res.labels_present) > 0


def test_extract_meshes_sphere(sphere_label_volume, default_cfg):
    from brainframe.reconstruction.marching import extract_meshes

    mr = extract_meshes(sphere_label_volume, cfg=default_cfg.reconstruction, spacing=(1, 1, 1))
    labels = set(mr.labels)
    assert "white_matter" in labels
    wm = next(m for m in mr.meshes if m.label == "white_matter")
    assert len(wm.vertices) > 0
    assert len(wm.faces) > 0


def test_mesh_metrics_sphere_analytic(solid_sphere_label_volume, default_cfg):
    from brainframe.reconstruction.marching import extract_meshes
    from brainframe.reconstruction.mesh_metrics import compute_metrics

    mr = extract_meshes(
        solid_sphere_label_volume, cfg=default_cfg.reconstruction, spacing=(1, 1, 1)
    )
    mm = compute_metrics(mr, solid_sphere_label_volume, (1, 1, 1), default_cfg.reconstruction)
    # white_matter solid sphere radius ~ 10 -> volume ~ 4/3 pi r^3
    wm = mm.per_label["white_matter"]
    analytic = 4.0 / 3.0 * math.pi * 10**3
    # Mesh volume from divergence theorem for a closed sphere should be close.
    assert wm.volume_mm3 == pytest.approx(analytic, rel=0.05)
    # compactness of a sphere ~ 1
    assert 0.9 <= wm.compactness <= 1.0


def test_atrophy_ratios(sphere_label_volume, default_cfg):
    from brainframe.reconstruction.marching import extract_meshes
    from brainframe.reconstruction.mesh_metrics import compute_metrics

    mr = extract_meshes(sphere_label_volume, cfg=default_cfg.reconstruction, spacing=(1, 1, 1))
    mm = compute_metrics(mr, sphere_label_volume, (1, 1, 1), default_cfg.reconstruction)
    assert "gray_matter" in mm.atrophy_ratios
    # gray_matter is the inner core; ratio relative to white_matter reference
    assert mm.atrophy_ratios["gray_matter"] > 0


def test_save_meshes(tmp_path, sphere_label_volume, default_cfg):
    from brainframe.reconstruction.marching import extract_meshes, save_meshes

    mr = extract_meshes(sphere_label_volume, cfg=default_cfg.reconstruction, spacing=(1, 1, 1))
    paths = save_meshes(mr, tmp_path / "meshes")
    assert len(paths) >= 1
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_save_cross_sections(tmp_path, synthetic_label_volume):
    from brainframe.reconstruction.visualize import save_cross_sections

    paths = save_cross_sections(synthetic_label_volume, tmp_path / "xs")
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
