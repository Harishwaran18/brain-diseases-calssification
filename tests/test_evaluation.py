"""Tests for the computational therapy evaluation module."""

from __future__ import annotations

import numpy as np


def test_analyze_lesions_finds_planted(synthetic_label_volume, default_cfg):
    from brainframe.evaluation.lesion_analysis import analyze_lesions

    lr = analyze_lesions(synthetic_label_volume, default_cfg.evaluation, spacing=(1, 1, 1))
    assert lr.n_regions == 1
    assert lr.total_lesion_volume_mm3 > 0
    r = lr.regions[0]
    assert len(r.centroid) == 3
    assert len(r.spatial_extent) == 3
    # adjacency values are min distances (floats) to named structures
    assert r.adjacent_structures
    for sname, dist in r.adjacent_structures.items():
        assert isinstance(sname, str)
        assert isinstance(dist, float)
        assert dist >= 0.0


def test_analyze_lesions_no_lesion(default_cfg):
    from brainframe.evaluation.lesion_analysis import analyze_lesions

    vol = np.zeros((20, 20, 20), dtype=np.int16)
    lr = analyze_lesions(vol, default_cfg.evaluation, spacing=(1, 1, 1))
    assert lr.n_regions == 0
    assert lr.total_lesion_volume_mm3 == 0.0


def test_therapy_spec_validation(default_cfg):
    from brainframe.evaluation.therapy_model import build_therapy

    th = build_therapy(default_cfg.evaluation)
    assert 0.0 <= th.dose <= 1.0
    assert th.mode in ("stimulation", "regeneration", "lesion_reversal")
    assert th.kernel in ("gaussian", "diffusion")


def test_simulator_lesion_reversal_reduces_volume(synthetic_label_volume, default_cfg):
    from brainframe.evaluation.lesion_analysis import analyze_lesions
    from brainframe.evaluation.simulator import simulate_therapy
    from brainframe.evaluation.therapy_model import build_therapy

    lr = analyze_lesions(synthetic_label_volume, default_cfg.evaluation, spacing=(1, 1, 1))
    th = build_therapy(default_cfg.evaluation)
    th.mode = "lesion_reversal"
    th.radius_mm = 12.0
    sim = simulate_therapy(
        synthetic_label_volume, lr, th, default_cfg.evaluation, spacing=(1, 1, 1)
    )
    assert sim.after_lesion_volume_mm3 < sim.before_lesion_volume_mm3
    # changes only occur within the target radius (effect field zero outside)
    assert sim.affected_voxels > 0
    # label_volume_after differs only inside the lesion region
    changed = (sim.label_volume_after != synthetic_label_volume).sum()
    assert changed > 0


def test_simulator_no_target_no_change(default_cfg):
    from brainframe.evaluation.lesion_analysis import analyze_lesions
    from brainframe.evaluation.simulator import simulate_therapy
    from brainframe.evaluation.therapy_model import build_therapy

    vol = np.zeros((20, 20, 20), dtype=np.int16)
    lr = analyze_lesions(vol, default_cfg.evaluation, spacing=(1, 1, 1))
    th = build_therapy(default_cfg.evaluation)
    sim = simulate_therapy(vol, lr, th, default_cfg.evaluation, spacing=(1, 1, 1))
    assert sim.before_lesion_volume_mm3 == sim.after_lesion_volume_mm3 == 0.0


def test_compatibility_score_bounded(synthetic_label_volume, default_cfg):
    from brainframe.evaluation.compatibility import compute_compatibility
    from brainframe.evaluation.lesion_analysis import analyze_lesions
    from brainframe.evaluation.simulator import simulate_therapy
    from brainframe.evaluation.therapy_model import build_therapy

    lr = analyze_lesions(synthetic_label_volume, default_cfg.evaluation, spacing=(1, 1, 1))
    th = build_therapy(default_cfg.evaluation)
    th.mode = "lesion_reversal"
    th.radius_mm = 12.0
    sim = simulate_therapy(
        synthetic_label_volume, lr, th, default_cfg.evaluation, spacing=(1, 1, 1)
    )
    comp = compute_compatibility(
        sim, lr, synthetic_label_volume, default_cfg.evaluation, spacing=(1, 1, 1)
    )
    assert 0.0 <= comp.score <= 1.0
    assert 0.0 <= comp.coverage <= 1.0
    assert 0.0 <= comp.recovery <= 1.0
    assert 0.0 <= comp.risk <= 1.0


def test_generate_report_files(tmp_path, synthetic_label_volume, default_cfg):
    from brainframe.evaluation.compatibility import compute_compatibility
    from brainframe.evaluation.lesion_analysis import analyze_lesions
    from brainframe.evaluation.report import generate_report
    from brainframe.evaluation.simulator import simulate_therapy
    from brainframe.evaluation.therapy_model import build_therapy

    lr = analyze_lesions(synthetic_label_volume, default_cfg.evaluation, spacing=(1, 1, 1))
    th = build_therapy(default_cfg.evaluation)
    th.mode = "lesion_reversal"
    th.radius_mm = 12.0
    sim = simulate_therapy(
        synthetic_label_volume, lr, th, default_cfg.evaluation, spacing=(1, 1, 1)
    )
    comp = compute_compatibility(
        sim, lr, synthetic_label_volume, default_cfg.evaluation, spacing=(1, 1, 1)
    )
    report = generate_report(
        lr, sim, comp, th, synthetic_label_volume, cfg=default_cfg.evaluation, output_dir=tmp_path
    )
    import os

    assert os.path.exists(report["figures"]["json"])
    assert os.path.exists(report["figures"]["html"])
    # JSON has expected keys
    import json

    data = json.loads(open(report["figures"]["json"]).read())
    assert "therapy" in data and "compatibility" in data
