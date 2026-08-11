"""End-to-end pipeline smoke test on a synthetic volume."""

from __future__ import annotations

import numpy as np


def test_run_pipeline_full(synthetic_volume, default_cfg, tmp_path):
    from brainframe.pipeline import run_pipeline

    # Use heuristic segmenter (no SAM download) by disabling auto_download.
    default_cfg.sam.auto_download = False
    default_cfg.pipeline.cache = False  # fresh run

    res = run_pipeline(
        synthetic_volume,
        default_cfg,
        output_dir=tmp_path / "pipeline",
        device="cpu",
        stages=["segment", "reconstruct", "evaluate"],
        spacing=(1, 1, 1),
    )
    assert res.stages == ["segment", "reconstruct", "evaluate"]
    # artifacts
    import os

    assert res.label_volume_path and os.path.exists(res.label_volume_path)
    assert len(res.mesh_paths) >= 1
    for p in res.mesh_paths:
        assert os.path.exists(p), f"missing mesh {p}"
    assert res.metrics_path and os.path.exists(res.metrics_path)
    # metrics content
    assert "reconstruction" in res.metrics
    assert "per_label" in res.metrics["reconstruction"]
    # evaluation report
    assert res.report_path and os.path.exists(res.report_path)
    assert res.compatibility_score is not None
    assert 0.0 <= res.compatibility_score <= 1.0
    # The unified single-page report is the primary deliverable.
    unified = tmp_path / "pipeline" / "report.html"
    assert unified.exists()
    html = unified.read_text()
    assert "BrainFrame" in html and "3D Reconstruction" in html
    assert "Disease Prediction" in html and "Therapy Simulation" in html


def test_unified_report_contains_all_sections(tmp_path):
    from brainframe.reporting.unified_report import generate_unified_report

    html = generate_unified_report(
        mesh_result=None,
        recon_metrics={"per_label": {}, "atrophy_ratios": {}, "total_volume_mm3": 0.0},
        lesion_report={"total_lesion_volume_mm3": 0.0, "n_regions": 0, "regions": []},
        simulation={"before_lesion_volume_mm3": 0.0, "after_lesion_volume_mm3": 0.0,
                    "affected_voxels": 0, "affected_fraction": 0.0, "propagated": False},
        compatibility={"coverage": 0.0, "recovery": 0.0, "risk": 0.0, "score": 0.0, "components": {}},
        therapy={"mode": "regeneration", "target_label": "lesion", "target_mode": "centroid",
                 "radius_mm": 10.0, "dose": 1.0, "kernel": "gaussian", "sigma_mm": 5.0},
        classification=None,
        out_path=tmp_path / "report.html",
    )
    assert "<!doctype html>" in html
    # All six tab sections are present on the single page.
    for section in ["t-3d", "t-clf", "t-les", "t-ther", "t-met", "t-xs"]:
        assert f'id="{section}"' in html
    assert (tmp_path / "report.html").exists()



def test_run_pipeline_reconstruct_only(synthetic_label_volume, default_cfg, tmp_path):
    # Pretend the label volume is already segmented by pre-seeding the cache.

    from brainframe.pipeline import run_pipeline

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.save(str(cache_dir / "label_volume.npy"), synthetic_label_volume)
    default_cfg.pipeline.cache_dir = str(cache_dir)
    default_cfg.sam.auto_download = False

    res = run_pipeline(
        synthetic_label_volume.astype(np.float32),
        default_cfg,
        output_dir=tmp_path / "out",
        device="cpu",
        stages=["reconstruct"],
        spacing=(1, 1, 1),
    )
    assert res.stages == ["reconstruct"]
    assert len(res.mesh_paths) >= 1


def test_cli_prepare(tmp_path):
    from brainframe.cli import main

    rc = main(
        ["prepare", "--config", "configs/default.yaml", "--output-dir", str(tmp_path / "out")]
    )
    assert rc == 0


def test_cli_run(tmp_path, nifti_path):
    from brainframe.cli import main

    rc = main(
        [
            "run",
            "--config",
            "configs/default.yaml",
            "--input",
            nifti_path,
            "--output",
            str(tmp_path / "cli_out"),
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    import os

    assert os.path.exists(tmp_path / "cli_out" / "pipeline_result.json")
    assert os.path.exists(tmp_path / "cli_out" / "evaluation" / "evaluation_report.html")
