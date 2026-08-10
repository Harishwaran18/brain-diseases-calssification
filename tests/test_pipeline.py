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
