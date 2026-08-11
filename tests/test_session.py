"""Tests for the end-to-end Session API driving the interactive platform."""

from __future__ import annotations

import numpy as np
import pytest

from brainframe.session import Session


def test_session_from_config_path(tmp_path):
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    assert sess.volume is None
    assert sess.label_volume is None
    assert sess.reconstruction is None
    assert sess.classification is None
    assert sess.recommendation is None
    assert sess.output_dir == tmp_path


def test_session_ingest_array(synthetic_volume, tmp_path):
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.ingest(synthetic_volume)
    assert sess.volume is synthetic_volume
    assert sess.spacing == (1.0, 1.0, 1.0)
    # Ingesting clears downstream state.
    assert sess.label_volume is None


def test_session_ingest_clears_downstream(synthetic_volume, tmp_path):
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.ingest(synthetic_volume)
    sess.label_volume = np.ones_like(synthetic_volume, dtype=np.int16)
    sess.reconstruction = {"dummy": True}
    # Re-ingest should clear the stale downstream state.
    sess.ingest(synthetic_volume)
    assert sess.label_volume is None
    assert sess.reconstruction is None


def test_session_load_demo_brain(tmp_path):
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.load_demo_brain()
    assert sess.volume is not None
    assert sess.volume.ndim == 3
    assert sess.volume.size > 0


def test_session_load_demo_brain_fallback(tmp_path, monkeypatch):
    # Force the demo asset to be missing so the phantom fallback is exercised.
    from pathlib import Path

    monkeypatch.setattr(Path, "exists", lambda self: False)
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.load_demo_brain()
    assert sess.volume is not None
    assert sess.volume.ndim == 3


def test_session_segment_requires_volume(tmp_path):
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="No volume ingested"):
        sess.segment()


def test_session_full_pipeline_integration(tmp_path):
    """End-to-end session flow on the bundled demo brain (heuristic segmenter)."""
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.config.sam.auto_download = False
    sess.config.pipeline.cache = False
    sess.load_demo_brain().segment().reconstruct().predict().evaluate().recommend().simulate()
    assert sess.label_volume is not None
    assert sess.reconstruction is not None
    assert "meshes" in sess.reconstruction
    assert sess.classification is not None
    assert "prediction" in sess.classification
    assert sess.recommendation is not None
    assert sess.evaluation is not None
    assert "simulation" in sess.evaluation
    assert "compatibility" in sess.evaluation
    sim = sess.evaluation["simulation"]
    assert sim.before_lesion_volume_mm3 >= sim.after_lesion_volume_mm3
    assert sess.simulation_override is not None


def test_session_reconstruct_keeps_label_volume(tmp_path):
    """Regression: reconstruct() must not wipe label_volume (fixed bug)."""
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.config.sam.auto_download = False
    sess.config.pipeline.cache = False
    sess.load_demo_brain().segment()
    label_before = sess.label_volume
    sess.reconstruct()
    assert sess.label_volume is label_before


def test_session_idempotent_reconstruct(tmp_path):
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.config.sam.auto_download = False
    sess.config.pipeline.cache = False
    sess.load_demo_brain().segment().reconstruct()
    first = sess.reconstruction
    sess.reconstruct()
    assert sess.reconstruction is first


def test_session_summary_empty(tmp_path):
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    s = sess.summary()
    assert isinstance(s, dict)
    assert s["stages"] == []
    assert s["volume_shape"] is None
    assert s["prediction"] is None


def test_session_summary_after_steps(tmp_path):
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.config.sam.auto_download = False
    sess.config.pipeline.cache = False
    sess.load_demo_brain().segment()
    s = sess.summary()
    assert "ingest" in s["stages"]
    assert "segment" in s["stages"]
    assert s["volume_shape"] is not None


def test_session_reset(tmp_path):
    sess = Session.from_config_path("configs/default.yaml", output_dir=tmp_path)
    sess.config.sam.auto_download = False
    sess.config.pipeline.cache = False
    sess.load_demo_brain().segment()
    sess.reset()
    assert sess.volume is None
    assert sess.label_volume is None
    assert sess.reconstruction is None
