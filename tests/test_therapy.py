"""Tests for the curated therapy library, recommender, and cure animation."""

from __future__ import annotations

import numpy as np

from brainframe.therapy.library import TECHNIQUE_LIBRARY, default_technique, techniques_for_class
from brainframe.therapy.recommender import TherapyRecommendation, recommend_therapy


def test_library_covers_every_disease_class():
    classes = {t.disease_class for t in TECHNIQUE_LIBRARY}
    assert classes == {0, 1, 2, 3}


def test_library_techniques_have_required_fields():
    for t in TECHNIQUE_LIBRARY:
        assert t.id and t.name
        assert t.mode in {"stimulation", "regeneration", "lesion_reversal"}
        assert t.target_mode in {"centroid", "largest_region", "manual"}
        assert 0.0 < t.dose <= 1.0
        assert t.radius_mm > 0.0
        assert t.sigma_mm > 0.0
        assert t.rationale and t.expected_effect


def test_default_technique_returns_class_match():
    for cls in (0, 1, 2, 3):
        tech = default_technique(cls)
        assert tech.disease_class == cls


def test_techniques_for_class_returns_only_matching():
    for cls in (0, 1, 2, 3):
        techs = techniques_for_class(cls)
        assert techs
        assert all(t.disease_class == cls for t in techs)


def test_to_therapy_spec_dict_keys_match_simulator():
    tech = TECHNIQUE_LIBRARY[0]
    spec = tech.to_therapy_spec_dict()
    assert set(spec) == {
        "target_label",
        "target_mode",
        "radius_mm",
        "dose",
        "mode",
        "kernel",
        "sigma_mm",
    }


def test_recommend_returns_recommendation_for_each_class():
    for cls in (0, 1, 2, 3):
        rec = recommend_therapy(cls, lesion_volume_mm3=0.0, n_regions=0, confidence=0.5)
        assert isinstance(rec, TherapyRecommendation)
        assert rec.disease_class == cls
        assert rec.technique.disease_class in {cls, min(3, cls + 1)}


def test_recommend_upscales_when_lesion_large():
    # A class-1 prediction with a large lesion should pick a heavier technique.
    rec = recommend_therapy(1, lesion_volume_mm3=8000.0, n_regions=2, confidence=0.4)
    assert rec.technique.disease_class >= 2


def test_recommend_does_not_upscale_at_top_class():
    # Class 3 is already the heaviest; large lesion must not exceed it.
    rec = recommend_therapy(3, lesion_volume_mm3=20000.0, n_regions=3, confidence=0.3)
    assert rec.technique.disease_class == 3


def test_recommend_to_dict_roundtrip():
    rec = recommend_therapy(2, lesion_volume_mm3=1500.0, n_regions=1, confidence=0.6)
    d = rec.to_dict()
    assert d["technique_id"]
    assert d["lesion_volume_mm3"] == 1500.0
    assert d["n_regions"] == 1
    assert d["disease_class"] == 2
    assert isinstance(d["references"], list)


def test_recommend_rationale_mentions_lesion_when_present():
    rec = recommend_therapy(2, lesion_volume_mm3=1500.0, n_regions=1, confidence=0.6)
    assert "1 region" in rec.rationale or "1 region(s)" in rec.rationale
    assert "1500" in rec.rationale


def test_recommend_rationale_mentions_no_lesion_when_absent():
    rec = recommend_therapy(0, lesion_volume_mm3=0.0, n_regions=0, confidence=0.9)
    assert "No focal lesions" in rec.rationale


def test_recommend_confidence_note_low_vs_high():
    low = recommend_therapy(1, confidence=0.2)
    high = recommend_therapy(1, confidence=0.8)
    assert "moderate" in low.rationale.lower()
    assert "well-supported" in high.rationale.lower()


def test_build_cure_frames_empty_when_no_lesion():
    from brainframe.therapy.animation import build_cure_frames

    assert build_cure_frames(None, [], 1000.0, 0.5) == []


def test_build_cure_frames_produces_expected_count():
    from brainframe.reconstruction.marching import MeshData
    from brainframe.therapy.animation import build_cure_frames

    # A small tetrahedron mesh as a stand-in lesion.
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    mesh = MeshData(
        label="lesion",
        label_idx=4,
        vertices=verts,
        faces=faces,
        normals=np.zeros_like(verts),
        spacing=(1.0, 1.0, 1.0),
    )
    frames = build_cure_frames(mesh, [], before_volume_mm3=1000.0, recovery_frac=0.5, n_frames=8)
    assert len(frames) == 8
    # Each frame shrinks the lesion; volume in hover should decrease.
    assert frames[0].name == "0"
    assert frames[-1].name == "7"
    # First frame has the full lesion; last frame reflects the recovery.
    assert frames[-1].data  # non-empty trace
