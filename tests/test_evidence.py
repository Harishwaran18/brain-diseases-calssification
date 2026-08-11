"""Tests for the evidence-based differential classifier."""

from __future__ import annotations

import numpy as np

from brainframe.classification.evidence import classify
from brainframe.config import LABELS

SHAPE = (96, 112, 84)


def _lesion_report(vol_mm3: float, n: int, centroids=None) -> dict:
    regions = []
    if centroids:
        for c in centroids:
            regions.append({"centroid": list(c), "volume_mm3": vol_mm3 / len(centroids)})
    return {"total_lesion_volume_mm3": vol_mm3, "n_regions": n, "regions": regions}


def test_healthy_volume_classifies_as_healthy_with_high_confidence():
    report = _lesion_report(0.0, 0)
    r = classify(report, None)
    assert r.prediction == 0
    assert r.disease.short_name == "Healthy"
    assert r.confidence > 0.90


def test_periventricular_bilateral_classifies_as_ms():
    labels = np.zeros(SHAPE, dtype=np.uint8)
    for x in (40, 56):
        for z in range(40, 55):
            for y in range(45, 65):
                labels[x, y, z] = LABELS["lesion"]
    report = _lesion_report(1500.0, 4, [[48.0, 55.0, 47.0]])
    r = classify(report, labels)
    assert r.prediction == 3
    assert r.disease.short_name == "MS"
    assert r.confidence > 0.75


def test_large_unilateral_frontal_lesion_favors_focal_mass():
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[30:45, 85:100, 40:60] = LABELS["lesion"]
    report = _lesion_report(25000.0, 1, [[37.0, 92.0, 50.0]])
    r = classify(report, labels)
    # Focal mass disease (Glioma or TBI or Stroke) — top-3 should contain one.
    top3 = {d["short_name"] for d in r.differential}
    assert top3 & {"Glioma", "TBI", "Stroke"}


def test_confidence_drops_for_ambiguous_central_lesion():
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[44:52, 55:65, 40:50] = LABELS["lesion"]
    report = _lesion_report(1500.0, 1, [[48.0, 60.0, 45.0]])
    r = classify(report, labels)
    # A generic central blob is ambiguous; confidence should not be near 1.0.
    assert r.confidence < 0.85


def test_differential_has_three_entries_normalised():
    labels = np.zeros(SHAPE, dtype=np.uint8)
    for x in (40, 56):
        for z in range(40, 55):
            for y in range(45, 65):
                labels[x, y, z] = LABELS["lesion"]
    report = _lesion_report(2000.0, 4, [[48.0, 55.0, 47.0]])
    r = classify(report, labels)
    assert len(r.differential) == 3
    total = sum(d["probability"] for d in r.differential)
    assert 0.0 < total <= 1.0


def test_features_extracted_with_label_volume():
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[30:40, 85:95, 40:50] = LABELS["lesion"]
    report = _lesion_report(3000.0, 1, [[35.0, 90.0, 45.0]])
    r = classify(report, labels)
    assert r.features.dominant_region  # not "unknown"
    assert r.features.pattern in {"focal", "diffuse", "symmetric", "periventricular"}
    assert r.features.laterality in {"left", "right", "bilateral", "any"}


def test_to_dict_roundtrip():
    report = _lesion_report(0.0, 0)
    r = classify(report, None)
    d = r.to_dict()
    assert d["prediction"] == 0
    assert "disease" in d and "features" in d
    assert len(d["scores"]) == 10


def test_scores_cover_all_ten_diseases():
    report = _lesion_report(0.0, 0)
    r = classify(report, None)
    class_ids = {s.class_id for s in r.scores}
    assert class_ids == set(range(10))
