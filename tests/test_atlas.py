"""Tests for the anatomical atlas (voxel -> region labelling)."""

from __future__ import annotations

import numpy as np

from brainframe.data.atlas import (
    classify_pattern,
    label_lesion_cluster,
    region_at,
)

SHAPE = (96, 112, 84)


def test_region_at_frontal():
    # anterior + superior-ish -> frontal
    assert region_at((48, 95, 60), SHAPE) == "frontal"


def test_region_at_occipital():
    # posterior + inferior -> occipital
    assert region_at((48, 10, 50), SHAPE) == "occipital"


def test_region_at_temporal():
    # inferior-lateral -> temporal
    assert region_at((20, 50, 30), SHAPE) == "temporal"


def test_region_at_cerebellum():
    # posterior-inferior -> cerebellum
    assert region_at((48, 15, 10), SHAPE) == "cerebellum"


def test_region_at_brainstem():
    # inferior-central -> brainstem
    assert region_at((48, 55, 8), SHAPE) == "brainstem"


def test_region_at_periventricular():
    assert region_at((48, 55, 55), SHAPE) == "periventricular"


def test_region_at_parietal():
    # posterior-superior -> parietal
    assert region_at((48, 28, 65), SHAPE) == "parietal"


def test_region_at_motor_cortex():
    assert region_at((48, 58, 62), SHAPE) == "motor_cortex"


def test_label_lesion_cluster_left_hemisphere():
    pts = np.array([[20, 55, 45]] * 10)
    out = label_lesion_cluster(pts, SHAPE)
    assert out["hemisphere"] == "left"
    assert "left" in out["location"]


def test_label_lesion_cluster_bilateral():
    pts = np.array([[20, 55, 45], [70, 55, 45], [21, 56, 45]])
    out = label_lesion_cluster(pts, SHAPE)
    assert out["hemisphere"] == "bilateral"
    assert "bilateral" in out["location"]


def test_label_lesion_cluster_empty():
    out = label_lesion_cluster(np.zeros((0, 3), dtype=int), SHAPE)
    assert out["region"] == "unknown"


def test_classify_pattern_periventricular():
    clusters = [
        {"region": "periventricular", "hemisphere": "bilateral", "voxels": 50},
        {"region": "periventricular", "hemisphere": "bilateral", "voxels": 40},
    ]
    assert classify_pattern(clusters, 90, SHAPE) == "periventricular"


def test_classify_pattern_symmetric():
    clusters = [
        {"region": "temporal", "hemisphere": "left", "voxels": 100},
        {"region": "temporal", "hemisphere": "right", "voxels": 110},
    ]
    assert classify_pattern(clusters, 210, SHAPE) == "symmetric"


def test_classify_pattern_focal_single_region():
    clusters = [{"region": "frontal", "hemisphere": "left", "voxels": 200}]
    assert classify_pattern(clusters, 200, SHAPE) == "focal"


def test_classify_pattern_empty_is_focal():
    assert classify_pattern([], 0, SHAPE) == "focal"
