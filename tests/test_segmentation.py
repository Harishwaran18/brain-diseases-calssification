"""Tests for the retraining-free segmentation module."""

from __future__ import annotations

import numpy as np


def test_heuristic_segmenter_returns_masks():
    from brainframe.segmentation.prompts import build_prompts
    from brainframe.segmentation.sam_wrapper import HeuristicSegmenter

    img = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.mgrid[0:64, 0:64]
    img[(xx - 32) ** 2 + (yy - 32) ** 2 < 28**2] = 0.4
    img[(xx - 32) ** 2 + (yy - 32) ** 2 < 20**2] = 0.7
    img[(xx - 32) ** 2 + (yy - 32) ** 2 < 8**2] = 0.15
    img[(xx - 20) ** 2 + (yy - 40) ** 2 < 4**2] = 0.95
    seg = HeuristicSegmenter()
    prompts = build_prompts(img, strategy="grid", spacing=32, max_points=8, use_bbox=True)
    masks = seg.predict(img, prompts)
    assert len(masks) > 0
    for m in masks:
        assert m.segmentation.shape == img.shape
        assert 0.0 <= m.score <= 1.0
        assert m.area > 0


def test_build_segmenter_falls_back_to_heuristic(default_cfg):
    from brainframe.segmentation.sam_wrapper import HeuristicSegmenter, build_segmenter

    cfg = default_cfg.segmentation
    seg = build_segmenter(cfg, device="cpu", allow_download=False)
    assert isinstance(seg, HeuristicSegmenter)
    assert seg.available


def test_segment_slice_produces_label_map(default_cfg):
    from brainframe.segmentation.inference import segment_slice
    from brainframe.segmentation.sam_wrapper import HeuristicSegmenter

    img = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.mgrid[0:64, 0:64]
    img[(xx - 32) ** 2 + (yy - 32) ** 2 < 28**2] = 0.4
    img[(xx - 32) ** 2 + (yy - 32) ** 2 < 20**2] = 0.7
    img[(xx - 32) ** 2 + (yy - 32) ** 2 < 8**2] = 0.15
    img[(xx - 20) ** 2 + (yy - 40) ** 2 < 4**2] = 0.95
    seg = HeuristicSegmenter()
    from brainframe.config import LABELS

    lm = segment_slice(seg, img, default_cfg.segmentation)
    assert lm.shape == img.shape
    labels = set(np.unique(lm).tolist())
    assert labels <= set(LABELS.values())
    # at least one non-background tissue detected
    assert labels - {0}, "Expected at least one tissue label"


def test_segment_volume_shape(default_cfg, synthetic_volume):
    from brainframe.segmentation.inference import segment_volume
    from brainframe.segmentation.sam_wrapper import build_segmenter

    cfg = default_cfg.segmentation
    seg = build_segmenter(cfg, device="cpu", allow_download=False)
    res = segment_volume(synthetic_volume, cfg, segmenter=seg)
    assert res.label_volume.shape == synthetic_volume.shape
    assert res.n_labels == 5
    assert len(res.slices_used) == synthetic_volume.shape[cfg.inference.slice_axis]


def test_postprocess_largest_cc():
    from brainframe.config import LABELS, SegmentationPostprocessConfig
    from brainframe.segmentation.postprocess import postprocess_mask

    lm = np.zeros((20, 20), dtype=np.int16)
    lm[2:5, 2:5] = LABELS["white_matter"]  # small region
    lm[12:18, 12:18] = LABELS["white_matter"]  # larger region
    cfg = SegmentationPostprocessConfig(
        largest_cc=True, fill_holes=True, smoothing_sigma=0.0, min_voxel_count=5
    )
    out = postprocess_mask(lm, cfg)
    # only the larger region should survive (min_voxel_count + largest_cc)
    assert int((out == LABELS["white_matter"]).sum()) == 36


def test_tta_disabled_is_noop(default_cfg):
    from brainframe.segmentation.sam_wrapper import HeuristicSegmenter
    from brainframe.segmentation.tta import adapt_segmenter

    cfg = default_cfg.segmentation.tta
    cfg.enabled = False
    seg = HeuristicSegmenter()
    state = adapt_segmenter(seg, np.zeros((32, 32), dtype=np.float32), cfg, device="cpu")
    assert state.adapted is False
    assert state.steps == 0


def test_prompts_grid_shape():
    from brainframe.segmentation.prompts import build_prompts, grid_points

    pts = grid_points((128, 128), spacing=64, max_points=6)
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert len(pts) <= 6
    p = build_prompts(np.zeros((64, 64)), strategy="grid", spacing=32, use_bbox=True)
    assert "point_coords" in p and "box" in p
