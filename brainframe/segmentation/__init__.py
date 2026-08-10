"""Retraining-free SAM-based segmentation (zero/few-shot + test-time adaptation)."""

from __future__ import annotations

from brainframe.segmentation.inference import SegmentResult, segment_volume
from brainframe.segmentation.sam_wrapper import SAMWrapper, build_segmenter

__all__ = ["SAMWrapper", "SegmentResult", "build_segmenter", "segment_volume"]
