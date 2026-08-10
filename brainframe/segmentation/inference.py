"""Slice-wise inference + label assignment for SAM masks.

Maps the candidate masks returned by :mod:`sam_wrapper` to canonical tissue classes
(``background``, ``gray_matter``, ``white_matter``, ``csf``, ``lesion``) using intensity
ranking: the brightest contiguous region is white matter, next is gray matter, the
lowest non-background is CSF, and an outlier hyper/hypo-intense region is flagged as a
potential lesion. This is the retraining-free, label-free assignment step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brainframe.config import LABELS, SegmentationConfig
from brainframe.data.preprocessing import normalize_intensity
from brainframe.data.samplers import SliceSampler
from brainframe.segmentation.postprocess import postprocess_mask
from brainframe.segmentation.prompts import build_prompts
from brainframe.segmentation.sam_wrapper import Mask, build_segmenter
from brainframe.utils.logging import get_logger

log = get_logger("segmentation.inference")


@dataclass
class SegmentResult:
    """Per-volume segmentation output."""

    label_volume: np.ndarray  # (X, Y, Z) int16 with LABELS indices
    slices_used: list[int]
    n_labels: int
    spacing: tuple[float, float, float]


def _assign_labels(masks: list[Mask], image: np.ndarray, classes: list[str]) -> np.ndarray:
    """Assign SAM candidate masks to tissue classes by intensity ranking.

    Candidate masks from the heuristic segmenter are typically *nested* (each is the
    region above a different intensity threshold). We therefore build concentric bands:
    the brightest core -> white matter, the next ring -> gray matter, the dimmest
    non-background ring -> CSF. An extreme-intensity outlier band is flagged as lesion.
    """
    if not masks:
        return np.zeros(image.shape[:2], dtype=np.int16)
    img = image.astype(np.float32)

    # Rank masks by mean intensity (brightest first). For nested masks this gives
    # smallest/brightest -> largest/dimmest.
    scored = []
    for m in masks:
        region = img[m.segmentation.astype(bool)]
        mean_int = float(region.mean()) if region.size else 0.0
        scored.append((mean_int, m))
    scored.sort(key=lambda t: t[0], reverse=True)

    out = np.zeros(image.shape[:2], dtype=np.int16)
    band_labels = ["white_matter", "gray_matter", "csf"]
    prev_mask = np.zeros(img.shape[:2], dtype=bool)
    for rank, (_mean_int, m) in enumerate(scored):
        cur = m.segmentation.astype(bool)
        band = cur & ~prev_mask  # the ring unique to this level
        if rank < len(band_labels) and band.any():
            out[band] = LABELS[band_labels[rank]]
        prev_mask = prev_mask | cur

    # Lesion detection: extreme hyper-intense outlier (e.g. >3 sigma above mean) that is
    # spatially small. Conservative so it does not hijack normal tissue.
    norm = normalize_intensity(img)
    hot = norm > 3.0
    if hot.sum() > 0:
        out[hot] = LABELS["lesion"]
    return out


def segment_slice(segmenter, image: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Segment a single 2D slice into a label map."""
    image = image.astype(np.float32)
    prompts = build_prompts(
        image,
        strategy=cfg.prompts.strategy,
        spacing=cfg.prompts.grid_spacing,
        max_points=cfg.prompts.max_points,
        use_bbox=cfg.prompts.use_bbox,
        bbox_from=cfg.prompts.bbox_from,
    )
    masks = segmenter.predict(image, prompts)
    label_map = _assign_labels(masks, image, cfg.inference.label_classes)
    label_map = postprocess_mask(label_map, cfg.postprocess)
    return label_map.astype(np.int16)


def segment_volume(
    volume: np.ndarray,
    cfg: SegmentationConfig,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cpu",
    segmenter=None,
) -> SegmentResult:
    """Segment a 3D volume slice-by-slice (no retraining)."""
    if segmenter is None:
        segmenter = build_segmenter(cfg, device=device)
    sampler = SliceSampler(volume, axis=cfg.inference.slice_axis, stride=1)
    label_volume = np.zeros(volume.shape, dtype=np.int16)
    used: list[int] = []
    for idx, sl in sampler:
        label_map = segment_slice(segmenter, sl, cfg)
        sl_setter = [slice(None)] * volume.ndim
        sl_setter[cfg.inference.slice_axis] = idx
        label_volume[tuple(sl_setter)] = label_map
        used.append(idx)
    log.info("Segmented %d slices (axis=%d)", len(used), cfg.inference.slice_axis)
    return SegmentResult(
        label_volume=label_volume,
        slices_used=used,
        n_labels=len(LABELS),
        spacing=spacing,
    )
