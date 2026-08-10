"""Segment Anything Model wrapper with auto-download and a heuristic fallback.

The real SAM (``segment_anything`` package + ViT-B checkpoint) is used when available.
When the checkpoint cannot be downloaded or the package is missing, a
:class:`HeuristicSegmenter` takes over so that the pipeline remains runnable end-to-end
on CPU/CI. The heuristic is intensity-based watershed-style region assignment -- it
mimics SAM's multi-mask output (a few candidate masks per slice) and is deterministic.

Both segmenters share the same interface: ``predict(image, prompts) -> list[Mask]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from brainframe.config import SegmentationConfig
from brainframe.utils.logging import get_logger

log = get_logger("segmentation.sam_wrapper")

SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec164.pth",
}
SAM_FILENAME = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec164.pth",
}


@dataclass
class Mask:
    """A single predicted segmentation mask."""

    segmentation: np.ndarray  # bool / uint8 (H, W)
    score: float
    label: str = "unknown"
    area: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x0, y0, x1, y1


def _download_checkpoint(model_type: str, dest: Path) -> Path | None:
    url = SAM_CHECKPOINT_URLS.get(model_type)
    if url is None:
        return None
    try:
        import urllib.request

        log.info("Downloading SAM %s checkpoint to %s ...", model_type, dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception as e:  # pragma: no cover - network dependent
        log.warning("SAM checkpoint download failed: %s", e)
        return None


def resolve_checkpoint(cfg: SegmentationConfig) -> Path | None:
    """Find or download the SAM checkpoint; return path or None."""
    if cfg.sam_checkpoint:
        p = Path(cfg.sam_checkpoint)
        return p if p.exists() else None
    fname = SAM_FILENAME.get(cfg.sam_model_type, SAM_FILENAME["vit_b"])
    dest = (
        Path(cfg.sam_checkpoint_dir if hasattr(cfg, "sam_checkpoint_dir") else "data/checkpoints")
        / fname
    )
    if dest.exists():
        return dest
    return None


class SAMWrapper:
    """Wrapper around the real Meta SAM model (zero/few-shot, no retraining)."""

    def __init__(
        self, model_type: str = "vit_b", checkpoint: str | Path | None = None, device: str = "cpu"
    ):
        self.model_type = model_type
        self.device = device
        self.sam = self._load(checkpoint)

    def _load(self, checkpoint: str | Path | None):
        try:
            from segment_anything import sam_model_registry

            ckpt = str(checkpoint) if checkpoint else None
            if ckpt is None:
                log.warning("No SAM checkpoint; SAMWrapper will not be usable.")
                return None
            sam = sam_model_registry[self.model_type](checkpoint=ckpt)
            sam.to(self.device)
            sam.eval()
            self._predictor = None
            return sam
        except ImportError:
            log.info("segment_anything not installed; SAM unavailable.")
            return None
        except Exception as e:  # pragma: no cover - checkpoint specific
            log.warning("SAM load failed (%s); falling back to heuristic.", e)
            return None

    @property
    def available(self) -> bool:
        return self.sam is not None

    def predict(self, image: np.ndarray, prompts: dict[str, Any] | None = None) -> list[Mask]:
        if not self.available:
            return []
        try:
            from segment_anything import SamPredictor
        except ImportError:
            return []
        if self._predictor is None:
            self._predictor = SamPredictor(self.sam)
        img = _to_uint8(image)
        self._predictor.set_image(img)
        masks_out: list[Mask] = []
        point_coords = prompts.get("point_coords") if prompts else None
        point_labels = prompts.get("point_labels") if prompts else None
        box = prompts.get("box") if prompts else None
        kwargs: dict[str, Any] = {"multimask_output": True}
        if point_coords is not None:
            kwargs["point_coords"] = np.asarray(point_coords)[None]
            kwargs["point_labels"] = np.asarray(point_labels)[None]
        if box is not None:
            kwargs["box"] = np.asarray(box)[None]
        masks, scores, _ = self._predictor.predict(**kwargs)
        for m, s in zip(masks, scores, strict=False):
            ys, xs = np.where(m)
            bbox = (
                (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                if len(xs)
                else (0, 0, 0, 0)
            )
            masks_out.append(Mask(m.astype(np.uint8), float(s), area=int(m.sum()), bbox=bbox))
        return masks_out


def _to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    out = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    if out.ndim == 2:  # SAM expects 3-channel
        out = np.stack([out, out, out], axis=-1)
    return out


class HeuristicSegmenter:
    """Intensity-based multi-region segmenter used when SAM is unavailable.

    Produces candidate masks in the spirit of SAM: a few masks ranked by score,
    derived from thresholding the normalized intensity and connected components. The
    output is suitable for label assignment downstream.
    """

    def __init__(self, n_levels: int = 4):
        self.n_levels = n_levels

    @property
    def available(self) -> bool:
        return True

    def predict(self, image: np.ndarray, prompts: dict[str, Any] | None = None) -> list[Mask]:
        img = image.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        if hi - lo < 1e-6:
            return []
        norm = (img - lo) / (hi - lo)
        masks: list[Mask] = []
        # Quantile-based thresholds produce distinct candidate regions
        thresholds = (
            np.quantile(norm[norm > 0.05], np.linspace(0.2, 0.9, self.n_levels))
            if (norm > 0.05).any()
            else [0.5]
        )
        for _i, t in enumerate(thresholds):
            m = (norm > float(t)).astype(np.uint8)
            score = float(1.0 - abs(0.5 - t) * 0.6)  # higher near mid-intensity
            ys, xs = np.where(m > 0)
            bbox = (
                (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                if len(xs)
                else (0, 0, 0, 0)
            )
            masks.append(Mask(m, max(0.0, score), area=int(m.sum()), bbox=bbox))
        return masks


def build_segmenter(
    cfg: SegmentationConfig, device: str = "cpu", allow_download: bool = True
) -> Any:
    """Construct the best available segmenter given the config.

    Order of preference: real SAM (with downloaded checkpoint) -> heuristic fallback.
    """
    ckpt = None
    if cfg.sam_checkpoint is None and allow_download:
        fname = SAM_FILENAME.get(cfg.sam_model_type, SAM_FILENAME["vit_b"])
        dest = Path(os.environ.get("BRAINFRAME_CHECKPOINT_DIR", "data/checkpoints")) / fname
        if not dest.exists():
            _download_checkpoint(cfg.sam_model_type, dest)
        if dest.exists():
            ckpt = dest
    elif cfg.sam_checkpoint:
        ckpt = Path(cfg.sam_checkpoint) if Path(cfg.sam_checkpoint).exists() else None

    try:
        import segment_anything  # noqa: F401
    except ImportError:
        log.info("segment_anything not installed; using heuristic segmenter.")
        return HeuristicSegmenter()

    if ckpt is None:
        log.info("No SAM checkpoint available; using heuristic segmenter.")
        return HeuristicSegmenter()

    wrapper = SAMWrapper(cfg.sam_model_type, checkpoint=ckpt, device=device)
    if not wrapper.available:
        log.info("SAM unavailable; using heuristic segmenter.")
        return HeuristicSegmenter()
    log.info("SAM %s loaded from %s", cfg.sam_model_type, ckpt)
    return wrapper
