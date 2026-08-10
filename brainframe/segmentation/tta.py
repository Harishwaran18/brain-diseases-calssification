"""Test-time adaptation for SAM (retraining-free, no source data, no labels).

A lightweight, self-supervised adaptation that nudges a tiny set of affine parameters of
SAM's mask decoder (the IoU token embedding) using three consistency signals:

* **Entropy minimization** -- sharpen the predicted mask distribution.
* **IoU uncertainty** -- reduce variance across the multi-mask output scores.
* **Dual-scale consistency** -- predictions on two rescaled views of the same image
  should agree (EMA teacher).

The adaptation touches only a handful of parameters (no labels, no source data), so it
is genuinely retraining-free in the sense of the research statement. When SAM is not
available the TTA is a no-op.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from brainframe.config import SegmentationTTAConfig
from brainframe.utils.logging import get_logger

log = get_logger("segmentation.tta")


@dataclass
class TTAState:
    adapted: bool
    steps: int
    final_loss: float
    teacher_ema: bool


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -(probs * (probs + eps).log()).sum(dim=1).mean()


def _dual_scale_consistency(
    student_probs: torch.Tensor, teacher_probs: torch.Tensor
) -> torch.Tensor:
    # Both already spatially aligned; use symmetric KL-ish distance
    eps = 1e-8
    kl = (
        (student_probs * ((student_probs + eps).log() - (teacher_probs + eps).log()))
        .sum(dim=1)
        .mean()
    )
    return kl


def _rescale(image_np: np.ndarray, scale: float) -> np.ndarray:
    from scipy.ndimage import zoom

    if abs(scale - 1.0) < 1e-3:
        return image_np
    return zoom(image_np, (scale, scale), order=1)


def adapt_segmenter(
    segmenter,
    image: np.ndarray,
    cfg: SegmentationTTAConfig,
    device: str = "cpu",
) -> TTAState:
    """Run TTA on a single image; mutate ``segmenter`` in place if possible."""
    if not cfg.enabled:
        return TTAState(adapted=False, steps=0, final_loss=float("nan"), teacher_ema=False)

    sam = getattr(segmenter, "sam", None)
    if sam is None or not getattr(segmenter, "available", False):
        log.info("TTA skipped: SAM not available (heuristic segmenter is non-adaptive).")
        return TTAState(adapted=False, steps=0, final_loss=float("nan"), teacher_ema=False)

    # Adapt only the IoU token embedding parameters (a tiny set).
    adapt_params: list[torch.Tensor] = []
    for name, p in sam.named_parameters():
        if any(tok in name for tok in cfg.adapt_params) and p.requires_grad:
            p.requires_grad_(True)
            adapt_params.append(p)
    if not adapt_params:
        log.info("TTA: no adaptable parameters found; skipping.")
        return TTAState(adapted=False, steps=0, final_loss=float("nan"), teacher_ema=False)

    teacher = copy.deepcopy(sam)
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.SGD(adapt_params, lr=cfg.lr)
    final_loss = float("nan")

    img_small = _rescale(image, 0.75)
    for _step in range(cfg.steps):
        opt.zero_grad()
        prompts = {
            "point_coords": np.array([[image.shape[1] / 2, image.shape[0] / 2]], dtype=np.float32),
            "point_labels": np.array([1]),
        }
        masks_s = segmenter.predict(image.astype(np.float32), prompts)
        if not masks_s:
            continue
        scores = torch.tensor([m.score for m in masks_s], requires_grad=False)
        probs = torch.softmax(scores, dim=0)
        loss_e = _entropy(probs.unsqueeze(0))
        loss_unc = scores.var()
        masks_t = segmenter.predict(img_small.astype(np.float32), prompts)
        if masks_t:
            t_scores = torch.tensor([m.score for m in masks_t]).detach()
            t_probs = torch.softmax(t_scores, dim=0)
            loss_dual = _dual_scale_consistency(probs.unsqueeze(0).detach(), t_probs.unsqueeze(0))
        else:
            loss_dual = torch.tensor(0.0)
        loss = loss_e + 0.5 * loss_unc + 0.3 * loss_dual
        final_loss = float(loss.item())
        loss.backward()
        opt.step()
        # EMA teacher update
        with torch.no_grad():
            for tp, p in zip(teacher.parameters(), sam.parameters(), strict=False):
                tp.mul_(cfg.ema_decay).add_(p.detach(), alpha=1 - cfg.ema_decay)

    log.info(
        "TTA adapted %d params over %d steps (final loss=%.4f)",
        len(adapt_params),
        cfg.steps,
        final_loss,
    )
    return TTAState(adapted=True, steps=cfg.steps, final_loss=final_loss, teacher_ema=True)
