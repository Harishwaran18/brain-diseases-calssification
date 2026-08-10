"""Therapeutic-interaction simulator.

Applies a configurable intervention (stimulation / regeneration / lesion-reversal) to the
segmented 3D volume. The effect is modelled as a Gaussian or diffusion kernel centered on
the target region and propagated over a small region graph. This is the tractable,
self-contained in-silico counterpart of lesion-as-perturbation models (TVB etc.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as cc_label

from brainframe.config import LABELS, EvaluationConfig
from brainframe.evaluation.lesion_analysis import LesionReport
from brainframe.evaluation.therapy_model import TherapySpec
from brainframe.utils.logging import get_logger

log = get_logger("evaluation.simulator")


@dataclass
class SimulationResult:
    """Result of applying a therapy to a volume."""

    before_lesion_volume_mm3: float
    after_lesion_volume_mm3: float
    affected_voxels: int
    affected_fraction: float
    therapy: dict
    propagated: bool
    label_volume_after: np.ndarray
    effect_field: np.ndarray  # the intervention strength field (0..dose)

    def to_dict(self) -> dict:
        return {
            "before_lesion_volume_mm3": self.before_lesion_volume_mm3,
            "after_lesion_volume_mm3": self.after_lesion_volume_mm3,
            "affected_voxels": self.affected_voxels,
            "affected_fraction": round(self.affected_fraction, 4),
            "therapy": self.therapy,
            "propagated": self.propagated,
        }


def _target_centroid(
    label_volume: np.ndarray, therapy: TherapySpec, lesion_report: LesionReport
) -> tuple[float, float, float] | None:
    if therapy.target_centroid is not None:
        return therapy.target_centroid
    if lesion_report.n_regions == 0:
        return None
    if therapy.target_mode == "largest_region":
        region = max(lesion_report.regions, key=lambda r: r.volume_mm3)
        return region.centroid
    # centroid mode: volume-weighted average of lesion centroids
    regions = lesion_report.regions
    w = np.array([r.volume_mm3 for r in regions])
    w = w / w.sum()
    cents = np.array([r.centroid for r in regions])
    return tuple(float(x) for x in (w[:, None] * cents).sum(axis=0))


def _build_region_graph(label_volume: np.ndarray, k: int) -> dict:
    """Build a simple k-nearest-neighbour graph between lesion sub-regions."""
    lesion_idx = LABELS["lesion"]
    lbl, n = cc_label(label_volume == lesion_idx)
    if n == 0:
        return {"centroids": [], "edges": []}
    from scipy.ndimage import center_of_mass

    cents = [center_of_mass(lbl == i) for i in range(1, n + 1)]
    cents = [tuple(float(c) for c in cm) for cm in cents]
    edges: list[tuple[int, int, float]] = []
    for i, c in enumerate(cents):
        dists = [
            (j, float(np.linalg.norm(np.array(c) - np.array(cents[j]))))
            for j in range(len(cents))
            if j != i
        ]
        dists.sort(key=lambda t: t[1])
        for j, d in dists[:k]:
            edges.append((i, j, d))
    return {"centroids": cents, "edges": edges}


def simulate_therapy(
    label_volume: np.ndarray,
    lesion_report: LesionReport,
    therapy: TherapySpec,
    cfg: EvaluationConfig | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> SimulationResult:
    """Apply a therapy to the label volume and return before/after state."""
    if cfg is None:
        from brainframe.config import default_config

        cfg = default_config().evaluation
    lesion_idx = LABELS["lesion"]
    voxel_volume = float(np.prod(spacing))
    before = float((label_volume == lesion_idx).sum() * voxel_volume)
    target = _target_centroid(label_volume, therapy, lesion_report)
    effect_field = np.zeros(label_volume.shape, dtype=np.float32)
    after = label_volume.astype(np.int16).copy()
    propagated = False

    if target is not None:
        zz, yy, xx = np.mgrid[
            0 : label_volume.shape[0], 0 : label_volume.shape[1], 0 : label_volume.shape[2]
        ]
        # distance in mm using voxel spacing
        sp = np.asarray(spacing, dtype=np.float32)
        dist = np.sqrt(
            ((zz - target[0]) * sp[0]) ** 2
            + ((yy - target[1]) * sp[1]) ** 2
            + ((xx - target[2]) * sp[2]) ** 2
        )
        sigma = therapy.sigma_mm or therapy.radius_mm / 2.0
        effect_field = np.exp(-(dist**2) / (2 * sigma**2)).astype(np.float32) * therapy.dose
        # Limit to the target radius
        effect_field[dist > therapy.radius_mm] = 0.0

        if therapy.mode == "lesion_reversal":
            # Reduce lesion voxels inside the effect field proportional to intensity.
            mask = label_volume == lesion_idx
            prob = effect_field * mask
            rng = np.random.default_rng(42)
            recover = rng.random(label_volume.shape) < prob
            after[recover] = LABELS["white_matter"]
        elif therapy.mode == "regeneration":
            # Promote CSF / damaged tissue to white matter within the field.
            csf = label_volume == LABELS["csf"]
            recover = (effect_field > 0.3) & csf
            after[recover] = LABELS["white_matter"]
            # Also slightly shrink lesion within the strong-effect region.
            rng = np.random.default_rng(7)
            lesion = label_volume == lesion_idx
            shrink = lesion & (effect_field > 0.4) & (rng.random(label_volume.shape) < 0.5)
            after[shrink] = LABELS["white_matter"]
        else:  # stimulation: locally enhance gray matter intensity marker (no label change)
            pass

        # Propagate effect over the region graph (diffuse the effect field)
        if cfg.simulator.propagation_steps > 0:
            propagated = True
            for _ in range(cfg.simulator.propagation_steps):
                effect_field = effect_field + cfg.simulator.diffusion_rate * (
                    gaussian_filter(effect_field, sigma=1.0) - effect_field
                )
            effect_field = np.clip(effect_field, 0.0, therapy.dose)

    after_lesion = float((after == lesion_idx).sum() * voxel_volume)
    affected = int((effect_field > 0.01).sum())
    total_voxels = int(label_volume.size)
    log.info(
        "Simulated therapy: before=%.1f after=%.1f mm^3 affected=%d voxels",
        before,
        after_lesion,
        affected,
    )
    return SimulationResult(
        before_lesion_volume_mm3=round(before, 3),
        after_lesion_volume_mm3=round(after_lesion, 3),
        affected_voxels=affected,
        affected_fraction=affected / total_voxels if total_voxels else 0.0,
        therapy=asdict(therapy) if hasattr(therapy, "__dataclass_fields__") else therapy.__dict__,
        propagated=propagated,
        label_volume_after=after,
        effect_field=effect_field,
    )
