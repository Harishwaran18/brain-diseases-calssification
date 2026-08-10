"""Structural-compatibility and coverage scoring for therapy plans.

The compatibility score is a weighted combination of:

* **Coverage** -- fraction of the target lesion region reached by the therapy field.
* **Recovery** -- reduction in lesion volume (regeneration / reversal modes).
* **Risk** -- proximity penalty: the closer the effect field extends to healthy
  structures, the higher the risk (lower score).

All components are in [0, 1]; the final score is a weighted sum bounded [0, 1].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from brainframe.config import LABELS, EvaluationConfig
from brainframe.evaluation.lesion_analysis import LesionReport
from brainframe.evaluation.simulator import SimulationResult
from brainframe.utils.logging import get_logger

log = get_logger("evaluation.compatibility")


@dataclass
class CompatibilityResult:
    coverage: float
    recovery: float
    risk: float
    score: float
    components: dict

    def to_dict(self) -> dict:
        return {
            "coverage": round(self.coverage, 4),
            "recovery": round(self.recovery, 4),
            "risk": round(self.risk, 4),
            "score": round(self.score, 4),
            "components": self.components,
        }


def compute_compatibility(
    simulation: SimulationResult,
    lesion_report: LesionReport,
    label_volume: np.ndarray,
    cfg: EvaluationConfig | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> CompatibilityResult:
    """Compute the structural-compatibility score for a simulated therapy."""
    if cfg is None:
        from brainframe.config import default_config

        cfg = default_config().evaluation
    comp = cfg.compatibility
    lesion_idx = LABELS["lesion"]

    # Coverage: fraction of lesion voxels within the effective field.
    lesion_mask = label_volume == lesion_idx
    lesion_voxels = int(lesion_mask.sum())
    if lesion_voxels > 0:
        covered = int((lesion_mask & (simulation.effect_field > 0.05)).sum())
        coverage = covered / lesion_voxels
    else:
        coverage = 1.0

    # Recovery: relative lesion volume reduction.
    if simulation.before_lesion_volume_mm3 > 0:
        recovery = 1.0 - (simulation.after_lesion_volume_mm3 / simulation.before_lesion_volume_mm3)
    else:
        recovery = 0.0
    recovery = float(np.clip(recovery, 0.0, 1.0))

    # Risk: effect field overlapping healthy structures (white/gray matter) too close.
    healthy_structures = [LABELS[s] for s in ["white_matter", "gray_matter"] if s in LABELS]
    healthy_mask = np.isin(label_volume, healthy_structures)
    overlap = float((healthy_mask & (simulation.effect_field > 0.2)).sum())
    healthy_total = int(healthy_mask.sum())
    risk = 1.0 - (overlap / healthy_total) if healthy_total > 0 else 0.0
    risk = float(np.clip(risk, 0.0, 1.0))

    score = (
        comp.coverage_weight * coverage + comp.recovery_weight * recovery + comp.risk_weight * risk
    )
    score = float(np.clip(score, 0.0, 1.0))

    log.info(
        "Compatibility: coverage=%.3f recovery=%.3f risk=%.3f score=%.3f",
        coverage,
        recovery,
        risk,
        score,
    )
    return CompatibilityResult(
        coverage=coverage,
        recovery=recovery,
        risk=risk,
        score=score,
        components={
            "coverage": coverage,
            "recovery": recovery,
            "risk": risk,
            "weights": asdict(comp),
        },
    )
