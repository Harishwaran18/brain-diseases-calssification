"""Therapy parameterization.

A therapy is fully described by a target region, an intervention radius, a dose, and a
mode (stimulation, regeneration, or lesion-reversal). This module turns the YAML
``evaluation.therapy`` config into a typed :class:`TherapySpec`.
"""

from __future__ import annotations

from dataclasses import dataclass

from brainframe.config import EvaluationConfig
from brainframe.utils.logging import get_logger

log = get_logger("evaluation.therapy_model")


@dataclass
class TherapySpec:
    target_label: str
    target_mode: str  # centroid | largest_region | manual
    radius_mm: float
    dose: float
    mode: str  # stimulation | regeneration | lesion_reversal
    kernel: str  # gaussian | diffusion
    sigma_mm: float
    target_centroid: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.dose <= 1.0:
            log.warning("Therapy dose %.3f outside [0,1]; clipping.", self.dose)
            self.dose = max(0.0, min(1.0, self.dose))
        if self.mode not in ("stimulation", "regeneration", "lesion_reversal"):
            raise ValueError(f"Unknown therapy mode: {self.mode}")
        if self.kernel not in ("gaussian", "diffusion"):
            raise ValueError(f"Unknown kernel: {self.kernel}")


def build_therapy(cfg: EvaluationConfig | None = None) -> TherapySpec:
    """Build a TherapySpec from the evaluation config."""
    if cfg is None:
        from brainframe.config import default_config

        cfg = default_config().evaluation
    th = cfg.therapy
    return TherapySpec(
        target_label=th.target_label,
        target_mode=th.target_mode,
        radius_mm=th.radius_mm,
        dose=th.dose,
        mode=th.mode,
        kernel=th.kernel,
        sigma_mm=th.sigma_mm,
    )
