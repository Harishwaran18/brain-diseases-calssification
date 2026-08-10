"""Computational evaluation of neuroregenerative therapy."""

from __future__ import annotations

from brainframe.evaluation.compatibility import CompatibilityResult, compute_compatibility
from brainframe.evaluation.lesion_analysis import LesionReport, analyze_lesions
from brainframe.evaluation.report import generate_report
from brainframe.evaluation.simulator import SimulationResult, simulate_therapy
from brainframe.evaluation.therapy_model import TherapySpec

__all__ = [
    "CompatibilityResult",
    "LesionReport",
    "SimulationResult",
    "TherapySpec",
    "analyze_lesions",
    "compute_compatibility",
    "generate_report",
    "simulate_therapy",
]
