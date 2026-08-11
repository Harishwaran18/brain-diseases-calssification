"""Therapy engine: technique library, recommender, and live animation frames.

This package bridges the research framework's "computational evaluation for
neuroregenerative therapy" objective with the user-facing platform. It provides:

- :mod:`brainframe.therapy.library`  -- a curated, parameterized library of
  curing/therapy techniques indexed by disease class and affected region.
- :mod:`brainframe.therapy.recommender` -- selects the best technique from a
  disease prediction and a lesion analysis, with a human-readable rationale.
- :mod:`brainframe.therapy.animation` -- converts a simulation timeline into
  per-timestep 3D frames for the live viewer.
"""

from brainframe.therapy.library import TECHNIQUE_LIBRARY, TherapyTechnique
from brainframe.therapy.recommender import recommend_therapy

__all__ = ["TECHNIQUE_LIBRARY", "TherapyTechnique", "recommend_therapy"]
