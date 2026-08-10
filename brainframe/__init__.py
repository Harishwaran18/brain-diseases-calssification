"""brainframe: Retraining-free AI framework for 3D neurodegenerative brain
reconstruction and computational therapy evaluation.

The package unifies four composable stages:

1. ``brainframe.classification`` -- supervised brain-disease classification baseline.
2. ``brainframe.segmentation`` -- retraining-free SAM segmentation (zero/few-shot + TTA).
3. ``brainframe.reconstruction`` -- 2D masks -> 3D volume -> isosurface mesh + metrics.
4. ``brainframe.evaluation`` -- computational evaluation of neuroregenerative therapy.

All stages share a common :mod:`brainframe.config` and :mod:`brainframe.data` core and
are orchestrated by :mod:`brainframe.pipeline` / :mod:`brainframe.cli`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
