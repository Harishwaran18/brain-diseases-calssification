"""Supervised brain-disease classification baseline.

This is the entry-point module that contrasts with the retraining-free segmentation
pipeline: it depends on labelled data and explicit training. Three backbones are
provided:

* ``densenet3d`` / ``resnet3d`` -- MONAI 3D networks over the full volume patch.
* ``slice_ensemble`` -- a 2D ResNet applied per axial slice with mean aggregation.

All models accept a ``(B, C, X, Y, Z)`` tensor and return logits ``(B, num_classes)``.
"""

from __future__ import annotations

from brainframe.classification.models import (
    ClassificationModel,
    SliceEnsembleModel,
    build_classifier,
)
from brainframe.classification.predict import predict_volume
from brainframe.classification.train import train_classifier

__all__ = [
    "ClassificationModel",
    "SliceEnsembleModel",
    "build_classifier",
    "predict_volume",
    "train_classifier",
]
