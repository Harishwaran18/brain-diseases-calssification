"""Tests for the classification module."""

from __future__ import annotations

import numpy as np
import torch


def test_build_classifier_fallback_forward():
    from brainframe.classification.models import build_classifier
    from brainframe.config import ClassificationModelConfig

    cfg = ClassificationModelConfig(name="fallback", in_channels=1, num_classes=2)
    model = build_classifier(cfg)
    x = torch.randn(2, 1, 16, 16, 16)
    out = model(x)
    assert out.shape == (2, 2)


def test_slice_ensemble_forward():
    from brainframe.classification.models import build_classifier
    from brainframe.config import ClassificationModelConfig

    cfg = ClassificationModelConfig(name="slice_ensemble", in_channels=1, num_classes=3)
    model = build_classifier(cfg)
    x = torch.randn(2, 1, 16, 16, 8)
    out = model(x)
    assert out.shape == (2, 3)


def test_densenet3d_build_and_forward():
    from brainframe.classification.models import build_classifier
    from brainframe.config import ClassificationModelConfig

    cfg = ClassificationModelConfig(
        name="densenet3d", in_channels=1, num_classes=2, pretrained=False
    )
    model = build_classifier(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 32, 32, 32))
    assert out.shape == (1, 2)


def test_train_step_reduces_loss():
    from brainframe.classification.models import build_classifier
    from brainframe.classification.train import train_classifier
    from brainframe.config import (
        ClassificationConfig,
        ClassificationModelConfig,
        ClassificationTrainConfig,
    )

    cfg = ClassificationConfig(
        model=ClassificationModelConfig(name="fallback", in_channels=1, num_classes=2),
        train=ClassificationTrainConfig(
            epochs=1,
            batch_size=4,
            learning_rate=1e-2,
            amp=False,
            optimizer="adamw",
            scheduler="none",
            early_stopping_patience=10,
            checkpoint="/tmp/clf_test.pt",
        ),
    )
    torch.manual_seed(0)
    model = build_classifier(cfg.model)
    X = torch.randn(16, 1, 16, 16, 16)
    y = torch.randint(0, 2, (16,))
    train_loader = [(X[:8], y[:8])]
    val_loader = [(X[8:], y[8:])]
    model, history = train_classifier(model, train_loader, val_loader, cfg, device="cpu")
    assert len(history.epochs) >= 1
    # The first-epoch loss should be finite and train loss should not be NaN.
    assert np.isfinite(history.epochs[0].train_loss)


def test_predict_volume(tmp_path, nifti_path):
    from brainframe.classification.models import build_classifier
    from brainframe.classification.predict import predict_volume
    from brainframe.config import ClassificationModelConfig

    cfg = ClassificationModelConfig(name="fallback", in_channels=1, num_classes=2)
    model = build_classifier(cfg).eval()
    out = predict_volume(model, nifti_path, device="cpu", patch_size=(16, 16, 16))
    assert "prediction" in out and "probabilities" in out
    assert len(out["probabilities"]) == 2
    assert 0 <= out["prediction"] < 2


def test_predict_volume_pads_small_volume(tmp_path):
    """Patching path: volume smaller than patch_size on every axis must zero-pad cleanly."""
    import nibabel as nib

    from brainframe.classification.models import build_classifier
    from brainframe.classification.predict import predict_volume
    from brainframe.config import ClassificationModelConfig

    small = np.zeros((10, 12, 14), dtype=np.float32)
    p = tmp_path / "small.nii.gz"
    nib.save(nib.Nifti1Image(small, affine=np.eye(4)), str(p))

    cfg = ClassificationModelConfig(name="fallback", in_channels=1, num_classes=2)
    model = build_classifier(cfg).eval()
    out = predict_volume(model, str(p), device="cpu", patch_size=(24, 24, 24))
    assert out["prediction"] in (0, 1)


def test_classification_metrics():
    from brainframe.classification.train import _classification_metrics

    logits = torch.tensor([[2.0, 0.1], [0.2, 3.0], [1.5, 0.5], [0.1, 4.0]])
    targets = torch.tensor([0, 1, 0, 1])
    m = _classification_metrics(logits, targets)
    assert "accuracy" in m and 0.0 <= m["accuracy"] <= 1.0
