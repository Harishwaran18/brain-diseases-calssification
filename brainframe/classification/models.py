"""Classification models (3D CNN + 2D slice ensemble)."""

from __future__ import annotations

import torch
import torch.nn as nn

from brainframe.config import ClassificationModelConfig
from brainframe.utils.logging import get_logger

log = get_logger("classification.models")


class ClassificationModel(nn.Module):
    """Thin wrapper around MONAI 3D networks.

    Supports ``densenet3d`` and ``resnet3d``. Falls back to a tiny pure-torch 3D CNN when
    MONAI is unavailable so that CI / tests do not require heavy weights.
    """

    def __init__(self, cfg: ClassificationModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        if cfg.name in ("densenet3d", "resnet3d"):
            self.net = self._build_monai(cfg)
        else:
            log.warning("Unknown model %r; using fallback CNN", cfg.name)
            self.net = Fallback3DCNN(cfg.in_channels, cfg.num_classes)

    @staticmethod
    def _build_monai(cfg: ClassificationModelConfig) -> nn.Module:
        try:
            from monai.networks.nets import DenseNet121, ResNet
        except ImportError as e:  # pragma: no cover - monai optional
            log.warning("MONAI unavailable (%s); using fallback 3D CNN", e)
            return Fallback3DCNN(cfg.in_channels, cfg.num_classes)

        if cfg.name == "densenet3d":
            return DenseNet121(
                spatial_dims=cfg.spatial_dims,
                in_channels=cfg.in_channels,
                out_channels=cfg.num_classes,
            )
        return ResNet(
            block="basic",
            layers=(2, 2, 2, 2),
            block_inplanes=(64, 128, 256, 512),
            spatial_dims=cfg.spatial_dims,
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if out.ndim == 4 and out.shape[-1] == 1:  # MONAI sometimes returns (B, C, 1, 1, 1)
            out = out.flatten(1)
        if out.ndim > 2:
            out = out.mean(dim=list(range(2, out.ndim)))
        return out


class Fallback3DCNN(nn.Module):
    """A tiny 3D CNN used when MONAI is unavailable (for tests / CPU CI)."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class SliceEnsembleModel(nn.Module):
    """2D ResNet applied per axial slice; logits averaged across slices.

    A lightweight, dependency-free 2D CNN is used as the slice backbone so that tests
    run without torchvision pretrained weights.
    """

    def __init__(self, in_channels: int, num_classes: int, backbone: str = "resnet18"):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.slice_net = _SliceCNN(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W, D) -> treat each (H, W) slice along D as a 2D image.
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B,C,H,W,D), got {x.shape}")
        b, c, h, w, d = x.shape
        slices = x.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
        logits = self.slice_net(slices)  # (B*D, num_classes)
        logits = logits.reshape(b, d, -1).mean(dim=1)
        return logits


class _SliceCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)


def build_classifier(cfg: ClassificationModelConfig) -> nn.Module:
    """Construct a classification model from config.

    Returns the 3D model for ``densenet3d``/``resnet3d`` (or fallback) and the slice
    ensemble for ``slice_ensemble``.
    """
    if cfg.name == "slice_ensemble":
        return SliceEnsembleModel(cfg.in_channels, cfg.num_classes)
    return ClassificationModel(cfg)
