"""Inference helpers for the classification baseline."""

from __future__ import annotations

import numpy as np
import torch

from brainframe.data.loaders import LoadResult, load_volume
from brainframe.data.preprocessing import normalize_intensity, resample_isotropic
from brainframe.utils.logging import get_logger

log = get_logger("classification.predict")


def _prepare_volume(result: LoadResult, patch_size=(96, 96, 96)) -> torch.Tensor:
    vol = resample_isotropic(result.volume, result.spacing, (1.0, 1.0, 1.0))
    vol = normalize_intensity(vol)
    out = np.zeros(patch_size, dtype=np.float32)
    for i in range(3):
        src = vol.shape[i]
        dst = patch_size[i]
        if src >= dst:
            start = (src - dst) // 2
            vol = vol[tuple(slice(None) if j != i else slice(start, start + dst) for j in range(3))]
        else:
            pad = (dst - src) // 2
            padded = np.zeros(dst, dtype=np.float32)
            padded[pad : pad + src] = vol if i == 0 else vol[tuple(slice(None) for _ in range(i))]
            vol = padded
    out[: vol.shape[0], : vol.shape[1], : vol.shape[2]] = vol[
        : patch_size[0], : patch_size[1], : patch_size[2]
    ]
    return torch.from_numpy(out).unsqueeze(0).unsqueeze(0).float()  # (1,1,X,Y,Z)


def predict_volume(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device | str = "cpu",
    patch_size=(96, 96, 96),
) -> dict:
    """Run ``model`` on a single NIfTI/DICOM volume; return prediction + probabilities."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device).eval()
    result = load_volume(image_path)
    x = _prepare_volume(result, patch_size).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    return {
        "subject": result.source,
        "prediction": pred,
        "probabilities": probs.tolist(),
        "num_classes": probs.shape[0],
    }
