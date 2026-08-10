"""PyTorch Dataset adapters for OASIS-1 and BraTS.

Datasets are optional PyTorch dependencies: the module imports torch lazily so that the
core pipeline can run without it. Each dataset yields ``(volume, label)`` tensors for
classification, or ``(slice, mask)`` for segmentation when ground truth is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brainframe.config import LABEL_NAMES
from brainframe.data.loaders import load_volume
from brainframe.data.preprocessing import normalize_intensity, resample_isotropic
from brainframe.utils.logging import get_logger

log = get_logger("data.datasets")


@dataclass
class SubjectRecord:
    """A single subject in a dataset."""

    subject_id: str
    image_path: str
    label: int
    mask_path: str | None = None


def discover_oasis(root: str) -> list[SubjectRecord]:
    """Discover OASIS-1 subjects under ``root``.

    OASIS-1 organises data as ``OAS1_xxxx_MR1/mpr-1.nii.gz``; labels are encoded in the
    clinical CSV (``oasis_cross-sectional.csv``) CDR column. Here we infer a binary label
    from the subject path when the CSV is absent (heuristic: subjects with ID >= OAS1_0200
    are treated as likely-demented only when a CSV is present; otherwise label=0).
    """
    root_path = Path(root)
    if not root_path.exists():
        log.warning("OASIS root not found: %s", root)
        return []
    records: list[SubjectRecord] = []
    csv_path = root_path / "oasis_cross-sectional.csv"
    cdr_map: dict[str, int] = {}
    if csv_path.exists():
        import csv as csvmod

        with csv_path.open() as fh:
            reader = csvmod.DictReader(fh)
            for row in reader:
                sid = row.get("ID", "")
                try:
                    cdr = float(row.get("CDR") or 0)
                except ValueError:
                    cdr = 0.0
                cdr_map[sid] = 1 if cdr > 0 else 0

    for img in sorted(root_path.rglob("*.nii.gz")) + sorted(root_path.rglob("*.nii")):
        sid = img.parent.name
        label = cdr_map.get(sid, 0)
        records.append(SubjectRecord(subject_id=sid, image_path=str(img), label=label))
    return records


def discover_brats(root: str) -> list[SubjectRecord]:
    """Discover BraTS subjects (each subject dir has modality + segmentation files)."""
    root_path = Path(root)
    if not root_path.exists():
        log.warning("BraTS root not found: %s", root)
        return []
    records: list[SubjectRecord] = []
    for sub in sorted(root_path.glob("BraTS*")):
        if not sub.is_dir():
            continue
        t1ce = next(sub.glob("*t1ce.nii.gz"), None)
        seg = next(sub.glob("*seg.nii.gz"), None)
        if t1ce is None:
            continue
        records.append(
            SubjectRecord(
                subject_id=sub.name,
                image_path=str(t1ce),
                label=0,  # BraTS labels are lesion masks, not disease classes
                mask_path=str(seg) if seg else None,
            )
        )
    return records


def _load_record_volume(record: SubjectRecord, target_spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    res = load_volume(record.image_path)
    vol = res.volume
    vol = resample_isotropic(vol, res.spacing, target_spacing)
    return normalize_intensity(vol)


class _TorchDatasetMixin:
    """Lazily import torch for the Dataset base class."""

    @staticmethod
    def _torch():
        import torch

        return torch


class OasisDataset(_TorchDatasetMixin):
    """OASIS-1 classification dataset -> ``(volume, label)`` tensors."""

    def __init__(self, root: str, patch_size=(96, 96, 96), augment=False, indices=None):
        self.patch_size = tuple(patch_size)
        self.augment = augment
        self.records = discover_oasis(root)
        if indices is not None:
            self.records = [self.records[i] for i in indices]

    def __len__(self) -> int:
        return len(self.records)

    def _crop_or_pad(self, vol: np.ndarray) -> np.ndarray:
        ph, pw, pd = self.patch_size
        out = np.zeros(self.patch_size, dtype=np.float32)
        sh, sw, sd = vol.shape
        for i in range(3):
            src = vol.shape[i]
            dst = self.patch_size[i]
            if src >= dst:
                start = (src - dst) // 2
                sl = slice(start, start + dst)
                tmp = (
                    vol[(slice(None),) * i + (sl,)]
                    if i < 2
                    else vol[(slice(None), slice(None), sl)]
                )
            else:
                pad = (dst - src) // 2
                tmp = np.zeros(dst, dtype=np.float32)
                tmp[pad : pad + src] = vol[(slice(None),) * i] if i < 2 else vol[..., :]
            out[(slice(None),) * i] = tmp if i == 0 else out[(slice(None),) * i + (slice(None),)]
        return out

    def __getitem__(self, idx: int):
        torch = self._torch()
        rec = self.records[idx]
        vol = _load_record_volume(rec)
        vol = self._crop_or_pad(vol)
        if self.augment and np.random.rand() < 0.5:
            vol = np.flip(vol, axis=0).copy()
        return torch.from_numpy(vol).unsqueeze(0).float(), torch.tensor(rec.label).long()


class BraTSDataset(_TorchDatasetMixin):
    """BraTS segmentation dataset -> ``(slice, mask)`` tensors along axial axis."""

    def __init__(self, root: str, slice_axis: int = 2, indices=None):
        self.slice_axis = slice_axis
        self.records = discover_brats(root)
        if indices is not None:
            self.records = [self.records[i] for i in indices]
        self._slice_cache: dict[int, list[tuple[int, int]]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        torch = self._torch()
        rec = self.records[idx]
        vol = _load_record_volume(rec)
        if rec.mask_path:
            mask_res = load_volume(rec.mask_path)
            mask = resample_isotropic(mask_res.volume.astype(np.int16), mask_res.spacing)
        else:
            mask = np.zeros_like(vol, dtype=np.int16)
        z = vol.shape[self.slice_axis] // 2
        sl_img = np.take(vol, z, axis=self.slice_axis)
        sl_mask = np.take(mask, z, axis=self.slice_axis)
        return torch.from_numpy(sl_img).unsqueeze(0).float(), torch.from_numpy(sl_mask).long()


def split_indices(
    n: int, ratios=(0.7, 0.15, 0.15), seed: int = 42
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic index split into train/val/test."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train = perm[:n_train].tolist()
    val = perm[n_train : n_train + n_val].tolist()
    test = perm[n_train + n_val :].tolist()
    return train, val, test


# Module-level convenience so other code can reference label names without torch.
__all__ = [
    "BraTSDataset",
    "LABEL_NAMES",
    "OasisDataset",
    "SubjectRecord",
    "discover_brats",
    "discover_oasis",
    "split_indices",
]
