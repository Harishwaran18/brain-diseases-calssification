"""Lesion-region spatial analysis from the segmented 3D volume.

Detects connected lesion components, computes centroid/volume/spatial extent, and
measures adjacency to key structures (white/gray matter, CSF). Output feeds the
therapy simulator and compatibility scorer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.ndimage import label as cc_label

from brainframe.config import LABELS, EvaluationConfig
from brainframe.utils.logging import get_logger

log = get_logger("evaluation.lesion_analysis")


@dataclass
class LesionRegion:
    region_id: int
    volume_mm3: float
    centroid: tuple[float, float, float]  # in voxel coords
    spatial_extent: tuple[int, int, int]  # bounding-box size in voxels
    adjacent_structures: dict[str, float]  # structure -> min distance (mm)


@dataclass
class LesionReport:
    total_lesion_volume_mm3: float
    n_regions: int
    regions: list[LesionRegion] = field(default_factory=list)
    lesion_label: str = "lesion"

    def to_dict(self) -> dict:
        return {
            "total_lesion_volume_mm3": self.total_lesion_volume_mm3,
            "n_regions": self.n_regions,
            "lesion_label": self.lesion_label,
            "regions": [asdict(r) for r in self.regions],
        }


def _distance_to_structure(
    lesion_voxels: np.ndarray,
    structure_mask: np.ndarray,
    spacing: tuple[float, float, float],
    sample: int = 200,
) -> float:
    """Approximate min Euclidean (in mm) from lesion voxels to a structure mask."""
    if lesion_voxels.shape[0] == 0 or structure_mask.sum() == 0:
        return float("inf")
    struct_coords = np.argwhere(structure_mask)
    if struct_coords.shape[0] > 5000:
        idx = np.random.default_rng(0).choice(struct_coords.shape[0], 5000, replace=False)
        struct_coords = struct_coords[idx]
    if lesion_voxels.shape[0] > sample:
        idx = np.random.default_rng(1).choice(lesion_voxels.shape[0], sample, replace=False)
        lesion_voxels = lesion_voxels[idx]
    sp = np.asarray(spacing, dtype=np.float32)
    diffs = (
        lesion_voxels[:, None, :].astype(np.float32) - struct_coords[None, :, :].astype(np.float32)
    ) * sp
    d = np.sqrt((diffs**2).sum(axis=2))
    return float(d.min())


def analyze_lesions(
    label_volume: np.ndarray,
    cfg: EvaluationConfig | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> LesionReport:
    """Detect and characterize lesion regions in the volume."""
    if cfg is None:
        from brainframe.config import default_config

        cfg = default_config().evaluation
    lesion_label = cfg.lesion_analysis.lesion_label
    lesion_idx = LABELS.get(lesion_label, LABELS["lesion"])
    lesion = label_volume == lesion_idx
    if lesion.sum() == 0:
        log.info("No lesion voxels detected.")
        return LesionReport(total_lesion_volume_mm3=0.0, n_regions=0, lesion_label=lesion_label)

    lbl, n = cc_label(lesion)
    voxel_volume = float(np.prod(spacing))
    regions: list[LesionRegion] = []
    for r in range(1, n + 1):
        m = lbl == r
        vol_mm3 = float(m.sum() * voxel_volume)
        if vol_mm3 < cfg.lesion_analysis.min_volume_mm3:
            continue
        com = center_of_mass(m)
        coords = np.argwhere(m)
        bb = coords.max(axis=0) - coords.min(axis=0) + 1
        adj = {}
        for sname in cfg.lesion_analysis.adjacency_structures:
            sidx = LABELS.get(sname, None)
            if sidx is None:
                continue
            smask = label_volume == sidx
            adj[sname] = round(_distance_to_structure(coords, smask, spacing), 3)
        regions.append(
            LesionRegion(
                region_id=int(r),
                volume_mm3=round(vol_mm3, 3),
                centroid=tuple(round(float(c), 3) for c in com),
                spatial_extent=tuple(int(b) for b in bb),
                adjacent_structures=adj,
            )
        )
    total = round(sum(r.volume_mm3 for r in regions), 3)
    log.info("Lesion analysis: %d regions, %.1f mm^3 total", len(regions), total)
    return LesionReport(
        total_lesion_volume_mm3=total,
        n_regions=len(regions),
        regions=regions,
        lesion_label=lesion_label,
    )
