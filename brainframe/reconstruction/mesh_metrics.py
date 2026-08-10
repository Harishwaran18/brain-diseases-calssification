"""Volumetric and surface morphometrics.

For each label mesh we compute volume, surface area, compactness (sphericity-like),
and per-label atrophy ratios relative to a reference region. Analytic checks for a
sphere validate the implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from brainframe.config import LABELS, ReconstructionConfig
from brainframe.reconstruction.marching import MeshResult
from brainframe.utils.logging import get_logger

log = get_logger("reconstruction.mesh_metrics")


@dataclass
class LabelMetrics:
    label: str
    volume_mm3: float
    surface_area_mm2: float
    compactness: float  # 1 = sphere
    n_vertices: int
    n_faces: int


@dataclass
class MeshMetrics:
    per_label: dict[str, LabelMetrics] = field(default_factory=dict)
    atrophy_ratios: dict[str, float] = field(default_factory=dict)
    total_volume_mm3: float = 0.0

    def to_dict(self) -> dict:
        return {
            "per_label": {k: asdict(v) for k, v in self.per_label.items()},
            "atrophy_ratios": self.atrophy_ratios,
            "total_volume_mm3": self.total_volume_mm3,
        }


def _mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed mesh via the divergence theorem."""
    if len(faces) == 0:
        return 0.0
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return float(abs(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2)))) / 6.0)


def _mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return float(0.5 * np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)))


def _compactness(volume: float, area: float) -> float:
    """Sphericity-like compactness: a sphere has compactness 1.

    Non-manifold / open meshes can yield values > 1 from the divergence-theorem volume;
    we clamp to [0, 1] so the metric stays interpretable.
    """
    if area <= 0 or volume <= 0:
        return 0.0
    return float(min(1.0, (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / area))


def compute_metrics(
    mesh_result: MeshResult,
    label_volume: np.ndarray | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cfg: ReconstructionConfig | None = None,
) -> MeshMetrics:
    """Compute per-label metrics from meshes (and voxel volume for atrophy)."""
    if cfg is None:
        from brainframe.config import default_config

        cfg = default_config().reconstruction
    out = MeshMetrics()

    for m in mesh_result.meshes:
        vol = _mesh_volume(m.vertices, m.faces)
        area = _mesh_surface_area(m.vertices, m.faces)
        comp = _compactness(vol, area)
        out.per_label[m.label] = LabelMetrics(
            label=m.label,
            volume_mm3=round(vol, 3),
            surface_area_mm2=round(area, 3),
            compactness=round(comp, 4),
            n_vertices=int(len(m.vertices)),
            n_faces=int(len(m.faces)),
        )
        out.total_volume_mm3 += vol

    # Atrophy ratios: volume relative to the reference region (voxel-based for robustness).
    ref_name = cfg.reference_region
    ref_idx = LABELS.get(ref_name, 2)
    if label_volume is not None:
        ref_voxels = int((label_volume == ref_idx).sum())
        for name, idx in LABELS.items():
            if name == "background" or name == ref_name:
                continue
            vox = int((label_volume == idx).sum())
            if ref_voxels > 0:
                out.atrophy_ratios[name] = round(vox / ref_voxels, 4)
    log.info(
        "Computed metrics for %d labels, total volume=%.1f mm^3",
        len(out.per_label),
        out.total_volume_mm3,
    )
    return out
