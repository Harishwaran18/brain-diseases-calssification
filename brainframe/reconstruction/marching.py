"""Marching Cubes / Flying Edges isosurface extraction from a label volume.

Uses :func:`skimage.measure.marching_cubes` (which implements Lewiner Marching Cubes).
``flying_edges`` is approximated by ``step_size=1`` + an optional Laplacian smoothing
pass. A separate mesh is extracted per tissue label when ``per_label`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from brainframe.config import LABEL_NAMES, LABELS, ReconstructionConfig
from brainframe.utils.logging import get_logger

log = get_logger("reconstruction.marching")


@dataclass
class MeshData:
    """A single isosurface mesh."""

    label: str
    label_idx: int
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3)
    normals: np.ndarray  # (V, 3)
    spacing: tuple[float, float, float]


@dataclass
class MeshResult:
    """Result of mesh extraction: a list of per-label meshes."""

    meshes: list[MeshData] = field(default_factory=list)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def labels(self) -> list[str]:
        return [m.label for m in self.meshes]


def _laplacian_smooth(
    vertices: np.ndarray, faces: np.ndarray, iterations: int, lam: float
) -> np.ndarray:
    """Simple Laplacian smoothing of vertices using trimesh if available."""
    if iterations <= 0 or len(vertices) == 0 or len(faces) == 0:
        return vertices
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
        trimesh.smoothing.filter_laplacian(mesh, lamb=lam, iterations=iterations)
        return np.asarray(mesh.vertices, dtype=np.float32)
    except Exception as e:  # pragma: no cover - optional dep
        log.debug("Laplacian smoothing unavailable (%s); skipping.", e)
        return vertices


def extract_meshes(
    label_volume: np.ndarray,
    cfg: ReconstructionConfig | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> MeshResult:
    """Extract a mesh per label from the label volume."""
    from skimage import measure

    if cfg is None:
        from brainframe.config import default_config

        cfg = default_config().reconstruction

    meshes: list[MeshData] = []
    target_labels = cfg.labels_of_interest if cfg.marching.per_label else LABEL_NAMES

    for name in target_labels:
        idx = LABELS.get(name, None)
        if idx is None or idx == 0:
            continue
        binary = (label_volume == idx).astype(np.float32)
        if binary.sum() < 8:  # need a few voxels for a mesh
            continue
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                binary,
                level=cfg.marching.level,
                spacing=spacing,
                step_size=cfg.marching.step_size,
            )
        except (RuntimeError, ValueError) as e:
            log.debug("marching_cubes failed for %s: %s", name, e)
            continue
        if len(verts) == 0:
            continue
        if cfg.marching.smoothing_enabled:
            verts = _laplacian_smooth(
                verts, faces, cfg.marching.smoothing_iterations, cfg.marching.smoothing_lambda
            )
        meshes.append(
            MeshData(
                label=name,
                label_idx=idx,
                vertices=verts.astype(np.float32),
                faces=faces.astype(np.int32),
                normals=normals.astype(np.float32),
                spacing=spacing,
            )
        )
        log.info("Mesh %s: %d verts, %d faces", name, len(verts), len(faces))
    return MeshResult(meshes=meshes, spacing=spacing)


def save_mesh(mesh: MeshData, path: str | Path) -> Path:
    """Save a single mesh as STL/PLY/GLB via trimesh."""
    import trimesh

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    trimesh.exchange.export.export_mesh(tm, str(p))
    return p


def save_meshes(mesh_result: MeshResult, out_dir: str | Path) -> list[Path]:
    """Save all meshes in ``mesh_result`` to ``out_dir`` as STL files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for m in mesh_result.meshes:
        p = out / f"{m.label}.stl"
        try:
            save_mesh(m, p)
            paths.append(p)
        except Exception as e:  # pragma: no cover - export specific
            log.warning("Failed to save mesh %s: %s", m.label, e)
    return paths
