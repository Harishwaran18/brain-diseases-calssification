"""3D visualization: plotly (default) / pyvista rendering + cross-section PNGs.

Plotly is the default engine because it renders to a self-contained HTML file without
requiring a display server. Pyvista is used when available and a window/PNG is desired.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from brainframe.config import ReconstructionConfig
from brainframe.reconstruction.marching import MeshResult
from brainframe.utils.logging import get_logger

log = get_logger("reconstruction.visualize")


def render_3d_plotly(mesh_result: MeshResult, out_path: str | Path | None = None) -> str | None:
    """Render meshes to an interactive HTML file with plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.warning("plotly not installed; skipping 3D render.")
        return None
    fig = go.Figure()
    colors = {"gray_matter": "#888", "white_matter": "#ddd", "csf": "#4aa", "lesion": "#e44"}
    for m in mesh_result.meshes:
        fig.add_trace(
            go.Mesh3d(
                x=m.vertices[:, 0],
                y=m.vertices[:, 1],
                z=m.vertices[:, 2],
                i=m.faces[:, 0],
                j=m.faces[:, 1],
                k=m.faces[:, 2],
                color=colors.get(m.label, "#999"),
                opacity=0.6,
                name=m.label,
            )
        )
    fig.update_layout(scene={"aspectmode": "data"}, title="3D Brain Reconstruction")
    if out_path is None:
        return fig.to_html(include_plotlyjs="cdn")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    log.info("Wrote 3D HTML to %s", out_path)
    return str(out_path)


def render_3d_pyvista(mesh_result: MeshResult, out_path: str | Path | None = None) -> str | None:
    """Render meshes with pyvista (off-screen PNG if out_path given)."""
    try:
        import pyvista as pv
        import trimesh
    except ImportError:
        log.warning("pyvista not installed; skipping render.")
        return None
    plotter = pv.Plotter(off_screen=out_path is not None)
    for m in mesh_result.meshes:
        tm = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
        plotter.add_mesh(tm, color="lightblue", opacity=0.7)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(out_path))
        plotter.close()
        log.info("Wrote 3D PNG to %s", out_path)
        return str(out_path)
    plotter.show()
    return None


def render_3d(
    mesh_result: MeshResult,
    cfg: ReconstructionConfig | None = None,
    out_path: str | Path | None = None,
) -> str | None:
    """Dispatch to the configured visualization engine."""
    if cfg is None:
        from brainframe.config import default_config

        cfg = default_config().reconstruction
    if cfg.viz_engine == "pyvista":
        return render_3d_pyvista(mesh_result, out_path=out_path)
    return render_3d_plotly(mesh_result, out_path=out_path)


def save_cross_sections(
    label_volume: np.ndarray,
    out_dir: str | Path,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    axes: tuple[int, ...] = (0, 1, 2),
) -> list[Path]:
    """Save mid-slice cross-section PNGs for each axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    names = ["sagittal", "coronal", "axial"]
    for ax in axes:
        z = label_volume.shape[ax] // 2
        sl = np.take(label_volume, z, axis=ax)
        fig, axp = plt.subplots(figsize=(5, 5))
        axp.imshow(sl.T, cmap="nipy_spectral", origin="lower")
        axp.set_title(f"{names[ax]} (slice {z})")
        axp.axis("off")
        p = out / f"cross_{names[ax]}.png"
        fig.savefig(p, dpi=80, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    log.info("Saved %d cross-sections to %s", len(paths), out)
    return paths
