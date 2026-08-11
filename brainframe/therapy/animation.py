"""Per-timestep animation frames for the live 3D cure viewer.

Given a lesion mesh and a recovery fraction (how much the lesion has shrunk),
:func:`build_cure_frames` produces a list of plotly ``Frame`` objects showing
the lesion mesh shrinking toward its centroid at each timestep. These frames
are consumed by the Streamlit live-simulation page to animate the cure acting
on the real brain.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from brainframe.reconstruction.marching import MeshData
from brainframe.reporting.unified_report import _decimate_mesh, _shrink_mesh


def build_cure_frames(
    lesion_mesh: MeshData | None,
    static_traces: list[Any],
    before_volume_mm3: float,
    recovery_frac: float,
    n_frames: int = 12,
    lesion_color: str = "#ff2b4a",
) -> list[go.Frame]:
    """Build animated plotly frames of the lesion shrinking over time.

    Parameters
    ----------
    lesion_mesh
        The reconstructed lesion mesh (vertices/faces). If ``None`` a single
        empty frame list is returned.
    static_traces
        The constant brain-tissue traces (gray/white/CSF) shown behind the
        shrinking lesion on every frame.
    before_volume_mm3
        Lesion volume before the cure (mm³) -- the animation starts here.
    recovery_frac
        Fraction (0..1) of the lesion that is reversed by the therapy.
    n_frames
        Number of timesteps to render.
    """
    if lesion_mesh is None or len(lesion_mesh.vertices) == 0:
        return []

    les_v, les_f = _decimate_mesh(lesion_mesh.vertices, lesion_mesh.faces, target_verts=2500)
    centroid = les_v.mean(axis=0)
    steps = np.linspace(0.0, recovery_frac, n_frames)
    frames: list[go.Frame] = []
    lesion_index = len(static_traces)

    for fi, frac in enumerate(steps):
        cur_v = before_volume_mm3 * (1.0 - frac)
        verts = _shrink_mesh(les_v, centroid, frac)
        trace = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=les_f[:, 0],
            j=les_f[:, 1],
            k=les_f[:, 2],
            color=lesion_color,
            opacity=0.92,
            name="lesion",
            flatshading=True,
            lighting={"ambient": 0.6, "diffuse": 0.9, "specular": 0.2, "roughness": 0.4},
            contour={"show": True, "color": "#ffb3bf", "width": 2},
            hovertemplate=f"<b>lesion</b><br>t={fi}/{n_frames - 1}<br>{cur_v:.0f} mm³<extra></extra>",
        )
        ts = fi / (n_frames - 1 if n_frames > 1 else 1)
        frames.append(
            go.Frame(
                data=[trace],
                name=str(fi),
                traces=[lesion_index],
                layout={
                    "title": {
                        "text": f"t={ts:.2f} · lesion {cur_v:.0f} mm³ ({frac * 100:.0f}% reversed)",
                        "font": {"color": "#e6edf3"},
                    }
                },
            )
        )
    return frames
