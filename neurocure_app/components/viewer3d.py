"""3D brain viewer components for the NeuroCure platform.

Uses ``st.plotly_chart`` to render an interactive, rotatable/zoomable 3D brain
(Plotly WebGL) -- this avoids the stpyvista/streamlit-version incompatibility
while still delivering a genuine in-browser 3D viewer. The animated cure
figure from :mod:`brainframe.reporting.unified_report` powers the live
simulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from brainframe.reconstruction.marching import MeshResult
from brainframe.reporting.unified_report import build_3d_figure


def render_brain_3d(
    mesh_result: MeshResult,
    label_volume: np.ndarray | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lesion_regions: list[dict] | None = None,
    simulation: dict | None = None,
    cortex_mesh: Any = None,
    disease_name: str | None = None,
    technique_name: str | None = None,
    *,
    height: int = 620,
) -> None:
    """Render the interactive 3D brain with optional cure animation in-page.

    The cure-animation frame titles name the disease being treated and the
    curing technique being applied, so the viewer always shows what is happening.
    """
    try:
        fig = build_3d_figure(
            mesh_result,
            label_volume=label_volume,
            spacing=spacing,
            lesion_regions=lesion_regions,
            simulation=simulation,
            cortex_mesh=cortex_mesh,
            disease_name=disease_name,
            technique_name=technique_name,
        )
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    except Exception as e:  # noqa: BLE001
        st.error(f"3D rendering failed: {e}")


def render_axial_slice(
    volume: np.ndarray, slice_idx: int | None = None, title: str = "Axial slice"
) -> None:
    """Show a single axial slice as an inline image."""
    import matplotlib.pyplot as plt

    if volume is None or volume.size == 0:
        st.info("No volume available to slice.")
        return
    if slice_idx is None:
        slice_idx = volume.shape[2] // 2
    sl = volume[:, :, slice_idx]
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#0d1117")
    ax.imshow(sl.T, cmap="gray", origin="lower", aspect="auto")
    ax.set_title(title, color="#e6edf3", fontsize=10)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_cure_animation(
    mesh_result: MeshResult,
    static_traces: list[Any],
    lesion_mesh: Any,
    before_volume: float,
    recovery_frac: float,
    n_frames: int = 12,
) -> None:
    """Render the live curing animation (lesion shrinking over time)."""
    import plotly.graph_objects as go

    from brainframe.therapy.animation import build_cure_frames

    frames = build_cure_frames(
        lesion_mesh, static_traces, before_volume, recovery_frac, n_frames=n_frames
    )
    if not frames:
        st.info("No lesion mesh to animate.")
        return
    base = static_traces + ([frames[0].data[0]] if frames[0].data else [])
    fig = go.Figure(data=base, frames=frames)
    steps = [
        {
            "args": [[str(i)], {"frame": {"duration": 400, "redraw": True}, "mode": "immediate"}],
            "label": f"t={i}",
            "method": "animate",
        }
        for i in range(n_frames)
    ]
    fig.update_layout(
        scene={
            "aspectmode": "data",
            "xaxis": {"backgroundcolor": "#0d1117", "gridcolor": "#1f2937"},
            "yaxis": {"backgroundcolor": "#0d1117", "gridcolor": "#1f2937"},
            "zaxis": {"backgroundcolor": "#0d1117", "gridcolor": "#1f2937"},
        },
        paper_bgcolor="#0d1117",
        font={"color": "#e6edf3"},
        height=600,
        sliders=[
            {
                "active": 0,
                "pad": {"t": 40},
                "len": 0.7,
                "x": 0.15,
                "y": 0.02,
                "steps": steps,
            }
        ],
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {"frame": {"duration": 400, "redraw": True}, "fromcurrent": True},
                        ],
                        "label": "▶ Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                        ],
                        "label": "⏸",
                        "method": "animate",
                    },
                ],
                "type": "buttons",
                "direction": "left",
                "x": 0.05,
                "y": 0.02,
                "xanchor": "right",
            }
        ],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
