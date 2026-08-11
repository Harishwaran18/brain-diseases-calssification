"""Plotly chart helpers for the NeuroCure platform."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

_THEME = {
    "paper_bgcolor": "#161b22",
    "plot_bgcolor": "#161b22",
    "font": {"color": "#e6edf3"},
    "margin": {"l": 40, "r": 20, "t": 30, "b": 40},
}


def disease_chart(probabilities: list[float], prediction: int) -> go.Figure:
    """Bar chart of per-class disease probabilities."""
    colors = ["#3fb950" if i != prediction else "#f85149" for i in range(len(probabilities))]
    fig = go.Figure(
        go.Bar(
            x=[f"Class {i}" for i in range(len(probabilities))],
            y=probabilities,
            marker_color=colors,
            text=[f"{p:.3f}" for p in probabilities],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis={"range": [0, 1], "title": "Probability"}, **_THEME, height=320)
    return fig


def lesion_volume_chart(regions: list[dict]) -> go.Figure:
    """Bar chart of per-region lesion volumes."""
    if not regions:
        fig = go.Figure()
        fig.update_layout(
            **_THEME,
            height=280,
            annotations=[
                {
                    "text": "No lesions detected",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#8b949e"},
                }
            ],
        )
        return fig
    fig = go.Figure(
        go.Bar(
            x=[f"R{r.get('region_id')}" for r in regions],
            y=[r.get("volume_mm3", 0) for r in regions],
            marker_color="#ff2b4a",
            text=[f"{r.get('volume_mm3', 0):.0f}" for r in regions],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis={"title": "Volume (mm³)"}, **_THEME, height=300)
    return fig


def cure_timeline_chart(before: float, after: float, n_steps: int = 20) -> go.Figure:
    """Line chart of lesion volume over the cure timeline."""
    import numpy as np

    recovery = (before - after) / before if before else 0.0
    t = np.linspace(0, 1, n_steps)
    vol = before * (1 - recovery * t)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=vol,
            mode="lines+markers",
            line={"color": "#3fb950", "width": 3},
            name="lesion volume",
            hovertemplate="t=%{x:.2f}<br>%{y:.0f} mm³<extra></extra>",
        )
    )
    fig.add_hrect(
        y0=after,
        y1=after,
        line_dash="dash",
        line_color="#3aa6e6",
        annotation_text=f"final: {after:.0f} mm³",
        annotation_position="right",
    )
    fig.update_layout(
        xaxis={"title": "Cure progress"},
        yaxis={"title": "Lesion volume (mm³)"},
        **_THEME,
        height=320,
    )
    return fig


def render(fig: go.Figure) -> None:
    """Render a plotly figure in the app."""
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def verdict_banner(recovery_frac: float, risk: float) -> None:
    """Show a prominent WORKING / PARTIAL / INEFFECTIVE verdict."""
    if recovery_frac >= 0.20 and risk < 0.85:
        verdict, color, icon = "MEDICINE WORKING", "#3fb950", "✅"
    elif recovery_frac < 0.05:
        verdict, color, icon = "MEDICINE INEFFECTIVE", "#f85149", "⛔"
    else:
        verdict, color, icon = "PARTIAL RESPONSE", "#d29922", "⚠️"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:16px;background:#161b22;
        border:2px solid {color};border-radius:12px;padding:18px 24px;margin:16px 0">
          <div style="font-size:2.4rem">{icon}</div>
          <div>
            <div style="font-size:1.4rem;font-weight:800;color:{color}">{verdict}</div>
            <div style="color:#8b949e;font-size:.9rem;margin-top:4px">
              Lesion volume reduced by <b>{recovery_frac * 100:.1f}%</b> · tissue risk {risk:.3f}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
