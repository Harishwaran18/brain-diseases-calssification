"""Unified single-page HTML report.

Combines 3D reconstruction, disease classification, lesion analysis, therapy
simulation, morphometric dashboards, and cross-sections into ONE self-contained
HTML file with an interactive (plotly) 3D brain viewer and tabbed sections.

All figures/images are inlined as base64 so the file is fully portable.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from brainframe.reconstruction.marching import MeshData, MeshResult
from brainframe.utils.logging import get_logger

log = get_logger("reporting.unified_report")

# Medically-inspired tissue palette (name -> (color, opacity, show_edges))
# Realistic cortical colors based on anatomical references.
TISSUE_STYLE: dict[str, tuple[str, float, bool]] = {
    "cortex": ("#b5651d", 0.85, False),  # real pial surface: warm cortical tan
    "gray_matter": ("#c9a96e", 0.30, False),  # cortical gray matter shell
    "white_matter": ("#efe8d8", 0.45, False),  # pale white matter core
    "csf": ("#3aa6e6", 0.18, False),  # cerebrospinal fluid (ventricles)
    "lesion": ("#ff2b4a", 0.95, True),  # hyper-intense lesion (red, outlined)
    "background": ("#222222", 0.0, False),
}
DISEASE_LABELS: dict[int, str] = {
    0: "Healthy / No neurodegeneration detected",
    1: "Early-stage neurodegenerative changes",
    2: "Moderate neurodegeneration (e.g., early AD/PD spectrum)",
    3: "Advanced neurodegeneration (severe atrophy / lesion burden)",
}


def _b64_file(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()


def _safe_float(v: Any, nd: int = 2) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _lesion_mesh(mesh_result: MeshResult) -> MeshData | None:
    for m in mesh_result.meshes:
        if m.label == "lesion" and len(m.vertices) > 0 and len(m.faces) > 0:
            return m
    return None


def _decimate_mesh(
    vertices: np.ndarray, faces: np.ndarray, target_verts: int = 4000
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a mesh to ~target_verts by uniform vertex subsampling + face filter."""
    if len(vertices) <= target_verts:
        return vertices, faces
    step = max(1, len(vertices) // target_verts)
    keep_idx = np.arange(0, len(vertices), step)
    idx_map = -np.ones(len(vertices), dtype=np.int64)
    idx_map[keep_idx] = np.arange(len(keep_idx))
    valid = (idx_map[faces[:, 0]] >= 0) & (idx_map[faces[:, 1]] >= 0) & (idx_map[faces[:, 2]] >= 0)
    new_faces = idx_map[faces[valid]]
    return vertices[keep_idx], new_faces


def _shrink_mesh(vertices: np.ndarray, centroid: np.ndarray, frac: float) -> np.ndarray:
    """Scale vertices toward the lesion centroid by (1 - frac) — models cure."""
    return centroid + (vertices - centroid) * (1.0 - frac)


def _align_to_volume(cortex_verts: np.ndarray, vol_pts: np.ndarray | None) -> np.ndarray:
    """Transform cortex MNI-mm vertices into the volume mesh coordinate space.

    The fsaverage pial surface lives in MNI millimetres (centered near origin),
    while the Marching-Cubes tissue meshes are in voxel-index space. This maps
    the cortex bounding box onto the volume bounding box so the realistic folded
    cortex overlays the segmented tissues.
    """
    if vol_pts is None or len(cortex_verts) == 0:
        return cortex_verts - cortex_verts.mean(axis=0)
    v_min, v_max = vol_pts.min(axis=0), vol_pts.max(axis=0)
    c_min, c_max = cortex_verts.min(axis=0), cortex_verts.max(axis=0)
    c_span = np.where(c_max - c_min < 1e-6, 1.0, c_max - c_min)
    scale = (v_max - v_min) / c_span
    # Uniform scale to preserve cortical fold proportions (use mean axis scale).
    scale = np.full(3, float(scale.mean()))
    shifted = (cortex_verts - c_min) * scale + v_min
    return shifted


def _medicine_verdict(recovery_frac: float, risk: float) -> dict:
    """Return a precise medicine-efficacy verdict from lesion reduction + risk."""
    if recovery_frac >= 0.20 and risk < 0.85:
        verdict, color, icon = "MEDICINE WORKING", "#3fb950", "✅"
    elif recovery_frac < 0.05:
        verdict, color, icon = "MEDICINE INEFFECTIVE", "#f85149", "⛔"
    else:
        verdict, color, icon = "PARTIAL RESPONSE", "#d29922", "⚠️"
    return {
        "verdict": verdict,
        "color": color,
        "icon": icon,
        "reduction_pct": recovery_frac * 100.0,
        "risk": risk,
    }


def build_3d_figure(
    mesh_result: MeshResult,
    label_volume: np.ndarray | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lesion_regions: list[dict] | None = None,
    simulation: dict | None = None,
    cortex_mesh: MeshData | None = None,
) -> Any:
    """Build an animated plotly 3D figure: multi-tissue brain + live cure frames.

    The lesion mesh shrinks across animation frames (medicine taking effect),
    with a play button + slider so the cure plays live in the viewer. Meshes are
    decimated to keep the self-contained HTML small and responsive. When a real
    fsaverage cortical surface is provided it is rendered with smooth shading as
    the realistic brain backdrop.
    """
    import plotly.graph_objects as go

    sim = simulation or {}
    before_v = float(sim.get("before_lesion_volume_mm3", 0.0) or 0.0)
    after_v = float(sim.get("after_lesion_volume_mm3", 0.0) or 0.0)
    recovery_frac = ((before_v - after_v) / before_v) if before_v > 0 else 0.0

    base_traces: list[Any] = []

    # Compute the segmented-volume mesh bounding box so the (MNI-mm) cortex can be
    # aligned into the same coordinate space as the voxel-index tissue meshes.
    vol_pts = (
        np.concatenate([m.vertices for m in mesh_result.meshes if len(m.vertices) > 0], axis=0)
        if any(len(m.vertices) > 0 for m in mesh_result.meshes)
        else None
    )

    # Real fsaverage pial cortex: render with smooth shading as the realistic
    # brain backdrop. The pial surface already carries genuine folded gyri/sulci;
    # to make the viewer look more anatomically lifelike we (a) keep more vertices
    # so the gyral detail is preserved, (b) split the mesh at the midline and
    # shade the two hemispheres with subtly different tissue tones so left/right
    # are visually distinguishable, and (c) use a two-light setup (key + rim) for
    # depth cueing.
    if cortex_mesh is not None and len(cortex_mesh.vertices) > 0 and len(cortex_mesh.faces) > 0:
        cv, cf = _decimate_mesh(cortex_mesh.vertices, cortex_mesh.faces, target_verts=24000)
        cv = _align_to_volume(cv, vol_pts)
        # Split at the midline (x mean) for hemisphere-tinted shading.
        left_mask = cv[:, 0] < float(cv[:, 0].mean())
        for hemi_mask, tint, hemi_name in (
            (left_mask, "#c98a4b", "Left cortex"),
            (~left_mask, "#c0793a", "Right cortex"),
        ):
            idx = np.where(hemi_mask)[0]
            if len(idx) < 3:
                continue
            # Remap faces whose three vertices all belong to this hemisphere.
            keep = np.isin(cf, idx).all(axis=1)
            if not np.any(keep):
                continue
            sub_f = cf[keep]
            remap = {old: new for new, old in enumerate(idx)}
            sub_f = np.vectorize(remap.get)(sub_f)
            sub_v = cv[idx]
            base_traces.append(
                go.Mesh3d(
                    x=sub_v[:, 0],
                    y=sub_v[:, 1],
                    z=sub_v[:, 2],
                    i=sub_f[:, 0],
                    j=sub_f[:, 1],
                    k=sub_f[:, 2],
                    color=tint,
                    opacity=0.95,
                    name=hemi_name,
                    showlegend=True,
                    flatshading=False,
                    lighting={
                        "ambient": 0.42,
                        "diffuse": 0.88,
                        "specular": 0.45,
                        "roughness": 0.45,
                        "fresnel": 0.28,
                    },
                    lightposition={"x": 150, "y": 250, "z": 200},
                    hovertemplate=f"<b>{hemi_name}</b><br>fsaverage pial<extra></extra>",
                )
            )

    ordered = sorted(
        mesh_result.meshes,
        key=lambda m: 0 if m.label == "lesion" else (1 if m.label == "white_matter" else 2),
    )
    lesion_mesh = None
    for m in ordered:
        if m.label == "lesion":
            lesion_mesh = m
            continue
        color, opacity, _edges = TISSUE_STYLE.get(m.label, ("#aaaaaa", 0.5, False))
        if len(m.vertices) == 0 or len(m.faces) == 0:
            continue
        v, f = _decimate_mesh(m.vertices, m.faces, target_verts=3000)
        base_traces.append(
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=color,
                opacity=opacity,
                name=m.label.replace("_", " ").title(),
                showlegend=True,
                flatshading=False,
                lighting={"ambient": 0.5, "diffuse": 0.8, "specular": 0.3, "roughness": 0.6},
                hovertemplate=f"<b>{m.label}</b><extra></extra>",
            )
        )

    if label_volume is not None and label_volume.size > 0:
        try:
            mid = label_volume.shape[2] // 2
            sl = label_volume[:, :, mid].astype(np.float32)
            H, W = sl.shape
            sx, sy, _ = spacing
            yy, xx = np.mgrid[0:H, 0:W]
            ss = max(1, H // 48)
            base_traces.append(
                go.Surface(
                    x=xx[::ss, ::ss] * sx,
                    y=yy[::ss, ::ss] * sy,
                    z=np.full(sl[::ss, ::ss].shape, mid * spacing[2], dtype=np.float32),
                    surfacecolor=sl[::ss, ::ss],
                    colorscale="Greys",
                    showscale=False,
                    opacity=0.22,
                    name="axial slice",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        except Exception as e:  # pragma: no cover - viz-only
            log.debug("slice plane skipped: %s", e)

    if lesion_regions:
        cx, cy, cz, txt = [], [], [], []
        for r in lesion_regions:
            c = r.get("centroid")
            if not c or len(c) < 3:
                continue
            cx.append(c[0] * spacing[0])
            cy.append(c[1] * spacing[1])
            cz.append(c[2] * spacing[2])
            txt.append(f"Region {r.get('region_id')}: {_safe_float(r.get('volume_mm3'), 0)} mm³")
        if cx:
            base_traces.append(
                go.Scatter3d(
                    x=cx,
                    y=cy,
                    z=cz,
                    mode="markers+text",
                    marker={
                        "size": 12,
                        "color": "#ff2b4a",
                        "symbol": "diamond",
                        "line": {"color": "#fff", "width": 2},
                    },
                    text=[t.split(":")[0] for t in txt],
                    textposition="top center",
                    name="lesion sites",
                    hovertext=txt,
                    hoverinfo="text",
                    showlegend=True,
                )
            )

    n_frames = 9
    cure_steps = np.linspace(0.0, recovery_frac, n_frames)
    frames: list[go.Frame] = []
    lesion_centroid = np.array([0.0, 0.0, 0.0])
    les_v = les_f = None
    if lesion_mesh is not None and len(lesion_mesh.vertices) > 0:
        les_v, les_f = _decimate_mesh(lesion_mesh.vertices, lesion_mesh.faces, target_verts=2500)
        lesion_centroid = les_v.mean(axis=0)

    def _lesion_trace(verts: np.ndarray, faces: np.ndarray, cur_v: float) -> go.Mesh3d:
        return go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="#ff2b4a",
            opacity=0.92,
            name="lesion",
            showlegend=True,
            flatshading=True,
            lighting={"ambient": 0.6, "diffuse": 0.9, "specular": 0.2, "roughness": 0.4},
            contour={"show": True, "color": "#ffb3bf", "width": 2},
            hovertemplate=f"<b>lesion</b><br>{_safe_float(cur_v, 0)} mm³<extra></extra>",
        )

    for fi, frac in enumerate(cure_steps):
        cur_v = before_v * (1.0 - frac)
        if les_v is not None:
            verts = _shrink_mesh(les_v, lesion_centroid, frac)
            data = [_lesion_trace(verts, les_f, cur_v)]
        else:
            data = []
        label = (
            "Before medicine"
            if fi == 0
            else (
                "Cured"
                if fi == n_frames - 1 and recovery_frac > 0
                else f"Cure {(fi / (n_frames - 1)) * 100:.0f}%"
            )
        )
        frames.append(
            go.Frame(
                data=data,
                name=str(fi),
                traces=[len(base_traces)],
                layout={
                    "title": {
                        "text": f"{label} · lesion {_safe_float(cur_v, 0)} mm³",
                        "font": {"color": "#e6edf3"},
                    }
                },
            )
        )

    base_lesion = frames[0].data[0] if frames and frames[0].data else None
    fig = go.Figure(
        data=base_traces + ([base_lesion] if base_lesion is not None else []),
        frames=frames if frames else None,
    )

    steps = [
        {
            "args": [[str(i)], {"frame": {"duration": 400, "redraw": True}, "mode": "immediate"}],
            "label": f"{int(cure_steps[i] * 100)}%",
            "method": "animate",
        }
        for i in range(n_frames)
    ]
    sliders = [
        {
            "active": 0,
            "pad": {"t": 40},
            "len": 0.7,
            "x": 0.15,
            "y": 0.02,
            "steps": steps,
            "currentvalue": {
                "visible": True,
                "prefix": "Cure progress: ",
                "font": {"color": "#3fb950"},
            },
        }
    ]
    play = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {"frame": {"duration": 400, "redraw": True}, "fromcurrent": True},
                    ],
                    "label": "▶ Play cure",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                    ],
                    "label": "⏸ Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 40},
            "showactive": False,
            "type": "buttons",
            "x": 0.05,
            "xanchor": "right",
            "y": 0.02,
            "yanchor": "bottom",
        }
    ]

    fig.update_layout(
        scene={
            "aspectmode": "data",
            "xaxis": {
                "title": "L-R (mm)",
                "backgroundcolor": "#0d1117",
                "showgrid": True,
                "gridcolor": "#1f2937",
            },
            "yaxis": {
                "title": "A-P (mm)",
                "backgroundcolor": "#0d1117",
                "showgrid": True,
                "gridcolor": "#1f2937",
            },
            "zaxis": {
                "title": "I-S (mm)",
                "backgroundcolor": "#0d1117",
                "showgrid": True,
                "gridcolor": "#1f2937",
            },
        },
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font={"color": "#e6edf3", "family": "Inter, system-ui, sans-serif"},
        legend={"bgcolor": "rgba(13,17,23,0.0)", "font": {"color": "#e6edf3"}},
        margin={"l": 0, "r": 0, "t": 50, "b": 60},
        height=600,
        sliders=sliders,
        updatemenus=play,
        title={
            "text": f"Before medicine · lesion {_safe_float(before_v, 0)} mm³",
            "font": {"color": "#e6edf3"},
        },
    )
    return fig


def _fig_to_html_div(fig: Any, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def _classification_section(classification: dict | None) -> tuple[str, str]:
    if not classification:
        return (
            '<div class="empty-state">Classification stage was not run for this subject. '
            "Run <code>brainframe classify</code> to generate a disease prediction.</div>"
        ), ""
    pred = classification.get("prediction", 0)
    probs = classification.get("probabilities", [])
    label = DISEASE_LABELS.get(pred, f"Class {pred}")
    conf = max(probs) if probs else 0.0
    risk = (
        "Low" if pred == 0 else ("Moderate" if pred <= 1 else ("High" if pred == 2 else "Critical"))
    )
    risk_color = {
        "Low": "#3fb950",
        "Moderate": "#d29922",
        "High": "#db6d28",
        "Critical": "#f85149",
    }[risk]
    body = f"""
    <div class="pred-hero" style="border-color:{risk_color}">
      <div class="pred-class">{label}</div>
      <div class="pred-meta">
        <span class="badge" style="background:{risk_color}">{risk} risk</span>
        <span>Predicted class: <b>{pred}</b></span>
        <span>Confidence: <b>{_safe_float(conf, 3)}</b></span>
        <span>Classes: {classification.get("num_classes", len(probs))}</span>
      </div>
    </div>"""
    div = ""
    if probs:
        import plotly.graph_objects as go

        bar_colors = ["#3fb950" if i != pred else risk_color for i in range(len(probs))]
        fig = go.Figure(
            go.Bar(
                x=[f"Class {i}" for i in range(len(probs))],
                y=probs,
                marker_color=bar_colors,
                text=[f"{p:.3f}" for p in probs],
                textposition="outside",
            )
        )
        fig.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font={"color": "#e6edf3"},
            yaxis={"range": [0, 1], "title": "Probability"},
            xaxis={"title": "Disease class"},
            margin={"l": 40, "r": 20, "t": 20, "b": 40},
            height=320,
        )
        div = _fig_to_html_div(fig, "clf-chart")
    return body, div


def _lesion_section(lesion: dict) -> tuple[str, str]:
    total = lesion.get("total_lesion_volume_mm3", 0.0)
    n = lesion.get("n_regions", 0)
    regions = lesion.get("regions", [])
    if n == 0:
        return (
            '<div class="empty-state">No lesion voxels detected in this subject. '
            "Tissue segmentation found no hyper-intense outliers above the detection threshold.</div>"
        ), ""
    rows = ""
    for r in regions:
        adj = ", ".join(
            f"{k}: {_safe_float(v, 1)}mm" for k, v in (r.get("adjacent_structures") or {}).items()
        )
        c = r.get("centroid", (0, 0, 0))
        ext = r.get("spatial_extent", (0, 0, 0))
        rows += (
            f"<tr><td>{r.get('region_id')}</td><td>{_safe_float(r.get('volume_mm3'), 0)}</td>"
            f"<td>({_safe_float(c[0])}, {_safe_float(c[1])}, {_safe_float(c[2])})</td>"
            f"<td>{ext[0]}×{ext[1]}×{ext[2]}</td><td>{adj}</td></tr>"
        )
    body = f"""
    <div class="stat-row">
      <div class="stat-card"><div class="stat-val">{_safe_float(total, 0)}</div><div class="stat-lbl">Total lesion volume (mm³)</div></div>
      <div class="stat-card"><div class="stat-val">{n}</div><div class="stat-lbl">Detected regions</div></div>
    </div>
    <table class="data-table"><thead><tr>
      <th>Region</th><th>Volume (mm³)</th><th>Centroid (vox)</th><th>Extent (vox)</th><th>Adjacent structures</th>
    </tr></thead><tbody>{rows}</tbody></table>"""
    div = ""
    if regions:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Bar(
                x=[f"R{r.get('region_id')}" for r in regions],
                y=[r.get("volume_mm3", 0) for r in regions],
                marker_color="#ff2b4a",
                text=[f"{r.get('volume_mm3', 0):.0f}" for r in regions],
                textposition="outside",
            )
        )
        fig.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font={"color": "#e6edf3"},
            yaxis={"title": "Volume (mm³)"},
            xaxis={"title": "Lesion region"},
            margin={"l": 40, "r": 20, "t": 20, "b": 40},
            height=300,
        )
        div = _fig_to_html_div(fig, "lesion-chart")
    return body, div


def _therapy_section(therapy: dict, sim: dict, comp: dict, figures: dict) -> str:
    before = sim.get("before_lesion_volume_mm3", 0.0)
    after = sim.get("after_lesion_volume_mm3", 0.0)
    delta = before - after
    pct = (delta / before * 100) if before else 0.0
    verdict = _medicine_verdict(pct / 100.0, float(comp.get("risk", 1.0) or 1.0))
    verdict_box = f"""
    <div class="verdict-box" style="border-color:{verdict["color"]}">
      <div class="verdict-icon">{verdict["icon"]}</div>
      <div>
        <div class="verdict-text" style="color:{verdict["color"]}">{verdict["verdict"]}</div>
        <div class="verdict-sub">Lesion volume reduced by <b>{_safe_float(verdict["reduction_pct"], 1)}%</b> · tissue risk {_safe_float(verdict["risk"], 3)}</div>
      </div>
    </div>"""
    ba_img = ""
    if figures.get("before_after"):
        ba_img = f'<img class="fig-img" src="data:image/png;base64,{_b64_file(figures["before_after"])}"/>'
    lm_img = ""
    if figures.get("lesion_map"):
        lm_img = (
            f'<img class="fig-img" src="data:image/png;base64,{_b64_file(figures["lesion_map"])}"/>'
        )
    return f"""
    {verdict_box}
    <div class="stat-row">
      <div class="stat-card"><div class="stat-val">{_safe_float(before, 0)}</div><div class="stat-lbl">Before (mm³)</div></div>
      <div class="stat-card"><div class="stat-val">{_safe_float(after, 0)}</div><div class="stat-lbl">After (mm³)</div></div>
      <div class="stat-card"><div class="stat-val">{_safe_float(delta, 0)}</div><div class="stat-lbl">Δ Reduction (mm³)</div></div>
      <div class="stat-card"><div class="stat-val">{_safe_float(pct, 1)}%</div><div class="stat-lbl">Volume change</div></div>
    </div>
    <div class="therapy-params">
      <h4>Therapy specification</h4>
      <table class="data-table"><tbody>
        <tr><th>Mode</th><td>{therapy.get("mode")}</td><th>Target</th><td>{therapy.get("target_label")} ({therapy.get("target_mode")})</td></tr>
        <tr><th>Radius</th><td>{therapy.get("radius_mm")} mm</td><th>Dose</th><td>{therapy.get("dose")}</td></tr>
        <tr><th>Kernel</th><td>{therapy.get("kernel")} (σ={therapy.get("sigma_mm")} mm)</td><th>Propagated</th><td>{sim.get("propagated")}</td></tr>
        <tr><th>Affected voxels</th><td>{sim.get("affected_voxels")} ({_safe_float(sim.get("affected_fraction", 0) * 100, 1)}%)</td><th>Compatibility</th><td>{_safe_float(comp.get("score"), 3)}</td></tr>
      </tbody></table>
    </div>
    <div class="fig-grid">{ba_img}{lm_img}</div>"""


def _metrics_section(recon_metrics: dict, comp: dict, figures: dict) -> tuple[str, str]:
    per_label = recon_metrics.get("per_label", {})
    atrophy = recon_metrics.get("atrophy_ratios", {})
    rows = ""
    for name, m in per_label.items():
        rows += (
            f"<tr><td>{name}</td><td>{_safe_float(m.get('volume_mm3'), 1)}</td>"
            f"<td>{_safe_float(m.get('surface_area_mm2'), 1)}</td>"
            f"<td>{_safe_float(m.get('compactness'), 3)}</td>"
            f"<td>{m.get('n_vertices')}</td><td>{m.get('n_faces')}</td></tr>"
        )
    body = f"""
    <div class="stat-row">
      <div class="stat-card"><div class="stat-val">{_safe_float(recon_metrics.get("total_volume_mm3"), 0)}</div><div class="stat-lbl">Total volume (mm³)</div></div>
      <div class="stat-card"><div class="stat-val">{_safe_float(comp.get("coverage"), 3)}</div><div class="stat-lbl">Coverage</div></div>
      <div class="stat-card"><div class="stat-val">{_safe_float(comp.get("recovery"), 3)}</div><div class="stat-lbl">Recovery</div></div>
      <div class="stat-card"><div class="stat-val">{_safe_float(comp.get("risk"), 3)}</div><div class="stat-lbl">Risk</div></div>
    </div>
    <table class="data-table"><thead><tr>
      <th>Structure</th><th>Volume (mm³)</th><th>Surface (mm²)</th><th>Compactness</th><th>Vertices</th><th>Faces</th>
    </tr></thead><tbody>{rows}</tbody></table>"""
    divs = ""
    if atrophy:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Bar(
                x=list(atrophy.keys()),
                y=list(atrophy.values()),
                marker_color="#3aa6e6",
            )
        )
        fig.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font={"color": "#e6edf3"},
            yaxis={"title": "Ratio vs WM"},
            margin={"l": 40, "r": 20, "t": 20, "b": 40},
            height=280,
        )
        divs += _fig_to_html_div(fig, "atrophy-chart")
    if figures.get("metric_bar"):
        body += (
            '<div class="fig-grid"><img class="fig-img" '
            f'src="data:image/png;base64,{_b64_file(figures["metric_bar"])}"/></div>'
        )
    return body, divs


def _cross_section_section(figures: dict) -> str:
    imgs = ""
    for key, title in [
        ("cross_axial", "Axial"),
        ("cross_coronal", "Coronal"),
        ("cross_sagittal", "Sagittal"),
    ]:
        p = figures.get(key)
        if p:
            imgs += (
                f'<figure class="xs-fig"><figcaption>{title}</figcaption>'
                f'<img src="data:image/png;base64,{_b64_file(p)}"/></figure>'
            )
    if not imgs:
        return '<div class="empty-state">Cross-section figures not available.</div>'
    return f'<div class="xs-grid">{imgs}</div>'


_CSS = """
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--txt:#e6edf3;--muted:#8b949e;--accent:#3aa6e6;--red:#ff2b4a;--green:#3fb950;--amber:#d29922}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font-family:Inter,system-ui,-apple-system,sans-serif;line-height:1.5}
.header{background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);border-bottom:1px solid var(--border);padding:24px 32px}
.header h1{margin:0;font-size:1.5rem;font-weight:700}
.header .sub{color:var(--muted);font-size:.85rem;margin-top:4px}
.score-badge{float:right;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 20px;text-align:center}
.score-badge .v{font-size:2rem;font-weight:800;color:var(--accent)}
.score-badge .l{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tabs{display:flex;background:var(--panel);border-bottom:1px solid var(--border);padding:0 24px;overflow-x:auto;position:sticky;top:0;z-index:10}
.tab{padding:14px 18px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;white-space:nowrap;font-size:.9rem;font-weight:500;transition:.15s}
.tab:hover{color:var(--txt)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.panel{display:none;padding:24px 32px;max-width:1200px;margin:0 auto}
.panel.active{display:block}
.stat-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.stat-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px 18px;flex:1;min-width:140px}
.stat-val{font-size:1.6rem;font-weight:700;color:var(--accent)}
.stat-lbl{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-top:2px}
.data-table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden;margin:12px 0;font-size:.85rem}
.data-table th,.data-table td{padding:9px 12px;border-bottom:1px solid var(--border);text-align:left}
.data-table th{background:#1c2330;color:var(--muted);font-weight:600;text-transform:uppercase;font-size:.7rem;letter-spacing:.04em}
.data-table tr:last-child td{border-bottom:none}
.fig-grid{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
.fig-img{max-width:48%;border:1px solid var(--border);border-radius:8px;background:var(--panel)}
.xs-grid{display:flex;gap:16px;flex-wrap:wrap}
.xs-fig{flex:1;min-width:260px;text-align:center}
.xs-fig figcaption{color:var(--muted);font-size:.8rem;margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em}
.xs-fig img{max-width:100%;border:1px solid var(--border);border-radius:8px}
.pred-hero{background:var(--panel);border:2px solid;border-radius:12px;padding:24px;margin-bottom:16px}
.pred-class{font-size:1.4rem;font-weight:700;margin-bottom:8px}
.pred-meta{display:flex;gap:16px;flex-wrap:wrap;align-items:center;color:var(--muted);font-size:.85rem}
.badge{padding:3px 10px;border-radius:12px;color:#fff;font-size:.75rem;font-weight:600}
.empty-state{background:var(--panel);border:1px dashed var(--border);border-radius:8px;padding:32px;text-align:center;color:var(--muted)}
.therapy-params{margin-top:16px}
.therapy-params h4{margin:0 0 8px;font-size:.95rem}
.verdict-box{display:flex;align-items:center;gap:16px;background:var(--panel);border:2px solid;border-radius:12px;padding:18px 24px;margin-bottom:16px}
.verdict-icon{font-size:2.2rem}
.verdict-text{font-size:1.3rem;font-weight:800;letter-spacing:.02em}
.verdict-sub{color:var(--muted);font-size:.85rem;margin-top:4px}
.footer{border-top:1px solid var(--border);padding:16px 32px;color:var(--muted);font-size:.75rem;text-align:center}
.lesion-warn{background:rgba(255,43,74,.08);border:1px solid var(--red);border-radius:8px;padding:12px 16px;margin-bottom:12px;color:#ffb3bf}
"""


def _js() -> str:
    return """
function showTab(id){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  const tab=document.querySelector(`.tab[data-target="${id}"]`);
  const panel=document.getElementById(id);
  if(tab)tab.classList.add('active');
  if(panel)panel.classList.add('active');
  if(window.Plotly){document.querySelectorAll('.js-plotly-plot').forEach(el=>{try{Plotly.Plots.resize(el)}catch(e){}});}
}
window.addEventListener('resize',()=>{if(window.Plotly){document.querySelectorAll('.js-plotly-plot').forEach(el=>{try{Plotly.Plots.resize(el)}catch(e){}});}});
"""


def generate_unified_report(
    *,
    mesh_result: MeshResult | None = None,
    recon_metrics: dict | None = None,
    lesion_report: dict | None = None,
    simulation: dict | None = None,
    compatibility: dict | None = None,
    therapy: dict | None = None,
    classification: dict | None = None,
    label_volume: np.ndarray | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    figures: dict[str, str] | None = None,
    subject: str = "N/A",
    out_path: str | Path | None = None,
    cortex_mesh: MeshData | None = None,
) -> str:
    """Build a single self-contained HTML page combining all pipeline outputs."""
    figures = figures or {}
    lesion_report = lesion_report or {}
    simulation = simulation or {}
    compatibility = compatibility or {}
    therapy = therapy or {}
    recon_metrics = recon_metrics or {}
    lesion_regions = lesion_report.get("regions", [])
    score = compatibility.get("score", 0.0)

    three_div = ""
    if mesh_result is not None:
        try:
            fig3d = build_3d_figure(
                mesh_result,
                label_volume,
                spacing,
                lesion_regions,
                simulation=simulation,
                cortex_mesh=cortex_mesh,
            )
            three_div = _fig_to_html_div(fig3d, "brain-3d")
        except Exception as e:  # pragma: no cover
            log.warning("3D figure build failed: %s", e)
            three_div = f'<div class="empty-state">3D rendering unavailable: {e}</div>'

    clf_body, clf_div = _classification_section(classification)
    les_body, les_div = _lesion_section(lesion_report)
    ther_body = _therapy_section(therapy, simulation, compatibility, figures)
    met_body, met_div = _metrics_section(recon_metrics, compatibility, figures)
    xs_body = _cross_section_section(figures)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    les_alert = ""
    if lesion_report.get("n_regions", 0) > 0:
        les_alert = (
            '<div class="lesion-warn">⚠ '
            f"{lesion_report['n_regions']} lesion region(s) detected — total volume "
            f"{_safe_float(lesion_report.get('total_lesion_volume_mm3'), 0)} mm³. "
            "See Lesion Analysis & Therapy Simulation tabs.</div>"
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BrainFrame — Neurodegenerative Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{_CSS}</style></head>
<body>
<div class="header">
  <div class="score-badge"><div class="v">{_safe_float(score, 3)}</div><div class="l">Compatibility Score</div></div>
  <h1>🧠 BrainFrame — Neurodegenerative Brain Analysis</h1>
  <div class="sub">Subject: <code>{subject}</code> &nbsp;·&nbsp; Generated {ts} &nbsp;·&nbsp; Retraining-free SAM + 3D reconstruction + computational therapy evaluation</div>
</div>
{les_alert}
<div class="tabs">
  <div class="tab active" data-target="t-3d" onclick="showTab('t-3d')">3D Reconstruction</div>
  <div class="tab" data-target="t-clf" onclick="showTab('t-clf')">Disease Prediction</div>
  <div class="tab" data-target="t-les" onclick="showTab('t-les')">Lesion Analysis</div>
  <div class="tab" data-target="t-ther" onclick="showTab('t-ther')">Therapy Simulation</div>
  <div class="tab" data-target="t-met" onclick="showTab('t-met')">Morphometrics</div>
  <div class="tab" data-target="t-xs" onclick="showTab('t-xs')">Cross-Sections</div>
</div>
<div id="t-3d" class="panel active">{three_div}</div>
<div id="t-clf" class="panel">{clf_body}{clf_div}</div>
<div id="t-les" class="panel">{les_body}{les_div}</div>
<div id="t-ther" class="panel">{ther_body}</div>
<div id="t-met" class="panel">{met_body}{met_div}</div>
<div id="t-xs" class="panel">{xs_body}</div>
<div class="footer">Generated by BrainFrame · Retraining-Free AI Framework for 3D Neurodegenerative Brain Reconstruction &amp; Computational Evaluation for Neuroregenerative Therapy</div>
<script>{_js()}</script>
</body></html>"""

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
        log.info("Wrote unified report to %s", p)
    return html
