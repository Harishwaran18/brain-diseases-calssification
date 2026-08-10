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

from brainframe.reconstruction.marching import MeshResult
from brainframe.utils.logging import get_logger

log = get_logger("reporting.unified_report")

# Medically-inspired tissue palette (name -> (color, opacity, show_edges))
TISSUE_STYLE: dict[str, tuple[str, float, bool]] = {
    "gray_matter": ("#c9a96e", 0.18, False),  # cortical gold, translucent shell
    "white_matter": ("#e8e0d0", 0.55, False),  # bright core
    "csf": ("#3aa6e6", 0.22, False),  # cerebrospinal fluid blue
    "lesion": ("#ff2b4a", 0.92, True),  # pathology red, solid + edges
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


def build_3d_figure(
    mesh_result: MeshResult,
    label_volume: np.ndarray | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lesion_regions: list[dict] | None = None,
) -> Any:
    """Build an advanced plotly 3D figure with multi-tissue meshes + lesion markers."""
    import plotly.graph_objects as go

    fig = go.Figure()
    # Re-order so the lesion (most important) renders last and stays on top.
    ordered = sorted(
        mesh_result.meshes,
        key=lambda m: 0 if m.label == "lesion" else (1 if m.label == "white_matter" else 2),
    )
    for m in ordered:
        color, opacity, edges = TISSUE_STYLE.get(m.label, ("#aaaaaa", 0.5, False))
        if len(m.vertices) == 0 or len(m.faces) == 0:
            continue
        fig.add_trace(
            go.Mesh3d(
                x=m.vertices[:, 0],
                y=m.vertices[:, 1],
                z=m.vertices[:, 2],
                i=m.faces[:, 0],
                j=m.faces[:, 1],
                k=m.faces[:, 2],
                color=color,
                opacity=opacity,
                name=m.label.replace("_", " ").title(),
                showlegend=True,
                lighting={"ambient": 0.5, "diffuse": 0.8, "specular": 0.3, "roughness": 0.6},
                flatshading=(m.label == "lesion"),
                contour={"show": edges, "color": "#ff8a9a", "width": 2} if edges else None,
                hovertemplate=f"<b>{m.label}</b><br>mesh vertex<br>extra=none<extra></extra>",
            )
        )

    # Anatomical mid-slice plane (semi-transparent surface) for spatial orientation.
    if label_volume is not None and label_volume.size > 0:
        try:
            mid = label_volume.shape[2] // 2
            sl = label_volume[:, :, mid].astype(np.float32)
            H, W = sl.shape
            sx, sy, _ = spacing
            yy, xx = np.mgrid[0:H, 0:W]
            fig.add_trace(
                go.Surface(
                    x=xx * sx,
                    y=yy * sy,
                    z=np.full_like(sl, mid * spacing[2], dtype=np.float32),
                    surfacecolor=sl,
                    colorscale="Greys",
                    showscale=False,
                    opacity=0.35,
                    name="axial slice plane",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        except Exception as e:  # pragma: no cover - viz-only
            log.debug("slice plane skipped: %s", e)

    # Lesion region centroids with annotated markers.
    if lesion_regions:
        cx, cy, cz, txt, sz = [], [], [], [], []
        for r in lesion_regions:
            c = r.get("centroid")
            if not c or len(c) < 3:
                continue
            cx.append(c[0] * spacing[0])
            cy.append(c[1] * spacing[1])
            cz.append(c[2] * spacing[2])
            txt.append(f"Region {r.get('region_id')}: {_safe_float(r.get('volume_mm3'), 0)} mm³")
            sz.append(18 + min(30, (r.get("volume_mm3", 0) or 0) / 50.0))
        if cx:
            fig.add_trace(
                go.Scatter3d(
                    x=cx,
                    y=cy,
                    z=cz,
                    mode="markers+text",
                    marker={
                        "size": sz,
                        "color": "#ff2b4a",
                        "symbol": "diamond",
                        "line": {"color": "#fff", "width": 2},
                    },
                    text=[t.split(":")[0] for t in txt],
                    textposition="top center",
                    name="lesion centroids",
                    hovertext=txt,
                    hoverinfo="text",
                    showlegend=True,
                )
            )

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
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        height=560,
        title={"text": "3D Neuroanatomical Reconstruction", "font": {"color": "#e6edf3"}},
    )
    return fig


def _fig_to_html_div(fig: Any, div_id: str) -> str:
    """Embed a plotly figure as an inlined <div> (uses page-level plotly.js)."""
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def _classification_section(classification: dict | None) -> tuple[str, str]:
    """Return (section_html, plotly_div_html) for the classification tab."""
    if not classification:
        body = (
            '<div class="empty-state">Classification stage was not run for this subject. '
            "Run <code>brainframe classify</code> to generate a disease prediction.</div>"
        )
        return body, ""
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
        body = (
            '<div class="empty-state">No lesion voxels detected in this subject. '
            "Tissue segmentation found no hyper-intense outliers above the detection threshold.</div>"
        )
        return body, ""
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
    ba_img = ""
    if figures.get("before_after"):
        ba_img = f'<img class="fig-img" src="data:image/png;base64,{_b64_file(figures["before_after"])}"/>'
    lm_img = ""
    if figures.get("lesion_map"):
        lm_img = (
            f'<img class="fig-img" src="data:image/png;base64,{_b64_file(figures["lesion_map"])}"/>'
        )
    return f"""
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

    # 3D figure
    three_div = ""
    if mesh_result is not None:
        try:
            fig3d = build_3d_figure(mesh_result, label_volume, spacing, lesion_regions)
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
