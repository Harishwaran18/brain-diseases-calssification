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
    from brainframe.classification.diseases import disease_names

    names = disease_names()[: len(probabilities)]
    colors = ["#3fb950" if i != prediction else "#f85149" for i in range(len(probabilities))]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=probabilities,
            marker_color=colors,
            text=[f"{p:.1%}" for p in probabilities],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis={"range": [0, 1], "title": "Probability"}, **_THEME, height=360)
    fig.update_xaxes(tickangle=-30)
    return fig


def evidence_chart(scores: list[dict]) -> go.Figure:
    """Grouped bar chart of the per-axis evidence scores for the top diseases."""
    top = scores[:5]
    names = [s["short_name"] for s in top]
    axes = ("region", "pattern", "laterality", "size")
    colors = {"region": "#3fb950", "pattern": "#d29922", "laterality": "#a371f7", "size": "#f85149"}
    fig = go.Figure()
    for ax in axes:
        fig.add_trace(
            go.Bar(
                name=ax.capitalize(),
                x=names,
                y=[s[f"{ax}_score"] for s in top],
                marker_color=colors[ax],
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0, 1], "title": "Evidence agreement"},
        legend={"orientation": "h", "y": -0.25},
        **_THEME,
        height=360,
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def differential_chart(differential: list[dict]) -> go.Figure:
    """Horizontal bar chart of the top-3 differential diagnosis probabilities."""
    names = [d["short_name"] for d in differential]
    probs = [d["probability"] for d in differential]
    fig = go.Figure(
        go.Bar(
            x=probs,
            y=names,
            orientation="h",
            marker_color=["#f85149", "#d29922", "#3fb950"][: len(names)],
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis={"range": [0, 1], "title": "Relative probability"},
        **_THEME,
        height=260,
    )
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


def confusion_matrix_chart(cm: dict) -> go.Figure:
    """Heatmap of the multi-class confusion matrix (rows=true, cols=predicted)."""
    import numpy as np

    matrix = np.array(cm["matrix"], dtype=float)
    names = cm.get("class_names") or [str(i) for i in range(len(matrix))]
    # Shorten names for axis ticks.
    short = [n if len(n) <= 10 else n[:9] + "…" for n in names]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=short,
            y=short,
            colorscale=[[0, "#0d1117"], [0.5, "#1f6feb"], [1, "#ff7b72"]],
            colorbar={"title": "count"},
            hovertemplate="true: %{y}<br>pred: %{x}<br>count: %{z}<extra></extra>",
            text=matrix.astype(int),
            texttemplate="%{text}",
        )
    )
    layout = _theme_layout()
    fig.update_layout(
        title="Confusion matrix (rows = true disease, columns = predicted)",
        xaxis={"title": "Predicted", "tickangle": -45, "scaleanchor": "y"},
        yaxis={"title": "True", "autorange": "reversed"},
        margin={"l": 60, "r": 20, "t": 50, "b": 80},
        **layout,
        height=560,
    )
    return fig


def f_test_chart(ft: dict) -> go.Figure:
    """Horizontal bar chart of per-feature ANOVA F-statistics (-log10 p)."""
    import numpy as np

    names = ft["feature_names"]
    f_vals = ft["f_statistic"]
    p_vals = ft["p_value"]
    sig = ft["significant"]
    neg_log_p = [-np.log10(max(p, 1e-300)) for p in p_vals]
    colors = ["#3fb950" if s else "#8b949e" for s in sig]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=f_vals,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"F={f:.1f}<br>p={p:.2g}" for f, p in zip(f_vals, p_vals, strict=False)],
            textposition="outside",
            hovertemplate="%{y}<br>F=%{x:.2f}<br>-log10p={customdata}<extra></extra>",
            customdata=neg_log_p,
        )
    )
    layout = _theme_layout()
    fig.update_layout(
        title="F-test (one-way ANOVA): feature discriminative power",
        xaxis={"title": "F-statistic (higher = more discriminative)"},
        legend={"orientation": "h", "y": -0.15},
        margin={"l": 160, "r": 20, "t": 50, "b": 60},
        **layout,
        height=max(360, 26 * len(names)),
    )
    return fig


def chi_square_chart(results: list[dict]) -> go.Figure:
    """Bar chart of chi-square statistics across the tested features."""
    names = [r["name"] for r in results]
    chi2 = [r["chi2"] for r in results]
    p = [r["p_value"] for r in results]
    sig = [r["significant"] for r in results]
    colors = ["#f85149" if s else "#8b949e" for s in sig]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=chi2,
            marker_color=colors,
            text=[f"χ²={c:.1f}<br>p={pv:.3g}" for c, pv in zip(chi2, p, strict=False)],
            textposition="outside",
        )
    )
    layout = _theme_layout()
    fig.update_layout(
        title="Chi-square test of independence (red = significant dependence)",
        yaxis={"title": "Chi-square statistic"},
        margin={"l": 50, "r": 20, "t": 50, "b": 90},
        **layout,
        height=360,
    )
    fig.update_xaxes(tickangle=-20)
    return fig


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


# ---------------------------------------------------------------------------
# Model-evaluation charts (advanced metrics).
# ---------------------------------------------------------------------------


def _theme_layout(**overrides: object) -> dict[str, object]:
    """Return a copy of ``_THEME`` with ``margin`` popped so callers can
    supply their own ``margin`` without a duplicate-keyword error.
    """
    layout: dict[str, object] = {**_THEME}
    layout.pop("margin", None)
    layout.update(overrides)
    return layout


def metric_tiles(tiles: list[dict[str, object]]) -> None:
    """Render a responsive grid of big coloured metric tiles.

    Each tile: ``{"label", "value", "hint"?, "color"?}``. Lays out up to 4
    per row, wrapping automatically.
    """
    n = len(tiles)
    if n == 0:
        return
    per_row = min(4, n)
    cells_html: list[str] = []
    for t in tiles:
        color = t.get("color", "#58a6ff")
        hint = t.get("hint", "")
        hint_html = f'<div style="color:#8b949e;font-size:.72rem;margin-top:6px">{hint}</div>' if hint else ""
        cells_html.append(
            f"""<div style="background:#0d1117;border:1px solid #30363d;border-top:3px solid {color};
            border-radius:10px;padding:14px 16px;text-align:center">
            <div style="color:#8b949e;font-size:.7rem;text-transform:uppercase;letter-spacing:.5px">{t["label"]}</div>
            <div style="font-size:1.7rem;font-weight:800;color:{color};margin-top:4px;line-height:1.1">{t["value"]}</div>
            {hint_html}</div>"""
        )
    st.markdown(
        f"""<div style="display:grid;grid-template-columns:repeat({per_row},1fr);gap:10px;margin:12px 0">
        {''.join(cells_html)}</div>""",
        unsafe_allow_html=True,
    )


def normalized_confusion_matrix_chart(cm: dict) -> go.Figure:
    """Row-normalised confusion matrix (recall view): each row sums to 1."""
    import numpy as np

    matrix = np.array(cm["matrix"], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm = matrix / row_sums
    names = cm.get("class_names") or [str(i) for i in range(len(matrix))]
    short = [n if len(n) <= 10 else n[:9] + "…" for n in names]
    fig = go.Figure(
        go.Heatmap(
            z=norm,
            x=short,
            y=short,
            colorscale=[[0, "#0d1117"], [0.5, "#1f6feb"], [1, "#ff7b72"]],
            colorbar={"title": "recall", "tickformat": ".0%"},
            hovertemplate="true: %{y}<br>pred: %{x}<br>recall: %{z:.1%}<extra></extra>",
            text=np.where(norm > 0.005, [f"{v:.0%}" for v in norm.ravel()], ""),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Normalized confusion matrix (row-normalized = per-class recall)",
        xaxis={"title": "Predicted", "tickangle": -45, "scaleanchor": "y"},
        yaxis={"title": "True", "autorange": "reversed"},
        margin={"l": 60, "r": 20, "t": 50, "b": 80},
        **_theme_layout(),
        height=560,
    )
    return fig


def per_class_metrics_chart(cm: dict) -> go.Figure:
    """Grouped horizontal bar of per-class precision / recall / F1 / specificity."""
    names = cm.get("class_names") or []
    n = len(names)
    short = [nm if len(nm) <= 12 else nm[:11] + "…" for nm in names]
    fig = go.Figure()
    for metric, color in [
        ("precision", "#58a6ff"), ("recall", "#3fb950"),
        ("f1", "#d29922"), ("specificity", "#bc8cff"),
    ]:
        fig.add_trace(go.Bar(
            name=metric.capitalize(), x=cm[metric], y=short, orientation="h",
            marker_color=color, hovertemplate=f"{metric}: %{{x:.1%}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        title="Per-class precision / recall / F1 / specificity",
        xaxis={"title": "score", "range": [0, 1.05], "tickformat": ".0%"},
        legend={"orientation": "h", "y": -0.08},
        margin={"l": 130, "r": 20, "t": 50, "b": 70},
        **_theme_layout(),
        height=max(420, 28 * n + 80),
    )
    return fig


def roc_curves_chart(roc: dict, max_classes: int = 12) -> go.Figure:
    """Multi-class one-vs-rest ROC curves with a diagonal reference + macro AUC."""
    names = roc["class_names"]
    aucs = roc["auc"]
    # Show the worst classes (lowest AUC) so the chart is informative.
    order = sorted(range(len(names)), key=lambda i: aucs[i])[:max_classes]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="chance",
        line={"color": "#8b949e", "dash": "dash"}, hoverinfo="skip",
    ))
    palette = _CURVE_PALETTE
    for idx, i in enumerate(order):
        fig.add_trace(go.Scatter(
            x=roc["fpr"][i], y=roc["tpr"][i], mode="lines",
            name=f"{names[i]} (AUC={aucs[i]:.3f})",
            line={"color": palette[idx % len(palette)], "width": 1.6},
            hovertemplate=f"{names[i]}<br>FPR=%{{x:.2f}} TPR=%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=f"ROC curves (one-vs-rest, macro-AUC = {roc['macro_auc']:.3f})",
        xaxis={"title": "False positive rate", "range": [0, 1]},
        yaxis={"title": "True positive rate", "range": [0, 1]},
        legend={"orientation": "v", "x": 1.02, "y": 1, "font": {"size": 10}},
        margin={"l": 50, "r": 180, "t": 50, "b": 50},
        **_theme_layout(),
        height=480,
    )
    return fig


def pr_curves_chart(pr: dict, max_classes: int = 12) -> go.Figure:
    """Multi-class one-vs-rest precision-recall curves with macro AP."""
    names = pr["class_names"]
    aps = pr["average_precision"]
    order = sorted(range(len(names)), key=lambda i: aps[i])[:max_classes]
    fig = go.Figure()
    palette = _CURVE_PALETTE
    for idx, i in enumerate(order):
        fig.add_trace(go.Scatter(
            x=pr["recall"][i], y=pr["precision"][i], mode="lines",
            name=f"{names[i]} (AP={aps[i]:.3f})",
            line={"color": palette[idx % len(palette)], "width": 1.6},
            hovertemplate=f"{names[i]}<br>recall=%{{x:.2f}} precision=%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Precision-Recall curves (one-vs-rest, macro-AP = {pr['macro_ap']:.3f})",
        xaxis={"title": "Recall", "range": [0, 1]},
        yaxis={"title": "Precision", "range": [0, 1.05]},
        legend={"orientation": "v", "x": 1.02, "y": 1, "font": {"size": 10}},
        margin={"l": 50, "r": 180, "t": 50, "b": 50},
        **_theme_layout(),
        height=480,
    )
    return fig


def calibration_chart(cal: dict) -> go.Figure:
    """Reliability diagram: confidence vs empirical accuracy + ideal-calibration line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
        line={"color": "#8b949e", "dash": "dash"}, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=cal["bin_confidences"], y=cal["bin_accuracies"], mode="lines+markers",
        name="classifier",
        line={"color": "#58a6ff", "width": 2},
        marker={"size": 9, "color": "#58a6ff"},
        customdata=cal["bin_counts"],
        hovertemplate="confidence=%{x:.2f}<br>accuracy=%{y:.2f}<br>samples=%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Calibration / reliability diagram (ECE = {cal['ece']:.3f})",
        xaxis={"title": "Mean predicted confidence", "range": [0, 1]},
        yaxis={"title": "Empirical accuracy", "range": [0, 1]},
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        **_theme_layout(),
        height=380,
    )
    return fig


def confidence_histogram_chart(
    y_true: list[int], y_proba: list[list[float]]
) -> go.Figure:
    """Histogram of max-softmax confidence, split by correct vs incorrect."""
    import numpy as np

    y_true_arr = np.asarray(y_true)
    proba = np.asarray(y_proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = pred == y_true_arr
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=conf[correct], name="correct", marker_color="#3fb950", opacity=0.75,
        nbinsx=20, hovertemplate="confidence: %{x:.2f}<br>count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=conf[~correct], name="incorrect", marker_color="#f85149", opacity=0.75,
        nbinsx=20, hovertemplate="confidence: %{x:.2f}<br>count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        barmode="overlay",
        title="Confidence distribution (correct vs incorrect predictions)",
        xaxis={"title": "Max softmax confidence"},
        yaxis={"title": "count"},
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        **_theme_layout(),
        height=360,
    )
    return fig


def top_k_chart(topk: dict) -> go.Figure:
    """Bar chart of top-K accuracy across K values."""
    fig = go.Figure(go.Bar(
        x=[f"Top-{k}" for k in topk["k_values"]],
        y=topk["accuracies"],
        marker_color=["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f85149"][:len(topk["k_values"])],
        text=[f"{a:.1%}" for a in topk["accuracies"]],
        textposition="outside",
        hovertemplate="%{x}<br>accuracy: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title="Top-K accuracy (differential-diagnosis quality)",
        yaxis={"title": "accuracy", "range": [0, 1.05], "tickformat": ".0%"},
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        **_theme_layout(),
        height=340,
    )
    return fig


def top_confused_pairs(cm: dict, top_n: int = 15) -> list[dict[str, object]]:
    """Return the worst off-diagonal confusions as a sorted list of dicts."""
    import numpy as np

    matrix = np.array(cm["matrix"], dtype=int)
    names = cm.get("class_names") or [str(i) for i in range(len(matrix))]
    pairs: list[dict[str, object]] = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j or matrix[i, j] == 0:
                continue
            row_total = int(matrix[i].sum()) or 1
            pairs.append({
                "True": names[i],
                "Predicted": names[j],
                "Count": int(matrix[i, j]),
                "Rate": float(matrix[i, j] / row_total),
            })
    pairs.sort(key=lambda p: p["Count"], reverse=True)
    return pairs[:top_n]


_CURVE_PALETTE = [
    "#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff",
    "#ff7b72", "#79c0ff", "#56d364", "#e3b341", "#a371f7",
    "#ffa657", "#7ee787", "#ff9eb1", "#d2a8ff", "#b1f4d4",
]
