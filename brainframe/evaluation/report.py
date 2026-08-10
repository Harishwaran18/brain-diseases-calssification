"""JSON + HTML evaluation report generation with before/after figures."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from brainframe.config import EvaluationConfig
from brainframe.evaluation.compatibility import CompatibilityResult
from brainframe.evaluation.lesion_analysis import LesionReport
from brainframe.evaluation.simulator import SimulationResult
from brainframe.evaluation.therapy_model import TherapySpec
from brainframe.utils.io import ensure_dir, save_json
from brainframe.utils.logging import get_logger

log = get_logger("evaluation.report")


def _save_metric_bar(compatibility: CompatibilityResult, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    labels = ["coverage", "recovery", "risk", "score"]
    vals = [compatibility.coverage, compatibility.recovery, compatibility.risk, compatibility.score]
    ax.bar(labels, vals, color=["#3a7", "#39c", "#fa3", "#555"])
    ax.set_ylim(0, 1)
    ax.set_title("Therapy compatibility metrics")
    ax.set_ylabel("score")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    return out_path


def _save_before_after(
    label_before: np.ndarray, label_after: np.ndarray, out_path: Path, axis: int = 2
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = label_before.shape[axis] // 2
    sb = np.take(label_before, z, axis=axis)
    sa = np.take(label_after, z, axis=axis)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sb.T, cmap="nipy_spectral", origin="lower")
    axes[0].set_title("Before therapy")
    axes[0].axis("off")
    axes[1].imshow(sa.T, cmap="nipy_spectral", origin="lower")
    axes[1].set_title("After therapy")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_lesion_map(
    effect_field: np.ndarray, label_volume: np.ndarray, out_path: Path, axis: int = 2
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = effect_field.shape[axis] // 2
    eff = np.take(effect_field, z, axis=axis)
    lab = np.take(label_volume, z, axis=axis)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(lab.T, cmap="nipy_spectral", origin="lower")
    axes[0].set_title("Labels")
    axes[0].axis("off")
    im2 = axes[1].imshow(eff.T, cmap="hot", origin="lower")
    axes[1].set_title("Therapy effect field")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_report(
    lesion_report: LesionReport,
    simulation: SimulationResult,
    compatibility: CompatibilityResult,
    therapy: TherapySpec,
    label_before: np.ndarray,
    cfg: EvaluationConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate JSON + HTML report and figures. Returns the report dict."""
    if cfg is None:
        from brainframe.config import default_config

        cfg = default_config().evaluation
    out_dir = Path(output_dir or cfg.output_dir)
    ensure_dir(out_dir)

    report: dict[str, Any] = {
        "therapy": asdict(therapy),
        "lesion_analysis": lesion_report.to_dict(),
        "simulation": simulation.to_dict(),
        "compatibility": compatibility.to_dict(),
    }

    json_path = out_dir / "evaluation_report.json"
    save_json(report, json_path)
    log.info("Wrote JSON report to %s", json_path)

    figures: dict[str, str] = {}
    if "json" in cfg.report_formats:
        figures["json"] = str(json_path)
    try:
        figures["metric_bar"] = str(_save_metric_bar(compatibility, out_dir / "metric_bar.png"))
        figures["before_after"] = str(
            _save_before_after(
                label_before, simulation.label_volume_after, out_dir / "before_after.png"
            )
        )
        figures["lesion_map"] = str(
            _save_lesion_map(simulation.effect_field, label_before, out_dir / "lesion_map.png")
        )
    except Exception as e:  # pragma: no cover - matplotlib/IO issues
        log.warning("Figure generation failed: %s", e)

    if "html" in cfg.report_formats:
        html = _render_html(report, figures)
        html_path = out_dir / "evaluation_report.html"
        html_path.write_text(html, encoding="utf-8")
        figures["html"] = str(html_path)
        log.info("Wrote HTML report to %s", html_path)

    report["figures"] = figures
    return report


def _render_html(report: dict, figures: dict[str, str]) -> str:
    therapy = report["therapy"]
    les = report["lesion_analysis"]
    sim = report["simulation"]
    comp = report["compatibility"]

    def _img(name: str) -> str:
        p = figures.get(name)
        if not p:
            return ""
        import base64
        from pathlib import Path as _P

        data = base64.b64encode(_P(p).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" style="max-width:48%;border:1px solid #ccc;margin:4px"/>'

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Therapy Evaluation Report</title>
<style>body{{font-family:sans-serif;margin:2em;max-width:1000px}}
.metric{{display:inline-block;width:120px;padding:6px;margin:4px;border-radius:6px;color:#fff;text-align:center}}
.bar{{width:200px;background:#eee;border-radius:6px;overflow:hidden;display:inline-block}}
.bar>div{{height:18px;background:#3a7}}</style></head><body>
<h1>Computational Therapy Evaluation Report</h1>
<h2>Therapy</h2>
<table border="0" cellpadding="4">
<tr><th>mode</th><td>{therapy["mode"]}</td></tr>
<tr><th>target</th><td>{therapy["target_label"]} ({therapy["target_mode"]})</td></tr>
<tr><th>radius (mm)</th><td>{therapy["radius_mm"]}</td></tr>
<tr><th>dose</th><td>{therapy["dose"]}</td></tr>
<tr><th>kernel</th><td>{therapy["kernel"]} (&sigma;={therapy["sigma_mm"]} mm)</td></tr>
</table>
<h2>Lesion Analysis</h2>
<p>Total lesion volume: <b>{les["total_lesion_volume_mm3"]} mm&sup3;</b> across {les["n_regions"]} region(s).</p>
<h2>Simulation</h2>
<p>Lesion volume before: <b>{sim["before_lesion_volume_mm3"]} mm&sup3;</b> &rarr; after: <b>{sim["after_lesion_volume_mm3"]} mm&sup3;</b></p>
<p>Affected voxels: {sim["affected_voxels"]} ({sim["affected_fraction"] * 100:.1f}% of volume), propagated={sim["propagated"]}</p>
<h2>Compatibility</h2>
<p><span class="metric" style="background:#3a7">coverage {comp["coverage"]:.2f}</span>
<span class="metric" style="background:#39c">recovery {comp["recovery"]:.2f}</span>
<span class="metric" style="background:#fa3">risk {comp["risk"]:.2f}</span>
<span class="metric" style="background:#555">score {comp["score"]:.2f}</span></p>
<h2>Figures</h2>
<h3>Before / After</h3>{_img("before_after")}
<h3>Lesion &amp; effect field</h3>{_img("lesion_map")}
<h3>Metric bars</h3>{_img("metric_bar")}
<hr/><small>Generated by brainframe evaluation.</small>
</body></html>"""
