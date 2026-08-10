"""End-to-end pipeline orchestration.

Wires segmentation -> reconstruction -> evaluation with stage caching. Each stage
returns a serializable artifact (NumPy arrays / dataclasses) cached to disk so that
re-runs are idempotent and individual stages can be skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from brainframe.config import BrainFrameConfig
from brainframe.evaluation.compatibility import compute_compatibility
from brainframe.evaluation.lesion_analysis import analyze_lesions
from brainframe.evaluation.report import generate_report
from brainframe.evaluation.simulator import simulate_therapy
from brainframe.evaluation.therapy_model import build_therapy
from brainframe.reconstruction.marching import extract_meshes, save_meshes
from brainframe.reconstruction.mesh_metrics import compute_metrics
from brainframe.reconstruction.stacking import stack_slices
from brainframe.reconstruction.visualize import render_3d, save_cross_sections
from brainframe.segmentation.inference import segment_volume
from brainframe.segmentation.sam_wrapper import build_segmenter
from brainframe.utils.io import ensure_dir, save_json
from brainframe.utils.logging import get_logger
from brainframe.utils.seed import set_seed

log = get_logger("pipeline")


@dataclass
class StageArtifact:
    name: str
    path: str | None = None
    data: Any = None


@dataclass
class PipelineResult:
    stages: list[str] = field(default_factory=list)
    label_volume_path: str | None = None
    mesh_paths: list[str] = field(default_factory=list)
    metrics_path: str | None = None
    report_path: str | None = None
    metrics: dict = field(default_factory=dict)
    compatibility_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "stages": self.stages,
            "label_volume_path": self.label_volume_path,
            "mesh_paths": self.mesh_paths,
            "metrics_path": self.metrics_path,
            "report_path": self.report_path,
            "metrics": self.metrics,
            "compatibility_score": self.compatibility_score,
        }


def _save_label_volume(label_volume: np.ndarray, path: Path, spacing=(1, 1, 1)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), label_volume)
    return path


def _load_label_volume(path: Path) -> np.ndarray:
    return np.load(str(path))


def run_segmentation(
    volume: np.ndarray,
    cfg: BrainFrameConfig,
    cache_dir: Path,
    device: str = "cpu",
    use_cache: bool = True,
) -> np.ndarray:
    """Stage 1: segmentation."""
    cache = cache_dir / "label_volume.npy"
    if use_cache and cache.exists():
        log.info("Using cached segmentation: %s", cache)
        return _load_label_volume(cache)
    segmenter = build_segmenter(
        cfg.segmentation, device=device, allow_download=cfg.sam.auto_download
    )
    res = segment_volume(volume, cfg.segmentation, segmenter=segmenter)
    label_vol = res.label_volume
    if use_cache:
        _save_label_volume(label_vol, cache)
    return label_vol


def run_reconstruction(
    label_volume: np.ndarray,
    cfg: BrainFrameConfig,
    output_dir: Path,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Stage 2: reconstruction (stacking -> meshes -> metrics -> viz)."""
    recon = stack_slices(label_volume, spacing=spacing, cfg=cfg.reconstruction)
    meshes = extract_meshes(recon.label_volume, cfg=cfg.reconstruction, spacing=recon.spacing)
    metrics = compute_metrics(meshes, recon.label_volume, recon.spacing, cfg=cfg.reconstruction)
    mesh_paths = save_meshes(meshes, output_dir / "meshes")
    # Cross-sections + 3D HTML (best-effort; skip if viz unavailable)
    try:
        save_cross_sections(recon.label_volume, output_dir / "figures", spacing=recon.spacing)
    except Exception as e:  # pragma: no cover
        log.warning("Cross-section rendering failed: %s", e)
    try:
        render_3d(
            meshes,
            cfg=cfg.reconstruction,
            out_path=output_dir / "figures" / "reconstruction_3d.html",
        )
    except Exception as e:  # pragma: no cover
        log.warning("3D rendering failed: %s", e)
    metrics_path = output_dir / "reconstruction_metrics.json"
    save_json(metrics.to_dict(), metrics_path)
    return {
        "label_volume": recon.label_volume,
        "spacing": recon.spacing,
        "meshes": meshes,
        "metrics": metrics.to_dict(),
        "mesh_paths": [str(p) for p in mesh_paths],
        "metrics_path": str(metrics_path),
    }


def run_evaluation(
    label_volume: np.ndarray,
    cfg: BrainFrameConfig,
    output_dir: Path,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Stage 3: therapy evaluation."""
    lesion_report = analyze_lesions(label_volume, cfg.evaluation, spacing=spacing)
    therapy = build_therapy(cfg.evaluation)
    sim = simulate_therapy(label_volume, lesion_report, therapy, cfg.evaluation, spacing=spacing)
    comp = compute_compatibility(sim, lesion_report, label_volume, cfg.evaluation, spacing=spacing)
    report = generate_report(
        lesion_report,
        sim,
        comp,
        therapy,
        label_volume,
        cfg=cfg.evaluation,
        output_dir=output_dir,
    )
    return {
        "lesion": lesion_report,
        "simulation": sim,
        "compatibility": comp,
        "report": report,
        "score": comp.score,
    }


def run_pipeline(
    volume: np.ndarray,
    cfg: BrainFrameConfig,
    output_dir: str | Path = "data/outputs/pipeline",
    device: str = "cpu",
    stages: list[str] | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PipelineResult:
    """Run the full segmentation -> reconstruction -> evaluation pipeline."""
    set_seed(cfg.seed)
    stages = stages or cfg.pipeline.stages
    out = Path(output_dir)
    ensure_dir(out)
    cache_dir = Path(cfg.pipeline.cache_dir)
    ensure_dir(cache_dir)

    result = PipelineResult()
    label_volume = None

    if "segment" in stages:
        log.info("=== Stage: segmentation ===")
        label_volume = run_segmentation(
            volume, cfg, cache_dir, device=device, use_cache=cfg.pipeline.cache
        )
        lv_path = out / "label_volume.npy"
        _save_label_volume(label_volume, lv_path)
        result.label_volume_path = str(lv_path)
        result.stages.append("segment")

    if "reconstruct" in stages:
        log.info("=== Stage: reconstruction ===")
        if label_volume is None:
            label_volume = run_segmentation(
                volume, cfg, cache_dir, device=device, use_cache=cfg.pipeline.cache
            )
        recon = run_reconstruction(label_volume, cfg, out, spacing=spacing)
        result.mesh_paths = recon["mesh_paths"]
        result.metrics_path = recon["metrics_path"]
        result.metrics["reconstruction"] = recon["metrics"]
        result.stages.append("reconstruct")

    if "evaluate" in stages:
        log.info("=== Stage: evaluation ===")
        if label_volume is None:
            label_volume = run_segmentation(
                volume, cfg, cache_dir, device=device, use_cache=cfg.pipeline.cache
            )
        ev = run_evaluation(label_volume, cfg, out / "evaluation", spacing=spacing)
        result.report_path = ev["report"].get("figures", {}).get("html")
        result.metrics["evaluation"] = ev["report"]
        result.compatibility_score = ev["score"]
        result.stages.append("evaluate")

    save_json(result.to_dict(), out / "pipeline_result.json")
    log.info("Pipeline complete. Stages: %s", result.stages)
    return result
