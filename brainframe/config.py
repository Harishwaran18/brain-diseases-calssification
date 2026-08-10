"""Typed configuration loading from YAML.

Config is represented as nested dataclasses. Each stage exposes its own dataclass; the
top-level :class:`BrainFrameConfig` aggregates them. ``load_config`` reads the master
``default.yaml`` and the stage configs it references, validating types and applying
sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from brainframe.utils.logging import get_logger

log = get_logger(__name__)

# Canonical label indices shared across segmentation/reconstruction/evaluation.
LABELS: dict[str, int] = {
    "background": 0,
    "gray_matter": 1,
    "white_matter": 2,
    "csf": 3,
    "lesion": 4,
}
LABEL_NAMES = list(LABELS.keys())


@dataclass
class PathsConfig:
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "data/checkpoints"
    output_dir: str = "data/outputs"


@dataclass
class SAMConfig:
    model_type: str = "vit_b"
    checkpoint_dir: str = "data/checkpoints"
    auto_download: bool = True
    fallback_to_mock: bool = True
    checkpoint: str | None = None


@dataclass
class DatasetsConfig:
    oasis_root: str = "data/raw/oasis"
    oasis_modality: str = "T1"
    brats_root: str = "data/raw/brats"
    brats_modality: str = "T1ce"


@dataclass
class PipelineConfig:
    stages: list[str] = field(default_factory=lambda: ["segment", "reconstruct", "evaluate"])
    cache: bool = True
    cache_dir: str = "data/outputs/cache"


@dataclass
class ClassificationModelConfig:
    name: str = "densenet3d"
    in_channels: int = 1
    num_classes: int = 2
    spatial_dims: int = 3
    pretrained: bool = False


@dataclass
class ClassificationTrainConfig:
    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    amp: bool = True
    early_stopping_patience: int = 8
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    checkpoint: str = "data/outputs/classification_best.pt"


@dataclass
class ClassificationDataConfig:
    dataset: str = "oasis"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    stratify: bool = True
    patch_size: tuple[int, int, int] = (96, 96, 96)
    augment: bool = True


@dataclass
class ClassificationConfig:
    model: ClassificationModelConfig = field(default_factory=ClassificationModelConfig)
    train: ClassificationTrainConfig = field(default_factory=ClassificationTrainConfig)
    data: ClassificationDataConfig = field(default_factory=ClassificationDataConfig)
    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "f1", "auc", "sensitivity", "specificity"]
    )


@dataclass
class SegmentationInferenceConfig:
    slice_axis: int = 2
    multimask_output: bool = True
    mask_threshold: float = 0.5
    label_classes: list[str] = field(
        default_factory=lambda: ["background", "gray_matter", "white_matter", "csf", "lesion"]
    )
    label_strategy: str = "intensity_ranking"


@dataclass
class SegmentationPromptsConfig:
    auto: bool = True
    strategy: str = "grid"
    grid_spacing: int = 64
    max_points: int = 12
    use_bbox: bool = True
    bbox_from: str = "intensity"


@dataclass
class SegmentationTTAConfig:
    enabled: bool = False
    steps: int = 3
    lr: float = 1.0e-5
    ema_decay: float = 0.999
    signals: list[str] = field(
        default_factory=lambda: ["entropy", "iou_uncertainty", "dual_scale_consistency"]
    )
    adapt_params: list[str] = field(default_factory=lambda: ["iou_token"])


@dataclass
class SegmentationPostprocessConfig:
    largest_cc: bool = True
    fill_holes: bool = True
    smoothing_sigma: float = 1.0
    min_voxel_count: int = 50


@dataclass
class SegmentationConfig:
    sam_model_type: str = "vit_b"
    sam_checkpoint: str | None = None
    inference: SegmentationInferenceConfig = field(default_factory=SegmentationInferenceConfig)
    prompts: SegmentationPromptsConfig = field(default_factory=SegmentationPromptsConfig)
    tta: SegmentationTTAConfig = field(default_factory=SegmentationTTAConfig)
    postprocess: SegmentationPostprocessConfig = field(
        default_factory=SegmentationPostprocessConfig
    )


@dataclass
class ReconstructionStackingConfig:
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    interpolation: str = "linear"
    fill_gaps: bool = True
    gap_method: str = "morphological"


@dataclass
class ReconstructionMarchingConfig:
    method: str = "marching_cubes"
    level: float = 0.5
    step_size: int = 1
    smoothing_enabled: bool = True
    smoothing_iterations: int = 3
    smoothing_lambda: float = 0.5
    per_label: bool = True


@dataclass
class ReconstructionConfig:
    stacking: ReconstructionStackingConfig = field(default_factory=ReconstructionStackingConfig)
    marching: ReconstructionMarchingConfig = field(default_factory=ReconstructionMarchingConfig)
    labels_of_interest: list[str] = field(
        default_factory=lambda: ["gray_matter", "white_matter", "csf", "lesion"]
    )
    reference_region: str = "white_matter"
    viz_engine: str = "plotly"
    viz_save_html: bool = True


@dataclass
class EvaluationLesionConfig:
    detect: bool = True
    lesion_label: str = "lesion"
    min_volume_mm3: float = 50.0
    adjacency_structures: list[str] = field(
        default_factory=lambda: ["white_matter", "gray_matter", "csf"]
    )


@dataclass
class EvaluationTherapyConfig:
    target_label: str = "lesion"
    target_mode: str = "centroid"
    radius_mm: float = 10.0
    dose: float = 1.0
    mode: str = "regeneration"
    kernel: str = "gaussian"
    sigma_mm: float = 5.0


@dataclass
class EvaluationSimulatorConfig:
    propagation_steps: int = 10
    diffusion_rate: float = 0.15
    region_graph_k: int = 6


@dataclass
class EvaluationCompatibilityConfig:
    coverage_weight: float = 0.5
    risk_weight: float = 0.3
    recovery_weight: float = 0.2
    risk_threshold_mm: float = 5.0


@dataclass
class EvaluationConfig:
    lesion_analysis: EvaluationLesionConfig = field(default_factory=EvaluationLesionConfig)
    therapy: EvaluationTherapyConfig = field(default_factory=EvaluationTherapyConfig)
    simulator: EvaluationSimulatorConfig = field(default_factory=EvaluationSimulatorConfig)
    compatibility: EvaluationCompatibilityConfig = field(
        default_factory=EvaluationCompatibilityConfig
    )
    report_formats: list[str] = field(default_factory=lambda: ["json", "html"])
    output_dir: str = "data/outputs/evaluation"


@dataclass
class BrainFrameConfig:
    seed: int = 42
    device: str = "auto"
    log_level: str = "INFO"
    paths: PathsConfig = field(default_factory=PathsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
            return tuple(value)
        return tuple(_to_tuple(v) for v in value)
    return value


def _patch_size(cfg: dict[str, Any]) -> tuple[int, int, int]:
    ps = cfg.get("patch_size", [96, 96, 96])
    return tuple(int(x) for x in ps)  # type: ignore[return-value]


def _build_classification(cfg: dict[str, Any]) -> ClassificationConfig:
    model_cfg = ClassificationModelConfig(
        name=cfg.get("model", {}).get("name", "densenet3d"),
        in_channels=cfg.get("model", {}).get("in_channels", 1),
        num_classes=cfg.get("model", {}).get("num_classes", 2),
        spatial_dims=cfg.get("model", {}).get("spatial_dims", 3),
        pretrained=cfg.get("model", {}).get("pretrained", False),
    )
    train_raw = cfg.get("train", {})
    train_cfg = ClassificationTrainConfig(
        epochs=int(train_raw.get("epochs", 50)),
        batch_size=int(train_raw.get("batch_size", 2)),
        learning_rate=float(train_raw.get("learning_rate", 1e-4)),
        weight_decay=float(train_raw.get("weight_decay", 1e-5)),
        optimizer=train_raw.get("optimizer", "adamw"),
        scheduler=train_raw.get("scheduler", "cosine"),
        amp=bool(train_raw.get("amp", True)),
        early_stopping_patience=int(train_raw.get("early_stopping", {}).get("patience", 8)),
        early_stopping_metric=train_raw.get("early_stopping", {}).get("metric", "val_loss"),
        early_stopping_mode=train_raw.get("early_stopping", {}).get("mode", "min"),
        checkpoint=train_raw.get("checkpoint", "data/outputs/classification_best.pt"),
    )
    data_raw = cfg.get("data", {})
    split = data_raw.get("split", {})
    data_cfg = ClassificationDataConfig(
        dataset=data_raw.get("dataset", "oasis"),
        train_split=float(split.get("train", 0.7)),
        val_split=float(split.get("val", 0.15)),
        test_split=float(split.get("test", 0.15)),
        stratify=bool(split.get("stratify", True)),
        patch_size=_patch_size(data_raw),
        augment=bool(data_raw.get("augment", True)),
    )
    metrics = cfg.get("metrics", {}).get("track", ["accuracy", "f1", "auc"])
    return ClassificationConfig(model=model_cfg, train=train_cfg, data=data_cfg, metrics=metrics)


def _build_segmentation(cfg: dict[str, Any]) -> SegmentationConfig:
    sam_raw = cfg.get("sam", {})
    inf_raw = cfg.get("inference", {})
    la = inf_raw.get("label_assignment", {})
    prompts_raw = cfg.get("prompts", {})
    tta_raw = cfg.get("tta", {})
    pp_raw = cfg.get("postprocess", {})
    return SegmentationConfig(
        sam_model_type=sam_raw.get("model_type", "vit_b"),
        sam_checkpoint=sam_raw.get("checkpoint"),
        inference=SegmentationInferenceConfig(
            slice_axis=int(inf_raw.get("slice_axis", 2)),
            multimask_output=bool(inf_raw.get("multimask_output", True)),
            mask_threshold=float(inf_raw.get("mask_threshold", 0.5)),
            label_classes=la.get("classes", LABEL_NAMES),
            label_strategy=la.get("strategy", "intensity_ranking"),
        ),
        prompts=SegmentationPromptsConfig(
            auto=bool(prompts_raw.get("auto", True)),
            strategy=prompts_raw.get("strategy", "grid"),
            grid_spacing=int(prompts_raw.get("grid_spacing", 64)),
            max_points=int(prompts_raw.get("max_points", 12)),
            use_bbox=bool(prompts_raw.get("use_bbox", True)),
            bbox_from=prompts_raw.get("bbox_from", "intensity"),
        ),
        tta=SegmentationTTAConfig(
            enabled=bool(tta_raw.get("enabled", False)),
            steps=int(tta_raw.get("steps", 3)),
            lr=float(tta_raw.get("lr", 1e-5)),
            ema_decay=float(tta_raw.get("ema_decay", 0.999)),
            signals=tta_raw.get(
                "signals", ["entropy", "iou_uncertainty", "dual_scale_consistency"]
            ),
            adapt_params=tta_raw.get("adapt_params", ["iou_token"]),
        ),
        postprocess=SegmentationPostprocessConfig(
            largest_cc=bool(pp_raw.get("largest_cc", True)),
            fill_holes=bool(pp_raw.get("fill_holes", True)),
            smoothing_sigma=float(pp_raw.get("smoothing_sigma", 1.0)),
            min_voxel_count=int(pp_raw.get("min_voxel_count", 50)),
        ),
    )


def _build_reconstruction(cfg: dict[str, Any]) -> ReconstructionConfig:
    st = cfg.get("stacking", {})
    mc = cfg.get("marching", {})
    metrics = cfg.get("metrics", {})
    viz = cfg.get("visualize", {})
    return ReconstructionConfig(
        stacking=ReconstructionStackingConfig(
            spacing=tuple(float(x) for x in st.get("spacing", [1.0, 1.0, 1.0])),
            interpolation=st.get("interpolation", "linear"),
            fill_gaps=bool(st.get("fill_gaps", True)),
            gap_method=st.get("gap_method", "morphological"),
        ),
        marching=ReconstructionMarchingConfig(
            method=mc.get("method", "marching_cubes"),
            level=float(mc.get("level", 0.5)),
            step_size=int(mc.get("step_size", 1)),
            smoothing_enabled=bool(mc.get("smoothing", {}).get("enabled", True)),
            smoothing_iterations=int(mc.get("smoothing", {}).get("iterations", 3)),
            smoothing_lambda=float(mc.get("smoothing", {}).get("lambda", 0.5)),
            per_label=bool(mc.get("per_label", True)),
        ),
        labels_of_interest=metrics.get(
            "labels_of_interest", ["gray_matter", "white_matter", "csf", "lesion"]
        ),
        reference_region=metrics.get("reference_region", "white_matter"),
        viz_engine=viz.get("engine", "plotly"),
        viz_save_html=bool(viz.get("save_html", True)),
    )


def _build_evaluation(cfg: dict[str, Any]) -> EvaluationConfig:
    la = cfg.get("lesion_analysis", {})
    th = cfg.get("therapy", {})
    sim = cfg.get("simulator", {})
    comp = cfg.get("compatibility", {})
    rep = cfg.get("report", {})
    return EvaluationConfig(
        lesion_analysis=EvaluationLesionConfig(
            detect=bool(la.get("detect", True)),
            lesion_label=la.get("lesion_label", "lesion"),
            min_volume_mm3=float(la.get("min_volume_mm3", 50.0)),
            adjacency_structures=la.get(
                "adjacency_structures", ["white_matter", "gray_matter", "csf"]
            ),
        ),
        therapy=EvaluationTherapyConfig(
            target_label=th.get("target_label", "lesion"),
            target_mode=th.get("target_mode", "centroid"),
            radius_mm=float(th.get("radius_mm", 10.0)),
            dose=float(th.get("dose", 1.0)),
            mode=th.get("mode", "regeneration"),
            kernel=th.get("kernel", "gaussian"),
            sigma_mm=float(th.get("sigma_mm", 5.0)),
        ),
        simulator=EvaluationSimulatorConfig(
            propagation_steps=int(sim.get("propagation_steps", 10)),
            diffusion_rate=float(sim.get("diffusion_rate", 0.15)),
            region_graph_k=int(sim.get("region_graph_k", 6)),
        ),
        compatibility=EvaluationCompatibilityConfig(
            coverage_weight=float(comp.get("coverage_weight", 0.5)),
            risk_weight=float(comp.get("risk_weight", 0.3)),
            recovery_weight=float(comp.get("recovery_weight", 0.2)),
            risk_threshold_mm=float(comp.get("risk_threshold_mm", 5.0)),
        ),
        report_formats=rep.get("format", ["json", "html"]),
        output_dir=rep.get("output_dir", "data/outputs/evaluation"),
    )


def _stage_path(base_dir: Path, ref: str) -> Path:
    """Resolve a stage-config reference.

    ``ref`` may be given relative to the master config's directory or relative to the
    repository root (the parent of ``configs/``). We return whichever exists.
    """
    p = Path(ref)
    if p.is_absolute():
        return p
    candidates = [base_dir / p, base_dir.parent / p]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[-1]  # fall back to repo-root-relative for the warning


def load_config(config_path: str | Path = "configs/default.yaml") -> BrainFrameConfig:
    """Load the master config and the referenced stage configs."""
    base_dir = Path(config_path).resolve().parent
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}

    cfg = BrainFrameConfig(
        seed=int(raw.get("seed", 42)),
        device=raw.get("device", "auto"),
        log_level=raw.get("log_level", "INFO"),
    )

    p_raw = raw.get("paths", {})
    cfg.paths = PathsConfig(
        data_dir=p_raw.get("data_dir", "data"),
        raw_dir=p_raw.get("raw_dir", "data/raw"),
        processed_dir=p_raw.get("processed_dir", "data/processed"),
        checkpoint_dir=p_raw.get("checkpoint_dir", "data/checkpoints"),
        output_dir=p_raw.get("output_dir", "data/outputs"),
    )

    pp = raw.get("pipeline", {})
    cfg.pipeline = PipelineConfig(
        stages=pp.get("stages", ["segment", "reconstruct", "evaluate"]),
        cache=bool(pp.get("cache", True)),
        cache_dir=pp.get("cache_dir", "data/outputs/cache"),
    )

    sam_raw = raw.get("sam", {})
    cfg.sam = SAMConfig(
        model_type=sam_raw.get("model_type", "vit_b"),
        checkpoint_dir=sam_raw.get("checkpoint_dir", "data/checkpoints"),
        auto_download=bool(sam_raw.get("auto_download", True)),
        fallback_to_mock=bool(sam_raw.get("fallback_to_mock", True)),
        checkpoint=sam_raw.get("checkpoint"),
    )

    ds_raw = raw.get("datasets", {})
    oasis = ds_raw.get("oasis", {})
    brats = ds_raw.get("brats", {})
    cfg.datasets = DatasetsConfig(
        oasis_root=oasis.get("root", "data/raw/oasis"),
        oasis_modality=oasis.get("modality", "T1"),
        brats_root=brats.get("root", "data/raw/brats"),
        brats_modality=brats.get("modality", "T1ce"),
    )

    stages = raw.get("stages", {})
    stage_files = {
        "classification": ("classification", ClassificationConfig, _build_classification),
        "segmentation": ("segmentation", SegmentationConfig, _build_segmentation),
        "reconstruction": ("reconstruction", ReconstructionConfig, _build_reconstruction),
        "evaluation": ("evaluation", EvaluationConfig, _build_evaluation),
    }
    for key, (_, _, builder) in stage_files.items():
        ref = stages.get(key, f"configs/{key}.yaml")
        sfile = _stage_path(base_dir, ref)
        if sfile.exists():
            sraw = yaml.safe_load(sfile.read_text(encoding="utf-8")) or {}
            top = sraw.get(key, sraw)
            setattr(cfg, key, builder(top))
        else:
            log.warning("Stage config not found: %s; using defaults", sfile)

    return cfg


def default_config() -> BrainFrameConfig:
    """Return a config built entirely from defaults (no files)."""
    return BrainFrameConfig()


def override(cfg: BrainFrameConfig, **changes: Any) -> BrainFrameConfig:
    """Return a copy of ``cfg`` with top-level fields replaced by ``changes``."""
    return replace(cfg, **changes)


# Supported device literal for type-checkers.
DeviceLiteral = Literal["auto", "cuda", "mps", "cpu"]
