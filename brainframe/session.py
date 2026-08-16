"""Typed end-to-end session API used by the interactive platform and CLI.

A :class:`Session` holds the evolving state of a single subject's analysis --
the ingested volume, segmentation, reconstruction, prediction, recommended
therapy, and simulation timeline -- and exposes one method per pipeline step.
Each step is idempotent: calling it again returns the cached result unless the
session is reset. This lets the Streamlit app drive the workflow page-by-page
without re-running expensive stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from brainframe.config import BrainFrameConfig, default_config, load_config
from brainframe.data.loaders import LoadResult, load_volume
from brainframe.evaluation.therapy_model import TherapySpec
from brainframe.reconstruction.marching import MeshResult
from brainframe.utils.logging import get_logger
from brainframe.utils.seed import set_seed

log = get_logger("session")


@dataclass
class Session:
    """Stateful end-to-end analysis session for one subject.

    Attributes are populated lazily as each step runs. Use :meth:`reset` to
    clear intermediate state and start over.
    """

    config: BrainFrameConfig = field(default_factory=default_config)
    device: str = "cpu"
    output_dir: Path = field(default_factory=lambda: Path("data/outputs/session"))
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Evolving state -------------------------------------------------------
    volume: np.ndarray | None = None
    volume_path: str | None = None
    load_result: LoadResult | None = None
    label_volume: np.ndarray | None = None
    reconstruction: dict | None = None
    evaluation: dict | None = None
    classification: dict | None = None
    recommendation: Any | None = None
    simulation_override: dict | None = None
    cure_timeline: dict | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_seed(self.config.seed)

    @classmethod
    def from_config_path(
        cls, config_path: str | Path = "configs/default.yaml", **kwargs: Any
    ) -> Session:
        """Create a session from a YAML config file path."""
        try:
            cfg = load_config(config_path)
        except FileNotFoundError:
            log.warning("Config %s not found; using defaults", config_path)
            cfg = default_config()
        return cls(config=cfg, **kwargs)

    # -- Step 1: ingest ----------------------------------------------------
    def ingest(self, source: str | Path | np.ndarray) -> Session:
        """Load a brain volume from a file path or accept a raw array."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(source, np.ndarray):
            self.volume = source
            self.volume_path = None
            self.spacing = (1.0, 1.0, 1.0)
            self.load_result = None
        else:
            result = load_volume(source)
            self.volume = result.volume
            self.volume_path = str(result.source)
            self.spacing = result.spacing
            self.load_result = result
        log.info("Ingested volume: shape=%s spacing=%s", self.volume.shape, self.spacing)
        self._clear_downstream()
        return self

    def load_demo_brain(self) -> Session:
        """Load the bundled demo brain (works with no upload/download).

        Prefers the **real** ICBM152 human brain template when the bundled
        assets are present; falls back to the synthetic brain phantom otherwise.
        """
        from brainframe.data.real_brain import has_real_brain, load_real_brain_volume

        if has_real_brain():
            vol, labels, spacing = load_real_brain_volume()
            # Seed the label volume so segmentation is skipped (already real).
            self.volume = vol
            self.spacing = spacing
            self.volume_path = None
            self.load_result = None
            self._clear_downstream()
            self.label_volume = labels
            log.info("Loaded REAL ICBM152 brain: shape=%s", vol.shape)
            return self
        demo = Path(__file__).resolve().parent.parent / "assets" / "demo_brain.nii.gz"
        if not demo.exists():
            # Fall back to the brain phantom generator.
            from brainframe.data.brain_phantom import generate_brain_volume

            vol, _ = generate_brain_volume(shape=(96, 128, 96), n_lesions=2, seed=7)
            return self.ingest(vol)
        return self.ingest(demo)

    def load_real_cortex(self) -> MeshResult | None:
        """Return the real fsaverage cortical surface mesh, or ``None``.

        Loaded once and cached on the session for the 3D viewer to render a
        genuine folded human brain cortex behind the segmented tissues.
        """
        from brainframe.data.real_brain import has_real_brain, load_real_cortex_mesh

        if not has_real_brain():
            return None
        if getattr(self, "_cortex_mesh", None) is None:
            self._cortex_mesh = load_real_cortex_mesh()
        return self._cortex_mesh

    # -- Step 2: segment ---------------------------------------------------
    def segment(self) -> Session:
        """Run retraining-free SAM segmentation -> tissue/lesion label volume."""
        if self.label_volume is not None:
            return self
        if self.volume is None:
            raise RuntimeError("No volume ingested; call ingest() first.")
        from brainframe.pipeline import run_segmentation

        cache = Path(self.config.pipeline.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        self.label_volume = run_segmentation(
            self.volume,
            self.config,
            cache,
            device=self.device,
            use_cache=self.config.pipeline.cache,
        )
        log.info("Segmentation complete: labels shape %s", self.label_volume.shape)
        self._clear_downstream(keep_label=True)
        return self

    # -- Step 3: reconstruct ----------------------------------------------
    def reconstruct(self) -> Session:
        """Reconstruct the 3D brain mesh + metrics from the label volume."""
        if self.reconstruction is not None:
            return self
        if self.label_volume is None:
            self.segment()
        assert self.label_volume is not None
        from brainframe.pipeline import run_reconstruction

        self.reconstruction = run_reconstruction(
            self.label_volume, self.config, self.output_dir, spacing=self.spacing
        )
        log.info("Reconstruction complete")
        self._clear_downstream(keep_label=True, keep_recon=True)
        return self

    # -- Step 4: predict ---------------------------------------------------
    def predict(self) -> Session:
        """Run disease classification on the ingested volume.

        Three complementary engines are blended:

        1. **Trained deep MLP** (primary, learned) — a deep residual network
           with self-attention trained on signature-derived data; gives a
           learned probability distribution over all 36 diseases.
        2. **Evidence-based differential classifier** (calibration, transparent)
           — scores lesion features against the 36-disease taxonomy on four
           interpretable axes (region, pattern, laterality, size) and produces
           an auditable per-axis breakdown plus a calibrated confidence.
        3. **3D CNN** (optional secondary) — when MONAI weights are available.

        The MLP provides the primary prediction and probability distribution;
        the evidence engine calibrates confidence and provides the auditable
        breakdown. When both agree, confidence is boosted; when they disagree,
        the MLP distribution is reported at reduced confidence.
        """
        if self.classification is not None:
            return self
        if self.evaluation is None:
            self.evaluate()
        assert self.evaluation is not None
        from brainframe.classification.evidence import classify

        lesion = self.evaluation["lesion"]
        lesion_dict = lesion.to_dict() if hasattr(lesion, "to_dict") else lesion
        report = classify(lesion_dict, self.label_volume, self.spacing)
        from brainframe.classification.diseases import get_disease
        features = report.features
        # Trained MLP over the same features -> learned probability vector.
        mlp_probs: list[float] = []
        try:
            from brainframe.classification.trained_model import TrainedClassifier

            clf = TrainedClassifier()
            if clf.available:
                mlp_probs = clf.predict_proba(
                    features.total_volume_mm3,
                    features.n_regions,
                    features.pattern,
                    features.laterality,
                    features.dominant_region,
                    features.cluster_regions,
                ).tolist()
        except Exception as e:  # pragma: no cover - optional
            log.debug("Trained MLP skipped: %s", e)
        # Materialise the in-memory volume to a temp NIfTI for the secondary NN.
        nn_probs: list[float] = []
        if self.volume_path is None and self.volume is not None:
            import nibabel as nib

            tmp = self.output_dir / "session_volume.nii.gz"
            affine = np.diag(list(self.spacing) + [1.0])
            nib.save(nib.Nifti1Image(self.volume.astype(np.float32), affine), str(tmp))
            self.volume_path = str(tmp)
        if self.volume_path is not None:
            try:
                from brainframe.pipeline import run_classification

                nn_out = run_classification(self.volume_path, self.config, device=self.device)
                if nn_out is not None:
                    nn_probs = nn_out.get("probabilities", [])
            except Exception as e:  # pragma: no cover - optional secondary
                log.debug("Secondary NN classifier skipped: %s", e)
        disease = report.disease
        # Build a full N-class probability vector from the evidence scores.
        order = np.argsort([s.class_id for s in report.scores])
        ordered = np.array([s.score for s in report.scores])[order]
        evi_probs = ordered / (ordered.sum() + 1e-9)
        # MLP is the primary predictor. If available, use its distribution
        # as the base and blend with evidence for calibration.
        if mlp_probs and len(mlp_probs) == len(evi_probs):
            mlp_arr = np.array(mlp_probs)
            mlp_pred = int(np.argmax(mlp_arr))
            evi_pred = report.prediction
            if mlp_pred == evi_pred:
                # Both agree: blend MLP (primary) with evidence (calibration).
                probs = 0.65 * mlp_arr + 0.35 * evi_probs
                probs = probs / probs.sum()
                prediction = mlp_pred
                # Boost confidence when both engines agree.
                confidence = min(0.99, report.confidence + 0.08)
            else:
                # Disagree: MLP takes priority but with reduced confidence.
                probs = 0.70 * mlp_arr + 0.30 * evi_probs
                probs = probs / probs.sum()
                prediction = mlp_pred
                # Lower confidence since the engines disagree.
                confidence = max(0.45, report.confidence - 0.05)
                disease = get_disease(prediction)
        else:
            # No MLP: fall back to evidence engine entirely.
            probs = evi_probs
            prediction = report.prediction
            confidence = report.confidence
        if nn_probs and len(nn_probs) == len(probs):
            nn_arr = np.array(nn_probs)
            top_nn = int(np.argmax(nn_arr))
            if top_nn == prediction:
                probs = 0.8 * probs + 0.2 * nn_arr
                probs = probs / probs.sum()
        self.classification = {
            "subject": self.volume_path or "in-memory volume",
            "prediction": prediction,
            "disease_name": disease.name,
            "disease_short_name": disease.short_name,
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "num_classes": len(probs),
            "differential": report.differential,
            "evidence": report.to_dict(),
            "evidence_summary": report.evidence_summary,
            "features": report.features.to_dict(),
            "mlp_probabilities": mlp_probs,
        }
        log.info(
            "Prediction: %s (conf %.3f)",
            disease.short_name,
            confidence,
        )
        return self

    # -- Step 5: evaluate + recommend --------------------------------------
    def evaluate(self) -> Session:
        """Run lesion analysis + therapy simulation + compatibility scoring."""
        if self.evaluation is not None:
            return self
        if self.label_volume is None:
            self.segment()
        assert self.label_volume is not None
        from brainframe.pipeline import run_evaluation

        self.evaluation = run_evaluation(
            self.label_volume, self.config, self.output_dir / "evaluation", spacing=self.spacing
        )
        log.info("Evaluation complete: score=%.3f", self.evaluation["score"])
        return self

    def recommend(self) -> Session:
        """Recommend a curing technique from the prediction + lesion analysis."""
        if self.recommendation is not None:
            return self
        if self.evaluation is None:
            self.evaluate()
        assert self.evaluation is not None
        from brainframe.therapy.recommender import recommend_therapy

        lesion = (
            self.evaluation["lesion"].to_dict()
            if hasattr(self.evaluation["lesion"], "to_dict")
            else self.evaluation["lesion"]
        )
        disease_class = 0
        confidence = 0.0
        if self.classification:
            disease_class = self.classification.get("prediction", 0)
            confidence = self.classification.get("confidence", 0.0)
        self.recommendation = recommend_therapy(
            disease_class=disease_class,
            lesion_volume_mm3=lesion.get("total_lesion_volume_mm3", 0.0),
            n_regions=lesion.get("n_regions", 0),
            confidence=confidence,
        )
        return self

    # -- Step 6: simulate with recommended technique -----------------------
    def simulate(self) -> Session:
        """Re-run the therapy simulation using the recommended technique.

        The recommended technique's parameters override the config therapy.
        """
        if self.simulation_override is not None:
            return self
        if self.label_volume is None:
            self.segment()
        if self.recommendation is None:
            self.recommend()
        assert self.label_volume is not None and self.recommendation is not None

        from brainframe.config import override
        from brainframe.evaluation.compatibility import compute_compatibility
        from brainframe.evaluation.lesion_analysis import analyze_lesions
        from brainframe.evaluation.simulator import simulate_therapy

        tech = self.recommendation.technique
        ev_cfg = override(self.config, **{}).evaluation
        # Build an evaluation config with the recommended therapy parameters.
        from dataclasses import replace

        ev_cfg = replace(ev_cfg, therapy=replace(ev_cfg.therapy, **tech.to_therapy_spec_dict()))
        lesion_report = analyze_lesions(self.label_volume, ev_cfg, spacing=self.spacing)
        sim = simulate_therapy(
            self.label_volume,
            lesion_report,
            TherapySpec(**tech.to_therapy_spec_dict()),
            ev_cfg,
            spacing=self.spacing,
        )
        comp = compute_compatibility(
            sim, lesion_report, self.label_volume, ev_cfg, spacing=self.spacing
        )
        self.evaluation = {
            "lesion": lesion_report,
            "simulation": sim,
            "compatibility": comp,
            "report": self.evaluation["report"] if self.evaluation else {},
            "score": comp.score,
            "therapy": tech.to_therapy_spec_dict(),
        }
        self.simulation_override = sim.to_dict()
        # Build the medically-accurate multi-phase cure timeline.
        disease_class = (
            self.classification["prediction"] if self.classification else 0
        )
        from brainframe.therapy.cure_phases import build_cure_timeline

        timeline = build_cure_timeline(
            disease_class,
            sim.before_lesion_volume_mm3,
            sim.after_lesion_volume_mm3,
            n_frames=36,
        )
        self.cure_timeline = timeline.to_dict()
        log.info(
            "Simulated recommended therapy '%s': before=%.1f after=%.1f mm³ (%d-phase cure)",
            tech.name,
            sim.before_lesion_volume_mm3,
            sim.after_lesion_volume_mm3,
            len(timeline.phases),
        )
        return self

    # -- Step 7: report ----------------------------------------------------
    def generate_report(self, out_path: str | Path | None = None) -> Path:
        """Generate the unified HTML + JSON report and return its path."""
        from brainframe.reporting.unified_report import generate_unified_report
        from brainframe.utils.io import save_json

        recon = self.reconstruction
        evaluation = self.evaluation
        if recon is None and evaluation is None:
            raise RuntimeError("Nothing to report: run reconstruct()/evaluate() first.")

        fig_paths: dict[str, str] = {}
        if evaluation is not None:
            fig_paths.update(evaluation.get("report", {}).get("figures", {}))
        xs_dir = self.output_dir / "figures"
        for name in ("cross_axial", "cross_coronal", "cross_sagittal"):
            p = xs_dir / f"{name}.png"
            if p.exists():
                fig_paths[name] = str(p)

        therapy = (
            self.recommendation.technique.to_therapy_spec_dict()
            if self.recommendation
            else (evaluation.get("therapy") if evaluation else {})
        )
        report_path = Path(out_path) if out_path else self.output_dir / "report.html"
        sim_dict = self.simulation_override or (
            evaluation["simulation"].to_dict() if evaluation else {}
        )
        comp_dict = evaluation["compatibility"].to_dict() if evaluation else {}
        lesion_dict = evaluation["lesion"].to_dict() if evaluation else {}
        generate_unified_report(
            mesh_result=recon["meshes"] if recon else None,
            recon_metrics=recon["metrics"] if recon else None,
            lesion_report=lesion_dict,
            simulation=sim_dict,
            compatibility=comp_dict,
            therapy=therapy,
            classification=self.classification,
            label_volume=self.label_volume,
            spacing=self.spacing,
            figures=fig_paths,
            subject=self.volume_path or "in-memory volume",
            out_path=report_path,
            cortex_mesh=getattr(self, "_cortex_mesh", None).meshes[0]
            if getattr(self, "_cortex_mesh", None)
            else None,
        )
        # JSON manifest
        manifest = {
            "subject": self.volume_path or "in-memory volume",
            "stages": self._completed_stages(),
            "classification": self.classification,
            "reconstruction": recon["metrics"] if recon else None,
            "lesion": lesion_dict,
            "simulation": sim_dict,
            "compatibility": comp_dict,
            "therapy": therapy,
            "recommendation": (self.recommendation.to_dict() if self.recommendation else None),
        }
        save_json(manifest, self.output_dir / "report_manifest.json")
        log.info("Report written to %s", report_path)
        return report_path

    # -- Helpers -----------------------------------------------------------
    def _completed_stages(self) -> list[str]:
        stages = []
        if self.volume is not None:
            stages.append("ingest")
        if self.label_volume is not None:
            stages.append("segment")
        if self.reconstruction is not None:
            stages.append("reconstruct")
        if self.classification is not None:
            stages.append("predict")
        if self.evaluation is not None:
            stages.append("evaluate")
        if self.recommendation is not None:
            stages.append("recommend")
        if self.simulation_override is not None:
            stages.append("simulate")
        return stages

    def _clear_downstream(self, *, keep_label: bool = False, keep_recon: bool = False) -> None:
        if not keep_label:
            self.label_volume = None
        if not keep_recon:
            self.reconstruction = None
        self.evaluation = None
        self.classification = None
        self.recommendation = None
        self.simulation_override = None
        self.cure_timeline = None

    def reset(self) -> Session:
        """Clear all session state."""
        self.volume = None
        self.volume_path = None
        self.load_result = None
        self._clear_downstream()
        return self

    def summary(self) -> dict:
        """Return a compact summary of the current session state."""
        rec = self.reconstruction
        ev = self.evaluation
        sim = self.simulation_override or (ev["simulation"].to_dict() if ev else {})
        comp = ev["compatibility"].to_dict() if ev else {}
        les = ev["lesion"].to_dict() if ev else {}
        return {
            "stages": self._completed_stages(),
            "volume_shape": list(self.volume.shape) if self.volume is not None else None,
            "prediction": self.classification,
            "total_volume_mm3": rec["metrics"]["total_volume_mm3"] if rec else None,
            "lesion_volume_mm3": les.get("total_lesion_volume_mm3"),
            "n_lesion_regions": les.get("n_regions"),
            "before_volume_mm3": sim.get("before_lesion_volume_mm3"),
            "after_volume_mm3": sim.get("after_lesion_volume_mm3"),
            "compatibility_score": comp.get("score"),
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
        }
