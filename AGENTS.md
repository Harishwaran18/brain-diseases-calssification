# AGENTS.md ŌĆö brainframe project memory

## Project
`brain-diseases-calssification` (repo name typo preserved) ŌĆö a retraining-free 3D
neurodegenerative brain reconstruction & therapy evaluation framework. Package import
name is `brainframe`. CLI entry: `python -m brainframe <subcommand>`.

## Disease Taxonomy (36 classes)
Classes 0-35: Healthy, AD, PD, MS, Glioma, Stroke, Epilepsy, HD, ALS, TBI,
Meningioma, Metastasis, MCA_Infarct, SDH, NPH, CJD, FTD, LBD, VaD, PSP, Abscess,
CBD, MSA, CADASIL, SAH, EDH, AVM, Cavernoma, Arachnoid, Colloid, Pituitary,
Schwannoma, PML, CPM, RadNecrosis, Wernicke.
23 anatomical regions, 7 patterns (focal/diffuse/symmetric/periventricular/ring_enhancing/cystic/hemorrhagic).

## Model Architecture
Deep residual MLP with self-attention (512->256->128->36, 3 residual blocks, LayerNorm, GELU).
43-dim features: volume, n_regions, density, bilateral, asymmetry, entropy, cortical/deep ratios,
pattern/laterality/region one-hot. 3-member ensemble with mixup, OneCycleLR, stratified split.
Weights: assets/models/disease_mlp.pt (~20MB). Train: scripts/train_disease_classifier.py (~35 min CPU).


## Environment
- Python 3.13, torch 2.13 CPU-only, monai 1.6, numpy 2.5, scipy, scikit-image, nibabel,
  trimesh, plotly, matplotlib.
- Install: `pip install -e ".[torch,dev]"` (core deps + torch/monai + test/lint).
- `brainframe` console script installs to `~/.local/bin` (may not be on PATH); use
  `python -m brainframe` instead.

## Architecture (3 stages + entry module)
classification (entry, supervised) -> segmentation (SAM zero-shot + TTA) ->
reconstruction (Marching Cubes + metrics) -> evaluation (lesion analysis + therapy
simulator + compatibility score). Canonical label index in `brainframe.config.LABELS`:
{background:0, gray_matter:1, white_matter:2, csf:3, lesion:4}.

## Evidence-based classifier + trained MLP (21-disease taxonomy) — updated 2026-08
- `brainframe/classification/diseases.py`: **21 diseases** (0=Healthy,1=AD,2=PD,
  3=MS,4=Glioma,5=Stroke,6=Epilepsy,7=HD,8=ALS,9=TBI,10=Meningioma,11=Metastasis,
  12=MCA_Infarct,13=SDH,14=NPH,15=CJD,16=FTD,17=LBD,18=VaD,19=PSP,20=Abscess).
  Each DiseaseSignature carries preferred_regions, pattern, laterality, size_mm3
  range, region_count, summary, references. REGIONS/PATTERNS/LATERALITIES expanded.
- `brainframe/data/atlas.py`: voxel->region labelling. `classify_pattern()` ->
  focal/diffuse/symmetric/periventricular/ring_enhancing.
- `brainframe/classification/evidence.py`: `classify(lesion_report, label_volume, shape)`
  -> EvidenceReport(prediction, confidence, disease, features, scores, differential).
  Scores 4 axes (region/pattern/laterality/size); confidence = top score + dominance
  margin bonus, capped. Honest: high confidence ONLY when all axes agree.
- `brainframe/classification/trained_model.py`: **trained 5-layer MLP ensemble**
  (DiseaseMLP: 256→128→64→32→N with BatchNorm + ReLU + Dropout, residual connection)
  over 31-dim lesion features (log_vol, n_regions, lesion_density, is_bilateral,
  pattern one-hot, laterality one-hot, multi-hot region set). Trained as a **3-member
  ensemble** (`scripts/train_disease_classifier.py`: minibatch SGD batch=64, 400 epochs,
  best-val checkpointing, cosine annealing, label smoothing 0.05) — **val acc 95.2%**
  (up from 83.3%). Weights saved to `assets/models/disease_mlp.pt` (gitignored; auto-trained
  on first use). `TrainedClassifier.predict_proba()` averages softmax across all ensemble
  members. Backward-compatible with legacy single-model checkpoints.
- `session.predict()` blends THREE engines: (1) evidence classifier (primary,
  transparent confidence), (2) trained MLP (learned probs, blended 0.55/0.45 when
  it agrees), (3) 3D CNN (optional secondary). Output dict adds `mlp_probabilities`.
- Output dict keys: prediction, confidence, disease_name, disease_short_name,
  features{pattern,laterality,dominant_region,total_volume_mm3}, evidence{scores},
  differential[{name,short_name,probability,score}], evidence_summary, probabilities,
  mlp_probabilities.

## Three.js WebGL brain viewer (replaces Plotly 3D) — updated 2026-08
- `neurocure_app/components/three_viewer.py`: genuine WebGL renderer (Three.js r169
  via importmap CDN) — the international standard for in-browser 3D brain viz
  (used by BrainBrowser / Allen Brain Atlas). **MeshPhysicalMaterial** for cortex
  (clearcoat wet meningeal sheen + sheen for subsurface pink glow), multi-light
  (hemisphere+key+rim+fill+back for subsurface scattering), OrbitControls,
  full-res fsaverage cortex split into L/R hemispheres with realistic pinkish-gray
  pial-surface tints.
- Cure animation: **smooth 60fps continuous interpolation** between keyframes
  (float `frame` advances at CURE_SPEED = NFRAMES/8s, vertex positions lerped
  between floor/ceil frames). Lesion mesh shrinks toward centroid; phase colours
  **lerp** (not snap) between phases. **Healing particle system** (120 additive-
  blended glowing motes drifting up from the lesion, opacity driven by cure
  progress). **Healing pulse wave** (expanding torus ring on each phase change).
  On-canvas overlay shows the DISEASE NAME + CURING TECHNIQUE NAME + per-frame
  lesion volume. ▶ Play cure / ↺ Reset view buttons.
- `render_three_brain(cortex_mesh, tissue_meshes, lesion_mesh, disease_name,
  technique_name, before_volume, after_volume)`.
- Pages `2_3D_Brain.py` and `5_Simulate.py` use the Three.js viewer.
- `build_3d_figure()` (unified_report.py, for the self-contained HTML report)
  still uses Plotly but now also takes disease_name/technique_name and shows them
  in the cure frame titles as a fallback renderer.

## Realistic brain rendering — updated 2026-08
- fsaverage pial cortex rebuilt at 32,770 verts / 65,532 faces (was 13k) via
  `scripts/build_real_brain_assets.py` (target_reduction=0.90, ~16k/hemisphere).
- `build_3d_figure()` splits cortex at midline -> Left/Right hemisphere traces with
  subtly different tissue tints (#c98a4b / #c0793a) for anatomical realism.
- Lighting: ambient 0.42, diffuse 0.88, specular 0.45, roughness 0.45, fresnel 0.28,
  lightposition (150,250,200). flatshading=False (smooth).

## Streamlit app (neurocure_app/) notes
- Multipage app; session state does NOT survive direct URL navigation -- must walk
  via in-page "Next" links (Upload -> 3D_Brain -> Predict -> Therapy -> Simulate).
- `state.run_step(label, fn)` runs fn under a spinner and reruns.
- Predict page: disease name + confidence (green >=90%, amber >=70%, red <70%),
  evidence breakdown chart, top-3 differential, per-class probs.
- charts.py: disease_chart (10 names), evidence_chart (grouped 4-axis bars),
  differential_chart (horizontal top-3).

## Tests
- 107 tests pass (`python -m pytest`). New: test_diseases.py, test_atlas.py,
  test_evidence.py (30 tests). Lint: `ruff check` + `ruff format` clean.

## Key conventions / gotchas
- Config: master `configs/default.yaml` references per-stage YAMLs; loaded by
  `brainframe.config.load_config()` into typed dataclasses. `_stage_path()` resolves
  relative to repo root.
- `device.py` lazily imports torch (so non-torch stages import cleanly).
- Segmentation: `HeuristicSegmenter` is the CI fallback when SAM checkpoint/package is
  absent. Candidate masks are nested quantile thresholds; `_assign_labels` builds
  concentric bands (brightest->WM, next->GM, dimmest->CSF; >3sigma outliers->lesion).
- `train.py` uses the new `torch.amp` API (not `torch.cuda.amp`).
- `postprocess.py` uses `np.unique` (not `set(array.tolist())` ŌĆö fails on 2D arrays).
- Mesh `_compactness` is clamped to [0,1] because non-manifold lesion fragments give
  spurious >1 values from the divergence-theorem volume.
- Simulator: avoid walrus operator in boolean array expressions (precedence bug).

## Testing
- `python -m pytest -q` -- 77 tests (was 36; +33 for therapy/session/app, +5 real-brain),
  all CPU-only, synthetic NIfTI fixtures in `tests/conftest.py`. No network, no real data
  (real-brain tests skip gracefully if assets absent).
- `python -m ruff check . && python -m ruff format --check .` -- must pass clean.
- mypy: `python -m mypy brainframe --ignore-missing-imports` (numpy stub syntax warning
  is a version quirk, not our code).
- AppTest (streamlit.testing.v1): `page_link` is not surfaced as an element attribute in
  streamlit 1.61; assert on `at.markdown`/`at.button` instead. `st.switch_page` raises
  under AppTest (no multipage context) ŌĆö tolerate that specific exception in tests.

## Commands
- `python -m brainframe prepare` ŌĆö create data/checkpoint dirs
- `python -m brainframe run --input <volume.nii.gz> --output <dir>` ŌĆö full pipeline
- `python -m brainframe segment|reconstruct|evaluate|visualize` ŌĆö per-stage

## NeuroCure interactive platform (Streamlit app)
- Package `neurocure_app/` (thin presentation shell) wraps the `brainframe` engine via
  `brainframe.session.Session` (typed, idempotent step API stored in st.session_state).
- Run: `python -m streamlit run neurocure_app/app.py --server.port 12000 --server.headless true`
- 6-page multipage app: app.py (home) + pages/1_Upload, 2_3D_Brain, 3_Predict,
  4_Therapy, 5_Simulate, 6_Report.
- **Real human brain data** (`brainframe/data/real_brain.py`): the demo scan uses the
  real ICBM152 T1 brain template (averaged human MRI) and the 3D viewer renders the real
  fsaverage pial cortical surface (genuine folded gyri/sulci, 13k verts) as the brain
  backdrop. Assets bundled as `assets/real_brain/{icbm152_volume.npz,fsaverage_pial.npz}`;
  build with `python scripts/build_real_brain_assets.py` (needs nilearn + fast-simplification,
  declared in the `realbrain`/`app` extras). `Session.load_real_cortex()` caches the mesh.
- 3D viewer uses `st.plotly_chart` (Plotly WebGL) with `flatshading=False` + vertex
  normals for smooth shading ŌĆö NOT stpyvista (streamlit-version incompatibility).
  `build_3d_figure` (in `brainframe.reporting.unified_report`) builds the static+animated
  figure and accepts a `cortex_mesh=` kwarg; `_align_to_volume` maps MNI-mm cortex coords
  onto the voxel-index tissue meshes. `build_cure_frames` in `brainframe.therapy.animation`
  produces the per-timestep lesion-shrinking frames.
- Therapy: `brainframe/therapy/{library,recommender,animation}.py` ŌĆö 4 curated
  techniques (disease classes 0-3), deterministic recommender, plotly cure animation.
- `Session` methods are idempotent (segment/reconstruct/predict/evaluate/recommend/
  simulate each early-return if their result is already cached); `ingest()` clears
  downstream state via `_clear_downstream()`.
- `use_container_width` deprecation warnings are cosmetic (streamlit 1.61) ŌĆö non-blocking.
- Headless test browser has no WebGL, so Plotly 3D shows a placeholder there; verify
  server-side via `fig.data` trace names + the streamlit log "Real cortex mesh" line.
- **3D viewer = Three.js WebGL** (`neurocure_app/components/three_viewer.py`): renders the
  real fsaverage cortex via Three.js r169 (CDN importmap) in a Streamlit
  `components.html` iframe. The cure animation (timeline bar segments, disease/technique
  overlay text, Play/Pause/Replay button, 2D-canvas fallback) lives in a SEPARATE plain
  `<script>` (not the `type="module"`) so it works even when the Three.js CDN import
  fails inside the sandboxed iframe. Data is bridged via `window.__NC_DATA__` /
  `window.__nc3d` (mesh-ref bag with `.ok` flag). Verify the cure-UI script logic with a
  node DOM-mock (`/tmp/mock_test.js`) since `browser_get_state` reads the iframe's static
  srcdoc, NOT the JS-mutated live DOM (so the Play button text won't reflect toggles).
  IMPORTANT: the mock must use STRICT element ids (only the real HTML ids exist, unknown
  getElementById returns null) — a permissive mock masks id/class-name mismatches that
  throw TypeErrors mid-`applyFrame`. The overlay element ids are `dv,tv,pv,mc,ds,mv`
  (NOT the class names `disease/technique/phase/mechanism/desc/meta`). To read the live
  iframe DOM, use `browser_get_content` (extracts JS-mutated HTML) — NOT `browser_get_state`.

## Statistical evaluation module — added 2026-08
- `brainframe/classification/stats.py`: classical + probabilistic stats for the trained MLP.
  **Aggregate metrics** (on ConfusionMatrix): macro/weighted precision/recall/F1, specificity,
  NPV (negative predictive value), balanced accuracy, Cohen's kappa, MCC (multiclass Gorodkin).
  **Probabilistic metrics** (require y_proba): ROC curves (OvR) + macro/weighted AUC,
  precision-recall curves + average precision, calibration/ECE, top-K accuracy, log loss,
  Brier score. `evaluate_trained_mlp` supports the 3-member ensemble checkpoint (averages
  softmax across members) and is backward-compatible with legacy single-model checkpoints.
  Dataclasses: FTestResult, ChiSquareResult, ConfusionMatrix, ROCResult,
  PRResult, CalibrationResult, TopKResult, StatsReport.
  Functions: f_test, confusion_matrix, chi_square_independence, chi_square_goodness_of_fit,
  roc_curves, pr_curves, calibration_curve, top_k_accuracy, multiclass_log_loss,
  multiclass_brier_score, evaluate_classifier (accepts optional y_proba), evaluate_trained_mlp.
- IMPORTANT: `_FEATURE_NAMES` and `_decode_categorical` are built from the live
  `diseases.PATTERNS/LATERALITIES/REGIONS` vocabularies (NOT hardcoded), so they stay in sync
  with `_FEATURE_DIM = 2 + 5 + 4 + 18 = 29`. Hardcoding region names caused a length mismatch
  (23 names vs 29 features) — keep using the live vocab.
- GOTCHA (fixed): Cohen's kappa `pe` must use element-wise dot product of 1-D marginals
  `np.dot(row, col)`, NOT `(row*col).sum()` with keepdims — that broadcasts to the full
  outer product whose sum is always total^2, making pe=1.0 and kappa=0.0.
- `neurocure_app/pages/7_Model_Evaluation.py`: 5-tab Streamlit dashboard (Overview /
  Confusion Matrix / ROC & PR / Calibration / Feature Analysis) with metric tiles at top,
  normalized confusion-matrix toggle, top confused pairs table, JSON report download.
  Stores raw y_true/y_proba in session_state for the confidence histogram.
- `neurocure_app/components/charts.py`: `confusion_matrix_chart()`, `normalized_confusion_matrix_chart()`,
  `f_test_chart()`, `chi_square_chart()`, `per_class_metrics_chart()`, `roc_curves_chart()`,
  `pr_curves_chart()`, `calibration_chart()`, `confidence_histogram_chart()`, `top_k_chart()`,
  `top_confused_pairs()`, `metric_tiles()`. GOTCHA: `_THEME` dict contains a `margin` key, so
  use `_theme_layout()` helper (pops margin) before passing explicit margin to update_layout.

## Three.js 3D viewer upgrades — added 2026-08
- `neurocure_app/components/three_viewer.py`: PBR rendering with ACESFilmic tone mapping,
  RoomEnvironment via PMREMGenerator, hemisphere light, UnrealBloomPass, clipping plane,
  auto-rotate toggle, PNG snapshot (toBlob -> Streamlit download_button), orientation
  gizmo (AxesHelper), ResizeObserver. Importmap pulls three + addons from unpkg.
  Reset button handles both cure-animation reset AND camera reset.
  Controls: Play cure / Reset / Spin / Clip / Snapshot.

## Verified end-to-end (2026-08-10)
- 119 tests pass (incl. 12 in tests/test_stats.py). Lint clean (ruff). mypy numpy-stub
  error is pre-existing environment issue (numpy 2.5 stubs vs py3.10 target), not our code.
- Live browser check: Model Evaluation page -> 473 samples, 21 classes, 83.3% accuracy,
  28/29 F-test features significant, chi-square confirms feature dependence. 3D Brain
  page -> all 5 control buttons render + Spin/Clip/Snapshot/Cure all work without errors.
