# AGENTS.md ŌĆö brainframe project memory

## Project
`brain-diseases-calssification` (repo name typo preserved) ŌĆö a retraining-free 3D
neurodegenerative brain reconstruction & therapy evaluation framework. Package import
name is `brainframe`. CLI entry: `python -m brainframe <subcommand>`.

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

## Evidence-based classifier (10-disease taxonomy) — added 2026-08
- `brainframe/classification/diseases.py`: 10 diseases (0=Healthy,1=AD,2=PD,3=MS,
  4=Glioma,5=Stroke,6=Epilepsy,7=HD,8=ALS,9=TBI). Each DiseaseSignature carries
  preferred_regions, pattern, laterality, size_mm3 range, region_count, summary, references.
- `brainframe/data/atlas.py`: voxel->region labelling (RAS: axis0=X/LR, axis1=Y/PA,
  axis2=Z/IS). `region_at(voxel_xyz, shape)` -> region str. `label_lesion_cluster()`
  -> {region, hemisphere, ...}. `classify_pattern()` -> focal/diffuse/symmetric/periventricular.
- `brainframe/classification/evidence.py`: `classify(lesion_report, label_volume, shape)`
  -> EvidenceReport(prediction, confidence, disease, features, scores, differential).
  Scores 4 axes (region/pattern/laterality/size); confidence = top score + dominance
  margin bonus, capped. Honest: high confidence ONLY when all axes agree.
  Demo MS lesion -> 95% confidence (periventricular bilateral Dawson's-finger pattern).
- `session.predict()` uses evidence classifier as PRIMARY engine; NN is secondary signal.
- Output dict keys: prediction, confidence, disease_name, disease_short_name,
  features{pattern,laterality,dominant_region,total_volume_mm3}, evidence{scores},
  differential[{name,short_name,probability,score}], evidence_summary, probabilities.

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
