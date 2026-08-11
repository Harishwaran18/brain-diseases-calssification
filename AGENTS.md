# AGENTS.md — brainframe project memory

## Project
`brain-diseases-calssification` (repo name typo preserved) — a retraining-free 3D
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

## Key conventions / gotchas
- Config: master `configs/default.yaml` references per-stage YAMLs; loaded by
  `brainframe.config.load_config()` into typed dataclasses. `_stage_path()` resolves
  relative to repo root.
- `device.py` lazily imports torch (so non-torch stages import cleanly).
- Segmentation: `HeuristicSegmenter` is the CI fallback when SAM checkpoint/package is
  absent. Candidate masks are nested quantile thresholds; `_assign_labels` builds
  concentric bands (brightest->WM, next->GM, dimmest->CSF; >3sigma outliers->lesion).
- `train.py` uses the new `torch.amp` API (not `torch.cuda.amp`).
- `postprocess.py` uses `np.unique` (not `set(array.tolist())` — fails on 2D arrays).
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
  under AppTest (no multipage context) — tolerate that specific exception in tests.

## Commands
- `python -m brainframe prepare` — create data/checkpoint dirs
- `python -m brainframe run --input <volume.nii.gz> --output <dir>` — full pipeline
- `python -m brainframe segment|reconstruct|evaluate|visualize` — per-stage

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
  normals for smooth shading — NOT stpyvista (streamlit-version incompatibility).
  `build_3d_figure` (in `brainframe.reporting.unified_report`) builds the static+animated
  figure and accepts a `cortex_mesh=` kwarg; `_align_to_volume` maps MNI-mm cortex coords
  onto the voxel-index tissue meshes. `build_cure_frames` in `brainframe.therapy.animation`
  produces the per-timestep lesion-shrinking frames.
- Therapy: `brainframe/therapy/{library,recommender,animation}.py` — 4 curated
  techniques (disease classes 0-3), deterministic recommender, plotly cure animation.
- `Session` methods are idempotent (segment/reconstruct/predict/evaluate/recommend/
  simulate each early-return if their result is already cached); `ingest()` clears
  downstream state via `_clear_downstream()`.
- `use_container_width` deprecation warnings are cosmetic (streamlit 1.61) — non-blocking.
- Headless test browser has no WebGL, so Plotly 3D shows a placeholder there; verify
  server-side via `fig.data` trace names + the streamlit log "Real cortex mesh" line.
