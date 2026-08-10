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
- `python -m pytest -q` — 36 tests, all CPU-only, synthetic NIfTI fixtures in
  `tests/conftest.py`. No network, no real data.
- `python -m ruff check brainframe tests` — must pass clean.
- mypy: `python -m mypy brainframe --ignore-missing-imports` (numpy stub syntax warning
  is a version quirk, not our code).

## Commands
- `python -m brainframe prepare` — create data/checkpoint dirs
- `python -m brainframe run --input <volume.nii.gz> --output <dir>` — full pipeline
- `python -m brainframe segment|reconstruct|evaluate|visualize` — per-stage
