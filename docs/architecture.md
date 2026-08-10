# Architecture

This document describes the module-by-module design of `brainframe`.

## Package layout

```
brainframe/
├── __init__.py        # version
├── __main__.py        # `python -m brainframe` entry
├── config.py          # typed YAML config loader
├── cli.py             # argparse subcommands
├── pipeline.py        # end-to-end orchestrator
├── data/
│   ├── loaders.py     # NIfTI/DICOM -> LoadResult
│   ├── preprocessing.py  # normalize, resample, slice
│   ├── datasets.py    # OASIS-1 / BraTS torch Datasets
│   └── samplers.py    # 2D slice sampler
├── classification/
│   ├── models.py      # MONAI 3D CNN + 2D slice ensemble
│   ├── train.py       # supervised training loop
│   └── predict.py     # single-volume inference
├── segmentation/
│   ├── sam_wrapper.py # SAM + HeuristicSegmenter fallback
│   ├── prompts.py     # auto-prompting
│   ├── inference.py   # slice-wise segmentation + label assignment
│   ├── tta.py         # test-time adaptation
│   └── postprocess.py # CC, hole filling, smoothing
├── reconstruction/
│   ├── stacking.py    # 2D -> isotropic 3D label volume
│   ├── marching.py    # Marching Cubes mesh extraction
│   ├── mesh_metrics.py # volume/area/compactness/atrophy
│   └── visualize.py   # plotly/pyvista rendering
├── evaluation/
│   ├── lesion_analysis.py # connected-component lesion detection
│   ├── therapy_model.py   # TherapySpec parameterization
│   ├── simulator.py       # intervention application + propagation
│   ├── compatibility.py   # weighted compatibility score
│   └── report.py          # JSON + HTML report
└── utils/
    ├── io.py, logging.py, seed.py, device.py, metrics.py
```

## Config

`brainframe/config.py` parses the master `configs/default.yaml` and the per-stage
configs it references into a tree of nested dataclasses rooted at `BrainFrameConfig`.
A shared canonical label index `LABELS = {background:0, gray_matter:1,
white_matter:2, csf:3, lesion:4}` is used across segmentation, reconstruction, and
evaluation so that labels stay consistent end-to-end.

## Data core

`data/loaders.py` standardises NIfTI (via `nibabel`) and DICOM (via `pydicom`) input
into a `LoadResult(volume, affine, spacing, origin, source, header)`.
`data/preprocessing.py` provides intensity normalization, isotropic resampling, slice
extraction, and a simple brain mask. `data/samplers.py` yields 2D slices with optional
brain-mask filtering. `data/datasets.py` provides OASIS-1 and BraTS PyTorch `Dataset`
adapters; torch is imported lazily so the rest of the package works without it.

## Stage 0 — Classification (entry module)

The supervised baseline in `classification/` is the data-dependent contrast the research
statement positions against retraining-free inference. `models.py` supports three
backbones: MONAI `DenseNet121`/`ResNet` 3D networks and a 2D slice-ensemble
(`SliceEnsembleModel`) that applies a small 2D CNN per axial slice and averages logits.
A `Fallback3DCNN` is used when MONAI is missing so tests run without heavy weights.
`train.py` implements the training loop with AMP, cosine/step schedulers, early
stopping, and checkpointing; `predict.py` loads a volume, crops/pads to the patch size,
and returns logits + probabilities.

## Stage 1 — Retraining-free segmentation

`sam_wrapper.py` wraps Meta SAM (zero-shot, no retraining). When the `segment_anything`
package or checkpoint is unavailable, a deterministic `HeuristicSegmenter` produces
candidate masks via intensity quantile thresholds so the pipeline still runs on CPU/CI.
`prompts.py` derives prompts automatically (grid points, intensity peaks, gradient or
intensity bounding boxes) so segmentation needs no manual interaction. `inference.py`
segments each slice, assigns candidate masks to tissue classes by intensity ranking
(brightest band → white matter, next → gray matter, dimmest → CSF; extreme outliers →
lesion), and post-processes. `tta.py` implements test-time adaptation: entropy
minimization + IoU uncertainty + dual-scale EMA-teacher consistency, adapting only a
tiny set of mask-decoder affine parameters — no source data, no labels. `postprocess.py`
applies largest-CC, hole filling, and small-region removal (2D and 3D variants).

## Stage 2 — 3D reconstruction

`stacking.py` resamples the per-slice label volume to isotropic target spacing and fills
inter-slice gaps (morphological or linear). `marching.py` extracts a per-label
isosurface mesh via `skimage.measure.marching_cubes` with optional Laplacian smoothing
(via `trimesh`) and STL/PLY/GLB export. `mesh_metrics.py` computes per-mesh volume (via
the divergence theorem), surface area, compactness (sphericity-like, 1 = sphere), and
voxel-based atrophy ratios relative to a reference region. `visualize.py` renders
plotly (default, headless HTML) or pyvista meshes and saves cross-section PNGs.

## Stage 3 — Computational therapy evaluation

`lesion_analysis.py` detects connected lesion components, computes centroid/volume/
spatial extent, and measures adjacency (min distance in mm) to white/gray matter and
CSF. `therapy_model.py` turns the YAML `evaluation.therapy` config into a validated
`TherapySpec`. `simulator.py` applies the intervention: a Gaussian effect field centered
on the target (radius/sigma in mm, dose 0–1), propagated over a small k-NN region graph
with a diffusion update, and mutates tissue labels depending on the mode (regeneration
promotes CSF→WM; lesion_reversal recovers lesion→WM; stimulation marks only the
field). `compatibility.py` scores the plan in [0,1] from coverage, recovery, and risk.
`report.py` writes a JSON + HTML report with before/after renders, the effect-field
map, and metric bars.

## Orchestration

`pipeline.py` wires the three stages with stage caching (`label_volume.npy` is reused
across re-runs). `cli.py` exposes `prepare`, `download-sam`, `download-data`,
`classify`, `segment`, `reconstruct`, `evaluate`, `visualize`, and `run` subcommands,
all sharing `--config`, `--device`, `--output-dir`, and `--seed`.
