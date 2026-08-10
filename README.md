# brainframe: Retraining-Free 3D Neurodegenerative Brain Reconstruction & Therapy Evaluation

A complete, production-grade PyTorch codebase implementing the research framework
**"Retraining-Free AI Framework for 3D Neurodegenerative Brain Reconstruction and
Computational Evaluation for Neuroregenerative Therapy."**

The system unifies three components into a single reproducible pipeline:

1. **Retraining-free segmentation** — the Segment Anything Model (SAM) with test-time
   adaptation, zero-shot / few-shot, *no retraining*.
2. **3D volumetric reconstruction** — stacking segmented 2D slices + Marching Cubes /
   Flying Edges isosurface extraction + volumetric metrics.
3. **Computational therapy evaluation** — lesion-region spatial analysis,
   structural-compatibility scoring, and a non-invasive therapeutic-interaction
   simulator over the reconstructed 3D volume.

A supervised **brain-disease classification** baseline (the data-hungry contrast the
research statement critiques) is retained as the entry-point module and umbrella task
that feeds the segmentation → reconstruction → evaluation pipeline.

## Architecture

```
            ┌──────────────────┐
 MRI volume │ classification    │  (supervised baseline: data-dependent contrast)
  ────────► │ (entry module)   │
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │ segmentation      │  SAM (zero/few-shot) + TTA, no retraining
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │ reconstruction    │  slice stacking → Marching Cubes → mesh + metrics
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │ evaluation        │  lesion analysis + therapy simulator + compatibility
            └──────────────────┘
```

Each stage is independently importable and testable and shares a common
[`config`](brainframe/config.py) and [`data`](brainframe/data/) core.

## Installation

Requires Python ≥ 3.10. The recommended package manager is [`uv`](https://docs.astral.sh/uv/),
but `pip` works too.

```bash
# with uv
uv pip install -e ".[torch,dev]"

# with pip
pip install -e ".[torch,dev]"
# optional visualization & SAM extras
pip install -e ".[sam,viz]"
```

Core (no torch) installs with just `pip install -e .` and is enough to run the
reconstruction/evaluation stages on CPU; the segmentation and classification stages
need the `[torch]` (and optionally `[sam]`) extras.

## Quickstart

```bash
# 1. Prepare the directory layout
python -m brainframe prepare

# 2. (optional) Download the SAM ViT-B checkpoint
python -m brainframe download-sam --model-type vit_b

# 3. Run the full pipeline on an MRI volume
python -m brainframe run --input data/raw/oasis/subject.nii.gz \
                         --output data/outputs/pipeline
```

This produces, under `data/outputs/pipeline/`:

| Artifact | Description |
|----------|-------------|
| `label_volume.npy` | per-voxel tissue label volume (0=bg, 1=GM, 2=WM, 3=CSF, 4=lesion) |
| `meshes/*.stl` | per-label isosurface meshes |
| `reconstruction_metrics.json` | volume / surface area / compactness / atrophy ratios |
| `figures/reconstruction_3d.html` | interactive 3D render (plotly) |
| `evaluation/evaluation_report.html` | before/after renders + metric bars + scores |
| `pipeline_result.json` | summary of all stages |

## Per-stage usage

```bash
# Segment only (writes label_volume.npy)
python -m brainframe segment --input subject.nii.gz --output label_volume.npy

# Reconstruct meshes + metrics from a saved label volume
python -m brainframe reconstruct --input label_volume.npy --spacing 1,1,2

# Run computational therapy evaluation on a saved label volume
python -m brainframe evaluate --input label_volume.npy

# Render 3D meshes / cross-sections
python -m brainframe visualize --input label_volume.npy

# Supervised classification: predict a single volume
python -m brainframe classify --image subject.nii.gz

# Supervised classification: train on an OASIS-1 root
python -m brainframe classify --data-root data/raw/oasis
```

## Data

The pipeline is dataset-agnostic: any NIfTI/DICOM MRI volume can be ingested. The two
public datasets referenced are:

- **OASIS-1** — cross-sectional T1 MRI, ~416 subjects (healthy vs Alzheimer's). Request
  access at <https://www.oasis-brains.org/> and place under `data/raw/oasis`.
- **BraTS** (optional) — multimodal MRI with glioma lesion annotations for validating
  the segmentation + lesion-region evaluation. See
  <https://www.med.upenn.edu/cbica/brats/> and place under `data/raw/brats`.

```bash
python -m brainframe download-data   # prints acquisition instructions
```

## Configuration

All stages are driven by YAML configs under [`configs/`](configs/):

- `default.yaml` — master config (seed, device, paths, stage references).
- `classification.yaml`, `segmentation.yaml`, `reconstruction.yaml`, `evaluation.yaml` —
  per-stage parameters.

The master config references stage configs by relative path and is parsed into typed
dataclasses in [`brainframe.config`](brainframe/config.py). Override device/seed/output
from the CLI with `--device`, `--seed`, `--output-dir`.

## Retraining-free segmentation

The segmentation component uses Meta's **Segment Anything Model (SAM)** — a
promptable, zero-shot segmenter that generalizes across domains *without fine-tuning*.
Auto-prompts (grid points / intensity peaks / gradient bounding boxes) are derived from
the image so segmentation requires no manual interaction.

Test-time adaptation ([`brainframe.segmentation.tta`](brainframe/segmentation/tta.py))
nudges a tiny set of SAM's mask-decoder affine parameters using three self-supervised
signals — entropy minimization, IoU uncertainty, and dual-scale EMA-teacher consistency
— with **no source data and no labels**, matching the "retraining-free,
context-aware inference" goal of the research statement.

When SAM is unavailable (no checkpoint / package), a deterministic
intensity-based [`HeuristicSegmenter`](brainframe/segmentation/sam_wrapper.py) takes over
so the pipeline runs end-to-end on CPU/CI.

## Computational therapy evaluation

The evaluation component ([`brainframe.evaluation`](brainframe/evaluation/)) implements a
tractable, self-contained counterpart of lesion-as-perturbation / whole-brain models
(e.g. The Virtual Brain):

- **Lesion analysis** — connected-component detection, centroid/volume/spatial extent,
  adjacency to white/gray matter and CSF.
- **Therapy model** — parameterize an intervention: target region, radius (mm), dose
  (0–1), mode (`stimulation` / `regeneration` / `lesion_reversal`), kernel
  (`gaussian` / `diffusion`).
- **Simulator** — apply a Gaussian/diffusion effect field centered on the target,
  propagate over a small region graph, and mutate tissue labels (e.g. promote CSF to
  white matter for regeneration; reverse lesion voxels for lesion-reversal).
- **Compatibility** — a weighted score in [0, 1] combining coverage of the lesion,
  recovery (volume reduction), and risk (proximity to healthy structures).

Output is a JSON + HTML report with before/after 3D renders and metric tables.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — module-by-module design.
- [`docs/research_alignment.md`](docs/research_alignment.md) — mapping each module to the
  research-statement objectives.
- [`docs/reproducibility.md`](docs/reproducibility.md) — exact commands to reproduce
  results.

## Testing

```bash
pytest -ra
```

Tests use synthetic NIfTI fixtures (no network, no real data, CPU-only) covering every
stage and an end-to-end pipeline smoke test.

## Project layout

```
brain-diseases-calssification/
├── configs/        # YAML configs (default + per-stage)
├── brainframe/      # main package
│   ├── config.py    # typed config loader
│   ├── data/        # loaders, preprocessing, datasets, samplers
│   ├── classification/  # supervised baseline
│   ├── segmentation/    # retraining-free SAM + TTA
│   ├── reconstruction/  # 3D mesh + metrics
│   ├── evaluation/      # therapy simulator + compatibility
│   ├── pipeline.py      # end-to-end orchestrator
│   └── cli.py           # `brainframe` CLI
├── tests/          # pytest suite
└── docs/            # architecture / research alignment / reproducibility
```

## License

MIT — see [`LICENSE`](LICENSE).