# Reproducibility

This document gives the exact commands to reproduce the results and run the framework.

## Environment

```bash
git clone <repo-url> brain-diseases-calssification
cd brain-diseases-calssification
uv pip install -e ".[torch,sam,viz,dev]"
# or with pip:
# pip install -e ".[torch,sam,viz,dev]"
```

Python ≥ 3.10. CPU-only is sufficient for the synthetic test suite and the
reconstruction/evaluation stages; a CUDA GPU speeds up segmentation and classification.

## Seeds

A fixed seed (`42` by default, configurable via `seed` in `configs/default.yaml` or
`--seed`) seeds Python, NumPy, and PyTorch RNGs through `brainframe.utils.seed.set_seed`.

## Acquiring data

### OASIS-1 (classification + segmentation)

1. Request access at <https://www.oasis-brains.org/>.
2. Download the cross-sectional NIfTI archive and extract to `data/raw/oasis/`, e.g.:
   ```
   data/raw/oasis/OAS1_0001_MR1/mpr-1.nii.gz
   data/raw/oasis/oasis_cross-sectional.csv   # CDR labels
   ```
3. (Optional) place the BraTS data under `data/raw/brats/`.

`python -m brainframe download-data` prints these instructions.

## Reproducing the test suite

```bash
pytest -ra
```

All tests run on CPU with synthetic NIfTI fixtures — no network, no real data.

## Reproducing the full pipeline on a real volume

```bash
python -m brainframe prepare
python -m brainframe download-sam --model-type vit_b   # ~358 MB
python -m brainframe run --input data/raw/oasis/OAS1_0001_MR1/mpr-1.nii.gz \
                         --output data/outputs/oasis_0001
```

Outputs (see [README](../README.md) table): `label_volume.npy`, `meshes/*.stl`,
`reconstruction_metrics.json`, `figures/reconstruction_3d.html`, and
`evaluation/evaluation_report.html` + `.json`.

## Per-stage reproduction

```bash
python -m brainframe segment --input <volume.nii.gz> --output label_volume.npy
python -m brainframe reconstruct --input label_volume.npy --spacing 1,1,2
python -m brainframe evaluate --input label_volume.npy
python -m brainframe visualize --input label_volume.npy
```

## Reproducing the classification baseline

```bash
python -m brainframe classify --data-root data/raw/oasis   # trains + checkpoints
python -m brainframe classify --image <volume.nii.gz>       # predict a single subject
```

Training writes `data/outputs/classification_best.pt` and `train_history.json`.

## Configurable knobs

All parameters live in the YAML configs under `configs/`. Key ones:

- `seed`, `device` — reproducibility and hardware.
- `segmentation.tta.{enabled,steps,lr}` — test-time adaptation strength.
- `reconstruction.marching.{level,smoothing}` — mesh smoothness.
- `evaluation.therapy.{mode,radius_mm,dose,sigma_mm}` — therapy plan.
- `evaluation.compatibility.{coverage,recovery,risk}_weight` — score weighting.

## Determinism notes

- The TTA simulator and the (optional) BraTS split use NumPy generators seeded by the
  global seed.
- Marching Cubes is deterministic; only the optional Laplacian smoothing is
  iterative (fixed iterations).
- Outputs are JSON-serialized with sorted-default formatting where applicable.
