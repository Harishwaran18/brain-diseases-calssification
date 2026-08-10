# Research Alignment

This document maps each `brainframe` module to the objectives of the research
statement *“Retraining-Free AI Framework for 3D Neurodegenerative Brain Reconstruction
and Computational Evaluation for Neuroregenerative Therapy.”*

## Objective 1 — Annotation-efficient / retraining-free segmentation

> “Retraining-free, context-aware inference” that avoids the dependence on labelled
> data of conventional AI.

| Module | Contribution |
|--------|--------------|
| `segmentation/sam_wrapper.py` | Uses the Segment Anything Model (SAM) — a promptable, zero-shot segmenter that generalizes across medical/MRI domains **without fine-tuning**. |
| `segmentation/prompts.py` | Auto-prompts derived from the image (grid / intensity peaks / gradient bbox) remove the need for manual interaction. |
| `segmentation/tta.py` | Test-time adaptation nudges a tiny set of mask-decoder parameters via self-supervised signals (entropy, IoU uncertainty, dual-scale EMA consistency) — **no source data, no labels**. |
| `segmentation/inference.py` | Zero/few-shot per-slice segmentation with label-free intensity-ranking assignment to tissue classes. |

When SAM is unavailable a deterministic `HeuristicSegmenter` keeps the pipeline
runnable on CPU/CI.

## Objective 2 — 3D volumetric reconstruction

> Stack segmented slices and reconstruct a 3D brain volume with isosurface extraction
> and morphometrics.

| Module | Contribution |
|--------|--------------|
| `reconstruction/stacking.py` | Spacing-aware interpolation to an isotropic 3D label volume with inter-slice gap filling. |
| `reconstruction/marching.py` | Marching Cubes (Lewiner) isosurface extraction, per-label meshes, optional Laplacian smoothing, STL/PLY/GLB export. |
| `reconstruction/mesh_metrics.py` | Volume, surface area, compactness (sphericity), and atrophy ratios — validated analytically on a sphere. |
| `reconstruction/visualize.py` | Plotly/pyvista 3D rendering + cross-sections. |

## Objective 3 — Computational evaluation of neuroregenerative therapy

> Beyond diagnosis: a non-invasive, in-silico decision-support layer that evaluates
> therapeutic strategies within the reconstructed 3D environment.

| Module | Contribution |
|--------|--------------|
| `evaluation/lesion_analysis.py` | Lesion-region detection: connected components, centroid, volume, spatial extent, adjacency to key structures. |
| `evaluation/therapy_model.py` | Parameterizes a therapy: target region, radius, dose, mode, kernel. |
| `evaluation/simulator.py` | Applies the intervention as a Gaussian/diffusion effect field, propagates over a region graph, and mutates tissue labels — a tractable, self-contained counterpart of lesion-as-perturbation / TVB-style whole-brain models. |
| `evaluation/compatibility.py` | Structural-compatibility score (coverage + recovery + risk) in [0,1] for therapy-plan ranking. |
| `evaluation/report.py` | JSON + HTML report with before/after 3D renders and metric tables — the decision-support output. |

## Umbrella task — brain-disease classification

> The repository name is `brain-diseases-calssification`; classification is the entry
> module that contrasts the data-dependent approach against the retraining-free path.

`classification/` provides a supervised 3D CNN baseline (MONAI DenseNet/ResNet + a
2D-slice ensemble) trained on labelled OASIS-1 data. It is deliberately the
"data-hungry" baseline the research statement critiques, retained to honour the repo
name and to frame the retraining-free pipeline as the proposed improvement.

## Literature grounding

- SAM zero-shot segmentation: Kirillov et al., *Segment Anything*, ICCV 2023.
- SAM test-time adaptation: entropy minimization + dual-scale consistency,
  SAM-TTA / training-free few-shot adaptation literature.
- Marching Cubes / Flying Edges: standard isosurface extraction from MRI label fields
  (corpus callosum, cerebral structures, fetal brain).
- Lesion-as-perturbation / whole-brain modelling: The Virtual Brain and
  diffusion-based lesion synthesis; the in-silico therapy simulator is a tractable,
  dependency-light adaptation of these ideas.
