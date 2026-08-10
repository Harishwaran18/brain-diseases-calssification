"""3D volumetric reconstruction from segmented 2D slices."""

from __future__ import annotations

from brainframe.reconstruction.marching import MeshResult, extract_meshes
from brainframe.reconstruction.mesh_metrics import MeshMetrics, compute_metrics
from brainframe.reconstruction.stacking import ReconstructionResult, stack_slices
from brainframe.reconstruction.visualize import render_3d, save_cross_sections

__all__ = [
    "MeshMetrics",
    "MeshResult",
    "ReconstructionResult",
    "compute_metrics",
    "extract_meshes",
    "render_3d",
    "save_cross_sections",
    "stack_slices",
]
