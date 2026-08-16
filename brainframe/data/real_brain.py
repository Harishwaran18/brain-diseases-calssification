"""Real human brain data for the NeuroCure platform.

Bundles two canonical real-brain datasets (fetched once from nilearn and cached
as compact ``.npz`` assets so no network is needed at runtime):

1. **ICBM152 T1 template** — a real averaged human brain MRI volume
   (``assets/real_brain/icbm152_volume.npz``). Used as the demo scan so that
   segmentation + reconstruction operate on a genuine brain shape rather than a
   synthetic sphere phantom.

2. **fsaverage pial surface** — the real FreeSurfer cortical surface mesh with
   anatomically correct gyri/sulci (``assets/real_brain/fsaverage_pial.npz``).
   Rendered as the realistic brain backdrop in the 3D viewer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from brainframe.reconstruction.marching import MeshData, MeshResult
from brainframe.utils.logging import get_logger

log = get_logger("data.real_brain")

_ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "real_brain"


def _has_assets() -> bool:
    return (_ASSET_DIR / "icbm152_volume.npz").exists()


def load_real_brain_volume(
    shape: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Return ``(volume, label_volume, spacing)`` from the real ICBM152 brain.

    The volume is the real T1 template (normalised 0..1). The label volume is a
    tissue segmentation produced by thresholding the real intensities — gray
    matter (cortex, mid-low T1), white matter (bright core), CSF (dark), plus a
    planted lesion so the therapy pipeline has a target.

    Parameters
    ----------
    shape
        Optional resample target; ``None`` keeps the bundled (96, 112, 84).
    """
    if not _has_assets():
        raise FileNotFoundError(
            "Real-brain assets not found. Run scripts/build_real_brain_assets.py."
        )
    data = np.load(_ASSET_DIR / "icbm152_volume.npz")
    vol = np.asarray(data["volume"], dtype=np.float32)
    spacing = tuple(float(s) for s in data["spacing"])

    if shape is not None and tuple(vol.shape) != shape:
        from scipy.ndimage import zoom

        f = [shape[i] / vol.shape[i] for i in range(3)]
        vol = zoom(vol, f, order=1).astype(np.float32)

    from brainframe.config import LABELS

    # Segment the real intensity histogram into tissues. ICBM152 T1 intensities
    # (normalised 0..1): CSF is dark, GM mid, WM bright.
    labels = np.zeros(vol.shape, dtype=np.int16)
    brain = vol > 0.12  # skull-stripped template: anything >0.12 is brain
    gm = brain & (vol > 0.12) & (vol <= 0.45)
    wm = brain & (vol > 0.45)
    # CSF: low-intensity pockets inside the brain (ventricles/sulci).
    csf = brain & (vol <= 0.12)
    labels[gm] = LABELS["gray_matter"]
    labels[wm] = LABELS["white_matter"]
    labels[csf] = LABELS["csf"]

    # Plant a realistic periventricular white-matter lesion cluster
    # (Dawson's-finger style) so the evidence classifier + therapy pipeline have
    # a target. Several small bilateral plaques near the ventricular system make
    # the demo classify as a periventricular demyelinating pattern (MS-like).
    s0, s1, s2 = vol.shape
    c0, c1, c2 = s0 // 2, s1 // 2 + 6, s2 // 2  # near the ventricular centre
    a0, a1, a2 = np.mgrid[0:s0, 0:s1, 0:s2]
    lesion = np.zeros(vol.shape, dtype=bool)
    for hemi in (-9, 9):  # bilateral periventricular plaques
        for dz, dy, dx in ((0, 0, 0), (3, 2, 0), (-3, -2, 1), (1, 4, -1)):
            cz, cy, cx = c0 + hemi + dx, c1 + dy, c2 + dz
            r = 3.5
            blob = (a0 - cx) ** 2 + (a1 - cy) ** 2 + (a2 - cz) ** 2 < r**2
            lesion |= blob
    labels[lesion] = LABELS["lesion"]
    vol[lesion] = np.float32(vol.max())
    log.info(
        "Real brain loaded: shape=%s spacing=%s lesion_voxels=%d",
        vol.shape,
        spacing,
        int(lesion.sum()),
    )
    return vol, labels, spacing


def load_real_cortex_mesh() -> MeshResult:
    """Return the real fsaverage pial cortex as a :class:`MeshResult`.

    The pial surface is the genuine folded cortical boundary (gray-matter / CSF
    interface) from the FreeSurfer fsaverage atlas — it shows real gyri and
    sulci, not a smooth sphere.
    """
    path = _ASSET_DIR / "fsaverage_pial.npz"
    if not path.exists():
        raise FileNotFoundError("Real cortex asset not found.")
    data = np.load(path)
    verts = np.asarray(data["vertices"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int32)
    # Compute vertex normals for smooth shading.
    normals = _compute_normals(verts, faces)
    mesh = MeshData(
        label="cortex",
        label_idx=1,
        vertices=verts,
        faces=faces,
        normals=normals,
        spacing=(1.0, 1.0, 1.0),
    )
    log.info("Real cortex mesh: %d verts, %d faces", len(verts), len(faces))
    return MeshResult(meshes=[mesh])


def _compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex normals (averaged face normals) for smooth shading."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.where(norms < 1e-12, 1.0, norms)
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    return vertex_normals / np.where(norms < 1e-12, 1.0, norms)


def has_real_brain() -> bool:
    """Whether the bundled real-brain assets are available."""
    return _has_assets() and (_ASSET_DIR / "fsaverage_pial.npz").exists()


# Colour + opacity for each deep structure overlay in the 3D viewer.
_DEEP_NUCLEI_STYLE = {
    "thalamus_left":      (0xE8A040, 0.55),
    "thalamus_right":     (0xE8A040, 0.55),
    "caudate_left":       (0xC064E8, 0.55),
    "caudate_right":      (0xC064E8, 0.55),
    "putamen_left":       (0x6496E8, 0.55),
    "putamen_right":      (0x6496E8, 0.55),
    "pallidum_left":      (0xE86496, 0.55),
    "pallidum_right":     (0xE86496, 0.55),
    "ventricle_left":     (0x40C0E8, 0.35),
    "ventricle_right":    (0x40C0E8, 0.35),
    "hippocampus_left":   (0x6BC97B, 0.60),
    "hippocampus_right":  (0x6BC97B, 0.60),
    "amygdala_left":      (0xE8C064, 0.60),
    "amygdala_right":     (0xE8C064, 0.60),
    "brainstem":          (0xA0886E, 0.50),
}


def load_deep_nuclei() -> list[MeshData]:
    """Load the deep-brain-nuclei meshes (thalamus, BG, ventricles, etc.).

    Returns a list of :class:`MeshData` in MNI-mm coordinates, ready to overlay
    on the fsaverage cortex in the 3D viewer. Returns an empty list if the
    asset is missing (graceful degradation).
    """
    path = _ASSET_DIR / "deep_nuclei.npz"
    if not path.exists():
        log.warning("deep_nuclei.npz not found; deep-nuclei overlay disabled.")
        return []
    data = np.load(path)
    meshes: list[MeshData] = []
    for name in _DEEP_NUCLEI_STYLE:
        vkey, fkey = f"{name}_v", f"{name}_f"
        if vkey not in data:
            continue
        verts = np.asarray(data[vkey], dtype=np.float32)
        faces = np.asarray(data[fkey], dtype=np.int32)
        if len(verts) < 3:
            continue
        normals = _compute_normals(verts, faces)
        meshes.append(
            MeshData(
                label=name,
                label_idx=0,
                vertices=verts,
                faces=faces,
                normals=normals,
                spacing=(1.0, 1.0, 1.0),
            )
        )
    log.info("Loaded %d deep-nuclei meshes", len(meshes))
    return meshes


def deep_nuclei_style(name: str) -> tuple[int, float]:
    """Return ``(color_hex, opacity)`` for a deep-nucleus structure name."""
    return _DEEP_NUCLEI_STYLE.get(name, (0xAAAAAA, 0.5))
