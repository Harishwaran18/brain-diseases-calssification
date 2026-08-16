"""Build deep-brain-nuclei meshes from the Harvard-Oxford subcortical atlas.

Extracts thalamus, caudate, putamen, pallidum, hippocampus, amygdala,
ventricles, and brainstem as individual triangle meshes (via marching cubes)
and caches them as a single compact ``.npz`` so the 3D viewer can overlay
realistic deep structures on the fsaverage cortex without any network fetch
at runtime.

Run once::

    python -m scripts.build_deep_nuclei_meshes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from nilearn import datasets, image
from skimage.measure import marching_cubes

_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "real_brain"

# Harvard-Oxford sub-maxprob atlas indices we care about.
_STRUCTURE_MAP = {
    "thalamus_left": 4,
    "thalamus_right": 15,
    "caudate_left": 5,
    "caudate_right": 16,
    "putamen_left": 6,
    "putamen_right": 17,
    "pallidum_left": 7,
    "pallidum_right": 18,
    "ventricle_left": 3,
    "ventricle_right": 14,
    "hippocampus_left": 9,
    "hippocampus_right": 20,
    "amygdala_left": 10,
    "amygdala_right": 21,
    "brainstem": 8,
}


def _extract_mesh(mask_3d: np.ndarray, affine: np.ndarray, spacing: tuple) -> tuple:
    """Run marching cubes on a binary mask and transform to MNI mm space."""
    if mask_3d.sum() < 20:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    # Pad so marching cubes has room at borders.
    padded = np.pad(mask_3d.astype(np.float32), 1)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)
    verts -= 1.0 * np.array(spacing)  # undo pad offset
    # Transform voxel -> MNI mm via affine (x, y, z, 1).
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    verts_mni = verts_h @ affine.T
    verts_mni = verts_mni[:, :3]
    return verts_mni.astype(np.float32), faces.astype(np.int32)


def main() -> None:
    atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-1mm")
    atlas_img = image.load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    spacing = atlas_img.header.get_zooms()[:3]
    print(f"Atlas shape: {atlas_data.shape}, spacing: {spacing}")

    _ASSET_DIR.mkdir(parents=True, exist_ok=True)
    meshes = {}
    for name, idx in _STRUCTURE_MAP.items():
        mask = (atlas_data == idx) | (atlas_data == idx + 1 if idx < 21 else False)
        mask = atlas_data == idx
        verts, faces = _extract_mesh(mask, affine, spacing)
        print(f"  {name:25s} idx={idx:2d}  verts={len(verts):6d}  faces={len(faces):6d}")
        if len(verts) > 0:
            meshes[name] = {"vertices": verts, "faces": faces}

    out = _ASSET_DIR / "deep_nuclei.npz"
    np.savez_compressed(
        out,
        **{f"{k}_v": v["vertices"] for k, v in meshes.items()},
        **{f"{k}_f": v["faces"] for k, v in meshes.items()},
    )
    print(f"Saved {len(meshes)} deep-nuclei meshes to {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
