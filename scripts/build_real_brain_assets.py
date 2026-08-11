#!/usr/bin/env python
"""Build the bundled real-human-brain assets under ``assets/real_brain/``.

Fetches canonical neuroimaging data via nilearn (one-time download) and caches
compact ``.npz`` bundles so the NeuroCure platform needs no network at runtime:

- ``icbm152_volume.npz`` — real ICBM152 T1 brain MRI volume (resampled, normalised)
- ``fsaverage_pial.npz`` — real FreeSurfer cortical surface mesh (decimated)

Run once after install::

    python scripts/build_real_brain_assets.py
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "real_brain"


def build_volume() -> None:
    from nilearn import datasets

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    t1 = datasets.fetch_icbm152_2009()
    vol = nib.load(t1.t1).get_fdata().astype(np.float32)
    factors = (96.0 / vol.shape[0], 112.0 / vol.shape[1], 84.0 / vol.shape[2])
    vol = zoom(vol, factors, order=1)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
    np.savez_compressed(
        ASSET_DIR / "icbm152_volume.npz", volume=vol, spacing=np.array([2.0, 2.0, 2.0])
    )
    print(f"volume: {vol.shape} -> {ASSET_DIR / 'icbm152_volume.npz'}")


def build_cortex() -> None:
    import fast_simplification
    from nilearn import datasets

    fs = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    all_v, all_f = [], []
    for hemi in ("left", "right"):
        coords, faces = nib.load(fs[f"pial_{hemi}"]).agg_data()
        # Keep ~16k vertices per hemisphere (≈32k total) so the viewer preserves
        # genuine gyral/sulcal detail rather than over-decimating.
        v_out, f_out = fast_simplification.simplify(coords, faces, target_reduction=0.90)
        all_v.append(v_out)
        all_f.append(np.asarray(f_out))
    offset = 0
    mv, mf = [], []
    for v, f in zip(all_v, all_f, strict=True):
        mv.append(v)
        mf.append(f + offset)
        offset += len(v)
    verts = np.concatenate(mv).astype(np.float32)
    faces = np.concatenate(mf).astype(np.int32)
    np.savez_compressed(ASSET_DIR / "fsaverage_pial.npz", vertices=verts, faces=faces)
    print(f"cortex: {len(verts)} verts, {len(faces)} faces -> {ASSET_DIR / 'fsaverage_pial.npz'}")


if __name__ == "__main__":
    build_volume()
    build_cortex()
    print("done")
