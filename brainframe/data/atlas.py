"""Anatomical atlas for lesion-region labelling.

Maps voxel coordinates in the (ICBM152-derived) brain volume to anatomical
regions (lobes and deep structures). This lets the platform report a
human-readable location for each detected lesion — e.g. "left temporal lobe" —
and is the spatial signal the evidence-based classifier scores disease
signatures against.

Coordinate convention
----------------------
The ICBM152 T1 template (and most NIfTI T1 scans) is stored in **RAS** order:
``X`` = Left→Right, ``Y`` = Posterior→Anterior, ``Z`` = Inferior→Superior. A
volume array of shape ``(nx, ny, nz)`` is indexed as ``(x, y, z)`` and the
normalised fraction ``(fx, fy, fz)`` in ``[0, 1]^3`` therefore means:

    fx  0→1 : patient left  →  right   (hemisphere)
    fy  0→1 : posterior     →  anterior (occipital → frontal)
    fz  0→1 : inferior      →  superior (brainstem → vertex)

The atlas is fully deterministic and side-effect free, so it is trivially
unit-testable without any real data.
"""

from __future__ import annotations

import numpy as np

from brainframe.utils.logging import get_logger

log = get_logger("data.atlas")

# Lateral (left/right) split: the brain midline is the middle of the X axis.
# (In RAS the patient's left is the high-X side; we keep convention-free labels
# "L"/"R" purely from the X fraction so downstream code stays unambiguous.)


def _hemi(fx: float) -> str:
    return "L" if fx < 0.5 else "R"


# Deep-structure / central bounding boxes in normalised (fx, fy, fz) space.
# All are central-in-X; the Y/Z extents encode the anatomy (RAS).
_DEEP_BOXES = {
    # basal ganglia / striatum: central-in-X, anterior-mid-Y, mid-superior-Z.
    "basal_ganglia": (0.40, 0.60, 0.50, 0.72, 0.48, 0.64),
    # thalamus: central, posterior to the striatum, mid-superior.
    "thalamus": (0.42, 0.58, 0.40, 0.55, 0.50, 0.66),
    # brainstem: inferior, central.
    "brainstem": (0.44, 0.56, 0.45, 0.60, 0.06, 0.22),
    # cerebellum: inferior + posterior.
    "cerebellum": (0.30, 0.70, 0.05, 0.40, 0.04, 0.22),
    # ventricular system: a thin central band used to detect periventricular lesions.
    "periventricular": (0.36, 0.64, 0.38, 0.78, 0.44, 0.74),
}


def _in_box(fx: float, fy: float, fz: float, box: tuple[float, ...]) -> bool:
    x0, x1, y0, y1, z0, z1 = box
    return x0 <= fx <= x1 and y0 <= fy <= y1 and z0 <= fz <= z1


def _lobe(fx: float, fy: float, fz: float) -> str:
    """Classify a cortical voxel into a lobe using RAS normalised coordinates.

    fy: posterior(0)→anterior(1); fz: inferior(0)→superior(1).
    """
    # Medial temporal / hippocampal region: anterior-inferior-medial (checked
    # first so anterior-inferior-medial voxels aren't swallowed by "frontal").
    if fy > 0.50 and fz < 0.42 and 0.36 <= fx <= 0.64:
        return "hippocampus"
    # Anterior third -> frontal (high Y).
    if fy > 0.66 and fz >= 0.42:
        return "frontal"
    # Posterior + inferior -> occipital (low Y, mid Z) — but only if not
    # superior (superior-posterior is parietal).
    if fy < 0.30 and 0.40 <= fz <= 0.62:
        return "occipital"
    # Posterior-superior (parietal association area) — high Z, behind motor strip.
    if fy < 0.46 and fz > 0.55:
        return "parietal"
    # Inferior-lateral (temporal lobe, below the Sylvian fissure).
    if fz < 0.45 and 0.28 <= fy <= 0.72:
        return "temporal"
    # Superior central strip around the central sulcus -> motor cortex.
    if 0.48 <= fy <= 0.62 and 0.55 <= fz <= 0.78 and 0.33 <= fx <= 0.67:
        return "motor_cortex"
    # default: deep white matter (corona radiata / centrum semiovale).
    return "corona_radiata"


def region_at(voxel_xyz: tuple[int, int, int], shape_xyz: tuple[int, int, int]) -> str:
    """Return the anatomical region name for a voxel in (X, Y, Z) coordinates.

    Parameters
    ----------
    voxel_xyz
        (x, y, z) integer index of the lesion voxel.
    shape_xyz
        (nx, ny, nz) shape of the volume.
    """
    x, y, z = voxel_xyz
    nx, ny, nz = shape_xyz
    fx = (x + 0.5) / nx
    fy = (y + 0.5) / ny
    fz = (z + 0.5) / nz
    # Specific deep structures first (small boxes) — a voxel in the striatum
    # should be labelled "basal_ganglia" not "frontal".
    for name in ("brainstem", "cerebellum", "thalamus", "basal_ganglia"):
        if _in_box(fx, fy, fz, _DEEP_BOXES[name]):
            return name
    # Periventricular white matter: near the ventricles but not in a specific
    # deep nucleus — checked before lobes so Dawson-finger / PV plaques map here.
    if _in_box(fx, fy, fz, _DEEP_BOXES["periventricular"]):
        return "periventricular"
    return _lobe(fx, fy, fz)


def label_lesion_cluster(voxel_coords: np.ndarray, shape_xyz: tuple[int, int, int]) -> dict:
    """Summarise the anatomical location of a lesion cluster.

    Returns a dict with the dominant region, hemisphere(s), centroid (xyz),
    and a human-readable location string.
    """
    pts = np.asarray(voxel_coords)
    if len(pts) == 0:
        return {
            "region": "unknown",
            "hemisphere": "unknown",
            "centroid_xyz": [0.0, 0.0, 0.0],
            "location": "unknown location",
        }
    nx, ny, nz = shape_xyz
    regions = [region_at((int(p[0]), int(p[1]), int(p[2])), shape_xyz) for p in pts]
    # Dominant region by vote.
    values, counts = np.unique(regions, return_counts=True)
    region = str(values[np.argmax(counts)])
    # Hemisphere(s): could be bilateral if the cluster spans the midline.
    fxs = (pts[:, 0] + 0.5) / nx
    has_left = bool(np.any(fxs < 0.5))
    has_right = bool(np.any(fxs >= 0.5))
    if has_left and has_right:
        hemi = "bilateral"
    elif has_left:
        hemi = "left"
    else:
        hemi = "right"
    centroid = pts.mean(axis=0).tolist()
    # "periventricular" wins over lobe labels when relevant.
    near_vent = bool(
        np.mean(
            [
                _in_box(
                    (p[0] + 0.5) / nx,
                    (p[1] + 0.5) / ny,
                    (p[2] + 0.5) / nz,
                    _DEEP_BOXES["periventricular"],
                )
                for p in pts
            ]
        )
        > 0.3
    )
    if near_vent:
        region = "periventricular"
    # Human-readable string.
    region_pretty = {
        "frontal": "frontal lobe",
        "parietal": "parietal lobe",
        "temporal": "temporal lobe",
        "occipital": "occipital lobe",
        "cerebellum": "cerebellum",
        "brainstem": "brainstem",
        "basal_ganglia": "basal ganglia",
        "thalamus": "thalamus",
        "periventricular": "periventricular white matter",
        "corona_radiata": "corona radiata",
        "motor_cortex": "motor cortex",
        "hippocampus": "hippocampus",
        "insula": "insula",
        "limbic": "limbic system",
    }.get(region, region)
    if hemi == "bilateral":
        location = f"bilateral {region_pretty}"
    elif hemi == "left":
        location = f"left {region_pretty}"
    else:
        location = f"right {region_pretty}"
    return {
        "region": region,
        "hemisphere": hemi,
        "centroid_xyz": [round(float(c), 2) for c in centroid],
        "location": location,
    }


def classify_pattern(
    clusters: list[dict], total_voxels: int, shape_xyz: tuple[int, int, int]
) -> str:
    """Infer the overall lesion pattern from the cluster analysis.

    Returns one of the signature patterns: focal / diffuse / symmetric /
    periventricular / ring_enhancing (the last is approximated as focal here).
    """
    n = len(clusters)
    if n == 0:
        return "focal"
    # Periventricular if the dominant regions are near the ventricles.
    pv_count = sum(1 for c in clusters if c["region"] == "periventricular")
    if pv_count >= max(2, n // 2):
        return "periventricular"
    # Bilateral symmetric: lesions present in both hemispheres with similar size.
    left = [c for c in clusters if c["hemisphere"] == "left"]
    right = [c for c in clusters if c["hemisphere"] == "right"]
    if left and right and abs(len(left) - len(right)) <= 1:
        # sizes comparable within 40%
        lv = sum(c.get("voxels", 0) for c in left)
        rv = sum(c.get("voxels", 0) for c in right)
        denom = min(lv, rv) + 1
        if max(lv, rv) / denom <= 1.6:
            return "symmetric"
    # Single dominant region -> focal; otherwise diffuse.
    sizes = [c.get("voxels", 0) for c in clusters]
    if n <= 2 or (max(sizes) / (sum(sizes) + 1) > 0.7):
        return "focal"
    return "diffuse"
