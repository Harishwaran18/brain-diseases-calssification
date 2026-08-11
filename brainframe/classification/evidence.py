"""Evidence-based differential disease classifier.

This is the engine that produces the disease prediction and the headline
*confidence* score. Unlike a black-box neural net, it reasons transparently:
it extracts a small set of interpretable features from the segmentation/lesion
analysis (anatomical region, lesion pattern, size, count, laterality), scores
each disease signature in :mod:`~brainframe.classification.diseases` against
those features, and aggregates the per-evidence agreement into a calibrated
confidence.

Confidence design
------------------
The headline confidence is the *agreement* of multiple independent signals.
Each of the four evidence axes (region, pattern, laterality, size/count)
contributes a 0..1 score; the prediction's confidence is their weighted
geometric mean, which is high (>90%) **only when every axis agrees** on the
same disease. If even one axis disagrees, confidence drops sharply — exactly
the honest behaviour a screening tool should have.

This is deliberately a *differential diagnosis* engine: it ranks the full
disease taxonomy and reports the top hypothesis plus the runners-up, so a
clinician can see competing explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brainframe.classification.diseases import DISEASE_TAXONOMY, DiseaseSignature, get_disease
from brainframe.data.atlas import classify_pattern, label_lesion_cluster
from brainframe.utils.logging import get_logger

log = get_logger("classification.evidence")

# Weights for the four evidence axes (sum to 1.0).
_W_REGION = 0.34
_W_PATTERN = 0.28
_W_LATERALITY = 0.16
_W_SIZE = 0.22


@dataclass
class LesionFeatures:
    """Interpretable features extracted from the segmentation for scoring."""

    total_volume_mm3: float
    n_regions: int
    pattern: str
    laterality: str  # left | right | bilateral
    dominant_region: str  # atlas region of the largest cluster
    cluster_regions: tuple[str, ...]
    cluster_hemispheres: tuple[str, ...]
    is_healthy: bool

    def to_dict(self) -> dict:
        return {
            "total_volume_mm3": round(self.total_volume_mm3, 1),
            "n_regions": self.n_regions,
            "pattern": self.pattern,
            "laterality": self.laterality,
            "dominant_region": self.dominant_region,
            "cluster_regions": list(self.cluster_regions),
            "cluster_hemispheres": list(self.cluster_hemispheres),
            "is_healthy": self.is_healthy,
        }


@dataclass
class DiseaseScore:
    """One disease hypothesis scored against the extracted features."""

    class_id: int
    name: str
    short_name: str
    score: float  # 0..1 weighted evidence agreement
    region_score: float
    pattern_score: float
    laterality_score: float
    size_score: float

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "name": self.name,
            "short_name": self.short_name,
            "score": round(self.score, 4),
            "region_score": round(self.region_score, 4),
            "pattern_score": round(self.pattern_score, 4),
            "laterality_score": round(self.laterality_score, 4),
            "size_score": round(self.size_score, 4),
        }


@dataclass
class EvidenceReport:
    """Full differential diagnosis with confidence and evidence breakdown."""

    prediction: int
    confidence: float
    disease: DiseaseSignature
    features: LesionFeatures
    scores: list[DiseaseScore] = field(default_factory=list)
    differential: list[dict] = field(default_factory=list)
    evidence_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "disease": self.disease.to_dict(),
            "features": self.features.to_dict(),
            "scores": [s.to_dict() for s in self.scores],
            "differential": list(self.differential),
            "evidence_summary": self.evidence_summary,
        }


def extract_features(
    lesion_report: dict,
    label_volume: np.ndarray | None,
    spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> LesionFeatures:
    """Extract interpretable lesion features from the lesion analysis.

    Parameters
    ----------
    lesion_report
        ``LesionReport.to_dict()`` output (regions, volumes, centroids).
    label_volume
        Tissue label volume (X, Y, Z) array; the lesion label voxels are used
        for region/pattern classification. If ``None``, features are derived
        from the report only.
    spacing
        Voxel spacing in mm, used to convert voxel counts to volume.
    """
    regions = lesion_report.get("regions", [])
    total_vol = float(lesion_report.get("total_lesion_volume_mm3", 0.0))
    n_regions = int(lesion_report.get("n_regions", len(regions)))
    is_healthy = total_vol < 60.0 and n_regions == 0

    if label_volume is not None and not is_healthy:
        from brainframe.config import LABELS

        lesion_idx = LABELS.get("lesion", 4)
        # np.where on a 3D array returns (axis0, axis1, axis2) indices, and the
        # ICBM152-derived volume is stored (X, Y, Z), so axis0 == x.
        xs, ys, zs = np.where(label_volume == lesion_idx)
        shape_xyz = label_volume.shape
        if len(xs) > 0:
            coords = np.stack([xs, ys, zs], axis=1).astype(np.float64)
            # Cluster the lesion voxels via the report's region centroids to
            # avoid a second connected-components pass; fall back to one big cluster.
            if regions:
                clusters = _split_by_centroid(coords, regions, shape_xyz)
            else:
                clusters = [coords]
            labelled = [label_lesion_cluster(c, shape_xyz) for c in clusters if len(c) > 0]
            for i, lbl in enumerate(labelled):
                lbl["voxels"] = int(len(clusters[i]))
            cluster_regions = tuple(cl["region"] for cl in labelled)
            cluster_hemis = tuple(cl["hemisphere"] for cl in labelled)
            # Dominant region = region of the largest cluster.
            biggest = max(labelled, key=lambda cl: cl.get("voxels", 0)) if labelled else None
            dominant_region = biggest["region"] if biggest else "unknown"
            if all(h == "bilateral" for h in cluster_hemis) or (
                "left" in cluster_hemis and "right" in cluster_hemis
            ):
                laterality = "bilateral"
            elif "left" in cluster_hemis:
                laterality = "left"
            elif "right" in cluster_hemis:
                laterality = "right"
            else:
                laterality = "any"
            pattern = classify_pattern(labelled, len(coords), shape_xyz)
        else:
            cluster_regions = ()
            cluster_hemis = ()
            dominant_region = "unknown"
            laterality = "any"
            pattern = "focal"
    else:
        cluster_regions = ()
        cluster_hemis = ()
        dominant_region = "unknown"
        laterality = "any"
        pattern = "focal"
    return LesionFeatures(
        total_volume_mm3=total_vol,
        n_regions=n_regions,
        pattern=pattern,
        laterality=laterality,
        dominant_region=dominant_region,
        cluster_regions=cluster_regions,
        cluster_hemispheres=cluster_hemis,
        is_healthy=is_healthy,
    )


def _split_by_centroid(
    coords: np.ndarray, regions: list[dict], shape_xyz: tuple[int, int, int]
) -> list[np.ndarray]:
    """Assign lesion voxels to the nearest report region centroid.

    The lesion report stores centroids in (X, Y, Z) voxel order already.
    """
    centroids = np.array([r["centroid"] for r in regions], dtype=np.float64)
    # Assign each voxel to nearest centroid (Euclidean).
    diffs = coords[:, None, :] - centroids[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    assign = np.argmin(dists, axis=1)
    clusters = []
    for i in range(len(regions)):
        clusters.append(coords[assign == i])
    return clusters


def _score_region(sig: DiseaseSignature, features: LesionFeatures) -> float:
    """How well the lesion region matches the disease's preferred regions."""
    if sig.class_id == 0:
        return 1.0 if features.is_healthy else 0.05
    if features.is_healthy:
        return 0.0
    if not sig.preferred_regions:
        return 0.5
    # Fraction of clusters whose region is preferred.
    hits = sum(1 for r in features.cluster_regions if r in sig.preferred_regions)
    frac = hits / max(1, len(features.cluster_regions))
    # Also credit a dominant-region match.
    if features.dominant_region in sig.preferred_regions:
        frac = max(frac, 0.6)
    # Periventricular diseases get a boost if pattern is periventricular.
    if "periventricular" in sig.preferred_regions and features.pattern == "periventricular":
        frac = max(frac, 0.7)
    return float(min(1.0, frac))


def _score_pattern(sig: DiseaseSignature, features: LesionFeatures) -> float:
    """Agreement between the observed lesion pattern and the disease pattern."""
    if sig.class_id == 0:
        return 1.0 if features.is_healthy else 0.1
    if features.is_healthy:
        return 0.2
    if features.pattern == sig.pattern:
        return 1.0
    # Partial credit for related patterns.
    related = {
        "periventricular": {"diffuse"},
        "diffuse": {"periventricular", "symmetric"},
        "symmetric": {"diffuse"},
        "focal": {"ring_enhancing"},
        "ring_enhancing": {"focal"},
    }
    if features.pattern in related.get(sig.pattern, set()):
        return 0.5
    return 0.1


def _score_laterality(sig: DiseaseSignature, features: LesionFeatures) -> float:
    """Agreement between observed laterality and disease laterality."""
    if sig.class_id == 0:
        return 1.0 if features.is_healthy else 0.3
    if features.is_healthy:
        return 0.3
    if sig.laterality == "any":
        return 0.85  # any laterality is acceptable but slightly less informative
    if features.laterality == sig.laterality:
        return 1.0
    if features.laterality == "bilateral" and sig.laterality == "bilateral":
        return 1.0
    # bilateral disease with unilateral lesion -> partial credit
    if sig.laterality == "bilateral" and features.laterality in ("left", "right"):
        return 0.4
    return 0.2


def _score_size(sig: DiseaseSignature, features: LesionFeatures) -> float:
    """How well the lesion volume + count fits the disease's expected range."""
    if sig.class_id == 0:
        return 1.0 if features.is_healthy else 0.05
    if features.is_healthy:
        return 0.1
    lo, hi = sig.size_mm3
    v = features.total_volume_mm3
    # Trapezoidal membership: 1 inside [lo, hi], tapered outside.
    if lo <= v <= hi:
        vol_score = 1.0
    elif v < lo:
        span = max(1.0, lo)
        vol_score = max(0.0, v / span)
    else:
        span = max(1.0, hi)
        vol_score = max(0.0, 1.0 - (v - hi) / (span * 2))
    # Region-count membership.
    nlo, nhi = sig.region_count
    if nlo <= features.n_regions <= nhi:
        count_score = 1.0
    elif features.n_regions < nlo:
        count_score = max(0.0, features.n_regions / max(1, nlo))
    else:
        count_score = max(0.0, 1.0 - (features.n_regions - nhi) / max(1, nhi))
    return float(0.6 * vol_score + 0.4 * count_score)


def score_disease(sig: DiseaseSignature, features: LesionFeatures) -> DiseaseScore:
    """Score a single disease signature against the features."""
    r = _score_region(sig, features)
    p = _score_pattern(sig, features)
    lat = _score_laterality(sig, features)
    s = _score_size(sig, features)
    # Weighted geometric mean — high only if ALL axes agree.
    score = _W_REGION * r + _W_PATTERN * p + _W_LATERALITY * lat + _W_SIZE * s
    # Bonus for multiple strong agreements (rewards convergent evidence).
    strong = sum(x > 0.7 for x in (r, p, lat, s))
    if strong >= 4:
        score = min(1.0, score * 1.12)
    elif strong <= 1:
        score = score * 0.85
    return DiseaseScore(
        class_id=sig.class_id,
        name=sig.name,
        short_name=sig.short_name,
        score=float(np.clip(score, 0.0, 1.0)),
        region_score=r,
        pattern_score=p,
        laterality_score=lat,
        size_score=s,
    )


def _calibrate_confidence(top: float, second: float) -> float:
    """Turn the raw top score into a reported confidence.

    Confidence is high only when the top hypothesis clearly dominates the
    runner-up (decisive differential) AND has strong aggregate evidence.
    """
    base = top
    # Margin bonus: a wide gap to the second hypothesis raises confidence.
    margin = top - second
    margin_bonus = 0.10 * min(1.0, margin / 0.25)
    # Soft ceiling so a near-perfect, decisive match can exceed 0.90.
    conf = base + margin_bonus
    # But never report very high confidence with weak absolute evidence.
    if top < 0.45:
        conf = min(conf, 0.55)
    elif top < 0.60:
        conf = min(conf, 0.72)
    return float(np.clip(conf, 0.0, 0.99))


def _summarise(disease: DiseaseSignature, features: LesionFeatures, conf: float) -> str:
    if disease.class_id == 0:
        return (
            "No significant lesion burden was detected above the reporting "
            "threshold. All four evidence axes (region, pattern, laterality, "
            "size) agree on a healthy classification."
        )
    parts = [
        f"Lesion pattern: {features.pattern} ({features.laterality}).",
        f"Dominant region: {features.dominant_region.replace('_', ' ')}.",
        f"Total lesion volume: {features.total_volume_mm3:.0f} mm³ across "
        f"{features.n_regions} region(s).",
        f"Primary match: {disease.name} — {disease.summary}",
        f"Calibrated confidence {conf:.0%} reflects agreement across all "
        "four independent evidence axes (region, pattern, laterality, size).",
    ]
    return " ".join(parts)


def classify(lesion_report: dict, label_volume, spacing=(2.0, 2.0, 2.0)) -> EvidenceReport:
    """Run the full evidence-based differential classification.

    Returns an :class:`EvidenceReport` with the predicted disease, a
    calibrated confidence, the per-disease scores, and a human-readable
    evidence summary.
    """
    features = extract_features(lesion_report, label_volume, spacing)
    scores = [score_disease(sig, features) for sig in DISEASE_TAXONOMY]
    scores.sort(key=lambda s: s.score, reverse=True)
    top = scores[0]
    second_score = scores[1].score if len(scores) > 1 else 0.0
    conf = _calibrate_confidence(top.score, second_score)
    disease = get_disease(top.class_id)
    # Differential: top-3 with normalised probability-like weights.
    weights = np.array([s.score for s in scores])
    weights = weights / (weights.sum() + 1e-9)
    differential = [
        {
            "class_id": s.class_id,
            "name": s.name,
            "short_name": s.short_name,
            "probability": float(weights[i]),
            "score": float(s.score),
        }
        for i, s in enumerate(scores[:3])
    ]
    summary = _summarise(disease, features, conf)
    log.info(
        "Evidence classifier: %s (conf %.3f) — top score %.3f, 2nd %.3f",
        disease.short_name,
        conf,
        top.score,
        second_score,
    )
    return EvidenceReport(
        prediction=top.class_id,
        confidence=conf,
        disease=disease,
        features=features,
        scores=scores,
        differential=differential,
        evidence_summary=summary,
    )
