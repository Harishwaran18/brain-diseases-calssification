"""Therapy recommendation engine.

Given a disease prediction and a lesion analysis, choose the most appropriate
curing technique from the :mod:`~brainframe.therapy.library` and return it with
a human-readable rationale tailored to the specific subject's lesion burden.

The recommender is deterministic and pure -- it has no side effects and is
trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from brainframe.therapy.library import TherapyTechnique, default_technique, techniques_for_class


@dataclass
class TherapyRecommendation:
    """A recommended therapy technique with subject-specific rationale."""

    technique: TherapyTechnique
    rationale: str
    lesion_volume_mm3: float
    n_regions: int
    disease_class: int

    def to_dict(self) -> dict:
        t = self.technique
        return {
            "technique_id": t.id,
            "technique_name": t.name,
            "mode": t.mode,
            "target_mode": t.target_mode,
            "radius_mm": t.radius_mm,
            "dose": t.dose,
            "sigma_mm": t.sigma_mm,
            "kernel": t.kernel,
            "target_label": t.target_label,
            "rationale": self.rationale,
            "expected_effect": t.expected_effect,
            "references": list(t.references),
            "lesion_volume_mm3": self.lesion_volume_mm3,
            "n_regions": self.n_regions,
            "disease_class": self.disease_class,
        }


_DISEASE_NAMES = {
    0: "no neurodegeneration",
    1: "early-stage neurodegenerative changes",
    2: "moderate neurodegeneration (Alzheimer's / Parkinson's spectrum)",
    3: "advanced neurodegeneration",
}


def recommend_therapy(
    disease_class: int,
    lesion_volume_mm3: float = 0.0,
    n_regions: int = 0,
    confidence: float = 0.0,
) -> TherapyRecommendation:
    """Recommend a therapy technique for a predicted disease and lesion burden.

    Parameters
    ----------
    disease_class
        Predicted disease class (0=healthy ... 3=advanced).
    lesion_volume_mm3
        Total detected lesion volume in cubic millimetres.
    n_regions
        Number of distinct lesion regions detected.
    confidence
        Classifier confidence (0..1), used to qualify the rationale.
    """
    technique = default_technique(disease_class)
    # If the predicted class has no curated technique, or if the lesion burden
    # clearly contradicts the prediction, prefer a technique matching the lesion.
    if lesion_volume_mm3 > 5000 and disease_class < 3:
        heavier = techniques_for_class(min(3, disease_class + 1))
        if heavier:
            technique = heavier[0]

    disease = _DISEASE_NAMES.get(disease_class, f"class {disease_class}")
    conf_note = (
        f" The prediction is well-supported (confidence {confidence:.0%})."
        if confidence >= 0.5
        else f" Note: prediction confidence is moderate ({confidence:.0%});"
        " monitor for progression."
    )
    if lesion_volume_mm3 > 0:
        lesion_note = (
            f" Lesion analysis found {n_regions} region(s) totalling {lesion_volume_mm3:.0f} mm³."
        )
    else:
        lesion_note = " No focal lesions were detected."

    rationale = f"Prediction: {disease}.{conf_note}{lesion_note}\n\n{technique.rationale}"
    return TherapyRecommendation(
        technique=technique,
        rationale=rationale,
        lesion_volume_mm3=lesion_volume_mm3,
        n_regions=n_regions,
        disease_class=disease_class,
    )
