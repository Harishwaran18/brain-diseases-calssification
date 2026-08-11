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
    0: "Healthy (no significant abnormality)",
    1: "Alzheimer's disease",
    2: "Parkinson's disease",
    3: "Multiple sclerosis",
    4: "Glioma / brain tumour",
    5: "Stroke / cerebral infarct",
    6: "Epilepsy (focal cortical / hippocampal sclerosis)",
    7: "Huntington's disease",
    8: "Amyotrophic lateral sclerosis",
    9: "Traumatic brain injury",
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
        Predicted disease class (0=healthy ... 9=TBI), matching
        :mod:`brainframe.classification.diseases`.
    lesion_volume_mm3
        Total detected lesion volume in cubic millimetres.
    n_regions
        Number of distinct lesion regions detected.
    confidence
        Classifier confidence (0..1), used to qualify the rationale.
    """
    technique = default_technique(disease_class)
    # If the predicted class has no curated technique, or if the lesion burden
    # is heavy and a higher-dose adjacent class exists, prefer the heavier one.
    if lesion_volume_mm3 > 5000:
        candidates = techniques_for_class(min(9, disease_class))
        if candidates:
            heavier = max(candidates, key=lambda t: t.dose)
            if heavier.dose > technique.dose:
                technique = heavier

    disease = _DISEASE_NAMES.get(disease_class, f"disease class {disease_class}")
    if confidence >= 0.5:
        conf_note = f" The prediction is well-supported (confidence {confidence:.0%})."
    elif confidence >= 0.2:
        conf_note = (
            f" Note: prediction confidence is moderate ({confidence:.0%});"
            " monitor for progression and seek clinical correlation."
        )
    else:
        conf_note = (
            f" Warning: prediction confidence is low ({confidence:.0%}); the"
            " evidence is equivocal and clinical correlation is essential."
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
