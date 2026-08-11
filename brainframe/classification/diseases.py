"""Comprehensive brain-disease taxonomy with evidence signatures.

Each disease carries a *signature* describing where its lesions typically appear,
what pattern they take, their expected size, and lateral symmetry. The
evidence-based differential classifier (:mod:`~brainframe.classification.evidence`)
scores extracted lesion features against these signatures to produce a
calibrated, transparent disease hypothesis with confidence.

The taxonomy covers 20 major categories encountered in clinical neuroimaging:

    0  Healthy / no significant abnormality
    1  Alzheimer's disease (medial-temporal / parietal atrophy)
    2  Parkinson's disease (nigrostriatal / basal-ganglia involvement)
    3  Multiple sclerosis (periventricular white-matter plaques)
    4  Glioma / brain tumour (focal mass, often unilateral)
    5  Stroke / cerebral infarct (vascular territory, unilateral)
    6  Epilepsy (focal cortical lesion / hippocampal sclerosis)
    7  Huntington's disease (caudate / putamen atrophy)
    8  Amyotrophic lateral sclerosis (motor-cortex / corticospinal tract)
    9  Traumatic brain injury (focal contusion / diffuse axonal)
   10  Meningioma (extra-axial dural-based mass)
   11  Brain metastasis (multiple ring-enhancing lesions)
   12  MCA territory infarction (large unilateral MCA stroke)
   13  Subdural haematoma (extra-axial crescent collection)
   14  Normal pressure hydrocephalus (ventricular enlargement)
   15  Creutzfeldt-Jakob disease (cortical / basal-ganglia diffusion change)
   16  Frontotemporal dementia (frontal / temporal atrophy)
   17  Lewy body dementia (cortical / subcortical Lewy pathology)
   18  Vascular dementia (diffuse small-vessel disease)
   19  Progressive supranuclear palsy (midbrain atrophy)
   20  Brain abscess (ring-enhancing encapsulated infection)

Each signature is literature-informed (see ``references``). This module is pure
data and has no side effects, so it is trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiseaseSignature:
    """A disease's expected lesion signature for evidence-based matching."""

    class_id: int
    name: str
    short_name: str
    # Preferred anatomical regions (lobes/structures) for lesions of this disease.
    # Matched against the atlas region of each detected lesion cluster.
    preferred_regions: tuple[str, ...]
    # Lesion pattern type expected for this disease.
    #   focal          - single well-circumscribed lesion
    #   diffuse        - widespread / poorly-circumscribed involvement
    #   symmetric      - bilaterally symmetric involvement
    #   periventricular- lesions hugging the ventricular system
    pattern: str
    # Expected laterality: left | right | bilateral | any.
    laterality: str
    # Typical lesion-volume range in cubic millimetres (min, max).
    size_mm3: tuple[float, float]
    # Typical number of distinct lesion regions (min, max).
    region_count: tuple[int, int]
    # ICD-11 block / chapter for clinical context.
    icd_block: str
    # Human-readable clinical summary.
    summary: str
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "name": self.name,
            "short_name": self.short_name,
            "preferred_regions": list(self.preferred_regions),
            "pattern": self.pattern,
            "laterality": self.laterality,
            "size_mm3": list(self.size_mm3),
            "region_count": list(self.region_count),
            "icd_block": self.icd_block,
            "summary": self.summary,
            "references": list(self.references),
        }


# Canonical anatomical regions surfaced by the atlas. Keep stable strings so
# signature matching is robust.
REGIONS = (
    "frontal",
    "parietal",
    "temporal",
    "occipital",
    "cerebellum",
    "brainstem",
    "basal_ganglia",
    "thalamus",
    "periventricular",
    "corona_radiata",
    "motor_cortex",
    "hippocampus",
    "insula",
    "limbic",
    "ventricular",
    "midbrain",
    "extra_axial",
    "gray_white_junction",
)

PATTERNS = ("focal", "diffuse", "symmetric", "periventricular", "ring_enhancing")
LATERALITIES = ("left", "right", "bilateral", "any")


DISEASE_TAXONOMY: list[DiseaseSignature] = [
    DiseaseSignature(
        class_id=0,
        name="Healthy",
        short_name="Healthy",
        preferred_regions=(),
        pattern="focal",
        laterality="any",
        size_mm3=(0.0, 50.0),
        region_count=(0, 0),
        icd_block="no diagnosis",
        summary=(
            "No significant structural abnormality detected. The brain volume is "
            "within normal limits and no focal lesion meets the reporting threshold."
        ),
        references=["ICD-11 11-A0: No diagnosed condition"],
    ),
    DiseaseSignature(
        class_id=1,
        name="Alzheimer's disease",
        short_name="AD",
        preferred_regions=("hippocampus", "temporal", "limbic", "parietal"),
        pattern="symmetric",
        laterality="bilateral",
        size_mm3=(200.0, 4000.0),
        region_count=(1, 6),
        icd_block="ICD-11 6A20 Alzheimer disease",
        summary=(
            "Medial-temporal and parietotemporal atrophy with relative sparing of "
            "sensorimotor cortex and occipital lobes. Bilateral hippocampal volume "
            "loss is an early radiological marker."
        ),
        references=[
            "Dubois B. et al., Lancet Neurol 2014 (NIA-AA criteria)",
            "Scheltens P. et al., Lancet 2021 (AD review)",
        ],
    ),
    DiseaseSignature(
        class_id=2,
        name="Parkinson's disease",
        short_name="PD",
        preferred_regions=("basal_ganglia", "thalamus", "motor_cortex"),
        pattern="symmetric",
        laterality="bilateral",
        size_mm3=(0.0, 800.0),
        region_count=(0, 3),
        icd_block="ICD-11 6A02 Parkinsonism",
        summary=(
            "Nigrostriatal degeneration with relative preservation of cortical "
            "volume. MR may show subtle iron deposition in the putamen; clinical "
            "features dominate over structural lesions."
        ),
        references=[
            "Postuma RB. et al., Mov Disord 2015 (MDS clinical criteria)",
            "Berg D. et al., Lancet Neurol 2013 (prodromal PD)",
        ],
    ),
    DiseaseSignature(
        class_id=3,
        name="Multiple sclerosis",
        short_name="MS",
        preferred_regions=("periventricular", "corona_radiata", "parietal", "occipital"),
        pattern="periventricular",
        laterality="bilateral",
        size_mm3=(50.0, 3000.0),
        region_count=(3, 20),
        icd_block="ICD-11 8A40 Multiple sclerosis",
        summary=(
            "Disseminated periventricular white-matter plaques in space and time, "
            "perpendicular to the ventricles (Dawson's fingers), often involving "
            "the corpus callosum."
        ),
        references=[
            "Thompson AJ. et al., Lancet Neurol 2018 (2017 McDonald criteria)",
            "Filippi M. et al., Lancet Neurol 2018 (MRI in MS)",
        ],
    ),
    DiseaseSignature(
        class_id=4,
        name="Glioma / brain tumour",
        short_name="Glioma",
        preferred_regions=("frontal", "temporal", "parietal", "insula", "occipital"),
        pattern="focal",
        laterality="any",
        size_mm3=(2000.0, 60000.0),
        region_count=(1, 3),
        icd_block="ICD-11 2A00.C0",
        summary=(
            "Focal intra-axial mass with possible mass effect and surrounding "
            "oedema. Low-grade gliomas are T2-hyperintense, well-circumscribed; "
            "high-grade gliomas show necrosis and ring enhancement."
        ),
        references=[
            "Wen PY. et al., Neuro Oncol 2021 (WHO CNS5 classification)",
            "Louis DN. et al., Acta Neuropathol 2021 (WHO CNS5)",
        ],
    ),
    DiseaseSignature(
        class_id=5,
        name="Stroke / cerebral infarct",
        short_name="Stroke",
        preferred_regions=("basal_ganglia", "corona_radiata", "frontal", "parietal", "occipital"),
        pattern="focal",
        laterality="left",  # any single hemisphere — see note below
        size_mm3=(500.0, 80000.0),
        region_count=(1, 2),
        icd_block="ICD-11 6B01-6B04 Cerebral infarction",
        summary=(
            "Focal lesion confined to a vascular territory (e.g. MCA, ACA, PCA), "
            "typically unilateral. Acute infarct is cytotoxic oedema; chronic shows "
            "encephalomalacia."
        ),
        references=[
            "Powers WJ. et al., Stroke 2019 (acute ischaemic stroke guidelines)",
            "Campbell BC. et al., Lancet 2019 (stroke imaging)",
        ],
    ),
    DiseaseSignature(
        class_id=6,
        name="Epilepsy (focal cortical / hippocampal sclerosis)",
        short_name="Epilepsy",
        preferred_regions=("hippocampus", "temporal", "frontal", "occipital"),
        pattern="focal",
        laterality="any",
        size_mm3=(0.0, 1500.0),
        region_count=(1, 2),
        icd_block="ICD-11 8A60-8A6Z Epilepsy & seizures",
        summary=(
            "Mesial temporal sclerosis (hippocampal atrophy with T2 "
            "hyperintensity) or a focal cortical dysplasia. Seizure semiology "
            "must localise the epileptogenic zone."
        ),
        references=[
            "Scheffer IE. et al., Epilepsia 2017 (ILAE classification)",
            "Wieser HG. et al., Epileptic Disord 2004 (mesial temporal sclerosis)",
        ],
    ),
    DiseaseSignature(
        class_id=7,
        name="Huntington's disease",
        short_name="HD",
        preferred_regions=("basal_ganglia",),
        pattern="symmetric",
        laterality="bilateral",
        size_mm3=(200.0, 2500.0),
        region_count=(1, 2),
        icd_block="ICD-11 8A01.1 Huntington disease",
        summary=(
            "Selective, progressive atrophy of the caudate nucleus and putamen "
            "(striatum) with ventricular enlargement of the frontal horns."
        ),
        references=[
            "Tabrizi SJ. et al., Lancet Neurol 2020 (HD)",
            "Reilmann R. et al., Mov Disord 2020 (HD UHDRS)",
        ],
    ),
    DiseaseSignature(
        class_id=8,
        name="Amyotrophic lateral sclerosis",
        short_name="ALS",
        preferred_regions=("motor_cortex", "brainstem", "corona_radiata"),
        pattern="symmetric",
        laterality="bilateral",
        size_mm3=(0.0, 1200.0),
        region_count=(1, 4),
        icd_block="ICD-11 8B62 Amyotrophic lateral sclerosis",
        summary=(
            "Degeneration of the corticospinal tracts (posterior limb of the "
            "internal capsule to the motor cortex) and brainstem motor nuclei. "
            "MRI may show T2 hyperintensity along the corticospinal tract."
        ),
        references=[
            "Hardiman O. et al., Nat Rev Dis Primers 2017 (ALS)",
            "Brooks BR. et al., Amyotroph Lateral Scler 2011 (El Escorial revised)",
        ],
    ),
    DiseaseSignature(
        class_id=9,
        name="Traumatic brain injury",
        short_name="TBI",
        preferred_regions=("frontal", "temporal", "corona_radiata", "brainstem"),
        pattern="focal",
        laterality="any",
        size_mm3=(100.0, 20000.0),
        region_count=(1, 8),
        icd_block="ICD-11 NA0B Traumatic brain injury",
        summary=(
            "Contusional haemorrhages at the frontal and temporal poles (coup-"
            "contrecoup), possible diffuse axonal injury at the grey-white "
            "junction and corpus callosum."
        ),
        references=[
            "Carney N. et al., Neurosurgery 2017 (TBI guidelines)",
            "Murray GD. et al., BMJ 1999 (TBI prognosis, CRASH)",
        ],
    ),
    DiseaseSignature(
        class_id=10,
        name="Meningioma",
        short_name="Meningioma",
        preferred_regions=("extra_axial", "frontal", "parietal", "occipital"),
        pattern="focal",
        laterality="any",
        size_mm3=(1000.0, 50000.0),
        region_count=(1, 2),
        icd_block="ICD-11 2A12.0 Meningioma",
        summary=(
            "Well-circumscribed extra-axial dural-based mass with a broad dural "
            "tail, typically slow-growing. Often convexity or parasagittal; "
            "may compress adjacent cortex."
        ),
        references=[
            "Goldsmith B. et al., J Neurooncol 2014 (WHO meningioma)",
            "Rogers L. et al., Neuro Oncol 2015 (meningioma management)",
        ],
    ),
    DiseaseSignature(
        class_id=11,
        name="Brain metastasis",
        short_name="Metastasis",
        preferred_regions=("gray_white_junction", "frontal", "parietal", "cerebellum"),
        pattern="ring_enhancing",
        laterality="bilateral",
        size_mm3=(100.0, 30000.0),
        region_count=(2, 10),
        icd_block="ICD-11 2A50 Secondary neoplasm",
        summary=(
            "Multiple well-circumscribed lesions at the grey-white junction, "
            "often ring-enhancing with surrounding vasogenic oedema. Lung, "
            "breast, and melanoma are common primaries."
        ),
        references=[
            "Soffietti R. et al., J Clin Oncol 2017 (brain metastases)",
            "Barnholtz-Sloan JS. et al., JCO 2014 (epidemiology)",
        ],
    ),
    DiseaseSignature(
        class_id=12,
        name="MCA territory infarction",
        short_name="MCA_Infarct",
        preferred_regions=("basal_ganglia", "corona_radiata", "frontal", "parietal", "temporal"),
        pattern="focal",
        laterality="left",
        size_mm3=(5000.0, 120000.0),
        region_count=(1, 1),
        icd_block="ICD-11 6B01 Cerebral infarction",
        summary=(
            "Large wedge-shaped lesion confined to the middle cerebral artery "
            "territory, involving the basal ganglia, corona radiata and the "
            "cortical MCA surface. Unilateral, often with mass effect."
        ),
        references=[
            "Powers WJ. et al., Stroke 2019 (acute ischaemic stroke)",
            "Adams HP. et al., Stroke 1993 (TOAST classification)",
        ],
    ),
    DiseaseSignature(
        class_id=13,
        name="Subdural haematoma",
        short_name="SDH",
        preferred_regions=("extra_axial", "frontal", "parietal", "temporal"),
        pattern="diffuse",
        laterality="any",
        size_mm3=(2000.0, 60000.0),
        region_count=(1, 2),
        icd_block="ICD-11 8D10 Nontraumatic subdural haematoma",
        summary=(
            "Extra-axial crescentic collection between dura and arachnoid, "
            "conforming to the calvarium and crossing sutures. May compress "
            "the underlying hemisphere."
        ),
        references=[
            "Adams H. et al., World Neurosurg 2017 (chronic SDH)",
            "Edlmann E. et al., Lancet 2022 (SDH review)",
        ],
    ),
    DiseaseSignature(
        class_id=14,
        name="Normal pressure hydrocephalus",
        short_name="NPH",
        preferred_regions=("ventricular", "periventricular"),
        pattern="symmetric",
        laterality="bilateral",
        size_mm3=(0.0, 500.0),
        region_count=(1, 2),
        icd_block="ICD-11 8D64 Hydrocephalus",
        summary=(
            "Symmetric ventricular enlargement out of proportion to sulcal "
            "atrophy, with periventricular transependymal interstitial oedema. "
            "The classic triad is gait, dementia, and urinary incontinence."
        ),
        references=[
            "Relkin N. et al., Neurosurgery 2005 (NPH guidelines)",
            "Toma K. et al., J Neurol 2020 (NPH imaging)",
        ],
    ),
    DiseaseSignature(
        class_id=15,
        name="Creutzfeldt-Jakob disease",
        short_name="CJD",
        preferred_regions=("basal_ganglia", "thalamus", "frontal", "parietal", "occipital"),
        pattern="symmetric",
        laterality="bilateral",
        size_mm3=(0.0, 1000.0),
        region_count=(1, 6),
        icd_block="ICD-11 8A20 Prion disease",
        summary=(
            "Rapidly progressive dementia with characteristic diffusion "
            "restriction (cortical ribboning and basal-ganglia/thalamic "
            "hyperintensity). Often bilateral and symmetric."
        ),
        references=[
            "Zerr I. et al., Brain 2009 (CJD MRI)",
            "Geschwind MD. et al., J Neurol Neurosurg Psychiatry 2012 (CJD)",
        ],
    ),
    DiseaseSignature(
        class_id=16,
        name="Frontotemporal dementia",
        short_name="FTD",
        preferred_regions=("frontal", "temporal", "limbic"),
        pattern="symmetric",
        laterality="bilateral",
        size_mm3=(0.0, 800.0),
        region_count=(1, 3),
        icd_block="ICD-11 6A82 Frontotemporal dementia",
        summary=(
            "Asymmetric or symmetric atrophy of the frontal and anterior "
            "temporal lobes with relative sparing of the posterior cortex. "
            "Behavioural and language variants are recognised."
        ),
        references=[
            "Rascovsky K. et al., Brain 2011 (bvFTD criteria)",
            "Gorno-Tempini ML. et al., Neurology 2004 (PPA variants)",
        ],
    ),
    DiseaseSignature(
        class_id=17,
        name="Lewy body dementia",
        short_name="LBD",
        preferred_regions=("frontal", "parietal", "occipital", "limbic"),
        pattern="symmetric",
        laterality="bilateral",
        size_mm3=(0.0, 700.0),
        region_count=(1, 4),
        icd_block="ICD-11 6A82 Lewy body dementia",
        summary=(
            "Cortical Lewy body deposition with relative hippocampal "
            "preservation. Occipital hypometabolism on functional imaging is "
            "a supportive feature; structural MRI is less specific."
        ),
        references=[
            "McKeith IG. et al., Neurology 2017 (DLB criteria)",
            "Donaghy PC. et al., Nat Rev Neurol 2015 (DLB)",
        ],
    ),
    DiseaseSignature(
        class_id=18,
        name="Vascular dementia",
        short_name="VaD",
        preferred_regions=("basal_ganglia", "corona_radiata", "periventricular", "frontal"),
        pattern="diffuse",
        laterality="bilateral",
        size_mm3=(100.0, 5000.0),
        region_count=(3, 15),
        icd_block="ICD-11 6B80 Vascular dementia",
        summary=(
            "Diffuse small-vessel disease with multiple lacunes and confluent "
            "white-matter hyperintensities (leukoaraiosis), often bilateral "
            "and periventricular."
        ),
        references=[
            "Gorelick PB. et al., Stroke 2011 (VCI definitions)",
            "Wardlaw JM. et al., Lancet Neurol 2013 (small vessel disease)",
        ],
    ),
    DiseaseSignature(
        class_id=19,
        name="Progressive supranuclear palsy",
        short_name="PSP",
        preferred_regions=("midbrain", "basal_ganglia", "brainstem"),
        pattern="symmetric",
        laterality="bilateral",
        size_mm3=(0.0, 600.0),
        region_count=(1, 2),
        icd_block="ICD-11 8A40.3 PSP",
        summary=(
            "Midbrain atrophy ('hummingbird' sign) with tegmental "
            "degeneration. Relative frontal atrophy and a vertical-gaze "
            "palsy distinguish it from Parkinson's disease."
        ),
        references=[
            "Höglinger GU. et al., Nat Rev Dis Primers 2017 (PSP)",
            "Litvan I. et al., Neurology 2003 (PSP criteria)",
        ],
    ),
    DiseaseSignature(
        class_id=20,
        name="Brain abscess",
        short_name="Abscess",
        preferred_regions=("frontal", "temporal", "parietal", "cerebellum"),
        pattern="ring_enhancing",
        laterality="any",
        size_mm3=(500.0, 30000.0),
        region_count=(1, 3),
        icd_block="ICD-11 1C40 Intracranial abscess",
        summary=(
            "Encapsulated infection with a thin, smooth, ring-enhancing "
            "capsule and surrounding vasogenic oedema. Typically unilateral "
            "and unilocular; often frontal or temporal."
        ),
        references=[
            "Brouwer MC. et al., N Engl J Med 2014 (brain abscess)",
            "Mathisen GE. et al., Infect Dis Clin North Am 2010 (abscess)",
        ],
    ),
]


# Fast lookups.
_BY_ID = {d.class_id: d for d in DISEASE_TAXONOMY}


def get_disease(class_id: int) -> DiseaseSignature:
    """Return the disease signature for ``class_id``."""
    return _BY_ID[class_id]


def num_classes() -> int:
    return len(DISEASE_TAXONOMY)


def disease_names() -> list[str]:
    """Return short disease names ordered by class id."""
    return [d.short_name for d in DISEASE_TAXONOMY]
