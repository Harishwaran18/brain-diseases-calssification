"""Curated library of neuroregenerative therapy techniques.

Each :class:`TherapyTechnique` is a parameterized curing strategy mapped to a
disease class and the affected brain region. The parameters feed directly into
the :class:`~brainframe.evaluation.therapy_model.TherapySpec` consumed by the
simulator, so a recommendation can be turned into a runnable simulation with
no extra translation layer.

Disease classes (matching the classifier output):
    0 -> healthy / no neurodegeneration
    1 -> early-stage neurodegenerative changes (prodromal Alzheimer's / MCI)
    2 -> moderate neurodegeneration (Alzheimer's / Parkinson's spectrum)
    3 -> advanced neurodegeneration (severe atrophy / large lesion burden)

The library is deliberately self-contained and literature-informed; no
technique is a stub -- each carries a dose, radius, kernel, and an expected
mechanism that the simulator realizes as a recovery field.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TherapyTechnique:
    """A parameterized curing/therapy technique."""

    id: str
    name: str
    disease_class: int
    mode: str  # stimulation | regeneration | lesion_reversal
    target_mode: str  # centroid | largest_region | manual
    radius_mm: float
    dose: float  # 0..1 intensity of intervention
    sigma_mm: float
    kernel: str = "gaussian"
    target_label: str = "lesion"
    rationale: str = ""
    expected_effect: str = ""
    references: list[str] = field(default_factory=list)

    def to_therapy_spec_dict(self) -> dict:
        """Return a dict matching ``TherapySpec`` fields for the simulator."""
        return {
            "target_label": self.target_label,
            "target_mode": self.target_mode,
            "radius_mm": self.radius_mm,
            "dose": self.dose,
            "mode": self.mode,
            "kernel": self.kernel,
            "sigma_mm": self.sigma_mm,
        }


# -- The curated library ---------------------------------------------------

TECHNIQUE_LIBRARY: list[TherapyTechnique] = [
    TherapyTechnique(
        id="preventive_stimulation",
        name="Transcranial cognitive stimulation (preventive)",
        disease_class=0,
        mode="stimulation",
        target_mode="largest_region",
        radius_mm=20.0,
        dose=0.35,
        sigma_mm=10.0,
        rationale=(
            "No neurodegeneration detected. Low-dose cortical stimulation is applied as a "
            "preventive neuroprotective measure to sustain gray-matter integrity."
        ),
        expected_effect="Slight gray-matter intensity enhancement; no lesion to reverse.",
        references=["Lubenova et al., 2019 (tDCS neuroprotection)"],
    ),
    TherapyTechnique(
        id="anti_amyloid_regen",
        name="Anti-amyloid immunotherapy + regenerative stimulation",
        disease_class=1,
        mode="regeneration",
        target_mode="largest_region",
        radius_mm=24.0,
        dose=0.55,
        sigma_mm=10.0,
        rationale=(
            "Alzheimer's-pattern medial-temporal / parietal involvement. Anti-amyloid "
            "monoclonal antibody (lecanemab) combined with a regenerative field to slow "
            "hippocampal atrophy and partially shrink plaque-associated signal."
        ),
        expected_effect="Slowed atrophy; ~15-30% reduction of the target region burden.",
        references=[
            "van Dyck CH. et al., NEJM 2023 (lecanemab, Clarity AD)",
            "Salloway S. et al., NEJM 2024 (anti-amyloid therapy)",
        ],
    ),
    TherapyTechnique(
        id="dopaminergic_stimulation",
        name="Dopaminergic neuromodulation + subthalamic stimulation",
        disease_class=2,
        mode="stimulation",
        target_mode="largest_region",
        radius_mm=22.0,
        dose=0.65,
        sigma_mm=11.0,
        rationale=(
            "Parkinson's-pattern nigrostriatal involvement. Deep-brain stimulation of "
            "the subthalamic nucleus models dopaminergic rescue and improves "
            "basal-ganglia functional reserve."
        ),
        expected_effect="Restoration of basal-ganglia signal toward control intensity.",
        references=[
            "Volkmann J. et al., N Engl J Med 2012 (STN-DBS for PD)",
            "Limousin P. et al., Lancet Neurol 1998 (STN stimulation)",
        ],
    ),
    TherapyTechnique(
        id="dmt_remyelination",
        name="Disease-modifying remyelination therapy (DMT)",
        disease_class=3,  # Multiple sclerosis
        mode="regeneration",
        target_mode="largest_region",
        radius_mm=26.0,
        dose=0.7,
        sigma_mm=11.0,
        rationale=(
            "Multiple-sclerosis-pattern periventricular demyelination. A remyelinating "
            "field ( modelling anti-CD20 + remyelinating agent effect ) targets the "
            "largest periventricular plaque to restore myelin signal and reduce plaque "
            "volume. Bilateral plaques are addressed by re-running the field per region."
        ),
        expected_effect="25-45% reduction of periventricular plaque volume.",
        references=[
            "Hauser SL. et al., NEJM 2017 (ocrelizumab, ORATORIO)",
            "Green AJ. et al., Nature 2017 (clemastine remyelination)",
        ],
    ),
    TherapyTechnique(
        id="tumor_resection",
        name="Image-guided tumour resection + adjuvant field",
        disease_class=4,  # Glioma / tumour
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=30.0,
        dose=0.95,
        sigma_mm=13.0,
        rationale=(
            "Focal intra-axial mass. The reversal field models maximal safe resection "
            "of the enhancing tumour core followed by a conformal adjuvant field to the "
            "residual cavity. Risk scales with eloquence of adjacent cortex."
        ),
        expected_effect="Gross-total resection of the modelled tumour core (>70% reduction).",
        references=[
            "Stummer W. et al., Lancet Oncol 2006 (5-ALA fluorescence-guided resection)",
            "Wen PY. et al., Neuro Oncol 2021 (WHO CNS5)",
        ],
    ),
    TherapyTechnique(
        id="thrombolysis_recovery",
        name="Thrombolytic reperfusion + neuroprotective field",
        disease_class=5,  # Stroke / cerebral infarct
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=32.0,
        dose=0.85,
        sigma_mm=13.0,
        rationale=(
            "Focal vascular-territory infarct. The reversal field models acute "
            "reperfusion (tPA/mechanical thrombectomy) shrinking the ischaemic core, "
            "with a neuroprotective penumbra field to salvage at-risk tissue. "
            "Time-window dependent; greatest effect when applied acutely."
        ),
        expected_effect="35-55% core-volume reduction (acute window); less in chronic infarcts.",
        references=[
            "Nogueira RG. et al., NEJM 2018 (DAWN thrombectomy)",
            "Powers WJ. et al., Stroke 2019 (AIS guidelines)",
        ],
    ),
    TherapyTechnique(
        id="epileptogenic_ablation",
        name="Epileptogenic-zone focal ablation",
        disease_class=6,  # Epilepsy
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=18.0,
        dose=0.75,
        sigma_mm=9.0,
        rationale=(
            "Focal epileptogenic lesion (e.g. hippocampal sclerosis or cortical "
            "dysplasia). A tightly conformal ablation field models laser interstitial "
            "thermal therapy or surgical resection of the epileptogenic zone, "
            "prioritising seizure freedom over volume reduction."
        ),
        expected_effect="60-80% reduction of the epileptogenic focus volume.",
        references=[
            "Engel J. et al., Epilepsia 2012 (ILAE surgery outcome)",
            "Gross RE. et al., Neurology 2020 (stereo-EEG + LaserITT)",
        ],
    ),
    TherapyTechnique(
        id="huntington_neuroprotect",
        name="Striatal neuroprotective modulation",
        disease_class=7,  # Huntington's disease
        mode="regeneration",
        target_mode="largest_region",
        radius_mm=20.0,
        dose=0.6,
        sigma_mm=10.0,
        rationale=(
            "Selective striatal (caudate/putamen) atrophy. A neuroprotective field "
            "targets the striatum to slow the rate of medium-spiny-neuron loss and "
            "model emerging huntingtin-lowering therapy. Curative potential is limited; "
            "the simulation shows slowed progression rather than reversal."
        ),
        expected_effect="Slowed caudate atrophy; modest (~10-20%) signal recovery.",
        references=[
            "Tabrizi SJ. et al., Lancet Neurol 2022 (tominersen, GENERATION HD1)",
            "Wild EJ. et al., Lancet Neurol 2015 (HD biomarkers)",
        ],
    ),
    TherapyTechnique(
        id="als_motor_protection",
        name="Corticospinal-tract neuroprotection",
        disease_class=8,  # ALS
        mode="regeneration",
        target_mode="largest_region",
        radius_mm=24.0,
        dose=0.5,
        sigma_mm=11.0,
        rationale=(
            "Corticospinal-tract and motor-cortex degeneration. A diffuse neuroprotective "
            "field models anti-oxidant + TDP-43-targeted therapy along the motor pathway. "
            "The disease is relentlessly progressive; the simulation shows modest slowing."
        ),
        expected_effect="Slowed CST degeneration; limited focal recovery (~10-15%).",
        references=[
            "van Es MA. et al., Lancet Neurol 2017 (ALS review)",
            "Andrews JA. et al., Muscle Nerve 2020 (edaravone / riluzole)",
        ],
    ),
    TherapyTechnique(
        id="tbi_repair",
        name="Contusion debridement + axonal regenerative field",
        disease_class=9,  # TBI
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=28.0,
        dose=0.8,
        sigma_mm=12.0,
        rationale=(
            "Traumatic contusional haemorrhage + diffuse axonal injury. The reversal "
            "field debrides the dominant haemorrhagic contusion while a wide regenerative "
            "field promotes repair at the grey-white junction and corpus callosum."
        ),
        expected_effect="35-55% reduction of the dominant contusion; diffuse recovery limited.",
        references=[
            "Carney N. et al., Neurosurgery 2017 (TBI guidelines)",
            "Bullock MR. et al., Neurosurgery 2006 (TBI surgical management)",
        ],
    ),
    TherapyTechnique(
        id="meningioma_resection",
        name="Microsurgical resection (Simpson grade I) + dural tail excision",
        disease_class=10,  # Meningioma
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=22.0,
        dose=0.9,
        sigma_mm=10.0,
        rationale=(
            "Meningiomas are benign, dural-based tumours. Complete resection with "
            "the involved dura and bone (Simpson I) offers the best progression-free "
            "survival. Stereotactic radiosurgery is an adjunct for residual tumour."
        ),
        expected_effect="60-80% reduction of the mass; Simpson I resection is often curative.",
        references=[
            "Simpson D. J Neurol Neurosurg Psychiatry 1957 (Simpson grading)",
            "Goldsmith B. et al., J Neurooncol 2014 (WHO meningioma)",
        ],
    ),
    TherapyTechnique(
        id="metastasis_radiosurgery",
        name="Stereotactic radiosurgery (SRS) + targeted systemic therapy",
        disease_class=11,  # Metastasis
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=18.0,
        dose=0.78,
        sigma_mm=9.0,
        rationale=(
            "Brain metastases are treated with SRS (Gamma Knife / LINAC) to each "
            "lesion, combined with targeted systemic therapy (e.g. anti-PD-1 for "
            "melanoma, TKI for EGFR+ lung). This shrinks the dominant lesion while "
            "controlling satellite lesions."
        ),
        expected_effect="50-70% reduction of dominant metastasis; local control ~85%.",
        references=[
            "Soffietti R. et al., J Clin Oncol 2017 (brain metastases)",
            "Brown PD. et al., Lancet Oncol 2017 (SRS vs WBRT)",
        ],
    ),
    TherapyTechnique(
        id="mca_reperfusion",
        name="Mechanical thrombectomy + decompressive hemicraniectomy",
        disease_class=12,  # MCA infarction
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=30.0,
        dose=0.7,
        sigma_mm=14.0,
        rationale=(
            "Large MCA infarcts need reperfusion (thrombectomy within 24h of DAWN/"
            "DEFUSE-3) and decompressive hemicraniectomy within 48h for malignant "
            "edema. The reversal field simulates penumbra salvage and oedema control."
        ),
        expected_effect="40-60% penumbra salvage; reduced mass effect; limited core recovery.",
        references=[
            "Nogueira RG. et al., NEJM 2018 (DAWN)",
            "Albers GW. et al., NEJM 2018 (DEFUSE-3)",
        ],
    ),
    TherapyTechnique(
        id="sdh_evacuation",
        name="Surgical evacuation (burr-hole / mini-craniotomy)",
        disease_class=13,  # Subdural haematoma
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=26.0,
        dose=0.85,
        sigma_mm=11.0,
        rationale=(
            "Subdural haematomas are drained via burr-hole or mini-craniotomy. "
            "Symptomatic collections >10 mm or with >5 mm midline shift are "
            "evacuated. The reversal field drains the dominant collection."
        ),
        expected_effect="70-90% reduction of the collection; relief of mass effect.",
        references=[
            "Adams H. et al., World Neurosurg 2017 (chronic SDH)",
            "Edlmann E. et al., Lancet 2022 (SDH review)",
        ],
    ),
    TherapyTechnique(
        id="nph_shunt",
        name="Ventriculoperitoneal (VP) shunt placement",
        disease_class=14,  # NPH
        mode="stimulation",
        target_mode="centroid",
        radius_mm=14.0,
        dose=0.75,
        sigma_mm=8.0,
        rationale=(
            "Normal pressure hydrocephalus is treated by CSF diversion via a VP "
            "shunt (or programmable valve) following a positive tap test. The "
            "stimulation field simulates normalised CSF dynamics."
        ),
        expected_effect="Reduction of ventricular dilation; gait and cognition improve in ~60%.",
        references=[
            "Relkin N. et al., Neurosurgery 2005 (NPH guidelines)",
            "Toma K. et al., J Neurol 2020 (NPH imaging)",
        ],
    ),
    TherapyTechnique(
        id="cjd_supportive",
        name="Supportive care + experimental PRP-targeted therapy",
        disease_class=15,  # CJD
        mode="stimulation",
        target_mode="largest_region",
        radius_mm=16.0,
        dose=0.3,
        sigma_mm=10.0,
        rationale=(
            "Creutzfeldt-Jakob disease has no proven disease-modifying therapy. "
            "Care is supportive (symptom control); experimental anti-prion agents "
            "(PRP replication inhibitors) are trialled. The simulation shows only "
            "modest slowing of progression."
        ),
        expected_effect="Modest symptomatic relief; disease course largely unchanged.",
        references=[
            "Geschwind MD. et al., J Neurol Neurosurg Psychiatry 2012 (CJD)",
            "Zerr I. et al., Brain 2009 (CJD MRI)",
        ],
    ),
    TherapyTechnique(
        id="ftd_symptomatic",
        name="Symptomatic therapy + behavioural support + tau-targeted trial",
        disease_class=16,  # FTD
        mode="stimulation",
        target_mode="largest_region",
        radius_mm=15.0,
        dose=0.5,
        sigma_mm=9.0,
        rationale=(
            "Frontotemporal dementia lacks disease-modifying therapy; SSRIs manage "
            "behavioural symptoms. Tau-targeted antibodies (e.g. semorinemab) are "
            "in trials for tau-variant FTD."
        ),
        expected_effect="Behavioural symptom improvement ~40-50%; structural change limited.",
        references=[
            "Rascovsky K. et al., Brain 2011 (bvFTD criteria)",
            "Boxer AL. et al., Nat Rev Neurol 2013 (FTD therapy)",
        ],
    ),
    TherapyTechnique(
        id="lbd_cholinesterase",
        name="Cholinesterase inhibitor + levodopa-sparing parkinsonism care",
        disease_class=17,  # LBD
        mode="stimulation",
        target_mode="largest_region",
        radius_mm=15.0,
        dose=0.55,
        sigma_mm=9.0,
        rationale=(
            "Lewy body dementia responds well to cholinesterase inhibitors "
            "(donepezil, rivastigmine). Levodopa is used cautiously for "
            "parkinsonism (avoid antipsychotics due to neuroleptic sensitivity)."
        ),
        expected_effect="Cognitive and hallucination improvement ~50-60%; modest.",
        references=[
            "McKeith IG. et al., Neurology 2017 (DLB criteria)",
            "Donaghy PC. et al., Nat Rev Neurol 2015 (DLB)",
        ],
    ),
    TherapyTechnique(
        id="vad_risk_control",
        name="Vascular risk-factor control + cognitive rehabilitation",
        disease_class=18,  # Vascular dementia
        mode="stimulation",
        target_mode="largest_region",
        radius_mm=16.0,
        dose=0.6,
        sigma_mm=10.0,
        rationale=(
            "Vascular dementia management targets secondary prevention: "
            "antihypertensives, statins, antiplatelets, and glycaemic control, "
            "plus cognitive rehabilitation. This slows progression of "
            "small-vessel disease."
        ),
        expected_effect="Slowed progression; ~30-40% reduction in new infarcts.",
        references=[
            "Gorelick PB. et al., Stroke 2011 (VCI definitions)",
            "Wardlaw JM. et al., Lancet Neurol 2013 (small vessel disease)",
        ],
    ),
    TherapyTechnique(
        id="psp_symptomatic",
        name="Symptomatic therapy + gait/balance rehabilitation",
        disease_class=19,  # PSP
        mode="stimulation",
        target_mode="largest_region",
        radius_mm=12.0,
        dose=0.35,
        sigma_mm=8.0,
        rationale=(
            "Progressive supranuclear palsy has no disease-modifying therapy. "
            "Levodopa gives limited benefit; physical therapy for gait/balance, "
            "and botulinum toxin for eyelid apraxia, are mainstays."
        ),
        expected_effect="Modest gait/symptom improvement; progression continues.",
        references=[
            "Höglinger GU. et al., Nat Rev Dis Primers 2017 (PSP)",
            "Litvan I. et al., Neurology 2003 (PSP criteria)",
        ],
    ),
    TherapyTechnique(
        id="abscess_drainage",
        name="Surgical drainage + targeted antibiotic therapy",
        disease_class=20,  # Brain abscess
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=20.0,
        dose=0.88,
        sigma_mm=10.0,
        rationale=(
            "Brain abscesses are drained (stereotactic aspiration or excision) "
            "and treated with 4-6 weeks of targeted IV antibiotics based on "
            "culture. The reversal field drains the dominant collection while "
            "antibiotics sterilise the cavity."
        ),
        expected_effect="70-90% resolution of the abscess; full recovery common.",
        references=[
            "Brouwer MC. et al., N Engl J Med 2014 (brain abscess)",
            "Mathisen GE. et al., Infect Dis Clin North Am 2010 (abscess)",
        ],
    ),
    TherapyTechnique(
        id="cbd_symptomatic_tau",
        name="Symptomatic therapy + tau-targeted experimental immunotherapy",
        disease_class=21,  # Corticobasal degeneration
        mode="stimulation",
        target_mode="largest_region",
        radius_mm=14.0,
        dose=0.35,
        sigma_mm=9.0,
        rationale=(
            "Corticobasal degeneration has no disease-modifying therapy. "
            "Levodopa is trialled (limited benefit); clonazepam/levetiracetam "
            "for myoclonus. Tau-targeted antibodies (semorinemab) are under "
            "investigation. Stimulation field models symptomatic cortical support."
        ),
        expected_effect="Modest symptomatic relief; progression continues.",
        references=[
            "Armstrong MJ. et al., Neurology 2013 (CBD criteria)",
            "Boxer AL. et al., Nat Rev Neurol 2013 (tau-targeted therapy)",
        ],
    ),
    TherapyTechnique(
        id="msa_supportive",
        name="Symptomatic therapy + autonomic support + physical rehabilitation",
        disease_class=22,  # Multiple system atrophy
        mode="stimulation",
        target_mode="largest_region",
        radius_mm=15.0,
        dose=0.30,
        sigma_mm=9.0,
        rationale=(
            "Multiple system atrophy has no disease-modifying therapy. "
            "Levodopa for parkinsonism (limited); midodrine/fludrocortisone for "
            "orthostatic hypotension; continuous positive airway pressure for "
            "stridor; physical therapy for gait/balance."
        ),
        expected_effect="Symptomatic improvement; disease-modifying therapy remains elusive.",
        references=[
            "Wenning GK. et al., Lancet Neurol 2022 (MSA review)",
            "Fanciulli A. et al., Mov Disord 2021 (MSA management consensus)",
        ],
    ),
    TherapyTechnique(
        id="cadasil_risk_control",
        name="Antiplatelet + vascular risk control + cognitive rehabilitation",
        disease_class=23,  # CADASIL
        mode="stimulation",
        target_mode="largest_region",
        radius_mm=16.0,
        dose=0.45,
        sigma_mm=10.0,
        rationale=(
            "CADASIL (NOTCH3 mutation) has no specific disease-modifying therapy. "
            "Management targets secondary stroke prevention: antiplatelets (avoid "
            "anticoagulation due to haemorrhage risk), aggressive vascular risk "
            "factor control, and cognitive rehabilitation."
        ),
        expected_effect="Slowed progression of white-matter lesions and cognitive decline.",
        references=[
            "Chabriat H. et al., Lancet Neurol 2009 (CADASIL)",
            "Adib-Samii P. et al., Stroke 2010 (CADASIL management)",
        ],
    ),
    TherapyTechnique(
        id="sah_clipping_coiling",
        name="Aneurysm securing (clip/coil) + nimodipine + vasospasm management",
        disease_class=24,  # Subarachnoid haemorrhage
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=24.0,
        dose=0.80,
        sigma_mm=12.0,
        rationale=(
            "Ruptured aneurysm requires urgent securing (endovascular coiling or "
            "surgical clipping). Nimodipine (21 days) prevents delayed cerebral "
            "ischaemia from vasospasm. The reversal field models clot clearance "
            "and vasospasm resolution."
        ),
        expected_effect="Secured aneurysm; reduced rebleeding risk; vasospasm mitigation.",
        references=[
            "Connolly ES. et al., Stroke 2012 (SAH guidelines)",
            "Molyneux AJ. et al., Lancet 2005 (ISAT coiling vs clipping)",
        ],
    ),
    TherapyTechnique(
        id="edh_evacuation",
        name="Emergency surgical evacuation (craniotomy)",
        disease_class=25,  # Epidural haematoma
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=28.0,
        dose=0.92,
        sigma_mm=12.0,
        rationale=(
            "Epidural haematoma is a neurosurgical emergency. Evacuation via "
            "craniotomy (or burr-hole drainage if craniotomy unavailable) relieves "
            "mass effect. The reversal field models rapid clot evacuation with "
            "brain re-expansion."
        ),
        expected_effect="80-95% clot evacuation; full neurological recovery if timely.",
        references=[
            "Bullock MR. et al., Neurosurgery 2006 (TBI guidelines, EDH)",
            "Narayan RK. et al., J Neurotrauma 2002 (EDH management)",
        ],
    ),
    TherapyTechnique(
        id="avm_embolization",
        name="Multimodal AVM treatment (embolization + surgery/radiosurgery)",
        disease_class=26,  # Arteriovenous malformation
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=22.0,
        dose=0.75,
        sigma_mm=11.0,
        rationale=(
            "AVM treatment depends on Spetzler-Martin grade: endovascular "
            "embolization (ONYX/nBCA), microsurgical resection, or stereotactic "
            "radiosurgery (Gamma Knife), often combined. The reversal field "
            "models progressive nidus obliteration."
        ),
        expected_effect="60-90% nidus obliteration; haemorrhage risk eliminated.",
        references=[
            "Spetzler RF. et al., J Neurosurg 1986 (AVM grading)",
            "Al-Shahi Salman R. et al., Lancet 2012 (AVM management)",
        ],
    ),
    TherapyTechnique(
        id="cavernoma_resection",
        name="Microsurgical resection + haemosiderin rim excision",
        disease_class=27,  # Cavernous malformation
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=18.0,
        dose=0.78,
        sigma_mm=9.0,
        rationale=(
            "Cavernous malformations are resected if symptomatic (haemorrhage, "
            "seizures) and surgically accessible. Complete excision including the "
            "haemosiderin rim reduces re-bleeding and seizure recurrence. Deep/"
            "brainstem lesions may be observed or treated with radiosurgery."
        ),
        expected_effect="75-90% complete resection; seizure freedom in 70-80%.",
        references=[
            "Moran NF. et al., Brain 1999 (cavernoma natural history)",
            "Zabramski JM. et al., J Neurosurg 1994 (cavernoma classification)",
        ],
    ),
    TherapyTechnique(
        id="arachnoid_fenestration",
        name="Endoscopic fenestration / cystoperitoneal shunt",
        disease_class=28,  # Arachnoid cyst
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=20.0,
        dose=0.70,
        sigma_mm=10.0,
        rationale=(
            "Symptomatic arachnoid cysts are treated by endoscopic fenestration "
            "(into the ventricle or cisterns) or cystoperitoneal shunting. The "
            "reversal field models cyst decompression and brain re-expansion. "
            "Asymptomatic cysts are monitored."
        ),
        expected_effect="60-80% cyst volume reduction; symptom resolution.",
        references=[
            "Pradilla G. et al., Neurosurgery 2013 (arachnoid cysts)",
            "Al-Holou WN. et al., J Neurosurg 2010 (arachnoid cyst epidemiology)",
        ],
    ),
    TherapyTechnique(
        id="colloid_resection",
        name="Neuroendoscopic resection / microsurgical removal",
        disease_class=29,  # Colloid cyst
        mode="lesion_reversal",
        target_mode="centroid",
        radius_mm=12.0,
        dose=0.85,
        sigma_mm=7.0,
        rationale=(
            "Colloid cysts at the foramen of Monro cause obstructive "
            "hydrocephalus and potential sudden death. Treated by "
            "neuroendoscopic fenestration/resection or microsurgical removal "
            "via a transcortical/transcallosal approach. The reversal field "
            "models cyst evacuation with CSF flow restoration."
        ),
        expected_effect="85-95% complete resection; hydrocephalus resolved.",
        references=[
            "Pollock BE. et al., Neurosurgery 2000 (colloid cyst management)",
            "Desai KI. et al., J Neurosurg 2002 (colloid cyst review)",
        ],
    ),
    TherapyTechnique(
        id="pituitary_resection",
        name="Transsphenoidal resection + hormonal management",
        disease_class=30,  # Pituitary adenoma
        mode="lesion_reversal",
        target_mode="centroid",
        radius_mm=16.0,
        dose=0.82,
        sigma_mm=9.0,
        rationale=(
            "Pituitary macroadenomas are resected via endoscopic transsphenoidal "
            "approach (microadenomas may be medical: dopamine agonists for "
            "prolactinomas, somatostatin analogues for GH adenomas). The reversal "
            "field models tumour debulking with chiasmal decompression."
        ),
        expected_effect="70-90% tumour resection; visual and hormonal recovery.",
        references=[
            "Molitch ME. et al., J Clin Endocrinol Metab 2011 (prolactinoma)",
            "Freda PU. et al., J Clin Endocrinol Metab 2011 (acromegaly)",
        ],
    ),
    TherapyTechnique(
        id="schwannoma_resection",
        name="Microsurgical resection / stereotactic radiosurgery",
        disease_class=31,  # Vestibular schwannoma
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=16.0,
        dose=0.80,
        sigma_mm=9.0,
        rationale=(
            "Vestibular schwannomas are managed by microsurgical resection "
            "(retrosigmoid/translabyrinthine) or stereotactic radiosurgery "
            "(Gamma Knife), or observation for small/asymptomatic lesions. The "
            "reversal field models tumour volume reduction with facial/cochlear "
            "nerve preservation."
        ),
        expected_effect="75-95% tumour control; hearing preservation variable.",
        references=[
            "Hasegawa T. et al., J Neurosurg 2005 (schwannoma gamma knife)",
            "Carlson ML. et al., J Neurosurg 2013 (vestibular schwannoma review)",
        ],
    ),
    TherapyTechnique(
        id="pml_antiviral",
        name="Immune reconstitution + antiviral therapy",
        disease_class=32,  # PML
        mode="regeneration",
        target_mode="largest_region",
        radius_mm=22.0,
        dose=0.50,
        sigma_mm=11.0,
        rationale=(
            "Progressive multifocal leukoencephalopathy (JC virus) requires "
            "immune reconstitution (HAART for HIV, plasmapheresis for "
            "natalizumab-associated PML). Mefloquine and mirtazapine have "
            "anti-JC virus activity in vitro. The regeneration field models "
            "white-matter recovery following immune restoration."
        ),
        expected_effect="30-50% lesion stabilisation/recovery if immune restored; IRIS risk.",
        references=[
            "Berger JR. et al., N Engl J Med 2012 (PML review)",
            "Major EO. et al., Lancet Neurol 2010 (PML pathogenesis)",
        ],
    ),
    TherapyTechnique(
        id="cpm_correction",
        name="Gradual correction + supportive care + osmotic management",
        disease_class=33,  # Central pontine myelinolysis
        mode="regeneration",
        target_mode="centroid",
        radius_mm=14.0,
        dose=0.40,
        sigma_mm=9.0,
        rationale=(
            "Central pontine myelinolysis is prevented by gradual sodium "
            "correction (<8-10 mmol/L/24h). Once established, treatment is "
            "supportive; experimental therapies include IV immunoglobulin, "
            "thyrotropin-releasing hormone, and plasmapheresis. The regeneration "
            "field models partial myelin recovery."
        ),
        expected_effect="Variable recovery; 30-50% of survivors have residual deficits.",
        references=[
            "Sterns RH. et al., N Engl J Med 1986 (osmotic demyelination)",
            "Singh TD. et al., Mayo Clin Proc 2015 (CPM review)",
        ],
    ),
    TherapyTechnique(
        id="radnecrosis_bevacizumab",
        name="Bevacizumab (anti-VEGF) + surgical debulking",
        disease_class=34,  # Radiation necrosis
        mode="lesion_reversal",
        target_mode="largest_region",
        radius_mm=18.0,
        dose=0.65,
        sigma_mm=10.0,
        rationale=(
            "Radiation necrosis is treated with bevacizumab (anti-VEGF, reduces "
            "vasogenic oedema and necrosis), hyperbaric oxygen, or surgical "
            "resection for mass effect. The reversal field models anti-VEGF-"
            "induced reduction of the necrotic lesion and surrounding oedema."
        ),
        expected_effect="40-60% lesion regression; significant oedema reduction.",
        references=[
            "Chao ST. et al., J Neurooncol 2015 (radiation necrosis)",
            "Sundgren PC. et al., Radiology 2004 (radiation necrosis MRS)",
        ],
    ),
    TherapyTechnique(
        id="wernicke_thiamine",
        name="Urgent intravenous thiamine replacement",
        disease_class=35,  # Wernicke encephalopathy
        mode="regeneration",
        target_mode="centroid",
        radius_mm=14.0,
        dose=0.85,
        sigma_mm=8.0,
        rationale=(
            "Wernicke encephalopathy is a medical emergency requiring prompt IV "
            "thiamine (500mg TDS for 3-5 days, then oral maintenance) BEFORE "
            "glucose. The regeneration field models rapid reversal of mammillary "
            "body and thalamic signal abnormality. Delayed treatment leads to "
            "Korsakoff syndrome."
        ),
        expected_effect="80-95% signal reversal if treated promptly; prevents Korsakoff.",
        references=[
            "Zuccoli G. et al., Radiology 2009 (Wernicke MRI)",
            "Sechi G. et al., Lancet Neurol 2002 (Wernicke review)",
        ],
    ),
]


def techniques_for_class(disease_class: int) -> list[TherapyTechnique]:
    """Return all techniques applicable to a given disease class."""
    return [t for t in TECHNIQUE_LIBRARY if t.disease_class == disease_class]


def get_technique(technique_id: str) -> TherapyTechnique | None:
    """Look up a technique by id."""
    for t in TECHNIQUE_LIBRARY:
        if t.id == technique_id:
            return t
    return None


def default_technique(disease_class: int) -> TherapyTechnique:
    """Return the canonical recommended technique for a disease class."""
    matches = techniques_for_class(disease_class)
    if matches:
        return matches[0]
    # Fall back to the closest available class.
    best = min(TECHNIQUE_LIBRARY, key=lambda t: abs(t.disease_class - disease_class))
    return best
