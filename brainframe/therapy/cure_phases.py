"""Medically-accurate multi-phase cure cascade for the real human brain.

Real neurotherapy does not happen in a single step. A disease-modifying
intervention unfolds as a *biological cascade* of distinct phases, each with
its own mechanism, timescale, and visible effect on brain tissue. This module
models that cascade as a sequence of :class:`CurePhase` objects and produces a
:class:`CureTimeline` — a sampled, frame-by-frame description of how the lesion
and surrounding tissue evolve over the course of treatment.

The six canonical phases (not every disease uses all six):

1. **Targeting / delivery** — the therapeutic agent reaches the lesion
   (antibody crosses BBB, radiation converges, resection planned). Visualised
   as a converging therapeutic field around the lesion.
2. **Anti-inflammatory / anti-oedema** — peri-lesional swelling and
   inflammation subside. The oedema halo fades; lesion volume dips slightly.
3. **Lesion reversal / resection** — the lesion core itself shrinks, is
   resected, or undergoes apoptosis. The dominant visible change.
4. **Neuroprotection** — at-risk penumbra tissue around the lesion is
   stabilised. A protective field shields viable neurons.
5. **Remyelination / regeneration** — white-matter myelin is restored and
   healthy tissue regrows into the lesion cavity. New healthy mesh appears.
6. **Functional recovery** — synapses reconnect; the cavity is filled with
   restored tissue. Final near-normal architecture.

Each disease class maps to a tailored subset/ordering of these phases, so the
animation is anatomically and therapeutically honest rather than a generic
"shrink a red blob".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brainframe.utils.logging import get_logger

log = get_logger("therapy.cure_phases")


@dataclass(frozen=True)
class CurePhase:
    """A single biological phase of a curative cascade."""

    id: str
    name: str
    mechanism: str
    weight: float
    lesion_scale_end: float
    edema_end: float
    regen_end: float
    protect_end: float
    delivery: float
    description: str
    color: str


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# -- Phase templates -------------------------------------------------------

_TARGETING: dict[str, Any] = {
    "id": "targeting",
    "name": "Therapeutic Targeting & Delivery",
    "mechanism": "drug_delivery",
    "weight": 0.14,
    "lesion_scale_end": 1.0,
    "edema_end": 1.0,
    "regen_end": 0.0,
    "protect_end": 0.0,
    "delivery": 0.9,
    "description": (
        "The therapeutic agent is delivered to the lesion site. For antibodies "
        "this is BBB penetration; for surgery it is stereotactic localisation; "
        "for radiation it is beam convergence."
    ),
    "color": "#3aa6e6",
}

_ANTI_INFLAM: dict[str, Any] = {
    "id": "anti_inflammatory",
    "name": "Anti-Inflammatory & Anti-Oedema",
    "mechanism": "inflammation_resolution",
    "weight": 0.13,
    "lesion_scale_end": 0.92,
    "edema_end": 0.45,
    "regen_end": 0.0,
    "protect_end": 0.2,
    "delivery": 0.5,
    "description": (
        "Corticosteroids and endogenous resolution pathways reduce peri-lesional "
        "oedema and inflammation. Swelling subsides; mass effect eases."
    ),
    "color": "#f0a040",
}

_REVERSAL: dict[str, Any] = {
    "id": "lesion_reversal",
    "name": "Lesion Reversal / Resection",
    "mechanism": "lesion_clearance",
    "weight": 0.23,
    "lesion_scale_end": 0.18,
    "edema_end": 0.15,
    "regen_end": 0.05,
    "protect_end": 0.6,
    "delivery": 0.4,
    "description": (
        "The lesion core is actively reversed — resected, lysed, or cleared by "
        "immune effectors. This is the dominant volumetric change of the cure."
    ),
    "color": "#ff2b4a",
}

_NEUROPROTECT: dict[str, Any] = {
    "id": "neuroprotection",
    "name": "Neuroprotection of Penumbra",
    "mechanism": "penumbra_salvage",
    "weight": 0.17,
    "lesion_scale_end": 0.12,
    "edema_end": 0.05,
    "regen_end": 0.15,
    "protect_end": 0.95,
    "delivery": 0.2,
    "description": (
        "At-risk tissue in the peri-lesional penumbra is stabilised. Free-radical "
        "scavenging and excitotoxicity blockade salvage viable neurons."
    ),
    "color": "#9a7ad0",
}

_REGEN: dict[str, Any] = {
    "id": "remyelination",
    "name": "Remyelination & Tissue Regeneration",
    "mechanism": "myelin_repair",
    "weight": 0.18,
    "lesion_scale_end": 0.05,
    "edema_end": 0.0,
    "regen_end": 0.7,
    "protect_end": 0.7,
    "delivery": 0.1,
    "description": (
        "Oligodendrocyte precursor cells differentiate and re-sheath demyelinated "
        "axons; healthy tissue regrows into the lesion cavity."
    ),
    "color": "#3fb950",
}

_RECOVERY: dict[str, Any] = {
    "id": "functional_recovery",
    "name": "Functional Recovery & Synaptic Reconnection",
    "mechanism": "circuit_repair",
    "weight": 0.15,
    "lesion_scale_end": 0.0,
    "edema_end": 0.0,
    "regen_end": 0.92,
    "protect_end": 0.4,
    "delivery": 0.0,
    "description": (
        "Plasticity and synaptic reconnection restore functional circuits. The "
        "cavity is filled with regenerated tissue; near-normal architecture returns."
    ),
    "color": "#2bd4a0",
}


def _phase(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    d = dict(base)
    d.update(overrides)
    return d


# -- Per-disease-class phase sequences ------------------------------------
# Each disease gets a tailored cascade reflecting its real-world therapy.
DISEASE_CURE_PHASES: dict[int, list[dict[str, Any]]] = {
    # MS — DMT: targeting, anti-inflam, reversal, remyelination, recovery.
    3: [_TARGETING, _ANTI_INFLAM, _REVERSAL, _REGEN, _RECOVERY],
    # Glioma — resection: targeting, resection, neuroprotect, regeneration.
    4: [_TARGETING, _ANTI_INFLAM, _REVERSAL, _NEUROPROTECT, _REGEN],
    # Stroke — reperfusion: targeting(reperfusion), neuroprotection, reversal, recovery.
    5: [_TARGETING, _NEUROPROTECT, _REVERSAL, _REGEN, _RECOVERY],
    # Epilepsy — ablation: targeting, reversal(ablation), recovery.
    6: [_TARGETING, _REVERSAL, _NEUROPROTECT, _RECOVERY],
    # Huntington — neuroprotective (limited): targeting, neuroprotect, mild regen.
    7: [_TARGETING, _NEUROPROTECT, _REGEN],
    # ALS — neuroprotective (limited): targeting, neuroprotect.
    8: [_TARGETING, _NEUROPROTECT],
    # TBI — debridement: targeting, anti-inflam, reversal, neuroprotect, regen, recovery.
    9: [_TARGETING, _ANTI_INFLAM, _REVERSAL, _NEUROPROTECT, _REGEN, _RECOVERY],
    # Meningioma — Simpson I resection: targeting, resection, recovery.
    10: [_TARGETING, _REVERSAL, _RECOVERY],
    # Metastasis — SRS: targeting(radiation), reversal, recovery.
    11: [_TARGETING, _REVERSAL, _NEUROPROTECT, _RECOVERY],
    # MCA infarct — thrombectomy: reperfusion, neuroprotect, reversal, recovery.
    12: [_TARGETING, _NEUROPROTECT, _REVERSAL, _REGEN, _RECOVERY],
    # SDH — evacuation: targeting, reversal(drain), recovery.
    13: [_TARGETING, _ANTI_INFLAM, _REVERSAL, _RECOVERY],
    # NPH — shunt: targeting(shunt), reversal(ventricular), recovery.
    14: [_TARGETING, _REVERSAL, _RECOVERY],
    # Abscess — drainage + antibiotics: targeting, reversal(drain), recovery.
    20: [_TARGETING, _ANTI_INFLAM, _REVERSAL, _REGEN, _RECOVERY],
    # CJD — supportive only: targeting(experimental), mild neuroprotect.
    15: [_TARGETING, _NEUROPROTECT],
    # FTD — symptomatic + tau trial: targeting, neuroprotect, mild regen.
    16: [_TARGETING, _NEUROPROTECT, _REGEN],
    # LBD — cholinergic: targeting, neuroprotect, regen.
    17: [_TARGETING, _NEUROPROTECT, _REGEN],
    # VaD — vascular control: targeting, neuroprotect, regen.
    18: [_TARGETING, _ANTI_INFLAM, _NEUROPROTECT, _REGEN],
    # PSP — symptomatic: targeting, neuroprotect.
    19: [_TARGETING, _NEUROPROTECT],
    # Alzheimer — anti-amyloid: targeting, reversal(plaque), neuroprotect, regen.
    1: [_TARGETING, _REVERSAL, _NEUROPROTECT, _REGEN],
    # Parkinson — dopaminergic: targeting, neuroprotect, regen.
    2: [_TARGETING, _NEUROPROTECT, _REGEN],
    # Healthy — preventive stimulation.
    0: [_phase(
        _TARGETING,
        name="Preventive Neurostimulation",
        weight=1.0,
        description="Low-dose preventive cortical stimulation sustains gray-matter integrity.",
    )],
}


@dataclass
class CureTimeline:
    """A sampled, frame-by-frame description of the cure cascade."""

    frames: list[dict[str, Any]]
    phases: list[CurePhase]
    before_volume: float
    after_volume: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "phases": [
                {
                    "id": p.id,
                    "name": p.name,
                    "mechanism": p.mechanism,
                    "description": p.description,
                    "color": p.color,
                    "weight": p.weight,
                }
                for p in self.phases
            ],
            "before_volume": self.before_volume,
            "after_volume": self.after_volume,
            "n_frames": len(self.frames),
        }


def build_cure_timeline(
    disease_class: int,
    before_volume: float,
    after_volume: float,
    n_frames: int = 36,
) -> CureTimeline:
    """Build a sampled cure timeline for a given disease class.

    Parameters
    ----------
    disease_class
        The predicted disease class index (0..20).
    before_volume, after_volume
        Lesion volume before/after the cure (mm³), from the simulator.
    n_frames
        Total animation frames across all phases.
    """
    phase_dicts = DISEASE_CURE_PHASES.get(
        disease_class, [_TARGETING, _REVERSAL, _RECOVERY]
    )
    # Normalise phase weights to sum to 1.
    total_w = sum(p["weight"] for p in phase_dicts)
    phases: list[CurePhase] = []
    for pd in phase_dicts:
        pd_norm = dict(pd)
        pd_norm["weight"] = pd["weight"] / total_w
        phases.append(
            CurePhase(**{k: pd_norm[k] for k in CurePhase.__dataclass_fields__})
        )

    # Allocate frames to phases proportionally (at least 2 per phase).
    alloc = [max(2, int(round(n_frames * p.weight))) for p in phases]
    diff = n_frames - sum(alloc)
    if diff:
        alloc[-1] = max(2, alloc[-1] + diff)

    frames: list[dict[str, Any]] = []
    cur_lesion = 1.0
    cur_edema = 1.0
    cur_regen = 0.0
    cur_protect = 0.0
    global_frame = 0

    for pi, phase in enumerate(phases):
        nf = alloc[pi]
        for fi in range(nf):
            frac = fi / max(1, nf - 1) if nf > 1 else 1.0
            ls = _lerp(cur_lesion, phase.lesion_scale_end, frac)
            ed = _lerp(cur_edema, phase.edema_end, frac)
            rg = _lerp(cur_regen, phase.regen_end, frac)
            pr = _lerp(cur_protect, phase.protect_end, frac)
            lesion_vol = before_volume * ls
            if pi == len(phases) - 1 and fi == nf - 1:
                lesion_vol = after_volume
                ls = lesion_vol / before_volume if before_volume else 0.0
            progress = global_frame / max(1, n_frames - 1)
            frames.append({
                "frame": global_frame,
                "phase_index": pi,
                "phase_name": phase.name,
                "phase_color": phase.color,
                "mechanism": phase.mechanism,
                "description": phase.description,
                "lesion_scale": round(ls, 4),
                "edema": round(ed, 4),
                "regen": round(rg, 4),
                "protect": round(pr, 4),
                "delivery": round(phase.delivery, 4),
                "lesion_volume": round(lesion_vol, 2),
                "progress": round(progress, 4),
            })
            global_frame += 1
        cur_lesion = phase.lesion_scale_end
        cur_edema = phase.edema_end
        cur_regen = phase.regen_end
        cur_protect = phase.protect_end

    log.info(
        "Built cure timeline: %d phases, %d frames, %s -> %s mm³",
        len(phases), len(frames), before_volume, after_volume,
    )
    return CureTimeline(
        frames=frames,
        phases=phases,
        before_volume=before_volume,
        after_volume=after_volume,
    )
