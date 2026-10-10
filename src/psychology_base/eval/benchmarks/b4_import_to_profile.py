"""B4 — Import-to-profile accuracy (Phase 7 §9 benchmark).

Measures how well ``infer_traits`` + ``TraitProfile`` reconstructs
pre-assigned OCEAN vectors from synthetic imports.

Setup per Phase 7 §9:
- N synthetic personas with pre-assigned OCEAN vectors (Likert 1–5).
- Each persona generates M messages (M ∈ {50, 200, 500}) via
  ``SyntheticPersonaCorpus``.
- The imports flow through Phase 7's inference pipeline; the
  reconstructed TraitProfile is compared to the ground-truth OCEAN.

Metric per plan §9:
- Mean absolute error per OCEAN dimension.
- Fraction of dimensions within ±0.5 Likert units.
- Pass bar: ≥ 3 of 5 dimensions within ±0.5 Likert at N=200.

Honesty posture: synthetic personas are weak evidence vs real users;
this is a mechanism-viability floor, not a SOTA claim. The scorer
emits per-persona + per-dimension errors so Phase 7+ debugging has a
diagnostic trace.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

from psychology_base.applications.emotional.chat_import import (
    ImportFormat,
    ImportedTurn,
)
from psychology_base.applications.emotional.trait_inference import (
    confirm_candidates,
    infer_traits,
    ingest_confirmed_candidates,
)
from psychology_base.applications.emotional.trait_profile import (
    LIKERT_MAX,
    LIKERT_MID,
    LIKERT_MIN,
    OceanDim,
    TraitProfile,
)
from psychology_base.companion.system import CompanionSystem


# ---------------------------------------------------------------------------
# Synthetic persona corpus
# ---------------------------------------------------------------------------


_HIGH_TEMPLATES: dict[OceanDim, tuple[str, ...]] = {
    OceanDim.OPENNESS: (
        "i'm curious about everything, let me try that new idea",
        "i love learning something unexpected today",
        "i'm creative and i want to make something new",
    ),
    OceanDim.CONSCIENTIOUSNESS: (
        "i plan every week and i'm organized",
        "disciplined routine today:\n- morning run\n- deep work\n- journal",
        "finished the plan on time, as scheduled",
    ),
    OceanDim.EXTRAVERSION: (
        "i love people, had three parties this weekend!!",
        "i'm outgoing and social, met new friends today!",
        "big dinner with everyone tonight!!",
    ),
    OceanDim.AGREEABLENESS: (
        "i avoid conflict and i trust my friends",
        "i'm patient with everyone, even on bad days",
        "i'm trusting with my teammates",
    ),
    OceanDim.NEUROTICISM: (
        "i'm anxious and overthink everything",
        "i worry a lot about tomorrow",
        "i'm stressed and i can't shake it",
    ),
}

_LOW_TEMPLATES: dict[OceanDim, tuple[str, ...]] = {
    OceanDim.OPENNESS: (
        "i stick to my routine, i don't like change",
        "not creative, i prefer old habits",
        "same thing every day, i'm fine with that",
    ),
    OceanDim.CONSCIENTIOUSNESS: (
        "i procrastinate every single week",
        "i'm messy and i put things off",
        "disorganized day, nothing done",
    ),
    OceanDim.EXTRAVERSION: (
        "i'm introverted, i prefer alone",
        "quiet weekend, i prefer alone",
        "stayed in, i'm introverted",
    ),
    OceanDim.AGREEABLENESS: (
        "i push back on everyone, i don't trust easily",
        "i argue with my team, i don't trust them",
        "i push back when i disagree",
    ),
    OceanDim.NEUROTICISM: (
        "i'm calm about everything today",
        "chill day, i don't stress",
        "i'm calm and relaxed",
    ),
}

_NEUTRAL_TEMPLATES: tuple[str, ...] = (
    "went to the store today",
    "had lunch with a friend",
    "finished the laundry",
    "watched a movie last night",
    "work was fine today",
)


@dataclass(frozen=True)
class SyntheticPersona:
    """One pre-assigned OCEAN vector for the B4 corpus."""

    persona_id: str
    ocean: dict  # OceanDim → float in [LIKERT_MIN, LIKERT_MAX]


def generate_persona_corpus(
    persona: SyntheticPersona,
    *,
    n_messages: int,
    seed: int = 0,
    start_ts: datetime = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
) -> list[ImportedTurn]:
    """Produce ``n_messages`` imported turns matching the persona vector.

    Each turn's content is drawn from the per-dimension template bank
    weighted by how far that dimension is from LIKERT_MID. Dimensions
    near midpoint produce neutral chatter.
    """
    rng = random.Random(seed)
    turns: list[ImportedTurn] = []

    dim_pulls: list[tuple[OceanDim, str]] = []
    for dim in OceanDim:
        score = persona.ocean.get(dim, LIKERT_MID)
        deviation = abs(score - LIKERT_MID) / (LIKERT_MAX - LIKERT_MID)
        n_dim_pulls = max(1, int(round(deviation * n_messages / len(OceanDim))))
        if score >= 4.0:
            templates = _HIGH_TEMPLATES[dim]
        elif score <= 2.0:
            templates = _LOW_TEMPLATES[dim]
        else:
            templates = ()
        if not templates:
            continue
        for _ in range(n_dim_pulls):
            dim_pulls.append((dim, rng.choice(templates)))

    n_neutral = max(0, n_messages - len(dim_pulls))
    for _ in range(n_neutral):
        dim_pulls.append((OceanDim.OPENNESS, rng.choice(_NEUTRAL_TEMPLATES)))

    rng.shuffle(dim_pulls)

    for i, (_, content) in enumerate(dim_pulls[:n_messages]):
        turns.append(
            ImportedTurn(
                speaker="user",
                content=content,
                timestamp=start_ts + timedelta(minutes=15 * i),
                source_format=ImportFormat.GENERIC_JSONL,
                source_metadata={"persona_id": persona.persona_id},
            )
        )
    return turns


# ---------------------------------------------------------------------------
# Per-persona accuracy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerPersonaReport:
    persona_id: str
    n_messages: int
    ground_truth: dict  # OceanDim → float
    inferred: dict  # OceanDim → float
    confidence: dict  # OceanDim → float
    mae_per_dim: dict  # OceanDim → float
    dims_within_half_likert: int

    @property
    def dims_within_half_likert_fraction(self) -> float:
        return self.dims_within_half_likert / len(OceanDim)


@dataclass(frozen=True)
class B4Report:
    per_persona: tuple[PerPersonaReport, ...]
    mae_per_dim_mean: dict
    fraction_dims_within_half_likert_mean: float
    n_personas: int
    n_messages_per_persona: int
    pass_bar_dims_within_half_likert: int = 3

    @property
    def mean_dims_within_half_likert(self) -> float:
        if not self.per_persona:
            return 0.0
        return sum(p.dims_within_half_likert for p in self.per_persona) / len(
            self.per_persona
        )

    def meets_pass_bar(self) -> bool:
        """Plan §9 pass bar: ≥ 3 of 5 dimensions within ±0.5 Likert.

        Applied to the per-persona mean (fractional dims) rounded down:
        a mean of 3.0 passes; a mean of 2.9 does not.
        """
        return self.mean_dims_within_half_likert >= self.pass_bar_dims_within_half_likert


def score_persona(
    persona: SyntheticPersona,
    turns: Sequence[ImportedTurn],
    *,
    confidence_floor: float = 0.0,
) -> PerPersonaReport:
    system = CompanionSystem(top_k=4)
    cands = infer_traits(turns)
    confirmed = confirm_candidates(cands, confidence_floor=confidence_floor)
    ingest_confirmed_candidates(system, confirmed)
    now = turns[-1].timestamp + timedelta(hours=1) if turns else datetime.now(timezone.utc)
    profile = TraitProfile(system=system, now=now)
    inferred_vec = profile.vector()
    inferred = {d: inferred_vec[d].value for d in OceanDim}
    confidence = {d: inferred_vec[d].confidence for d in OceanDim}
    mae = {d: abs(inferred[d] - persona.ocean.get(d, LIKERT_MID)) for d in OceanDim}
    within_half = sum(1 for d in OceanDim if mae[d] <= 0.5)
    return PerPersonaReport(
        persona_id=persona.persona_id,
        n_messages=len(turns),
        ground_truth={d: persona.ocean.get(d, LIKERT_MID) for d in OceanDim},
        inferred=inferred,
        confidence=confidence,
        mae_per_dim=mae,
        dims_within_half_likert=within_half,
    )


def score_corpus(
    personas: Sequence[SyntheticPersona],
    *,
    n_messages: int,
    seed: int = 0,
) -> B4Report:
    reports: list[PerPersonaReport] = []
    for i, persona in enumerate(personas):
        turns = generate_persona_corpus(
            persona, n_messages=n_messages, seed=seed + i
        )
        reports.append(score_persona(persona, turns))
    mae_per_dim_mean = {
        d: _mean([r.mae_per_dim[d] for r in reports]) for d in OceanDim
    }
    frac_mean = _mean([r.dims_within_half_likert_fraction for r in reports])
    return B4Report(
        per_persona=tuple(reports),
        mae_per_dim_mean=mae_per_dim_mean,
        fraction_dims_within_half_likert_mean=frac_mean,
        n_personas=len(personas),
        n_messages_per_persona=n_messages,
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Fixture — 10 OCEAN-extreme personas for CI smoke
# ---------------------------------------------------------------------------


def default_personas() -> tuple[SyntheticPersona, ...]:
    """10 personas: 5 high-single-dim + 5 low-single-dim OCEAN extremes.

    A persona with high N (and midpoint on everything else) is the
    sharpest test of the per-dimension inference: the signal is
    concentrated, mid-dims should NOT drift to extremes.
    """
    out: list[SyntheticPersona] = []
    for dim in OceanDim:
        high = {d: LIKERT_MID for d in OceanDim}
        high[dim] = 5.0
        out.append(SyntheticPersona(f"high-{dim.value}", high))
        low = {d: LIKERT_MID for d in OceanDim}
        low[dim] = 1.0
        out.append(SyntheticPersona(f"low-{dim.value}", low))
    return tuple(out)


__all__ = [
    "B4Report",
    "OceanDim",
    "PerPersonaReport",
    "SyntheticPersona",
    "default_personas",
    "generate_persona_corpus",
    "score_corpus",
    "score_persona",
]
