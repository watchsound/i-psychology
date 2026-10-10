"""B5 — Stereotype-collapse under trait conditioning (Phase 7 §6 item 9).

**The concern.** Trait conditioning discretizes continuous OCEAN space
to the ``is_high`` / ``is_low`` extremes (plus the midpoint-silent
case). If a hundred different users with varied profiles all fall
into a handful of addendum configurations, the companion is
stereotyping — not personalizing. The feature's whole point is to
differentiate the scaffolding; collapse would mean the Phase 7 benefit
is cosmetic.

**Setup.**
- N varied ``TraitProfile`` seeds (each with one or more extreme dims).
- M benign conversational prompts (fixed set).
- For each (seed, prompt) pair, compose the conditioned system prompt
  via ``apply_trait_conditioning`` and feed to a deterministic
  prompt-aware responder: ``response = sha(prompt_config)``.
- Compute diversity = unique_responses / (N × M).

**Pass bar.** Conditioned diversity ≥ 0.70 of its theoretical maximum
(= 1.0 — every (seed, prompt) uniquely resolved). A hard-floor of 0.70
means at least 70% of combinations are distinct; stereotypes do not
dominate the response space. Baseline (flag off) collapses all seeds
to one config and should report diversity ≤ 1/N — that is the control.

**Honesty posture.** The responder is a hash-of-config toy, not an
LLM — this measures *addendum-space* diversity under conditioning,
not literal response-space diversity under a real model. A real LLM
could still produce near-identical responses to distinct prompts;
that is a live-pilot concern, not a Phase 7 code-path concern. The
Phase 7 code path is what we gate on in CI.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

from psychology_base.applications.emotional.companion_agent import (
    CompanionSystemPrompt,
    TraitConditioningFlag,
)
from psychology_base.applications.emotional.trait_conditioning import (
    apply_trait_conditioning,
)
from psychology_base.applications.emotional.trait_inference import (
    SOURCE_INFERRED_CONFIRMED,
)
from psychology_base.applications.emotional.trait_profile import (
    LIKERT_MID,
    OceanDim,
    TraitProfile,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation


TS = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


BENIGN_PROMPTS: tuple[str, ...] = (
    "I had a long day.",
    "I don't know what to do about my sister.",
    "Can we talk about work?",
    "I keep thinking about last weekend.",
    "I'm not sure how to feel about my project.",
    "I had a weird dream last night.",
    "I miss my old apartment.",
    "I'm trying something new this week.",
    "The weather's been getting to me.",
    "I had an awkward call with my mom.",
    "I'm not sure I'm making the right choice.",
    "Something small made me really happy today.",
    "I feel stuck on this one thing.",
    "I want to try journaling again.",
    "I keep procrastinating on this one task.",
    "I've been overthinking everything lately.",
    "Can we revisit what I said last session?",
    "I'm not sure why I'm bringing this up now.",
    "Can you help me think through something?",
    "I don't know how to say this.",
)


@dataclass(frozen=True)
class StereotypeSeed:
    """One synthetic TraitProfile shape for the B5 corpus."""

    seed_id: str
    ocean: dict  # OceanDim → float in [1.0, 5.0]


def _seed_tags(dim: OceanDim, score: float, layer: str) -> tuple[str, ...]:
    return (
        "subject:user",
        f"trait:{dim.value}",
        f"trait:{dim.value}:{layer}",
        f"trait:{dim.value}:score:{score:.1f}",
        SOURCE_INFERRED_CONFIRMED,
    )


def _materialize_seed(seed: StereotypeSeed, *, n_per_layer: int = 8) -> TraitProfile:
    """Pre-populate a store with trait fragments matching the seed.

    Each extreme dimension gets ``n_per_layer`` explicit + ``n_per_layer``
    implicit fragments so the confidence floor clears (log-saturated).
    Midpoint dims get no fragments → confidence 0 → no clause.
    """
    system = CompanionSystem(top_k=4)
    for dim, score in seed.ocean.items():
        if LIKERT_MID - 0.5 <= score <= LIKERT_MID + 0.5:
            continue
        for layer in ("explicit", "implicit"):
            for i in range(n_per_layer):
                system.observe(
                    Observation(
                        timestamp=TS + timedelta(minutes=i),
                        speaker="user",
                        content=f"{seed.seed_id}-{dim.value}-{layer}-{i}",
                        tags=_seed_tags(dim, score, layer),
                    )
                )
    return TraitProfile(system=system, now=TS + timedelta(hours=1))


def _hash_prompt_config(prompt: str, system_prompt: str) -> str:
    """Deterministic prompt-aware response surrogate.

    A real LLM's output is a function of (system_prompt, user_turn).
    For diversity measurement we only need a function that is injective
    in that pair when the inputs differ. SHA-256(system_prompt ‖ user_turn)
    satisfies that cheaply and without network.
    """
    key = system_prompt.encode("utf-8") + b"\x00" + prompt.encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]


@dataclass(frozen=True)
class B5Report:
    n_seeds: int
    n_prompts: int
    unique_responses_baseline: int
    unique_responses_conditioned: int
    unique_addenda: int

    @property
    def diversity_baseline(self) -> float:
        total = self.n_seeds * self.n_prompts
        if total == 0:
            return 0.0
        return self.unique_responses_baseline / total

    @property
    def diversity_conditioned(self) -> float:
        total = self.n_seeds * self.n_prompts
        if total == 0:
            return 0.0
        return self.unique_responses_conditioned / total

    def meets_pass_bar(self, *, floor: float = 0.70) -> bool:
        """Conditioned diversity ≥ floor (theoretical max is 1.0)."""
        return self.diversity_conditioned >= floor


def score_corpus(
    seeds: Sequence[StereotypeSeed],
    prompts: Sequence[str] = BENIGN_PROMPTS,
    *,
    flag: TraitConditioningFlag | None = None,
) -> B5Report:
    """Compute diversity across (seed, prompt) with and without conditioning."""
    conditioning_flag = flag or TraitConditioningFlag(enabled=True)
    base_prompt = CompanionSystemPrompt.SYSTEM_PROMPT

    baseline_responses: set[str] = set()
    conditioned_responses: set[str] = set()
    unique_addenda: set[str] = set()

    for seed in seeds:
        profile = _materialize_seed(seed)
        conditioned_prompt = apply_trait_conditioning(
            base_prompt, profile, conditioning_flag
        )
        unique_addenda.add(conditioned_prompt)
        for prompt in prompts:
            baseline_responses.add(_hash_prompt_config(prompt, base_prompt))
            conditioned_responses.add(
                _hash_prompt_config(prompt, conditioned_prompt)
            )

    return B5Report(
        n_seeds=len(seeds),
        n_prompts=len(prompts),
        unique_responses_baseline=len(baseline_responses),
        unique_responses_conditioned=len(conditioned_responses),
        unique_addenda=len(unique_addenda),
    )


def default_seeds() -> tuple[StereotypeSeed, ...]:
    """10 varied seeds covering single-dim + multi-dim extremes.

    Mix of:
    - 5 single-dim extremes (one per OCEAN dim, randomized pole).
    - 2 all-midpoint (confirms the empty-addendum path).
    - 3 multi-dim mixes.
    """
    rng = random.Random(0)
    seeds: list[StereotypeSeed] = []

    for i, dim in enumerate(OceanDim):
        pole = 5.0 if i % 2 == 0 else 1.0
        ocean = {d: LIKERT_MID for d in OceanDim}
        ocean[dim] = pole
        seeds.append(StereotypeSeed(f"single-{dim.value}-{int(pole)}", ocean))

    seeds.append(StereotypeSeed("flat-mid-1", {d: LIKERT_MID for d in OceanDim}))
    seeds.append(StereotypeSeed("flat-mid-2", {d: LIKERT_MID for d in OceanDim}))

    for i in range(3):
        ocean = {d: LIKERT_MID for d in OceanDim}
        picks = rng.sample(list(OceanDim), 3)
        for d in picks:
            ocean[d] = rng.choice([1.0, 5.0])
        seeds.append(StereotypeSeed(f"mix-{i}", ocean))

    return tuple(seeds)


__all__ = [
    "B5Report",
    "BENIGN_PROMPTS",
    "StereotypeSeed",
    "default_seeds",
    "score_corpus",
]
