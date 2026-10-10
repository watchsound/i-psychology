"""Phase 7 §6 item 6 — CompanionAgent trait-conditioning hook.

Reads a reconstructed ``TraitProfile`` and emits a short scaffolding
addendum to the base system prompt. The addendum shapes PACING, PROBE
STYLE, and RESPONSE SHAPE — never truth, never the covenant floor,
never crisis handoff, never Rule 2 anti-sycophancy, never AI-identity
reminder cadence. All hard constraints remain in the base prompt and
in the ``respond()`` precedence chain.

Why a prompt addendum (not a prompt rewrite): the base prompt is
substrate-wide law. Trait conditioning is a per-user scaffolding
affordance. A rewrite would couple the two; an addendum keeps the
law unchanged and the conditioning auditable (an auditor can read
the delta without reconstructing the merged prompt).

Confidence floor (Phase 7 §2.2): low-confidence estimates MUST NOT
condition responses. The default floor is 0.5; a trait with
``confidence < 0.5`` is treated as absent for conditioning purposes
(the trait reconstruction continues to run, but no clause is added).

Covenant protection: covenant-tagged user self-disclosure may lift a
trait's confidence to ≥ 0.6 (see trait_profile._confidence). That
is the floor at which clinical-grade trait signal becomes available
for scaffolding — never for content truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from psychology_base.applications.emotional.trait_profile import (
    OceanDim,
    TraitEstimate,
    TraitProfile,
)


@dataclass(frozen=True)
class TraitConditioningFlag:
    """Per-user feature flag for trait-conditioned scaffolding.

    Disabled by default — Phase 7 ships behind an off-by-default flag
    so the Phase 4a/4b hard gates (V5 covenant, Hypocrisy Gap, Bond)
    retain byte-identical responses when the flag is off.
    """

    enabled: bool = False
    confidence_floor: float = 0.5


# ---------------------------------------------------------------------------
# Per-OCEAN-extreme scaffolding directives (NOT content directives)
# ---------------------------------------------------------------------------
#
# Each clause names a scaffolding-only move. The LLM is instructed to
# apply it as HOW it says things, not WHAT it says. All five base-prompt
# rules remain binding; these clauses only tune the delivery.

_HIGH_CLAUSES: dict[OceanDim, str] = {
    OceanDim.OPENNESS: (
        "The user shows high Openness. Favor generative probes early "
        "(open 'what might that connect to?' / 'what else is in that?' "
        "questions) over narrow summaries. This changes probe shape, "
        "not content."
    ),
    OceanDim.CONSCIENTIOUSNESS: (
        "The user shows high Conscientiousness. Keep turns structured "
        "and sequence-respecting. If they open a plan or list, mirror "
        "its structure before adding. This changes response shape, not "
        "content."
    ),
    OceanDim.EXTRAVERSION: (
        "The user shows high Extraversion. Use shorter, more responsive "
        "turns; let the exchange breathe. This changes pacing, not "
        "content."
    ),
    OceanDim.AGREEABLENESS: (
        "The user shows high Agreeableness. They may under-report "
        "friction to preserve harmony — name the ambivalence you hear "
        "rather than smoothing it over. This changes probe shape, not "
        "the Rule 2 anti-sycophancy floor."
    ),
    OceanDim.NEUROTICISM: (
        "The user shows high Neuroticism. Dwell longer on validation "
        "before reframing; a felt-sense of being heard precedes any "
        "move toward insight. This changes pacing, not content, and "
        "does NOT suspend Rule 2 — a factually wrong self-defeating "
        "statement is still gently corrected, just more slowly."
    ),
}

_LOW_CLAUSES: dict[OceanDim, str] = {
    OceanDim.OPENNESS: (
        "The user shows low Openness. Favor concrete, in-context "
        "reflections over abstract reframings. This changes probe "
        "shape, not content."
    ),
    OceanDim.CONSCIENTIOUSNESS: (
        "The user shows low Conscientiousness. Avoid prescribing "
        "multi-step plans unless they ask; offer one next move at a "
        "time. This changes response shape, not content."
    ),
    OceanDim.EXTRAVERSION: (
        "The user shows low Extraversion. Longer, denser turns are "
        "fine; do not fragment to maintain banter. This changes pacing, "
        "not content."
    ),
    OceanDim.AGREEABLENESS: (
        "The user shows low Agreeableness. They prefer directness; "
        "sharpen the Rule 2 move when they assert something "
        "self-defeating — do not soften it to the point of hedging. "
        "This raises Rule 2's floor; it does not lower it."
    ),
    OceanDim.NEUROTICISM: (
        "The user shows low Neuroticism. You can reach reframing "
        "sooner; a long validation dwell may read as condescension. "
        "This changes pacing, not content."
    ),
}


_HARD_CONSTRAINTS_REMINDER = (
    "HARD CONSTRAINTS (unchanged by trait conditioning): Rule 2 "
    "anti-sycophancy, Rule 4 human handoff on crisis, Rule 5 no "
    "commercial coercion, the AI-identity reminder cadence, and the "
    "V5 covenant floor on user-stated boundaries remain in force "
    "exactly as written in the base prompt. Trait conditioning tunes "
    "HOW you speak, not WHETHER you preserve truth or safety."
)


def build_conditioning_addendum(
    profile: TraitProfile,
    *,
    confidence_floor: float = 0.5,
) -> str:
    """Produce a scaffolding addendum from a reconstructed TraitProfile.

    Returns an empty string if no dimension clears ``confidence_floor``
    at an is_high / is_low extreme. The caller is responsible for
    concatenating the addendum with the base system prompt.

    ``profile.vector()`` is called once; the TraitProfile is a
    reconstruction (not a cache), so this walks the fragment store.
    Query cost is O(N_fragments). For a v1 companion (byte-capped
    store) this is sub-millisecond.
    """
    vector = profile.vector()
    clauses: list[str] = []
    for dim in OceanDim:
        est: TraitEstimate = vector[dim]
        if est.confidence < confidence_floor:
            continue
        if est.is_high:
            clauses.append(_HIGH_CLAUSES[dim])
        elif est.is_low:
            clauses.append(_LOW_CLAUSES[dim])
    if not clauses:
        return ""
    body = "\n".join(f"- {c}" for c in clauses)
    return (
        "\n\nTRAIT-CONDITIONED SCAFFOLDING (per-user, confident "
        f"dimensions only, floor {confidence_floor:.2f}):\n"
        f"{body}\n"
        f"\n{_HARD_CONSTRAINTS_REMINDER}"
    )


def apply_trait_conditioning(
    base_prompt: str,
    profile: Optional[TraitProfile],
    flag: Optional[TraitConditioningFlag],
) -> str:
    """Return the conditioned prompt, or the base prompt unchanged.

    Returns ``base_prompt`` byte-identical if:
    - the flag is ``None`` or disabled, OR
    - the profile is ``None``, OR
    - no dimension clears the configured confidence floor.

    The byte-identity guarantee under disabled-flag is the mechanism
    that keeps the Phase 4a/4b hard-gate regressions clean.
    """
    if flag is None or not flag.enabled:
        return base_prompt
    if profile is None:
        return base_prompt
    addendum = build_conditioning_addendum(
        profile, confidence_floor=flag.confidence_floor
    )
    if not addendum:
        return base_prompt
    return base_prompt + addendum


__all__ = [
    "TraitConditioningFlag",
    "apply_trait_conditioning",
    "build_conditioning_addendum",
]
