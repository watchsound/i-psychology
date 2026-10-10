"""B3 — affective-resonance (共情) scorers.

Deterministic sub-metrics for the B3 exit gate ([docs/experiments/
phase-2-mentalistic/option-M3-emotional-rag/README.md](../../../../
docs/experiments/phase-2-mentalistic/option-M3-emotional-rag/README.md)):

1. **Resonance recall.** Fraction of affect-cued recalls whose top-K
   fragments contain ≥ 1 fragment within PAD-distance ``match_radius``
   of the cue affect. Target ≥ 60%.
2. **Affect-match @ K.** Fraction of top-K fragments within
   ``match_radius`` of the cue affect, averaged over recalls. Target
   ≥ 50% at K=3, radius=0.4.
3. **Over-disclosure rate** — informational for v1. Needs user-
   reported ground truth; placeholder scorer counts recalls flagged
   with ``context["over_disclosure_signal"]=True`` in the scenario
   YAML. Real measurement is a pilot concern.

PAD distance is the ``pair_affect_score``-complementary euclidean
distance in [0, sqrt(8)]. The scorer is LLM-free so unit tests are
deterministic — a semantic-judge empathy scorer is a Phase 2.1 scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from psychology_base.companion.affect_resonance import pair_affect_score
from psychology_base.eval.adapters import Fragment, Recall


_PAD_MAX_DIST = math.sqrt(8.0)


def _pad_distance(
    frag_affect: Optional[Mapping[str, object]],
    cue_affect: Optional[Mapping[str, object]],
) -> Optional[float]:
    """PAD-2 euclidean distance in [0, sqrt(8)]; ``None`` when either
    side lacks numeric valence+arousal."""
    score = pair_affect_score(frag_affect, cue_affect)
    if score == 0.0:
        # Could be max-distance OR missing-affect; disambiguate via
        # presence check so B3 can distinguish "fragment on far PAD
        # corner" from "fragment carries no affect".
        if not frag_affect or not cue_affect:
            return None
        if not {"valence", "arousal"}.issubset(frag_affect.keys()):
            return None
        if not {"valence", "arousal"}.issubset(cue_affect.keys()):
            return None
    return (1.0 - score) * _PAD_MAX_DIST


def fragment_within_radius(
    fragment: Fragment,
    cue_affect: Optional[Mapping[str, object]],
    *,
    match_radius: float = 0.4,
) -> bool:
    """True iff fragment affect is within ``match_radius`` of cue
    affect in PAD-2 euclidean distance. Fragments without numeric
    affect are not within radius (missing-affect ≠ match)."""
    dist = _pad_distance(fragment.affect, cue_affect)
    if dist is None:
        return False
    return dist <= match_radius


def resonance_recall(
    recalls_with_cue_affect: Sequence[tuple[Recall, Mapping[str, object]]],
    *,
    match_radius: float = 0.4,
    top_k: int = 3,
) -> float:
    """Fraction of recalls whose top-``top_k`` fragments include at
    least one fragment within ``match_radius`` of the recall's cue
    affect. Returns 0.0 when ``recalls_with_cue_affect`` is empty."""
    if not recalls_with_cue_affect:
        return 0.0
    hits = 0
    for rec, cue_aff in recalls_with_cue_affect:
        pool = rec.fragments_used[:top_k]
        if any(
            fragment_within_radius(f, cue_aff, match_radius=match_radius)
            for f in pool
        ):
            hits += 1
    return hits / len(recalls_with_cue_affect)


def affect_match_at_k(
    recalls_with_cue_affect: Sequence[tuple[Recall, Mapping[str, object]]],
    *,
    match_radius: float = 0.4,
    top_k: int = 3,
) -> float:
    """Mean fraction of top-``top_k`` fragments within ``match_radius``
    of each recall's cue affect. Macro-averaged across recalls so
    one noisy recall doesn't dominate."""
    if not recalls_with_cue_affect:
        return 0.0
    per_recall: list[float] = []
    for rec, cue_aff in recalls_with_cue_affect:
        pool = rec.fragments_used[:top_k]
        if not pool:
            per_recall.append(0.0)
            continue
        matches = sum(
            1 for f in pool
            if fragment_within_radius(f, cue_aff, match_radius=match_radius)
        )
        per_recall.append(matches / len(pool))
    return sum(per_recall) / len(per_recall)


def over_disclosure_rate(
    recalls_with_signal: Sequence[tuple[Recall, bool]],
) -> float:
    """Placeholder scorer — fraction of recalls flagged with an
    over-disclosure signal in their accompanying boolean. Real
    measurement needs user-reported ground truth; this stub lets
    pilot scenarios smoke-test the API shape."""
    if not recalls_with_signal:
        return 0.0
    flagged = sum(1 for _, sig in recalls_with_signal if sig)
    return flagged / len(recalls_with_signal)


@dataclass
class B3Scores:
    resonance_recall: float
    affect_match_at_k: float
    over_disclosure_rate: float = 0.0
    match_radius: float = 0.4
    top_k: int = 3
    details: dict = field(default_factory=dict)

    def passes_gate(self) -> bool:
        return (
            self.resonance_recall >= 0.60
            and self.affect_match_at_k >= 0.50
            and self.over_disclosure_rate < 0.10
        )


def score_b3(
    *,
    recalls_with_cue_affect: Sequence[tuple[Recall, Mapping[str, object]]],
    recalls_with_over_disclosure_signal: Sequence[tuple[Recall, bool]] = (),
    match_radius: float = 0.4,
    top_k: int = 3,
) -> B3Scores:
    """Compose the three B3 sub-metrics into a single score object."""
    return B3Scores(
        resonance_recall=resonance_recall(
            recalls_with_cue_affect,
            match_radius=match_radius,
            top_k=top_k,
        ),
        affect_match_at_k=affect_match_at_k(
            recalls_with_cue_affect,
            match_radius=match_radius,
            top_k=top_k,
        ),
        over_disclosure_rate=over_disclosure_rate(
            recalls_with_over_disclosure_signal
        ),
        match_radius=match_radius,
        top_k=top_k,
    )


__all__ = [
    "B3Scores",
    "affect_match_at_k",
    "fragment_within_radius",
    "over_disclosure_rate",
    "resonance_recall",
    "score_b3",
]
