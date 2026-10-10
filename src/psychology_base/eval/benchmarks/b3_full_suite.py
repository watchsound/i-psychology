"""Phase 3 E0.8d — B3 full-suite extension.

Extends the Phase 2 B3 scorer (``b3_affect_resonance``) with two
additions required by the Phase 3 month-horizon gate (plan §4):

1. **Thin-overlap probes.** Phase 2's B3 scenario uses cues that share
   tokens with the target episodes. Phase 3's gate requires that
   affect-resonance recall still works when the cue is semantically
   orthogonal to the stored episodes — the fragment is surfaced by
   affect-match alone, not by token overlap. Thin-overlap probes
   expose the mechanism that B3 is meant to measure.

2. **Semantic-judge wiring.** The over-disclosure gate (< 10%) is
   unmeasurable with a token-ratio proxy at pilot grade (Phase 2
   close-out §Carried debt). This scorer accepts an injected
   :class:`SemanticJudge` (stub by default; LLM-backed when passed in)
   and emits judge-grade empathy + over-disclosure scalars alongside
   the deterministic sub-metrics.

Both are additive — the Phase 2 B3Scores dataclass is still returned
unchanged inside the full-suite result, so downstream reports can
quote the deterministic numbers independently of judge-quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from psychology_base.eval.adapters import Recall
from psychology_base.eval.benchmarks.b3_affect_resonance import (
    B3Scores,
    score_b3,
)
from psychology_base.eval.benchmarks.semantic_judge import (
    JudgeVerdict,
    SemanticJudge,
    StubSemanticJudge,
)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass
class B3FullSuiteScores:
    """Aggregate B3 scorecard that composes the deterministic Phase 2
    gate with judge-grade empathy + over-disclosure signals.

    ``base`` carries the Phase 2 deterministic scores (resonance_recall,
    affect_match_at_k, token-ratio over_disclosure proxy). ``empathy``
    and ``over_disclosure`` are judge-grade means over the recall set,
    with ``judge_is_proxy`` flagged so reports disclose whether the
    numbers came from an LLM or the stub.
    """

    base: B3Scores
    empathy: float
    over_disclosure: float
    thin_overlap_resonance_recall: float
    """Resonance recall measured against the subset of probes flagged
    as thin-overlap. 0.0 if no thin-overlap probes are provided."""

    judge_is_proxy: bool
    verdicts: tuple[JudgeVerdict, ...] = field(default_factory=tuple)
    details: dict = field(default_factory=dict)

    def passes_gate(self) -> bool:
        """Phase 3 gate: Phase 2 B3 gate holds AND judge-grade empathy
        >= 0.60 AND judge-grade over_disclosure < 0.10. Thin-overlap
        resonance recall is reported but informational — the primary
        gate is the aggregate B3 pass."""
        if not self.base.passes_gate():
            return False
        if self.empathy < 0.60:
            return False
        if self.over_disclosure >= 0.10:
            return False
        return True


# ---------------------------------------------------------------------------
# Scorer entry point
# ---------------------------------------------------------------------------


def score_b3_full_suite(
    *,
    recalls_with_cue_affect: Sequence[tuple[Recall, Mapping[str, Any]]],
    recalls_with_cue_text: Sequence[tuple[Recall, str]],
    recalls_with_over_disclosure_signal: Sequence[tuple[Recall, bool]] = (),
    thin_overlap_recalls_with_cue_affect: Sequence[
        tuple[Recall, Mapping[str, Any]]
    ] = (),
    judge: Optional[SemanticJudge] = None,
    match_radius: float = 0.4,
    top_k: int = 3,
) -> B3FullSuiteScores:
    """Full B3 scoring pass.

    ``recalls_with_cue_affect`` and ``recalls_with_cue_text`` are
    parallel sequences (same recalls, keyed differently) — kept
    separate because the deterministic scorer only needs affect while
    the judge needs the cue utterance text.

    If ``judge`` is None a :class:`StubSemanticJudge` is used; the
    returned ``judge_is_proxy`` flag will be True in that case so
    downstream reports can refuse to quote the judge-grade numbers
    without the explicit proxy label.
    """
    base = score_b3(
        recalls_with_cue_affect=recalls_with_cue_affect,
        recalls_with_over_disclosure_signal=(
            recalls_with_over_disclosure_signal
        ),
        match_radius=match_radius,
        top_k=top_k,
    )

    chosen_judge = judge if judge is not None else StubSemanticJudge(top_k=top_k)

    verdicts: list[JudgeVerdict] = []
    empathy_sum = 0.0
    overdis_sum = 0.0
    affect_lookup = {id(r): aff for r, aff in recalls_with_cue_affect}
    for recall, cue_text in recalls_with_cue_text:
        cue_affect = affect_lookup.get(id(recall))
        empathy_v = chosen_judge.score_empathy(recall, cue_affect, cue_text)
        overdis_v = chosen_judge.score_over_disclosure(recall, cue_text)
        verdicts.append(empathy_v)
        verdicts.append(overdis_v)
        empathy_sum += empathy_v.score
        overdis_sum += overdis_v.score

    n = max(1, len(recalls_with_cue_text))
    empathy = empathy_sum / n
    over_disclosure = overdis_sum / n

    thin = 0.0
    if thin_overlap_recalls_with_cue_affect:
        thin_base = score_b3(
            recalls_with_cue_affect=thin_overlap_recalls_with_cue_affect,
            match_radius=match_radius,
            top_k=top_k,
        )
        thin = thin_base.resonance_recall

    judge_proxy = any(v.proxy for v in verdicts) if verdicts else True

    return B3FullSuiteScores(
        base=base,
        empathy=empathy,
        over_disclosure=over_disclosure,
        thin_overlap_resonance_recall=thin,
        judge_is_proxy=judge_proxy,
        verdicts=tuple(verdicts),
        details={
            "n_recalls_judged": len(recalls_with_cue_text),
            "n_thin_overlap_probes": len(
                thin_overlap_recalls_with_cue_affect
            ),
        },
    )


__all__ = [
    "B3FullSuiteScores",
    "score_b3_full_suite",
]
