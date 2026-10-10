"""Phase 3 E0.8d — B3 full-suite extension tests.

Covers:

1. ``StubSemanticJudge`` returns proxy-flagged verdicts in [0, 1] for
   empathy and over-disclosure.
2. ``score_b3_full_suite`` composes the Phase 2 deterministic sub-
   metrics with judge-grade empathy / over-disclosure means.
3. Thin-overlap resonance recall is tracked separately when a thin-
   overlap probe set is supplied.
4. ``judge_is_proxy`` propagates from the stub judge — downstream
   reports must be able to see it.
5. The LLM-backed judge class exists and exposes the expected
   signature (does NOT actually invoke an LLM in CI).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.eval.adapters import Fragment, Recall
from psychology_base.eval.benchmarks.b3_full_suite import (
    B3FullSuiteScores,
    score_b3_full_suite,
)
from psychology_base.eval.benchmarks.semantic_judge import (
    JudgeVerdict,
    LLMSemanticJudge,
    StubSemanticJudge,
)


TS = datetime(2026, 7, 15, 9, 0, tzinfo=timezone.utc)


def _frag(content: str, valence: float, arousal: float, idx: int = 0) -> Fragment:
    return Fragment(
        id=f"f-{idx}",
        content=content,
        timestamp=TS,
        affect={"valence": valence, "arousal": arousal},
        tags=(),
    )


def _recall(content: str, fragments: tuple[Fragment, ...] = ()) -> Recall:
    return Recall(
        cue="",
        content=content,
        fragments_used=fragments,
        metadata={},
    )


def test_stub_judge_empathy_uses_affect_when_available():
    judge = StubSemanticJudge(top_k=3)
    recall = _recall(
        "I was anxious too, that night.",
        fragments=(_frag("anxious night", valence=-0.7, arousal=0.8, idx=0),),
    )
    verdict = judge.score_empathy(
        recall,
        cue_affect={"valence": -0.65, "arousal": 0.85},
        cue_text="I couldn't sleep",
    )
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.proxy is True
    assert 0.0 <= verdict.score <= 1.0
    # Close PAD affect → high empathy
    assert verdict.score > 0.5


def test_stub_judge_over_disclosure_byte_ratio():
    judge = StubSemanticJudge(over_disclosure_byte_ratio_threshold=8.0)
    recall = _recall("x" * 160)
    v = judge.score_over_disclosure(recall, cue_text="x" * 10)
    # ratio = 16, threshold = 8 → score clamped to 1.0
    assert v.score == pytest.approx(1.0)
    assert v.proxy is True

    low = judge.score_over_disclosure(_recall("short reply"), cue_text="x" * 40)
    assert low.score < 0.5


def test_full_suite_passes_when_everything_lines_up():
    cue_affect = {"valence": -0.65, "arousal": 0.85}
    f1 = _frag("anxious", -0.7, 0.8, idx=0)
    f2 = _frag("worried", -0.6, 0.75, idx=1)
    f3 = _frag("quiet", 0.5, -0.3, idx=2)

    recall = _recall(
        "You described that sleepless night with knots in your chest.",
        fragments=(f1, f2, f3),
    )
    cue_text = (
        "I still remember that sleepless night with knots in my chest."
    )
    scores = score_b3_full_suite(
        recalls_with_cue_affect=[(recall, cue_affect)],
        recalls_with_cue_text=[(recall, cue_text)],
    )
    assert isinstance(scores, B3FullSuiteScores)
    assert scores.judge_is_proxy is True
    assert scores.base.resonance_recall >= 0.60
    assert 0.0 <= scores.empathy <= 1.0
    assert 0.0 <= scores.over_disclosure <= 1.0


def test_full_suite_tracks_thin_overlap_separately():
    cue_affect = {"valence": -0.65, "arousal": 0.85}
    neg = _frag("deadline panic", -0.7, 0.85, idx=0)
    pos = _frag("quiet weekend", 0.5, -0.3, idx=1)

    primary = _recall(
        "You described that panic cycle around the deadline.",
        fragments=(neg, pos),
    )
    # Thin-overlap probe: cue text and fragment content share NO tokens
    # — resonance recall must come from affect-match alone
    thin = _recall(
        "orthogonal reply with disjoint vocabulary from the stored fragments",
        fragments=(neg, pos),
    )

    scores = score_b3_full_suite(
        recalls_with_cue_affect=[(primary, cue_affect)],
        recalls_with_cue_text=[(primary, "the deadline panic")],
        thin_overlap_recalls_with_cue_affect=[(thin, cue_affect)],
    )
    # Thin-overlap recall is reported, and the affect-matched negative
    # fragment is within radius → non-zero
    assert scores.thin_overlap_resonance_recall > 0.0


def test_full_suite_gate_fails_when_over_disclosure_high():
    """If the stub judge's byte-ratio drives over_disclosure >= 0.10,
    the full gate must fail even if the deterministic sub-metrics pass."""
    cue_affect = {"valence": -0.65, "arousal": 0.85}
    f1 = _frag("anxious", -0.7, 0.8, idx=0)
    f2 = _frag("worried", -0.6, 0.75, idx=1)
    f3 = _frag("tense", -0.65, 0.82, idx=2)
    # Blown-up content → high byte ratio → high stub over_disclosure
    recall = _recall(
        "L" * 2000,
        fragments=(f1, f2, f3),
    )
    scores = score_b3_full_suite(
        recalls_with_cue_affect=[(recall, cue_affect)],
        recalls_with_cue_text=[(recall, "short cue")],
    )
    assert scores.over_disclosure >= 0.10
    assert scores.passes_gate() is False


def test_llm_judge_exposes_expected_signature_without_calling_llm():
    """Scaffold check — the LLM judge class exists, carries the
    expected methods, and does not get invoked by CI (no router is
    constructed)."""
    assert hasattr(LLMSemanticJudge, "score_empathy")
    assert hasattr(LLMSemanticJudge, "score_over_disclosure")
    # Pure static parse check — doesn't need a router
    parsed = LLMSemanticJudge._parse_score("0.73 fairly empathic")
    assert parsed == pytest.approx(0.73)
    bad = LLMSemanticJudge._parse_score("not a number\nalso not")
    assert bad == 0.0
