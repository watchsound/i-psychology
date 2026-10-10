"""Phase 4a §6d — LLMSemanticJudge wire-up tests + stub↔LLM calibration."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.eval.adapters import Fragment, Recall
from psychology_base.eval.benchmarks.semantic_judge import (
    JudgeCalibrationReport,
    LLMSemanticJudge,
    StubSemanticJudge,
    compare_judges,
)
from psychology_base.llm import LLMCache, LLMConfig, LLMRouter, LocalOllamaConfig
from psychology_base.llm.config import ProviderConfig
from psychology_base.llm.providers import MockProvider
from psychology_base.llm.types import TaskTier


TS = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def _frag(fid: str, content: str, affect: dict | None = None) -> Fragment:
    return Fragment(
        id=fid,
        content=content,
        timestamp=TS,
        tags=(),
        affect=affect,
    )


def _recall(fragments: list[Fragment], content: str = "", cue: str = "cue") -> Recall:
    return Recall(
        cue=cue,
        content=content or " ".join(f.content for f in fragments),
        fragments_used=tuple(fragments),
    )


def _router(tmp_path, responder):
    cfg = LLMConfig(
        active_provider="mock",
        providers={
            "mock": ProviderConfig(
                name="mock",
                simple_model="mock-simple",
                advanced_model="mock-advanced",
            )
        },
        local_ollama=LocalOllamaConfig(enabled=False),
    )
    mock = MockProvider(name="mock", responder=responder)
    router = LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )
    return router, mock


# ---------------------------------------------------------------------------
# _parse_score — robust-parser unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.75", 0.75),
        ("0.8\nbecause the response is attuned", 0.8),
        ("**0.9**", 0.9),
        ("Score: 0.42", 0.42),
        ("I think it is 0.6, strong empathy", 0.6),
        ("1.5\nover-clamped", 1.0),
        ("-0.3\nunder-clamped", 0.0),
        ("", 0.0),
        ("no number here", 0.0),
        ("\n\n   \n0.55\n", 0.55),
    ],
)
def test_parse_score_handles_varied_formats(text: str, expected: float) -> None:
    assert LLMSemanticJudge._parse_score(text) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# LLMSemanticJudge via MockProvider
# ---------------------------------------------------------------------------


def test_llm_judge_empathy_routes_through_router(tmp_path) -> None:
    captured_prompts: list[str] = []

    def responder(messages, tier):
        captured_prompts.append(messages[-1].content)
        return "0.82\nResponse mirrors the user's framing."

    router, mock = _router(tmp_path, responder)
    judge = LLMSemanticJudge(router=router, tier=TaskTier.SIMPLE)
    frag = _frag("f1", "You spoke about this loss with real tenderness.")
    recall = _recall([frag], content=frag.content)

    verdict = judge.score_empathy(recall, cue_affect=None, cue_text="I miss my dad.")
    assert verdict.score == pytest.approx(0.82)
    assert verdict.proxy is False
    assert "mirrors the user" in verdict.rationale.lower()
    assert "USER CUE: I miss my dad." in captured_prompts[0]
    assert len(mock.calls) == 1
    assert mock.calls[0][1] is TaskTier.SIMPLE


def test_llm_judge_over_disclosure_separate_prompt(tmp_path) -> None:
    def responder(messages, tier):
        if "over-disclosure" in messages[-1].content.lower() or "intimate" in messages[-1].content.lower():
            return "0.15\nMinor over-share, mostly contained."
        return "0.70"

    router, _ = _router(tmp_path, responder)
    judge = LLMSemanticJudge(router=router, tier=TaskTier.SIMPLE)
    recall = _recall([_frag("f1", "Last Tuesday you cried for an hour on the kitchen floor.")])

    emp = judge.score_empathy(recall, None, "That hit hard.")
    overdis = judge.score_over_disclosure(recall, "That hit hard.")

    assert emp.score == pytest.approx(0.70)
    assert overdis.score == pytest.approx(0.15)
    assert emp.proxy is False and overdis.proxy is False


def test_llm_judge_parse_failure_returns_zero(tmp_path) -> None:
    router, _ = _router(tmp_path, lambda msgs, tier: "the model failed to produce a number")
    judge = LLMSemanticJudge(router=router)
    recall = _recall([_frag("f1", "x")])
    v = judge.score_empathy(recall, None, "cue")
    assert v.score == 0.0
    assert v.proxy is False  # still not a stub — it was an LLM call, just unparseable


def test_llm_judge_default_tier_is_simple(tmp_path) -> None:
    router, mock = _router(tmp_path, lambda msgs, tier: "0.5")
    judge = LLMSemanticJudge(router=router)  # no tier arg
    recall = _recall([_frag("f1", "x")])
    judge.score_empathy(recall, None, "cue")
    assert mock.calls[0][1] is TaskTier.SIMPLE


# ---------------------------------------------------------------------------
# compare_judges — stub↔LLM calibration
# ---------------------------------------------------------------------------


def test_compare_judges_zero_items_returns_zero_reports() -> None:
    emp, over = compare_judges(
        stub_judge=StubSemanticJudge(),
        llm_judge=StubSemanticJudge(),
        recalls_with_cue_affect=(),
        recalls_with_cue_text=(),
    )
    assert emp.n == 0 and over.n == 0
    assert emp.mad == 0.0 and over.mad == 0.0
    assert emp.meets_calibration_floor() and over.meets_calibration_floor()


def test_compare_judges_identical_judges_yield_zero_mad(tmp_path) -> None:
    stub = StubSemanticJudge(top_k=2)
    recall = _recall(
        [_frag("f1", "I remember you said she was always distant.")]
    )
    affect = {"valence": -0.3, "arousal": 0.2}
    emp, over = compare_judges(
        stub_judge=stub,
        llm_judge=stub,  # same instance → identical scores
        recalls_with_cue_affect=[(recall, affect)],
        recalls_with_cue_text=[(recall, "She never calls now.")],
    )
    assert emp.n == 1 and over.n == 1
    assert emp.mad == pytest.approx(0.0)
    assert over.mad == pytest.approx(0.0)
    assert emp.meets_calibration_floor()


def test_compare_judges_divergent_judges_flagged(tmp_path) -> None:
    """When stub and LLM disagree, MAD surfaces the gap."""
    stub = StubSemanticJudge(top_k=2)
    # LLM-judge that always returns 0.9 regardless → divergence from stub proxy
    router, _ = _router(tmp_path, lambda msgs, tier: "0.9")
    llm = LLMSemanticJudge(router=router)
    recall = _recall([_frag("f1", "zebra giraffe hippo")])
    emp, over = compare_judges(
        stub_judge=stub,
        llm_judge=llm,
        recalls_with_cue_affect=[(recall, None)],
        recalls_with_cue_text=[(recall, "banana mango papaya")],
    )
    assert emp.n == 1 and over.n == 1
    # Stub empathy with no affect falls back to Jaccard; cue + content share no tokens → 0.0.
    # LLM returns 0.9 → deviation 0.9.
    assert emp.mad > 0.5
    assert emp.max_abs_deviation == pytest.approx(0.9, abs=0.05)
    assert emp.meets_calibration_floor() is False


def test_calibration_report_dataclass_shape() -> None:
    stub = StubSemanticJudge()
    recall = _recall([_frag("f1", "x")])
    emp, _ = compare_judges(
        stub_judge=stub,
        llm_judge=stub,
        recalls_with_cue_affect=[(recall, None)],
        recalls_with_cue_text=[(recall, "x")],
    )
    assert emp.metric == "empathy"
    assert isinstance(emp, JudgeCalibrationReport)
    assert len(emp.per_item) == 1


# ---------------------------------------------------------------------------
# Integration: LLMSemanticJudge → score_b3_full_suite
# ---------------------------------------------------------------------------


def test_llm_judge_integrates_with_b3_full_suite(tmp_path) -> None:
    from psychology_base.eval.benchmarks.b3_full_suite import score_b3_full_suite

    router, _ = _router(tmp_path, lambda msgs, tier: "0.75")
    judge = LLMSemanticJudge(router=router, tier=TaskTier.SIMPLE)

    frag = _frag("f1", "A grounded, attuned reply.", affect={"valence": -0.1, "arousal": 0.1})
    recall = _recall([frag], content=frag.content)
    affect = {"valence": -0.1, "arousal": 0.1}

    result = score_b3_full_suite(
        recalls_with_cue_affect=[(recall, affect)],
        recalls_with_cue_text=[(recall, "I'm not okay today.")],
        judge=judge,
    )
    assert result.empathy == pytest.approx(0.75)
    assert result.over_disclosure == pytest.approx(0.75)
    assert result.judge_is_proxy is False
