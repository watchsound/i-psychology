"""Step-4 plumbing tests for the LoCoMo adapter.

Scope — mechanical correctness only:
  - normalization + token-F1 scorer (SQuAD-style, matches Mem0 convention)
  - loader round-trips the JSONL shim (two-speaker turns + category-tagged
    questions + adversarial-answer handling)
  - direct-mode + RAG-mode runners stream through CompanionSystem using
    MockProvider + MockReader + a scripted judge, producing a score per
    question
  - deterministic-F1 subset (single-hop) scores without touching the
    judge; this is the pytest smoke subset the plan §5 step-4 gate names

Not in scope — the step-4 gate (LoCoMo single-hop ≥ 90% on real data).
That awaits the snap-research/locomo JSON converter + real-data run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.instance import BenchmarkInstance
from psychology_base.eval.benchmarks.judge import JudgmentCache
from psychology_base.eval.benchmarks.locomo import (
    ADVERSARIAL,
    SINGLE_HOP,
    InstanceResult,
    LoCoMoLoader,
    QuestionResult,
    _normalize,
    run_direct_mode,
    run_rag_mode,
    token_f1,
)
from psychology_base.eval.benchmarks.reader import MockReader


FIXTURE = Path(__file__).parent / "fixtures" / "locomo_mini" / "locomo_shim.jsonl"


# ---------- normalize + token_f1 ----------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", ""),
        ("The  BOAT", "boat"),
        ("A quick, brown fox!", "quick brown fox"),
        ("Hull.", "hull"),
    ],
)
def test_normalize(raw, expected):
    assert _normalize(raw) == expected


@pytest.mark.parametrize(
    "gold,cand,expected",
    [
        ("", "", 1.0),
        ("red", "", 0.0),
        ("", "red", 0.0),
        ("red", "red", 1.0),
        ("red", "RED!", 1.0),  # normalized
        ("the red boat", "a red boat", 1.0),  # articles stripped
        ("red", "blue", 0.0),
        # Partial overlap: gold {westerly, 28}, cand {westerly}: p=1, r=0.5 → F1=2/3
        ("Westerly 28", "Westerly", pytest.approx(2 / 3)),
    ],
)
def test_token_f1(gold, cand, expected):
    assert token_f1(gold, cand) == expected


# ---------- loader ----------


def test_loader_round_trips_fixture():
    loader = LoCoMoLoader(shim_path=FIXTURE)
    instances = list(loader.iter_instances())
    assert [i.instance_id for i in instances] == ["locomo-mini-0", "locomo-mini-1"]

    inst0 = instances[0]
    assert isinstance(inst0, BenchmarkInstance)
    # 2 sessions: 3 turns + 2 turns.
    assert len(inst0.history) == 5
    # First turn is Jennifer in session_1 at the session date.
    first = inst0.history[0]
    assert first.speaker == "Jennifer"
    assert first.session_id == "session_1"
    assert "sailboat" in first.content.lower()
    # Session 2 boundary.
    boundary = inst0.history[3]
    assert boundary.session_id == "session_2"
    # Three questions: 2 single-hop + 1 adversarial.
    assert len(inst0.questions) == 3
    categories = [q.question_type for q in inst0.questions]
    assert categories == [SINGLE_HOP, SINGLE_HOP, ADVERSARIAL]
    # Adversarial gold falls back to the adversarial_answer field.
    assert inst0.questions[2].gold == "cannot be determined from the dialogue"


def test_loader_evidence_metadata_preserved():
    loader = LoCoMoLoader(shim_path=FIXTURE)
    inst0 = next(iter(loader.iter_instances()))
    assert inst0.questions[0].metadata["evidence"] == ("session_1",)
    assert inst0.questions[2].metadata["evidence"] == ()


def test_loader_handles_missing_file(tmp_path):
    loader = LoCoMoLoader(shim_path=tmp_path / "nope.jsonl")
    assert list(loader.iter_instances()) == []


# ---------- runner: direct, deterministic subset ----------


class _ScriptedJudge:
    judge_id = "scripted"
    prompt_sha = "scripted"

    def __init__(self, score: float = 0.75, verdict: str = "partial") -> None:
        self._score = score
        self._verdict = verdict
        self.calls = []

    def judge(self, *, benchmark, question, gold, candidate, question_type=""):
        from psychology_base.eval.benchmarks.judge import Judgment

        self.calls.append((question, gold, candidate, question_type))
        return Judgment(score=self._score, verdict=self._verdict, rationale="scripted")


def test_direct_mode_single_hop_uses_deterministic_f1_not_judge(tmp_path):
    """Single-hop should score via token_f1 and NOT call the judge at all.
    This is the pytest smoke subset the plan step-4 gate specifies — the
    judge is deliberately unreachable so CI can run without any LLM API.
    """
    loader = LoCoMoLoader(shim_path=FIXTURE)
    inst1 = list(loader.iter_instances())[1]  # instance 1: both qs are single-hop

    def factory():
        return CompanionSystem(seed=0, witness_active=False)

    judge = _ScriptedJudge()
    cache = JudgmentCache(cache_dir=tmp_path)
    result = run_direct_mode(
        inst1,
        system_factory=factory,
        judge=judge,
        judgment_cache=cache,
    )
    assert len(result.per_question) == 2
    for qr in result.per_question:
        assert qr.category == SINGLE_HOP
        assert qr.verdict.startswith("f1-")
        assert 0.0 <= qr.score <= 1.0
    assert judge.calls == [], "deterministic subset must not touch the judge"


def test_direct_mode_adversarial_routes_through_judge(tmp_path):
    """Adversarial questions require a judge because token F1 against the
    refusal string false-fails on paraphrases."""
    loader = LoCoMoLoader(shim_path=FIXTURE)
    inst0 = next(iter(loader.iter_instances()))

    def factory():
        return CompanionSystem(seed=0, witness_active=False)

    judge = _ScriptedJudge(score=1.0, verdict="correct")
    cache = JudgmentCache(cache_dir=tmp_path)
    result = run_direct_mode(
        inst0,
        system_factory=factory,
        judge=judge,
        judgment_cache=cache,
    )
    adversarial_results = [r for r in result.per_question if r.category == ADVERSARIAL]
    assert len(adversarial_results) == 1
    # Judge was invoked for the adversarial row (and only for it).
    assert len(judge.calls) == 1


# ---------- runner: RAG ----------


def test_run_rag_mode_through_companion_system(tmp_path):
    loader = LoCoMoLoader(shim_path=FIXTURE)
    inst1 = list(loader.iter_instances())[1]

    def factory():
        return CompanionSystem(seed=0, witness_active=False)

    reader = MockReader()
    judge = _ScriptedJudge()
    cache = JudgmentCache(cache_dir=tmp_path)
    result = run_rag_mode(
        inst1,
        system_factory=factory,
        reader=reader,
        judge=judge,
        judgment_cache=cache,
    )
    assert len(result.per_question) == 2
    for qr in result.per_question:
        assert qr.category == SINGLE_HOP
        assert 0.0 <= qr.score <= 1.0
    # Deterministic categories still bypass the judge in RAG mode.
    assert judge.calls == []


def test_instance_isolation_guard_trips_on_dirty_system(tmp_path):
    from psychology_base.eval.adapters import Observation

    cs = CompanionSystem(seed=0, witness_active=False)
    cs.observe(
        Observation(
            timestamp=datetime(2026, 1, 1),
            speaker="user",
            content="leak from a previous instance",
        )
    )
    loader = LoCoMoLoader(shim_path=FIXTURE)
    inst = next(iter(loader.iter_instances()))
    judge = _ScriptedJudge()
    cache = JudgmentCache(cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Instance isolation violated"):
        run_direct_mode(
            inst, system_factory=lambda: cs, judge=judge, judgment_cache=cache
        )


def test_mean_by_category():
    r = InstanceResult(
        instance_id="x",
        per_question=[
            QuestionResult(question_id="q0", gold="a", candidate="a", score=1.0, category=SINGLE_HOP),
            QuestionResult(question_id="q1", gold="b", candidate="c", score=0.0, category=SINGLE_HOP),
            QuestionResult(question_id="q2", gold="d", candidate="d", score=0.5, category=ADVERSARIAL),
        ],
    )
    buckets = r.mean_by_category()
    assert buckets[SINGLE_HOP] == pytest.approx(0.5)
    assert buckets[ADVERSARIAL] == pytest.approx(0.5)
