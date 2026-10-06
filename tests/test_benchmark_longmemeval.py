"""Step-3 plumbing tests for the LongMemEval adapter + RAG reader + judge.

Scope — mechanical correctness only:
  - loader round-trips the JSONL shim into BenchmarkInstance (flattens
    sessions, carries question metadata including oracle session ids)
  - reader surfaces the fragment under the locked prompt
  - judge parses well-formed and malformed JSON responses
  - judge cache round-trips a judgment
  - direct-mode + RAG-mode runners stream through CompanionSystem using a
    MockProvider router and MockReader + MockJudge, producing a score per
    question

Not in scope — the step-3 gate (LongMemEval ≥ 90% on a 50-instance smoke
subset in RAG mode). That requires the HuggingFace dataset + a Tongyi
qwen-max judge spend; tracked as the real-data integration task alongside
the EpBench parquet shim.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Fragment
from psychology_base.eval.benchmarks.instance import BenchmarkInstance
from psychology_base.eval.benchmarks.judge import (
    JudgmentCache,
    JudgmentKey,
    LLMJudge,
    _parse_judge_response,
)
from psychology_base.eval.benchmarks.longmemeval import (
    InstanceResult,
    LongMemEvalLoader,
    QuestionResult,
    run_direct_mode,
    run_rag_mode,
)
from psychology_base.eval.benchmarks.reader import (
    LLMReader,
    MockReader,
    _format_fragments,
)
from psychology_base.llm.config import LLMConfig, LocalOllamaConfig, CacheConfig
from psychology_base.llm.providers import MockProvider
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


FIXTURE = Path(__file__).parent / "fixtures" / "longmemeval_mini" / "longmemeval_shim.jsonl"


# ---------- loader ----------


def test_loader_round_trips_fixture():
    loader = LongMemEvalLoader(shim_path=FIXTURE)
    instances = list(loader.iter_instances())
    assert [i.instance_id for i in instances] == ["lm-mini-0", "lm-mini-1", "lm-mini-2"]

    inst0 = instances[0]
    assert isinstance(inst0, BenchmarkInstance)
    # 2 sessions: session 0 has 5 turns (intro + 2 user + 2 asst),
    # session 1 has 2 turns.
    assert len(inst0.history) == 7
    # Session boundaries preserved as speaker + session_id.
    first = inst0.history[0]
    assert first.speaker == "user"
    assert first.session_id == "s0"
    assert "Mira" in first.content
    # Session 1 anchors at its own date.
    turn_after_boundary = inst0.history[5]
    assert turn_after_boundary.session_id == "s1"
    # Question metadata survives.
    assert len(inst0.questions) == 1
    q = inst0.questions[0]
    assert q.id == "lm-mini-0"
    assert q.question_type == "single-session-user"
    assert q.gold == "red"
    assert q.metadata["answer_session_ids"] == ("s0",)


def test_loader_handles_missing_file(tmp_path):
    loader = LongMemEvalLoader(shim_path=tmp_path / "nope.jsonl")
    assert list(loader.iter_instances()) == []


def test_loader_timestamp_cadence_within_session():
    """Within a session, turn N is observed one minute after turn N-1."""
    loader = LongMemEvalLoader(shim_path=FIXTURE)
    inst0 = next(iter(loader.iter_instances()))
    # Session 0 has 5 turns. First at session date, successors +1min each.
    session_0_turns = [t for t in inst0.history if t.session_id == "s0"]
    assert len(session_0_turns) == 5
    deltas = [
        (session_0_turns[i + 1].timestamp - session_0_turns[i].timestamp).total_seconds()
        for i in range(len(session_0_turns) - 1)
    ]
    assert deltas == [60.0, 60.0, 60.0, 60.0]


# ---------- reader ----------


def _frag(idx: int, content: str) -> Fragment:
    return Fragment(
        id=f"f{idx}",
        content=content,
        timestamp=datetime(2026, 5, 1, 10, 0, 0),
        affect={},
        tags=(),
    )


def test_mock_reader_prefers_first_fragment():
    reader = MockReader()
    out = reader.answer(
        question="color?",
        recall_content="reconstruction",
        fragments=[_frag(0, "the bike is red")],
    )
    assert out == "the bike is red"


def test_mock_reader_falls_back_to_recall_content():
    reader = MockReader()
    out = reader.answer(
        question="color?",
        recall_content="reconstruction says red",
        fragments=[],
    )
    assert out == "reconstruction says red"


def test_mock_reader_falls_back_to_not_stated():
    reader = MockReader()
    out = reader.answer(question="color?", recall_content="", fragments=[])
    assert out == "not stated"


def test_format_fragments_caps_at_limit():
    frags = [_frag(i, f"content-{i}") for i in range(20)]
    rendered = _format_fragments(frags, limit=5)
    assert "content-0" in rendered
    assert "content-4" in rendered
    assert "content-5" not in rendered
    assert "15 more fragments omitted" in rendered


def test_format_fragments_empty():
    assert _format_fragments([]) == "(no fragments surfaced)"


def test_llm_reader_calls_router_advanced_tier():
    """Reader must route to ADVANCED tier and surface the completion content."""
    mock = MockProvider(
        responder=lambda msgs, tier: "red" if tier == TaskTier.ADVANCED else "WRONG"
    )
    config = LLMConfig(
        active_provider="mock",
        providers={},
        local_ollama=LocalOllamaConfig(enabled=False),
        cache=CacheConfig(enabled=False),
    )
    router = LLMRouter(config, providers={"mock": mock})
    reader = LLMReader(router=router)
    out = reader.answer(
        question="color?",
        recall_content="reconstruction",
        fragments=[_frag(0, "the bike is red")],
    )
    assert out == "red"
    assert len(mock.calls) == 1
    _, tier = mock.calls[0]
    assert tier == TaskTier.ADVANCED


def test_llm_reader_prompt_shape():
    """Verify the locked system prompt + user prompt contain the expected
    pieces. The prompt is step-3-locked; drift here invalidates every cached
    judgment, so this test guards against accidental edits."""
    recorded = []

    def responder(msgs, tier):
        recorded.append(list(msgs))
        return "ok"

    mock = MockProvider(responder=responder)
    config = LLMConfig(
        active_provider="mock",
        providers={},
        local_ollama=LocalOllamaConfig(enabled=False),
        cache=CacheConfig(enabled=False),
    )
    router = LLMRouter(config, providers={"mock": mock})
    reader = LLMReader(router=router)
    reader.answer(
        question="what color?",
        recall_content="rebuilt reconstruction",
        fragments=[_frag(0, "the bike is red")],
    )
    msgs = recorded[0]
    assert msgs[0].role == "system"
    assert "not stated" in msgs[0].content.lower()
    assert msgs[1].role == "user"
    assert "what color?" in msgs[1].content
    assert "rebuilt reconstruction" in msgs[1].content
    assert "the bike is red" in msgs[1].content


# ---------- judge parse ----------


@pytest.mark.parametrize(
    "raw,expected_score,expected_verdict",
    [
        ('{"score": 1.0, "verdict": "correct", "rationale": "match"}', 1.0, "correct"),
        ('{"score": 0.5, "verdict": "partial", "rationale": "half"}', 0.5, "partial"),
        ('{"score": 0, "verdict": "incorrect", "rationale": "miss"}', 0.0, "incorrect"),
        # Code-fenced
        ('```json\n{"score": 1, "verdict": "correct", "rationale": "x"}\n```', 1.0, "correct"),
        # Prefix prose + JSON
        ('Thinking... {"score": 0.5, "verdict": "partial", "rationale": "half"} done.', 0.5, "partial"),
        # Score clamped above 1
        ('{"score": 5.0, "verdict": "correct", "rationale": "x"}', 1.0, "correct"),
        # Score clamped below 0
        ('{"score": -1.0, "verdict": "incorrect", "rationale": "x"}', 0.0, "incorrect"),
    ],
)
def test_parse_judge_response_well_formed(raw, expected_score, expected_verdict):
    j = _parse_judge_response(raw)
    assert j.score == expected_score
    assert j.verdict == expected_verdict


def test_parse_judge_response_malformed_falls_back():
    j = _parse_judge_response("this is not json at all")
    assert j.score == 0.0
    assert j.verdict == "unparsed"
    assert "judge_response_unparsed" in j.rationale


def test_llm_judge_resolves_judge_id_from_router():
    mock = MockProvider(
        responder=lambda msgs, tier: '{"score": 1.0, "verdict": "correct", "rationale": "ok"}'
    )
    config = LLMConfig(
        active_provider="mock",
        providers={},
        local_ollama=LocalOllamaConfig(enabled=False),
        cache=CacheConfig(enabled=False),
    )
    router = LLMRouter(config, providers={"mock": mock})
    judge = LLMJudge(router=router)
    # judge_id is "<provider>:<advanced-model-id>" from the router routing.
    assert judge.judge_id.startswith("mock:")
    assert "mock-advanced" in judge.judge_id
    # prompt_sha is a non-empty short hex from the locked prompt.
    assert len(judge.prompt_sha) == 16


def test_llm_judge_round_trip():
    mock = MockProvider(
        responder=lambda msgs, tier: '{"score": 0.5, "verdict": "partial", "rationale": "half-right"}'
    )
    config = LLMConfig(
        active_provider="mock",
        providers={},
        local_ollama=LocalOllamaConfig(enabled=False),
        cache=CacheConfig(enabled=False),
    )
    router = LLMRouter(config, providers={"mock": mock})
    judge = LLMJudge(router=router)
    verdict = judge.judge(
        benchmark="longmemeval",
        question="color?",
        gold="red",
        candidate="crimson",
        question_type="single-session-user",
    )
    assert verdict.score == 0.5
    assert verdict.verdict == "partial"
    # ADVANCED tier used.
    assert all(call_tier == TaskTier.ADVANCED for _, call_tier in mock.calls)


# ---------- judge cache ----------


def test_judgment_cache_round_trip(tmp_path):
    cache = JudgmentCache(cache_dir=tmp_path)
    mock = MockProvider(
        responder=lambda msgs, tier: '{"score": 1.0, "verdict": "correct", "rationale": "ok"}'
    )
    config = LLMConfig(
        active_provider="mock",
        providers={},
        local_ollama=LocalOllamaConfig(enabled=False),
        cache=CacheConfig(enabled=False),
    )
    router = LLMRouter(config, providers={"mock": mock})
    judge = LLMJudge(router=router)

    # First call hits the judge.
    _ = cache.judge_cached(
        client=judge,
        benchmark="longmemeval",
        instance_id="x",
        question_id="q0",
        question="color?",
        gold="red",
        candidate="red",
    )
    assert len(mock.calls) == 1
    assert cache.misses == 1
    # Second call hits the cache.
    _ = cache.judge_cached(
        client=judge,
        benchmark="longmemeval",
        instance_id="x",
        question_id="q0",
        question="color?",
        gold="red",
        candidate="red",
    )
    assert len(mock.calls) == 1  # no second LLM call
    assert cache.hits == 1


# ---------- direct-mode + RAG-mode runners ----------


class _ScriptedJudge:
    """Scripted judge for runner tests; always returns a fixed Judgment."""

    judge_id = "scripted"
    prompt_sha = "scripted"

    def __init__(self, score: float = 1.0, verdict: str = "correct") -> None:
        self._score = score
        self._verdict = verdict
        self.calls = []

    def judge(self, *, benchmark, question, gold, candidate, question_type=""):
        from psychology_base.eval.benchmarks.judge import Judgment

        self.calls.append((benchmark, question, gold, candidate, question_type))
        return Judgment(score=self._score, verdict=self._verdict, rationale="scripted")


def test_run_direct_mode_through_companion_system(tmp_path):
    loader = LongMemEvalLoader(shim_path=FIXTURE)
    instance = next(iter(loader.iter_instances()))

    def factory():
        return CompanionSystem(seed=0, witness_active=False)

    judge = _ScriptedJudge(score=1.0, verdict="correct")
    cache = JudgmentCache(cache_dir=tmp_path)

    result = run_direct_mode(
        instance,
        system_factory=factory,
        judge=judge,
        judgment_cache=cache,
    )
    assert isinstance(result, InstanceResult)
    assert result.instance_id == "lm-mini-0"
    assert len(result.per_question) == 1
    qr = result.per_question[0]
    assert isinstance(qr, QuestionResult)
    assert qr.score == 1.0
    # Judge was called with the SUT's raw recall.content.
    assert len(judge.calls) == 1
    _, _, gold, candidate, _ = judge.calls[0]
    assert gold == "red"
    # Candidate is whatever CompanionSystem reconstructed; it's non-empty
    # because the fixture turns include "red bicycle".
    assert candidate is not None


def test_run_rag_mode_through_companion_system(tmp_path):
    loader = LongMemEvalLoader(shim_path=FIXTURE)
    instance = next(iter(loader.iter_instances()))

    def factory():
        return CompanionSystem(seed=0, witness_active=False)

    reader = MockReader()
    judge = _ScriptedJudge(score=0.5, verdict="partial")
    cache = JudgmentCache(cache_dir=tmp_path)

    result = run_rag_mode(
        instance,
        system_factory=factory,
        reader=reader,
        judge=judge,
        judgment_cache=cache,
    )
    assert len(result.per_question) == 1
    qr = result.per_question[0]
    assert qr.score == 0.5
    # The RAG runner should have called the reader → judge with the reader's
    # output, which under MockReader is the first fragment or recall content.
    assert len(judge.calls) == 1


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
    loader = LongMemEvalLoader(shim_path=FIXTURE)
    instance = next(iter(loader.iter_instances()))
    judge = _ScriptedJudge()
    cache = JudgmentCache(cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Instance isolation violated"):
        run_direct_mode(
            instance,
            system_factory=lambda: cs,
            judge=judge,
            judgment_cache=cache,
        )


def test_mean_score_averages_per_question():
    r = InstanceResult(
        instance_id="x",
        per_question=[
            QuestionResult(question_id="q0", gold="a", candidate="a", score=1.0),
            QuestionResult(question_id="q1", gold="b", candidate="c", score=0.0),
            QuestionResult(question_id="q2", gold="d", candidate="d", score=0.5),
        ],
    )
    assert r.mean_score == pytest.approx(0.5)


def test_mean_score_empty_result_is_zero():
    assert InstanceResult(instance_id="empty").mean_score == 0.0
