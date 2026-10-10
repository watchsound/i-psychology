"""Phase 2 R6 — per-row recall diagnostics.

Covers:
- ``_recall_diagnostics`` extracts evidence sessions, counts top-K
  fragments, computes session overlap, substring presence, and token
  coverage without touching an LLM.
- ``run_rag_mode`` and ``run_direct_mode`` populate ``recall_diagnostics``
  on every ``QuestionResult`` using fragments actually surfaced by the
  substrate.
- The classification buckets (reader-side vs recall-side) line up with
  the stacked ablation story: an answer literally in the pool but
  mis-extracted is reader-side; an evidence session completely absent
  from the pool is recall-side.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Fragment
from psychology_base.eval.benchmarks.instance import BenchmarkQuestion
from psychology_base.eval.benchmarks.judge import Judgment, JudgmentCache
from psychology_base.eval.benchmarks.locomo import (
    LoCoMoLoader,
    _normalize_evidence_sessions,
    _recall_diagnostics,
    run_direct_mode,
    run_rag_mode,
)
from psychology_base.eval.benchmarks.reader import MockReader


FIXTURE = Path(__file__).parent / "fixtures" / "locomo_mini" / "locomo_shim.jsonl"


class _ScriptedJudge:
    judge_id = "scripted"
    prompt_sha = "scripted"

    def __init__(self) -> None:
        self.calls = []

    def judge(self, *, benchmark, question, gold, candidate, question_type=""):
        self.calls.append((question, gold, candidate, question_type))
        return Judgment(score=1.0, verdict="correct", rationale="scripted")


def _frag(text: str, *tags: str, idx: int = 0) -> Fragment:
    return Fragment(
        id=f"f-{idx}",
        content=text,
        timestamp=datetime(2026, 1, 1),
        tags=tags,
    )


def _q(gold: str, *evidence: str, cue: str = "where?") -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id="q-x",
        cue=cue,
        gold=gold,
        question_type="single-hop",
        metadata={"evidence": tuple(evidence), "raw_answer": gold, "adversarial_answer": ""},
    )


# ---------- evidence-format normalization ----------


def test_normalize_evidence_locomo_native_turn_ids():
    """LoCoMo evidence ships as `D<N>:<T>` turn addresses; map to our
    session_id tagging scheme so the session-overlap check isn't
    spuriously False."""
    assert _normalize_evidence_sessions(("D9:2", "D3:3")) == ("session_9", "session_3")
    # Deduplicates when multiple turns point at the same session.
    assert _normalize_evidence_sessions(("D7:11", "D7:4")) == ("session_7",)


def test_normalize_evidence_passthrough_for_shim_format():
    """Fixture-style ``session_<N>`` evidence passes through unchanged."""
    assert _normalize_evidence_sessions(("session_1", "session_2")) == (
        "session_1",
        "session_2",
    )


def test_normalize_evidence_handles_empty_and_unknown():
    assert _normalize_evidence_sessions(()) == ()
    assert _normalize_evidence_sessions(("",)) == ()
    # Unknown schema → literal pass-through so other benchmarks aren't broken.
    assert _normalize_evidence_sessions(("other-123",)) == ("other-123",)


def test_diagnostics_resolves_locomo_turn_evidence_against_session_tags():
    q = BenchmarkQuestion(
        id="q-x",
        cue="where?",
        gold="Westerly 28",
        question_type="single-hop",
        metadata={"evidence": ("D9:2", "D3:3"), "raw_answer": "Westerly 28", "adversarial_answer": ""},
    )
    frags = (_frag("some content", "session_9", idx=0),)
    diag = _recall_diagnostics(question=q, fragments=frags)
    # evidence sessions are normalized to session_9 / session_3; session_9
    # is present in the pool tags.
    assert diag["evidence_sessions"] == ("session_9", "session_3")
    assert diag["evidence_session_in_top_k"] is True


# ---------- unit tests on _recall_diagnostics ----------


def test_diagnostics_flags_gold_in_pool_substring_match():
    q = _q("Westerly 28", "session_1")
    frags = (
        _frag("Jennifer: Westerly 28, hull is scratched.", "session_1", idx=0),
        _frag("David: no way.", "session_1", idx=1),
    )
    diag = _recall_diagnostics(question=q, fragments=frags)
    assert diag["gold_substring_in_top_k"] is True
    assert diag["evidence_session_in_top_k"] is True
    assert diag["gold_token_coverage"] == 1.0
    assert diag["top_k_sessions"] == ("session_1",)
    assert diag["top_k_fragment_count"] == 2


def test_diagnostics_session_present_gold_absent():
    q = _q("Westerly 28", "session_1")
    frags = (
        _frag("David: how's the weather?", "session_1", idx=0),
        _frag("Jennifer: fine, thanks.", "session_1", idx=1),
    )
    diag = _recall_diagnostics(question=q, fragments=frags)
    assert diag["evidence_session_in_top_k"] is True
    assert diag["gold_substring_in_top_k"] is False
    # Neither "westerly" nor "28" in the fragments.
    assert diag["gold_token_coverage"] == 0.0


def test_diagnostics_recall_missing_evidence_session_not_present():
    q = _q("Westerly 28", "session_1")
    frags = (
        _frag("completely unrelated turn", "session_7", idx=0),
        _frag("another unrelated turn", "session_9", idx=1),
    )
    diag = _recall_diagnostics(question=q, fragments=frags)
    assert diag["evidence_session_in_top_k"] is False
    assert diag["gold_substring_in_top_k"] is False
    assert diag["top_k_sessions"] == ("session_7", "session_9")


def test_diagnostics_no_evidence_metadata():
    q = _q("whatever")  # no evidence sessions
    frags = (_frag("whatever", "session_1", idx=0),)
    diag = _recall_diagnostics(question=q, fragments=frags)
    assert diag["evidence_sessions"] == ()
    # No evidence metadata ⇒ session-level signal is False by definition.
    assert diag["evidence_session_in_top_k"] is False
    # Lexical check still works.
    assert diag["gold_substring_in_top_k"] is True


def test_diagnostics_partial_token_coverage():
    q = _q("19 October 2023", "session_3")
    frags = (
        _frag("we met in October to talk about the move.", "session_3", idx=0),
    )
    diag = _recall_diagnostics(question=q, fragments=frags)
    # "19 october 2023" is not a substring of the normalized pool.
    assert diag["gold_substring_in_top_k"] is False
    # "october" matches, "19" and "2023" don't → coverage = 1/3.
    assert diag["gold_token_coverage"] == round(1 / 3, 3)


def test_diagnostics_normalization_matches_locomo_scorer():
    """Normalization should be the SQuAD-style path, same as token_f1."""
    q = _q("The Westerly 28", "session_1")
    frags = (_frag("A Westerly 28!", "session_1", idx=0),)
    diag = _recall_diagnostics(question=q, fragments=frags)
    # Articles stripped on both sides → substring match.
    assert diag["gold_substring_in_top_k"] is True


def test_diagnostics_empty_fragments():
    q = _q("Westerly 28", "session_1")
    diag = _recall_diagnostics(question=q, fragments=())
    assert diag["top_k_fragment_count"] == 0
    assert diag["top_k_sessions"] == ()
    assert diag["evidence_session_in_top_k"] is False
    assert diag["gold_substring_in_top_k"] is False
    assert diag["gold_token_coverage"] == 0.0


# ---------- integration: run_rag_mode / run_direct_mode populate diag ----------


def test_run_rag_mode_populates_recall_diagnostics(tmp_path):
    loader = LoCoMoLoader(shim_path=FIXTURE)
    inst0 = next(iter(loader.iter_instances()))

    def factory():
        return CompanionSystem(seed=0, witness_active=False)

    reader = MockReader()
    judge = _ScriptedJudge()
    cache = JudgmentCache(cache_dir=tmp_path)
    result = run_rag_mode(
        inst0,
        system_factory=factory,
        reader=reader,
        judge=judge,
        judgment_cache=cache,
    )
    assert len(result.per_question) == 3
    for qr in result.per_question:
        assert qr.recall_diagnostics is not None
        # Every single-hop row has evidence metadata preserved.
        assert "evidence_sessions" in qr.recall_diagnostics
        assert "gold_substring_in_top_k" in qr.recall_diagnostics
        assert isinstance(qr.recall_diagnostics["top_k_fragment_count"], int)


def test_run_direct_mode_populates_recall_diagnostics(tmp_path):
    loader = LoCoMoLoader(shim_path=FIXTURE)
    inst1 = list(loader.iter_instances())[1]

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
        assert qr.recall_diagnostics is not None
        assert qr.recall_diagnostics["evidence_sessions"] == ("session_1",)
