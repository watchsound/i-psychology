"""Step-2 plumbing tests for the EpBench adapter.

Scope — mechanical correctness only:
  - answer-string parsing (format shim ↔ item set)
  - F1 scorer edge cases
  - loader round-trips the JSONL shim into BenchmarkInstance
  - direct-mode runner streams through CompanionSystem without crashing
  - instance-isolation guard trips on dirty systems

Not in scope — the step-2 gate (F1 ≥ 0.70 on a 50-instance smoke subset on
real EpBench data). That requires the Figshare bundle + a parquet→JSONL
converter; tracked as the real-data integration task.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.epbench import (
    EpBenchLoader,
    InstanceResult,
    QuestionResult,
    parse_answer_items,
    run_direct_mode,
    score_candidate,
    set_f1,
)
from psychology_base.eval.benchmarks.instance import BenchmarkInstance


FIXTURE = Path(__file__).parent / "fixtures" / "epbench_mini" / "epbench_shim.jsonl"


# ---------- parse_answer_items ----------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", ()),
        ("alone", ("alone",)),
        ("mira and ellis", ("mira", "ellis")),
        ("harborfront cafe, chandlery, and harborfront cafe", ("harborfront cafe", "chandlery", "harborfront cafe")),
        ("\"a, b, and c\"", ("a", "b", "c")),
        ("A, B And C", ("a", "b", "c")),
    ],
)
def test_parse_answer_items(raw, expected):
    assert parse_answer_items(raw) == expected


# ---------- set_f1 ----------


@pytest.mark.parametrize(
    "gold,cand,expected",
    [
        ((), (), 1.0),
        (("a",), (), 0.0),
        ((), ("a",), 0.0),
        (("a", "b"), ("a", "b"), 1.0),
        (("a", "b"), ("a",), pytest.approx(2 / 3)),
        (("a",), ("a", "b"), pytest.approx(2 / 3)),
        (("a", "b"), ("c", "d"), 0.0),
        (("a", "a", "b"), ("a", "b"), 1.0),  # set-F1 collapses duplicates
    ],
)
def test_set_f1(gold, cand, expected):
    assert set_f1(gold, cand) == expected


def test_score_candidate_end_to_end():
    f1 = score_candidate("mira and ellis", "ellis, mira, and tomas")
    # gold {mira, ellis}; cand {mira, ellis, tomas}; tp=2, p=2/3, r=1 → F1=0.8
    assert f1 == pytest.approx(0.8)


# ---------- loader ----------


def test_loader_round_trips_fixture():
    loader = EpBenchLoader(shim_path=FIXTURE)
    instances = list(loader.iter_instances())
    assert [i.instance_id for i in instances] == ["mini-0", "mini-1"]

    inst0 = instances[0]
    assert isinstance(inst0, BenchmarkInstance)
    assert len(inst0.history) == 3
    assert len(inst0.questions) == 3
    # First event: scaffold prepended to content.
    first_turn = inst0.history[0]
    assert "harborfront cafe" in first_turn.content
    assert "mira" in first_turn.content
    assert "entities=" in first_turn.content
    assert first_turn.session_id == "chapter-0"
    # First question: gold passthrough + retrieval_type captured.
    q0 = inst0.questions[0]
    assert q0.gold == "harborfront cafe, chandlery, and harborfront cafe"
    assert q0.question_type == "entity-to-places"
    assert q0.metadata["n_items"] == 2


def test_loader_handles_missing_file(tmp_path):
    missing = tmp_path / "nope.jsonl"
    loader = EpBenchLoader(shim_path=missing)
    assert list(loader.iter_instances()) == []


def test_loader_resolves_root_default(tmp_path):
    """If shim_path is None, loader falls back to <root>/epbench_shim.jsonl."""
    (tmp_path / "epbench_shim.jsonl").write_text(
        '{"instance_id": "x", "events": [], "questions": []}\n',
        encoding="utf-8",
    )
    loader = EpBenchLoader()
    instances = list(loader.iter_instances(root=tmp_path))
    assert [i.instance_id for i in instances] == ["x"]


# ---------- direct-mode runner ----------


def test_run_direct_mode_against_companion_system():
    """End-to-end: observe each event, recall per question, score. Does not
    check hitting the F1 ≥ 0.70 gate — that requires real EpBench data. Only
    checks that the pipeline runs clean and produces a result per question."""
    loader = EpBenchLoader(shim_path=FIXTURE)
    instance = next(iter(loader.iter_instances()))

    def factory():
        return CompanionSystem(seed=0, witness_active=False)

    result = run_direct_mode(instance, system_factory=factory)
    assert isinstance(result, InstanceResult)
    assert result.instance_id == "mini-0"
    assert len(result.per_question) == len(instance.questions)
    for qr in result.per_question:
        assert isinstance(qr, QuestionResult)
        assert 0.0 <= qr.f1 <= 1.0
        assert qr.gold != ""
        assert qr.candidate != ""


def test_instance_isolation_guard_trips_on_dirty_system():
    from datetime import datetime

    from psychology_base.eval.adapters import Observation

    cs = CompanionSystem(seed=0, witness_active=False)
    cs.observe(
        Observation(
            timestamp=datetime(2026, 1, 1),
            speaker="user",
            content="leak from a previous instance",
        )
    )
    loader = EpBenchLoader(shim_path=FIXTURE)
    instance = next(iter(loader.iter_instances()))
    with pytest.raises(RuntimeError, match="Instance isolation violated"):
        run_direct_mode(instance, system_factory=lambda: cs)


def test_mean_f1_averages_per_question_scores():
    r = InstanceResult(
        instance_id="x",
        per_question=[
            QuestionResult(question_id="q0", gold="a", candidate="a", f1=1.0),
            QuestionResult(question_id="q1", gold="b", candidate="c", f1=0.0),
            QuestionResult(question_id="q2", gold="a and b", candidate="a", f1=2 / 3),
        ],
    )
    assert r.mean_f1 == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)


def test_mean_f1_empty_result_is_zero():
    assert InstanceResult(instance_id="empty").mean_f1 == 0.0
