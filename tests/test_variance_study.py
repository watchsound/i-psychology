"""Phase 4a §6d — Multi-trial variance study tests.

Uses a scripted run_fn + fake InstanceResult so the aggregation math is
exercised independently of any live LLM path.
"""

from __future__ import annotations

import math

import pytest

from psychology_base.eval.benchmarks.locomo import InstanceResult, QuestionResult
from psychology_base.eval.benchmarks.variance_study import (
    PerQuestionVariance,
    VarianceReport,
    run_multi_trial,
)


def _make_result(instance_id: str, scores: list[tuple[str, str, float]]) -> InstanceResult:
    r = InstanceResult(instance_id=instance_id)
    for qid, cat, score in scores:
        r.per_question.append(
            QuestionResult(
                question_id=qid,
                gold="",
                candidate="",
                score=score,
                category=cat,
            )
        )
    return r


def test_deterministic_run_yields_zero_variance() -> None:
    def run_fn(*, instance, system_factory, **kwargs):
        return _make_result("inst-1", [
            ("q1", "single_hop", 0.6),
            ("q2", "multi_hop", 0.5),
        ])

    report = run_multi_trial(
        instance=object(),
        system_factory=lambda: None,
        run_fn=run_fn,
        n_trials=5,
    )
    assert report.instance_mean == pytest.approx(0.55)
    assert report.instance_stderr == pytest.approx(0.0)
    assert report.ci95_lower == pytest.approx(0.55)
    assert report.ci95_upper == pytest.approx(0.55)
    assert all(pq.stderr == pytest.approx(0.0) for pq in report.per_question)
    assert report.ci_warning is not None  # N < 10


def test_stochastic_run_captures_per_question_variance() -> None:
    trial_scores = [0.3, 0.5, 0.7, 0.4, 0.6]
    counter = {"i": 0}

    def run_fn(*, instance, system_factory, **kwargs):
        i = counter["i"]
        counter["i"] += 1
        return _make_result("inst-1", [("q1", "single_hop", trial_scores[i])])

    report = run_multi_trial(
        instance=object(),
        system_factory=lambda: None,
        run_fn=run_fn,
        n_trials=5,
    )
    assert report.n_trials == 5
    assert report.per_question[0].scores == tuple(trial_scores)
    assert report.per_question[0].mean == pytest.approx(0.5)
    # Sample stddev of [0.3, 0.5, 0.7, 0.4, 0.6] = sqrt(((0.2^2 + 0^2 + 0.2^2 + 0.1^2 + 0.1^2) / 4))
    expected_stddev = math.sqrt((0.04 + 0.0 + 0.04 + 0.01 + 0.01) / 4)
    expected_stderr = expected_stddev / math.sqrt(5)
    assert report.per_question[0].stderr == pytest.approx(expected_stderr)
    assert report.instance_stderr == pytest.approx(expected_stderr)


def test_system_factory_called_each_trial() -> None:
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return object()

    def run_fn(*, instance, system_factory, **kwargs):
        system_factory()  # emulate real run_fn behaviour
        return _make_result("inst-1", [("q1", "single_hop", 0.5)])

    run_multi_trial(
        instance=object(),
        system_factory=factory,
        run_fn=run_fn,
        n_trials=3,
    )
    assert calls["n"] == 3


def test_mismatched_question_ids_raises() -> None:
    counter = {"i": 0}

    def run_fn(*, instance, system_factory, **kwargs):
        i = counter["i"]
        counter["i"] += 1
        if i == 0:
            return _make_result("inst-1", [("q1", "sh", 0.5), ("q2", "sh", 0.5)])
        return _make_result("inst-1", [("q1", "sh", 0.5), ("q3", "sh", 0.5)])

    with pytest.raises(ValueError, match="question_ids differ"):
        run_multi_trial(
            instance=object(),
            system_factory=lambda: None,
            run_fn=run_fn,
            n_trials=2,
        )


def test_zero_trials_raises() -> None:
    def run_fn(*, instance, system_factory, **kwargs):
        return _make_result("inst-1", [])

    with pytest.raises(ValueError, match="≥ 1"):
        run_multi_trial(
            instance=object(),
            system_factory=lambda: None,
            run_fn=run_fn,
            n_trials=0,
        )


def test_ten_trials_removes_ci_warning() -> None:
    def run_fn(*, instance, system_factory, **kwargs):
        return _make_result("inst-1", [("q1", "sh", 0.5)])

    report = run_multi_trial(
        instance=object(),
        system_factory=lambda: None,
        run_fn=run_fn,
        n_trials=10,
    )
    assert report.ci_warning is None


def test_per_category_breakdown() -> None:
    def run_fn(*, instance, system_factory, **kwargs):
        return _make_result("inst-1", [
            ("q1", "single_hop", 0.8),
            ("q2", "single_hop", 0.6),
            ("q3", "multi_hop", 0.4),
            ("q4", "multi_hop", 0.2),
        ])

    report = run_multi_trial(
        instance=object(),
        system_factory=lambda: None,
        run_fn=run_fn,
        n_trials=3,
    )
    assert report.per_category_mean["single_hop"] == pytest.approx(0.7)
    assert report.per_category_mean["multi_hop"] == pytest.approx(0.3)
    # Deterministic run → zero per-category stderr.
    assert report.per_category_stderr["single_hop"] == pytest.approx(0.0)


def test_meets_confidence_floor_default_regression_gate() -> None:
    scores = [0.60, 0.55, 0.65, 0.58, 0.62, 0.59, 0.61, 0.57, 0.63, 0.60]
    counter = {"i": 0}

    def run_fn(*, instance, system_factory, **kwargs):
        i = counter["i"]
        counter["i"] += 1
        return _make_result("inst-1", [("q1", "sh", scores[i])])

    report = run_multi_trial(
        instance=object(),
        system_factory=lambda: None,
        run_fn=run_fn,
        n_trials=10,
    )
    # Mean 0.60, stderr small → CI95 lower > 0.514.
    assert report.meets_confidence_floor(floor=0.514) is True
    assert report.meets_confidence_floor(floor=0.70) is False


def test_run_kwargs_passed_through() -> None:
    captured: dict = {}

    def run_fn(*, instance, system_factory, judge, cache, **kwargs):
        captured["judge"] = judge
        captured["cache"] = cache
        return _make_result("inst-1", [("q1", "sh", 0.5)])

    run_multi_trial(
        instance=object(),
        system_factory=lambda: None,
        run_fn=run_fn,
        n_trials=2,
        run_kwargs={"judge": "judge-A", "cache": "cache-X"},
    )
    assert captured == {"judge": "judge-A", "cache": "cache-X"}


def test_single_trial_degenerate_case() -> None:
    def run_fn(*, instance, system_factory, **kwargs):
        return _make_result("inst-1", [("q1", "sh", 0.7)])

    report = run_multi_trial(
        instance=object(),
        system_factory=lambda: None,
        run_fn=run_fn,
        n_trials=1,
    )
    assert report.n_trials == 1
    assert report.instance_stderr == 0.0
    assert report.ci_warning is not None
    assert "N=1" in report.ci_warning
