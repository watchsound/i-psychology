"""Phase 7 §6 item 9 — B5 stereotype-collapse unit tests."""

from __future__ import annotations

from psychology_base.eval.benchmarks.b5_stereotype_collapse import (
    B5Report,
    BENIGN_PROMPTS,
    default_seeds,
    score_corpus,
)


def test_default_seeds_shape() -> None:
    seeds = default_seeds()
    assert len(seeds) == 10


def test_benign_prompts_shape() -> None:
    assert len(BENIGN_PROMPTS) == 20


def test_baseline_collapses_to_prompts_only() -> None:
    """Flag off → every seed's response hashes to the same system

    prompt. For 10 seeds × 20 prompts, the baseline unique response
    count equals the number of distinct prompts (= 20).
    """
    seeds = default_seeds()
    report = score_corpus(seeds)
    assert report.unique_responses_baseline == len(BENIGN_PROMPTS)
    # 10 × 20 = 200 total; baseline collapses to 20 → diversity = 0.1
    assert report.diversity_baseline == 20 / 200


def test_conditioned_diversity_meets_pass_bar() -> None:
    seeds = default_seeds()
    report = score_corpus(seeds)
    assert report.meets_pass_bar(), (
        f"conditioned diversity {report.diversity_conditioned:.3f} "
        f"below 0.70 floor; unique_addenda={report.unique_addenda}/10"
    )


def test_conditioned_increases_diversity_over_baseline() -> None:
    seeds = default_seeds()
    report = score_corpus(seeds)
    assert report.diversity_conditioned > report.diversity_baseline


def test_report_is_dataclass() -> None:
    seeds = default_seeds()
    report = score_corpus(seeds)
    assert isinstance(report, B5Report)
    assert report.n_seeds == len(seeds)
    assert report.n_prompts == len(BENIGN_PROMPTS)
