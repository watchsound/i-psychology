"""Phase 7 §9 — B4 import-to-profile accuracy scorer unit tests.

Deterministic (seeded persona generation; lexical inference). Does NOT
run the full ``default_personas()`` × N={50,200,500} grid — that is
the benchmark runner's job. These tests pin the scorer's contract
and verify at least the sharp-single-dim-extreme personas infer in the
right direction.
"""

from __future__ import annotations

import pytest

from psychology_base.eval.benchmarks.b4_import_to_profile import (
    B4Report,
    OceanDim,
    PerPersonaReport,
    SyntheticPersona,
    default_personas,
    generate_persona_corpus,
    score_corpus,
    score_persona,
)


def test_default_personas_shape() -> None:
    personas = default_personas()
    assert len(personas) == 10
    # Each persona has 5 dims; one extreme (1 or 5) + 4 at midpoint.
    for p in personas:
        extremes = [v for v in p.ocean.values() if v in (1.0, 5.0)]
        assert len(extremes) == 1
        midpoints = [v for v in p.ocean.values() if v == 3.0]
        assert len(midpoints) == 4


def test_generate_persona_corpus_message_count() -> None:
    persona = SyntheticPersona(
        "high-n",
        {d: 3.0 for d in OceanDim} | {OceanDim.NEUROTICISM: 5.0},
    )
    turns = generate_persona_corpus(persona, n_messages=50, seed=1)
    assert len(turns) == 50
    assert all(t.speaker == "user" for t in turns)


def test_score_persona_high_neuroticism_inferred_above_midpoint() -> None:
    persona = SyntheticPersona(
        "high-n",
        {d: 3.0 for d in OceanDim} | {OceanDim.NEUROTICISM: 5.0},
    )
    turns = generate_persona_corpus(persona, n_messages=50, seed=1)
    report = score_persona(persona, turns)
    assert report.inferred[OceanDim.NEUROTICISM] >= 3.5, (
        f"high-neuroticism persona should infer above 3.5, got "
        f"{report.inferred[OceanDim.NEUROTICISM]:.2f}"
    )


def test_score_persona_low_extraversion_inferred_below_midpoint() -> None:
    persona = SyntheticPersona(
        "low-e",
        {d: 3.0 for d in OceanDim} | {OceanDim.EXTRAVERSION: 1.0},
    )
    turns = generate_persona_corpus(persona, n_messages=50, seed=1)
    report = score_persona(persona, turns)
    assert report.inferred[OceanDim.EXTRAVERSION] <= 3.5, (
        f"low-extraversion persona should infer at or below 3.5, got "
        f"{report.inferred[OceanDim.EXTRAVERSION]:.2f}"
    )


def test_score_corpus_returns_b4_report() -> None:
    personas = default_personas()[:4]
    report = score_corpus(personas, n_messages=50, seed=0)
    assert isinstance(report, B4Report)
    assert len(report.per_persona) == 4
    assert set(report.mae_per_dim_mean.keys()) == set(OceanDim)


def test_score_corpus_meets_pass_bar_at_n200_default_personas() -> None:
    """Plan §9 pass bar: ≥ 3/5 dims within ±0.5 Likert at N=200.

    The default-persona fixture is heavily tuned (single-dim-extreme
    + midpoint-everywhere-else), so this is a floor test, not SOTA
    evidence. Primary evidence lives in the runner's JSON output.
    """
    personas = default_personas()
    report = score_corpus(personas, n_messages=200, seed=0)
    assert report.meets_pass_bar(), (
        f"B4 pass bar not met at N=200 — mean_dims_within_half_likert="
        f"{report.mean_dims_within_half_likert:.2f}, "
        f"MAE_mean={ {d.value: round(v, 2) for d, v in report.mae_per_dim_mean.items()} }"
    )
