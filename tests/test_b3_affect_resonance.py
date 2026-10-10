"""M3 unit tests — B3 scorer sub-metrics.

Covers:

1. ``fragment_within_radius`` matches at distance ≤ radius, rejects
   fragments beyond radius, rejects fragments without numeric affect.
2. ``resonance_recall`` counts the fraction of recalls whose top-K
   fragments contain ≥ 1 within-radius fragment.
3. ``affect_match_at_k`` macro-averages the within-radius fraction
   across recalls (so one noisy recall doesn't dominate).
4. ``over_disclosure_rate`` placeholder behavior.
5. ``B3Scores.passes_gate`` enforces the three gate thresholds from
   the M3 README (≥ 60% resonance, ≥ 50% match@K, < 10% over-disc).
"""

from __future__ import annotations

from datetime import datetime, timezone

from psychology_base.eval.adapters import Fragment, Recall, make_fragment
from psychology_base.eval.benchmarks.b3_affect_resonance import (
    B3Scores,
    affect_match_at_k,
    fragment_within_radius,
    over_disclosure_rate,
    resonance_recall,
    score_b3,
)


def _dt(seconds: int = 0) -> datetime:
    return datetime(2026, 1, 1, 12, 0, seconds, tzinfo=timezone.utc)


def _frag(aff: dict, t: int = 0) -> Fragment:
    return make_fragment(
        content=f"frag-{t}",
        timestamp=_dt(t),
        affect=aff,
    )


# ------------------- fragment_within_radius ------------------------- #


def test_fragment_within_radius_true_on_exact_match():
    cue = {"valence": -0.7, "arousal": 0.8}
    frag = _frag(cue)
    assert fragment_within_radius(frag, cue) is True


def test_fragment_within_radius_false_on_opposite_corner():
    cue = {"valence": 1.0, "arousal": 1.0}
    frag = _frag({"valence": -1.0, "arousal": -1.0})
    assert fragment_within_radius(frag, cue, match_radius=0.4) is False


def test_fragment_within_radius_boundary_is_inclusive():
    # distance = 0.4 exactly → inclusive.
    cue = {"valence": 0.0, "arousal": 0.0}
    frag = _frag({"valence": 0.4, "arousal": 0.0})
    # pair_affect_score = 1 - 0.4/sqrt(8); _pad_distance = 0.4 exactly.
    assert fragment_within_radius(frag, cue, match_radius=0.4) is True


def test_fragment_without_numeric_affect_not_within_radius():
    cue = {"valence": -0.7, "arousal": 0.8}
    bare = _frag({})
    non_numeric = _frag({"valence": "high", "arousal": "low"})
    assert fragment_within_radius(bare, cue) is False
    assert fragment_within_radius(non_numeric, cue) is False


def test_fragment_within_radius_cue_without_affect_returns_false():
    frag = _frag({"valence": -0.7, "arousal": 0.8})
    assert fragment_within_radius(frag, {}) is False
    assert fragment_within_radius(frag, None) is False


# ----------------------- resonance_recall --------------------------- #


def _recall(frags: list[Fragment]) -> Recall:
    return Recall(cue="x", content="", fragments_used=tuple(frags))


def test_resonance_recall_empty_returns_zero():
    assert resonance_recall([]) == 0.0


def test_resonance_recall_all_match_returns_one():
    cue = {"valence": -0.7, "arousal": 0.8}
    r1 = _recall([_frag(cue, t=1), _frag(cue, t=2)])
    r2 = _recall([_frag(cue, t=3)])
    assert resonance_recall([(r1, cue), (r2, cue)]) == 1.0


def test_resonance_recall_none_match_returns_zero():
    cue = {"valence": -0.7, "arousal": 0.8}
    far = {"valence": 0.7, "arousal": -0.8}
    r1 = _recall([_frag(far, t=1)])
    assert resonance_recall([(r1, cue)]) == 0.0


def test_resonance_recall_mixed_counts_only_hits():
    cue = {"valence": -0.7, "arousal": 0.8}
    far = {"valence": 0.7, "arousal": -0.8}
    hit = _recall([_frag(cue, t=1), _frag(far, t=2)])  # one in-radius
    miss = _recall([_frag(far, t=3), _frag(far, t=4)])
    out = resonance_recall([(hit, cue), (miss, cue)])
    assert out == 0.5


def test_resonance_recall_respects_top_k_cutoff():
    cue = {"valence": -0.7, "arousal": 0.8}
    far = {"valence": 0.7, "arousal": -0.8}
    # Hit is at position 3; top_k=2 should miss it.
    r = _recall([_frag(far, t=1), _frag(far, t=2), _frag(cue, t=3)])
    assert resonance_recall([(r, cue)], top_k=2) == 0.0
    assert resonance_recall([(r, cue)], top_k=3) == 1.0


# ---------------------- affect_match_at_k --------------------------- #


def test_affect_match_at_k_counts_fraction_within_radius():
    cue = {"valence": -0.7, "arousal": 0.8}
    far = {"valence": 0.7, "arousal": -0.8}
    r = _recall([_frag(cue, t=1), _frag(cue, t=2), _frag(far, t=3)])
    # 2 of 3 fragments within radius.
    out = affect_match_at_k([(r, cue)], top_k=3)
    assert abs(out - 2 / 3) < 1e-6


def test_affect_match_at_k_macro_averages_across_recalls():
    cue = {"valence": -0.7, "arousal": 0.8}
    far = {"valence": 0.7, "arousal": -0.8}
    r_clean = _recall([_frag(cue, t=1), _frag(cue, t=2)])  # 1.0
    r_noisy = _recall([_frag(far, t=3), _frag(far, t=4)])  # 0.0
    out = affect_match_at_k([(r_clean, cue), (r_noisy, cue)], top_k=2)
    assert out == 0.5


def test_affect_match_at_k_empty_pool_scores_zero():
    cue = {"valence": -0.7, "arousal": 0.8}
    empty = _recall([])
    assert affect_match_at_k([(empty, cue)]) == 0.0


def test_affect_match_at_k_empty_input_returns_zero():
    assert affect_match_at_k([]) == 0.0


# ---------------------- over_disclosure_rate ------------------------ #


def test_over_disclosure_rate_counts_flags():
    r = _recall([])
    assert over_disclosure_rate([(r, True), (r, False), (r, True)]) == 2 / 3


def test_over_disclosure_rate_empty_returns_zero():
    assert over_disclosure_rate([]) == 0.0


# ------------------------- B3Scores.passes_gate ------------------------ #


def test_passes_gate_requires_all_thresholds():
    good = B3Scores(resonance_recall=0.62, affect_match_at_k=0.55, over_disclosure_rate=0.05)
    assert good.passes_gate() is True

    low_resonance = B3Scores(resonance_recall=0.55, affect_match_at_k=0.55, over_disclosure_rate=0.0)
    assert low_resonance.passes_gate() is False

    low_match = B3Scores(resonance_recall=0.70, affect_match_at_k=0.45, over_disclosure_rate=0.0)
    assert low_match.passes_gate() is False

    high_disclosure = B3Scores(resonance_recall=0.70, affect_match_at_k=0.55, over_disclosure_rate=0.15)
    assert high_disclosure.passes_gate() is False


# -------------------------- score_b3 composer -------------------------- #


def test_score_b3_composes_sub_metrics():
    cue = {"valence": -0.7, "arousal": 0.8}
    far = {"valence": 0.7, "arousal": -0.8}
    hit = _recall([_frag(cue, t=1), _frag(cue, t=2)])
    mixed = _recall([_frag(cue, t=3), _frag(far, t=4)])
    scores = score_b3(
        recalls_with_cue_affect=[(hit, cue), (mixed, cue)],
        recalls_with_over_disclosure_signal=[(hit, False), (mixed, False)],
    )
    assert scores.resonance_recall == 1.0  # both have ≥1 match
    assert scores.affect_match_at_k == 0.75  # (1.0 + 0.5) / 2
    assert scores.over_disclosure_rate == 0.0
    assert scores.match_radius == 0.4
    assert scores.top_k == 3
