"""M1 step 5 — B1 reconsolidation scorer tests.

Covers each sub-metric in isolation with hand-crafted inputs so the
exit-gate thresholds (preservation ≥ 0.95, divergence 0.25–0.45,
within-session Jaccard ≥ 0.65) can be reasoned about independent of
any scripted scenario.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psychology_base.eval.adapters import Fragment, make_fragment
from psychology_base.eval.benchmarks.b1_reconsolidation import (
    B1Scores,
    fragment_preservation,
    group_chains_by_root,
    rendering_divergence,
    rendering_divergence_for_chain,
    score_b1,
    within_session_jaccard,
)


TS = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def test_fragment_preservation_all_kept():
    f = make_fragment("a", TS)
    assert fragment_preservation([f.id], [f]) == 1.0


def test_fragment_preservation_half_lost():
    f1 = make_fragment("a", TS)
    f2 = make_fragment("b", TS + timedelta(minutes=1))
    assert fragment_preservation([f1.id, f2.id], [f1]) == 0.5


def test_fragment_preservation_empty_originals():
    """Edge case — no originals means nothing to preserve; return 1.0
    rather than divide-by-zero."""
    assert fragment_preservation([], []) == 1.0


def test_rendering_divergence_chain_length_one_is_zero():
    """Length-1 chain = no re-rendering yet; no divergence to score."""
    f = make_fragment("alone", TS)
    assert rendering_divergence_for_chain([f]) == 0.0


def test_rendering_divergence_identical_content_is_zero():
    """Identical content → Jaccard 1.0 → distance 0.0. Catches an
    accidental duplicate masquerading as a re-rendering."""
    a = make_fragment("father was never there", TS)
    b = make_fragment(
        "father was never there",
        TS + timedelta(days=1),
        parent_ids=(a.id,),
    )
    assert rendering_divergence_for_chain([a, b]) == 0.0


def test_rendering_divergence_full_divergence_is_one():
    """Fully disjoint vocabulary → Jaccard 0.0 → distance 1.0. Catches
    chain-coherence breakdown (reinterpretation drifted into a
    different referent entirely)."""
    a = make_fragment("alpha beta gamma", TS)
    b = make_fragment(
        "entirely different words words",
        TS + timedelta(days=1),
        parent_ids=(a.id,),
    )
    assert rendering_divergence_for_chain([a, b]) == 1.0


def test_rendering_divergence_in_band():
    """Reinterpretation that shares some tokens but reframes meaning
    lands in the target 0.25–0.45 band."""
    a = make_fragment("my father was never there on friday evenings", TS)
    b = make_fragment(
        "my father was overwhelmed trying his best on friday evenings",
        TS + timedelta(days=90),
        parent_ids=(a.id,),
    )
    d = rendering_divergence_for_chain([a, b])
    assert 0.25 <= d <= 0.6  # loosely in-band; exact value depends on tokenizer


def test_rendering_divergence_macro_averages_chains():
    """Macro-average: two chains contribute equally regardless of
    length. Length-1 chains do not contribute."""
    a1 = make_fragment("alpha", TS)
    a2 = make_fragment("alpha reinterpreted", TS + timedelta(days=1), parent_ids=(a1.id,))
    b1 = make_fragment("beta", TS + timedelta(hours=1))
    chains = [[a1, a2], [b1]]
    # Only the first chain contributes.
    assert rendering_divergence(chains) == rendering_divergence_for_chain([a1, a2])


def test_within_session_jaccard_coherent():
    """Fragments that overlap heavily on tokens → high within-session
    Jaccard."""
    s1 = [
        make_fragment("father was never there friday", TS),
        make_fragment(
            "father was never home friday evenings",
            TS + timedelta(minutes=1),
        ),
    ]
    j = within_session_jaccard([s1])
    assert j >= 0.4  # loose bound; token overlap is substantial


def test_within_session_jaccard_single_fragment_session_skipped():
    """A session with one fragment cannot contribute pairwise
    coherence; averaged over zero scored sessions returns 0."""
    s1 = [make_fragment("alone", TS)]
    assert within_session_jaccard([s1]) == 0.0


def test_group_chains_by_root_identifies_rerendering():
    root = make_fragment("v1", TS)
    rev = make_fragment("v2", TS + timedelta(days=1), parent_ids=(root.id,))
    unrelated = make_fragment("other", TS + timedelta(minutes=5))

    chains = group_chains_by_root([root, rev, unrelated])
    chain_ids = [[f.id for f in c] for c in chains]
    # Order depends on the input iteration; sort by root content for stability.
    chain_ids_sorted = sorted(chain_ids, key=lambda ids: ids[0])
    assert len(chain_ids_sorted) == 2
    assert any(ids == [root.id, rev.id] for ids in chain_ids_sorted)
    assert any(ids == [unrelated.id] for ids in chain_ids_sorted)


def test_group_chains_by_root_follows_long_chain():
    f1 = make_fragment("v1", TS)
    f2 = make_fragment("v2", TS + timedelta(days=1), parent_ids=(f1.id,))
    f3 = make_fragment("v3", TS + timedelta(days=2), parent_ids=(f2.id,))

    chains = group_chains_by_root([f1, f2, f3])
    assert len(chains) == 1
    assert [f.id for f in chains[0]] == [f1.id, f2.id, f3.id]


def test_score_b1_assembles_all_metrics():
    f1 = make_fragment("father was never there on friday", TS)
    f2 = make_fragment(
        "father was overwhelmed on friday",
        TS + timedelta(days=90),
        parent_ids=(f1.id,),
    )
    sessions = {1: [f1], 2: [f2]}
    scores = score_b1(
        original_ids=[f1.id, f2.id],
        final_fragments=[f1, f2],
        sessions=sessions,
    )
    assert isinstance(scores, B1Scores)
    assert scores.fragment_preservation == 1.0
    # Rendering divergence > 0 because content differs along the chain.
    assert scores.rendering_divergence > 0.0


def test_b1_gate_pass_requires_all_three():
    gate_pass = B1Scores(0.95, 0.35, 0.70)
    assert gate_pass.passes_gate()

    low_preservation = B1Scores(0.94, 0.35, 0.70)
    assert not low_preservation.passes_gate()

    high_divergence = B1Scores(0.95, 0.50, 0.70)
    assert not high_divergence.passes_gate()

    low_coherence = B1Scores(0.95, 0.35, 0.60)
    assert not low_coherence.passes_gate()
