"""M3 unit tests — PAD-2 affect-resonance primitives + retrieval boost.

Covers:

1. ``pair_affect_score`` returns 1.0 on identical PAD states, 0.0 at
   the max-distance corner, and 0.0 when either side lacks numeric
   valence+arousal ("no boost" rather than penalty).
2. ``cue_affect_from_context`` extracts a PAD dict from
   ``context={"affect": {...}}`` and returns None otherwise. Scalar
   clamping is applied on the way out.
3. ``reconstruct(affect_weight=0.0)`` is byte-identical to the
   pre-M3 scoring (regression guard — M1+M2 must still work when
   M3 is off).
4. ``reconstruct(affect_weight>0, cue_affect=...)`` boosts fragments
   whose stored affect matches the cue, flipping the top ranking
   when semantic Jaccard alone can't separate two equally-matching
   fragments.
5. Scope + affect compose: scoped pool pre-filters, then affect
   boost re-ranks inside the scoped pool.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from psychology_base.companion.affect_resonance import (
    cue_affect_from_context,
    pair_affect_score,
)
from psychology_base.companion.reconstruction import ActivationState, reconstruct
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import (
    Observation,
    make_fragment,
)


def _dt(seconds: int = 0) -> datetime:
    return datetime(2026, 1, 1, 12, 0, seconds, tzinfo=timezone.utc)


# ------------------------- pair_affect_score ------------------------- #


def test_pair_affect_score_identical_states_returns_one():
    pad = {"valence": 0.3, "arousal": -0.2}
    assert pair_affect_score(pad, pad) == 1.0


def test_pair_affect_score_max_distance_returns_zero():
    # Corners (-1, -1) and (1, 1): euclidean = sqrt(8) → score = 0.
    low = {"valence": -1.0, "arousal": -1.0}
    high = {"valence": 1.0, "arousal": 1.0}
    assert pair_affect_score(low, high) == 0.0


def test_pair_affect_score_scales_linearly_with_distance():
    # Half-way: euclidean = sqrt(2); score = 1 - sqrt(2)/sqrt(8) = 0.5.
    a = {"valence": 0.0, "arousal": 0.0}
    b = {"valence": 1.0, "arousal": 1.0}
    assert math.isclose(pair_affect_score(a, b), 0.5, abs_tol=1e-6)


def test_pair_affect_score_missing_valence_arousal_returns_zero():
    # Missing-affect yields 0.0 (no boost). Not a penalty: downstream
    # retrieval math treats 0 pair-score as "add nothing", so M1/M2
    # fragments without affect are not demoted when M3 lights up.
    assert pair_affect_score({}, {"valence": 0.0, "arousal": 0.0}) == 0.0
    assert pair_affect_score({"valence": 0.0, "arousal": 0.0}, {}) == 0.0
    assert pair_affect_score({"valence": 0.0}, {"arousal": 0.0}) == 0.0


def test_pair_affect_score_non_numeric_affect_returns_zero():
    bad = {"valence": "high", "arousal": "low"}
    good = {"valence": 0.5, "arousal": 0.5}
    assert pair_affect_score(bad, good) == 0.0


def test_pair_affect_score_clamps_out_of_range_values():
    # valence=10 clamps to 1.0 — pair with valence=1.0 (same arousal)
    # must therefore score 1.0, not drop to 0 or negative.
    runaway = {"valence": 10.0, "arousal": 0.3}
    clamped = {"valence": 1.0, "arousal": 0.3}
    assert pair_affect_score(runaway, clamped) == 1.0


# ------------------- cue_affect_from_context ------------------------- #


def test_cue_affect_from_context_pulls_affect_dict():
    out = cue_affect_from_context({"affect": {"valence": 0.5, "arousal": -0.3}})
    assert out == {"valence": 0.5, "arousal": -0.3}


def test_cue_affect_from_context_returns_none_on_empty():
    assert cue_affect_from_context(None) is None
    assert cue_affect_from_context({}) is None
    assert cue_affect_from_context({"other": "thing"}) is None


def test_cue_affect_from_context_returns_none_on_bad_shape():
    assert cue_affect_from_context({"affect": "not-a-dict"}) is None
    assert cue_affect_from_context({"affect": {"valence": "high"}}) is None
    assert cue_affect_from_context({"affect": {"valence": 0.5}}) is None


def test_cue_affect_from_context_clamps_out_of_range():
    out = cue_affect_from_context({"affect": {"valence": 5.0, "arousal": -3.0}})
    assert out == {"valence": 1.0, "arousal": -1.0}


# ------------------ reconstruct(affect_weight) ----------------------- #


def _frag(content: str, affect: dict, t: int = 0):
    return make_fragment(
        content=content,
        timestamp=_dt(t),
        affect=affect,
    )


def test_reconstruct_affect_weight_zero_is_m1_m2_byte_identical():
    # Two fragments share a cue token so Jaccard overlap is equal;
    # with affect_weight=0 and cue_affect=None, ranking must be
    # independent of affect — the deterministic tiebreak (id) decides.
    a = _frag("I am overwhelmed", {"valence": -0.7, "arousal": 0.8}, t=1)
    b = _frag("I am overwhelmed", {"valence": 0.5, "arousal": -0.3}, t=2)
    state_off = ActivationState()
    state_on = ActivationState()
    off = reconstruct([a, b], "overwhelmed", None, state_off, 0)
    on = reconstruct(
        [a, b], "overwhelmed", None, state_on, 0,
        cue_affect={"valence": -0.7, "arousal": 0.8},
        affect_weight=0.0,
    )
    # affect_weight=0 means the M3 branch is a no-op even with a
    # cue_affect attached — ranking is identical.
    assert [f.id for f in off.fragments_used] == [f.id for f in on.fragments_used]


def test_reconstruct_affect_boost_flips_tie_when_cue_matches_only_one():
    # Both fragments have identical cue tokens so pure Jaccard rank is
    # a tie. Affect boost should promote the fragment whose stored
    # affect matches cue_affect.
    neg = _frag(
        "I feel overwhelmed and panicked",
        {"valence": -0.7, "arousal": 0.8},
        t=1,
    )
    pos = _frag(
        "I feel overwhelmed and panicked",
        {"valence": 0.6, "arousal": -0.3},
        t=2,
    )
    state = ActivationState()
    boosted = reconstruct(
        [neg, pos], "overwhelmed", None, state, 0,
        cue_affect={"valence": -0.75, "arousal": 0.8},
        affect_weight=1.0,
    )
    # The negative-affect fragment should outrank the positive one
    # under affect boost even though pure Jaccard is tied.
    assert boosted.fragments_used[0].id == neg.id


def test_reconstruct_affect_boost_rescues_zero_relevance_fragment():
    """Thin-cue + strong-affect case — the specific scenario M3 is
    designed for. A fragment with zero cue-token overlap AND zero
    edge-propagation reach (no shared tokens with any other fragment)
    should still surface when its stored affect matches cue_affect,
    because the affect term is ADDITIVE, not multiplicative. A
    multiplicative boost cannot rescue a zero-relevance score — which
    was the pre-fix behavior and is wrong for the "I'm spiraling but
    about a different topic than any prior fragment" case the M3
    README names as the target shape.
    """
    hit = _frag(
        "wholly disjoint content tokens here",
        {"valence": -0.7, "arousal": 0.8},
        t=1,
    )
    miss = _frag(
        "entirely separate words elsewhere",
        {"valence": 0.6, "arousal": -0.3},
        t=2,
    )
    state = ActivationState()
    out = reconstruct(
        [hit, miss], "xylophone zebra", None, state, 0,
        cue_affect={"valence": -0.7, "arousal": 0.8},
        affect_weight=1.0,
    )
    # Pool must not be empty — the affect-matched fragment must survive
    # the zero-score filter.
    assert len(out.fragments_used) >= 1
    assert out.fragments_used[0].id == hit.id


def test_reconstruct_zero_score_filter_keeps_residual_only_fragment():
    """Principle A — a fragment with zero cue overlap in the CURRENT
    recall but carrying residual activation from a PRIOR recall must
    survive the post-additive ``score > 0`` filter. The residual is
    what makes memory a living process; filtering it out would turn
    the recall into a stateless lookup.
    """
    primed = _frag(
        "a b c",  # unique tokens, no second-call cue overlap
        {"valence": -0.7, "arousal": 0.8},
        t=1,
    )
    state = ActivationState()
    # First recall: cue overlaps "a" → primed gets residual.
    reconstruct([primed], "a", None, state, 0)
    assert state.get(primed.id) > 0.0  # residual seeded
    # Second recall: cue "zzz" has no overlap with primed. Without
    # residual, raw = 0 and score = 0 → filtered. With residual
    # (via spread_activation seeding from state.residual), raw > 0
    # and the fragment survives — Principle A intact.
    out = reconstruct([primed], "zzz", None, state, 0)
    assert len(out.fragments_used) == 1
    assert out.fragments_used[0].id == primed.id


def test_reconstruct_zero_score_filter_drops_truly_signal_less_fragment():
    """Retirement evidence for the ``score > 0`` filter under additive:
    a fragment with NO cue overlap, NO residual (fresh state), NO
    edge reach (singleton pool), AND NO affect match is correctly
    dropped. The filter is tight — it drops only fragments with
    nothing retrievable about them. All other dimensions (cue, residual,
    edges, affect) are each tested elsewhere to admit survivors.
    """
    dead_weight = _frag(
        "wholly disjoint tokens here",
        {},  # missing affect → pair_affect_score = 0
        t=1,
    )
    state = ActivationState()  # fresh — no residual
    out = reconstruct(
        [dead_weight], "entirely unrelated cue", None, state, 0,
        cue_affect={"valence": -0.7, "arousal": 0.8},
        affect_weight=1.0,
    )
    assert out.fragments_used == ()


def test_reconstruct_missing_affect_gets_no_boost_no_penalty():
    """Fragments without numeric affect participate at pair_score=0.
    They should neither be lifted nor demoted — the M1/M2 relevance
    signal alone decides their rank, which is the compatibility
    guarantee from the M3 README."""
    with_affect = _frag(
        "overwhelmed and panicked today",
        {"valence": -0.7, "arousal": 0.8},
        t=1,
    )
    no_affect = _frag(
        "overwhelmed and panicked today also",
        {},
        t=2,
    )
    state = ActivationState()
    out = reconstruct(
        [with_affect, no_affect], "overwhelmed", None, state, 0,
        cue_affect={"valence": -0.7, "arousal": 0.8},
        affect_weight=1.0,
    )
    # Both should still appear in the pool — missing-affect must not
    # drop no_affect below the ranked > 0.0 threshold.
    assert len(out.fragments_used) == 2


# -------- CompanionSystem recall(affect_weight=...) --------------------- #


def test_companion_system_default_affect_weight_zero_is_backcompat():
    # Default construction stores affect_weight=0 so pre-M3 scenarios
    # behave byte-identically.
    sys = CompanionSystem(byte_cap=None, witness_active=False)
    assert sys.affect_weight == 0.0


def test_companion_recall_affect_boost_prefers_matched_affect():
    sys = CompanionSystem(
        byte_cap=None,
        witness_active=False,
        affect_weight=1.0,
    )
    sys.observe(
        Observation(
            timestamp=_dt(1),
            speaker="user",
            content="I feel overwhelmed and I cannot sleep",
            affect={"valence": -0.7, "arousal": 0.8},
            subject_id="user",
            entity_id="user",
        )
    )
    sys.observe(
        Observation(
            timestamp=_dt(2),
            speaker="user",
            content="I feel overwhelmed by this quiet weekend",
            affect={"valence": 0.5, "arousal": -0.3},
            subject_id="user",
            entity_id="user",
        )
    )
    rec = sys.recall(
        "overwhelmed",
        {"affect": {"valence": -0.7, "arousal": 0.8}},
        scope=("user", "user"),
    )
    # The negative-arousal fragment should rank first under the affect
    # boost even though both share the "overwhelmed" cue.
    assert rec.fragments_used
    top_affect = rec.fragments_used[0].affect
    assert top_affect.get("valence", 0.0) < 0.0


def test_companion_recall_affect_weight_per_call_overrides_default():
    sys = CompanionSystem(
        byte_cap=None,
        witness_active=False,
        affect_weight=0.0,  # system default off
    )
    sys.observe(
        Observation(
            timestamp=_dt(1),
            speaker="user",
            content="I feel overwhelmed and I cannot sleep",
            affect={"valence": -0.7, "arousal": 0.8},
            subject_id="user",
            entity_id="user",
        )
    )
    sys.observe(
        Observation(
            timestamp=_dt(2),
            speaker="user",
            content="I feel overwhelmed by this quiet weekend",
            affect={"valence": 0.5, "arousal": -0.3},
            subject_id="user",
            entity_id="user",
        )
    )
    rec = sys.recall(
        "overwhelmed",
        {"affect": {"valence": -0.7, "arousal": 0.8}},
        scope=("user", "user"),
        affect_weight=2.0,
    )
    assert rec.metadata["affect_weight"] == 2.0
    assert rec.metadata["cue_affect"] == {"valence": -0.7, "arousal": 0.8}


def test_companion_recall_without_cue_affect_leaves_scoring_untouched():
    """``affect_weight > 0`` with no cue_affect must behave like
    ``affect_weight=0``: the retrieval math gets no multiplier (the
    README's "default off" compatibility clause when the application
    layer forgets to attach cue affect)."""
    sys = CompanionSystem(
        byte_cap=None,
        witness_active=False,
        affect_weight=1.0,
    )
    sys.observe(
        Observation(
            timestamp=_dt(1),
            speaker="user",
            content="overwhelmed negative",
            affect={"valence": -0.7, "arousal": 0.8},
            subject_id="user",
            entity_id="user",
        )
    )
    sys.observe(
        Observation(
            timestamp=_dt(2),
            speaker="user",
            content="overwhelmed positive",
            affect={"valence": 0.5, "arousal": -0.3},
            subject_id="user",
            entity_id="user",
        )
    )
    rec = sys.recall("overwhelmed", scope=("user", "user"))
    # No context → cue_affect=None → the M3 branch inside reconstruct
    # must stay off. Both fragments present; order driven by non-affect
    # signal.
    assert rec.metadata["cue_affect"] is None
    assert len(rec.fragments_used) == 2
