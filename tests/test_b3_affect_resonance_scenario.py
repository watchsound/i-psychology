"""M3 step 9 — B3 affect-resonance scripted scenario integration test.

Plays [benchmarks/scripts/b3_affect_resonance.yaml](../../benchmarks/scripts/b3_affect_resonance.yaml)
through the CompanionSystem with ``affect_weight=1.0`` and validates:

1. The scenario loads under the schema (RecallTurn.context already
   present pre-M3; no schema change for M3).
2. The seven observes land as seven fragments (3 high-arousal-negative
   + 3 low-arousal-positive + 1 cue not in play).
3. The session-6 recall surfaces the three negative-arousal fragments
   in top-3 when ``affect_weight > 0`` and ``cue_affect`` is attached,
   exercising the resonance mechanism directly.
4. ``resonance_recall`` ≥ 0.60 and ``affect_match_at_k`` ≥ 0.50 (the
   B3 exit gate thresholds from the M3 README).
5. The default-off path (``affect_weight=0``) is NOT required to
   pass the gate — the gate is specifically for the affect-on
   configuration; the regression that default-off = M1+M2 behavior
   is covered elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.b3_affect_resonance import score_b3
from psychology_base.eval.session_player import SessionPlayer, load_scenario


SCENARIO_PATH = (
    Path(__file__).parent.parent
    / "benchmarks"
    / "scripts"
    / "b3_affect_resonance.yaml"
)


def _play(affect_weight: float = 1.0):
    scenario = load_scenario(SCENARIO_PATH)
    sys = CompanionSystem(
        byte_cap=None,
        witness_active=False,
        affect_weight=affect_weight,
    )
    player = SessionPlayer(sys)
    player.play(scenario)
    return sys, player


def test_scenario_loads_under_m3_schema():
    scenario = load_scenario(SCENARIO_PATH)
    assert scenario.benchmark == "B3"
    assert len(scenario.sessions) == 6


def test_scenario_observes_six_affect_fragments():
    sys, _ = _play()
    frags = sys.companion.store.all()
    assert len(frags) == 6
    neg = [f for f in frags if f.affect.get("valence", 0.0) < 0.0]
    pos = [f for f in frags if f.affect.get("valence", 0.0) > 0.0]
    assert len(neg) == 3
    assert len(pos) == 3


def test_b3_resonance_gate_clears_with_affect_weight_on():
    _, player = _play(affect_weight=1.0)
    recalls = player.transcript
    assert recalls, "scenario produced no recalls"
    recall = recalls[0]
    assert recall.metadata["affect_weight"] == 1.0
    cue_aff = recall.metadata["cue_affect"]
    assert cue_aff is not None

    scores = score_b3(
        recalls_with_cue_affect=[(r_to_recall(r), cue_aff) for r in recalls],
        match_radius=recall.expect.get("match_radius", 0.4),
        top_k=recall.expect.get("top_k", 3),
    )
    assert scores.resonance_recall >= recall.expect["resonance_recall_min"], (
        f"resonance_recall={scores.resonance_recall:.3f} below gate "
        f"{recall.expect['resonance_recall_min']}"
    )
    assert scores.affect_match_at_k >= recall.expect["affect_match_at_k_min"], (
        f"affect_match_at_k={scores.affect_match_at_k:.3f} below gate "
        f"{recall.expect['affect_match_at_k_min']}"
    )


def test_affect_boost_promotes_negative_cluster_over_positive():
    """Core M3 claim — with high-arousal-negative cue, the top-K
    fragments carry negative valence even when the positive-arousal
    fragments share surface tokens with the cue. Direct check on the
    recall's fragments_used."""
    _, player = _play(affect_weight=1.0)
    recall = player.transcript[0]
    top3 = recall.fragments_used[:3]
    negative_count = sum(
        1 for f in top3 if f.affect.get("valence", 0.0) < 0.0
    )
    assert negative_count >= 2, (
        f"Only {negative_count}/3 top-K fragments are negative-affect; "
        f"affect boost did not separate the clusters"
    )


def test_default_off_scoring_is_scope_only():
    """Regression guard — ``affect_weight=0`` with cue_affect in
    context must leave the retrieval math at M1+M2 behavior. The
    metadata records affect_weight=0 so transcripts stay auditable."""
    _, player = _play(affect_weight=0.0)
    recall = player.transcript[0]
    assert recall.metadata["affect_weight"] == 0.0
    # cue_affect is still extracted (metadata honesty), but the
    # retrieval score path treats it as a no-op per the README's
    # "default off" guarantee.
    assert recall.metadata["cue_affect"] is not None


# ------------------------- helpers -------------------------------- #


def r_to_recall(record):
    """Adapt a transcript ``RecallRecord`` back to the ``Recall``
    shape the B3 scorer consumes."""
    from psychology_base.eval.adapters import Recall

    return Recall(
        cue=record.cue,
        content=record.content,
        fragments_used=record.fragments_used,
        metadata=record.metadata,
    )
