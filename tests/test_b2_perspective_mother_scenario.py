"""M2 step 8 — B2 perspective-mother scripted scenario integration test.

Plays [benchmarks/scripts/b2_perspective_mother.yaml](../../benchmarks/scripts/b2_perspective_mother.yaml)
through the CompanionSystem with scope-filtered recall. Validates:

1. The scenario loads under the M2-extended schema (subject/entity on
   ObserveTurn, scope on RecallTurn).
2. The player drives all observe turns, writing ``subject:<id>`` /
   ``entity:<id>`` tags onto each fragment.
3. The ``(user, mother)`` scope filter returns ONLY the 10 mother-
   referent fragments (no self-model, no AI observations), so the
   perspective-taking retrieval is actually filtered.
4. The ``(user, user)`` and ``(ai, user)`` scopes return disjoint
   pools covering the self-model-gap sub-scenario.
5. B2 attribute reconstruction clears ≥ 10/12 under the scope-filtered
   recall content.
6. B2 gap detection identifies the planted gap (user says "not a math
   person"; AI observes first-try solves).
"""

from __future__ import annotations

from pathlib import Path

from psychology_base.companion.system import (
    CompanionSystem,
    _filter_fragments_by_scope,
)
from psychology_base.eval.benchmarks.b2_perspective_taking import (
    AttributeSpec,
    attribute_reconstruction,
    score_user_model_gap,
)
from psychology_base.eval.session_player import SessionPlayer, load_scenario


SCENARIO_PATH = (
    Path(__file__).parent.parent
    / "benchmarks"
    / "scripts"
    / "b2_perspective_mother.yaml"
)


MOTHER_ATTRIBUTES: tuple[AttributeSpec, ...] = (
    AttributeSpec("money-security", ("village", "cash", "mattress")),
    AttributeSpec("gas-bill-worry", ("gas bill",)),
    AttributeSpec("hyper-vigilance", ("thermometer", "hospital")),
    AttributeSpec("refrigerator-inspection", ("refrigerator", "expiration")),
    AttributeSpec("directness-as-love", ("directness", "marry")),
    AttributeSpec("oldest-sibling-caretaker", ("oldest", "six", "night shift")),
    AttributeSpec("private-language-of-care", ("hundred-dollar", "eat something warm")),
    AttributeSpec("no-vacation", ("vacation", "22 years")),
    AttributeSpec("superstition-four", ("number 4", "superstition", "death-adjacent")),
    AttributeSpec("language-barrier-blunt", ("English", "blunt", "first language")),
    AttributeSpec("sacrifice", ("letting the family down", "rest")),
    AttributeSpec("village-origin", ("village",)),
)

GAP_ATTRIBUTES: tuple[AttributeSpec, ...] = (
    AttributeSpec("math-first-try-solve", ("first try", "first-try", "correct")),
)


def _play():
    scenario = load_scenario(SCENARIO_PATH)
    # byte_cap=None disables S1 eviction so scope filter is measured
    # on the full fragment pool, not under capacity pressure (same
    # posture as B1 — a different metric would catch eviction behaviour).
    sys = CompanionSystem(byte_cap=None)
    player = SessionPlayer(sys)
    player.play(scenario)
    return sys, player


def test_scenario_loads_under_m2_schema():
    scenario = load_scenario(SCENARIO_PATH)
    assert scenario.benchmark == "B2"
    assert len(scenario.sessions) == 6


def test_observations_carry_perspective_tags():
    sys, _ = _play()
    tagged = [
        f for f in sys.companion.store.all()
        if any(t.startswith("subject:") for t in f.tags)
    ]
    # Every observe in the scenario has subject+entity → every
    # fragment is tagged.
    assert len(tagged) == len(sys.companion.store.all())
    assert all(
        any(t.startswith("entity:") for t in f.tags) for f in tagged
    )


def test_scope_filter_isolates_mother_pool():
    sys, _ = _play()
    pool = _filter_fragments_by_scope(
        sys.companion.store.all(), ("user", "mother")
    )
    # Scenario puts 2+2+2+2+2 = 10 user-about-mother observes across
    # s1-s5. Scope filter must return exactly those 10.
    assert len(pool) == 10
    for f in pool:
        assert "subject:user" in f.tags
        assert "entity:mother" in f.tags
        # No self-model turns leak in.
        assert "subject:ai" not in f.tags


def test_scope_filter_separates_self_and_ai_views():
    sys, _ = _play()
    user_self = _filter_fragments_by_scope(
        sys.companion.store.all(), ("user", "user")
    )
    ai_user = _filter_fragments_by_scope(
        sys.companion.store.all(), ("ai", "user")
    )
    # 4 "I'm not a math person" statements across s1,s2,s3,s5.
    assert len(user_self) == 4
    # 5 AI observations of first-try solve across s1-s5.
    assert len(ai_user) == 5
    assert set(f.id for f in user_self).isdisjoint(f.id for f in ai_user)


def test_attribute_reconstruction_clears_gate_on_direct_pool():
    """Deterministic attribute scoring against the raw scope-filtered
    pool. The downstream reconstructive-recall path is tested
    separately — this test isolates the attribute-recovery metric
    from reconstruction's top-k truncation."""
    sys, _ = _play()
    pool = _filter_fragments_by_scope(
        sys.companion.store.all(), ("user", "mother")
    )
    merged_content = "\n".join(f.content for f in pool)
    score = attribute_reconstruction(merged_content, MOTHER_ATTRIBUTES)
    assert score >= 10 / 12, (
        f"attribute_reconstruction={score:.3f} below gate 10/12; "
        f"scope filter dropped attributes or scenario tokens drifted"
    )


def test_user_model_gap_detects_math_identity_divergence():
    sys, _ = _play()
    user_self = _filter_fragments_by_scope(
        sys.companion.store.all(), ("user", "user")
    )
    ai_user = _filter_fragments_by_scope(
        sys.companion.store.all(), ("ai", "user")
    )
    gap = score_user_model_gap(
        user_self_fragments=user_self,
        ai_observations=ai_user,
        expected_attributes=GAP_ATTRIBUTES,
        expected_gap_names=["math-first-try-solve"],
    )
    assert "math-first-try-solve" in gap.predicted_gaps
    assert gap.precision >= 0.60
    assert gap.recall >= 0.50
