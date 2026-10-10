"""M1 step 8 — B1 childhood-event scripted scenario integration test.

Plays [benchmarks/scripts/b1_childhood_event.yaml](../../benchmarks/scripts/b1_childhood_event.yaml)
through the CompanionSystem and the B1 scorer. Validates:

1. The scenario loads under the extended schema (RerenderTurn + label).
2. The player drives 1 anchor + 8 rerenderings + 10 ambient observes
   through observe/rerender, chaining via parent_label.
3. The Merkle chain rooted at ``anchor`` has 9 nodes (anchor + 8 reinterpretations).
4. Fragment preservation = 1.0 (append-only + no eviction configured).
5. Rendering divergence lands > 0.0 (chain is not a trivial duplicate chain).
6. Within-session Jaccard > 0.0 (ambient + thematic turns share tokens).
7. The YAML expect-block (fragment_count_min, chain_min_length) is honoured.
"""

from __future__ import annotations

from pathlib import Path

from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.b1_reconsolidation import (
    group_chains_by_root,
    score_b1,
)
from psychology_base.eval.session_player import SessionPlayer, load_scenario


SCENARIO_PATH = (
    Path(__file__).parent.parent / "benchmarks" / "scripts" / "b1_childhood_event.yaml"
)


def _play():
    scenario = load_scenario(SCENARIO_PATH)
    # byte_cap=None disables S1 eviction. B1 scores reconsolidation as
    # append-only-by-construction; running under the default 4096-byte
    # cap would conflate M1 preservation with S1 capacity-pressure
    # eviction, which is a separate mechanism (Principle D selection
    # pressure) measured by the V-track, not by B1.
    sys = CompanionSystem(byte_cap=None)
    player = SessionPlayer(sys)
    player.play(scenario)
    return sys, player


def test_scenario_loads_with_rerender_turns():
    scenario = load_scenario(SCENARIO_PATH)
    assert scenario.benchmark == "B1"
    assert len(scenario.sessions) == 10


def test_player_drives_anchor_and_eight_rerenderings():
    sys, player = _play()
    # 1 anchor + 8 chained rerenderings + 9 ambient observes = 18.
    assert len(sys.companion.store.all()) == 18
    # Rerenderings chain linearly: each s_{n+1}_rev → s_n_rev → … → anchor.
    # So anchor has exactly ONE child (s2_rev) and every rerendering (except
    # the leaf s9_rev) has exactly one child.
    anchor_id = player.labels["anchor"]
    direct_children = [
        f for f in sys.companion.store.all() if f.parent_ids == (anchor_id,)
    ]
    assert len(direct_children) == 1
    # Count all rerenderings regardless of parent: 8 chained rerenderings.
    all_rerenderings = [
        f for f in sys.companion.store.all() if f.parent_ids != ()
    ]
    assert len(all_rerenderings) == 8


def test_chain_has_nine_nodes():
    sys, player = _play()
    chains = group_chains_by_root(sys.companion.store.all())
    # The anchor chain is the longest; follow the first-parent-chain rule.
    anchor_chain = next(
        c for c in chains if c[0].id == player.labels["anchor"]
    )
    assert len(anchor_chain) == 9


def test_b1_scorer_passes_on_scenario():
    sys, player = _play()
    scores = score_b1(
        original_ids=player.original_ids,
        final_fragments=sys.companion.store.all(),
        sessions=player.sessions,
    )
    # Append-only + no eviction → 1.0.
    assert scores.fragment_preservation == 1.0
    # Chain has real content divergence (reframings, not duplicates).
    assert scores.rendering_divergence > 0.0
    # Thematic + ambient turns share tokens (father, friday, waited, etc.).
    assert scores.within_session_jaccard > 0.0


def test_expect_block_is_honoured():
    """The YAML ``expect`` on the s10 probe asserts fragment_count_min
    = 18 and chain_min_length = 9. Round-trip them against the actual
    final state."""
    sys, player = _play()
    assert len(sys.companion.store.all()) >= 18
    anchor_chain = next(
        c
        for c in group_chains_by_root(sys.companion.store.all())
        if c[0].id == player.labels["anchor"]
    )
    assert len(anchor_chain) >= 9
