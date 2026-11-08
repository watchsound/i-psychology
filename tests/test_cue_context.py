"""Frozen contracts for the one-hop candidate; not application-quality evidence."""
from __future__ import annotations

import copy
from datetime import datetime, timezone

import pytest

from psychology_base.companion import reconstruction as memory
from psychology_base.companion import turn_provenance as turn
from psychology_base.companion.pam_export import pam_export, pam_import
from psychology_base.eval.adapters import make_fragment


STAMP = datetime(2026, 10, 18, tzinfo=timezone.utc)


def fragment(position, text, *, owner="alice", session="one", group="sequence", tags=(), parents=()):
    return make_fragment(text, STAMP, tags=(f"user:{owner}", f"session:{session}",
        "subject:alice", "entity:memory", turn.sequence_tag("source-pin", group, position), *tags),
        parent_ids=parents)


def select(fragments, cue, scores):
    snapshot = memory.recall_tokens(fragments, cue, "unicode")
    return memory.cue_context_scores(fragments, scores, snapshot)


@pytest.mark.parametrize("anchor_position", [0, 2])
def test_support_is_symmetric_nonrecursive_and_conserves_mass(anchor_position):
    fs = [fragment(i, "anchor" if i == anchor_position else f"other {i}") for i in range(4)]
    scores = {f.id: (0.8 if i == anchor_position else 0.02) for i, f in enumerate(fs)}
    before = copy.deepcopy(scores)
    result = select(fs, "anchor", scores)
    assert result[fs[1].id] > scores[fs[1].id]
    if anchor_position == 0:
        assert result[fs[2].id] < scores[fs[2].id]  # scaled only, no two-hop support
    assert sum(result.values()) == pytest.approx(sum(scores.values()))
    assert all(value >= 0 for value in result.values()) and scores == before


def test_two_neighbors_do_not_accumulate_and_six_stronger_facts_keep_their_slots():
    fs = [fragment(0, "anchor"), fragment(1, "unrelated successor"), fragment(2, "anchor")]
    fs += [fragment(0, f"anchor direct {i}", group=f"direct{i}") for i in range(6)]
    scores = {f.id: (0.001 if i == 1 else 0.2) for i, f in enumerate(fs)}
    result = select(fs, "anchor", scores)
    # 8 matching anchors: single support=.0625, even with two neighbors.
    ratio = result[fs[0].id] / scores[fs[0].id]
    assert result[fs[1].id] / ratio == pytest.approx(0.0625)
    assert fs[1].id not in sorted(result, key=result.get, reverse=True)[:6]


@pytest.mark.parametrize("change", [
    {"owner": "bob"}, {"session": "two"}, {"group": "elsewhere"},
    {"tags": ("subject:bob",)}, {"tags": ("witness:other",)},
    {"parents": ("parent",)},
])
def test_existing_provenance_boundaries_block_context_support(change):
    fs = [fragment(0, "anchor"), fragment(1, "answer", **change)]
    scores = {fs[0].id: 1.0, fs[1].id: 0.01}
    assert select(fs, "anchor", scores) == scores


def test_gap_legacy_opaque_and_ambiguous_positions_do_not_invent_relationships():
    anchor, gap = fragment(0, "anchor"), fragment(2, "gap")
    legacy = make_fragment("legacy", STAMP, tags=("user:alice", "session:one"))
    opaque = make_fragment("opaque", STAMP, tags=(*legacy.tags, "pb-turn:v2:unknown"))
    conflict_a, conflict_b = fragment(1, "one"), fragment(1, "conflict")
    fs = [anchor, gap, legacy, opaque, conflict_a, conflict_b]
    scores = {f.id: (1.0 if f == anchor else 0.01) for f in fs}
    assert select(fs, "anchor", scores) == scores


def test_no_match_preserves_continuation_and_restore_preserves_relationships():
    fs = [fragment(0, "anchor"), fragment(1, "fact")]
    scores = {fs[0].id: 1.0, fs[1].id: 0.01}
    assert select(fs, "continue", scores) is scores
    restored = list(reversed(pam_import(pam_export(fs))))
    assert select(restored, "anchor", scores) == select(fs, "anchor", scores)


def test_candidate_uses_actual_selected_pool_but_writes_raw_activation():
    from psychology_base.eval.benchmarks import cue_context as study

    fs = [fragment(0, "anchor"), fragment(1, "context fact")]
    fs += [fragment(0, f"distractor {i}", group=f"other{i}") for i in range(6)]
    ordinary_state, candidate_state = memory.ActivationState(), memory.ActivationState()
    ordinary = memory.reconstruct(fs, "anchor", None, ordinary_state, 0, hebbian_delta=0)
    candidate = study.reconstruction()(fs, "anchor", None, candidate_state, 0, hebbian_delta=0)
    assert candidate.activation_profile == ordinary.activation_profile
    assert candidate_state == ordinary_state
    assert candidate.selection_profile != candidate.activation_profile
    assert fs[1] in candidate.fragments_used and len(candidate.fragments_used) <= 6
    assert {f.id for f in candidate.fragments_used} <= {f.id for f in fs}


@pytest.mark.parametrize("descriptor", ["pb-turn:v1:{", "pb-turn:v2:future"])
def test_invalid_or_multiple_descriptors_do_not_support_neighbors(descriptor):
    fs = [fragment(0, "anchor"), fragment(1, "fact", tags=(descriptor,))]
    scores = {fs[0].id: 1.0, fs[1].id: 0.01}
    assert select(fs, "anchor", scores) is scores


def test_empty_zero_and_already_supported_profiles_are_exact_identity():
    assert select([], "anchor", {}) == {}
    fs = [fragment(0, "anchor"), fragment(1, "fact")]
    for scores in ({f.id: 0.0 for f in fs}, {f.id: 1.0 for f in fs}):
        assert select(fs, "anchor", scores) is scores


def test_companion_and_independent_witness_use_one_hook_without_global_patch():
    from psychology_base.eval.benchmarks import cue_context as study

    system = study.system_for_arm("cue_context", cjk_tokenizer="unicode")
    call_globals = system.recall.__func__.__globals__
    bound = call_globals["reconstruct"]
    witness_bound = call_globals["witness_couple"].__globals__["witness_second_perspective"].__globals__["reconstruct"]
    assert bound is witness_bound
    assert bound.__globals__["_selection_scores"] is memory.cue_context_scores
    assert memory.reconstruct.__globals__["_selection_scores"] is memory._selection_scores
    assert system.activation is not system.witness_state.activation and system.edges is not system.witness_edges
    fs = [fragment(0, "anchor"), fragment(1, "context fact")]
    for f in fs:
        system._ingest_fragment(f)
    witness_before = copy.deepcopy(system.witness_edges)
    edge_before = system.edges.get(fs[0].id, fs[1].id)
    result = system.recall("anchor", scope=("alice", "memory"))
    assert {f.id for f in result.fragments_used} == {f.id for f in fs}
    assert system.edges.get(fs[0].id, fs[1].id) > edge_before
    assert system.witness_edges == witness_before
    assert system.activation.recall_count == system.witness_state.activation.recall_count == 1


def test_filtered_out_anchor_cannot_support_an_eligible_neighbor():
    from psychology_base.companion.perspective.routing import PerspectiveSpec, ScopeSpec
    from psychology_base.eval.benchmarks import cue_context as study

    system = study.system_for_arm("cue_context", cjk_tokenizer="unicode")
    anchor = fragment(0, "anchor")
    neighbor = fragment(1, "fact", tags=("witness:private",))
    system._ingest_fragment(anchor)
    system._ingest_fragment(neighbor)
    result = system.recall("anchor", perspective=PerspectiveSpec(scope=ScopeSpec("bob", "memory")))
    assert result.fragments_used == ()
    assert system.activation.residual == {} and system.witness_state.activation.residual == {}


def test_duplicate_evidence_still_spends_one_selection_slot():
    from psychology_base.eval.benchmarks import cue_context as study

    fs = [fragment(0, "anchor"), fragment(1, "same fact"),
          fragment(0, "anchor", session="two"), fragment(1, "same fact", session="two")]
    result = study.reconstruction()(fs, "anchor", None, memory.ActivationState(), 0)
    assert len(result.fragments_used) == 2
    assert {f.content for f in result.fragments_used} == {"anchor", "same fact"}


def test_token_snapshot_is_shared_and_idf_research_owner_is_preserved(monkeypatch):
    from psychology_base.eval.benchmarks import chinese_retrieval as shared
    from psychology_base.eval.benchmarks import recall_carryover as carry
    from psychology_base.eval.benchmarks import cue_context as study

    fs = [fragment(0, "anchor rare"), fragment(1, "anchor common"), fragment(2, "common")]
    snapshots = []
    original = memory.recall_tokens

    def record(*args):
        snapshot = original(*args)
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(memory, "recall_tokens", record)
    result = study.reconstruction()(fs, "rare common", None, memory.ActivationState(), 0)
    assert len(snapshots) == 1 and result.selection_profile
    spread = shared._bind(memory.spread_activation, _cue_activation=carry.idf_weights)
    reconstruct = shared._bind(memory.reconstruct, spread_activation=spread)
    actual = reconstruct(fs, "rare common", None, memory.ActivationState(), 0)
    raw = spread(fs, "rare common", memory.ActivationState())
    assert actual.activation_profile == {f.id: raw.get(f.id, 0.0) for f in fs}
    assert actual.activation_profile != result.activation_profile
