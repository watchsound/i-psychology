"""Prospective allocation controls; authored statements are software fixtures."""
import copy
from itertools import combinations
import math

import pytest

from psychology_base.applications.context import render_context_block
from psychology_base.companion import reconstruction as memory
from psychology_base.companion.edges import EdgeStore
from psychology_base.companion.system import CompanionSystem
from tests.test_cue_context import fragment

MODES = ("query_occurrences", "span_allocation", "query_span_allocation")


def render(fs):
    return render_context_block(fs, assistant_speaker="companion")


def select(fs, scores, cue, mode, limit=6, max_bytes=6000):
    return memory.ordered_selection(fs, scores, limit,
        memory.ResearchSelection(mode, render, max_bytes),
        snapshot=memory.recall_tokens(fs, cue, "unicode"))


def test_query_statistics_use_occurrence_document_frequency_and_ignore_unsupported_terms():
    rows = [("a", {"rare", "common"}), ("b", {"common"}), ("c", {"common"})]
    frequencies, weights = memory.query_information(rows, {"rare", "common", "missing"})
    assert dict(frequencies) == {"rare": 1, "common": 3}
    assert weights["common"] == 1
    assert weights["rare"] == 1 + math.log(4 / 2)
    assert "missing" not in weights


@pytest.mark.parametrize("mode", MODES)
def test_empty_and_zero_overlap_fall_back_without_inventing_context(mode):
    fs = [fragment(0, "unrelated"), fragment(1, "detail")]
    scores = {fs[0].id: .9}
    selected, trace = select(fs, scores, "absent", mode)
    assert selected == [(fs[0], .9)]
    assert trace["fallback"] == "no_query_support"
    assert select([], {}, "absent", mode)[0] == []


def test_distinct_query_coverage_protects_distant_singleton_from_repetitive_matches():
    fs = [fragment(0, "common", group=str(i)) for i in range(7)]
    distant = fragment(0, "personname", group="distant")
    fs.append(distant)
    scores = {f.id: 1 - i / 100 for i, f in enumerate(fs)}
    legacy, _ = memory.ordered_selection(fs, scores, 6, memory.ResearchSelection("occurrences", render))
    result, trace = select(fs, scores, "common personname", "query_occurrences")
    assert distant not in [f for f, _ in legacy]
    assert distant in [f for f, _ in result]
    assert trace["objective"][0] == sum(trace["query_weights"].values())


def test_query_only_matches_exhaustive_objective_on_small_inventory():
    fs = [fragment(0, text, group=str(i)) for i, text in enumerate([
        "red", "red blue", "green", "blue", "irrelevant"])]
    scores = {f.id: .5 + i / 20 for i, f in enumerate(fs)}
    result, trace = select(fs, scores, "red blue green", "query_occurrences", limit=2)
    weights = trace["query_weights"]
    def objective(group):
        words = set().union(*(set(memory.tokens(f.content)) for f in group))
        return (sum(weights[t] for t in sorted(words & weights.keys())),
                sum(scores[f.id] for f in fs if f in group))
    expected = max(objective(group) for k in range(3) for group in combinations(fs, k))
    assert objective([f for f, _ in result]) == expected


@pytest.mark.parametrize("mode", ("span_allocation", "query_span_allocation"))
def test_two_competing_three_turn_spans_retain_zero_score_outcomes(mode):
    fs = []
    for group, outcome in [("first", "accepted"), ("second", "rejected")]:
        fs.extend([fragment(0, "interview", group=group), fragment(1, "question", group=group),
                   fragment(2, outcome, group=group)])
    scores = {fs[0].id: 1, fs[3].id: .9}
    selected, trace = select(fs, scores, "interview", mode)
    assert {f.id for f, _ in selected} == {f.id for f in fs}
    assert len(trace["spans"]) == 2
    assert [score for f, score in selected if f in (fs[2], fs[5])] == [0, 0]


def test_short_source_uses_pair_and_gap_cannot_supply_a_third_turn():
    fs = [fragment(0, "interview"), fragment(1, "accepted"), fragment(3, "other")]
    selected, trace = select(fs, {fs[0].id: 1}, "interview", "query_span_allocation")
    assert [f for f, _ in selected] == fs[:2]
    assert trace["spans"][0]["ids"] == [f.id for f in fs[:2]]
    selected, trace = select([fs[0], fs[2]], {fs[0].id: 1}, "interview", "query_span_allocation")
    assert [f for f, _ in selected] == [fs[0]]
    assert trace["spans"] == []


@pytest.mark.parametrize("change", [{"owner": "other"}, {"session": "other"},
    {"group": "other"}, {"tags": ("subject:other",)}, {"parents": ("parent",)}])
def test_span_expansion_reuses_provenance_boundaries(change):
    fs = [fragment(0, "interview"), fragment(1, "outcome", **change)]
    result, trace = select(fs, {fs[0].id: 1}, "interview", "query_span_allocation")
    assert [f for f, _ in result] == [fs[0]] and trace["spans"] == []


@pytest.mark.parametrize("mode", MODES)
def test_exact_budget_duplicate_ids_and_stable_order(mode):
    fs = [fragment(i, text) for i, text in enumerate(["query", "\u4e2d" * 1000, "detail"])]
    scores = {fs[0].id: 1, fs[2].id: .5}
    budget = len(render([fs[0]]).encode("utf-8"))
    first, trace = select(fs, scores, "query", mode, max_bytes=budget)
    second, again = select(fs, scores, "query", mode, max_bytes=budget)
    assert first == second and trace == again
    assert len({f.id for f, _ in first}) == len(first)
    assert len(render([f for f, _ in first]).encode("utf-8")) <= budget
    assert select(fs, scores, "query", mode, limit=0)[0] == []


def test_actual_emission_order_controls_arbitrary_renderer_byte_budget():
    fs = [fragment(0, "query"), fragment(1, "context"), fragment(2, "tail")]
    def order_sensitive(group):
        return "X" * (100 if group and group[0] == fs[1] else len(group))
    option = memory.ResearchSelection("query_span_allocation", order_sensitive, 3)
    result, trace = memory.ordered_selection(fs, {fs[1].id: 1}, 6, option,
        snapshot=memory.recall_tokens(fs, "query", "unicode"))
    assert len(order_sensitive([f for f, _ in result]).encode()) <= 3
    assert trace["rendered_bytes"] <= 3


def test_normal_round_completion_and_expansion_failure_are_distinct(monkeypatch):
    fs = [fragment(0, "query"), fragment(1, "context"), fragment(2, "tail")]
    scores = {fs[0].id: 1}
    _, trace = select(fs, scores, "query", "query_span_allocation")
    assert trace["fallback"] is None and trace["search"]["approximate"] is True
    monkeypatch.setattr(memory, "_ALLOCATION_MAX_EXPANSIONS", 0)
    result, trace = select(fs, scores, "query", "query_span_allocation")
    assert result == [(fs[0], 1)] and trace["fallback"] == "expansion_limit"


@pytest.mark.parametrize("mode", MODES)
def test_single_state_advance_raw_writeback_same_hebbian_pool_and_independent_witness(mode):
    fs = [fragment(0, "interview"), fragment(1, "question"), fragment(2, "accepted")]
    state, edges = memory.ActivationState(), EdgeStore()
    result = memory.reconstruct(fs, "interview", None, state, 0, edges=edges,
        research_selection=memory.ResearchSelection(mode, render))
    assert state.recall_count == 1
    assert state.residual == {fs[0].id: 1}
    for a, b in combinations(fs, 2):
        assert edges.get(a.id, b.id) == (.05 if a in result.fragments_used and b in result.fragments_used else 0)
    systems = [CompanionSystem(byte_cap=None, research_selection=memory.ResearchSelection(mode, render)),
               CompanionSystem(byte_cap=None)]
    for system in systems:
        for f in fs:
            system._ingest_fragment(f)
    a, b = [s.recall("interview") for s in systems]
    assert systems[0].witness_state.activation == systems[1].witness_state.activation
    assert systems[0].witness_edges == systems[1].witness_edges


def test_source_groups_are_not_claimed_to_be_semantic_events():
    fs = [fragment(i, text) for i, text in enumerate([
        "interview first", "question", "accepted", "interview second", "question", "rejected"])]
    scores = {fs[0].id: 1, fs[3].id: .9}
    _, trace = select(fs, scores, "interview", "query_span_allocation")
    assert trace["objective"][1] <= 1
