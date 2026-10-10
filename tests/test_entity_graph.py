"""Phase 2b Step C — entity graph unit tests.

Pure-function coverage of ``EntityGraph`` + ``score_fragments``. No
LLM wiring; builds graphs from hand-written Fact fixtures and scores
synthetic fragments.
"""

from __future__ import annotations

from datetime import datetime, timezone

from psychology_base.companion.entity_graph import (
    EntityGraph,
    build_entity_graph,
    score_fragments,
)
from psychology_base.companion.facts import Fact
from psychology_base.eval.adapters import Fragment


def _fact(entity: str, relation: str, value: str, fid: str = "f") -> Fact:
    return Fact(
        id=fid,
        entity=entity,
        relation=relation,
        value=value,
        confidence=0.9,
        source_fragment_ids=(),
        created_tick=0,
        updated_tick=0,
    )


def _frag(fid: str, content: str) -> Fragment:
    return Fragment(
        id=fid,
        content=content,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_build_from_empty():
    g = build_entity_graph([])
    assert len(g) == 0
    assert g.entities_from_cue("anything") == set()


def test_single_fact_adds_two_nodes_and_edge():
    g = build_entity_graph([_fact("Caroline", "origin", "Sweden")])
    assert len(g) == 2
    assert g.neighbors("Caroline") == {"sweden": 1}
    assert g.neighbors("sweden") == {"caroline": 1}


def test_edge_weight_counts_facts():
    g = build_entity_graph(
        [
            _fact("Caroline", "origin", "Sweden", "f1"),
            _fact("Caroline", "lived_in", "Sweden", "f2"),
            _fact("Caroline", "visited", "Rome", "f3"),
        ]
    )
    assert g.neighbors("Caroline")["sweden"] == 2
    assert g.neighbors("Caroline")["rome"] == 1


def test_entities_from_cue_single_token_match():
    g = build_entity_graph([_fact("Caroline", "origin", "Sweden")])
    assert "caroline" in g.entities_from_cue("Where did Caroline grow up?")


def test_entities_from_cue_multi_token_phrase_match():
    g = build_entity_graph([_fact("Caroline", "home_country", "Sweden")])
    # Build a graph node that's multi-word by using a multi-word value.
    g.ingest_fact("Caroline", "favorite thing")
    assert "favorite thing" in g.entities_from_cue(
        "tell me about her favorite thing"
    )


def test_entities_from_cue_single_token_requires_token_match_not_substring():
    """``Rome`` as a node should NOT match ``chrome`` in a cue — token
    boundary matters for precision.
    """
    g = build_entity_graph([_fact("Caroline", "visited", "Rome")])
    assert "rome" not in g.entities_from_cue("Chrome keeps crashing")
    assert "rome" in g.entities_from_cue("Did she visit Rome?")


def test_expand_one_hop_includes_seed_and_neighbors():
    g = build_entity_graph(
        [
            _fact("Caroline", "origin", "Sweden"),
            _fact("Caroline", "visited", "Rome"),
            _fact("Caroline", "owns", "cat"),
        ]
    )
    expanded = g.expand(["Caroline"])
    assert "caroline" in expanded
    assert "sweden" in expanded
    assert "rome" in expanded
    assert "cat" in expanded


def test_expand_cap_per_seed_limits_fan_out():
    """With cap=1, only the single highest-weight neighbor + seed is kept."""
    g = build_entity_graph(
        [
            _fact("Caroline", "origin", "Sweden", "f1"),
            _fact("Caroline", "origin", "Sweden", "f2"),
            _fact("Caroline", "visited", "Rome"),
            _fact("Caroline", "owns", "cat"),
        ]
    )
    expanded = g.expand(["Caroline"], cap_per_seed=1)
    assert expanded == {"caroline", "sweden"}


def test_score_fragments_lifts_entity_mentioning_turn():
    g = build_entity_graph([_fact("Caroline", "origin", "Sweden")])
    fragments = [
        _frag("t1", "The weather is nice today."),
        _frag("t2", "I grew up in Sweden with my family."),
        _frag("t3", "Caroline mentioned Sweden last week."),
    ]
    scored = score_fragments(g, "where is Caroline from?", fragments)
    # Caroline mentioned → caroline + sweden expanded;
    # t3 has both → hits=2; t2 has sweden only → hits=1; t1 has none.
    assert scored[0][0].id == "t3"
    assert scored[0][1] == 2
    assert scored[1][0].id == "t2"
    assert scored[1][1] == 1
    assert all(f.id != "t1" for f, _ in scored)


def test_score_fragments_empty_when_no_cue_match():
    g = build_entity_graph([_fact("Caroline", "origin", "Sweden")])
    fragments = [_frag("t1", "Sweden is cold.")]
    # Cue mentions neither "Caroline" nor "Sweden" → no seeds → empty.
    assert score_fragments(g, "the weather today", fragments) == []


def test_score_fragments_dedupes_entity_within_fragment():
    """Mentioning the same expanded entity 3 times in one fragment
    still counts as 1 hit. Prevents a single verbose turn from
    dominating via repetition.
    """
    g = build_entity_graph([_fact("Caroline", "origin", "Sweden")])
    fragments = [_frag("t1", "Sweden Sweden Sweden!")]
    scored = score_fragments(g, "ask about Caroline", fragments)
    assert scored == [(fragments[0], 1)]


def test_normalization_is_case_insensitive():
    g = build_entity_graph([_fact("CAROLINE", "origin", "sweden")])
    assert "caroline" in g.entities_from_cue("about Caroline")
    assert g.neighbors("caroline") == {"sweden": 1}


def test_ingest_fact_skips_empty_strings():
    g = EntityGraph()
    g.ingest_fact("", "something")
    g.ingest_fact("something", "")
    g.ingest_fact("  ", "  ")
    assert len(g) == 0
