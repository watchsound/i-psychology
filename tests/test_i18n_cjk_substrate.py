"""Phase 9+ i18n follow-on: CJK coverage of the secondary substrates.

The main-recall path jieba knob (``CompanionSystem(cjk_tokenizer="jieba")``)
shipped the primary cue→content matching migration. Two secondary paths
were initially flagged as "unicode-default, follow-up migration" but
turned out to need baseline-correctness fixes:

  1. ``companion/witness.py`` W4 pain-divergence computed Jaccard on
     ``tokens(content)`` without a ``cjk_tokenizer`` kwarg, so a jieba
     Companion coupled with a unicode Witness would compare
     jieba-segmented word tokens against one-blob-per-clause tokens.
     The two sets would never intersect — divergence pinned near 1.0 —
     and W4 pain feedback would fire on every Chinese recall even when
     the two reconstructions were semantically identical.

  2. ``companion/entity_graph.py`` ``_ENTITY_TOKEN_RE`` was ASCII-only
     (``[A-Za-z0-9][A-Za-z0-9'\\-]*``), so ``_tokens("焦虑")`` returned
     ``[]`` and every Chinese fact entity was silently dropped from
     ``_token_to_nodes``. Cue "焦虑" could never reach any Chinese node.

Both fixes land under the same byte-identical English contract: the
regex now includes ``\\u4e00-\\u9fff``, mirroring the primary-path
``_TOKEN_RE`` in ``reconstruction.py``; witness threads the same
``cjk_tokenizer`` as the Companion so the two legs stay sign-coherent.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.companion.entity_graph import (
    EntityGraph,
    _tokens,
    build_entity_graph,
)
from psychology_base.companion.facts import Fact
from psychology_base.companion.reconstruction import (
    ActivationState,
    Reconstruction,
    reconstruct,
)
from psychology_base.companion.witness import (
    WitnessState,
    _token_divergence,
    witness_couple,
)
from psychology_base.eval.adapters import Fragment


TS = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)


# ---------- entity_graph CJK regex ----------


def test_entity_graph_tokens_cjk_now_returns_cjk_run() -> None:
    """Pre-fix, _tokens('焦虑') returned [] because the ASCII-only regex
    skipped every CJK char. Post-fix the full CJK run is one token."""
    assert _tokens("焦虑") == ["焦虑"]
    assert _tokens("用户") == ["用户"]


def test_entity_graph_tokens_mixed_latin_cjk() -> None:
    """Mixed-script input tokenises both sides. Space separates runs;
    the two scripts don't merge into one blob because they are both
    letter-run classes."""
    assert _tokens("user 焦虑 anxiety") == ["user", "焦虑", "anxiety"]


def test_entity_graph_tokens_english_byte_identical() -> None:
    """Regression lock: adding the CJK range to the character class
    must not change English tokenization. ASCII input stays identical
    to the pre-fix behavior (lowercased alphanumeric runs with
    apostrophes/hyphens kept)."""
    assert _tokens("Hello world") == ["hello", "world"]
    assert _tokens("don't go") == ["don't", "go"]
    assert _tokens("state-of-the-art AI") == ["state-of-the-art", "ai"]
    assert _tokens("User123") == ["user123"]


def _cjk_fact(entity: str, relation: str, value: str, fid: str = "f") -> Fact:
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


def test_entity_graph_cjk_fact_reachable_from_cjk_cue() -> None:
    """End-to-end: build a graph from a Chinese fact, probe it with a
    Chinese cue, confirm the entity is returned. Pre-fix this returned
    ``set()`` because ``_tokens("焦虑")`` was empty, so the token-index
    pass had nothing to look up."""
    g = build_entity_graph([_cjk_fact("用户", "情绪", "焦虑")])
    assert len(g) == 2
    # Both normalized nodes are reachable from either CJK cue.
    assert "用户" in g.entities_from_cue("用户")
    assert "焦虑" in g.entities_from_cue("焦虑")


def test_entity_graph_cjk_cue_reaches_both_endpoints() -> None:
    """Both edge endpoints land in the token index, so a cue naming
    either entity surfaces its neighbour via graph expansion downstream."""
    g = build_entity_graph(
        [
            _cjk_fact("用户", "情绪", "焦虑", "f1"),
            _cjk_fact("用户", "压力源", "工作", "f2"),
        ]
    )
    assert g.neighbors("用户") == {"焦虑": 1, "工作": 1}


# ---------- witness W4 pain divergence under jieba knob ----------


def _rec(content: str, fragments: tuple[Fragment, ...] = ()) -> Reconstruction:
    """Build a bare Reconstruction with just the content string set.
    _token_divergence only inspects .content, so fragments_used can be
    empty for this unit."""
    return Reconstruction(
        content=content, fragments_used=fragments, activation_profile={}
    )


def test_token_divergence_zh_pair_near_identical_jieba_knob() -> None:
    """Two reconstructions with the SAME word-level content but
    DIFFERENT punctuation / spacing collapse to one CJK blob each under
    the unicode tokenizer, so the Jaccard divergence pins at 1.0 even
    though the two strings are near-identical word-level.

    Under jieba the strings tokenize into shared word tokens and
    divergence drops accordingly."""
    pytest.importorskip("jieba")
    a = _rec("我最近感到很焦虑")
    b = _rec("最近感到焦虑")
    assert _token_divergence(a, b, cjk_tokenizer="unicode") == 1.0
    jieba_div = _token_divergence(a, b, cjk_tokenizer="jieba")
    assert jieba_div < 1.0, (
        "jieba should segment both strings into word tokens with "
        "non-empty intersection — pain should NOT pin at 1.0."
    )


def test_token_divergence_english_byte_identical_under_both_knobs() -> None:
    """English content under jieba knob is routed to text.split() → the
    legacy letter-run path, so the divergence computation must be
    byte-identical to the unicode default."""
    pytest.importorskip("jieba")
    a = _rec("i feel anxious about work")
    b = _rec("i am anxious about work today")
    uni = _token_divergence(a, b, cjk_tokenizer="unicode")
    jie = _token_divergence(a, b, cjk_tokenizer="jieba")
    assert uni == jie


def test_witness_couple_zh_pain_coherent_under_jieba_pair() -> None:
    """End-to-end: companion + witness BOTH on jieba produce a coherent
    pain signal on a Chinese fragment set. Pre-fix, Witness
    reconstructed with unicode tokens even if Companion was on jieba —
    the two pools would never share tokens so divergence pinned at 1.0.
    """
    pytest.importorskip("jieba")

    fragments = [
        Fragment(
            id=f"f{i}",
            content=content,
            timestamp=TS,
            tags=("speaker:user",),
        )
        for i, content in enumerate(
            [
                "我最近感到很焦虑",
                "工作压力很大",
                "我昨天去跑步了",
            ]
        )
    ]
    state = ActivationState()
    rec = reconstruct(
        fragments=fragments,
        cue="焦虑",
        context=None,
        state=state,
        tick_count=0,
        system_seed=0,
        top_k=3,
        cjk_tokenizer="jieba",
    )
    witness_state = WitnessState()
    pain = witness_couple(
        fragments=fragments,
        cue="焦虑",
        companion_context=None,
        companion_reconstruction=rec,
        witness_state=witness_state,
        companion_activation=state,
        witness_edges=None,
        tick_count=0,
        system_seed=0,
        top_k=3,
        cjk_tokenizer="jieba",
    )
    # Pain must be a real [0, 1] float — not pinned at 1.0 by the
    # unicode-vs-jieba tokenizer mismatch bug.
    assert 0.0 <= pain < 1.0


def test_witness_couple_zh_pain_degenerate_without_jieba_dispatch() -> None:
    """Baseline — under the unicode default, three mono-clause Chinese
    fragments share zero tokens pairwise, so Companion and Witness both
    collapse to one-blob reconstructions and (depending on context flip)
    the pain signal pins degenerately. This is the pre-jieba honest
    limit; the test documents it so a future substrate change doesn't
    silently drift the behavior."""
    fragments = [
        Fragment(
            id=f"f{i}",
            content=content,
            timestamp=TS,
            tags=("speaker:user",),
        )
        for i, content in enumerate(
            [
                "我最近感到很焦虑",
                "工作压力很大",
                "我昨天去跑步了",
            ]
        )
    ]
    state = ActivationState()
    rec = reconstruct(
        fragments=fragments,
        cue="焦虑",
        context=None,
        state=state,
        tick_count=0,
        system_seed=0,
        top_k=3,
    )
    # Companion recall under unicode default returns no fragments
    # because '焦虑' as a cue does not share any token with the stored
    # mono-clause blobs. This is the Phase 9+ documented honest limit.
    assert rec.fragments_used == ()


@pytest.mark.parametrize("tokenizer", ["unicode", "jieba"])
@pytest.mark.parametrize("tags", [(), ("user:u1", "session:s1", "speaker:user")], ids=["tag_free", "application_tags"])
def test_initial_edges_use_configured_tokens(chinese_cases, tokenizer, tags):
    from psychology_base.companion.reconstruction import jaccard_prior
    from psychology_base.companion.system import CompanionSystem
    from psychology_base.eval.adapters import Observation

    if tokenizer == "jieba":
        pytest.importorskip("jieba")
    system = CompanionSystem(cjk_tokenizer=tokenizer)
    for text in chinese_cases["edge_pair"]:
        system.observe(Observation(timestamp=TS, speaker="user", content=text, tags=tags))
    a, b = system.companion.store.all()
    expected = jaccard_prior(a, b, cjk_tokenizer=tokenizer)
    if not tags:
        assert expected == pytest.approx(0.36 if tokenizer == "jieba" else 0.0)
    assert system.activation.recall_count == 0
    assert system.edges.get(a.id, b.id) == pytest.approx(expected)
    assert system.witness_edges.get(a.id, b.id) == pytest.approx(expected)
    assert system.edges is not system.witness_edges


def test_configured_system_witness_uses_same_token_policy(chinese_cases):
    from psychology_base.companion.system import CompanionSystem
    from psychology_base.eval.adapters import Observation

    pytest.importorskip("jieba")
    system = CompanionSystem(cjk_tokenizer="jieba")
    for text in chinese_cases["edge_pair"]:
        system.observe(Observation(timestamp=TS, speaker="user", content=text))
    rec = system.recall("焦虑")
    assert rec.fragments_used
    assert system.witness_state.pain_log
    assert system.witness_state.activation.recall_count == 1
    assert 0.0 <= system.witness_state.pain_log[-1].pain < 1.0
