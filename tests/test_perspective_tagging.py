"""M2 unit tests — write-path tag merge + retrieval-time scope filter.

Covers:

1. ``Observation.subject_id`` + ``entity_id`` default to None (M1
   back-compat; LoCoMo adapter is unchanged).
2. ``CompanionSystem.observe`` merges ``subject:<id>`` and
   ``entity:<id>`` into fragment.tags when the observation carries
   them. No merge when both are None.
3. ``_filter_fragments_by_scope`` returns scope-less fragments
   unchanged (back-compat), excludes mismatched scoped fragments,
   and keeps matched scoped fragments.
4. ``CompanionSystem.recall(scope=...)`` routes through the pure
   reconstruct() path (bypassing the hierarchical fact-store path) and
   returns only fragments whose scope matches.
5. Content-addressing stays stable: adding perspective tags shifts
   the hash (expected), but re-constructing with the same inputs
   produces the same hash (determinism).
"""

from __future__ import annotations

from datetime import datetime, timezone

from psychology_base.companion.system import (
    CompanionSystem,
    _filter_fragments_by_scope,
)
from psychology_base.eval.adapters import (
    Fragment,
    Observation,
    make_fragment,
)


def _dt(seconds: int = 0) -> datetime:
    return datetime(2026, 1, 1, 12, 0, seconds, tzinfo=timezone.utc)


def test_observation_defaults_leave_perspective_tags_off():
    obs = Observation(
        timestamp=_dt(),
        speaker="user",
        content="hello",
    )
    assert obs.subject_id is None
    assert obs.entity_id is None


def test_observe_without_perspective_leaves_tags_unchanged():
    sys = CompanionSystem(byte_cap=None)
    sys.observe(
        Observation(
            timestamp=_dt(),
            speaker="user",
            content="hello world",
            tags=("greeting",),
        )
    )
    frag = sys.companion.store.all()[0]
    assert frag.tags == ("greeting",)
    # No scope tags leaked in.
    assert not any(
        t.startswith("subject:") or t.startswith("entity:")
        for t in frag.tags
    )


def test_observe_with_subject_and_entity_merges_tag_prefixes():
    sys = CompanionSystem(byte_cap=None)
    sys.observe(
        Observation(
            timestamp=_dt(),
            speaker="user",
            content="my mother called",
            tags=("mother",),
            subject_id="user",
            entity_id="mother",
        )
    )
    frag = sys.companion.store.all()[0]
    assert "mother" in frag.tags
    assert "subject:user" in frag.tags
    assert "entity:mother" in frag.tags


def test_observe_with_only_subject_merges_subject_only():
    sys = CompanionSystem(byte_cap=None)
    sys.observe(
        Observation(
            timestamp=_dt(),
            speaker="ai",
            content="observation of user",
            subject_id="ai",
        )
    )
    frag = sys.companion.store.all()[0]
    assert "subject:ai" in frag.tags
    assert not any(t.startswith("entity:") for t in frag.tags)


def test_scope_filter_passes_through_scope_less_fragments():
    f = make_fragment(
        content="untagged",
        timestamp=_dt(),
        tags=("greeting",),
    )
    kept = _filter_fragments_by_scope([f], ("user", "mother"))
    assert kept == [f]


def test_scope_filter_keeps_matching_scope_only():
    user_mother = make_fragment(
        content="mother called",
        timestamp=_dt(1),
        tags=("subject:user", "entity:mother"),
    )
    ai_user = make_fragment(
        content="observation of user",
        timestamp=_dt(2),
        tags=("subject:ai", "entity:user"),
    )
    user_user = make_fragment(
        content="I'm not a math person",
        timestamp=_dt(3),
        tags=("subject:user", "entity:user"),
    )
    all_frags = [user_mother, ai_user, user_user]
    assert _filter_fragments_by_scope(all_frags, ("user", "mother")) == [
        user_mother
    ]
    assert _filter_fragments_by_scope(all_frags, ("ai", "user")) == [ai_user]
    assert _filter_fragments_by_scope(all_frags, ("user", "user")) == [
        user_user
    ]


def test_scope_filter_excludes_partial_match():
    # Fragment with subject:user but entity:boss shouldn't match
    # scope=(user, mother).
    f = make_fragment(
        content="boss meeting",
        timestamp=_dt(),
        tags=("subject:user", "entity:boss"),
    )
    kept = _filter_fragments_by_scope([f], ("user", "mother"))
    assert kept == []


def test_recall_with_scope_pulls_scoped_fragments_only():
    sys = CompanionSystem(byte_cap=None, witness_active=False)
    # Three observations across three scopes with overlapping content
    # so the Jaccard reconstructor would otherwise pull from all pools.
    sys.observe(
        Observation(
            timestamp=_dt(1),
            speaker="user",
            content="mother called about the gas bill",
            subject_id="user",
            entity_id="mother",
        )
    )
    sys.observe(
        Observation(
            timestamp=_dt(2),
            speaker="user",
            content="I am not good with gas bill math",
            subject_id="user",
            entity_id="user",
        )
    )
    sys.observe(
        Observation(
            timestamp=_dt(3),
            speaker="ai",
            content="user solved the gas bill derivation first try",
            subject_id="ai",
            entity_id="user",
        )
    )
    rec = sys.recall("gas bill", scope=("user", "mother"))
    frag_contents = [f.content for f in rec.fragments_used]
    # Only the mother-scoped fragment can appear in the pool.
    assert any("mother called" in c for c in frag_contents)
    assert not any("not good with" in c for c in frag_contents)
    assert not any("first try" in c for c in frag_contents)
    assert rec.metadata["scope"] == ("user", "mother")


def test_recall_without_scope_sees_everything():
    sys = CompanionSystem(byte_cap=None, witness_active=False)
    sys.observe(
        Observation(
            timestamp=_dt(1),
            speaker="user",
            content="mother called about the gas bill",
            subject_id="user",
            entity_id="mother",
        )
    )
    sys.observe(
        Observation(
            timestamp=_dt(2),
            speaker="user",
            content="I am not good with gas bill math",
            subject_id="user",
            entity_id="user",
        )
    )
    rec = sys.recall("gas bill")
    assert rec.metadata.get("scope") is None


def test_content_hash_is_deterministic_with_perspective_tags():
    a = make_fragment(
        content="mother called",
        timestamp=_dt(),
        tags=("subject:user", "entity:mother"),
    )
    b = make_fragment(
        content="mother called",
        timestamp=_dt(),
        tags=("subject:user", "entity:mother"),
    )
    assert a.id == b.id


def test_session_player_persists_recall_metadata_including_scope():
    """Transcript-level audit: ``RecallRecord.metadata`` must carry the
    scope the recall was invoked under (and the other substrate signals
    the memory system exposes), so downstream analysis does not have to
    re-run the scenario to find out which recalls were scope-filtered.
    """
    from datetime import datetime, timezone

    from psychology_base.eval.scripts.schema import (
        ObserveTurn,
        RecallTurn,
        ScriptedScenario,
        Session,
    )
    from psychology_base.eval.session_player import SessionPlayer

    at_observe = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    at_recall_scoped = datetime(2026, 1, 1, 12, 1, 0, tzinfo=timezone.utc)
    at_recall_plain = datetime(2026, 1, 1, 12, 2, 0, tzinfo=timezone.utc)

    scenario = ScriptedScenario(
        id="metadata-audit",
        benchmark="B2",
        description="",
        sessions=[
            Session(
                id="s1",
                description="",
                turns=[
                    ObserveTurn(
                        type="observe",
                        at=at_observe,
                        speaker="user",
                        content="mother called about the gas bill",
                        subject="user",
                        entity="mother",
                    ),
                    RecallTurn(
                        type="recall",
                        at=at_recall_scoped,
                        cue="mother",
                        scope=["user", "mother"],
                    ),
                    RecallTurn(
                        type="recall",
                        at=at_recall_plain,
                        cue="mother",
                    ),
                ],
            )
        ],
    )
    sys = CompanionSystem(byte_cap=None, witness_active=False)
    player = SessionPlayer(sys)
    transcript = player.play(scenario)

    assert len(transcript) == 2
    assert transcript[0].metadata["scope"] == ("user", "mother")
    assert transcript[1].metadata.get("scope") is None
    # Spot-check that substrate signals ride along too — not just scope.
    assert "pool_size" in transcript[0].metadata


def test_content_hash_shifts_when_perspective_tag_added():
    """Content-hash covers tags, so a bare fragment and a
    perspective-tagged fragment with identical content hash to
    different ids. This is the provenance guarantee M2 relies on."""
    bare = make_fragment(content="mother called", timestamp=_dt())
    tagged = make_fragment(
        content="mother called",
        timestamp=_dt(),
        tags=("subject:user", "entity:mother"),
    )
    assert bare.id != tagged.id
