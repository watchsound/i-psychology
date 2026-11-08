"""Real memory/application contracts; fixed replies are not quality evidence."""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from psychology_base.applications.emotional.companion_agent import CompanionAgent
from psychology_base.applications.learning.tutor_agent import TutorAgent
from psychology_base.applications.measurement import ScaffoldingTelemetry
from psychology_base.applications.safety import CrisisEscalation
from psychology_base.companion.fragment import fragment_bytes
from psychology_base.companion.pam_cli import restore_system_from_path
from psychology_base.companion.pam_export import pam_export, pam_import, validate_content_addressing
from psychology_base.companion.reconstruction import (
    ActivationState, _distinct_ranked_fragments, jaccard_prior, spread_activation,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.companion import turn_provenance as turn
from psychology_base.eval.adapters import Observation, make_fragment
from psychology_base.eval.benchmarks import attachment_longitudinal as harness
from psychology_base.eval.benchmarks.graph_priors import system_for_policy, local_association_prior


STAMP = datetime(2026, 10, 18, tzinfo=timezone.utc)
GROUP = "6225f741-f20b-40d3-936a-47d7d6e425df"
OTHER = "d22b1598-249c-4fc1-9c52-56c44623c8b2"


def fragment(position=0, *, group=GROUP, actor=None, text=None, tags=(), parents=(), sequence=False):
    actor = actor or ("user" if position == 0 else "companion")
    descriptor = (turn.sequence_tag("pinned-source", group, position) if sequence else
                  turn.exchange_tag(group, actor, position))
    return make_fragment(text or f"observation {position}", STAMP,
                         tags=("user:alice", "session:one", *tags, descriptor), parent_ids=parents)


@pytest.mark.parametrize("changes", [
    {"kind": "unknown"}, {"group": "resetting-counter"}, {"position": True}, {"position": -1},
    {"position": 2}, {"position": 2**63}, {"position": 0.0}, {"actor": "assistant"},
    {"actor": None}, {"actor": []}, {"actor": "companion"}, {"group": None},
])
def test_writer_rejects_invalid_exchange(changes):
    fields = dict(kind="exchange", group=GROUP, position=0, actor="user")
    with pytest.raises(ValueError):
        turn.TurnDescriptor(**(fields | changes))


@pytest.mark.parametrize("kind,tag,status", [
    ("valid", turn.exchange_tag(GROUP, "user", 0), "valid"),
    ("future", "pb-turn:v2:anything", "unsupported"),
    ("json", "pb-turn:v1:{", "malformed"),
    ("long", "pb-turn:v1:" + "x" * 257, "malformed"),
    ("noncanonical", turn.exchange_tag(GROUP, "user", 0).replace(':0', ': 0'), "malformed"),
    ("fields", 'pb-turn:v1:{"actor":"user"}', "malformed"),
    ("array", "pb-turn:v1:[]", "malformed"),
])
def test_reader_preserves_opaque_tags_and_reports_status(kind, tag, status):
    f = make_fragment("payload", STAMP, tags=(tag,))
    assert turn.parse_turn(f.tags).status == status
    assert pam_import(pam_export((f,))) == [f]
    assert validate_content_addressing((f,)) == []
    assert fragment_bytes(f) == len(f.content.encode()) + len(tag.encode())


def test_ambiguous_metadata_does_not_invent_an_actor_or_relationship():
    a, b = fragment(), fragment(1)
    for tags in ((*a.tags, a.tags[-1]), (*a.tags, "pb-turn:v2:future"), (*a.tags, "speaker:tutor")):
        ambiguous = replace(a, tags=tags)
        assert turn.parse_turn(tags).status == "ambiguous"
        assert turn.event_key(ambiguous) is None
        assert not turn.locally_associated(ambiguous, b)


@pytest.mark.parametrize("mode", ["exchange", "sequence"])
def test_local_pair_uses_original_positions_and_is_symmetric(mode):
    a, b = (fragment(i, sequence=mode == "sequence") for i in (0, 1))
    assert turn.locally_associated(a, b) and turn.locally_associated(b, a)
    assert local_association_prior(a, b) == 1.0
    assert local_association_prior(a, a) == jaccard_prior(a, a)
    if mode == "sequence":
        assert turn.parse_turn(a.tags).descriptor.actor is None
        assert not turn.locally_associated(a, fragment(2, sequence=True))


@pytest.mark.parametrize("change", ["owner", "session", "missing_owner", "missing_session", "ambiguous_owner",
                                    "subject", "entity", "witness", "group", "parent", "kind", "multi_subject"])
@pytest.mark.parametrize("sequence", [False, True])
def test_no_bonus_across_identity_scope_or_version_boundaries(change, sequence):
    a = fragment(sequence=sequence, tags=("speaker:user",))
    b = fragment(1, sequence=sequence, tags=("speaker:companion",))
    if change == "group":
        b = fragment(1, group=OTHER, sequence=sequence)
    elif change == "parent":
        b = fragment(1, parents=(a.id,), sequence=sequence)
    elif change == "kind":
        b = fragment(1, sequence=not sequence)
    elif change.startswith("missing_"):
        prefix = "user:" if change == "missing_owner" else "session:"
        b = replace(b, tags=tuple(t for t in b.tags if not t.startswith(prefix)))
    elif change == "owner":
        b = replace(b, tags=tuple("user:bob" if t == "user:alice" else t for t in b.tags))
    elif change == "session":
        b = replace(b, tags=tuple("session:two" if t == "session:one" else t for t in b.tags))
    elif change == "ambiguous_owner":
        b = replace(b, tags=(*b.tags, "user:bob"))
    elif change == "multi_subject":
        a = replace(a, tags=(*a.tags, "subject:a", "subject:b"))
        b = replace(b, tags=(*b.tags, "subject:a", "subject:b"))
    else:
        b = replace(b, tags=(*b.tags, f"{change}:different"))
    assert not turn.locally_associated(a, b)
    assert local_association_prior(a, b) == jaccard_prior(a, b)


def test_source_group_depends_on_pinned_identity_and_dialogue_not_window():
    first = turn.parse_turn((turn.sequence_tag("sha-one", "dialogue", 4),)).descriptor
    later = turn.parse_turn((turn.sequence_tag("sha-one", "dialogue", 9),)).descriptor
    assert first.group == later.group
    assert first.group != turn.parse_turn((turn.sequence_tag("sha-two", "dialogue", 4),)).descriptor.group
    assert first.group != turn.parse_turn((turn.sequence_tag("sha-one", "other", 4),)).descriptor.group


@pytest.mark.parametrize("speaker", ["user", "companion", "tutor", "source-actor-0", "source-actor-1"])
def test_source_sequence_keeps_side_speaker_without_inventing_descriptor_actor(speaker):
    a = fragment(sequence=True, tags=(f"speaker:{speaker}",))
    b = fragment(1, sequence=True, tags=("speaker:source-actor-1",))
    parsed = turn.parse_turn(a.tags)
    assert parsed.status == "valid" and parsed.descriptor.actor is None
    assert turn.evidence_tags(a) == tuple(sorted(t for t in a.tags if not t.startswith(turn.NAMESPACE)))
    assert turn.local_neighbors((a, b)) == {a.id: (b.id,), b.id: (a.id,)}
    assert pam_import(pam_export((a, b))) == [a, b]
    assert not validate_content_addressing((a, b))


@pytest.mark.parametrize("speakers", [("",), ("user", "tutor"), ("source-actor-0", "source-actor-1")])
def test_source_sequence_rejects_empty_or_conflicting_side_speakers(speakers):
    f = fragment(sequence=True, tags=tuple(f"speaker:{s}" for s in speakers))
    assert turn.parse_turn(f.tags).status == "ambiguous"
    assert turn.event_key(f) is None


@pytest.mark.parametrize("sequence", [False, True])
def test_explicit_descriptor_actor_still_requires_matching_side_speaker(sequence):
    descriptor = (turn.TurnDescriptor("source-sequence", "a" * 64, 0, "user").encode()
                  if sequence else turn.exchange_tag(GROUP, "user", 0))
    for speaker in ("tutor", "source-actor-0", ""):
        assert turn.parse_turn((descriptor, f"speaker:{speaker}")).status == "ambiguous"
    assert turn.parse_turn((descriptor, "speaker:user")).status == "valid"


def test_source_speaker_event_conflict_fails_before_live_or_restore_mutation(tmp_path):
    a = fragment(sequence=True, tags=("speaker:source-actor-0",))
    conflict = fragment(sequence=True, text="different source payload", tags=("speaker:source-actor-0",))
    system = system_for_policy("local_association", cjk_tokenizer="unicode")
    system._ingest_fragment(a)
    before = copy.deepcopy((system.companion.store.all(), system.witness.store.all(),
                            system.edges, system.witness_edges))
    path = tmp_path / "source.json"
    path.write_text(json.dumps(pam_export((conflict,))), encoding="utf-8")
    for intake in (lambda: system._ingest_fragment(conflict),
                   lambda: restore_system_from_path(path, system=system)):
        with pytest.raises(ValueError, match="event position conflicts"):
            intake()
        assert (system.companion.store.all(), system.witness.store.all(),
                system.edges, system.witness_edges) == before


def test_occurrence_bytes_do_not_change_prior_duplicate_budget_or_residual_ties():
    original = make_fragment("same words", STAMP, tags=("user:alice", "session:one", "speaker:user"))
    a = fragment(text=original.content, tags=("speaker:user",))
    b = fragment(group=OTHER, text=original.content, tags=("speaker:user",))
    assert a.id != b.id != original.id
    assert jaccard_prior(a, b) == jaccard_prior(original, original)
    assert turn.activation_tie_key(a) == turn.activation_tie_key(b) == original.id
    assert len(_distinct_ranked_fragments([(a, 2.0), (b, 1.0), (original, 0.5)])) == 1
    reply = fragment(1, text=original.content)
    unknown = make_fragment(original.content, STAMP, tags=("user:alice", "session:one"))
    assert len(_distinct_ranked_fragments([(a, 1.0), (reply, 1.0), (unknown, 1.0)])) == 3


@pytest.mark.parametrize("field", ["affect", "parents", "perspective"])
def test_distinct_evidence_keeps_semantic_differences(field):
    a = fragment()
    b = {"affect": replace(a, affect={"valence": 0.8}),
         "parents": replace(a, parent_ids=("ancestor",)),
         "perspective": replace(a, tags=(*a.tags, "subject:other"))}[field]
    assert len(_distinct_ranked_fragments([(a, 1.0), (b, 1.0)])) == 2


@pytest.mark.parametrize("policy", ["production", "local_association"])
@pytest.mark.parametrize("sequence", [False, True])
def test_real_restore_repeat_partial_store_and_rendering_lifecycle(tmp_path, policy, sequence):
    system = system_for_policy(policy, cjk_tokenizer="unicode")
    a = fragment(sequence=sequence, tags=("speaker:user",))
    b = fragment(1, sequence=sequence, tags=("speaker:companion",))
    for f in (a, b):
        system._ingest_fragment(f)
    expected = 1.0 if policy == "local_association" else jaccard_prior(a, b)
    assert system.edges.get(a.id, b.id) == expected
    assert system.witness_edges.get(a.id, b.id) == expected
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(pam_export((a, b))), encoding="utf-8")
    fresh = system_for_policy(policy, cjk_tokenizer="unicode")
    restore_system_from_path(path, system=fresh)
    assert fresh.edges == system.edges
    system.edges.bump(a.id, b.id, 0.1)
    learned = copy.deepcopy(system.edges)
    system.witness.store._fragments.remove(b)
    system.witness_edges.remove_fragment(b.id)
    restore_system_from_path(path, system=system)
    assert system.edges == learned
    assert system.witness_edges.get(a.id, b.id) == expected
    child = system.rerender(a.id, "new rendering", timestamp=STAMP, tags=a.tags)
    assert child.parent_ids == (a.id,)
    assert not turn.locally_associated(child, b)
    assert not validate_content_addressing(system.companion.store.all())
    before = copy.deepcopy(system.edges)
    restore_system_from_path(path, system=system)
    assert system.edges == before
    assert len(system.companion.store.all()) == 3


@pytest.mark.parametrize("where", ["companion", "witness", "incoming"])
def test_bulk_conflict_rejected_before_any_store_or_edge_mutation(tmp_path, where):
    system = CompanionSystem(byte_cap=None)
    a, conflict, unrelated = fragment(), fragment(text="different payload"), fragment(group=OTHER)
    if where != "incoming":
        getattr(system, where).store.add(a)
    before = (system.companion.store.all(), system.witness.store.all(), copy.deepcopy(system.edges))
    incoming = (unrelated, a, conflict) if where == "incoming" else (unrelated, conflict)
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(pam_export(incoming)), encoding="utf-8")
    with pytest.raises(ValueError, match="event position conflicts"):
        restore_system_from_path(path, system=system)
    assert (system.companion.store.all(), system.witness.store.all(), system.edges) == before


def test_single_conflict_does_not_advance_aging_or_graphs():
    from psychology_base.companion.aging import EbbinghausPolicy

    system = CompanionSystem(byte_cap=None, aging_policy=EbbinghausPolicy())
    system._ingest_fragment(fragment())
    before = copy.deepcopy((system.companion.store.all(), system.witness.store.all(),
                            system.edges, system.witness_edges, system.aging_policy))
    with pytest.raises(ValueError, match="event position conflicts"):
        system._ingest_fragment(fragment(text="conflicting event"))
    assert (system.companion.store.all(), system.witness.store.all(), system.edges,
            system.witness_edges, system.aging_policy) == before


def build_agent(kind, system, responder):
    fields = dict(user_id="alice", session_id="one", companion_system=system, llm_responder=responder)
    return (CompanionAgent(**fields) if kind == "companion" else
            TutorAgent(**fields, telemetry=ScaffoldingTelemetry(user_id="alice")))


@pytest.mark.parametrize("kind", ["companion", "tutor"])
def test_real_agent_receipts_equal_timestamps_recreation_and_duplicate_slots(kind):
    system = CompanionSystem(byte_cap=None)
    contexts = []

    def responder(prompt, context, utterance):
        contexts.append(context)
        return "What do you notice?"

    receipts = []
    for _ in range(8):
        agent = build_agent(kind, system, responder)
        response = agent.respond("A repeated question", timestamp=STAMP)
        assert response.turn_index == 0
        receipts.append(response.exchange_id)
    assert contexts[0] == ""
    assert len(set(receipts)) == 8
    stored = system.companion.store.all()
    assert len(stored) == 16
    assert len(system.recall("repeated question notice").fragments_used) == 2
    for f in stored:
        d = turn.parse_turn(f.tags).descriptor
        assert d.group in receipts
        assert d.actor == ("user" if d.position == 0 else kind)
    assert not validate_content_addressing(stored)


@pytest.mark.parametrize("kind", ["companion", "tutor"])
def test_failed_responder_leaves_only_actual_input_and_retry_is_new_exchange(kind):
    system = CompanionSystem(byte_cap=None)

    def fail(*args):
        raise RuntimeError("injected responder failure")

    agent = build_agent(kind, system, fail)
    with pytest.raises(RuntimeError, match="injected"):
        agent.respond("real input", timestamp=STAMP)
    only = system.companion.store.all()
    assert len(only) == 1 and turn.parse_turn(only[0].tags).descriptor.position == 0
    previous_group = turn.parse_turn(only[0].tags).descriptor.group
    agent.llm_responder = lambda *args: "What is next?"
    receipt = agent.respond("real input", timestamp=STAMP).exchange_id
    assert receipt != previous_group
    assert len(system.companion.store.all()) == 3


def test_handoff_has_no_receipt_or_synthetic_observations():
    agent = build_agent("companion", CompanionSystem(byte_cap=None), lambda *a: pytest.fail("responder ran"))
    agent.crisis_escalation = CrisisEscalation(handoff_sink=lambda incident: None)
    response = agent.respond("I want to kill myself", timestamp=STAMP)
    assert response.handoff_triggered and response.exchange_id is None
    assert agent.companion_system.companion.store.all() == []


def short_checkpoint(tmp_path):
    payload = harness.build_trajectory("supportive-en").model_dump(mode="json")
    payload["sessions"] = payload["sessions"][:3]
    manifest = harness.TrajectoryManifest.model_validate(payload)
    path = tmp_path / "checkpoint.json"
    result = harness.run_trajectory(manifest, checkpoint_path=path, stop_after=2)
    assert result.state.sessions_completed == 2
    return manifest, path, result


@pytest.mark.parametrize("corruption", ["missing", "malformed", "duplicate", "swapped", "old_schema", "old_policy"])
def test_checkpoint_does_not_accept_invented_or_stale_receipts(tmp_path, corruption):
    manifest, path, _ = short_checkpoint(tmp_path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    receipts = payload["ledger"][0]["exchange_ids"]
    if corruption == "missing":
        payload["ledger"][0].pop("exchange_ids")
    elif corruption == "malformed":
        receipts[0] = "pretend"
    elif corruption == "duplicate":
        receipts[1] = receipts[0]
    elif corruption == "swapped":
        receipts.reverse()
    elif corruption == "old_schema":
        payload["schema_version"] = "attachment-checkpoint-v2"
    else:
        payload["configuration"]["fragment_recall_policy"] = "bounded-spread-distinct-evidence-novel-cue-v4"
    envelope["sha256"] = harness.digest(payload)
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError):
        harness.load_checkpoint(path, manifest)


def test_checkpoint_receipts_are_actual_persisted_metadata(tmp_path):
    manifest, path, result = short_checkpoint(tmp_path)
    state, ledger, restored = harness.load_checkpoint(path, manifest)
    assert state == result.state
    assert harness.canonical_json(ledger) == harness.canonical_json(result.ledger)
    assert pam_export(restored.companion.store.all()) == pam_export(result.system.companion.store.all())
    for spec, record in zip(manifest.sessions, ledger):
        for index, group in enumerate(record["exchange_ids"], 1):
            expected = harness._turn_fragments(manifest, spec, spec.seeker_turns[index - 1],
                                              spec.companion_turns[index - 1], index, group)
            assert all(restored.companion.store.by_id(f.id) == f for f in expected)


def test_eviction_removes_local_edges_without_bridging_sequence_gaps():
    system = system_for_policy("local_association", cjk_tokenizer="unicode")
    fragments = [fragment(i, sequence=True) for i in range(3)]
    for f in fragments:
        system._ingest_fragment(f)
    first, middle, last = fragments
    assert system.edges.get(first.id, middle.id) == 1.0
    ordinary = jaccard_prior(first, last)
    assert system.edges.get(first.id, last.id) == ordinary
    system.companion.store._fragments.remove(middle)
    system.edges.remove_fragment(middle.id)
    assert system.edges.get(first.id, middle.id) == 0.0
    assert system.edges.get(first.id, last.id) == ordinary
    # Actual capacity enforcement counts descriptor bytes, including opaque tags.
    system.companion.store.byte_cap = 0
    system._ingest_fragment(fragment(group=OTHER))
    assert system.companion.store.total_bytes(system.edges) == 0
    assert len(system.edges) == 0


def test_metadata_independent_residual_ties_preserve_topic_gate():
    originals = [make_fragment(f"topic {word}", STAMP, tags=("user:alice", "session:one", "speaker:user"))
                 for word in ("bicycle", "garden", "wall", "tree", "roof", "bench", "color")]
    annotated = [make_fragment(f.content, STAMP, tags=(*f.tags, turn.exchange_tag(turn.new_exchange_id(), "user", 0)))
                 for f in originals]
    base = spread_activation(originals, "bicycle color", ActivationState(residual={f.id: 1.0 for f in originals}))
    current = spread_activation(annotated, "bicycle color", ActivationState(residual={f.id: 1.0 for f in annotated}))
    assert [base[f.id] for f in originals] == [current[f.id] for f in annotated]


def test_legacy_pam_payload_byte_counts_and_duplicate_tag_semantics_are_preserved():
    f = make_fragment("legacy payload", STAMP, tags=("tag", "tag", "speaker:user"))
    artifact = pam_export((f,))
    assert pam_export(pam_import(artifact)) == artifact
    assert fragment_bytes(f) == len(f.content.encode())
    assert turn.evidence_tags(f) == tuple(sorted(f.tags))


def test_canonical_parser_rejects_duplicate_json_keys():
    tag = turn.exchange_tag(GROUP, "user", 0).replace('{"actor":"user",', '{"actor":"tutor","actor":"user",')
    assert turn.parse_turn((tag,)).status == "malformed"


def test_initial_local_graph_rebuild_is_independent_of_import_order(tmp_path):
    fragments = [fragment(i, sequence=True) for i in range(4)]
    forward = system_for_policy("local_association", cjk_tokenizer="unicode")
    reverse = system_for_policy("local_association", cjk_tokenizer="unicode")
    for name, items, system in (("forward", fragments, forward), ("reverse", fragments[::-1], reverse)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(pam_export(items)), encoding="utf-8")
        restore_system_from_path(path, system=system)
    assert forward.edges == reverse.edges
    assert forward.witness_edges == reverse.witness_edges
    assert {f.id for f in forward.companion.store.all()} == {f.id for f in reverse.companion.store.all()}


@pytest.mark.parametrize("boundary", ["perspective", "copresence"])
def test_local_association_does_not_bypass_real_recall_filters(boundary):
    from psychology_base.companion.copresence import CopresenceScope

    system = system_for_policy("local_association", cjk_tokenizer="unicode")
    allowed = ("subject:alice", "entity:story", "witness:alice")
    foreign = ("subject:bob", "entity:story", "witness:bob")
    own = [fragment(i, tags=allowed, text=f"shared bicycle own {i}") for i in range(2)]
    other = [fragment(i, group=OTHER, tags=foreign, text=f"shared bicycle foreign {i}") for i in range(2)]
    for f in (*own, *other):
        system._ingest_fragment(f)
    kwargs = {"scope": ("alice", "story")} if boundary == "perspective" else {"copresence": CopresenceScope(("alice",))}
    recall = system.recall("shared bicycle", **kwargs)
    assert {f.id for f in recall.fragments_used} == {f.id for f in own}
    assert system.edges is not system.witness_edges


@pytest.mark.parametrize("kind", ["companion", "tutor"])
def test_console_adapter_exposes_the_actual_exchange_receipt(kind):
    from psychology_base.applications.emotional.companion_cli import _companion_metadata
    from psychology_base.applications.learning.tutor_cli import _tutor_metadata

    system = CompanionSystem(byte_cap=None)
    response = build_agent(kind, system, lambda *args: "What do you notice?").respond("real input", timestamp=STAMP)
    metadata = (_companion_metadata if kind == "companion" else _tutor_metadata)(response)
    assert metadata["exchange_id"] == response.exchange_id
    assert {turn.parse_turn(f.tags).descriptor.group for f in system.companion.store.all()} == {response.exchange_id}
