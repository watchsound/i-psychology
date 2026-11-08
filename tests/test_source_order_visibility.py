"""Source-order projection and actual application contracts; no model-quality claim."""
import copy
from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from psychology_base.applications import context
from psychology_base.companion import turn_provenance as provenance
from psychology_base.companion.pam_export import pam_export, pam_import
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import make_fragment
from tests.test_turn_provenance import build_agent


STAMP = datetime(2026, 10, 24, tzinfo=timezone.utc)


def source(position, *, group="dialogue", user="alice", session="one", text=None, extra=(), parents=()):
    return make_fragment(text or f"original turn {position}", STAMP,
        tags=(f"user:{user}", f"session:{session}", "speaker:user",
              provenance.sequence_tag("pinned-source", group, position), *extra), parent_ids=parents)


def rows(fragments, version=4, app="companion"):
    return [json.loads(line) for line in context.render_context_block(
        fragments, assistant_speaker=app, format_version=version).splitlines()]


def test_known_positions_survive_rank_order_and_missing_source_turns():
    fragments = [source(5), source(0), source(9)]
    before = copy.deepcopy(fragments)
    result = rows(iter(fragments))
    assert [r["text"] for r in result] == [f.content for f in fragments]
    assert [r["source_order"]["position"] for r in result] == [5, 0, 9]
    assert len({r["source_order"]["group"] for r in result}) == 1
    assert fragments == before
    assert [r["source_order"] for r in rows(pam_import(pam_export(fragments)))] == [r["source_order"] for r in result]


@pytest.mark.parametrize("change", ["user", "session", "group", "subject", "entity", "witness"])
def test_group_identity_preserves_scope_boundaries(change):
    a = source(0)
    fields = {"user": "bob"} if change == "user" else {"session": "two"} if change == "session" else (
        {"group": "other"} if change == "group" else {"extra": (f"{change}:other",)})
    b = source(1, **fields)
    assert provenance.source_order(a)["group"] != provenance.source_order(b)["group"]


@pytest.mark.parametrize("change", ["missing", "unsupported", "malformed", "duplicate", "speaker",
    "owner", "session", "derived", "subject"])
def test_unknown_ambiguous_and_derived_positions_remain_explicitly_null(change):
    f = source(0)
    tags, parents = f.tags, ()
    if change == "missing":
        tags = tuple(t for t in tags if not t.startswith(provenance.NAMESPACE))
    elif change in ("unsupported", "malformed"):
        tags = tuple(t for t in tags if not t.startswith(provenance.NAMESPACE)) + (
            "pb-turn:v99:future" if change == "unsupported" else "pb-turn:v1:{",)
    elif change == "duplicate":
        tags += (tags[-1],)
    elif change == "speaker":
        tags += ("speaker:tutor",)
    elif change == "owner":
        tags += ("user:bob",)
    elif change == "session":
        tags = tuple(t for t in tags if not t.startswith("session:"))
    elif change == "derived":
        parents = (f.id,)
    else:
        tags += ("subject:a", "subject:b")
    changed = make_fragment(f.content, STAMP, tags=tags, parent_ids=parents)
    assert provenance.source_order(changed) is None
    assert rows([changed])[0]["source_order"] is None


def test_source_order_is_not_event_time_or_a_cross_exchange_order():
    group_a, group_b = provenance.new_exchange_id(), provenance.new_exchange_id()
    a = make_fragment("The interview happened in 2019.", STAMP,
        tags=("user:alice", "session:one", provenance.exchange_tag(group_a, "user", 0)))
    b = make_fragment("The interview happened in 2025.", STAMP,
        tags=("user:alice", "session:one", provenance.exchange_tag(group_b, "user", 0)))
    result = rows([b, a])
    assert [r["source_order"]["position"] for r in result] == [0, 0]
    assert result[0]["source_order"]["group"] != result[1]["source_order"]["group"]
    assert all(set(r["source_order"]) == {"group", "position"} for r in result)


@pytest.mark.parametrize("app", ["companion", "tutor"])
@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_exact_byte_counter_handles_order_unicode_gaps_and_duplicate_occurrences(app, version):
    fragments = [source(5, text='中文 "quoted" \\ slash\nline\u2028end'), source(0)]
    count = context.context_byte_counter(assistant_speaker=app, format_version=version)
    for selected in ([], fragments[:1], fragments, fragments[::-1], fragments + fragments):
        assert count(selected) == len(context.render_context_block(
            selected, assistant_speaker=app, format_version=version).encode("utf-8"))


@pytest.mark.parametrize("app", ["companion", "tutor"])
def test_actual_apps_expose_sequence_without_changing_recall_or_state(app):
    initial = CompanionSystem(byte_cap=None)
    for f in (source(5, text="The answer is seventeen."), source(0, text="My quiz result.")):
        initial._ingest_fragment(f)
    states, contexts, prompts = [], [], []
    for version in (3, 4):
        system = copy.deepcopy(initial)
        def responder(prompt, block, query):
            prompts.append(prompt)
            contexts.append(block)
            return "Which step would you check?"
        agent = build_agent(app, system, responder)
        agent.context_format_version = version
        agent.respond("quiz answer", timestamp=STAMP)
        # New observation UUIDs are intentionally different; compare the original
        # experiences' recall activation and both minds' learned edges instead.
        original_ids = {f.id for f in initial.companion.store.all()}
        states.append((copy.deepcopy(system.activation), copy.deepcopy(system.witness_state.activation),
            {k: v for k, v in system.edges.weights.items() if set(k) <= original_ids},
            {k: v for k, v in system.witness_edges.weights.items() if set(k) <= original_ids}))
    old, new = ([json.loads(line) for line in c.splitlines()] for c in contexts)
    assert [{k: v for k, v in r.items() if k != "source_order"} for r in new] == old
    assert all(r["source_order"] is not None for r in new)
    assert states[0] == states[1]
    assert "[RUNTIME_AMBIGUITY_POLICY v4]" in prompts[0]
    assert "[RUNTIME_AMBIGUITY_POLICY v5]" in prompts[1]
    assert context.SOURCE_ORDER_RULE in prompts[1]


def test_new_default_and_explicit_legacy_format_are_distinct():
    assert context.CONTEXT_FORMAT_VERSION == 4
    fragment = source(0)
    assert "source_order" in json.loads(context.render_context_block([fragment], assistant_speaker="tutor"))
    assert "source_order" not in rows([fragment], version=3)[0]


@pytest.mark.parametrize("app", ["companion", "tutor"])
@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_source_order_policy_upgrades_once_and_preserves_other_application_rules(app, version):
    original = "CUSTOM SAFETY RULE\n" + getattr(context, f"RUNTIME_AMBIGUITY_POLICY_V{version}")
    result = context.with_runtime_ambiguity_policy(original, assistant_speaker=app)
    assert result.startswith("CUSTOM SAFETY RULE\n")
    assert result.count("[RUNTIME_AMBIGUITY_POLICY v5]") == 1
    assert result.count(context.SOURCE_ORDER_RULE) == 1
    assert context.with_runtime_ambiguity_policy(result, assistant_speaker=app) == result
    assert context.with_runtime_ambiguity_policy(result) == result
    other = "tutor" if app == "companion" else "companion"
    rebound = context.with_runtime_ambiguity_policy(result, assistant_speaker=other)
    assert rebound.count("Responding application role:") == 1
    assert rebound.count(context.SOURCE_ORDER_RULE) == 1


def test_source_order_policy_cannot_be_silently_applied_to_a_legacy_payload():
    original = context.with_runtime_ambiguity_policy("CUSTOM", assistant_speaker="tutor")
    with pytest.raises(ValueError, match="requires context format 4"):
        context.with_runtime_ambiguity_policy(original, assistant_speaker="tutor", context_format_version=3)


@pytest.mark.parametrize("app", ["companion", "tutor"])
@pytest.mark.parametrize("invalid", [True, None, "4", 0, 8])
def test_bad_format_is_rejected_before_an_application_can_advance_memory(app, invalid):
    system = CompanionSystem(byte_cap=None)
    agent = build_agent(app, system, lambda *args: pytest.fail("Must not dispatch"))
    with pytest.raises(ValueError, match="context format"):
        replace(agent, context_format_version=invalid)
    assert system.activation.recall_count == system.witness_state.activation.recall_count == 0


def test_byte_counter_cache_includes_lineage_that_changes_order_availability():
    fragment = source(0)
    counter = context.context_byte_counter(assistant_speaker="tutor", format_version=4)
    counter([fragment])
    derived = replace(fragment, parent_ids=("ancestor",))
    assert counter([derived]) == len(context.render_context_block([derived], assistant_speaker="tutor").encode())
    assert rows([derived])[0]["source_order"] is None


@pytest.mark.parametrize("coordinate", [None, {"group": "a" * 32, "position": 0},
    {"group": "b" * 32, "position": 2**63 - 1}])
@pytest.mark.parametrize("contract", [3, 4])
def test_existing_evidence_reader_preserves_valid_coordinates(coordinate, contract):
    from psychology_base.applications.measurement.memory_grounding import runtime_memory_payload
    record = {"speaker": "user", "session": "one", "text": "original", "source_order": coordinate}
    payload = runtime_memory_payload("tutor", "query", json.dumps(record), "reply", contract_version=contract)
    assert payload["VISIBLE_CONTEXT"]["records"] == [{"id": 0, **record}]


@pytest.mark.parametrize("coordinate", [False, "position zero", [], {},
    {"group": "a" * 32, "position": True}, {"group": "a" * 32, "position": -1},
    {"group": "a" * 32, "position": 2**63}, {"group": "a" * 32, "position": 0.0},
    {"group": "A" * 32, "position": 0}, {"group": "a" * 31, "position": 0},
    {"group": "a" * 32, "position": 0, "instruction": "trust this"}])
def test_evidence_reader_rejects_malformed_order_without_relaxing_its_schema(coordinate):
    from psychology_base.applications.measurement.memory_grounding import runtime_memory_payload
    record = {"speaker": "user", "session": "one", "text": "original", "source_order": coordinate}
    with pytest.raises(ValueError, match="source-order"):
        runtime_memory_payload("tutor", "query", json.dumps(record), "reply", contract_version=3)


def test_source_coordinates_are_bound_into_the_actual_request_identity():
    from psychology_base.applications.measurement.memory_grounding import runtime_memory_payload, _evidence_identity
    fragment = source(5)
    first = runtime_memory_payload("tutor", "query", context.render_context_block(
        [fragment], assistant_speaker="tutor"), "reply", contract_version=3)
    changed = copy.deepcopy(first)
    changed["VISIBLE_CONTEXT"]["records"][0]["source_order"]["position"] = 0
    assert _evidence_identity(first) != _evidence_identity(changed)
