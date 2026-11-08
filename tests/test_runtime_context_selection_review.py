"""Review coverage: real source/context plumbing; no generated-answer claims."""
from __future__ import annotations

import copy
import json

import pytest

from psychology_base.applications.emotional.companion_agent import CompanionAgent
from psychology_base.applications.learning.tutor_agent import TutorAgent
from psychology_base.applications.measurement import ScaffoldingTelemetry
from psychology_base.applications.pilot.layout import PilotApp, PilotLayout, UserManifest, pam_latest_path
from psychology_base.companion import reconstruction as memory
from psychology_base.companion.edges import EdgeStore
from psychology_base.companion.pam_export import pam_export
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks import attachment_longitudinal as longitudinal
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks import scaled_context as scaled
from psychology_base.eval.benchmarks import turn_association as association
from psychology_base.llm import MemoryConfig, load_config
from tests.test_cue_context import STAMP, fragment
from tests.test_focused_context_review import public_sources


def write_config(path):
    # Required provider slots are parsed only; these tests make no model calls.
    path.write_text("active_provider: local\nproviders:\n  local:\n"
                    "    simple_model: unused\n    advanced_model: unused\n"
                    "memory:\n  context_selection: scaled_focus\n", encoding="utf-8")


@pytest.mark.parametrize("kind", ["companion", "tutor"])
@pytest.mark.parametrize("absent", [False, True])
def test_pinned_source_reaches_real_application_responder(public_sources, kind, absent):
    source, case, protocol = public_sources["film/test/82/13"]
    base, ids, _ = association.setup(source, case, "scaled_focus", protocol, absent=absent,
                                     arms=scaled.ARMS, system_factory=scaled.system_for_arm)
    system = longitudinal._AuditedSystem(cjk_tokenizer="jieba", byte_cap=None,
                                         fragment_context_selection="scaled_focus")
    for f in base.companion.store.all():
        system._ingest_fragment(f)
    captured = []
    fields = dict(user_id="study", session_id="application-review", companion_system=system,
                  llm_responder=lambda prompt, context, utterance:
                  captured.append(context) or "Response capture control; no quality claim.")
    agent = (CompanionAgent(**fields) if kind == "companion" else
             TutorAgent(**fields, telemetry=ScaffoldingTelemetry(user_id="study")))
    query = shared.message(source, case["query"])["message"]
    response = agent.respond(query, timestamp=STAMP)
    used = system.last_context_fragments
    assert 0 < len(used) <= 6 and len(captured) == 1
    assert {f.id for f in used} <= set(ids.values())
    assert all(f.content in captured[0] for f in used)
    assert system.activation.recall_count == system.witness_state.activation.recall_count == 1
    if absent:
        assert all(tuple(ref) not in ids for ref in case["gold"])
    else:
        expected = {ids[tuple(ref)] for ref in case["gold"]}
        assert expected <= {f.id for f in used}
    assert response.exchange_id


@pytest.mark.parametrize("kind", ["companion", "tutor"])
@pytest.mark.parametrize("restore", [False, True])
def test_actual_yaml_reaches_application_init_and_pam_restore(tmp_path, monkeypatch, kind, restore):
    from psychology_base.applications.emotional import companion_cli
    from psychology_base.applications.learning import tutor_cli
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    monkeypatch.setenv("PSYCHOLOGY_BASE_LLM_CONFIG", str(config_path))
    app = PilotApp.EMOTIONAL_COMPANION if kind == "companion" else PilotApp.LEARNING_TUTOR
    layout = PilotLayout(pilot_root=tmp_path / "pilot", app=app)
    manifest = UserManifest(user_id="alice", app=app, policy_id="S1",
                            enrolled_at=STAMP.isoformat(), age_bracket="26-45", region="US")
    fragments = [fragment(0, "current question"), fragment(1, "answer detail")] if restore else []
    if restore:
        path = pam_latest_path(layout, "alice")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(pam_export(fragments)), encoding="utf-8")
    module = companion_cli if kind == "companion" else tutor_cli
    system = module._resume_or_init_system(layout, "alice", manifest)
    assert system.fragment_context_selection == "scaled_focus"
    assert pam_export(system.companion.store.all()) == pam_export(fragments)
    assert system.witness.store.all() == system.companion.store.all()
    if restore:
        result = system.recall("current question")
        assert {f.id for f in result.fragments_used} == {f.id for f in fragments}
        assert result.metadata["fragment_context_selection"]["name"] == "scaled_focus"


def test_yaml_public_config_type_and_explicit_override(tmp_path):
    path = tmp_path / "config.yaml"
    write_config(path)
    config = load_config(path)
    assert isinstance(config.memory, MemoryConfig)
    assert CompanionSystem.from_llm_config(config).fragment_context_selection == "scaled_focus"
    assert CompanionSystem.from_llm_config(config, fragment_context_selection="identity").fragment_context_selection == "identity"


def test_explicit_lowlevel_identity_preserves_default_reconstruction_and_commits():
    fragments = [fragment(0, "current question"), fragment(1, "answer detail")]
    state = memory.ActivationState()
    memory.reconstruct(fragments, "answer detail", None, state, 0)
    states = [copy.deepcopy(state), copy.deepcopy(state)]
    graphs = [EdgeStore(), EdgeStore()]
    omitted = memory.reconstruct(fragments, "current question", None, states[0], 1, edges=graphs[0])
    explicit = memory.reconstruct(fragments, "current question", None, states[1], 1, edges=graphs[1],
                                  fragment_context_selection="identity")
    assert explicit == omitted and explicit.fragments_used
    assert states[0] == states[1] and graphs[0] == graphs[1]
