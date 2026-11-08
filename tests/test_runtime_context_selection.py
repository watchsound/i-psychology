"""Runtime integration contracts; scripted replies are software controls only."""
from __future__ import annotations

import copy
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path

import pytest

from psychology_base.companion import reconstruction as memory
from psychology_base.companion.pam_export import pam_export
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks import attachment_longitudinal as longitudinal
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks import focused_context as focused
from psychology_base.eval.benchmarks import graph_priors as graph
from psychology_base.eval.benchmarks import recall_carryover as carry
from psychology_base.eval.benchmarks import scaled_context as scaled
from psychology_base.eval.benchmarks import turn_association as association
from psychology_base.llm.config import _parse_config
from tests.test_cue_context import STAMP, fragment
from tests.test_focused_context_review import public_sources
from tests.test_turn_provenance import build_agent


EVIDENCE = Path(__file__).resolve().parents[1] / "docs/experiments/phase-9-runtime-context-selection"


def configured(mode="scaled_focus"):
    return _parse_config({"active_provider": "local", "providers": {
        "local": {"simple_model": "unused", "advanced_model": "unused"}},
        "memory": {"context_selection": mode}})


@pytest.mark.parametrize("mode", [None, True, 1, [], {}, "", "unknown"])
def test_invalid_selector_fails_before_state_creation(mode):
    system = CompanionSystem.__new__(CompanionSystem)
    with pytest.raises(ValueError, match="context.selection|context_selection"):
        system.__init__(fragment_context_selection=mode)
    assert not hasattr(system, "companion")


@pytest.mark.parametrize("mode", [None, True, [], "unknown"])
def test_config_does_not_coerce_invalid_selector(mode):
    with pytest.raises(ValueError, match="context.selection|context_selection"):
        CompanionSystem.from_llm_config(configured(mode))


def test_default_config_override_readonly_and_detached_metadata():
    assert CompanionSystem().fragment_context_selection == "identity"
    system = CompanionSystem.from_llm_config(configured())
    assert system.fragment_context_selection == "scaled_focus"
    assert CompanionSystem.from_llm_config(configured(), fragment_context_selection="identity").fragment_context_selection == "identity"
    with pytest.raises(AttributeError):
        system.fragment_context_selection = "identity"
    identity = system.inspect()["fragment_context_selection"]
    assert identity == {"name": "scaled_focus", "version": "scaled-focus-v1"}
    identity["name"] = "tampered"
    assert system.inspect()["fragment_context_selection"]["name"] == "scaled_focus"
    assert copy.deepcopy(system).fragment_context_selection == "scaled_focus"


def test_identity_configuration_only_changes_the_versioned_han_seed_policy():
    baseline = json.loads((EVIDENCE / "before.json").read_text(encoding="utf-8"))
    system = CompanionSystem(top_k=6, byte_cap=longitudinal.BYTE_CAP)
    actual = longitudinal.scorer_configuration(system)
    prior = baseline["identity_configuration"]
    assert actual['fragment_recall_policy'] == longitudinal.RECALL_SELECTION_POLICY
    assert actual['fragment_recall_policy'] != prior['fragment_recall_policy']
    assert {k:v for k,v in actual.items() if k != 'fragment_recall_policy'} == {
        k:v for k,v in prior.items() if k != 'fragment_recall_policy'}


@pytest.mark.parametrize("state", carry.STATES)
@pytest.mark.parametrize("absent", [False, True])
def test_runtime_matches_real_pinned_source_binding_and_trace(public_sources, state, absent):
    source, case, protocol = public_sources["film/test/82/13"]
    legacy, ids, _ = association.setup(
        source, case, "metadata_only", protocol, absent=absent,
        arms=scaled.ARMS, system_factory=scaled.system_for_arm)
    carry.bind_reconstruction(legacy, shared._bind(memory.reconstruct,
        _selection_scores=memory.scaled_focus_scores), witness_reconstruct=memory.reconstruct)
    runtime = CompanionSystem(top_k=6, byte_cap=None, cjk_tokenizer="jieba",
                              fragment_context_selection="scaled_focus")
    for f in legacy.companion.store.all():
        runtime._ingest_fragment(f)
    assert runtime.edges == legacy.edges
    systems = [carry.prepare_recall(source, case, s, "production", state, focus_current=True)
               for s in (legacy, runtime)]
    results = [s.recall(cue, scope=("study", "memory"),
        focus_cue=shared.message(source, case["query"])["message"]) for s, cue in systems]
    if not absent:
        gold_ids = {ids[tuple(ref)] for ref in case["gold"]}
        assert gold_ids <= {f.id for f in results[1].fragments_used}
    assert results[0].content == results[1].content
    assert results[0].fragments_used == results[1].fragments_used
    for field in ("activation", "edges", "witness_edges", "witness_state"):
        assert getattr(systems[0][0], field) == getattr(systems[1][0], field)
    if not absent:  # The existing graph tracer requires the gold-containing pool.
        before = runtime.inspect()
        expected = graph.trace_recall(source, case, legacy, ids, state, focus_current=True)
        actual = graph.trace_recall(source, case, runtime, ids, state, focus_current=True)
        assert actual == expected
        assert runtime.inspect() == before


def test_runtime_commits_once_from_selected_pool_and_keeps_witness_independent():
    fs = [fragment(0, "current question", group="focus"), fragment(1, "answer detail", group="focus")]
    fs += [fragment(0, f"history old broad topic {i}", group=f"d{i}") for i in range(6)]
    system = CompanionSystem(top_k=2, byte_cap=None, fragment_context_selection="scaled_focus")
    baseline = CompanionSystem(top_k=2, byte_cap=None)
    for s in (system, baseline):
        for f in fs:
            s._ingest_fragment(f)
    edges = copy.deepcopy(system.edges)
    cue, focus = "history old broad topic current question", "current question"
    result = system.recall(cue, scope=("alice", "memory"), focus_cue=focus)
    baseline.recall(cue, scope=("alice", "memory"), focus_cue=focus)
    assert fs[1] in result.fragments_used
    selected = {f.id for f in result.fragments_used}
    for a, b in combinations(fs, 2):
        delta = .05 if {a.id, b.id} <= selected else 0
        assert system.edges.get(a.id, b.id) == pytest.approx(edges.get(a.id, b.id) + delta)
    assert system.activation.recall_count == system.witness_state.activation.recall_count == 1
    assert system.witness_state.activation == baseline.witness_state.activation
    assert system.witness_edges == baseline.witness_edges
    assert result.metadata["fragment_context_selection_applied"] is True
    assert system.recall(cue, scope=("bob", "memory")).fragments_used == ()


def test_invalid_low_level_selection_does_not_decay_or_commit():
    state = memory.ActivationState()
    state.residual["sentinel"] = .5
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="context.selection|context_selection"):
        memory.reconstruct([], "question", None, state, 0, fragment_context_selection="unknown")
    assert state == before


@pytest.mark.parametrize("arm", scaled.ARMS)
def test_factory_rejects_conflicting_mode_and_retains_one_normal_runtime_primary(arm):
    expected = "scaled_focus" if arm == "scaled_focus" else "identity"
    other = "identity" if expected == "scaled_focus" else "scaled_focus"
    with pytest.raises(ValueError, match="selector|selection"):
        scaled.system_for_arm(arm, cjk_tokenizer="unicode", fragment_context_selection=other)
    system = scaled.system_for_arm(arm, cjk_tokenizer="unicode")
    if arm == "scaled_focus":
        assert system.recall.__func__ is CompanionSystem.recall
        assert system.fragment_context_selection == expected


@pytest.mark.parametrize("kind", ["companion", "tutor"])
def test_real_agents_render_one_actual_recall_before_observing_current_input(kind):
    class Audit(CompanionSystem):
        def recall(self, *args, **kwargs):
            self.before_recall = tuple(self.companion.store.all())
            result = super().recall(*args, **kwargs)
            self.actual_recall = result
            self.calls = getattr(self, "calls", 0) + 1
            return result
    system = Audit.from_llm_config(configured(), byte_cap=None)
    expected_fragments = (fragment(0, "current question"), fragment(1, "answer detail"))
    for f in expected_fragments:
        system._ingest_fragment(f)
    captured = []
    agent = build_agent(kind, system, lambda prompt, context, utterance:
                        captured.append(context) or "What would you try next?")
    response = agent.respond("current question today", timestamp=STAMP)
    assert system.calls == 1
    assert len(system.before_recall) == 2 and len(system.companion.store.all()) == 4
    assert system.actual_recall.fragments_used == expected_fragments
    assert system.actual_recall.metadata["fragment_context_selection"]["name"] == "scaled_focus"
    assert len(captured) == 1
    rendered = list(map(json.loads, captured[0].splitlines()))
    assert [{k: v for k, v in row.items() if k != 'source_order'} for row in rendered] == [
        {'speaker': None, 'session': 'one', 'text': text} for text in ('current question', 'answer detail')]
    assert [row['source_order']['position'] for row in rendered] == [0, 1]
    assert rendered[0]['source_order']['group'] == rendered[1]['source_order']['group']
    assert response.exchange_id and "recall_receipt" not in asdict(response)


def short_manifest():
    data = longitudinal.build_trajectory("supportive-en").model_dump(mode="json")
    data["sessions"] = data["sessions"][:3]
    return longitudinal.TrajectoryManifest.model_validate(data)


def test_real_v5_checkpoint_cannot_resume_under_the_v6_memory_policy():
    manifest = longitudinal.TrajectoryManifest.model_validate_json(
        (EVIDENCE / "legacy/manifest.json").read_text(encoding="utf-8"))
    path = EVIDENCE / "legacy/checkpoint.json"
    original = path.read_bytes()
    payload = json.loads(original)['payload']
    assert payload['configuration']['fragment_recall_policy'] != longitudinal.RECALL_SELECTION_POLICY
    for mode in ('identity', 'scaled_focus'):
        with pytest.raises(ValueError, match="configuration mismatch"):
            longitudinal.load_checkpoint(path, manifest, fragment_context_selection=mode)
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["identity", "scaled_focus"])
def test_checkpoint_roundtrip_reports_actual_policy_and_rejects_wrong_expected_policy(tmp_path, monkeypatch, mode):
    manifest, path = short_manifest(), tmp_path / "checkpoint.json"
    result = longitudinal.run_trajectory(manifest, checkpoint_path=path, stop_after=2,
                                         fragment_context_selection=mode)
    configuration = result.report()["configuration"]
    payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
    assert payload["schema_version"] == "attachment-checkpoint-v3"
    assert payload["configuration"] == configuration == longitudinal.scorer_configuration(result.system)
    assert ("fragment_context_selection" in configuration) == (mode == "scaled_focus")
    _, _, restored = longitudinal.load_checkpoint(path, manifest, fragment_context_selection=mode)
    assert restored.fragment_context_selection == mode
    assert pam_export(restored.companion.store.all()) == pam_export(result.system.companion.store.all())
    assert result.report()["memory_checks"]["runtime_equivalence_claimed"] is False
    original = Path.read_text
    def guard(path_obj, *args, **kwargs):
        assert not path_obj.name.startswith("memory-"), "PAM opened before configuration check"
        return original(path_obj, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guard)
    wrong = "identity" if mode == "scaled_focus" else "scaled_focus"
    with pytest.raises(ValueError, match="configuration mismatch"):
        longitudinal.load_checkpoint(path, manifest, fragment_context_selection=wrong)


@pytest.mark.parametrize("corruption", ["missing", "version", "name", "tokenizer"])
def test_scaled_checkpoint_requires_exact_metadata_before_ingestion(tmp_path, monkeypatch, corruption):
    manifest, path = short_manifest(), tmp_path / "checkpoint.json"
    longitudinal.run_trajectory(manifest, checkpoint_path=path, stop_after=1,
                                fragment_context_selection="scaled_focus")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    cfg = envelope["payload"]["configuration"]
    if corruption == "missing":
        cfg.pop("fragment_context_selection")
    elif corruption == "tokenizer":
        cfg["tokenizer_identity"] = {}
    else:
        cfg["fragment_context_selection"][corruption] = "incorrect"
    envelope["sha256"] = longitudinal.digest(envelope["payload"])
    path.write_text(json.dumps(envelope), encoding="utf-8")
    monkeypatch.setattr(CompanionSystem, "_ingest_fragment", lambda *a: pytest.fail("ingested mismatched checkpoint"))
    with pytest.raises(ValueError, match="configuration mismatch"):
        longitudinal.load_checkpoint(path, manifest, fragment_context_selection="scaled_focus")


def test_hierarchical_bypass_preserves_actual_path_and_reports_applicability(tmp_path):
    # Existing deterministic provider isolates routing, not model-quality evidence.
    from tests.test_companion_consolidation import _router
    router, _ = _router(tmp_path, "")
    systems = [CompanionSystem(consolidation_enabled=True, consolidation_router=router,
                fragment_context_selection=mode) for mode in ("identity", "scaled_focus")]
    for system in systems:
        system._ingest_fragment(fragment(0, "current question"))
    results = [system.recall("current question") for system in systems]
    assert results[0].content == results[1].content
    assert results[0].fragments_used == results[1].fragments_used
    for result, system in zip(results, systems):
        assert result.fragments_used == (fragment(0, "current question"),)
        assert "current question" in result.content
        assert result.metadata["path"] == "hierarchical"
        assert result.metadata["fragment_context_selection_applied"] is False
        assert result.metadata["fragment_context_selection"]["name"] == system.fragment_context_selection
        assert system.activation.recall_count == 0


def test_research_binding_cannot_be_overridden_or_misreported():
    bound = shared._bind(memory.reconstruct, _selection_scores=memory.focused_context_scores)
    state = memory.ActivationState()
    with pytest.raises(ValueError, match="bound research selector"):
        bound([], "question", None, state, 0, fragment_context_selection="scaled_focus")
    assert state.recall_count == 0
    system = focused.system_for_arm("focus_companion", cjk_tokenizer="unicode")
    assert system.recall("question").metadata["fragment_context_selection"]["name"] == "research:focused_context_scores"
    runtime = CompanionSystem(fragment_context_selection="scaled_focus")
    with pytest.raises(ValueError, match="Research binding"):
        carry.bind_reconstruction(runtime, bound)


def test_scaled_option_still_requires_real_requested_tokenizer(monkeypatch):
    import psychology_base.i18n as i18n
    monkeypatch.setattr(i18n, "_load_jieba", lambda: (None, {}, "not installed"))
    system = CompanionSystem.__new__(CompanionSystem)
    with pytest.raises(ValueError, match="jieba.*not installed"):
        system.__init__(cjk_tokenizer="jieba", fragment_context_selection="scaled_focus")
    assert not hasattr(system, "companion")


def test_roundtrip_uses_original_selector_for_actual_restored_recall(monkeypatch):
    system = CompanionSystem(byte_cap=None, fragment_context_selection="scaled_focus")
    system._ingest_fragment(fragment(0, "current question"))
    seen, original = [], CompanionSystem.recall
    def audited(self, *args, **kwargs):
        seen.append(self.fragment_context_selection)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CompanionSystem, "recall", audited)
    checks = longitudinal.memory_round_trip_checks(system, "current question", "alice")
    assert seen == ["scaled_focus"]
    assert checks["fragment_set_preserved"] and checks["sentinel_recall_after_restore"]
    assert checks["runtime_equivalence_claimed"] is False


def test_scaled_crisis_path_skips_recall_and_observation(monkeypatch):
    from psychology_base.applications.safety import CrisisEscalation
    system = CompanionSystem(fragment_context_selection="scaled_focus")
    monkeypatch.setattr(system, "recall", lambda *a: pytest.fail("crisis performed recall"))
    agent = build_agent("companion", system, lambda *a: pytest.fail("crisis called responder"))
    agent.crisis_escalation = CrisisEscalation(handoff_sink=lambda incident: None)
    response = agent.respond("I want to kill myself", timestamp=STAMP)
    assert response.handoff_triggered and response.exchange_id is None
    assert not system.companion.store.all()


@pytest.mark.parametrize("mode", ["identity", "scaled_focus"])
def test_real_fresh_process_resume_forwards_selector(tmp_path, mode):
    from benchmarks.scripts import phase9_attachment_longitudinal_run as cli
    manifest = short_manifest()
    path = tmp_path / "manifest.json"
    longitudinal.atomic_write_json(path, manifest.model_dump(mode="json"))
    uninterrupted = longitudinal.run_trajectory(manifest, fragment_context_selection=mode).report()
    verdict = cli._verify_resume(manifest, path, tmp_path, uninterrupted, fragment_context_selection=mode)
    assert verdict["passed"] and verdict["fresh_process"]
    assert verdict["checks"]["configuration"]
    assert verdict["comparison_kind"] == "fixed_transcript_attachment_bookkeeping"
    resumed = json.loads((tmp_path / "resume-verification" / manifest.trajectory_id / "report.json").read_text(encoding="utf-8"))
    assert resumed["configuration"] == uninterrupted["configuration"]


@pytest.mark.parametrize("raw", [None, False, [], "scaled_focus"])
def test_memory_section_requires_mapping(raw):
    with pytest.raises(ValueError, match="memory.*mapping"):
        _parse_config({"active_provider": "local", "providers": {
            "local": {"simple_model": "unused", "advanced_model": "unused"}}, "memory": raw})
