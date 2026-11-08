"""Application, renderer and versioned evaluator contracts for allocation."""
import copy
from functools import partial
import importlib.util
import json
from pathlib import Path

import pytest

from psychology_base.applications.context import context_byte_counter, render_context_block
from psychology_base.companion import reconstruction as memory
from psychology_base.companion.pam_export import pam_export
from psychology_base.companion.system import CompanionSystem
from tests.test_context_allocation import MODES, render
from tests.test_cue_context import STAMP, fragment
from tests.test_turn_provenance import build_agent


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("allocation_runner", Path(
        "docs/experiments/phase-9-memory-navigation/run-episode-context-v1.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("app", ["companion", "tutor"])
def test_byte_counter_uses_actual_unicode_and_escape_rules(version, app):
    fs = [fragment(i, text) for i, text in enumerate([
        'quote" and \\ slash\nnewline', '\u4e2d\u6587\u0085\u2028\u2029', ''])]
    count = context_byte_counter(assistant_speaker=app, format_version=version)
    for selected in ([], fs[:1], fs, list(reversed(fs)), fs + fs):
        assert count(selected) == len(render_context_block(selected, assistant_speaker=app,
                                                         format_version=version).encode("utf-8"))


@pytest.mark.parametrize("mode", MODES)
def test_optimized_counter_does_not_change_selection_or_trace(mode):
    fs = [fragment(i, text) for i, text in enumerate(["question", "detail", "result"])]
    snapshot = memory.recall_tokens(fs, "question", "unicode")
    scores = {fs[0].id: 1}
    options = [memory.ResearchSelection(mode, render), memory.ResearchSelection(mode, render,
        byte_counter_factory=partial(context_byte_counter, assistant_speaker="companion"))]
    assert memory.ordered_selection(fs, scores, 6, options[0], snapshot=snapshot) == memory.ordered_selection(
        fs, scores, 6, options[1], snapshot=snapshot)


@pytest.mark.parametrize("kind", ["companion", "tutor"])
@pytest.mark.parametrize("mode", MODES)
def test_actual_application_uses_one_selected_list_before_observation(kind, mode):
    fs = [fragment(i, text) for i, text in enumerate(["question", "detail", "result"])]
    system = CompanionSystem(byte_cap=None, research_selection=memory.ResearchSelection(mode, render))
    for f in fs:
        system._ingest_fragment(f)
    actual, contexts, original = [], [], system.recall
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        actual.append(result)
        return result
    system.recall = capture
    agent = build_agent(kind, system, lambda prompt, context, utterance:
                        contexts.append(context) or "What would you try next?")
    agent.respond("question", timestamp=STAMP)
    assert len(actual) == len(contexts) == 1
    assert contexts[0] == render(actual[0].fragments_used)
    assert all(f in fs for f in actual[0].fragments_used)
    assert system.activation.recall_count == system.witness_state.activation.recall_count == 1


def test_versioned_span_verifier_accepts_two_hops_and_rejects_gap_reverse_or_rebinding(runner):
    fs = [fragment(i, text) for i, text in enumerate(["query", "question", "outcome", "tail"])]
    system = CompanionSystem(byte_cap=None)
    for f in fs:
        system._ingest_fragment(f)
    item = {"app": "companion", "query": "query", "memory": pam_export(fs)}
    trial = runner.trial_system(system, item, "query_span_allocation", runner.ALLOCATION)
    row = {"arm": "query_span_allocation", **runner.execute(trial, item)}
    runner.validate_record(row, item, runner.ALLOCATION)
    assert len(row["trace"]["spans"][0]["ids"]) == 3
    for ids in ([fs[0].id, fs[2].id], [fs[2].id, fs[1].id, fs[0].id], ["invented", fs[1].id]):
        corrupt = copy.deepcopy(row)
        corrupt["trace"]["spans"][0]["ids"] = ids
        corrupt["trace_bytes"] = len(json.dumps(corrupt["trace"], ensure_ascii=False).encode("utf-8"))
        with pytest.raises(ValueError, match="trace"):
            runner.validate_record(corrupt, item, runner.ALLOCATION)


def test_six_arm_spec_has_explicit_additional_query_only_comparison(runner):
    from tests.test_episode_context_run import ref
    spec = runner.ALLOCATION
    refs = {"one": ref(), "two": ref()}
    rows = [{"regime": regime, "arm": arm, "id": identity, "order": order, "repeat": repeat,
             "selected": ["a", "context"] if arm in spec.arms[3:] else ["a"],
             "retained": ["a", "context", "equivalent"]}
            for regime in ("primary", "stress") for arm in spec.arms for identity in refs
            for order in range(3) for repeat in range(2)]
    gates = runner.mechanism_gates(rows, runner.semantic_summary(rows, refs, spec), refs, spec)
    assert gates["query_occurrences"]["passed"]
    assert not gates["query_span_allocation"]["passed"]
    assert any(row["baseline"] == "query_occurrences" and not row["passed"]
               for row in gates["query_span_allocation"]["comparisons"])


def test_new_freeze_has_unchanged_runtime_inputs_and_preimplementation_controls(runner):
    manifest, items, refs = runner.load_packet(runner.ALLOCATION)
    assert len(items) == len(refs) == 32
    assert sum(len(orders) for orders in manifest["input_permutations"].values()) == 92
    assert manifest["protocol"]["expected_recalls"]["total"] == 2712
    assert manifest["protocol"]["paid_requests"] == 0


def test_study_clone_preserves_learned_state_and_uses_constructor_validation(runner):
    fs = [fragment(0, "query"), fragment(1, "detail")]
    system = CompanionSystem(byte_cap=None)
    for f in fs:
        system._ingest_fragment(f)
    system.recall("query")
    before = copy.deepcopy((system.activation, system.edges, system.witness_state, system.witness_edges))
    trial = runner.trial_system(system, {"app": "companion"}, "query_occurrences", runner.ALLOCATION)
    assert (trial.activation, trial.edges, trial.witness_state, trial.witness_edges) == before
    assert trial.activation is not system.activation and trial.edges is not system.edges
    system.consolidation_enabled = True
    with pytest.raises(ValueError, match="Research selection"):
        runner.trial_system(system, {"app": "companion"}, "query_occurrences", runner.ALLOCATION)


def test_large_candidate_pool_obeys_frozen_bounds_and_tokenizes_once(monkeypatch):
    fs = [fragment(0, "background", group=f"background{i}") for i in range(12)]
    fs += [fragment(position, "query" if position == 0 else "detail", group=f"source{i}")
           for i in range(20) for position in range(3)]
    scores = {f.id: 2 if i < 12 else 1 for i, f in enumerate(fs)}
    result, trace = memory.ordered_selection(fs, scores, 6,
        memory.ResearchSelection("query_span_allocation", render),
        snapshot=memory.recall_tokens(fs, "query", "unicode"))
    assert len(trace["candidate_ids"]) == 24
    assert trace["search"]["options"] <= 96 and trace["search"]["expansions"] <= 9216
    assert len(result) <= 6
    calls, original = [], memory.recall_tokens
    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(memory, "recall_tokens", capture)
    memory.reconstruct(fs[:3], "background", None, memory.ActivationState(), 0,
        research_selection=memory.ResearchSelection("query_occurrences", render))
    assert len(calls) == 1


def test_invalid_duplicate_pool_cannot_corrupt_bitset_but_repeated_ingestion_is_idempotent():
    f = fragment(0, "query")
    option = memory.ResearchSelection("query_occurrences", render)
    with pytest.raises(ValueError, match="unique eligible occurrence"):
        memory.ordered_selection([f, f], {f.id: 1}, 6, option,
            snapshot=memory.recall_tokens([f, f], "query", "unicode"))
    system = CompanionSystem(byte_cap=None, research_selection=option)
    system._ingest_fragment(f)
    system._ingest_fragment(f)
    assert system.recall("query").fragments_used == (f,)
