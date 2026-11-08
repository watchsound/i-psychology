"""Stateful carryover contracts and actual-source experiment integrity."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from benchmarks.scripts import phase9_carryover_run as cli
from psychology_base.companion import reconstruction as memory
from psychology_base.companion.edges import EdgeStore
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation, make_fragment
from psychology_base.eval.benchmarks import recall_carryover as study
from psychology_base.i18n import tokenizer_identity


@pytest.fixture(scope="module")
def real_backend():
    pytest.importorskip("jieba", reason="Optional real jieba backend unavailable")
    assert tokenizer_identity("jieba")["module_version"] == "0.42.1"


@pytest.fixture(scope="module")
def controls():
    return json.loads((cli.FIXTURES / "controls.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen_sources():
    protocol = json.loads((cli.FIXTURES / "protocol.json").read_text(encoding="utf-8"))
    assert study.shared.digest(cli.FIXTURES / "catalog.json") == protocol["catalog_sha256"]
    assert study.shared.digest(cli.FIXTURES / "controls.json") == protocol["controls_sha256"]
    if not all((cli.ROOT / ".data/kdconv/data" / f"{key}.json").exists() for key in protocol["sources"]):
        pytest.skip("Optional pinned public KdConv sources not provisioned")
    sources = study.load_sources(cli.ROOT / ".data/kdconv/data", protocol)
    catalog = json.loads((cli.FIXTURES / "catalog.json").read_text(encoding="utf-8"))
    return sources, catalog, protocol


def test_idf_rewards_discriminating_words_and_is_order_independent():
    documents = [("target", {"the", "platform"}), ("a", {"the", "garden"}), ("b", {"the", "garden"})]
    result = study.idf_weights(documents, {"the", "platform"})
    assert result["target"] == pytest.approx(2 + math.log(2))
    assert result["a"] == result["b"] == 1
    assert study.idf_weights(list(reversed(documents)), {"platform", "the"}) == result
    assert study.idf_weights([], {"missing"}) == {}


def test_production_suppresses_unrelated_carryover_without_erasing_live_state():
    fragments = [make_fragment("garden plants", study.shared.BASE)] + [
        make_fragment(text, study.shared.BASE + study.shared.timedelta(minutes=i + 1))
        for i, text in enumerate(("train platform", "astronomy telescope", "ceramic boats",
                                  "piano sonata", "mountain walking", "bicycle repair", "library books"))]
    state = memory.ActivationState(residual={fragments[0].id: 1.0})
    before = copy.deepcopy(state)
    switched = memory.spread_activation(fragments, "train platform", state, edges=EdgeStore())
    assert switched[fragments[0].id] == 0.0
    assert switched[fragments[1].id] == 1.0
    assert state == before
    continued = memory.spread_activation(fragments, "why", state, edges=EdgeStore())
    assert continued[fragments[0].id] == 1.0


def test_supported_cue_novelty_and_common_words_do_not_drop_followups():
    docs = [("bike", {"bicycle", "blue"})] + [(str(i), {"color", "classroom"}) for i in range(7)]
    residual = {"bike": 1.0}
    assert memory._residual_gate({"bike": 1.0}, residual, docs, {"bicycle"}) == 1.0
    assert memory._residual_gate({str(i): 1 / 7 for i in range(7)}, residual, docs, {"color"}) == 1.0
    assert memory._residual_gate({}, residual, docs, {"unknown"}) == 1.0


def test_disabling_fresh_seeding_does_not_discard_the_state_budget():
    old = make_fragment("garden plants", study.shared.BASE)
    new = make_fragment("train platform", study.shared.BASE + study.shared.timedelta(minutes=1))
    state = memory.ActivationState(residual={old.id: 1.0})
    result = memory.spread_activation([old, new], "train platform", state, edges=EdgeStore(), seed_boost=0)
    assert result == {old.id: 1.0, new.id: 0.0}


def test_cold_seeding_remains_count_based_with_the_same_budget():
    a = make_fragment("guide dog platform", study.shared.BASE)
    b = make_fragment("platform", study.shared.BASE + study.shared.timedelta(minutes=1))
    values = memory.spread_activation([a, b], "guide dog platform", memory.ActivationState(), edges=EdgeStore())
    assert values == {a.id: 0.75, b.id: 0.25}


def test_residual_focus_ties_do_not_depend_on_mapping_insertion_order():
    docs = [(str(i), {"familiar"} if i == 6 else {"garden"}) for i in range(8)]
    residual = {str(i): 1.0 for i in range(7)}
    forward = memory._residual_gate({"6": 1.0}, residual, docs, {"familiar"})
    reverse = memory._residual_gate({"6": 1.0}, dict(reversed(list(residual.items()))), docs, {"familiar"})
    assert forward == reverse == 0.0  # IDs 0..5 are the canonical six-way focus.


@pytest.mark.parametrize("familiar,expected", [(True, 1.0), (False, 0.0)])
def test_gate_half_information_boundary_and_strict_document_frequency(familiar, expected):
    docs = [("a", {"old", "common"}), ("b", {"new", "common"}), ("c", set()), ("d", set())]
    cue = {"old", "new", "common"} if familiar else {"new", "common"}
    # 'common' occurs in exactly half the pool and cannot signal a switch.
    assert memory._residual_gate({"b": 1.0}, {"a": 1.0}, docs, cue) == expected


@pytest.mark.parametrize("seed,residual,overlap", [
    ({"a": 1.0}, {"a": 7.0}, 1.0),
    ({"a": 1.0}, {"b": 7.0}, 0.0),
    ({"a": 0.5, "b": 0.5}, {"a": 1.0}, 0.5),
    ({}, {"a": 1.0}, 1.0), ({"a": 1.0}, {}, 1.0),
])
def test_carryover_bounds_alignment_and_implicit_followup(seed, residual, overlap):
    before = copy.deepcopy((seed, residual))
    assert study.overlap_carryover(seed, residual) == pytest.approx(overlap)
    assert study.conservative_carryover(seed, residual) == pytest.approx(0.5 + 0.5 * overlap)
    assert (seed, residual) == before
    assert study.overlap_carryover({k: v * 9 for k, v in seed.items()},
                                   {k: v * 3 for k, v in residual.items()}) == pytest.approx(overlap)


def test_coherent_catalog_is_source_backed_and_excludes_missing_answers(frozen_sources):
    sources, catalog, protocol = frozen_sources
    assert study.build_catalog(sources, protocol)["cases"] == catalog["cases"]
    assert len(catalog["cases"]) == 36
    assert {c["source"] for c in catalog["cases"] if c["split"] == "development"} == {"music/dev"}
    assert {c["source"] for c in catalog["cases"] if c["split"] == "evaluation"} == {"music/test", "film/test"}
    for case in catalog["cases"]:
        source = sources[case["source"]]
        for field in ("history", "absent_history"):
            history = case[field]
            assert len(history) == len(set(map(tuple, history))) == 24
            assert len({ref[0] for ref in history}) == 3
            for di in {ref[0] for ref in history}:
                turns = [ref[1] for ref in history if ref[0] == di]
                assert turns == sorted(turns)
        assert case["answer"] in case["history"] and case["answer"] not in case["absent_history"]
        start, end = case["span"]
        assert study.shared.message(source, case["answer"])["message"][start:end] == case["fact"]["attrvalue"]


@pytest.mark.parametrize("policy,failed", [
    ("baseline", set()), ("idf", set()), ("conservative", set()), ("novel_cue", set()), ("production", set()),
    ("cue_gate", {"implicit-en"}), ("cue_idf", {"implicit-en"}),
    ("reset", {"implicit-en", "implicit-zh", "no-overlap-en", "no-overlap-zh"}),
])
def test_frozen_controls_detect_topic_switch_and_followup_regressions(controls, real_backend, policy, failed):
    results = study.control_results(controls, policy)
    assert {r["case"] for r in results if not r["passed"]} == failed


def test_same_real_policy_runs_in_companion_and_witness(real_backend):
    original_spread = memory.spread_activation
    system = study.bind_policy(CompanionSystem(cjk_tokenizer="jieba"), "conservative")
    globals_ = system.recall.__func__.__globals__
    companion = globals_["reconstruct"].__globals__["spread_activation"]
    second = globals_["witness_couple"].__globals__["witness_second_perspective"]
    witness_spread = second.__globals__["reconstruct"].__globals__["spread_activation"]
    assert companion is witness_spread
    assert companion.__code__ is original_spread.__code__
    assert companion.__globals__["_cue_activation"] is study.idf_weights
    assert memory.spread_activation is original_spread
    with pytest.raises(ValueError, match="Unknown carryover"):
        study.bind_policy(system, "unknown")


@pytest.mark.parametrize("policy", study.POLICIES)
def test_each_policy_keeps_mass_bounded_and_drops_out_of_scope_state(policy, real_backend):
    system = study.bind_policy(CompanionSystem(byte_cap=None, cjk_tokenizer="jieba"), policy)
    for i, text in enumerate(("平台需要维修。", "我的火车从七号站台出发。", "明天恢复天文观测。")):
        system.observe(Observation(study.shared.BASE + study.shared.timedelta(minutes=i), "user", text))
    spread = system.recall.__func__.__globals__["reconstruct"].__globals__["spread_activation"]
    fragments = system.companion.store.all()
    state = memory.ActivationState(residual={f.id: 1e8 for f in fragments} | {"foreign": 1e12})
    before = copy.deepcopy(state)
    values = spread(fragments, "火车站台", state, edges=system.edges, cjk_tokenizer="jieba")
    assert "foreign" not in values
    assert sum(values.values()) <= 3.5 + 1e-12
    assert all(math.isfinite(value) and value >= 0 for value in values.values())
    assert state == before


@pytest.fixture(scope="module")
def source_trial(frozen_sources, real_backend):
    sources, catalog, _ = frozen_sources
    case = catalog["cases"][0]
    source = sources[case["source"]]
    bases = {absent: study.shared.prepare_trial(source, case, "precise", absent=absent, source_sessions=True)
             for absent in (False, True)}
    return source, case, bases


@pytest.mark.parametrize("policy", study.POLICIES)
@pytest.mark.parametrize("state", study.STATES)
@pytest.mark.parametrize("absent", [False, True])
def test_actual_source_contexts_use_original_ids_scope_and_equal_budgets(source_trial, policy, state, absent):
    source, case, bases = source_trial
    base, ids, history = bases[absent]
    before = copy.deepcopy(base.inspect())
    result = study.evaluate(source, case, base, ids, policy, state, absent)
    assert base.inspect() == before
    assert set(map(tuple, result["selected_sources"])) <= set(map(tuple, history))
    assert set(result["selected_ids"]) <= set(ids.values())
    if absent:
        assert result["gold_ids"] == [] and result["recall_at_6"] is None
    else:
        hit = bool(set(result["gold_ids"]) & set(result["selected_ids"]))
        assert result["recall_at_6"] == float(hit)


def test_development_inputs_do_not_load_evaluation_sources(frozen_sources):
    _, _, protocol = frozen_sources
    sources = study.load_sources(cli.ROOT / ".data/kdconv/data", protocol, "development")
    assert set(sources) == {"music/dev"}


def test_cli_refuses_missing_development_evidence_and_overwrite(tmp_path):
    output = tmp_path / "old.json"
    output.write_text("previous", encoding="utf-8")
    assert cli.main(["development", "--output", str(output)]) == 1
    assert output.read_text() == "previous"


def test_evaluation_requires_frozen_development_selection(frozen_sources, real_backend, tmp_path):
    output = tmp_path / "report.json"
    assert cli.main(["evaluation", "--output", str(output)]) == 1
    assert not output.exists()


def historical_report(phase):
    path = cli.ROOT / f"docs/experiments/phase-9-recall-carryover/{phase}-v2.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("damage", ["reset", "wrong_winner", "missing", "duplicate", "metric", "control", "guards", "row_metric", "nonfinite"])
def test_invalid_development_evidence_is_rejected_before_scoring(frozen_sources, real_backend, tmp_path, monkeypatch, damage):
    report = historical_report("development")
    # Only align the code metadata for this negative report-validation test.
    # These edited reports are never accepted as new benchmark evidence.
    report["code_sha256"] = cli.fingerprints()
    if damage == "reset":
        report["selected"] = "reset"
    elif damage == "wrong_winner":
        report["selected"] = "idf"
    elif damage == "missing":
        report["trials"].pop()
    elif damage == "duplicate":
        report["trials"].append(report["trials"][0])
    elif damage == "metric":
        report["summary"]["novel_cue"]["switch"]["recall_at_6"] = 1.0
    elif damage == "control":
        report["software_controls"]["novel_cue"] = []
    elif damage == "guards":
        report["guards"]["idf"] = {key: True for key in report["guards"]["idf"]}
    elif damage == "row_metric":
        report["trials"][0]["mrr_at_6"] = 2.0
    else:
        report["trials"][0]["latency_ms"] = float("nan")
    path = tmp_path / "development.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Scoring started before development evidence was validated")

    monkeypatch.setattr(study, "control_results", forbidden)
    output = tmp_path / "evaluation.json"
    assert cli.main(["evaluation", "--development-report", str(path), "--output", str(output)]) == 1
    assert not output.exists()


@pytest.mark.parametrize("section,key,value", [
    ("selection", "target_window_turns", 4),
    ("selection", "distractor_windows", 3),
    ("selection", "distractor_window_turns", 4),
    ("gates", "cold_recall_must_not_decrease", False),
    ("gates", "software_controls_must_pass", False),
])
def test_unsupported_protocol_settings_fail_instead_of_being_ignored(frozen_sources, tmp_path, section, key, value):
    _, _, protocol = frozen_sources
    protocol = copy.deepcopy(protocol)
    protocol[section][key] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        cli.load_inputs(path, cli.FIXTURES / "catalog.json", cli.FIXTURES / "controls.json",
                        cli.ROOT / ".data/kdconv/data", "development")


def test_production_verification_rejects_duplicate_reference_trials_before_replay(frozen_sources, real_backend, tmp_path, monkeypatch):
    report = historical_report("evaluation")
    report["trials"].append(next(r for r in report["trials"] if r["candidate"] == "novel_cue"))
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Replay started with an invalid reference trial matrix")

    monkeypatch.setattr(study, "control_results", forbidden)
    output = tmp_path / "verification.json"
    assert cli.main(["verify-production", "--reference-report", str(path), "--output", str(output)]) == 1
    assert not output.exists()


def test_input_changes_during_real_replay_cannot_publish_a_report(frozen_sources, real_backend, tmp_path, monkeypatch, capsys):
    controls_path = tmp_path / "controls.json"
    controls_path.write_bytes((cli.FIXTURES / "controls.json").read_bytes())
    evaluate = study.evaluate
    calls = 0

    def change_input_after_actual_recall(*args, **kwargs):
        nonlocal calls
        row = evaluate(*args, **kwargs)
        calls += 1
        if calls == 1:
            with controls_path.open("a", encoding="utf-8") as output:
                output.write("\n")
        return row

    monkeypatch.setattr(study, "evaluate", change_input_after_actual_recall)
    output = tmp_path / "verification.json"
    reference = cli.ROOT / "docs/experiments/phase-9-recall-carryover/development-v2.json"
    assert cli.main(["verify-production", "--reference-report", str(reference), "--controls", str(controls_path),
                     "--output", str(output)]) == 1
    assert calls == 72  # All replays are real; only the on-disk input was perturbed.
    assert "inputs changed" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("gate", ["switch_recall", "cold_recall", "followup_recall", "switch_mrr",
                                 "absent", "latency", "memory", "controls"])
def test_each_integration_guard_can_block_a_regression(gate):
    report = historical_report("evaluation")
    baseline = report["summary"]["baseline"]
    candidate = copy.deepcopy(report["summary"]["novel_cue"])
    rules = json.loads((cli.FIXTURES / "protocol.json").read_text(encoding="utf-8"))["gates"]
    if gate == "switch_recall":
        candidate["switch"]["recall_at_6"] = baseline["switch"]["recall_at_6"]
    elif gate in ("cold_recall", "followup_recall"):
        state = gate.removesuffix("_recall")
        candidate[state]["recall_at_6"] = baseline[state]["recall_at_6"] - 0.01
    elif gate == "switch_mrr":
        candidate["switch"]["mrr_at_6"] = baseline["switch"]["mrr_at_6"] - 0.01
    elif gate == "absent":
        baseline = copy.deepcopy(baseline)
        baseline["cold"]["absent_nonempty_rate"] = 0.0
    elif gate == "latency":
        candidate["cold"]["p95_latency_ms"] = baseline["cold"]["p95_latency_ms"] * 2.01
    elif gate == "memory":
        candidate["cold"]["peak_python_bytes"] = baseline["cold"]["peak_python_bytes"] * 2.01
    checks = study.guards(candidate, baseline, gate != "controls", rules)
    assert not checks[gate]


@pytest.mark.parametrize("prior", ["bounded-spread-distinct-evidence-v1", "bounded-spread-distinct-evidence-novel-cue-v2",
                                   "bounded-spread-distinct-evidence-novel-cue-v3",
                                   "bounded-spread-distinct-evidence-novel-cue-v4",
                                   "bounded-spread-distinct-evidence-novel-cue-v5-turn-provenance"])
def test_longitudinal_checkpoint_rejects_the_prior_runtime_policy(tmp_path, prior):
    from psychology_base.eval.benchmarks import attachment_longitudinal as harness

    value = harness.build_trajectory("supportive-en").model_dump(mode="json")
    value["sessions"] = value["sessions"][:3]
    manifest = harness.TrajectoryManifest.model_validate(value)
    path = tmp_path / "checkpoint.json"
    harness.run_trajectory(manifest, checkpoint_path=path, stop_after=1)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["configuration"]["fragment_recall_policy"] = prior
    envelope["sha256"] = harness.digest(envelope["payload"])
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration mismatch"):
        harness.load_checkpoint(path, manifest)


@pytest.mark.parametrize("agent_type", ["companion", "tutor"])
@pytest.mark.parametrize("language", ["en", "zh"])
def test_real_application_context_retains_the_new_topic(controls, real_backend, agent_type, language):
    from psychology_base.applications.emotional import CompanionAgent
    from psychology_base.applications.learning import TutorAgent
    from psychology_base.applications.measurement.scaffolding_telemetry import ScaffoldingTelemetry

    case = next(c for c in controls["cases"] if c["id"] == f"switch-{language}")
    system = CompanionSystem(top_k=6, byte_cap=None, cjk_tokenizer="jieba")
    for index, text in enumerate(case["facts"]):
        system.observe(Observation(study.shared.BASE + study.shared.timedelta(minutes=index), "user", text))
    system.recall(case["warm"])
    contexts = []

    def capture(prompt, context, utterance):
        contexts.append(context)
        return "Acknowledged."

    cls = CompanionAgent if agent_type == "companion" else TutorAgent
    options = {"telemetry": ScaffoldingTelemetry(user_id="study")} if agent_type == "tutor" else {}
    agent = cls(user_id="study", session_id="later", companion_system=system, llm_responder=capture, **options)
    agent.respond(case["cue"], timestamp=study.shared.BASE + study.shared.timedelta(days=1))
    assert len(contexts) == 1 and case["facts"][1] in contexts[0]


def test_graph_tag_projection_preserves_perspective_semantics_and_fragment_bytes():
    from psychology_base.eval.benchmarks import graph_priors as graph

    tags = ("user:a", "session:s", "session_7", "speaker:user", "subject:a", "entity:mother",
            "mode:self", "move:hint", "music", "custom:value")
    fragment = make_fragment("a remembered conversation", study.shared.BASE, tags=tags)
    before = copy.deepcopy(fragment)
    assert graph.routing_tags(fragment) == set(tags[4:])
    assert memory._graph_prior_tags(fragment) == set(tags)
    assert fragment == before


@pytest.mark.parametrize("policy", ["production", "routing_filtered", "content_only", "no_initial_edges"])
def test_graph_experiment_uses_real_ingestion_for_both_stores_and_pam(policy, tmp_path):
    from psychology_base.companion.pam_cli import restore_system_from_path
    from psychology_base.companion.pam_export import pam_export
    from psychology_base.eval.benchmarks import graph_priors as graph

    original = memory.jaccard_prior
    system = graph.system_for_policy(policy, cjk_tokenizer="unicode")
    for i, text in enumerate(("orchid garden", "railway platform")):
        system.observe(Observation(study.shared.BASE + study.shared.timedelta(minutes=i), "user", text,
                                   tags=("user:a", "session:s")))
    a, b = system.companion.store.all()
    expected = 0.4 if policy == "production" else 0.0
    assert system.edges.get(a.id, b.id) == system.witness_edges.get(a.id, b.id) == expected
    assert system.edges is not system.witness_edges
    assert system._ingest_fragment.__func__.__code__ is CompanionSystem._ingest_fragment.__code__
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(pam_export(system.companion.store.all())), encoding="utf-8")
    restored = graph.system_for_policy(policy, cjk_tokenizer="unicode")
    for _ in range(2):
        _, offenders = restore_system_from_path(path, system=restored)
        assert not offenders
        assert restored.edges.weights == system.edges.weights
        assert restored.witness_edges.weights == system.witness_edges.weights
        assert restored.companion.store.all() == system.companion.store.all()
    child = restored.rerender(a.id, "orchid flower", timestamp=study.shared.BASE + study.shared.timedelta(days=1),
                              tags=a.tags)
    assert child.parent_ids == (a.id,)
    assert restored.companion.store.by_id(child.id) == restored.witness.store.by_id(child.id) == child
    assert restored.edges.get(a.id, child.id) == restored.witness_edges.get(a.id, child.id)
    assert memory.jaccard_prior is original
    with pytest.raises(ValueError, match="Unknown graph-prior"):
        graph.system_for_policy("missing", cjk_tokenizer="unicode")


@pytest.mark.parametrize("policy", ["production", "routing_filtered", "content_only", "no_initial_edges"])
@pytest.mark.parametrize("state", study.STATES)
def test_graph_trace_matches_actual_source_context_without_mutating_the_baseline(source_trial, policy, state):
    from psychology_base.eval.benchmarks import graph_priors as graph

    source, case, _ = source_trial
    system, ids, _ = study.shared.prepare_trial(source, case, "precise", source_sessions=True,
                                               system_factory=lambda: graph.system_for_policy(policy))
    before = copy.deepcopy(system.inspect())
    traced = graph.trace_recall(source, case, system, ids, state)
    actual = study.evaluate(source, case, system, ids, "production", state, False)
    assert traced["selected_ids"] == actual["selected_ids"]
    assert set(traced["gold_ids"]) == set(actual["gold_ids"])
    assert system.inspect() == before
    for row in traced["fragments"]:
        assert row["final_score"] == pytest.approx(row["seed_and_residual"] + row["hop_1"] + row["hop_2"])
        if state == "cold":
            eligible = set(ids.values()) - {row["id"]}
            assert row["current_edge_weight"] == pytest.approx(sum(system.edges.get(row["id"], fid) for fid in eligible))
    if policy == "no_initial_edges":
        assert len(system.edges) == len(system.witness_edges) == 0
        if state == "cold":
            assert all(row["hop_1"] == row["hop_2"] == row["current_edge_weight"] == 0 for row in traced["fragments"])
        else:
            # Keep normal learning in this initial-prior intervention. Warm
            # recalls really create edges; it is not a zero-spreading ablation.
            warmed, _ = study.prepare_recall(source, case, system, "production", state)
            assert len(warmed.edges) > 0 and any(row["current_edge_weight"] > 0 for row in traced["fragments"])


def test_exposure_audit_counts_actual_unscored_targets_and_background_groups(frozen_sources, tmp_path):
    from psychology_base.eval.benchmarks import graph_priors as graph

    _, catalog, protocol = frozen_sources
    audit = graph.data_audit(cli.ROOT, catalog, protocol)
    assert audit["fresh_cases_scored"] == 0
    assert audit["fresh_target_count"] == 32
    assert len(audit["sources"]["film/dev"]["exposed_dialogues"]) == 127
    for source, details in audit["sources"].items():
        assert not set(details["fresh_eligible_targets"]) & set(details["exposed_dialogues"])
    for case in catalog["cases"]:
        details = audit["sources"][case["source"]]
        for field in ("history", "absent_history", "cue_refs", "followup_refs", "switch_refs", "gold"):
            assert {ref[0] for ref in case[field]} <= set(details["exposed_dialogues"])
    with pytest.raises(FileNotFoundError):
        graph.data_audit(cli.ROOT, catalog, protocol, data_root=tmp_path)


def test_graph_cli_requires_reference_before_scoring_and_preserves_old_evidence(tmp_path):
    output = tmp_path / "report.json"
    assert cli.main(["diagnose-graph", "--output", str(output)]) == 1
    assert not output.exists()
    output.write_text("previous", encoding="utf-8")
    assert cli.main(["diagnose-graph", "--output", str(output)]) == 1
    assert output.read_text(encoding="utf-8") == "previous"


def test_unknown_recall_state_is_rejected_before_source_access():
    with pytest.raises(ValueError, match="Unknown recall state"):
        study.prepare_recall(None, None, None, "production", "unknown")


def test_repeated_pam_restore_preserves_learned_edges_residuals_and_aging(tmp_path):
    from psychology_base.companion.aging import EbbinghausPolicy
    from psychology_base.companion.pam_cli import export_system_to_path, restore_system_from_path

    aging = EbbinghausPolicy()
    system = CompanionSystem(byte_cap=None, aging_policy=aging)
    for i, text in enumerate(("orchid garden", "orchid flower")):
        system.observe(Observation(study.shared.BASE + study.shared.timedelta(minutes=i), "user", text))
    system.recall("orchid")
    fragments = system.companion.store.all()
    assert system.edges.get(fragments[0].id, fragments[1].id) > memory.jaccard_prior(*fragments)
    path = tmp_path / "memory.json"
    export_system_to_path(system, path)
    before = copy.deepcopy((system.inspect(), system.edges, system.witness_edges,
                            system.activation, system.witness_state, aging))
    for _ in range(2):
        restore_system_from_path(path, system=system)
        assert system.companion.store.all() == fragments
        assert (system.inspect(), system.edges, system.witness_edges,
                system.activation, system.witness_state, aging) == before
    # Same words at a different time remain a new content-addressed occurrence.
    system.observe(Observation(study.shared.BASE + study.shared.timedelta(days=1), "user", fragments[0].content))
    assert len(system.companion.store.all()) == 3


@pytest.mark.parametrize("missing", ["companion", "witness"])
def test_reingestion_repairs_only_the_missing_store(missing):
    from psychology_base.companion.fragment import FragmentStore

    system = CompanionSystem(byte_cap=None)
    for i, text in enumerate(("orchid garden", "orchid flower")):
        system.observe(Observation(study.shared.BASE + study.shared.timedelta(minutes=i), "user", text))
    system.recall("orchid")
    fragments = system.companion.store.all()
    retained = system.witness if missing == "companion" else system.companion
    retained_edges = system.witness_edges if missing == "companion" else system.edges
    before = copy.deepcopy(retained_edges)
    getattr(system, missing).store = FragmentStore()
    if missing == "companion":
        system.edges = EdgeStore()
    else:
        system.witness_edges = EdgeStore()
    for fragment in fragments:
        system._ingest_fragment(fragment)
    assert system.companion.store.all() == system.witness.store.all() == fragments
    assert retained.store.all() == fragments
    assert retained_edges == before


def test_conflicting_existing_fragment_id_is_rejected_before_mutation():
    from dataclasses import replace

    system = CompanionSystem(byte_cap=None)
    system.observe(Observation(study.shared.BASE, "user", "original evidence"))
    original = system.companion.store.all()[0]
    before = copy.deepcopy(system.inspect())
    corrupted = replace(original, content="different evidence under the same ID")
    with pytest.raises(ValueError, match="ID conflicts"):
        system._ingest_fragment(corrupted)
    assert system.inspect() == before
    assert system.companion.store.all() == system.witness.store.all() == [original]


@pytest.mark.parametrize('state', ['cold','followup','switch'])
def test_navigation_trace_reuses_the_actual_preparation_and_keeps_live_state(source_trial, state):
    from psychology_base.eval.benchmarks import graph_priors as graph
    source, case, _ = source_trial
    system, ids, _ = study.shared.prepare_trial(source, case, 'precise', source_sessions=True,
        system_factory=lambda: graph.system_for_policy('production'))
    graph.bind_navigation(system, 'lexical')
    before = copy.deepcopy(system.inspect())
    traced = graph.trace_recall(source, case, system, ids, state)
    actual = study.evaluate(source, case, system, ids, 'production', state, False)
    assert traced['selected_ids'] == actual['selected_ids']
    assert system.inspect() == before
