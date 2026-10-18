"""Review regressions: actual memory provenance, safety, persistence and CLI.

Synthetic fixtures test software contracts only. Fault injection below makes
specific failures observable; successful model behavior is never mocked.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from pathlib import Path

import pytest

from benchmarks.scripts import phase9_attachment_longitudinal_run as runner
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.benchmarks import attachment_longitudinal as harness


def short_manifest(name="supportive-en", count=4):
    value = harness.build_trajectory(name).model_dump(mode="json")
    value["sessions"] = value["sessions"][:count]
    return harness.TrajectoryManifest.model_validate(value)


def mutate(manifest, callback):
    value = manifest.model_dump(mode="json")
    callback(value)
    return harness.TrajectoryManifest.model_validate(value)


def short_catalog(tmp_path):
    catalog = harness.load_catalog()
    recipe = next(r for r in catalog["scenarios"] if r["name"] == "supportive-en")
    recipe["phases"][0]["count"] = 3
    catalog["scenarios"] = [recipe]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def pause(tmp_path, name="supportive-en", count=4):
    manifest = short_manifest(name, count)
    path = tmp_path / manifest.trajectory_id / "checkpoint.json"
    result = harness.run_trajectory(manifest, checkpoint_path=path, stop_after=2)
    assert result.stop_reason == "paused_at_checkpoint"
    return manifest, path, result


def rehash_checkpoint(path, change):
    envelope = json.loads(path.read_text(encoding="utf-8"))
    change(envelope["payload"])
    envelope["sha256"] = harness.digest(envelope["payload"])
    path.write_text(json.dumps(envelope), encoding="utf-8")


def test_repeated_questions_do_not_prove_original_fact_recall(monkeypatch):
    manifest = harness.build_trajectory("supportive-zh")
    anchor = manifest.anchor_fragment
    original_observe = CompanionSystem.observe

    def lose_fact(system, observation):
        if observation.content != anchor.content:
            original_observe(system, observation)

    # Simulate losing exactly the original fact, while all ordinary question,
    # reply, retrieval and scoring paths continue to run normally.
    monkeypatch.setattr(CompanionSystem, "observe", lose_fact)
    result = harness.run_trajectory(manifest)
    report = result.report()
    assert all(f.id != anchor.id for f in result.system.companion.store.all())
    assert any(manifest.sentinel in context for record in result.ledger for context in record["contexts"])
    assert not report["continuity_probe_pass"]
    assert all(not t["original_user_fact_in_context"] for t in report["block_transitions"])


def test_fragment_audits_match_real_response_context_and_observed_memory():
    result = harness.run_trajectory(short_manifest())
    stored = {f.id for f in result.system.companion.store.all()}
    seen = 0
    for record in result.ledger:
        assert len(record["context_fragments"]) == len(record["contexts"]) == 2
        for text, snapshots in zip(record["contexts"], record["context_fragments"]):
            fragments = harness.pam_import({**harness.pam_export(()), "fragments": snapshots})
            assert harness._render_context_block(fragments) == text
            assert all(f.id in stored for f in fragments)
            seen += len(fragments)
    assert seen > 0


@pytest.mark.parametrize("kind", ["missing", "bad_type", "too_many", "duplicate", "future", "wrong_text", "forged_label"])
def test_checkpoint_rejects_false_or_future_context_evidence(tmp_path, kind):
    manifest, path, _ = pause(tmp_path)

    def corrupt(payload):
        record = payload["ledger"][1]
        snapshot = record["context_fragments"][0]
        assert snapshot
        if kind == "missing":
            record.pop("context_fragments")
        elif kind == "bad_type":
            record["context_fragments"] = {}
        elif kind == "too_many":
            record["context_fragments"][0] = snapshot[:1] * 7
        elif kind == "duplicate":
            record["context_fragments"][0] = snapshot[:1] * 2
        elif kind == "future":
            future = harness._turn_fragments(manifest, manifest.sessions[-1], "future", "reply", 1)
            record["context_fragments"][0] = harness.pam_export(future)["fragments"]
        elif kind == "wrong_text":
            snapshot[0]["content"] = "This was never the original memory."
        else:
            record["contexts"][0] = "- user said (session imagined): fabricated anchor"

    rehash_checkpoint(path, corrupt)
    with pytest.raises(ValueError, match="context"):
        harness.load_checkpoint(path, manifest)


def test_extreme_emotion_is_exercised_and_rebuilt_across_resume(tmp_path):
    manifest, path, paused = pause(tmp_path, "extreme-emotion-en")
    assert paused.review_flags == []
    resumed = harness.run_trajectory(manifest, checkpoint_path=path, resume=True)
    uninterrupted = harness.run_trajectory(manifest)
    assert resumed.review_flags == uninterrupted.review_flags
    assert len(resumed.review_flags) == 1
    assert resumed.review_flags[0]["last_session_id"] == manifest.sessions[2].session_id
    assert resumed.ledger[0]["affect_provenance"] == "explicit_fixture_input"


@pytest.mark.parametrize("gap", ["unknown_affect", "empty_session"])
def test_missing_affect_does_not_fabricate_neutral_or_bridge_a_streak(gap):
    def add_gap(value):
        if gap == "unknown_affect":
            value["sessions"][2]["affect"] = None
        else:
            value["sessions"][2]["companion_turns"] = []

    manifest = mutate(short_manifest("extreme-emotion-en"), add_gap)
    result = harness.run_trajectory(manifest)
    assert result.review_flags == []
    if gap == "unknown_affect":
        assert result.ledger[2]["affect"] is None
        assert result.ledger[2]["affect_provenance"] == "not_measured"


def test_rejected_checkpoint_boundary_does_not_keep_unaccepted_safety_flags(tmp_path, monkeypatch):
    original = harness.save_checkpoint

    def disk_failure(path, manifest, state, ledger, system):
        if state.sessions_completed == 3:
            raise OSError("failure after the third affect observation")
        return original(path, manifest, state, ledger, system)

    monkeypatch.setattr(harness, "save_checkpoint", disk_failure)
    result = harness.run_trajectory(short_manifest("extreme-emotion-en"),
                                    checkpoint_path=tmp_path / "checkpoint.json", checkpoint_every=1)
    assert result.state.sessions_completed == 2
    assert result.review_flags == []
    assert result.report()["failed_sessions"] == 1


def test_scripted_results_explicitly_do_not_claim_model_evaluation():
    report = harness.run_trajectory(short_manifest()).report()
    assert report["evidence_class"] == "scripted_software_control"
    assert report["response_uses_recalled_context"] is False
    assert report["model_response_quality_measured"] is False
    assert report["attachment_validity_established"] is False
    assert report["simulated_time_only"] is True
    assert report["memory_capacity_pressure_exercised"] is False
    assert all(r["affect"] is None for r in report["per_session"])


def test_external_callable_is_not_mislabeled_as_a_fixed_script():
    result = harness.run_trajectory(short_manifest(), responder_factory=lambda spec: lambda *_: "Okay.")
    report = result.report()
    assert report["completed_sessions"] == 4
    assert report["evidence_class"] == "unverified_injected_callable"
    assert report["response_uses_recalled_context"] is None


def test_empty_response_is_an_actual_execution_failure():
    report = harness.run_trajectory(short_manifest(), responder_factory=lambda spec: lambda *_: "").report()
    assert report["completed_sessions"] == 0
    assert report["failed_sessions"] == 1
    assert not report["duration_qualified_proxy_pass"]


def test_foreign_ownership_is_detected_without_a_continuity_marker():
    result = harness.run_trajectory(short_manifest())
    result.system.observe(Observation(harness.BASE_TIME, "user", "An unrelated user's private sentence.",
                                      tags=("user:another-user",)))
    check = harness.isolation_checks([result])
    assert not check["passed"]
    assert check["leaks"][0]["reason"] == "foreign or missing ownership tag"


def test_duplicate_markers_across_distinct_users_are_rejected():
    a = harness.run_trajectory(short_manifest())
    other = mutate(short_manifest("supportive-zh"), lambda value: value.update(sentinel=a.manifest.sentinel))
    b = harness.run_trajectory(other)
    with pytest.raises(ValueError, match="sentinels"):
        harness.isolation_checks([a, b])


@pytest.mark.parametrize("field,value", [("stop_after", 0), ("stop_after", True), ("checkpoint_every", 0), ("checkpoint_every", False)])
def test_invalid_execution_controls_fail_early(field, value):
    with pytest.raises(ValueError):
        harness.run_trajectory(short_manifest(), **{field: value})


def test_resume_and_injected_checkpoint_guards(tmp_path):
    manifest, path, _ = pause(tmp_path)
    with pytest.raises(ValueError, match="needs a checkpoint"):
        harness.run_trajectory(manifest, resume=True)
    with pytest.raises(ValueError, match="fixed scripted responses"):
        harness.run_trajectory(manifest, checkpoint_path=path, responder_factory=lambda _: lambda *_: "Okay.")
    with pytest.raises(ValueError, match="precedes"):
        harness.run_trajectory(manifest, checkpoint_path=path, resume=True, stop_after=1)
    with pytest.raises(ValueError, match="already exists"):
        harness.run_trajectory(manifest, checkpoint_path=path)


def test_checkpoint_save_rejects_state_and_transcript_memory_mismatches(tmp_path):
    manifest, _, paused = pause(tmp_path)
    with pytest.raises(ValueError, match="state does not match"):
        harness.save_checkpoint(tmp_path / "bad.json", manifest, replace(paused.state, trust_ema=1), paused.ledger, paused.system)
    paused.system.observe(Observation(harness.BASE_TIME + timedelta(seconds=1), "user", "Unaccepted extra memory."))
    with pytest.raises(ValueError, match="memory differs"):
        harness.save_checkpoint(tmp_path / "bad.json", manifest, paused.state, paused.ledger, paused.system)


@pytest.mark.parametrize("kind", ["envelope", "checksum", "state_fields", "old_schema", "memory_binding"])
def test_additional_checkpoint_validation(tmp_path, kind):
    manifest, path, _ = pause(tmp_path)
    if kind == "envelope":
        path.write_text("{}", encoding="utf-8")
    elif kind == "checksum":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["sha256"] = "bad"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif kind == "memory_binding":
        envelope = json.loads(path.read_text(encoding="utf-8"))
        artifact = harness.pam_export(())
        name = f"memory-{harness.digest(artifact)}.json"
        harness.atomic_write_json(path.with_name(name), artifact)
        rehash_checkpoint(path, lambda p: p.update(memory_file=name, memory_sha256=harness.digest(artifact)))
    elif kind == "old_schema":
        rehash_checkpoint(path, lambda p: p.update(schema_version="attachment-checkpoint-v1"))
    else:
        rehash_checkpoint(path, lambda p: p["state"].pop("bond_history"))
    with pytest.raises(ValueError):
        harness.load_checkpoint(path, manifest)


@pytest.mark.parametrize("value", [[], {}, {"scenarios": [{"name": "a"}, {"name": "a"}], "languages": {}}])
def test_invalid_catalogs_are_rejected(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="catalog"):
        harness.load_catalog(path)


@pytest.mark.parametrize("change", [
    lambda r: r.update(phases=[]),
    lambda r: r["phases"][0].update(count=True),
    lambda r: r["phases"][0].update(profile="unknown"),
    lambda r: r.update(empty_at=40),
    lambda r: r.update(empty_at=3, handoff_at=3),
])
def test_bad_recipe_controls_fail_instead_of_silently_omitting_behavior(change):
    catalog = harness.load_catalog()
    change(catalog["scenarios"][0])
    with pytest.raises(ValueError):
        harness.build_trajectory("supportive-en", catalog=catalog)


@pytest.mark.parametrize("args", [
    ["--resume"], ["--verify-resume", "--stop-after", "2"],
    ["--manifest", "missing.json", "--scenario", "supportive-en"],
    ["--scenario", "supportive-en", "--scenario", "supportive-en"],
    ["--msc-limit", "-1"], ["--stop-after", "0"],
])
def test_cli_rejects_invalid_invocations_before_execution(tmp_path, args):
    with pytest.raises(SystemExit) as error:
        runner.main(["--output-dir", str(tmp_path), *args])
    assert error.value.code == 2
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("source_kind", ["missing", "malformed", "insufficient"])
def test_cli_retains_unavailable_source_counts_without_dummy_replacements(tmp_path, source_kind):
    source = tmp_path / "source.jsonl"
    if source_kind == "malformed":
        source.write_text("{not valid json}\n", encoding="utf-8")
    elif source_kind == "insufficient":
        source.write_text("", encoding="utf-8")
    out = tmp_path / "output"
    status = runner.main(["--catalog", str(short_catalog(tmp_path)), "--msc-shim", str(source),
                          "--msc-limit", "3", "--output-dir", str(out)])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert status == 1
    assert summary["selected_trajectories"] == 4
    assert summary["reported_trajectories"] == 1
    assert summary["setup_errors"][0]["trajectory_count"] == 3


def test_self_contained_manifest_resume_preserves_previous_cohort_summary(tmp_path):
    manifest, checkpoint, _ = pause(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    harness.atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    previous = b'{"previous_cohort":true}\n'
    (tmp_path / "summary.json").write_bytes(previous)
    status = runner.main(["--manifest", str(manifest_path), "--resume", "--output-dir", str(tmp_path),
                          "--catalog", str(tmp_path / "missing-catalog.json")])
    assert status == 0
    assert (tmp_path / "summary.json").read_bytes() == previous
    assert (checkpoint.parent / "resume-summary.json").is_file()


@pytest.mark.parametrize("failure", ["timeout", "crash"])
def test_resume_verification_failure_keeps_the_reference_report(tmp_path, monkeypatch, failure):
    def failed_process(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 60)
        return SimpleNamespace(returncode=1, stderr="injected child crash")

    monkeypatch.setattr(runner.subprocess, "run", failed_process)
    out = tmp_path / "output"
    status = runner.main(["--catalog", str(short_catalog(tmp_path)), "--msc-limit", "0",
                          "--output-dir", str(out), "--verify-resume"])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((out / "supportive-en/report.json").read_text(encoding="utf-8"))
    assert status == 1
    assert summary["reported_trajectories"] == 1
    assert summary["setup_errors"] == []
    assert report["completed_sessions"] == 3
    assert not report["checkpoint_verification"]["passed"]


def test_cli_verifies_a_short_trajectory_in_a_real_child_process(tmp_path):
    out = tmp_path / "output"
    assert runner.main(["--catalog", str(short_catalog(tmp_path)), "--msc-limit", "0",
                        "--output-dir", str(out), "--verify-resume"]) == 0
    report = json.loads((out / "supportive-en/report.json").read_text(encoding="utf-8"))
    assert report["checkpoint_verification"]["pause_after_completed_sessions"] == 2
    assert report["checkpoint_verification"]["fresh_process"]
    assert report["checkpoint_verification"]["comparison_kind"] == "fixed_transcript_attachment_bookkeeping"
    assert not report["checkpoint_verification"]["companion_response_quality_tested"]


@pytest.mark.parametrize("kind", ["not_a_list", "too_long", "status", "completion", "context_type", "context_item", "context_count"])
def test_checkpoint_ledger_contracts_are_enforced(tmp_path, kind):
    manifest, path, _ = pause(tmp_path)

    def change(payload):
        if kind == "not_a_list":
            payload["ledger"] = {}
        elif kind == "too_long":
            payload["ledger"] *= 3
        elif kind == "status":
            payload["ledger"][0]["status"] = "unverified"
        elif kind == "completion":
            payload["ledger"][0]["status"] = "empty"
        elif kind == "context_type":
            payload["ledger"][0]["contexts"] = {}
        elif kind == "context_item":
            payload["ledger"][0]["contexts"][0] = 0
        else:
            payload["ledger"][0]["contexts"] = []

    rehash_checkpoint(path, change)
    with pytest.raises(ValueError):
        harness.load_checkpoint(path, manifest)


@pytest.mark.parametrize("change", [
    lambda value: value["sessions"][0].update(companion_turns=["One reply only."]),
    lambda value: value["sessions"][0].update(seeker_provenance=["scripted"]),
    lambda value: value["sessions"][0].update(seeker_turns=["", "Blank is invalid."]),
    lambda value: value["sessions"][0].update(seeker_provenance=["msc_source_opener", "scripted"]),
    lambda value: value["sessions"][1].update(block_id="middle-block") or value["sessions"][2].update(block_id=value["sessions"][0]["block_id"]),
    lambda value: value["continuity_anchor"].update(session_id="not-a-session"),
    lambda value: value["continuity_anchor"].update(user_turn=3),
    lambda value: value["sessions"][0].update(affect={"valence": "0.8", "arousal": 0.8}),
])
def test_additional_manifest_validation(change):
    with pytest.raises(ValueError):
        mutate(short_manifest(), change)


def one_source():
    return harness.MSCConversation("source", (("I enjoy ceramics.",), ("I hike.",)), (
        harness.MSCSession(1, 0, (harness.MSCTurn(0, "My real source opener."), harness.MSCTurn(1, "Hi."))),
    ))


@pytest.mark.parametrize("kind", ["missing_metadata", "wrong_provenance", "source_hash", "source_ids", "source_after_synthetic"])
def test_source_manifest_cannot_hide_or_relabel_provenance(kind):
    source = one_source()
    if kind == "source_after_synthetic":
        source = harness.MSCConversation(source.conversation_id, source.personas, source.sessions + (
            harness.MSCSession(2, 2, (harness.MSCTurn(0, "A second source opener."),)),
        ))
    manifest = harness.build_trajectory("supportive-en", source=source, source_file_sha256="a" * 64)

    def change(value):
        if kind == "missing_metadata":
            value["sessions"][0]["source_gap_days"] = None
        elif kind == "wrong_provenance":
            value["sessions"][0]["seeker_provenance"] = ["scripted", "scripted"]
        elif kind == "source_hash":
            value["source_file_sha256"] = None
        elif kind == "source_ids":
            value["source_conversation_ids"] = []
        else:
            value["sessions"][0].update(
                provenance="synthetic_continuation", source_gap_days=None, source_session_id=None,
                source_conversation_id=None, source_elapsed_days=None, seeker_provenance=["scripted", "scripted"],
            )

    with pytest.raises(ValueError):
        mutate(manifest, change)


def test_source_builder_rejects_empty_sources_and_source_opener_replacement():
    empty = harness.MSCConversation("empty", ((), ()), ())
    with pytest.raises(ValueError, match="room for continuations"):
        harness.build_trajectory("supportive-en", source=empty, source_file_sha256="a" * 64)
    catalog = harness.load_catalog()
    catalog["scenarios"][0]["handoff_at"] = 1
    with pytest.raises(ValueError, match="source opener cannot be replaced"):
        harness.build_trajectory("supportive-en", catalog=catalog, source=one_source(), source_file_sha256="a" * 64)
    with pytest.raises(ValueError, match="unknown scenario"):
        harness.build_trajectory("does-not-exist")


def test_atomic_write_failure_preserves_prior_file_and_cleans_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint.json"
    destination.write_text("original\n", encoding="utf-8")

    def disk_failure(*_):
        raise OSError("injected replace failure")

    monkeypatch.setattr(harness.os, "replace", disk_failure)
    with pytest.raises(OSError):
        harness.atomic_write_json(destination, {"new": "data"})
    assert destination.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_cli_uses_real_source_input_and_rejects_duplicate_source_identities(tmp_path):
    catalog_path = short_catalog(tmp_path)
    source = tmp_path / "source.jsonl"
    row = {"conversation_id": "actual-input-row", "personas": [["I enjoy ceramics."], []],
           "sessions": [{"session_id": 1, "time_interval_days": 0,
                         "turns": [{"speaker_idx": 0, "utterance": "My exact source opening."}]}]}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "output"
    assert runner.main(["--catalog", str(catalog_path), "--msc-shim", str(source), "--msc-limit", "1",
                        "--output-dir", str(out)]) == 0
    summary_bytes = (out / "summary.json").read_bytes()
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in out.glob("*/manifest.json")]
    native = next(m for m in manifests if m["source_conversation_ids"])
    assert native["sessions"][0]["seeker_turns"][0] == "My exact source opening."
    assert native["sessions"][0]["seeker_provenance"][0] == "msc_source_opener"
    assert native["sessions"][0]["companion_provenance"] == "scripted"
    with pytest.raises(SystemExit):
        runner.main(["--catalog", str(catalog_path), "--msc-limit", "0", "--output-dir", str(out)])
    assert (out / "summary.json").read_bytes() == summary_bytes
    source.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    with pytest.raises(SystemExit):
        runner.main(["--catalog", str(catalog_path), "--msc-shim", str(source), "--msc-limit", "2",
                     "--output-dir", str(tmp_path / "duplicates")])


def test_cli_rejects_mismatched_manifests_without_replacing_them(tmp_path):
    manifest = short_manifest()
    input_path = tmp_path / "input.json"
    harness.atomic_write_json(input_path, manifest.model_dump(mode="json"))
    destination = tmp_path / "output" / manifest.trajectory_id / "manifest.json"
    different = mutate(manifest, lambda p: p.update(seed=1))
    harness.atomic_write_json(destination, different.model_dump(mode="json"))
    previous = destination.read_bytes()
    assert runner.main(["--manifest", str(input_path), "--output-dir", str(destination.parent.parent)]) == 1
    assert destination.read_bytes() == previous


def test_cli_rejects_an_existing_checkpoint_before_writing_a_manifest(tmp_path):
    manifest, checkpoint, _ = pause(tmp_path)
    input_path = tmp_path / "input.json"
    harness.atomic_write_json(input_path, manifest.model_dump(mode="json"))
    previous = checkpoint.read_bytes()
    assert runner.main(["--manifest", str(input_path), "--output-dir", str(tmp_path)]) == 1
    assert checkpoint.read_bytes() == previous
    assert not checkpoint.with_name("manifest.json").exists()


def test_single_session_cannot_claim_an_exercised_resume_comparison(tmp_path):
    manifest = short_manifest(count=1)
    path = tmp_path / "manifest.json"
    harness.atomic_write_json(path, manifest.model_dump(mode="json"))
    reference = harness.run_trajectory(manifest).report()
    check = runner._verify_resume(manifest, path, tmp_path, reference)
    assert not check["passed"]
    assert "pause boundary" in check["reason"]


def test_real_child_score_mismatch_is_detected(tmp_path, monkeypatch):
    manifest = short_manifest(count=3)
    path = tmp_path / "manifest.json"
    harness.atomic_write_json(path, manifest.model_dump(mode="json"))
    reference = harness.run_trajectory(manifest).report()
    original_run = runner.subprocess.run

    def corrupt_child_report(command, **kwargs):
        child = original_run(command, **kwargs)
        assert child.returncode == 0
        output = Path(command[command.index("--output-dir") + 1])
        report_path = output / manifest.trajectory_id / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["attachment_state"]["trust_ema"] = 1.0
        harness.atomic_write_json(report_path, report)
        return child

    monkeypatch.setattr(runner.subprocess, "run", corrupt_child_report)
    comparison = runner._verify_resume(manifest, path, tmp_path, reference)
    assert comparison["fresh_process"]
    assert not comparison["checks"]["attachment_state"]
    assert not comparison["passed"]
