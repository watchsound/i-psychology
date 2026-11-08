"""Software controls, provenance, isolation and checkpoint failures.

All integration tests use the real companion/safety/memory path with fixed
scripts. No model, network, downloaded corpus, or clinical labels are needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta

import pytest

from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.benchmarks import attachment_longitudinal as harness
from psychology_base.eval.benchmarks.public_datasets.msc import MSCConversation, MSCSession, MSCTurn


SCRIPT = harness.REPO / "benchmarks/scripts/phase9_attachment_longitudinal_run.py"


def altered(manifest, change):
    payload = manifest.model_dump(mode="json")
    change(payload)
    return harness.TrajectoryManifest.model_validate(payload)


def source_conversation(name="source-one"):
    return MSCConversation(
        conversation_id=name, personas=(("I teach pottery.",), ("I enjoy cycling.",)),
        sessions=tuple(MSCSession(i + 1, 0 if i == 0 else 2.0, (
            MSCTurn(0, f"Hello, this is source opening {i + 1}."), MSCTurn(1, "Hi."),
        )) for i in range(4)),
    )


@pytest.mark.parametrize("name,legacy,duration", [
    ("supportive-en", True, True), ("supportive-zh", True, True),
    ("neutral-en", True, True), ("neutral-zh", True, True),
    ("rupture-en", False, False), ("rupture-zh", False, False),
    ("sycophancy-en", False, False), ("repair-en", True, True), ("repair-zh", True, True),
    ("short-duration-en", True, False), ("fifteen-sessions-en", False, False),
])
def test_offline_control_matrix(name, legacy, duration):
    result = harness.run_trajectory(harness.build_trajectory(name))
    report = result.report()
    assert report["trajectory_complete"]
    assert report["legacy_gate_pass"] is legacy
    assert report["duration_qualified_proxy_pass"] is duration
    assert len(result.state.bond_history) == report["completed_sessions"]
    assert result.state.cumulative_companion_turns == 2 * report["completed_sessions"]
    if name.startswith("neutral"):
        assert result.state.trust_ema == 4.0
        assert report["neutral_control_passes_legacy_gate"]


def test_repair_improves_over_matched_unrepaired_trajectory():
    repaired = harness.run_trajectory(harness.build_trajectory("repair-en"))
    unrepaired = harness.run_trajectory(harness.build_trajectory("rupture-en"))
    assert repaired.state.trust_ema > unrepaired.state.trust_ema
    assert repaired.state.cumulative_rupture_turns == unrepaired.state.cumulative_rupture_turns / 2


def test_repeat_runs_reproduce_scores_with_distinct_audited_occurrences():
    manifest = harness.build_trajectory("supportive-en")
    a = harness.run_trajectory(manifest)
    b = harness.run_trajectory(manifest)
    assert a.state == b.state
    assert harness._replay(manifest, a.ledger) == a.state
    assert harness._replay(manifest, b.ledger) == b.state
    for result in (a, b):
        assert harness.pam_export(harness._expected_fragments(manifest, result.ledger)) == harness.pam_export(
            result.system.companion.store.all())
    assert {g for r in a.ledger for g in r["exchange_ids"]}.isdisjoint(
        g for r in b.ledger for g in r["exchange_ids"])
    assert [r["scoring"] for r in a.ledger] == [r["scoring"] for r in b.ledger]


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p["sessions"][1].update(user_id="another-user"), "mixed-user"),
    (lambda p: p["sessions"][1].update(session_id=p["sessions"][0]["session_id"]), "unique ordered"),
    (lambda p: p["sessions"][1].update(index=1), "unique ordered"),
    (lambda p: p["sessions"][1].update(starts_at=p["sessions"][0]["starts_at"]), "non-overlapping"),
    (lambda p: p["sessions"][0].update(starts_at="2026-10-18T00:00:00"), "timezone"),
    (lambda p: p["sessions"][0].update(ends_at=p["sessions"][0]["starts_at"]), "duration"),
    (lambda p: p["sessions"][0].update(source_conversation_id="invented"), "source provenance"),
    (lambda p: p.update(trajectory_id="../other"), "pattern"),
    (lambda p: p["sessions"][0].update(unrecognized=True), "Extra inputs"),
])
def test_invalid_manifests_fail_before_execution(mutate, match):
    with pytest.raises(ValueError, match=match):
        altered(harness.build_trajectory("supportive-en"), mutate)


def test_seeded_manifest_preserves_source_identity_timing_and_provenance():
    source = source_conversation()
    manifest = harness.build_trajectory("supportive-en", source=source, source_file_sha256="a" * 64)
    assert manifest.persona == source.personas[0]
    assert manifest.source_conversation_ids == (source.conversation_id,)
    assert source.personas[0][0] in manifest.sessions[0].seeker_turns[1]
    assert len(manifest.sessions) == 16
    assert [s.source_elapsed_days for s in manifest.sessions[:4]] == [0, 2, 4, 6]
    for spec, original in zip(manifest.sessions[:4], source.sessions):
        assert spec.seeker_turns[0] == original.turns[0].utterance
        assert spec.seeker_provenance == ("msc_source_opener", "scripted")
        assert spec.source_gap_days == original.time_interval_days
    assert all(s.provenance == "synthetic_continuation" for s in manifest.sessions[4:])
    assert manifest.sessions[-1].starts_at - manifest.sessions[0].starts_at == timedelta(days=28)
    result = harness.run_trajectory(manifest)
    assert result.report()["completed_sessions"] == 16


@pytest.mark.parametrize("kind", ["duplicate", "foreign", "timing", "gap"])
def test_source_corruption_is_rejected(kind):
    manifest = harness.build_trajectory("supportive-en", source=source_conversation(), source_file_sha256="a" * 64)

    def corrupt(payload):
        if kind == "duplicate":
            payload["sessions"][1]["source_session_id"] = 1
        elif kind == "foreign":
            payload["sessions"][1]["source_conversation_id"] = "another-source-person"
        elif kind == "gap":
            payload["sessions"][1]["source_gap_days"] = 1
        else:
            spec = manifest.sessions[1]
            payload["sessions"][1]["starts_at"] = (spec.starts_at + timedelta(hours=1)).isoformat()
            payload["sessions"][1]["ends_at"] = (spec.ends_at + timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError):
        altered(manifest, corrupt)


def test_scripted_callback_cannot_fake_a_missing_memory_probe(monkeypatch):
    manifest = altered(harness.build_trajectory("supportive-en"), lambda payload: [
        session.update(companion_turns=["You mentioned an earlier topic."] * 2) for session in payload["sessions"]
    ])
    original_observe = CompanionSystem.observe

    def lose_original_fact(system, event):
        if event.content != manifest.anchor_fragment.content:
            original_observe(system, event)

    monkeypatch.setattr(CompanionSystem, "observe", lose_original_fact)
    result = harness.run_trajectory(manifest)
    report = result.report()
    assert report["legacy_gate_pass"]
    assert result.state.cumulative_consistency_turns > 0
    assert not report["continuity_probe_pass"]
    assert all(not t["original_user_fact_in_context"] for t in report["block_transitions"])


def test_isolated_users_and_deliberate_contamination_detection():
    a = harness.run_trajectory(harness.build_trajectory("supportive-en"))
    b = harness.run_trajectory(harness.build_trajectory("supportive-zh"))
    assert harness.isolation_checks([a, b])["passed"]
    a.system.observe(Observation(
        timestamp=harness.BASE_TIME, speaker="user", content=b.manifest.sentinel,
        tags=(f"user:{a.manifest.user_id}",),
    ))
    assert not harness.isolation_checks([a, b])["passed"]
    with pytest.raises(ValueError, match="identities"):
        harness.isolation_checks([a, a])


def test_empty_session_is_excluded_without_incrementing_state():
    result = harness.run_trajectory(harness.build_trajectory("empty-session-en"))
    report = result.report()
    assert result.state.sessions_completed == 15
    assert report["attempted_sessions"] == 16
    assert report["excluded_sessions"] == 1
    assert report["failed_sessions"] == 0
    assert not report["trajectory_complete"]
    assert not report["duration_qualified_proxy_pass"]


@pytest.mark.parametrize("failure_stage", ["response", "factory"])
def test_execution_error_retains_denominator_and_last_accepted_state(failure_stage):
    manifest = harness.build_trajectory("supportive-en")

    def factory(spec):
        if spec.index == 9 and failure_stage == "factory":
            raise RuntimeError("injected interruption")
        replies = iter(spec.companion_turns)

        def reply(*_):
            if spec.index == 9:
                raise RuntimeError("injected interruption")
            return next(replies)
        return reply

    result = harness.run_trajectory(manifest, responder_factory=factory)
    report = result.report()
    assert report["planned_sessions"] == 16
    assert report["attempted_sessions"] == 9
    assert report["completed_sessions"] == 8
    assert report["failed_sessions"] == 1
    assert report["unattempted_sessions"] == 7
    assert result.state.sessions_completed == 8
    assert report["per_session"][-1]["error"] == "RuntimeError: injected interruption"
    assert not report["duration_qualified_proxy_pass"]


def test_crisis_handoff_stops_chain_and_reaches_local_sink():
    result = harness.run_trajectory(harness.build_trajectory("handoff-en"))
    report = result.report()
    assert report["stop_reason"] == "handoff"
    assert report["attempted_sessions"] == 9
    assert report["completed_sessions"] == 8
    assert report["excluded_sessions"] == 1
    assert len(report["handoff_incidents"]) == 1
    assert report["per_session"][-1]["contexts"] == []  # Model call short-circuited.
    assert not report["trajectory_complete"]


def test_age_gate_is_exercised_on_the_real_agent_path():
    manifest = altered(harness.build_trajectory("supportive-en"), lambda p: p.update(age_years=13))
    report = harness.run_trajectory(manifest).report()
    assert report["completed_sessions"] == 0
    assert report["failed_sessions"] == 1
    assert "PermissionError" in report["per_session"][0]["error"]


def paused_checkpoint(tmp_path, name="supportive-en"):
    manifest = harness.build_trajectory(name)
    checkpoint = tmp_path / manifest.trajectory_id / "checkpoint.json"
    result = harness.run_trajectory(manifest, checkpoint_path=checkpoint, stop_after=8)
    assert result.stop_reason == "paused_at_checkpoint"
    assert result.state.sessions_completed == 8
    return manifest, checkpoint


@pytest.mark.parametrize("name", ["supportive-en", "supportive-zh"])
def test_fresh_process_resume_matches_uninterrupted_attachment_trace(tmp_path, name):
    manifest, checkpoint = paused_checkpoint(tmp_path, name)
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest_path), "--output-dir", str(tmp_path), "--resume"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    resumed = json.loads((checkpoint.parent / "report.json").read_text(encoding="utf-8"))
    uninterrupted = harness.run_trajectory(manifest).report()
    assert harness.canonical_json(resumed["attachment_state"]) == harness.canonical_json(uninterrupted["attachment_state"])
    assert resumed["duration_qualified_proxy_pass"] == uninterrupted["duration_qualified_proxy_pass"]
    assert [(r["session_id"], r["scoring"]) for r in resumed["per_session"]] == [
        (r["session_id"], r["scoring"]) for r in uninterrupted["per_session"]
    ]


@pytest.mark.parametrize("kind", ["user", "state", "boolean_count", "ledger", "next_index", "clock", "path", "unknown"])
def test_checkpoint_validation_rejects_rehashed_invalid_payloads(tmp_path, kind):
    manifest, path = paused_checkpoint(tmp_path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    if kind == "user":
        payload["user_id"] = "other-user"
    elif kind == "state":
        payload["state"]["trust_ema"] = 1.0
    elif kind == "boolean_count":
        payload["state"]["cumulative_repair_turns"] = False
    elif kind == "ledger":
        payload["ledger"][1] = payload["ledger"][0]
    elif kind == "next_index":
        payload["next_index"] = 1
    elif kind == "clock":
        payload["simulated_clock"] = None
    elif kind == "path":
        payload["memory_file"] = "../outside.json"
    elif kind == "unknown":
        payload["extra"] = 1
    envelope["sha256"] = harness.digest(payload)
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError):
        harness.load_checkpoint(path, manifest)


def test_checkpoint_configuration_and_manifest_changes_are_rejected(tmp_path, monkeypatch):
    manifest, path = paused_checkpoint(tmp_path)
    changed = altered(manifest, lambda p: p.update(seed=1))
    with pytest.raises(ValueError, match="configuration mismatch"):
        harness.load_checkpoint(path, changed)
    monkeypatch.setattr(harness.attachment, "_TRUST_EMA_ALPHA", 0.5)
    with pytest.raises(ValueError, match="configuration mismatch"):
        harness.load_checkpoint(path, manifest)


def test_truncated_checkpoint_and_corrupt_memory_are_rejected(tmp_path):
    manifest, path = paused_checkpoint(tmp_path)
    valid = path.read_text(encoding="utf-8")
    path.write_text(valid[:50], encoding="utf-8")
    with pytest.raises(ValueError):
        harness.load_checkpoint(path, manifest)
    path.write_text(valid, encoding="utf-8")
    payload = json.loads(valid)["payload"]
    memory = path.with_name(payload["memory_file"])
    artifact = json.loads(memory.read_text(encoding="utf-8"))
    artifact["fragments"][0]["content"] = "altered content"
    memory.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="memory checksum"):
        harness.load_checkpoint(path, manifest)


def test_failed_session_does_not_replace_an_accepted_checkpoint(tmp_path, monkeypatch):
    manifest, path = paused_checkpoint(tmp_path)
    accepted_bytes = path.read_bytes()
    original = harness.run_msc_session

    def interrupted(session, *args, **kwargs):
        if session.session_id == 9:
            raise RuntimeError("session interrupted")
        return original(session, *args, **kwargs)

    monkeypatch.setattr(harness, "run_msc_session", interrupted)
    interrupted_result = harness.run_trajectory(manifest, checkpoint_path=path, resume=True)
    assert interrupted_result.state.sessions_completed == 8
    assert path.read_bytes() == accepted_bytes
    state, ledger, _ = harness.load_checkpoint(path, manifest)
    assert state.sessions_completed == len(ledger) == 8
    monkeypatch.setattr(harness, "run_msc_session", original)
    resumed = harness.run_trajectory(manifest, checkpoint_path=path, resume=True)
    assert resumed.state.sessions_completed == 16
    assert len({r["session_id"] for r in resumed.ledger}) == 16


def test_checkpoint_write_failure_does_not_double_count_the_current_session(tmp_path, monkeypatch):
    def failed_write(*_):
        raise OSError("injected disk failure")

    monkeypatch.setattr(harness, "save_checkpoint", failed_write)
    result = harness.run_trajectory(harness.build_trajectory("supportive-en"),
                                    checkpoint_path=tmp_path / "checkpoint.json")
    report = result.report()
    assert report["completed_sessions"] == 7
    assert report["attempted_sessions"] == 8
    assert report["failed_sessions"] == 1
    assert len({r["session_id"] for r in result.ledger}) == 8
    assert result.state.sessions_completed == 7


def test_memory_export_preserves_a_real_reconsolidation_parent_chain():
    system = CompanionSystem(byte_cap=harness.BYTE_CAP)
    system.observe(Observation(
        timestamp=harness.BASE_TIME, speaker="user", content="Private continuity marker amberorchid.",
        tags=("user:alice", "speaker:user"),
    ))
    parent = system.companion.store.all()[0]
    system.rerender(parent.id, "A new meaning for amberorchid.",
                    timestamp=harness.BASE_TIME + timedelta(days=1), tags=("user:alice",))
    checks = harness.memory_round_trip_checks(system, "amberorchid", "alice")
    assert checks["fragment_count"] == 2
    assert checks["fragment_set_preserved"]
    assert checks["parent_chains_preserved"]
    assert checks["content_addressing_valid"]
    assert checks["sentinel_recall_after_restore"]
    assert not checks["runtime_equivalence_claimed"]


def test_cli_refuses_to_overwrite_existing_evidence_on_a_second_run(tmp_path):
    command = [sys.executable, str(SCRIPT), "--scenario", "supportive-en", "--msc-limit", "0",
               "--output-dir", str(tmp_path)]
    first = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert first.returncode == 0, first.stdout + first.stderr
    original_summary = (tmp_path / "summary.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert second.returncode == 2
    assert (tmp_path / "summary.json").read_bytes() == original_summary


def test_cli_keeps_invalid_selected_sources_in_the_denominator(tmp_path):
    source = tmp_path / "msc.jsonl"
    source.write_text(json.dumps({
        "conversation_id": "missing-self-opener", "personas": [[], []],
        "sessions": [{"session_id": 1, "time_interval_days": 0,
                      "turns": [{"speaker_idx": 1, "utterance": "Other speaker only."}]}],
    }) + "\n", encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--scenario", "supportive-en", "--msc-limit", "1",
         "--msc-shim", str(source), "--output-dir", str(tmp_path / "output")],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert process.returncode == 1, process.stdout + process.stderr
    summary = json.loads((tmp_path / "output/summary.json").read_text(encoding="utf-8"))
    assert summary["selected_trajectories"] == 2
    assert summary["reported_trajectories"] == 1
    assert len(summary["setup_errors"]) == 1
    assert summary["setup_errors"][0]["source_conversation_id"] == "missing-self-opener"
