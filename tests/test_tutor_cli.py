"""TutorCLI — enrol, inspect, session (stub responder, no live LLM)."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Mapping

import pytest

from psychology_base.applications.learning import tutor_cli
from psychology_base.applications.learning.tutor_agent import TutorResponse
from psychology_base.applications.pilot.layout import (
    PilotApp,
    PilotLayout,
    load_manifest,
    pam_latest_path,
    pam_session_path,
    session_log_path,
)
from psychology_base.applications.pilot.session_runner import SLASH_QUIT


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = tutor_cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def test_enroll_writes_manifest(tmp_path: Path) -> None:
    rc, out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "enroll",
            "--user",
            "u01",
            "--policy",
            "S1",
            "--age-bracket",
            "26-45",
            "--region",
            "CN",
        ]
    )
    assert rc == 0, err
    layout = PilotLayout(pilot_root=tmp_path, app=PilotApp.LEARNING_TUTOR)
    manifest = load_manifest(layout, "u01")
    assert manifest.policy_id == "S1"
    assert manifest.region == "CN"
    assert manifest.app is PilotApp.LEARNING_TUTOR
    assert "wrote" in out


def test_enroll_rejects_duplicate_without_force(tmp_path: Path) -> None:
    base = [
        "--pilot-root",
        str(tmp_path),
        "enroll",
        "--user",
        "u01",
        "--policy",
        "S1",
        "--age-bracket",
        "26-45",
    ]
    _run_cli(base)
    rc, _out, err = _run_cli(base)
    assert rc == 2
    assert "already enrolled" in err


def test_enroll_force_overwrites_prior(tmp_path: Path) -> None:
    base = [
        "--pilot-root",
        str(tmp_path),
        "enroll",
        "--user",
        "u01",
        "--age-bracket",
        "26-45",
    ]
    _run_cli(base + ["--policy", "S1"])
    rc, _out, _err = _run_cli(base + ["--policy", "S2", "--force"])
    assert rc == 0
    layout = PilotLayout(pilot_root=tmp_path, app=PilotApp.LEARNING_TUTOR)
    assert load_manifest(layout, "u01").policy_id == "S2"


def test_enroll_rejects_bad_policy(tmp_path: Path) -> None:
    rc, _out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "enroll",
            "--user",
            "u01",
            "--policy",
            "S9",
            "--age-bracket",
            "26-45",
        ]
    )
    assert rc == 2
    assert "--policy must be one of" in err


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_reports_zero_sessions_right_after_enrol(tmp_path: Path) -> None:
    _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "enroll",
            "--user",
            "u01",
            "--policy",
            "S1",
            "--age-bracket",
            "26-45",
        ]
    )
    rc, out, _err = _run_cli(
        ["--pilot-root", str(tmp_path), "inspect", "--user", "u01"]
    )
    assert rc == 0
    report = json.loads(out)
    assert report["session_count"] == 0
    assert report["pam_latest_present"] is False
    assert report["pam_archive_count"] == 0


def test_inspect_missing_user_errors(tmp_path: Path) -> None:
    rc, _out, err = _run_cli(
        ["--pilot-root", str(tmp_path), "inspect", "--user", "nobody"]
    )
    assert rc == 2
    assert "No manifest" in err


# ---------------------------------------------------------------------------
# session (direct call to cmd_session with injected pieces)
# ---------------------------------------------------------------------------


def _enrolled_layout(tmp_path: Path, user: str = "u01") -> PilotLayout:
    _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "enroll",
            "--user",
            user,
            "--policy",
            "S1",
            "--age-bracket",
            "26-45",
        ]
    )
    return PilotLayout(pilot_root=tmp_path, app=PilotApp.LEARNING_TUTOR)


def test_session_runs_a_turn_with_stub_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _enrolled_layout(tmp_path)
    stdin = io.StringIO(f"what is a derivative?\n{SLASH_QUIT}\n")
    monkeypatch.setattr("sys.stdin", stdin)
    rc, out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "u01",
            "--mode",
            "stub",
        ]
    )
    assert rc == 0, err
    log_path = session_log_path(layout, "u01", 1)
    assert log_path.exists()
    log = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines()]
    user_turns = [e for e in log if e["event"] == "user_turn"]
    agent_turns = [e for e in log if e["event"] == "agent_turn"]
    assert len(user_turns) == 1
    assert len(agent_turns) == 1
    assert "tutor_move" in agent_turns[0]["metadata"]
    assert "stub responder" in agent_turns[0]["text"]
    assert pam_latest_path(layout, "u01").exists()
    assert pam_session_path(layout, "u01", 1).exists()
    assert "session saved" in out


def test_session_resumes_pam_on_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _enrolled_layout(tmp_path)
    # First session.
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(f"first turn\n{SLASH_QUIT}\n")
    )
    rc1, _o1, _e1 = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "u01",
            "--mode",
            "stub",
        ]
    )
    assert rc1 == 0
    assert pam_latest_path(layout, "u01").exists()
    # Second session — session index advances to 2 and PAM is restored.
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(f"second turn\n{SLASH_QUIT}\n")
    )
    rc2, _o2, _e2 = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "u01",
            "--mode",
            "stub",
        ]
    )
    assert rc2 == 0
    assert session_log_path(layout, "u01", 2).exists()
    assert pam_session_path(layout, "u01", 2).exists()


def test_session_missing_user_errors(tmp_path: Path) -> None:
    rc, _out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "ghost",
            "--mode",
            "stub",
        ]
    )
    assert rc == 2
    assert "No manifest" in err


# ---------------------------------------------------------------------------
# build_tutor_agent
# ---------------------------------------------------------------------------


def test_build_tutor_agent_composes_safety_surface(tmp_path: Path) -> None:
    layout = _enrolled_layout(tmp_path)
    manifest = load_manifest(layout, "u01")
    from psychology_base.companion.system import CompanionSystem
    from psychology_base.applications.pilot.layout import build_aging_policy

    system = CompanionSystem(aging_policy=build_aging_policy(manifest.policy_id))

    def _stub(system_prompt: str, context: str, user: str) -> str:
        return "ok"

    agent = tutor_cli.build_tutor_agent(
        manifest, system, llm_responder=_stub, session_id="u01-s0001"
    )
    assert agent.ai_identity_reminder is not None
    assert agent.minor_exclusion_gate is not None
    # Smoke call to prove end-to-end composition.
    resp = agent.respond(
        "what next?",
        attestation=tutor_cli._attestation_from_manifest(manifest),
    )
    assert isinstance(resp, TutorResponse)
    assert resp.response_text == "ok"
