"""CompanionCLI — enrol gate, inspect, session + crisis handoff."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from psychology_base.applications.emotional import companion_cli
from psychology_base.applications.emotional.companion_agent import HANDOFF_SCRIPT
from psychology_base.applications.pilot.layout import (
    PilotApp,
    PilotLayout,
    load_manifest,
    pam_latest_path,
    pam_session_path,
    session_log_path,
)
from psychology_base.applications.pilot.session_runner import SLASH_QUIT


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = companion_cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# enroll — includes minor-exclusion defence
# ---------------------------------------------------------------------------


def test_enroll_writes_companion_manifest(tmp_path: Path) -> None:
    rc, _out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "enroll",
            "--user",
            "c01",
            "--policy",
            "S2",
            "--age-bracket",
            "26-45",
            "--region",
            "CN",
        ]
    )
    assert rc == 0, err
    layout = PilotLayout(
        pilot_root=tmp_path, app=PilotApp.EMOTIONAL_COMPANION
    )
    manifest = load_manifest(layout, "c01")
    assert manifest.app is PilotApp.EMOTIONAL_COMPANION
    assert manifest.policy_id == "S2"


def test_enroll_rejects_invalid_policy(tmp_path: Path) -> None:
    rc, _out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "enroll",
            "--user",
            "c01",
            "--policy",
            "S9",
            "--age-bracket",
            "26-45",
        ]
    )
    assert rc == 2
    assert "--policy must be one of" in err


# ---------------------------------------------------------------------------
# session + crisis handoff
# ---------------------------------------------------------------------------


def _enrolled_layout(tmp_path: Path, user: str = "c01") -> PilotLayout:
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
    return PilotLayout(
        pilot_root=tmp_path, app=PilotApp.EMOTIONAL_COMPANION
    )


def test_session_runs_a_turn_and_saves_pam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _enrolled_layout(tmp_path)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(f"i'm feeling a bit stuck today\n{SLASH_QUIT}\n"),
    )
    rc, out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "c01",
            "--mode",
            "stub",
        ]
    )
    assert rc == 0, err
    log_path = session_log_path(layout, "c01", 1)
    log = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines()]
    agent_turns = [e for e in log if e["event"] == "agent_turn"]
    assert len(agent_turns) == 1
    md = agent_turns[0]["metadata"]
    assert md["handoff_triggered"] is False
    assert md["crisis_severity"] is None
    assert pam_latest_path(layout, "c01").exists()
    assert pam_session_path(layout, "c01", 1).exists()
    assert "session saved" in out


def test_session_crisis_cue_triggers_handoff_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _enrolled_layout(tmp_path)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(f"i want to kill myself\n{SLASH_QUIT}\n"),
    )
    rc, _out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "c01",
            "--mode",
            "stub",
        ]
    )
    assert rc == 0, err
    log_path = session_log_path(layout, "c01", 1)
    log = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines()]
    agent_turns = [e for e in log if e["event"] == "agent_turn"]
    assert agent_turns[0]["text"] == HANDOFF_SCRIPT
    assert agent_turns[0]["metadata"]["handoff_triggered"] is True
    assert agent_turns[0]["metadata"]["crisis_severity"] is not None
    crisis_log = layout.user_dir("c01") / companion_cli.CRISIS_LOG_FILENAME
    assert crisis_log.exists()
    rows = [json.loads(ln) for ln in crisis_log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["user_id"] == "c01"
    assert rows[0]["utterance"] == "i want to kill myself"


# ---------------------------------------------------------------------------
# inspect reflects crisis log state
# ---------------------------------------------------------------------------


def test_inspect_reports_crisis_log_absent_initially(tmp_path: Path) -> None:
    _enrolled_layout(tmp_path)
    rc, out, _err = _run_cli(
        ["--pilot-root", str(tmp_path), "inspect", "--user", "c01"]
    )
    assert rc == 0
    report = json.loads(out)
    assert report["crisis_log_present"] is False


def test_inspect_reports_crisis_log_after_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enrolled_layout(tmp_path)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(f"i want to kill myself\n{SLASH_QUIT}\n"),
    )
    _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "c01",
            "--mode",
            "stub",
        ]
    )
    rc, out, _err = _run_cli(
        ["--pilot-root", str(tmp_path), "inspect", "--user", "c01"]
    )
    assert rc == 0
    report = json.loads(out)
    assert report["crisis_log_present"] is True


# ---------------------------------------------------------------------------
# crisis sink unit
# ---------------------------------------------------------------------------


def test_make_crisis_sink_appends_jsonl_record(tmp_path: Path) -> None:
    from psychology_base.applications.safety.crisis_escalation import (
        CrisisIncident,
        CrisisSeverity,
    )

    sink = companion_cli._make_crisis_sink(tmp_path / "crisis.jsonl")
    incident = CrisisIncident(
        severity=CrisisSeverity.EXPLICIT_IDEATION,
        matched_cue="kill myself",
        utterance="i want to kill myself",
        timestamp_seconds=1700000000.0,
        user_id="c01",
    )
    sink(incident)
    sink(incident)
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "crisis.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["matched_cue"] == "kill myself"
    assert rows[0]["severity"] == "explicit_ideation"


# ---------------------------------------------------------------------------
# --notify-provider wiring
# ---------------------------------------------------------------------------


def test_build_notifier_requires_recipient_for_sms() -> None:
    with pytest.raises(ValueError, match="requires at least one --notify-to"):
        companion_cli._build_notifier("twilio", [])


def test_build_notifier_console_needs_no_recipient() -> None:
    from psychology_base.applications.safety.notifications import (
        ConsoleNotificationSink,
    )

    sink = companion_cli._build_notifier("console", [])
    assert isinstance(sink, ConsoleNotificationSink)


def test_session_notify_console_writes_pager_audit_on_crisis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _enrolled_layout(tmp_path)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(f"i want to kill myself\n{SLASH_QUIT}\n"),
    )
    rc, _out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "c01",
            "--mode",
            "stub",
            "--notify-provider",
            "console",
        ]
    )
    assert rc == 0, err
    # Legacy crisis-log preserved.
    crisis_log = layout.user_dir("c01") / companion_cli.CRISIS_LOG_FILENAME
    assert crisis_log.exists()
    # Pager-audit log additionally written in NotificationEvent schema.
    pager = layout.user_dir("c01") / companion_cli.PAGER_AUDIT_FILENAME
    assert pager.exists()
    rows = [json.loads(ln) for ln in pager.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["severity"] == "critical"
    assert rows[0]["user_id"] == "c01"
    assert rows[0]["metadata"]["app"] == "companion"
    assert rows[0]["metadata"]["pilot_policy"] == "S1"


def test_session_notify_none_leaves_pager_audit_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _enrolled_layout(tmp_path)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(f"i want to kill myself\n{SLASH_QUIT}\n"),
    )
    rc, _out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "c01",
            "--mode",
            "stub",
        ]
    )
    assert rc == 0, err
    pager = layout.user_dir("c01") / companion_cli.PAGER_AUDIT_FILENAME
    assert not pager.exists()


def test_session_notify_twilio_without_recipients_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enrolled_layout(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{SLASH_QUIT}\n"))
    rc, _out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "c01",
            "--mode",
            "stub",
            "--notify-provider",
            "twilio",
        ]
    )
    assert rc == 2
    assert "--notify-to" in err


def test_session_notify_custom_audit_log_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enrolled_layout(tmp_path)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(f"i want to kill myself\n{SLASH_QUIT}\n"),
    )
    custom = tmp_path / "custom-pager.jsonl"
    rc, _out, err = _run_cli(
        [
            "--pilot-root",
            str(tmp_path),
            "session",
            "--user",
            "c01",
            "--mode",
            "stub",
            "--notify-provider",
            "console",
            "--notify-audit-log",
            str(custom),
        ]
    )
    assert rc == 0, err
    assert custom.exists()
