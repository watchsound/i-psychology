"""Session runner — slash commands, agent flow, EOF handling, log shape."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping

from psychology_base.applications.pilot.session_runner import (
    HELP_TEXT,
    SLASH_BOUNDARY,
    SLASH_EXPORT,
    SLASH_HELP,
    SLASH_QUIT,
    SessionLogWriter,
    run_session,
)


def _echo_agent(utterance: str) -> tuple[str, Mapping[str, Any]]:
    return f"echo:{utterance}", {"len": len(utterance)}


def _read_log(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]


def _drive(
    tmp_path: Path,
    inputs: str,
    *,
    on_export=None,
    on_boundary=None,
    on_end=None,
) -> tuple[list[dict], str, int]:
    log_path = tmp_path / "session.jsonl"
    out = io.StringIO()
    with SessionLogWriter(path=log_path) as writer:
        turns = run_session(
            user_id="u",
            session_idx=1,
            agent_fn=_echo_agent,
            log_writer=writer,
            on_export=on_export or (lambda _p: 0),
            on_boundary=on_boundary,
            on_end=on_end,
            welcome_text="",
            input_stream=io.StringIO(inputs),
            output_stream=out,
        )
    return _read_log(log_path), out.getvalue(), turns


# ---------------------------------------------------------------------------
# Basic I/O
# ---------------------------------------------------------------------------


def test_single_turn_logs_user_and_agent(tmp_path: Path) -> None:
    log, out, turns = _drive(tmp_path, "hello\n")
    assert turns == 1
    events = [e["event"] for e in log]
    assert events == ["session_start", "user_turn", "agent_turn", "session_end"]
    assert "echo:hello" in out


def test_eof_quits_cleanly_with_zero_turns(tmp_path: Path) -> None:
    log, _out, turns = _drive(tmp_path, "")
    assert turns == 0
    assert log[0]["event"] == "session_start"
    assert log[-1]["event"] == "session_end"
    assert log[-1]["turn_count"] == 0


def test_blank_lines_do_not_fire_agent(tmp_path: Path) -> None:
    log, _out, turns = _drive(tmp_path, "\n   \n\n")
    assert turns == 0
    assert all(e["event"] in ("session_start", "session_end") for e in log)


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def test_slash_quit_ends_session_before_eof(tmp_path: Path) -> None:
    log, _out, turns = _drive(tmp_path, f"one\n{SLASH_QUIT}\n")
    assert turns == 1
    assert log[-1]["event"] == "session_end"


def test_slash_help_prints_help_and_does_not_log(tmp_path: Path) -> None:
    log, out, turns = _drive(tmp_path, f"{SLASH_HELP}\n")
    assert HELP_TEXT in out
    assert turns == 0
    assert all(e["event"] in ("session_start", "session_end") for e in log)


def test_slash_boundary_fires_callback_and_logs(tmp_path: Path) -> None:
    called = []
    log, out, _turns = _drive(
        tmp_path,
        f"{SLASH_BOUNDARY}\n",
        on_boundary=lambda: called.append(True),
    )
    assert called == [True]
    boundary_events = [e for e in log if e["event"] == "boundary"]
    assert len(boundary_events) == 1
    assert "session boundary marked" in out


def test_slash_export_invokes_callback_and_logs(tmp_path: Path) -> None:
    captured_path: list[Path] = []

    def _on_export(path: Path) -> int:
        captured_path.append(path)
        return 42

    log, out, _turns = _drive(
        tmp_path,
        f"{SLASH_EXPORT} {tmp_path / 'archive.json'}\n",
        on_export=_on_export,
    )
    assert len(captured_path) == 1
    export_events = [e for e in log if e["event"] == "export"]
    assert export_events[0]["fragment_count"] == 42
    assert "exported 42 fragments" in out


def test_slash_export_without_path_prints_usage_and_continues(tmp_path: Path) -> None:
    log, out, turns = _drive(tmp_path, f"{SLASH_EXPORT}\n")
    assert f"Usage: {SLASH_EXPORT}" in out
    assert turns == 0
    assert not any(e["event"] == "export" for e in log)


def test_slash_export_reports_failure_without_crashing(tmp_path: Path) -> None:
    def _fail(_path: Path) -> int:
        raise OSError("disk full")

    log, out, turns = _drive(
        tmp_path,
        f"{SLASH_EXPORT} {tmp_path / 'archive.json'}\n",
        on_export=_fail,
    )
    assert "export failed" in out
    assert turns == 0
    assert not any(e["event"] == "export" for e in log)


# ---------------------------------------------------------------------------
# session_end hook
# ---------------------------------------------------------------------------


def test_on_end_fires_once_after_session(tmp_path: Path) -> None:
    counter = {"n": 0}
    _drive(
        tmp_path,
        f"hello\n{SLASH_QUIT}\n",
        on_end=lambda: counter.__setitem__("n", counter["n"] + 1),
    )
    assert counter["n"] == 1


def test_on_end_fires_even_on_eof(tmp_path: Path) -> None:
    counter = {"n": 0}
    _drive(
        tmp_path,
        "",
        on_end=lambda: counter.__setitem__("n", counter["n"] + 1),
    )
    assert counter["n"] == 1


# ---------------------------------------------------------------------------
# SessionLogWriter
# ---------------------------------------------------------------------------


def test_session_log_writer_appends_in_order(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    with SessionLogWriter(path=path) as w:
        w.write({"event": "a"})
        w.write({"event": "b", "x": 1})
    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in lines] == ["a", "b"]
    assert lines[1]["x"] == 1


def test_session_log_writer_opens_lazily(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    with SessionLogWriter(path=path):
        pass
    # No write → no file.
    assert not path.exists()
