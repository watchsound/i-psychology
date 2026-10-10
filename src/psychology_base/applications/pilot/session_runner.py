"""Interactive REPL + session-log writer shared by TutorCLI and CompanionCLI.

The two app-specific CLIs differ in their *agent configuration* (safety
surface, prompt, telemetry) but share the same session shape: enrol →
read-respond loop → session-end PAM export. This module hosts the shape
so the two CLIs don't drift.

Design notes:

- The REPL does NOT know what agent is behind it. ``run_session`` takes
  an ``agent_fn: (user_utterance) -> (response_text, metadata_dict)``
  closure the CLIs build. This keeps the TutorAgent / CompanionAgent
  types out of this module (and avoids a cycle through the agent
  modules).
- Slash commands (``/quit``, ``/export``, ``/boundary``, ``/help``) are
  handled before the agent sees them. ``/export`` and session-end PAM
  export both call back into ``on_export`` the CLI provides — this
  module does not import the PAM subsystem.
- The session log is one JSONL file per session. One line per turn
  (user then agent) plus bookend lines for session start and end. The
  log is pilot auditor-facing; the memory substrate's internal
  fragment store is distinct and exported separately via PAM.
- Streams (``input_stream``, ``output_stream``) are injectable so tests
  can drive the loop without stdin/stdout. In production both default
  to ``sys.stdin`` / ``sys.stdout``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TextIO


SLASH_QUIT = "/quit"
SLASH_EXPORT = "/export"
SLASH_BOUNDARY = "/boundary"
SLASH_HELP = "/help"


SessionAgentCallback = Callable[[str], tuple[str, Mapping[str, Any]]]
"""Signature: ``(user_utterance) -> (response_text, metadata)``.

``metadata`` is logged verbatim alongside the agent turn — the CLIs put
tutor_move / reminder_fired / handoff_triggered etc. here. Keep it
JSON-serializable."""


SessionEndCallback = Callable[[], None]
"""Hook invoked once the REPL exits (clean quit or EOF). The CLI uses
this to run PAM auto-export + any end-of-session telemetry."""


ExportCallback = Callable[[Path], int]
"""Signature for the ``/export`` slash-command. Returns the number of
fragments written to the given path."""


BoundaryCallback = Callable[[], None]
"""Hook invoked when the user types ``/boundary`` to trigger the
consolidation path mid-session."""


@dataclass
class SessionLogWriter:
    """Append-only JSONL writer for one pilot session.

    One line per event. Opens the file lazily on first write so a
    session that aborts before any turn (crisis-interrupt at enrol,
    e.g.) does not leave behind an empty log.
    """

    path: Path
    _fh: Optional[TextIO] = None

    def _open(self) -> TextIO:
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        return self._fh

    def write(self, record: Mapping[str, Any]) -> None:
        fh = self._open()
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "SessionLogWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


HELP_TEXT = (
    f"Commands:\n"
    f"  {SLASH_QUIT}             end this session (auto-saves memory via PAM)\n"
    f"  {SLASH_EXPORT} <path>    export a PAM-v1 artefact to <path> now\n"
    f"  {SLASH_BOUNDARY}         mark a within-session consolidation boundary\n"
    f"  {SLASH_HELP}             show this help\n"
    f"\n"
    f"Anything else is sent to the agent."
)


def _is_blank(text: str) -> bool:
    return text.strip() == ""


def run_session(
    *,
    user_id: str,
    session_idx: int,
    agent_fn: SessionAgentCallback,
    log_writer: SessionLogWriter,
    on_export: ExportCallback,
    on_boundary: Optional[BoundaryCallback] = None,
    on_end: Optional[SessionEndCallback] = None,
    welcome_text: str = "",
    prompt: str = "you> ",
    agent_label: str = "agent> ",
    input_stream: Optional[TextIO] = None,
    output_stream: Optional[TextIO] = None,
) -> int:
    """Drive one interactive pilot session.

    Returns the number of agent turns produced (not counting slash
    commands or blank lines).

    The loop:

    1. Write welcome text (if any) to ``output_stream`` and log a
       ``session_start`` record.
    2. Loop over ``input_stream`` lines:
       - EOF → clean quit.
       - Blank line → ignore (do not fire the agent on empty input).
       - ``/quit`` → clean quit.
       - ``/help`` → print HELP_TEXT and continue.
       - ``/boundary`` → call ``on_boundary`` (if provided).
       - ``/export <path>`` → call ``on_export(path)``; print count.
       - Anything else → call ``agent_fn``, log both the user turn
         and the agent turn, print the agent response.
    3. On exit, log ``session_end`` and call ``on_end`` (if provided).

    All agent exceptions propagate — the pilot *wants* a crash log over
    a silently-swallowed error. Only slash-command parse errors are
    reported inline.
    """
    stdin = input_stream if input_stream is not None else sys.stdin
    stdout = output_stream if output_stream is not None else sys.stdout

    log_writer.write(
        {
            "event": "session_start",
            "user_id": user_id,
            "session_idx": session_idx,
            "ts": _utc_now_iso(),
        }
    )

    if welcome_text:
        stdout.write(welcome_text.rstrip() + "\n")
        stdout.flush()

    turn_count = 0
    try:
        while True:
            stdout.write(prompt)
            stdout.flush()
            raw = stdin.readline()
            if raw == "":
                # EOF — treat as /quit
                break
            line = raw.rstrip("\n").rstrip("\r")
            if _is_blank(line):
                continue
            if line == SLASH_QUIT:
                break
            if line == SLASH_HELP:
                stdout.write(HELP_TEXT + "\n")
                stdout.flush()
                continue
            if line == SLASH_BOUNDARY:
                if on_boundary is not None:
                    on_boundary()
                log_writer.write(
                    {
                        "event": "boundary",
                        "ts": _utc_now_iso(),
                    }
                )
                stdout.write("(session boundary marked)\n")
                stdout.flush()
                continue
            if line.startswith(SLASH_EXPORT):
                rest = line[len(SLASH_EXPORT) :].strip()
                if not rest:
                    stdout.write(
                        f"Usage: {SLASH_EXPORT} <path>  (where to write the "
                        f"PAM-v1 artefact)\n"
                    )
                    stdout.flush()
                    continue
                out_path = Path(rest).expanduser()
                try:
                    frag_count = on_export(out_path)
                except (OSError, ValueError) as exc:
                    stdout.write(f"(export failed: {exc})\n")
                    stdout.flush()
                    continue
                log_writer.write(
                    {
                        "event": "export",
                        "path": str(out_path),
                        "fragment_count": frag_count,
                        "ts": _utc_now_iso(),
                    }
                )
                stdout.write(
                    f"(exported {frag_count} fragments to {out_path})\n"
                )
                stdout.flush()
                continue

            # Normal agent turn.
            log_writer.write(
                {
                    "event": "user_turn",
                    "turn_index": turn_count,
                    "text": line,
                    "ts": _utc_now_iso(),
                }
            )
            response_text, metadata = agent_fn(line)
            log_writer.write(
                {
                    "event": "agent_turn",
                    "turn_index": turn_count,
                    "text": response_text,
                    "metadata": dict(metadata),
                    "ts": _utc_now_iso(),
                }
            )
            stdout.write(agent_label + response_text + "\n")
            stdout.flush()
            turn_count += 1
    finally:
        log_writer.write(
            {
                "event": "session_end",
                "turn_count": turn_count,
                "ts": _utc_now_iso(),
            }
        )
        log_writer.close()
        if on_end is not None:
            on_end()

    return turn_count


__all__ = [
    "BoundaryCallback",
    "ExportCallback",
    "HELP_TEXT",
    "SLASH_BOUNDARY",
    "SLASH_EXPORT",
    "SLASH_HELP",
    "SLASH_QUIT",
    "SessionAgentCallback",
    "SessionEndCallback",
    "SessionLogWriter",
    "run_session",
]
