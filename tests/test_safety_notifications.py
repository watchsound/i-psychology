"""Notification scaffold — sinks, bridge, and crisis-handoff composition."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from psychology_base.applications.safety.crisis_escalation import (
    CrisisEscalation,
    CrisisIncident,
    CrisisSeverity,
)
from psychology_base.applications.safety.notifications import (
    AlibabaCloudSMSNotificationSink,
    ConsoleNotificationSink,
    JsonlFileNotificationSink,
    MultiNotificationSink,
    NotificationEvent,
    NotificationNotConfigured,
    NotificationResult,
    NotificationSeverity,
    SMTPEmailNotificationSink,
    TwilioSMSNotificationSink,
    build_crisis_handoff_sink,
    crisis_incident_to_notification_event,
)


# ---------------------------------------------------------------------------
# Console + file sinks
# ---------------------------------------------------------------------------


def _event(severity: NotificationSeverity = NotificationSeverity.CRITICAL) -> NotificationEvent:
    return NotificationEvent(
        severity=severity,
        title="t",
        body="b",
        user_id="u01",
        timestamp_seconds=1_700_000_000.0,
        metadata={"k": "v"},
    )


def test_console_sink_writes_to_stream() -> None:
    buf = io.StringIO()
    sink = ConsoleNotificationSink(stream=buf)
    result = sink.send(_event())
    assert result.ok
    out = buf.getvalue()
    assert "[NOTIFY CRITICAL]" in out
    assert "user_id=u01" in out


def test_jsonl_file_sink_appends_records(tmp_path: Path) -> None:
    path = tmp_path / "pager.jsonl"
    sink = JsonlFileNotificationSink(path=path)
    r1 = sink.send(_event(NotificationSeverity.WARNING))
    r2 = sink.send(_event(NotificationSeverity.CRITICAL))
    assert r1.ok and r2.ok
    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["severity"] == "warning"
    assert lines[1]["severity"] == "critical"
    assert lines[0]["metadata"] == {"k": "v"}


def test_jsonl_file_sink_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "deep" / "pager.jsonl"
    sink = JsonlFileNotificationSink(path=nested)
    result = sink.send(_event())
    assert result.ok
    assert nested.exists()


# ---------------------------------------------------------------------------
# Multi sink fan-out
# ---------------------------------------------------------------------------


class _FailingSink:
    name = "boom"

    def send(self, event: NotificationEvent) -> NotificationResult:
        raise RuntimeError("fire alarm")


class _NotConfiguredSink:
    name = "stub_provider"

    def send(self, event: NotificationEvent) -> NotificationResult:
        raise NotificationNotConfigured("no creds")


def test_multi_sink_ok_when_any_child_ok(tmp_path: Path) -> None:
    log = tmp_path / "a.jsonl"
    multi = MultiNotificationSink(
        sinks=(JsonlFileNotificationSink(path=log), _FailingSink())
    )
    result = multi.send(_event())
    assert result.ok
    # provider_message_id carries a JSON list of per-sink outcomes.
    outcomes = json.loads(result.provider_message_id)
    assert len(outcomes) == 2
    assert any(not o["ok"] and "fire alarm" in o["error"] for o in outcomes)


def test_multi_sink_fails_when_all_children_fail() -> None:
    multi = MultiNotificationSink(sinks=(_FailingSink(), _NotConfiguredSink()))
    result = multi.send(_event())
    assert result.ok is False
    assert result.error == "all child sinks failed"


def test_multi_sink_captures_not_configured_errors() -> None:
    multi = MultiNotificationSink(sinks=(_NotConfiguredSink(),))
    result = multi.send(_event())
    outcomes = json.loads(result.provider_message_id)
    assert outcomes[0]["ok"] is False
    assert "NotificationNotConfigured" in outcomes[0]["error"]


# ---------------------------------------------------------------------------
# Provider stubs — raise cleanly without credentials
# ---------------------------------------------------------------------------


def test_twilio_sink_without_env_raises_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER"):
        monkeypatch.delenv(v, raising=False)
    sink = TwilioSMSNotificationSink(to_numbers=("+13105551212",))
    with pytest.raises(NotificationNotConfigured, match="TWILIO"):
        sink.send(_event())


def test_aliyun_sink_without_env_raises_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ("ALIYUN_ACCESS_KEY_ID", "ALIYUN_ACCESS_KEY_SECRET", "ALIYUN_SMS_SIGN_NAME", "ALIYUN_SMS_TEMPLATE_CODE"):
        monkeypatch.delenv(v, raising=False)
    sink = AlibabaCloudSMSNotificationSink(to_numbers=("13800000000",))
    with pytest.raises(NotificationNotConfigured, match="ALIYUN"):
        sink.send(_event())


def test_smtp_sink_without_env_raises_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ("PSYCH_SMTP_HOST", "PSYCH_SMTP_PORT", "PSYCH_SMTP_USER", "PSYCH_SMTP_PASSWORD", "PSYCH_SMTP_FROM"):
        monkeypatch.delenv(v, raising=False)
    sink = SMTPEmailNotificationSink(to_addresses=("pi@example.org",))
    with pytest.raises(NotificationNotConfigured, match="SMTP"):
        sink.send(_event())


# ---------------------------------------------------------------------------
# Bridge from CrisisIncident → NotificationEvent
# ---------------------------------------------------------------------------


def test_crisis_incident_to_notification_event_maps_severity() -> None:
    incident = CrisisIncident(
        severity=CrisisSeverity.EXPLICIT_IDEATION,
        matched_cue="kill myself",
        utterance="i want to kill myself",
        timestamp_seconds=1700000000.0,
        user_id="c01",
    )
    event = crisis_incident_to_notification_event(incident, extra_metadata={"app": "companion"})
    assert event.severity is NotificationSeverity.CRITICAL
    assert event.user_id == "c01"
    assert "kill myself" in event.title
    assert "i want to kill myself" in event.body
    assert event.metadata["crisis_severity"] == "explicit_ideation"
    assert event.metadata["app"] == "companion"


def test_crisis_incident_ambiguous_maps_to_warning() -> None:
    incident = CrisisIncident(
        severity=CrisisSeverity.AMBIGUOUS,
        matched_cue="hopeless",
        utterance="i feel hopeless",
        timestamp_seconds=0.0,
        user_id="c02",
    )
    event = crisis_incident_to_notification_event(incident)
    assert event.severity is NotificationSeverity.WARNING


# ---------------------------------------------------------------------------
# build_crisis_handoff_sink end-to-end composition
# ---------------------------------------------------------------------------


def test_build_crisis_handoff_sink_writes_audit_and_fires_notifier(tmp_path: Path) -> None:
    audit = tmp_path / "pager-audit.jsonl"
    captured_results: list[NotificationResult] = []
    stream = io.StringIO()
    sink = build_crisis_handoff_sink(
        audit_log_path=audit,
        notifier=ConsoleNotificationSink(stream=stream),
        on_notification_result=captured_results.append,
        extra_metadata={"app": "companion", "cohort": "preview"},
    )
    incident = CrisisIncident(
        severity=CrisisSeverity.IMMINENT_DANGER,
        matched_cue="kill myself",
        utterance="i want to kill myself",
        timestamp_seconds=1700000000.0,
        user_id="c01",
    )
    sink(incident)

    rows = [json.loads(ln) for ln in audit.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["severity"] == "critical"
    assert rows[0]["metadata"]["cohort"] == "preview"
    assert captured_results and captured_results[0].ok
    assert "[NOTIFY CRITICAL]" in stream.getvalue()


def test_build_crisis_handoff_sink_captures_notifier_failure(tmp_path: Path) -> None:
    audit = tmp_path / "pager-audit.jsonl"
    captured: list[NotificationResult] = []
    sink = build_crisis_handoff_sink(
        audit_log_path=audit,
        notifier=_FailingSink(),
        on_notification_result=captured.append,
    )
    sink(
        CrisisIncident(
            severity=CrisisSeverity.EXPLICIT_IDEATION,
            matched_cue="kill myself",
            utterance="i want to kill myself",
            timestamp_seconds=0.0,
            user_id="c01",
        )
    )
    assert audit.exists()
    assert captured and captured[0].ok is False
    assert "fire alarm" in captured[0].error


def test_build_crisis_handoff_sink_integrates_with_crisis_escalation(tmp_path: Path) -> None:
    audit = tmp_path / "pager-audit.jsonl"
    stream = io.StringIO()
    handoff = build_crisis_handoff_sink(
        audit_log_path=audit, notifier=ConsoleNotificationSink(stream=stream)
    )
    gate = CrisisEscalation(handoff_sink=handoff)
    incident = gate.scan(
        "i want to kill myself", timestamp_seconds=0.0, user_id="c01"
    )
    assert incident is not None
    assert audit.exists()
    assert "[NOTIFY CRITICAL]" in stream.getvalue()
