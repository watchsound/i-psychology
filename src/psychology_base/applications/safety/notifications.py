"""Notification scaffold for Phase 4b safety handoff — §11 item 4.

Per `docs/phases/phase-4b-signoff.md` §9.6b: when a crisis incident
fires, the PI needs a Tier-1 pager alert within 2 h. User has not
picked the provider at the time of this scaffold (Twilio / Alibaba
Cloud SMS / email+SMS / other), so we ship the abstraction + wiring
and leave the provider-specific credentials-and-API code as plug-in
sinks that fail loudly if invoked without configuration.

Shape:

- `NotificationEvent` — the normalised event the system emits.
- `NotificationSink` — protocol a provider implements.
- Three default sinks ship:
    - `ConsoleNotificationSink` (always safe; dev + CI default)
    - `JsonlFileNotificationSink` (durable local audit trail; always on
      in parallel with any live provider)
    - `MultiNotificationSink` (fan-out; collects per-sink results)
- Three provider sinks ship as **ready-to-configure stubs**:
    - `TwilioSMSNotificationSink` (uses twilio-sdk if installed; raises
      `NotificationNotConfigured` otherwise)
    - `AlibabaCloudSMSNotificationSink` (uses aliyun-python-sdk-core if
      installed; raises `NotificationNotConfigured` otherwise)
    - `SMTPEmailNotificationSink` (uses stdlib `smtplib`; no external
      dependency — ready to use once SMTP credentials are supplied)
- `crisis_incident_to_notification_event()` — bridges `CrisisIncident`
  from `safety.crisis_escalation`.
- `build_crisis_notification_sink()` — convenience constructor the
  CLI calls; composes `JsonlFileNotificationSink` + chosen provider
  into a `MultiNotificationSink` returning a `handoff_sink`-compatible
  callable.

Deliberate non-goals in this scaffold:

- No retry logic (fire-and-forget; failures are logged, not retried).
  Durable retry needs a persistent queue; outside §11 item 4 scope.
- No rate-limit on paging. A storm of crisis events in one session is
  itself a signal — not something to suppress.
- No per-recipient routing (one event → all configured sinks). If the
  PI wants different events to go to different humans, that's a
  product feature for Phase 5.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from psychology_base.applications.safety.crisis_escalation import (
    CrisisIncident,
    CrisisSeverity,
)


class NotificationSeverity(str, Enum):
    """Pager severity. Maps from `CrisisSeverity` via the bridge helper."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_CRISIS_TO_NOTIFY: Mapping[CrisisSeverity, NotificationSeverity] = {
    CrisisSeverity.NONE: NotificationSeverity.INFO,
    CrisisSeverity.AMBIGUOUS: NotificationSeverity.WARNING,
    CrisisSeverity.EXPLICIT_IDEATION: NotificationSeverity.CRITICAL,
    CrisisSeverity.IMMINENT_DANGER: NotificationSeverity.CRITICAL,
}


@dataclass(frozen=True)
class NotificationEvent:
    """A pager-level event emitted by the safety layer.

    `title` is the human-readable one-liner shown in SMS / push preview.
    `body` is the full message body (longer; used in email / audit log).
    `metadata` is provider-agnostic key/value; providers may ignore it.
    """

    severity: NotificationSeverity
    title: str
    body: str
    user_id: Optional[str] = None
    timestamp_seconds: float = field(default_factory=lambda: time.time())
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotificationResult:
    """Per-sink outcome of a `send()` call."""

    sink_name: str
    ok: bool
    provider_message_id: Optional[str] = None
    error: Optional[str] = None


class NotificationNotConfigured(RuntimeError):
    """A provider sink was invoked without the credentials / SDK it needs."""


class NotificationSink(Protocol):
    """Protocol — provider sinks implement this."""

    name: str

    def send(self, event: NotificationEvent) -> NotificationResult:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Built-in sinks (always safe to construct)
# ---------------------------------------------------------------------------


@dataclass
class ConsoleNotificationSink:
    """Stderr writer. Default when nothing else is configured."""

    name: str = "console"
    stream: Any = None

    def send(self, event: NotificationEvent) -> NotificationResult:
        stream = self.stream if self.stream is not None else sys.stderr
        stream.write(
            f"[NOTIFY {event.severity.value.upper()}] {event.title}\n"
            f"  user_id={event.user_id} ts={event.timestamp_seconds:.0f}\n"
            f"  body: {event.body}\n"
        )
        stream.flush()
        return NotificationResult(sink_name=self.name, ok=True)


@dataclass
class JsonlFileNotificationSink:
    """Appends events as JSONL. Durable local audit trail — recommended as
    the always-on second sink regardless of which cloud provider is wired.

    Parent directory is created on first write. One line per event.
    """

    path: Path
    name: str = "jsonl_file"

    def send(self, event: NotificationEvent) -> NotificationResult:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "severity": event.severity.value,
                            "title": event.title,
                            "body": event.body,
                            "user_id": event.user_id,
                            "timestamp_seconds": event.timestamp_seconds,
                            "metadata": dict(event.metadata),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return NotificationResult(sink_name=self.name, ok=True)
        except OSError as exc:
            return NotificationResult(
                sink_name=self.name, ok=False, error=f"{type(exc).__name__}: {exc}"
            )


@dataclass
class MultiNotificationSink:
    """Fan-out sink. Collects per-child results; one child's failure does
    not short-circuit the others. The PI reads the result list to decide
    whether to escalate to a backup channel."""

    sinks: tuple[NotificationSink, ...]
    name: str = "multi"

    def send(self, event: NotificationEvent) -> NotificationResult:
        results: list[NotificationResult] = []
        for sink in self.sinks:
            try:
                results.append(sink.send(event))
            except NotificationNotConfigured as exc:
                results.append(
                    NotificationResult(
                        sink_name=sink.name,
                        ok=False,
                        error=f"NotificationNotConfigured: {exc}",
                    )
                )
            except Exception as exc:
                results.append(
                    NotificationResult(
                        sink_name=sink.name,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        any_ok = any(r.ok for r in results)
        return NotificationResult(
            sink_name=self.name,
            ok=any_ok,
            error=None if any_ok else "all child sinks failed",
            provider_message_id=json.dumps(
                [
                    {"sink": r.sink_name, "ok": r.ok, "error": r.error}
                    for r in results
                ]
            ),
        )


# ---------------------------------------------------------------------------
# Provider stubs — the user picks one at pilot-start and the credentials
# land via env vars.
# ---------------------------------------------------------------------------


@dataclass
class TwilioSMSNotificationSink:
    """Twilio SMS sink. Reads credentials from env vars at send-time.

    Required env vars:
      - TWILIO_ACCOUNT_SID
      - TWILIO_AUTH_TOKEN
      - TWILIO_FROM_NUMBER (E.164)

    Requires the ``twilio`` SDK: ``pip install twilio``.

    The constructor is cheap and does NOT validate credentials; validation
    runs at ``send()`` time so a CI-mode CLI boot does not need credentials
    to construct. Boot-time validation would couple test ergonomics to
    pilot-time secrets.
    """

    to_numbers: tuple[str, ...]
    name: str = "twilio_sms"

    def send(self, event: NotificationEvent) -> NotificationResult:
        sid = os.environ.get("TWILIO_ACCOUNT_SID")
        token = os.environ.get("TWILIO_AUTH_TOKEN")
        from_num = os.environ.get("TWILIO_FROM_NUMBER")
        if not (sid and token and from_num):
            raise NotificationNotConfigured(
                "Twilio sink requires TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM_NUMBER env vars."
            )
        try:
            from twilio.rest import Client  # type: ignore
        except ImportError as exc:
            raise NotificationNotConfigured(
                "Twilio sink requires the `twilio` package (`pip install twilio`)."
            ) from exc
        client = Client(sid, token)
        msg_body = f"[{event.severity.value.upper()}] {event.title}\n{event.body}"
        message_ids: list[str] = []
        for to in self.to_numbers:
            msg = client.messages.create(body=msg_body, from_=from_num, to=to)
            message_ids.append(str(getattr(msg, "sid", "unknown")))
        return NotificationResult(
            sink_name=self.name,
            ok=True,
            provider_message_id=",".join(message_ids),
        )


@dataclass
class AlibabaCloudSMSNotificationSink:
    """Alibaba Cloud (阿里云短信) SMS sink.

    Required env vars:
      - ALIYUN_ACCESS_KEY_ID
      - ALIYUN_ACCESS_KEY_SECRET
      - ALIYUN_SMS_SIGN_NAME   (阿里云短信签名)
      - ALIYUN_SMS_TEMPLATE_CODE  (template that includes {severity}/{title}/{body})

    Requires `aliyun-python-sdk-core` + `aliyun-python-sdk-dysmsapi`.

    Same lazy-validation pattern as the Twilio sink.
    """

    to_numbers: tuple[str, ...]
    name: str = "aliyun_sms"

    def send(self, event: NotificationEvent) -> NotificationResult:
        ak = os.environ.get("ALIYUN_ACCESS_KEY_ID")
        sk = os.environ.get("ALIYUN_ACCESS_KEY_SECRET")
        sign = os.environ.get("ALIYUN_SMS_SIGN_NAME")
        tpl = os.environ.get("ALIYUN_SMS_TEMPLATE_CODE")
        if not (ak and sk and sign and tpl):
            raise NotificationNotConfigured(
                "Alibaba Cloud SMS sink requires ALIYUN_ACCESS_KEY_ID / "
                "ALIYUN_ACCESS_KEY_SECRET / ALIYUN_SMS_SIGN_NAME / ALIYUN_SMS_TEMPLATE_CODE env vars."
            )
        try:
            from aliyunsdkcore.client import AcsClient  # type: ignore
            from aliyunsdkcore.request import CommonRequest  # type: ignore
        except ImportError as exc:
            raise NotificationNotConfigured(
                "Alibaba Cloud SMS sink requires `aliyun-python-sdk-core` "
                "(`pip install aliyun-python-sdk-core`)."
            ) from exc
        client = AcsClient(ak, sk, "cn-hangzhou")
        request = CommonRequest()
        request.set_accept_format("json")
        request.set_domain("dysmsapi.aliyuncs.com")
        request.set_method("POST")
        request.set_version("2017-05-25")
        request.set_action_name("SendSms")
        params = json.dumps(
            {
                "severity": event.severity.value,
                "title": event.title[:60],
                "body": event.body[:300],
            },
            ensure_ascii=False,
        )
        message_ids: list[str] = []
        for to in self.to_numbers:
            request.add_query_param("PhoneNumbers", to)
            request.add_query_param("SignName", sign)
            request.add_query_param("TemplateCode", tpl)
            request.add_query_param("TemplateParam", params)
            response = client.do_action_with_exception(request)
            parsed = json.loads(response)
            message_ids.append(str(parsed.get("BizId", "unknown")))
        return NotificationResult(
            sink_name=self.name,
            ok=True,
            provider_message_id=",".join(message_ids),
        )


@dataclass
class SMTPEmailNotificationSink:
    """SMTP email sink using the stdlib `smtplib`. No external dependency.

    Required env vars:
      - PSYCH_SMTP_HOST
      - PSYCH_SMTP_PORT (int)
      - PSYCH_SMTP_USER
      - PSYCH_SMTP_PASSWORD
      - PSYCH_SMTP_FROM
    """

    to_addresses: tuple[str, ...]
    name: str = "smtp_email"
    use_tls: bool = True

    def send(self, event: NotificationEvent) -> NotificationResult:
        host = os.environ.get("PSYCH_SMTP_HOST")
        port_s = os.environ.get("PSYCH_SMTP_PORT")
        user = os.environ.get("PSYCH_SMTP_USER")
        pw = os.environ.get("PSYCH_SMTP_PASSWORD")
        sender = os.environ.get("PSYCH_SMTP_FROM")
        if not (host and port_s and user and pw and sender):
            raise NotificationNotConfigured(
                "SMTP email sink requires PSYCH_SMTP_HOST / PORT / USER / PASSWORD / FROM env vars."
            )
        try:
            port = int(port_s)
        except ValueError as exc:
            raise NotificationNotConfigured(
                f"PSYCH_SMTP_PORT={port_s!r} is not an integer."
            ) from exc
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = f"[{event.severity.value.upper()}] {event.title}"
        msg["From"] = sender
        msg["To"] = ", ".join(self.to_addresses)
        msg.set_content(
            f"Severity: {event.severity.value}\n"
            f"Title: {event.title}\n"
            f"User: {event.user_id}\n"
            f"Timestamp: {event.timestamp_seconds}\n\n"
            f"{event.body}\n\n"
            f"Metadata: {json.dumps(dict(event.metadata), ensure_ascii=False, indent=2)}\n"
        )
        with smtplib.SMTP(host, port) as smtp:
            if self.use_tls:
                smtp.starttls()
            smtp.login(user, pw)
            smtp.send_message(msg)
        return NotificationResult(sink_name=self.name, ok=True)


# ---------------------------------------------------------------------------
# Bridge: CrisisIncident → NotificationEvent → `handoff_sink` callable
# ---------------------------------------------------------------------------


def crisis_incident_to_notification_event(
    incident: CrisisIncident, *, extra_metadata: Optional[Mapping[str, Any]] = None
) -> NotificationEvent:
    """Normalise a `CrisisIncident` into a pager-level `NotificationEvent`.

    Title is kept short enough for an SMS preview; body carries the
    offending utterance (truncated) and the matched cue. The utterance
    is included under the assumption that the Tier-1 pager recipient
    (the PI) is the only reader and has the right to see raw content
    for triage.
    """
    sev = _CRISIS_TO_NOTIFY[incident.severity]
    title = f"Crisis cue '{incident.matched_cue}' ({incident.severity.value})"
    body = (
        f"User: {incident.user_id or '(unknown)'}\n"
        f"Severity: {incident.severity.value}\n"
        f"Matched cue: {incident.matched_cue}\n"
        f"Utterance: {incident.utterance[:500]}"
    )
    md: dict[str, Any] = {
        "crisis_severity": incident.severity.value,
        "matched_cue": incident.matched_cue,
    }
    if extra_metadata:
        md.update(dict(extra_metadata))
    return NotificationEvent(
        severity=sev,
        title=title,
        body=body,
        user_id=incident.user_id,
        timestamp_seconds=incident.timestamp_seconds,
        metadata=md,
    )


def build_crisis_handoff_sink(
    *,
    audit_log_path: Path,
    notifier: Optional[NotificationSink] = None,
    on_notification_result: Optional[Callable[[NotificationResult], None]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Callable[[CrisisIncident], None]:
    """Build a `handoff_sink` callable compatible with `CrisisEscalation`.

    The returned callable:
      1. Writes an audit record to `audit_log_path` (JSONL).
      2. Converts the incident to a `NotificationEvent`.
      3. Sends via the configured `notifier` (if any) — or `ConsoleNotificationSink`
         by default so dev + CI get visibility.
      4. Invokes `on_notification_result` with the aggregate result (useful
         for the CLI to log the pager outcome to the session log).

    The audit record is written BEFORE notification is attempted — crash
    during notification still leaves a durable trail of the triggering
    incident.
    """
    audit_sink = JsonlFileNotificationSink(path=audit_log_path, name="crisis_audit")
    live_sink: NotificationSink = notifier or ConsoleNotificationSink()

    def _sink(incident: CrisisIncident) -> None:
        event = crisis_incident_to_notification_event(
            incident, extra_metadata=extra_metadata
        )
        audit_sink.send(event)
        try:
            result = live_sink.send(event)
        except NotificationNotConfigured as exc:
            result = NotificationResult(
                sink_name=live_sink.name,
                ok=False,
                error=f"NotificationNotConfigured: {exc}",
            )
        except Exception as exc:
            result = NotificationResult(
                sink_name=live_sink.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        if on_notification_result is not None:
            on_notification_result(result)

    return _sink
