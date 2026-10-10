"""Phase 4b pilot CLI for the long-term emotional companion.

Mirror of :mod:`psychology_base.applications.learning.tutor_cli` with the
companion-specific safety surface wired on:

- Minor-exclusion gate is enforced strictly (`《办法》 Art. 7` applies to
  pseudo-kin / pseudo-romantic surfaces). Pilot additionally excludes
  under-18 entirely at enrol time per §9.8.
- Crisis-escalation sink is wired to a per-user ``crisis-log.jsonl``.
  Severity ≥ EXPLICIT_IDEATION produces a handoff script injection and
  halts the LLM path for that turn — the user is routed to the Tier-2
  clinician and the published crisis line.
- Chain-reference audit hook is left dormant at the CLI layer; the
  pilot auditor pulls it off the agent's internal log at wind-down.

Subcommands + usage mirror the tutor CLI::

    python -m psychology_base.applications.emotional.companion_cli enroll \\
        --user c01 --policy S2 --age-bracket 26-45 --region CN

    python -m psychology_base.applications.emotional.companion_cli session \\
        --user c01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from psychology_base.applications.emotional.companion_agent import (
    CompanionAgent,
    CompanionResponse,
    CompanionSystemPrompt,
    HANDOFF_SCRIPT,
    LLMResponder,
)
from psychology_base.applications.learning.tutor_cli import (
    BRACKET_TO_YEARS,
    DEFAULT_PILOT_ROOT,
    VALID_BRACKETS,
    _build_llm_responder,
)
from psychology_base.applications.pilot.layout import (
    POLICY_IDS,
    PilotApp,
    PilotLayout,
    UserManifest,
    build_aging_policy,
    load_manifest,
    next_session_index,
    pam_latest_path,
    pam_session_path,
    read_manifest_if_present,
    session_log_path,
    utc_now_iso,
    write_manifest,
)
from psychology_base.applications.pilot.session_runner import (
    SessionLogWriter,
    run_session,
)
from psychology_base.applications.safety.ai_identity_reminder import (
    AIIdentityReminder,
)
from psychology_base.applications.safety.compliance_floor import (
    get_builtin_policy,
)
from psychology_base.applications.safety.crisis_escalation import (
    CrisisEscalation,
    CrisisIncident,
)
from psychology_base.applications.safety.minor_exclusion import (
    AgeAttestation,
    ApplicationSurface,
    MinorExclusionGate,
)
from psychology_base.applications.safety.notifications import (
    AlibabaCloudSMSNotificationSink,
    ConsoleNotificationSink,
    NotificationResult,
    NotificationSink,
    SMTPEmailNotificationSink,
    TwilioSMSNotificationSink,
    build_crisis_handoff_sink,
)
from psychology_base.companion.pam_cli import (
    export_system_to_path,
    restore_system_from_path,
)
from psychology_base.companion.system import CompanionSystem


CRISIS_LOG_FILENAME = "crisis-log.jsonl"
PAGER_AUDIT_FILENAME = "pager-audit.jsonl"
NOTIFY_PROVIDERS = ("none", "console", "twilio", "aliyun", "smtp")


# ---------------------------------------------------------------------------
# Session resume + save helpers (mirror of tutor_cli; local copies to
# keep the two apps independently evolvable)
# ---------------------------------------------------------------------------


def _resume_or_init_system(
    layout: PilotLayout,
    user_id: str,
    manifest: UserManifest,
) -> CompanionSystem:
    aging_policy = build_aging_policy(manifest.policy_id)
    latest = pam_latest_path(layout, user_id)
    if latest.exists():
        system, offenders = restore_system_from_path(
            latest,
            system=CompanionSystem(aging_policy=aging_policy),
        )
        if offenders:
            sys.stderr.write(
                f"[warn] {len(offenders)} fragments failed content-"
                f"addressing on resume (first id: {offenders[0]})\n"
            )
        return system
    return CompanionSystem(aging_policy=aging_policy)


def _save_pam(
    layout: PilotLayout,
    user_id: str,
    session_idx: int,
    system: CompanionSystem,
) -> tuple[int, Path, Path]:
    layout.pam_dir(user_id).mkdir(parents=True, exist_ok=True)
    latest = pam_latest_path(layout, user_id)
    per_session = pam_session_path(layout, user_id, session_idx)
    count = export_system_to_path(system, latest)
    per_session.write_bytes(latest.read_bytes())
    return count, latest, per_session


def _make_crisis_sink(
    crisis_log_path: Path,
    *,
    on_incident: Optional[Callable[[CrisisIncident], None]] = None,
) -> Callable[[CrisisIncident], None]:
    """Build a handoff sink that appends a JSONL record per incident.

    The sink is intentionally synchronous: in the pilot, we want the
    disk write to complete BEFORE the LLM path short-circuits so the
    audit trail cannot be lost if the process crashes mid-response.
    """

    def _sink(incident: CrisisIncident) -> None:
        crisis_log_path.parent.mkdir(parents=True, exist_ok=True)
        with crisis_log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "severity": incident.severity.value,
                        "matched_cue": incident.matched_cue,
                        "utterance": incident.utterance,
                        "timestamp_seconds": incident.timestamp_seconds,
                        "user_id": incident.user_id,
                        "ts": utc_now_iso(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if on_incident is not None:
            on_incident(incident)

    return _sink


def _build_notifier(provider: str, to_list: list[str]) -> NotificationSink:
    """Instantiate the requested notification sink.

    Credential validation is deferred to :meth:`NotificationSink.send`; the
    CLI boots cleanly in CI (and in dev shells without secrets) and only
    raises :class:`NotificationNotConfigured` on an actual crisis incident.
    Providers that require recipients (SMS, email) require ``--notify-to``.
    """
    if provider == "console":
        return ConsoleNotificationSink()
    if provider in ("twilio", "aliyun", "smtp") and not to_list:
        raise ValueError(
            f"--notify-provider {provider!r} requires at least one "
            f"--notify-to recipient."
        )
    if provider == "twilio":
        return TwilioSMSNotificationSink(to_numbers=tuple(to_list))
    if provider == "aliyun":
        return AlibabaCloudSMSNotificationSink(to_numbers=tuple(to_list))
    if provider == "smtp":
        return SMTPEmailNotificationSink(to_addresses=tuple(to_list))
    raise ValueError(f"Unknown notify-provider {provider!r}.")


def _make_combined_crisis_sink(
    crisis_log_path: Path,
    pager_audit_path: Path,
    notifier: NotificationSink,
    *,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    on_notification_result: Optional[Callable[[NotificationResult], None]] = None,
) -> Callable[[CrisisIncident], None]:
    """Fan an incident out to the legacy crisis log AND the pager pipeline.

    The legacy ``crisis-log.jsonl`` is preserved so existing wind-down
    tooling and tests keep reading the same schema; the pager-audit log
    captures the richer :class:`NotificationEvent` representation that
    downstream human-handoff tooling (clinician dashboard, PI inbox)
    consumes.
    """
    legacy = _make_crisis_sink(crisis_log_path)
    handoff = build_crisis_handoff_sink(
        audit_log_path=pager_audit_path,
        notifier=notifier,
        on_notification_result=on_notification_result,
        extra_metadata=extra_metadata,
    )

    def _sink(incident: CrisisIncident) -> None:
        legacy(incident)
        handoff(incident)

    return _sink


def build_companion_agent(
    manifest: UserManifest,
    system: CompanionSystem,
    *,
    llm_responder: LLMResponder,
    session_id: str,
    crisis_log_path: Path,
    handoff_sink: Optional[Callable[[CrisisIncident], None]] = None,
) -> CompanionAgent:
    """Compose the companion agent with the full 《办法》 safety surface.

    - AI-identity reminder is driven from the manifest region's
      compliance overlay.
    - Minor-exclusion gate enforces 14/18 (or regional equivalents).
    - Crisis escalation wires the sink to ``crisis_log_path``.
    """
    policy = get_builtin_policy(manifest.region)
    reminder = AIIdentityReminder.from_policy(policy)
    gate = MinorExclusionGate(
        surface=ApplicationSurface.EMOTIONAL_COMPANION, policy=policy
    )
    sink = handoff_sink if handoff_sink is not None else _make_crisis_sink(
        crisis_log_path
    )
    crisis = CrisisEscalation(handoff_sink=sink, policy=policy)
    return CompanionAgent(
        user_id=manifest.user_id,
        session_id=session_id,
        companion_system=system,
        llm_responder=llm_responder,
        ai_identity_reminder=reminder,
        minor_exclusion_gate=gate,
        crisis_escalation=crisis,
        system_prompt=CompanionSystemPrompt.SYSTEM_PROMPT,
        handoff_script=HANDOFF_SCRIPT,
    )


def _attestation_from_manifest(manifest: UserManifest) -> AgeAttestation:
    years = BRACKET_TO_YEARS.get(manifest.age_bracket)
    if years is None:
        raise ValueError(
            f"Manifest age_bracket {manifest.age_bracket!r} is not one of "
            f"{VALID_BRACKETS}."
        )
    return AgeAttestation(
        age_years=years,
        attested_at=time.time(),
        attestation_source=f"manifest@{manifest.enrolled_at}",
    )


def _companion_metadata(resp: CompanionResponse) -> Mapping[str, Any]:
    return {
        "turn_index": resp.turn_index,
        "reminder_fired": resp.reminder_fired,
        "handoff_triggered": resp.handoff_triggered,
        "crisis_severity": (
            resp.crisis_incident.severity.value
            if resp.crisis_incident is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_enroll(args: argparse.Namespace) -> int:
    layout = PilotLayout(
        pilot_root=args.pilot_root, app=PilotApp.EMOTIONAL_COMPANION
    )
    existing = read_manifest_if_present(layout, args.user)
    if existing is not None and not args.force:
        sys.stderr.write(
            f"User {args.user!r} already enrolled "
            f"(policy={existing.policy_id}, enrolled_at={existing.enrolled_at}). "
            f"Pass --force to overwrite.\n"
        )
        return 2
    policy_id = args.policy.upper()
    if policy_id not in POLICY_IDS:
        sys.stderr.write(
            f"--policy must be one of {POLICY_IDS}; got {args.policy!r}.\n"
        )
        return 2
    if args.age_bracket not in VALID_BRACKETS:
        sys.stderr.write(
            f"--age-bracket must be one of {VALID_BRACKETS}; "
            f"got {args.age_bracket!r}.\n"
        )
        return 2
    # Pilot §9.8: no under-18 users for the emotional-companion surface.
    # The VALID_BRACKETS list already starts at 18-25, but we also run
    # the gate at enrol time as a defence in depth: a future bracket
    # (e.g. 14-17) added for a non-pilot deployment should not accidentally
    # become selectable here.
    policy = get_builtin_policy(args.region.upper())
    gate = MinorExclusionGate(
        surface=ApplicationSurface.EMOTIONAL_COMPANION, policy=policy
    )
    years = BRACKET_TO_YEARS[args.age_bracket]
    attestation = AgeAttestation(
        age_years=years,
        attested_at=time.time(),
        attestation_source="enrol_cli",
    )
    decision = gate.decide(attestation)
    if decision.value != "allow":
        sys.stderr.write(
            f"Pilot enrolment rejected for age_bracket={args.age_bracket} in "
            f"region={args.region.upper()}: gate decision={decision.value}.\n"
        )
        return 2
    manifest = UserManifest(
        user_id=args.user,
        app=PilotApp.EMOTIONAL_COMPANION,
        policy_id=policy_id,
        enrolled_at=utc_now_iso(),
        age_bracket=args.age_bracket,
        region=args.region.upper(),
        language=args.language,
        guardian_consent_artefact_id=None,
        pilot_root=str(args.pilot_root.resolve()),
    )
    path = write_manifest(layout, manifest)
    print(json.dumps({"wrote": str(path), "manifest": manifest.to_json()}, indent=2))
    return 0


def cmd_session(args: argparse.Namespace) -> int:
    layout = PilotLayout(
        pilot_root=args.pilot_root, app=PilotApp.EMOTIONAL_COMPANION
    )
    try:
        manifest = load_manifest(layout, args.user)
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    system = _resume_or_init_system(layout, args.user, manifest)
    session_idx = next_session_index(layout, args.user)
    session_id = f"{manifest.user_id}-s{session_idx:04d}"
    try:
        responder = _build_llm_responder(args.mode)
    except (RuntimeError, ValueError, OSError) as exc:
        sys.stderr.write(f"Could not build LLM responder: {exc}\n")
        return 3
    crisis_log_path = layout.user_dir(args.user) / CRISIS_LOG_FILENAME
    handoff_sink: Optional[Callable[[CrisisIncident], None]] = None
    notify_provider = getattr(args, "notify_provider", "none")
    if notify_provider != "none":
        try:
            notifier = _build_notifier(notify_provider, list(args.notify_to or []))
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        pager_audit_path = Path(
            args.notify_audit_log
            if getattr(args, "notify_audit_log", None)
            else layout.user_dir(args.user) / PAGER_AUDIT_FILENAME
        )
        handoff_sink = _make_combined_crisis_sink(
            crisis_log_path,
            pager_audit_path,
            notifier,
            extra_metadata={
                "app": "companion",
                "pilot_user": manifest.user_id,
                "pilot_region": manifest.region,
                "pilot_policy": manifest.policy_id,
            },
        )
    agent = build_companion_agent(
        manifest,
        system,
        llm_responder=responder,
        session_id=session_id,
        crisis_log_path=crisis_log_path,
        handoff_sink=handoff_sink,
    )
    attestation = _attestation_from_manifest(manifest)

    def _agent_fn(utterance: str) -> tuple[str, Mapping[str, Any]]:
        resp = agent.respond(utterance, attestation=attestation)
        return resp.response_text, _companion_metadata(resp)

    def _on_boundary() -> None:
        system.mark_session_boundary()

    def _on_export(out_path: Path) -> int:
        return export_system_to_path(system, out_path)

    def _on_end() -> None:
        count, latest, per_session = _save_pam(
            layout, args.user, session_idx, system
        )
        agent.mark_session_boundary()
        sys.stdout.write(
            f"\n(session saved: {count} fragments → {latest.name}, "
            f"{per_session.name})\n"
        )
        sys.stdout.flush()

    log_path = session_log_path(layout, args.user, session_idx)
    welcome = (
        f"Companion session {session_idx} for user {manifest.user_id} "
        f"(policy={manifest.policy_id}, region={manifest.region}).\n"
        f"You are talking to an AI, not a human. If you are in immediate "
        f"danger please contact the published crisis line directly.\n"
        f"Type /help for commands, /quit to end."
    )
    with SessionLogWriter(path=log_path) as log_writer:
        try:
            turn_count = run_session(
                user_id=manifest.user_id,
                session_idx=session_idx,
                agent_fn=_agent_fn,
                log_writer=log_writer,
                on_export=_on_export,
                on_boundary=_on_boundary,
                on_end=_on_end,
                welcome_text=welcome,
            )
        except KeyboardInterrupt:
            sys.stdout.write("\n(interrupted; session memory NOT saved)\n")
            sys.stdout.flush()
            return 130
    sys.stdout.write(f"(session complete: {turn_count} turns)\n")
    sys.stdout.flush()
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    layout = PilotLayout(
        pilot_root=args.pilot_root, app=PilotApp.EMOTIONAL_COMPANION
    )
    manifest = read_manifest_if_present(layout, args.user)
    if manifest is None:
        sys.stderr.write(f"No manifest for user {args.user!r}.\n")
        return 2
    sessions_dir = layout.sessions_dir(args.user)
    pam_dir = layout.pam_dir(args.user)
    crisis_log = layout.user_dir(args.user) / CRISIS_LOG_FILENAME
    report = {
        "manifest": manifest.to_json(),
        "session_count": sum(
            1
            for f in sessions_dir.iterdir()
            if sessions_dir.exists() and f.suffix == ".jsonl"
        )
        if sessions_dir.exists()
        else 0,
        "pam_latest_present": pam_latest_path(layout, args.user).exists(),
        "pam_archive_count": sum(
            1 for f in pam_dir.iterdir() if f.name.startswith("session-")
        )
        if pam_dir.exists()
        else 0,
        "crisis_log_present": crisis_log.exists(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# argparse scaffolding
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companion_cli",
        description="Phase 4b long-term emotional-companion pilot CLI",
    )
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=DEFAULT_PILOT_ROOT,
        help="Pilot root directory (default: ./pilot-data/).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    enroll = sub.add_parser("enroll", help="Create a user manifest.")
    enroll.add_argument("--user", required=True)
    enroll.add_argument("--policy", required=True)
    enroll.add_argument(
        "--age-bracket",
        required=True,
        choices=VALID_BRACKETS,
    )
    enroll.add_argument("--region", default="CN")
    enroll.add_argument("--language", default="en")
    enroll.add_argument("--force", action="store_true")
    enroll.set_defaults(func=cmd_enroll)

    session = sub.add_parser("session", help="Run one interactive session.")
    session.add_argument("--user", required=True)
    session.add_argument(
        "--mode",
        choices=("stub", "local", "cloud"),
        default="cloud",
    )
    session.add_argument(
        "--notify-provider",
        choices=NOTIFY_PROVIDERS,
        default="none",
        help=(
            "Notification sink for Tier-2 handoff on crisis incidents. "
            "'none' preserves legacy crisis-log.jsonl-only behavior; other "
            "providers additionally fire a NotificationEvent and append a "
            "pager-audit.jsonl record."
        ),
    )
    session.add_argument(
        "--notify-to",
        action="append",
        default=[],
        help=(
            "Recipient (phone number for twilio/aliyun, email for smtp). "
            "Repeat for multiple recipients. Ignored for 'console' provider."
        ),
    )
    session.add_argument(
        "--notify-audit-log",
        type=Path,
        default=None,
        help=(
            "Override path for the pager-audit JSONL log "
            f"(default: <user_dir>/{PAGER_AUDIT_FILENAME})."
        ),
    )
    session.set_defaults(func=cmd_session)

    inspect = sub.add_parser(
        "inspect", help="Print a user's manifest + archive summary."
    )
    inspect.add_argument("--user", required=True)
    inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CRISIS_LOG_FILENAME",
    "NOTIFY_PROVIDERS",
    "PAGER_AUDIT_FILENAME",
    "build_companion_agent",
    "build_parser",
    "cmd_enroll",
    "cmd_inspect",
    "cmd_session",
    "main",
]
