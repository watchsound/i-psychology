"""Phase 4b pilot CLI for the resilient-learning tutor.

Three subcommands:

- ``enroll``   — write a per-user manifest (one-shot; refuses to overwrite
  an existing manifest unless ``--force`` is passed).
- ``session``  — run one interactive session for an enrolled user.
  Resumes memory from the user's PAM archive if present; appends a
  session log; auto-exports PAM at session end.
- ``inspect``  — print a user's manifest + PAM archive summary (no
  side effects, no memory ingestion).

Usage::

    # Day-one enrolment:
    python -m psychology_base.applications.learning.tutor_cli enroll \\
        --user u01 --policy S1 --age-bracket 26-45 --region CN

    # Each pilot session (resume + run + save):
    python -m psychology_base.applications.learning.tutor_cli session \\
        --user u01

The tutor pilot has no minor-exclusion block (`《办法》 Art. 7` applies to
emotional-companion only) but attestation is still recorded for audit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from psychology_base.applications.learning.tutor_agent import (
    LLMResponder,
    TutorAgent,
    TutorResponse,
    TutorSystemPrompt,
)
from psychology_base.applications.measurement.scaffolding_telemetry import (
    ScaffoldingTelemetry,
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
from psychology_base.applications.safety.minor_exclusion import (
    AgeAttestation,
    ApplicationSurface,
    MinorExclusionGate,
)
from psychology_base.companion.pam_cli import (
    export_system_to_path,
    restore_system_from_path,
)
from psychology_base.companion.system import CompanionSystem


DEFAULT_PILOT_ROOT = Path("pilot-data")

# Age-bracket → representative mid-bracket age (years). The pilot only
# uses this when constructing an `AgeAttestation`; the attestation
# itself is bucket-aware (`AgeAttestation.age_years`), and the bucket
# lives on the manifest.
BRACKET_TO_YEARS: dict[str, int] = {
    "18-25": 22,
    "26-45": 35,
    "46-65": 55,
    "65+": 70,
}
VALID_BRACKETS = tuple(BRACKET_TO_YEARS.keys())


# ---------------------------------------------------------------------------
# Session resume + save helpers
# ---------------------------------------------------------------------------


def _resume_or_init_system(
    layout: PilotLayout,
    user_id: str,
    manifest: UserManifest,
) -> CompanionSystem:
    """Load the user's PAM archive if present; otherwise init a fresh system.

    Uses the manifest's policy_id to select the aging policy; the
    reconstructive-retrieval knobs stay at defaults.
    """
    aging_policy = build_aging_policy(manifest.policy_id)
    latest = pam_latest_path(layout, user_id)
    if latest.exists():
        system, offenders = restore_system_from_path(
            latest,
            system=CompanionSystem(aging_policy=aging_policy),
        )
        if offenders:
            # Not fatal; the pilot auditor flags these downstream. Print
            # a warning so the pilot ops operator sees it immediately.
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
    """Write both ``latest.pam-v1.json`` and ``session-NNNN.pam-v1.json``.

    Returns ``(fragment_count, latest_path, session_path)``.
    """
    layout.pam_dir(user_id).mkdir(parents=True, exist_ok=True)
    latest = pam_latest_path(layout, user_id)
    per_session = pam_session_path(layout, user_id, session_idx)
    count = export_system_to_path(system, latest)
    # Re-use the latest payload for the per-session archive to avoid
    # double-serializing a large fragment set.
    per_session.write_bytes(latest.read_bytes())
    return count, latest, per_session


# ---------------------------------------------------------------------------
# Agent factory + stub responder
# ---------------------------------------------------------------------------


def _stub_responder(
    system_prompt: str, context_block: str, user_utterance: str
) -> str:
    """Deterministic placeholder for ``--mode stub``.

    The pilot never ships with this responder — the LLM shakeout
    (Phase 4a §6e-prime) covered the stub-shape path. It exists so
    operators can smoke-test the CLI loop without an LLM backend
    available.
    """
    hint = user_utterance.strip()
    return (
        "(stub responder) I hear you. Can you walk me through the step "
        f"you tried before this? You wrote: {hint!r}"
    )


def _build_llm_responder(mode: str) -> LLMResponder:
    """Resolve ``--mode`` to a concrete :class:`LLMResponder`.

    ``stub`` returns :func:`_stub_responder`. ``local`` / ``cloud`` go
    through :func:`make_llm_responder` with the appropriate tier.
    Importing the LLM router is deferred to this function so the stub
    path has zero LLM dependencies at import time (tests + CI smoke
    runs stay cheap).
    """
    if mode == "stub":
        return _stub_responder

    # Lazy imports — stub mode does not need the LLM router at all.
    from psychology_base.applications.llm_responders import make_llm_responder
    from psychology_base.llm.config import load_config
    from psychology_base.llm.router import LLMRouter
    from psychology_base.llm.types import TaskTier
    import dataclasses as _dc

    cfg = load_config()
    if mode == "local":
        cfg = _dc.replace(
            cfg,
            local_ollama=_dc.replace(
                cfg.local_ollama, enabled=True, force_cloud_chat=False
            ),
        )
    elif mode == "cloud":
        cfg = _dc.replace(
            cfg,
            local_ollama=_dc.replace(cfg.local_ollama, force_cloud_chat=True),
        )
    else:
        raise ValueError(f"Unknown --mode {mode!r}; expected stub/local/cloud.")
    router = LLMRouter(cfg)
    return make_llm_responder(router, tier=TaskTier.ADVANCED)


def build_tutor_agent(
    manifest: UserManifest,
    system: CompanionSystem,
    *,
    llm_responder: LLMResponder,
    session_id: str,
    telemetry: Optional[ScaffoldingTelemetry] = None,
) -> TutorAgent:
    """Compose the tutor agent for one session.

    - Pulls the regional-compliance policy from the manifest's region
      and wires both the AI-identity reminder and the minor-exclusion
      gate against it.
    - Records attestation via the gate even though the tutor surface
      always `ALLOW`s (《办法》 Art. 7 is companion-only).
    """
    policy = get_builtin_policy(manifest.region)
    reminder = AIIdentityReminder.from_policy(policy)
    gate = MinorExclusionGate(
        surface=ApplicationSurface.LEARNING_TUTOR, policy=policy
    )
    tele = telemetry if telemetry is not None else ScaffoldingTelemetry(
        user_id=manifest.user_id
    )
    return TutorAgent(
        user_id=manifest.user_id,
        session_id=session_id,
        companion_system=system,
        llm_responder=llm_responder,
        telemetry=tele,
        ai_identity_reminder=reminder,
        minor_exclusion_gate=gate,
        system_prompt=TutorSystemPrompt.SYSTEM_PROMPT,
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


def _tutor_metadata(resp: TutorResponse) -> Mapping[str, Any]:
    return {
        "tutor_move": resp.tutor_move.value,
        "turn_index": resp.turn_index,
        "reminder_fired": resp.reminder_fired,
    }


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_enroll(args: argparse.Namespace) -> int:
    layout = PilotLayout(pilot_root=args.pilot_root, app=PilotApp.LEARNING_TUTOR)
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
    manifest = UserManifest(
        user_id=args.user,
        app=PilotApp.LEARNING_TUTOR,
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
    layout = PilotLayout(pilot_root=args.pilot_root, app=PilotApp.LEARNING_TUTOR)
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
    telemetry = ScaffoldingTelemetry(user_id=manifest.user_id)
    agent = build_tutor_agent(
        manifest,
        system,
        llm_responder=responder,
        session_id=session_id,
        telemetry=telemetry,
    )
    attestation = _attestation_from_manifest(manifest)

    def _agent_fn(utterance: str) -> tuple[str, Mapping[str, Any]]:
        resp = agent.respond(utterance, attestation=attestation)
        return resp.response_text, _tutor_metadata(resp)

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
        f"Tutor session {session_idx} for user {manifest.user_id} "
        f"(policy={manifest.policy_id}, region={manifest.region}).\n"
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
    layout = PilotLayout(pilot_root=args.pilot_root, app=PilotApp.LEARNING_TUTOR)
    manifest = read_manifest_if_present(layout, args.user)
    if manifest is None:
        sys.stderr.write(f"No manifest for user {args.user!r}.\n")
        return 2
    sessions_dir = layout.sessions_dir(args.user)
    pam_dir = layout.pam_dir(args.user)
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
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# argparse scaffolding
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tutor_cli",
        description="Phase 4b resilient-learning tutor pilot CLI",
    )
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=DEFAULT_PILOT_ROOT,
        help="Pilot root directory (default: ./pilot-data/).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    enroll = sub.add_parser("enroll", help="Create a user manifest.")
    enroll.add_argument("--user", required=True, help="Pseudonymous user handle.")
    enroll.add_argument(
        "--policy",
        required=True,
        help=f"Aging policy id (one of {POLICY_IDS}).",
    )
    enroll.add_argument(
        "--age-bracket",
        required=True,
        choices=VALID_BRACKETS,
        help="Age bucket (pilot excludes <18).",
    )
    enroll.add_argument(
        "--region",
        default="CN",
        help="ISO-3166 region code for compliance overlay (default CN).",
    )
    enroll.add_argument(
        "--language",
        default="en",
        help="BCP-47 language tag for user-facing text (default en).",
    )
    enroll.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing manifest.",
    )
    enroll.set_defaults(func=cmd_enroll)

    session = sub.add_parser("session", help="Run one interactive session.")
    session.add_argument("--user", required=True)
    session.add_argument(
        "--mode",
        choices=("stub", "local", "cloud"),
        default="cloud",
        help="LLM responder backend (default: cloud; pilot uses advanced tier).",
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
    "BRACKET_TO_YEARS",
    "DEFAULT_PILOT_ROOT",
    "VALID_BRACKETS",
    "build_parser",
    "build_tutor_agent",
    "cmd_enroll",
    "cmd_inspect",
    "cmd_session",
    "main",
]
