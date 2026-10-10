"""Phase 4b pilot A/B — pre-registered S1/S2 assignment + comparison report.

Per `docs/phases/phase-4b-signoff.md` §9.5, each pilot app enrols 5 users
on S1 and 5 users on S2. Sign-off requires the assignment be
**pre-registered**: fixed BEFORE enrolment opens, deterministic given a
committed seed, and auditable later from the artefact alone. Picking
policies after seeing enrol order would let cohort differences bias the
comparison.

Two public functions:

- :func:`plan_assignment` takes a list of user ids and a seed, returns a
  frozen, exactly-balanced assignment. The caller writes this to disk
  BEFORE calling ``enroll``; the enrol wizard reads the plan rather than
  letting the PI type ``--policy`` by hand.

- :func:`build_comparison_report` scans one app's pilot directory, pulls
  session + crisis + PAM metrics per user, aggregates by policy, and
  returns a side-by-side S1-vs-S2 summary (plus a Markdown rendering).

Metrics covered (per the sign-off §10 approved axes):

- Enrolment counts per policy.
- Session counts, turn counts, and mean turns/session.
- Agent-side signals from the session log: crisis-handoff rate,
  AI-identity reminder fire rate.
- PAM archive persistence (sessions saved, mean fragments per archive).

Metrics NOT covered here are the ones that require a judge or auditor
label (WAI-SR, Socratic compliance κ, chain-reference quality, empathy
grade). Those arrive from the Argilla pipeline (§11 item 7) and should
be joined into this report by the pilot wind-down tooling, not computed
inline — this module stays dependency-free and reads only the local
pilot directory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from psychology_base.applications.pilot.layout import (
    POLICY_IDS,
    PilotApp,
    PilotLayout,
    load_manifest,
    pam_latest_path,
    read_manifest_if_present,
)


# ---------------------------------------------------------------------------
# Pre-registered assignment
# ---------------------------------------------------------------------------


ASSIGNMENT_PLAN_SCHEMA_VERSION = 1
"""Bump when the on-disk plan schema changes incompatibly. The pilot
audit requires that a plan written months ago can be re-loaded today
without ambiguity about the policy ids it names."""


@dataclass(frozen=True)
class AssignmentPlan:
    """Pre-registered A/B assignment artefact.

    Written to disk BEFORE enrolment; read by the enrol wizard so the
    policy for a given user_id is fixed by the plan, not by whoever
    happens to run the CLI.
    """

    schema_version: int
    app: PilotApp
    seed: str
    """Human-chosen committed seed. The hash function + seed together
    determine the assignment given the user_id list."""

    user_ids: tuple[str, ...]
    assignment: Mapping[str, str]
    """``user_id → policy_id`` ("S1" or "S2")."""

    counts: Mapping[str, int]
    """Policy counts for a quick glance at balance."""

    def policy_for(self, user_id: str) -> Optional[str]:
        return self.assignment.get(user_id)

    def to_json(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "app": self.app.value,
            "seed": self.seed,
            "user_ids": list(self.user_ids),
            "assignment": dict(self.assignment),
            "counts": dict(self.counts),
        }

    @classmethod
    def from_json(cls, obj: Mapping) -> "AssignmentPlan":
        schema = int(obj.get("schema_version", 1))
        if schema != ASSIGNMENT_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"AssignmentPlan schema_version {schema} not supported "
                f"(expected {ASSIGNMENT_PLAN_SCHEMA_VERSION})."
            )
        return cls(
            schema_version=schema,
            app=PilotApp(obj["app"]),
            seed=str(obj["seed"]),
            user_ids=tuple(obj["user_ids"]),
            assignment=dict(obj["assignment"]),
            counts=dict(obj["counts"]),
        )


def _hash_rank(user_id: str, seed: str) -> int:
    """Stable integer rank for a user id under a seed.

    Uses blake2b-128 for cross-platform determinism; built-in ``hash()``
    is PYTHONHASHSEED-salted, so cannot be used for a reproducible
    pre-registered plan.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(seed.encode("utf-8"))
    h.update(b"\x00")
    h.update(user_id.encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="big")


def plan_assignment(
    user_ids: Sequence[str],
    *,
    seed: str,
    app: PilotApp,
) -> AssignmentPlan:
    """Deterministically assign users to S1/S2 with exact balance.

    Algorithm:

    1. De-duplicate and sort ``user_ids`` to make the plan independent
       of input order.
    2. Rank each user id by ``blake2b(seed || user_id)``.
    3. Alternate policies down the ranked list — the lowest-hash user
       goes to S1, next to S2, next to S1, …

    This yields exact balance (``n // 2`` on each policy; the remainder
    for an odd count lands on S1 by construction). Alternation is
    stronger than modulo-2-hash: the latter can land 7/3 by bad luck,
    which the sign-off's N=5-per-policy requirement explicitly forbids.
    """
    if not user_ids:
        raise ValueError("plan_assignment requires at least one user_id.")
    deduped = sorted(set(user_ids))
    ranked = sorted(deduped, key=lambda uid: _hash_rank(uid, seed))
    assignment: dict[str, str] = {}
    for idx, uid in enumerate(ranked):
        assignment[uid] = POLICY_IDS[idx % 2]
    counts: dict[str, int] = {pid: 0 for pid in POLICY_IDS}
    for pid in assignment.values():
        counts[pid] += 1
    return AssignmentPlan(
        schema_version=ASSIGNMENT_PLAN_SCHEMA_VERSION,
        app=app,
        seed=seed,
        user_ids=tuple(deduped),
        assignment=assignment,
        counts=counts,
    )


def write_assignment_plan(path: Path, plan: AssignmentPlan) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_assignment_plan(path: Path) -> AssignmentPlan:
    return AssignmentPlan.from_json(
        json.loads(path.read_text(encoding="utf-8"))
    )


# ---------------------------------------------------------------------------
# Pilot comparison report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserMetrics:
    """Per-user metrics scraped from the pilot directory."""

    user_id: str
    policy_id: str
    session_count: int
    turn_count: int
    agent_turn_count: int
    handoff_events: int
    reminder_events: int
    pam_archive_count: int
    pam_latest_bytes: int

    @property
    def handoff_rate(self) -> float:
        if self.agent_turn_count == 0:
            return 0.0
        return self.handoff_events / self.agent_turn_count


@dataclass(frozen=True)
class PolicyAggregate:
    """Aggregated metrics for one policy arm (S1 or S2)."""

    policy_id: str
    user_count: int
    session_count: int
    turn_count: int
    agent_turn_count: int
    handoff_events: int
    reminder_events: int
    pam_archive_count: int
    pam_latest_bytes_total: int

    @property
    def handoff_rate(self) -> float:
        if self.agent_turn_count == 0:
            return 0.0
        return self.handoff_events / self.agent_turn_count

    @property
    def mean_sessions_per_user(self) -> float:
        if self.user_count == 0:
            return 0.0
        return self.session_count / self.user_count

    @property
    def mean_turns_per_session(self) -> float:
        if self.session_count == 0:
            return 0.0
        return self.turn_count / self.session_count


@dataclass(frozen=True)
class ComparisonReport:
    """Side-by-side S1-vs-S2 comparison for one pilot app."""

    app: PilotApp
    pilot_root: Path
    per_policy: Mapping[str, PolicyAggregate]
    per_user: Sequence[UserMetrics]
    unassigned_users: Sequence[str] = field(default_factory=tuple)
    """Users enrolled under a policy id that is not S1 or S2 — flagged
    so the operator spots a typo rather than silently dropping rows."""

    def to_json(self) -> dict:
        return {
            "app": self.app.value,
            "pilot_root": str(self.pilot_root.resolve()),
            "per_policy": {
                pid: {
                    "policy_id": agg.policy_id,
                    "user_count": agg.user_count,
                    "session_count": agg.session_count,
                    "turn_count": agg.turn_count,
                    "agent_turn_count": agg.agent_turn_count,
                    "handoff_events": agg.handoff_events,
                    "reminder_events": agg.reminder_events,
                    "pam_archive_count": agg.pam_archive_count,
                    "pam_latest_bytes_total": agg.pam_latest_bytes_total,
                    "handoff_rate": agg.handoff_rate,
                    "mean_sessions_per_user": agg.mean_sessions_per_user,
                    "mean_turns_per_session": agg.mean_turns_per_session,
                }
                for pid, agg in self.per_policy.items()
            },
            "per_user": [
                {
                    "user_id": u.user_id,
                    "policy_id": u.policy_id,
                    "session_count": u.session_count,
                    "turn_count": u.turn_count,
                    "agent_turn_count": u.agent_turn_count,
                    "handoff_events": u.handoff_events,
                    "reminder_events": u.reminder_events,
                    "pam_archive_count": u.pam_archive_count,
                    "pam_latest_bytes": u.pam_latest_bytes,
                    "handoff_rate": u.handoff_rate,
                }
                for u in self.per_user
            ],
            "unassigned_users": list(self.unassigned_users),
        }


def _enumerate_user_ids(layout: PilotLayout) -> list[str]:
    users_dir = layout.users_dir
    if not users_dir.exists():
        return []
    return sorted(
        entry.name
        for entry in users_dir.iterdir()
        if entry.is_dir() and (entry / "user.json").exists()
    )


def _scrape_session_metrics(session_log: Path) -> tuple[int, int, int, int]:
    """Return (agent_turn_count, user_turn_count, handoff_events, reminder_events).

    Session log is append-only JSONL; this reader is tolerant of mid-write
    truncation (one broken line) because a pilot machine losing power is
    more likely than a corrupt log and we would rather see partial data
    than crash the whole report.
    """
    agent_turns = 0
    user_turns = 0
    handoffs = 0
    reminders = 0
    with session_log.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = rec.get("event")
            if event == "agent_turn":
                agent_turns += 1
                md = rec.get("metadata") or {}
                if md.get("handoff_triggered"):
                    handoffs += 1
                if md.get("reminder_fired"):
                    reminders += 1
            elif event == "user_turn":
                user_turns += 1
    return agent_turns, user_turns, handoffs, reminders


def _collect_user_metrics(
    layout: PilotLayout, user_id: str
) -> UserMetrics:
    manifest = load_manifest(layout, user_id)
    sessions_dir = layout.sessions_dir(user_id)
    pam_dir = layout.pam_dir(user_id)
    session_count = 0
    agent_turns_total = 0
    user_turns_total = 0
    handoffs_total = 0
    reminders_total = 0
    if sessions_dir.exists():
        for entry in sessions_dir.iterdir():
            if not (entry.suffix == ".jsonl" and entry.name.startswith("session-")):
                continue
            session_count += 1
            agent_t, user_t, handoffs, reminders = _scrape_session_metrics(entry)
            agent_turns_total += agent_t
            user_turns_total += user_t
            handoffs_total += handoffs
            reminders_total += reminders
    pam_archive_count = 0
    if pam_dir.exists():
        pam_archive_count = sum(
            1 for f in pam_dir.iterdir() if f.name.startswith("session-")
        )
    latest = pam_latest_path(layout, user_id)
    pam_latest_bytes = latest.stat().st_size if latest.exists() else 0
    return UserMetrics(
        user_id=user_id,
        policy_id=manifest.policy_id,
        session_count=session_count,
        turn_count=user_turns_total,
        agent_turn_count=agent_turns_total,
        handoff_events=handoffs_total,
        reminder_events=reminders_total,
        pam_archive_count=pam_archive_count,
        pam_latest_bytes=pam_latest_bytes,
    )


def _aggregate_policy(policy_id: str, users: Sequence[UserMetrics]) -> PolicyAggregate:
    cohort = [u for u in users if u.policy_id == policy_id]
    return PolicyAggregate(
        policy_id=policy_id,
        user_count=len(cohort),
        session_count=sum(u.session_count for u in cohort),
        turn_count=sum(u.turn_count for u in cohort),
        agent_turn_count=sum(u.agent_turn_count for u in cohort),
        handoff_events=sum(u.handoff_events for u in cohort),
        reminder_events=sum(u.reminder_events for u in cohort),
        pam_archive_count=sum(u.pam_archive_count for u in cohort),
        pam_latest_bytes_total=sum(u.pam_latest_bytes for u in cohort),
    )


def build_comparison_report(
    pilot_root: Path, app: PilotApp
) -> ComparisonReport:
    """Scan one app's pilot directory + return the S1-vs-S2 report."""
    layout = PilotLayout(pilot_root=pilot_root, app=app)
    users = _enumerate_user_ids(layout)
    per_user: list[UserMetrics] = []
    unassigned: list[str] = []
    for uid in users:
        if read_manifest_if_present(layout, uid) is None:
            continue
        metrics = _collect_user_metrics(layout, uid)
        if metrics.policy_id not in POLICY_IDS:
            unassigned.append(uid)
            continue
        per_user.append(metrics)
    per_policy = {pid: _aggregate_policy(pid, per_user) for pid in POLICY_IDS}
    return ComparisonReport(
        app=app,
        pilot_root=pilot_root,
        per_policy=per_policy,
        per_user=tuple(per_user),
        unassigned_users=tuple(unassigned),
    )


def render_comparison_markdown(report: ComparisonReport) -> str:
    """Render the comparison as a human-readable Markdown block.

    The format is intentionally terse — one headline table + a per-user
    table — so pilot wind-down can paste it into the §10 sign-off follow-up
    memo without reformatting.
    """
    lines: list[str] = []
    lines.append(f"# Pilot comparison report — {report.app.value}")
    lines.append("")
    lines.append(f"Pilot root: `{report.pilot_root}`")
    lines.append("")
    lines.append("## Headline — S1 vs S2")
    lines.append("")
    lines.append(
        "| metric | S1 (Ebbinghaus decay) | S2 (capacity-only) |"
    )
    lines.append("|---|---|---|")
    s1 = report.per_policy["S1"]
    s2 = report.per_policy["S2"]
    rows: list[tuple[str, str, str]] = [
        ("users enrolled", str(s1.user_count), str(s2.user_count)),
        ("sessions", str(s1.session_count), str(s2.session_count)),
        (
            "mean sessions / user",
            f"{s1.mean_sessions_per_user:.2f}",
            f"{s2.mean_sessions_per_user:.2f}",
        ),
        ("user turns", str(s1.turn_count), str(s2.turn_count)),
        ("agent turns", str(s1.agent_turn_count), str(s2.agent_turn_count)),
        (
            "mean turns / session",
            f"{s1.mean_turns_per_session:.2f}",
            f"{s2.mean_turns_per_session:.2f}",
        ),
        (
            "crisis handoffs",
            str(s1.handoff_events),
            str(s2.handoff_events),
        ),
        (
            "handoff rate (per agent turn)",
            f"{s1.handoff_rate:.4f}",
            f"{s2.handoff_rate:.4f}",
        ),
        (
            "AI-identity reminders",
            str(s1.reminder_events),
            str(s2.reminder_events),
        ),
        (
            "PAM session archives",
            str(s1.pam_archive_count),
            str(s2.pam_archive_count),
        ),
        (
            "PAM latest total bytes",
            str(s1.pam_latest_bytes_total),
            str(s2.pam_latest_bytes_total),
        ),
    ]
    for name, a, b in rows:
        lines.append(f"| {name} | {a} | {b} |")
    lines.append("")
    if report.unassigned_users:
        lines.append("## Unassigned users (not S1/S2)")
        lines.append("")
        for uid in report.unassigned_users:
            lines.append(f"- `{uid}`")
        lines.append("")
    lines.append("## Per-user breakdown")
    lines.append("")
    lines.append(
        "| user_id | policy | sessions | user_turns | agent_turns | "
        "handoffs | reminders | pam_archives |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for u in sorted(report.per_user, key=lambda m: (m.policy_id, m.user_id)):
        lines.append(
            f"| {u.user_id} | {u.policy_id} | {u.session_count} | "
            f"{u.turn_count} | {u.agent_turn_count} | "
            f"{u.handoff_events} | {u.reminder_events} | "
            f"{u.pam_archive_count} |"
        )
    lines.append("")
    lines.append(
        "> Judge-grade metrics (WAI-SR, Socratic compliance κ, chain-"
        "reference quality, empathy score) are NOT included; they land "
        "here after Argilla auditor labelling (§11 item 7)."
    )
    return "\n".join(lines)


__all__ = [
    "ASSIGNMENT_PLAN_SCHEMA_VERSION",
    "AssignmentPlan",
    "ComparisonReport",
    "PolicyAggregate",
    "UserMetrics",
    "build_comparison_report",
    "load_assignment_plan",
    "plan_assignment",
    "render_comparison_markdown",
    "write_assignment_plan",
]
