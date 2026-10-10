"""CLI for the Phase 4b S1/S2 A/B — ``assign`` + ``report``.

Two subcommands::

    python -m psychology_base.applications.pilot.ab_cli assign \\
        --app learning_tutor --seed phase4b-2026Q4 \\
        --user t01 --user t02 ... --out plan.json

    python -m psychology_base.applications.pilot.ab_cli report \\
        --app emotional_companion --pilot-root ./pilot-data \\
        --out report.json --markdown report.md

``assign`` writes a frozen :class:`AssignmentPlan`; the PI commits that
file to the pilot repo BEFORE enrolment opens. ``report`` scans the
pilot directory and emits the S1-vs-S2 comparison tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from psychology_base.applications.learning.tutor_cli import DEFAULT_PILOT_ROOT
from psychology_base.applications.pilot.ab import (
    build_comparison_report,
    load_assignment_plan,
    plan_assignment,
    render_comparison_markdown,
    write_assignment_plan,
)
from psychology_base.applications.pilot.layout import PilotApp


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_assign(args: argparse.Namespace) -> int:
    if not args.user:
        sys.stderr.write(
            "assign requires at least one --user (repeat for multiple).\n"
        )
        return 2
    try:
        app = PilotApp(args.app)
    except ValueError:
        sys.stderr.write(
            f"--app must be one of "
            f"{[a.value for a in PilotApp]}; got {args.app!r}.\n"
        )
        return 2
    plan = plan_assignment(args.user, seed=args.seed, app=app)
    if args.out is not None:
        path = write_assignment_plan(args.out, plan)
        sys.stderr.write(f"(wrote plan: {path})\n")
    print(json.dumps(plan.to_json(), ensure_ascii=False, indent=2))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    try:
        plan = load_assignment_plan(args.plan)
    except FileNotFoundError:
        sys.stderr.write(f"No plan at {args.plan}.\n")
        return 2
    if args.user is not None:
        pid = plan.policy_for(args.user)
        if pid is None:
            sys.stderr.write(
                f"User {args.user!r} is not in plan "
                f"(plan covers {len(plan.user_ids)} users).\n"
            )
            return 3
        print(pid)
        return 0
    print(json.dumps(plan.to_json(), ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    try:
        app = PilotApp(args.app)
    except ValueError:
        sys.stderr.write(
            f"--app must be one of "
            f"{[a.value for a in PilotApp]}; got {args.app!r}.\n"
        )
        return 2
    report = build_comparison_report(pilot_root=args.pilot_root, app=app)
    payload = report.to_json()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        sys.stderr.write(f"(wrote JSON report: {args.out})\n")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            render_comparison_markdown(report),
            encoding="utf-8",
        )
        sys.stderr.write(f"(wrote Markdown report: {args.markdown})\n")
    if args.out is None and args.markdown is None:
        # No destination → emit Markdown to stdout for a quick glance.
        print(render_comparison_markdown(report))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# argparse scaffolding
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pilot_ab_cli",
        description=(
            "Pre-registered S1/S2 policy assignment and comparison report "
            "for the Phase 4b pilot."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    assign = sub.add_parser(
        "assign",
        help=(
            "Produce a pre-registered S1/S2 assignment for a committed "
            "user_id list + seed."
        ),
    )
    assign.add_argument(
        "--app", required=True, choices=[a.value for a in PilotApp]
    )
    assign.add_argument(
        "--seed",
        required=True,
        help=(
            "Human-chosen committed seed string, e.g. "
            "`phase4b-2026Q4-tutor`. Commit this value to the pilot "
            "repository BEFORE running assign."
        ),
    )
    assign.add_argument(
        "--user",
        action="append",
        default=[],
        help="User id to include. Repeat for multiple users.",
    )
    assign.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the plan JSON. If omitted, prints to stdout only.",
    )
    assign.set_defaults(func=cmd_assign)

    show = sub.add_parser(
        "show", help="Print an existing plan or look up one user's policy."
    )
    show.add_argument("--plan", type=Path, required=True)
    show.add_argument(
        "--user",
        default=None,
        help="If given, print only this user's policy id (S1/S2).",
    )
    show.set_defaults(func=cmd_show)

    report = sub.add_parser(
        "report",
        help=(
            "Scan an app's pilot directory + produce the S1-vs-S2 comparison "
            "report (JSON + optional Markdown)."
        ),
    )
    report.add_argument(
        "--app", required=True, choices=[a.value for a in PilotApp]
    )
    report.add_argument(
        "--pilot-root",
        type=Path,
        default=DEFAULT_PILOT_ROOT,
        help="Pilot root directory (default: ./pilot-data/).",
    )
    report.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the JSON report.",
    )
    report.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Where to write the Markdown rendering.",
    )
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "cmd_assign",
    "cmd_report",
    "cmd_show",
    "main",
]
