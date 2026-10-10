"""Pilot A/B — pre-registered assignment + comparison report."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from psychology_base.applications.pilot import ab, ab_cli
from psychology_base.applications.pilot.ab import (
    AssignmentPlan,
    build_comparison_report,
    load_assignment_plan,
    plan_assignment,
    render_comparison_markdown,
    write_assignment_plan,
)
from psychology_base.applications.pilot.layout import (
    PilotApp,
    PilotLayout,
    UserManifest,
    utc_now_iso,
    write_manifest,
)
from psychology_base.applications.pilot.session_runner import SessionLogWriter


# ---------------------------------------------------------------------------
# Assignment determinism + balance
# ---------------------------------------------------------------------------


def test_plan_assignment_deterministic_under_same_seed() -> None:
    users = [f"t{i:02d}" for i in range(1, 11)]
    plan_a = plan_assignment(users, seed="phase4b-tutor", app=PilotApp.LEARNING_TUTOR)
    plan_b = plan_assignment(users, seed="phase4b-tutor", app=PilotApp.LEARNING_TUTOR)
    assert dict(plan_a.assignment) == dict(plan_b.assignment)


def test_plan_assignment_different_seeds_may_differ() -> None:
    users = [f"t{i:02d}" for i in range(1, 11)]
    plan_a = plan_assignment(users, seed="seed-a", app=PilotApp.LEARNING_TUTOR)
    plan_b = plan_assignment(users, seed="seed-b", app=PilotApp.LEARNING_TUTOR)
    # Different seeds should re-rank the users; with 10 users this will
    # almost certainly produce at least one differently-assigned user.
    assert dict(plan_a.assignment) != dict(plan_b.assignment)


def test_plan_assignment_exact_balance_even_count() -> None:
    plan = plan_assignment(
        [f"c{i:02d}" for i in range(1, 11)],
        seed="s",
        app=PilotApp.EMOTIONAL_COMPANION,
    )
    assert plan.counts == {"S1": 5, "S2": 5}


def test_plan_assignment_odd_count_puts_remainder_on_s1() -> None:
    plan = plan_assignment(
        [f"c{i:02d}" for i in range(1, 8)],
        seed="s",
        app=PilotApp.EMOTIONAL_COMPANION,
    )
    # 7 users → 4 on S1, 3 on S2 (remainder lands on S1 by alternation).
    assert plan.counts == {"S1": 4, "S2": 3}


def test_plan_assignment_dedupes_input() -> None:
    plan = plan_assignment(
        ["t01", "t02", "t01", "t03"],
        seed="s",
        app=PilotApp.LEARNING_TUTOR,
    )
    assert set(plan.user_ids) == {"t01", "t02", "t03"}


def test_plan_assignment_input_order_independent() -> None:
    users_a = ["t03", "t01", "t02"]
    users_b = ["t01", "t02", "t03"]
    plan_a = plan_assignment(users_a, seed="s", app=PilotApp.LEARNING_TUTOR)
    plan_b = plan_assignment(users_b, seed="s", app=PilotApp.LEARNING_TUTOR)
    assert dict(plan_a.assignment) == dict(plan_b.assignment)


def test_plan_assignment_empty_rejects() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plan_assignment([], seed="s", app=PilotApp.LEARNING_TUTOR)


def test_assignment_plan_round_trips_through_disk(tmp_path: Path) -> None:
    plan = plan_assignment(
        [f"t{i:02d}" for i in range(1, 11)],
        seed="s",
        app=PilotApp.LEARNING_TUTOR,
    )
    path = write_assignment_plan(tmp_path / "plan.json", plan)
    loaded = load_assignment_plan(path)
    assert loaded.app is PilotApp.LEARNING_TUTOR
    assert dict(loaded.assignment) == dict(plan.assignment)
    assert loaded.counts == plan.counts
    assert loaded.seed == "s"


def test_assignment_plan_schema_version_check(tmp_path: Path) -> None:
    bad = {
        "schema_version": 999,
        "app": "learning_tutor",
        "seed": "s",
        "user_ids": ["t01"],
        "assignment": {"t01": "S1"},
        "counts": {"S1": 1, "S2": 0},
    }
    (tmp_path / "bad.json").write_text(
        json.dumps(bad), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_assignment_plan(tmp_path / "bad.json")


def test_assignment_plan_policy_for_known_and_unknown() -> None:
    plan = plan_assignment(
        ["t01", "t02", "t03"], seed="s", app=PilotApp.LEARNING_TUTOR
    )
    pid = plan.policy_for("t01")
    assert pid in ("S1", "S2")
    assert plan.policy_for("ghost") is None


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def _write_manifest(
    layout: PilotLayout,
    user_id: str,
    policy_id: str,
    *,
    app: PilotApp,
) -> None:
    manifest = UserManifest(
        user_id=user_id,
        app=app,
        policy_id=policy_id,
        enrolled_at=utc_now_iso(),
        age_bracket="26-45",
        region="CN",
    )
    write_manifest(layout, manifest)


def _write_session_log(
    layout: PilotLayout,
    user_id: str,
    session_idx: int,
    *,
    agent_turns: int,
    handoffs: int = 0,
    reminders: int = 0,
) -> None:
    sessions_dir = layout.sessions_dir(user_id)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"session-{session_idx:04d}.jsonl"
    with SessionLogWriter(path=path) as log:
        log.write(
            {
                "event": "session_start",
                "user_id": user_id,
                "session_idx": session_idx,
                "ts": utc_now_iso(),
            }
        )
        for i in range(agent_turns):
            log.write(
                {
                    "event": "user_turn",
                    "turn_index": i,
                    "text": "hi",
                    "ts": utc_now_iso(),
                }
            )
            md = {
                "handoff_triggered": i < handoffs,
                "reminder_fired": i < reminders,
            }
            log.write(
                {
                    "event": "agent_turn",
                    "turn_index": i,
                    "text": "hello",
                    "metadata": md,
                    "ts": utc_now_iso(),
                }
            )
        log.write(
            {
                "event": "session_end",
                "turn_count": agent_turns,
                "ts": utc_now_iso(),
            }
        )


def test_report_aggregates_per_policy_counts(tmp_path: Path) -> None:
    layout = PilotLayout(
        pilot_root=tmp_path, app=PilotApp.EMOTIONAL_COMPANION
    )
    _write_manifest(layout, "c01", "S1", app=PilotApp.EMOTIONAL_COMPANION)
    _write_manifest(layout, "c02", "S1", app=PilotApp.EMOTIONAL_COMPANION)
    _write_manifest(layout, "c03", "S2", app=PilotApp.EMOTIONAL_COMPANION)
    _write_session_log(layout, "c01", 1, agent_turns=4, handoffs=1)
    _write_session_log(layout, "c01", 2, agent_turns=2)
    _write_session_log(layout, "c02", 1, agent_turns=3, reminders=1)
    _write_session_log(layout, "c03", 1, agent_turns=5, handoffs=2, reminders=1)

    report = build_comparison_report(
        pilot_root=tmp_path, app=PilotApp.EMOTIONAL_COMPANION
    )
    assert report.per_policy["S1"].user_count == 2
    assert report.per_policy["S2"].user_count == 1
    assert report.per_policy["S1"].session_count == 3
    assert report.per_policy["S2"].session_count == 1
    assert report.per_policy["S1"].agent_turn_count == 4 + 2 + 3
    assert report.per_policy["S2"].agent_turn_count == 5
    assert report.per_policy["S1"].handoff_events == 1
    assert report.per_policy["S2"].handoff_events == 2
    assert report.per_policy["S1"].reminder_events == 1
    assert report.per_policy["S2"].reminder_events == 1


def test_report_handles_empty_pilot_dir(tmp_path: Path) -> None:
    report = build_comparison_report(
        pilot_root=tmp_path, app=PilotApp.LEARNING_TUTOR
    )
    assert report.per_policy["S1"].user_count == 0
    assert report.per_policy["S2"].user_count == 0
    assert report.per_user == ()
    assert report.unassigned_users == ()


def test_report_flags_unassigned_policy(tmp_path: Path) -> None:
    layout = PilotLayout(
        pilot_root=tmp_path, app=PilotApp.LEARNING_TUTOR
    )
    _write_manifest(layout, "t01", "S9", app=PilotApp.LEARNING_TUTOR)
    report = build_comparison_report(
        pilot_root=tmp_path, app=PilotApp.LEARNING_TUTOR
    )
    assert "t01" in report.unassigned_users


def test_report_handles_truncated_session_log(tmp_path: Path) -> None:
    """A partial last line (e.g. crash mid-flush) must not blow up."""
    layout = PilotLayout(
        pilot_root=tmp_path, app=PilotApp.LEARNING_TUTOR
    )
    _write_manifest(layout, "t01", "S1", app=PilotApp.LEARNING_TUTOR)
    _write_session_log(layout, "t01", 1, agent_turns=2)
    sessions_dir = layout.sessions_dir("t01")
    log_path = sessions_dir / "session-0001.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "agent_turn", "turn_index": 2, "text": "p')
    report = build_comparison_report(
        pilot_root=tmp_path, app=PilotApp.LEARNING_TUTOR
    )
    # 2 complete agent turns survive; the partial line is silently dropped.
    assert report.per_policy["S1"].agent_turn_count == 2


def test_markdown_renderer_includes_headline_rows(tmp_path: Path) -> None:
    layout = PilotLayout(
        pilot_root=tmp_path, app=PilotApp.EMOTIONAL_COMPANION
    )
    _write_manifest(layout, "c01", "S1", app=PilotApp.EMOTIONAL_COMPANION)
    _write_manifest(layout, "c02", "S2", app=PilotApp.EMOTIONAL_COMPANION)
    _write_session_log(layout, "c01", 1, agent_turns=3, handoffs=1)
    _write_session_log(layout, "c02", 1, agent_turns=4)
    report = build_comparison_report(
        pilot_root=tmp_path, app=PilotApp.EMOTIONAL_COMPANION
    )
    md = render_comparison_markdown(report)
    assert "S1 (Ebbinghaus decay)" in md
    assert "S2 (capacity-only)" in md
    assert "users enrolled" in md
    assert "crisis handoffs" in md
    assert "c01" in md and "c02" in md
    assert "Judge-grade metrics" in md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = ab_cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def test_cli_assign_writes_plan_and_prints_json(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    rc, out, _err = _run_cli(
        [
            "assign",
            "--app",
            "learning_tutor",
            "--seed",
            "phase4b-tutor",
            "--user",
            "t01",
            "--user",
            "t02",
            "--user",
            "t03",
            "--user",
            "t04",
            "--out",
            str(plan_path),
        ]
    )
    assert rc == 0
    assert plan_path.exists()
    payload = json.loads(out)
    assert payload["app"] == "learning_tutor"
    assert set(payload["user_ids"]) == {"t01", "t02", "t03", "t04"}
    assert payload["counts"]["S1"] + payload["counts"]["S2"] == 4


def test_cli_assign_without_users_errors() -> None:
    rc, _out, err = _run_cli(
        ["assign", "--app", "learning_tutor", "--seed", "s"]
    )
    assert rc == 2
    assert "at least one --user" in err


def test_cli_show_single_user_policy(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _run_cli(
        [
            "assign",
            "--app",
            "learning_tutor",
            "--seed",
            "phase4b-tutor",
            "--user",
            "t01",
            "--user",
            "t02",
            "--out",
            str(plan_path),
        ]
    )
    rc, out, _err = _run_cli(
        ["show", "--plan", str(plan_path), "--user", "t01"]
    )
    assert rc == 0
    assert out.strip() in ("S1", "S2")


def test_cli_show_unknown_user_errors(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _run_cli(
        [
            "assign",
            "--app",
            "learning_tutor",
            "--seed",
            "s",
            "--user",
            "t01",
            "--out",
            str(plan_path),
        ]
    )
    rc, _out, err = _run_cli(
        ["show", "--plan", str(plan_path), "--user", "ghost"]
    )
    assert rc == 3
    assert "ghost" in err


def test_cli_report_renders_markdown_on_bare_invocation(
    tmp_path: Path,
) -> None:
    layout = PilotLayout(
        pilot_root=tmp_path, app=PilotApp.EMOTIONAL_COMPANION
    )
    _write_manifest(layout, "c01", "S1", app=PilotApp.EMOTIONAL_COMPANION)
    _write_manifest(layout, "c02", "S2", app=PilotApp.EMOTIONAL_COMPANION)
    _write_session_log(layout, "c01", 1, agent_turns=2)
    _write_session_log(layout, "c02", 1, agent_turns=2)

    rc, out, _err = _run_cli(
        [
            "report",
            "--app",
            "emotional_companion",
            "--pilot-root",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert "S1 (Ebbinghaus decay)" in out
    assert "c01" in out
    assert "c02" in out


def test_cli_report_writes_json_and_markdown(tmp_path: Path) -> None:
    layout = PilotLayout(
        pilot_root=tmp_path, app=PilotApp.LEARNING_TUTOR
    )
    _write_manifest(layout, "t01", "S1", app=PilotApp.LEARNING_TUTOR)
    _write_session_log(layout, "t01", 1, agent_turns=1)
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    rc, _out, err = _run_cli(
        [
            "report",
            "--app",
            "learning_tutor",
            "--pilot-root",
            str(tmp_path),
            "--out",
            str(out_json),
            "--markdown",
            str(out_md),
        ]
    )
    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["app"] == "learning_tutor"
    md = out_md.read_text(encoding="utf-8")
    assert "t01" in md
