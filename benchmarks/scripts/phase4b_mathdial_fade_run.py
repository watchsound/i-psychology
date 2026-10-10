"""Phase 4b MathDial fade-schedule live-fire.

For each MathDial problem in the shim, run `TutorAgent` × simulated
student through `run_mathdial_tutor_session`, aggregate into a
`FadeScheduleReport`, write a numeric report under
`docs/experiments/phase-3-month-horizon/phase-4b/mathdial-fade.json`.

Pivot §3 target:
- unaided-solve rate ≥ 0.60
- ≥ 100 pooled fade events
- ≥ 30 distinct problems

Each session runs `max_turns` tutor replies. With local Ollama at ~30s/
turn and ~2 LLM calls per cycle (tutor + student), budget scales linearly
with ``--limit × --max-turns``. Default --limit=20 is the floor for a
plumbing-sanity run; the gate itself requires --limit≥30 and more if
many problems surrender.

Usage::

    python benchmarks/scripts/phase4b_mathdial_fade_run.py --limit 20 --max-turns 8
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

# Windows background `python` block-buffers stdout without -u / PYTHONUNBUFFERED,
# producing 0-byte logs for hour-long runs. Force line buffering so progress
# prints flush as they happen.
sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.applications.learning import TutorAgent  # noqa: E402
from psychology_base.applications.llm_responders import make_llm_responder  # noqa: E402
from psychology_base.applications.measurement.scaffolding_telemetry import (  # noqa: E402
    ScaffoldingTelemetry,
)
from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.eval.benchmarks.public_datasets.mathdial import (  # noqa: E402
    iter_mathdial_conversations,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial_fade_harness import (  # noqa: E402
    aggregate_fade_schedule,
    make_mathdial_student,
    run_mathdial_tutor_session,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = REPO / ".data" / "mathdial" / "train.jsonl"
DEFAULT_OUT = REPO / "docs" / "experiments" / "phase-3-month-horizon" / "phase-4b" / "mathdial-fade.json"
TS = _dt.datetime(2026, 10, 7, 12, 0, 0, tzinfo=_dt.timezone.utc)


def _config_with_overrides(mode: str):
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
    return cfg


def _build_router(mode: str) -> tuple[LLMRouter, str]:
    cfg = _config_with_overrides(mode)
    if mode == "cloud":
        env = cfg.get_provider(cfg.active_provider).api_key_env
        if env and not os.environ.get(env):
            raise SystemExit(f"--mode cloud requires {env} set.")
    router = LLMRouter(cfg)
    dec = router.route(TaskTier.SIMPLE)[1]
    if mode == "local" and dec.provider != "ollama":
        raise SystemExit(
            f"--mode local but router routed to {dec.provider} ({dec.reason})."
        )
    return router, f"{dec.provider} / {dec.model_id} (reason={dec.reason})"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase4b_mathdial_fade_run")
    p.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Problems to run. Gate requires ≥30 problems + ≥100 events.",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Tutor replies per session.",
    )
    p.add_argument(
        "--mode",
        choices=("local", "cloud"),
        default="local",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.shim.exists():
        raise SystemExit(
            f"Shim not found at {args.shim}; run provision_mathdial.py first."
        )
    convs = []
    for conv in iter_mathdial_conversations(args.shim):
        if not conv.ground_truth or not conv.student_incorrect_solution:
            continue
        convs.append(conv)
        if len(convs) >= args.limit:
            break
    print(f"Loaded {len(convs)} problems from {args.shim}")

    router, routing_note = _build_router(args.mode)
    print(f"LLM routing: {routing_note}")

    responder = make_llm_responder(router, tier=TaskTier.SIMPLE)

    results = []
    t0 = time.time()
    for idx, convo in enumerate(convs):
        telemetry = ScaffoldingTelemetry(user_id=f"u-fade-{idx:03d}")
        cs = CompanionSystem(top_k=4)
        agent = TutorAgent(
            user_id=f"u-fade-{idx:03d}",
            session_id=f"s-fade-{idx:03d}",
            companion_system=cs,
            llm_responder=responder,
            telemetry=telemetry,
        )
        student = make_mathdial_student(router, convo)
        t_prob = time.time()
        result = run_mathdial_tutor_session(
            convo,
            tutor_agent=agent,
            telemetry=telemetry,
            student=student,
            max_turns=args.max_turns,
            starting_timestamp=TS + _dt.timedelta(hours=idx),
            problem_id=convo.conversation_id,
        )
        results.append(result)
        dur = time.time() - t_prob
        print(
            f"  [{idx+1}/{len(convs)}] {convo.conversation_id}: "
            f"solved_unaided={result.student_solved_unaided} "
            f"surrender={result.tutor_handed_answer} "
            f"turns={len(result.turns)} wall={dur:.1f}s"
        )

    wall = time.time() - t0
    report = aggregate_fade_schedule(results)
    pass_gate = report.meets_pivot_gate()
    print(
        f"\nproblem_count={report.problem_count} session_count={report.session_count} "
        f"fade_events={report.fade_event_count} successful={report.successful_fades}"
    )
    print(f"  unaided_solve_rate   = {report.unaided_solve_rate:.3f} (floor 0.60)")
    print(f"  tutor_surrender_rate = {report.tutor_surrender_rate:.3f}")
    print(f"  pivot_gate_PASS      = {pass_gate} (needs ≥30 problems + ≥100 events)")

    out = {
        "benchmark": "phase4b_mathdial_fade",
        "mode": args.mode,
        "routing": routing_note,
        "shim": str(args.shim),
        "limit": args.limit,
        "max_turns": args.max_turns,
        "problem_count": report.problem_count,
        "session_count": report.session_count,
        "fade_event_count": report.fade_event_count,
        "successful_fades": report.successful_fades,
        "tutor_surrender_count": report.tutor_surrender_count,
        "unaided_solve_rate": report.unaided_solve_rate,
        "tutor_surrender_rate": report.tutor_surrender_rate,
        "pivot_gate_pass": pass_gate,
        "wall_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
