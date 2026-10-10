"""Phase 8 Pillar A — MSC × WAI-SR longitudinal Bond live-fire.

For each MSC conversation, build a persona-grounded simulated seeker,
run `run_msc_conversation` end-to-end across all its sessions against a
live `CompanionAgent` (memory persists across sessions, with in-world
day-level gaps between them), administer WAI-SR at session 1 and session
5, aggregate into a `LongitudinalBondReport`.

Pillar A hard gate (per
`docs/phases/phase-8-dataset-replay-expansion-plan.md` §2):
- mean WAI-SR Bond ≥ 3.89 at session 5 across ≥ 50 simulated users
  (Woebot aggregate floor)
- cohort spans MSC personas (reported, not gated)

Pillar A soft signal (reported, not gated):
- paired Δ(session_5 − session_1) ≥ 0 — longitudinal bond growth

Budget shape: 50 users × 5 sessions × 6 turns × 2 LLM calls ≈ 3000 LLM
calls. At qwen-turbo ~2–4s/call this is 2–3 hours cache-warm; qwen-max
is ~4–6×. Smoke: --limit 3 --max-turns-per-session 2 for wiring check.

Usage::

    python benchmarks/scripts/phase8_a_msc_run.py --limit 50 --max-turns-per-session 6
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

from psychology_base.applications.emotional import CompanionAgent  # noqa: E402
from psychology_base.applications.llm_responders import make_llm_responder  # noqa: E402
from psychology_base.applications.measurement.wai_sr import score_wai_sr  # noqa: E402
from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.eval.benchmarks.public_datasets.msc import (  # noqa: E402
    iter_msc_conversations,
)
from psychology_base.eval.benchmarks.public_datasets.msc_bond_longitudinal import (  # noqa: E402
    aggregate_longitudinal_bond,
    make_msc_seeker,
    run_msc_conversation,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = REPO / ".data" / "msc" / "train.jsonl"
DEFAULT_OUT_SIMPLE = (
    REPO
    / "docs"
    / "experiments"
    / "phase-8-dataset-replay-expansion"
    / "pillar-a"
    / "msc-bond-longitudinal.json"
)
DEFAULT_OUT_ADVANCED = (
    REPO
    / "docs"
    / "experiments"
    / "phase-8-dataset-replay-expansion"
    / "pillar-a"
    / "msc-bond-longitudinal-advanced.json"
)
BASE_TS = _dt.datetime(2026, 10, 8, 12, 0, 0, tzinfo=_dt.timezone.utc)


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


def _build_router(mode: str, tier: TaskTier) -> tuple[LLMRouter, str]:
    cfg = _config_with_overrides(mode)
    if mode == "cloud":
        env = cfg.get_provider(cfg.active_provider).api_key_env
        if env and not os.environ.get(env):
            raise SystemExit(f"--mode cloud requires {env} set.")
    router = LLMRouter(cfg)
    dec = router.route(tier)[1]
    if mode == "local" and dec.provider != "ollama":
        raise SystemExit(
            f"--mode local but router routed to {dec.provider} ({dec.reason})."
        )
    return router, f"{dec.provider} / {dec.model_id} (reason={dec.reason})"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase8_a_msc_run")
    p.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Simulated users to run. Pillar A gate requires ≥50.",
    )
    p.add_argument("--max-turns-per-session", type=int, default=6)
    p.add_argument("--mode", choices=("local", "cloud"), default="local")
    p.add_argument(
        "--tier",
        choices=("simple", "advanced"),
        default="simple",
        help="LLM tier for CompanionAgent responder (simple=qwen-turbo, advanced=qwen-max).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; defaults to msc-bond-longitudinal{,-advanced}.json based on --tier.",
    )
    p.add_argument(
        "--session-1-id",
        type=int,
        default=1,
        help="Session id for early-bond measurement (default: 1).",
    )
    p.add_argument(
        "--session-5-id",
        type=int,
        default=5,
        help="Session id for longitudinal-bond measurement (default: 5).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE
    if args.output is None:
        args.output = DEFAULT_OUT_ADVANCED if args.tier == "advanced" else DEFAULT_OUT_SIMPLE
    if not args.shim.exists():
        raise SystemExit(
            f"Shim not found at {args.shim}; run provision_msc.py first."
        )
    convs = []
    for conv in iter_msc_conversations(args.shim):
        convs.append(conv)
        if len(convs) >= args.limit:
            break
    print(f"Loaded {len(convs)} MSC conversations from {args.shim}")

    router, routing_note = _build_router(args.mode, tier)
    print(f"LLM routing [tier={args.tier}]: {routing_note}")
    responder = make_llm_responder(router, tier=tier)

    user_runs = []
    persona_labels = []
    t0 = time.time()
    for idx, convo in enumerate(convs):
        user_id = f"u-msc-{idx:03d}"
        session_id = f"s-msc-{idx:03d}"
        cs = CompanionSystem(top_k=6)
        agent = CompanionAgent(
            user_id=user_id,
            session_id=session_id,
            companion_system=cs,
            llm_responder=responder,
        )
        seeker = make_msc_seeker(router, convo)
        t_u = time.time()
        try:
            run = run_msc_conversation(
                convo,
                companion_agent=agent,
                seeker=seeker,
                max_turns_per_session=args.max_turns_per_session,
                base_timestamp=BASE_TS + _dt.timedelta(days=idx * 60),
                wai_sr_at_sessions=(args.session_1_id, args.session_5_id),
                user_id=user_id,
            )
        except Exception as exc:
            print(f"  [{idx+1}/{len(convs)}] {convo.conversation_id}: FAIL conv {exc!r}")
            continue
        user_runs.append(run)
        label = _persona_label(convo)
        persona_labels.append(label)
        dur = time.time() - t_u
        s1 = run.wai_sr_by_session_id.get(args.session_1_id)
        s5 = run.wai_sr_by_session_id.get(args.session_5_id)
        s1_bond = f"{score_wai_sr(s1).bond:.2f}" if s1 is not None else "—"
        s5_bond = f"{score_wai_sr(s5).bond:.2f}" if s5 is not None else "—"
        tag = " HANDOFF" if run.handoff_triggered else ""
        print(
            f"  [{idx+1}/{len(convs)}] {convo.conversation_id} [{label}]: "
            f"s{args.session_1_id}={s1_bond} s{args.session_5_id}={s5_bond} "
            f"wall={dur:.1f}s{tag}"
        )

    wall = time.time() - t0
    report = aggregate_longitudinal_bond(
        user_runs,
        persona_labels=persona_labels,
        session_1_id=args.session_1_id,
        session_5_id=args.session_5_id,
    )
    pass_gate = report.meets_pillar_gate()
    print(
        f"\nn_s1={report.n_session_1} n_s5={report.n_session_5} "
        f"paired={report.n_paired}"
    )
    print(
        f"  mean_bond_s{args.session_1_id}={report.mean_session_1_bond:.3f} "
        f"mean_bond_s{args.session_5_id}={report.mean_session_5_bond:.3f} "
        f"delta={report.mean_paired_delta:+.3f}"
    )
    print(
        f"  s{args.session_5_id} ci95_half={report.session_5_ci95_half_width:.3f} "
        f"handoffs_excluded={report.excluded_handoffs}"
    )
    print(f"  persona labels: {dict(report.persona_counts)}")
    print(f"  pillar_a_PASS = {pass_gate} (floor Bond≥3.89, N≥50)")

    out = {
        "benchmark": "phase8_a_msc_bond_longitudinal",
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "shim": str(args.shim),
        "limit": args.limit,
        "max_turns_per_session": args.max_turns_per_session,
        "session_1_id": args.session_1_id,
        "session_5_id": args.session_5_id,
        "n_session_1": report.n_session_1,
        "n_session_5": report.n_session_5,
        "n_paired": report.n_paired,
        "mean_bond_session_1": report.mean_session_1_bond,
        "mean_bond_session_5": report.mean_session_5_bond,
        "mean_paired_delta": report.mean_paired_delta,
        "session_5_stderr": report.session_5_stderr,
        "session_5_ci95_half_width": report.session_5_ci95_half_width,
        "paired_delta_stderr": report.paired_delta_stderr,
        "excluded_users": report.excluded_users,
        "excluded_handoffs": report.excluded_handoffs,
        "persona_counts": dict(report.persona_counts),
        "session_1_bond_scores": list(report.session_1_bond_scores),
        "session_5_bond_scores": list(report.session_5_bond_scores),
        "paired_deltas": list(report.paired_deltas),
        "pillar_a_pass": pass_gate,
        "wall_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


def _persona_label(convo) -> str:
    """Coarse first-bullet label for the self-persona.

    Readability-only — used to populate ``persona_counts`` so a reader can
    see the cohort spans distinct personas without exposing full persona
    text in the summary.
    """
    try:
        bullets = convo.personas[0]
    except (IndexError, AttributeError):
        return "unlabeled"
    if not bullets:
        return "unlabeled"
    first = bullets[0]
    return first[:60].strip() or "unlabeled"


if __name__ == "__main__":
    raise SystemExit(main())
