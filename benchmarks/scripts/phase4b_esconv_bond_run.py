"""Phase 4b ESConv WAI-SR Bond live-fire.

For each ESConv conversation, build a simulated seeker, run
`run_esconv_session` against a live `CompanionAgent`, administer the
5-item WAI-SR, aggregate into a `CohortBondReport`.

Pivot §3 target:
- mean WAI-SR Bond ≥ 3.8 across ≥ 50 simulated users
- cohort spans ESConv emotion categories (reported, not gated)

Each user runs `max_turns` dialogue turns + 5 WAI-SR prompts.
Budget at ~30s/turn: 50 users × (6 turns × 2 LLM + 5 WAI-SR) × 30s
= ~4-6 hours. Smoke mode: --limit 3 to validate wiring first.

Usage::

    python benchmarks/scripts/phase4b_esconv_bond_run.py --limit 50 --max-turns 6
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
from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.eval.benchmarks.public_datasets.esconv import (  # noqa: E402
    iter_esconv_conversations,
)
from psychology_base.eval.benchmarks.public_datasets.esconv_wai_sr_harness import (  # noqa: E402
    administer_wai_sr_simulated,
    aggregate_cohort_bond,
    make_esconv_seeker,
    run_esconv_session,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = REPO / ".data" / "esconv" / "train.jsonl"
DEFAULT_OUT = REPO / "docs" / "experiments" / "phase-3-month-horizon" / "phase-4b" / "esconv-bond.json"
DEFAULT_OUT_ADVANCED = (
    REPO
    / "docs"
    / "experiments"
    / "phase-3-month-horizon"
    / "phase-4b"
    / "esconv-bond-advanced.json"
)
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
    p = argparse.ArgumentParser(prog="phase4b_esconv_bond_run")
    p.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Simulated users to run. Gate requires ≥50.",
    )
    p.add_argument("--max-turns", type=int, default=6)
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
        help="Output path; defaults to esconv-bond{,-advanced}.json based on --tier.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE
    if args.output is None:
        args.output = DEFAULT_OUT_ADVANCED if args.tier == "advanced" else DEFAULT_OUT
    if not args.shim.exists():
        raise SystemExit(
            f"Shim not found at {args.shim}; run provision_esconv.py first."
        )
    convs = []
    for conv in iter_esconv_conversations(args.shim):
        convs.append(conv)
        if len(convs) >= args.limit:
            break
    print(f"Loaded {len(convs)} ESConv conversations from {args.shim}")

    router, routing_note = _build_router(args.mode, tier)
    print(f"LLM routing [tier={args.tier}]: {routing_note}")
    responder = make_llm_responder(router, tier=tier)

    response_sets = []
    emotion_types = []
    excluded_handoffs = 0
    t0 = time.time()
    for idx, convo in enumerate(convs):
        user_id = f"u-bond-{idx:03d}"
        session_id = f"s-bond-{idx:03d}"
        cs = CompanionSystem(top_k=6)
        agent = CompanionAgent(
            user_id=user_id,
            session_id=session_id,
            companion_system=cs,
            llm_responder=responder,
        )
        seeker = make_esconv_seeker(router, convo)
        t_u = time.time()
        try:
            transcript = run_esconv_session(
                convo,
                companion_agent=agent,
                seeker=seeker,
                max_turns=args.max_turns,
                starting_timestamp=TS + _dt.timedelta(hours=idx),
            )
        except Exception as exc:
            print(f"  [{idx+1}/{len(convs)}] {convo.conversation_id}: FAIL session {exc!r}")
            continue
        if transcript.handoff_triggered:
            excluded_handoffs += 1
            print(
                f"  [{idx+1}/{len(convs)}] {convo.conversation_id}: handoff — skipped"
            )
            continue
        try:
            rs = administer_wai_sr_simulated(
                seeker,
                user_id=user_id,
                administered_at_seconds=(
                    TS + _dt.timedelta(hours=idx, minutes=30)
                ).timestamp(),
            )
        except Exception as exc:
            print(
                f"  [{idx+1}/{len(convs)}] {convo.conversation_id}: WAI-SR parse fail {exc!r}"
            )
            continue
        response_sets.append(rs)
        emotion_types.append(convo.emotion_type)
        dur = time.time() - t_u
        from psychology_base.applications.measurement.wai_sr import score_wai_sr
        bond = score_wai_sr(rs).bond
        print(
            f"  [{idx+1}/{len(convs)}] {convo.conversation_id} "
            f"[{convo.emotion_type}]: bond={bond:.2f} wall={dur:.1f}s"
        )

    wall = time.time() - t0
    report = aggregate_cohort_bond(
        response_sets,
        emotion_types=emotion_types,
        excluded_handoffs=excluded_handoffs,
    )
    pass_gate = report.meets_pivot_gate()
    print(
        f"\nn={report.n} mean_bond={report.mean_bond:.3f} "
        f"ci95_half={report.ci95_half_width:.3f} handoffs_excluded={excluded_handoffs}"
    )
    print(f"  emotion categories: {dict(report.emotion_type_counts)}")
    print(f"  pivot_gate_PASS = {pass_gate} (floor Bond≥3.8, N≥50)")

    out = {
        "benchmark": "phase4b_esconv_bond",
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "shim": str(args.shim),
        "limit": args.limit,
        "max_turns": args.max_turns,
        "n": report.n,
        "mean_bond": report.mean_bond,
        "stderr": report.stderr,
        "ci95_half_width": report.ci95_half_width,
        "excluded_handoffs": excluded_handoffs,
        "emotion_type_counts": dict(report.emotion_type_counts),
        "bond_scores": list(report.bond_scores),
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
