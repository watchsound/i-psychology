"""Phase 4b MathDial Socratic + EMPATH live-fire.

Pools MathDial teacher turns from the provisioned shim, scores each
through the lexical classifier (auto-compliance + taxonomy overlap) and
the EMPATH judge (κ against gold). Writes a numeric report under
`docs/experiments/phase-4-month-horizon/phase-4b/mathdial-empath.json`.

Pivot §3 target (triple floor):
- auto-compliance rate ≥ 0.95
- MathDial taxonomy overlap ≥ 0.90
- EMPATH κ against gold ≥ 0.60
- min 100 pooled teacher turns

Usage::

    python benchmarks/scripts/phase4b_mathdial_empath_run.py --mode local --limit 25
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

from psychology_base.applications.measurement.empath_judge import (  # noqa: E402
    LLMEMPATHJudge,
    StubEMPATHJudge,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial import (  # noqa: E402
    iter_mathdial_conversations,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial_empath_scorer import (  # noqa: E402
    score_empath_against_mathdial,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = REPO / ".data" / "mathdial" / "train.jsonl"
DEFAULT_OUT = REPO / "docs" / "experiments" / "phase-3-month-horizon" / "phase-4b" / "mathdial-empath.json"
DEFAULT_OUT_ADVANCED = (
    REPO
    / "docs"
    / "experiments"
    / "phase-3-month-horizon"
    / "phase-4b"
    / "mathdial-empath-advanced.json"
)


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
    p = argparse.ArgumentParser(prog="phase4b_mathdial_empath_run")
    p.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    p.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Conversations to pool from the shim (each ~3-10 teacher turns).",
    )
    p.add_argument(
        "--mode",
        choices=("local", "cloud", "stub"),
        default="local",
        help="Judge backend: local Ollama / cloud / stub (lexical, no LLM).",
    )
    p.add_argument(
        "--tier",
        choices=("simple", "advanced"),
        default="simple",
        help="LLM tier for the EMPATH judge (simple=qwen-turbo, advanced=qwen-max).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; defaults to mathdial-empath{,-advanced}.json based on --tier.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE
    if args.output is None:
        args.output = DEFAULT_OUT_ADVANCED if args.tier == "advanced" else DEFAULT_OUT
    if not args.shim.exists():
        raise SystemExit(
            f"Shim not found at {args.shim}; run provision_mathdial.py first."
        )
    convs = []
    for conv in iter_mathdial_conversations(args.shim):
        convs.append(conv)
        if len(convs) >= args.limit:
            break
    print(f"Loaded {len(convs)} conversations from {args.shim}")

    if args.mode == "stub":
        judge = StubEMPATHJudge()
        routing_note = "stub (lexical, no LLM)"
    else:
        router, routing_note = _build_router(args.mode, tier)
        judge = LLMEMPATHJudge(router=router, tier=tier)
    print(f"EMPATH judge [tier={args.tier}]: {routing_note}")

    t0 = time.time()
    report = score_empath_against_mathdial(convs, judge)
    wall = time.time() - t0

    kappa = report.empath_kappa()
    pivot_pass = report.meets_pivot_gate()
    print(
        f"\nsampled_teacher_turns={report.sampled_teacher_turns} "
        f"wall={wall:.1f}s"
    )
    print(f"  auto_compliance_rate = {report.auto_compliance_rate():.3f} (floor 0.95)")
    print(f"  taxonomy_overlap_rate= {report.taxonomy_overlap_rate():.3f} (floor 0.90)")
    print(
        f"  empath_kappa         = {kappa.kappa:.3f} on n={kappa.n_items} non-BORDERLINE (floor 0.60)"
    )
    print(f"  all_proxy_verdicts   = {report.all_proxy_verdicts}")
    print(
        f"  judge_parse_failures = {report.judge_parse_failures} "
        f"({report.judge_failure_rate():.3f} rate, floor 0.20)"
    )
    print(f"  pivot_gate_PASS      = {pivot_pass}")

    out = {
        "benchmark": "phase4b_mathdial_empath",
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "shim": str(args.shim),
        "conversations_pooled": len(convs),
        "sampled_teacher_turns": report.sampled_teacher_turns,
        "auto_compliance_rate": report.auto_compliance_rate(),
        "taxonomy_overlap_rate": report.taxonomy_overlap_rate(),
        "empath_kappa": kappa.kappa,
        "empath_kappa_n": kappa.n_items,
        "empath_kappa_po": kappa.po,
        "empath_kappa_pe": kappa.pe,
        "all_proxy_verdicts": report.all_proxy_verdicts,
        "judge_parse_failures": report.judge_parse_failures,
        "judge_failure_rate": report.judge_failure_rate(),
        "pivot_gate_pass": pivot_pass,
        "wall_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
