"""Phase 8 Pillar F — ExTES counseling-depth live-fire.

For each ExTES example, build an LLM-simulated seeker from the client
utterance, run a short counseling dialogue against a fresh
:class:`CompanionAgent` + full safety stack, administer WAI-SR Bond to
the simulated seeker, score lexical empathy on both companion + the
dataset's reference counselor response for an EMPATH-κ proxy.

Pillar F ExTES sub-gates (per
``docs/phases/phase-8-dataset-replay-expansion-plan.md`` §2 pillar F +
the task's partial-pass pass-criteria):
- mean WAI-SR Bond ≥ 3.89 (Woebot aggregate floor)
- EMPATH-κ ≥ 0.50 (Cohen's κ between lexical empathy judge's
  companion-side and reference-side binary labels across the dialog sample)

Budget shape: N dialogs × ``max_turns`` turns × 2 LLM calls per turn
(companion + seeker) + 5 WAI-SR items per dialog ≈ N × (2M + 5). At
N=100 and max_turns=4 this is ~1300 cloud calls (≈ 1.0h at qwen-turbo
cache-warm). Smoke: ``--limit 3 --max-turns 2`` for wiring check.

Usage::

    python benchmarks/scripts/phase8_f_extes_run.py --limit 100 --max-turns 4
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

sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.applications.emotional import CompanionAgent  # noqa: E402
from psychology_base.applications.llm_responders import make_llm_responder  # noqa: E402
from psychology_base.applications.safety import (  # noqa: E402
    AIIdentityReminder,
    CrisisEscalation,
    ExtremeEmotionWatcher,
)
from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_scorer import (  # noqa: E402
    CounselingDepthReport,
    make_extes_seeker,
    score_extes_sweep,
)
from psychology_base.eval.benchmarks.public_datasets.extes import (  # noqa: E402
    iter_extes_examples,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = REPO / ".data" / "extes" / "test.jsonl"
DEFAULT_OUT = (
    REPO
    / "docs"
    / "experiments"
    / "phase-8-dataset-replay-expansion"
    / "pillar-f"
    / "extes-counseling-depth.json"
)
BASE_TS = _dt.datetime(2026, 10, 9, 12, 0, 0, tzinfo=_dt.timezone.utc)


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
    p = argparse.ArgumentParser(prog="phase8_f_extes_run")
    p.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Dialogs to run. Pillar F plan uses up to 100 for cost reasons.",
    )
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--mode", choices=("local", "cloud"), default="cloud")
    p.add_argument("--tier", choices=("simple", "advanced"), default="simple")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--bond-floor", type=float, default=3.89,
        help="Woebot aggregate floor (default 3.89).",
    )
    p.add_argument(
        "--kappa-floor", type=float, default=0.50,
        help="EMPATH-κ gate floor (default 0.50).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE
    if not args.shim.exists():
        raise SystemExit(
            f"Shim not found at {args.shim}; "
            "run `python benchmarks/scripts/provision_extes.py` first."
        )
    examples = list(iter_extes_examples(args.shim, limit=args.limit))
    print(f"Loaded {len(examples)} ExTES examples from {args.shim}")

    router, routing_note = _build_router(args.mode, tier)
    print(f"LLM routing [tier={args.tier}]: {routing_note}")
    base_responder = make_llm_responder(router, tier=tier)
    crisis_sink: list = []
    extreme_sink: list = []

    def _system_factory():
        return CompanionSystem(top_k=6)

    def _agent_factory(system, example):
        return CompanionAgent(
            user_id=f"extes_{example.example_id}",
            session_id="extes",
            companion_system=system,
            llm_responder=base_responder,
            ai_identity_reminder=AIIdentityReminder(),
            crisis_escalation=CrisisEscalation(handoff_sink=crisis_sink.append),
            extreme_emotion_watcher=ExtremeEmotionWatcher(
                review_sink=extreme_sink.append,
                user_id=f"extes_{example.example_id}",
            ),
        )

    def _seeker_factory(example):
        return make_extes_seeker(router, example, tier=tier)

    t0 = time.time()

    def _progress(idx: int, result) -> None:
        bond_s = f"{result.bond:.2f}" if result.bond is not None else "—"
        tag = " HANDOFF" if result.handoff_triggered else ""
        print(
            f"  [{idx+1}/{len(examples)}] {result.example_id}: "
            f"bond={bond_s} comp_e={result.companion_empathy_score:.2f} "
            f"ref_e={result.reference_empathy_score:.2f}{tag}"
        )

    report = score_extes_sweep(
        examples,
        system_factory=_system_factory,
        agent_factory=_agent_factory,
        seeker_factory=_seeker_factory,
        max_turns_per_dialog=args.max_turns,
        base_timestamp=BASE_TS,
        progress_sink=_progress,
    )
    wall = time.time() - t0

    bond_pass = report.meets_bond_gate(args.bond_floor)
    kappa_pass = report.meets_kappa_gate(args.kappa_floor)
    print(
        f"\nn={report.n} mean_bond={report.mean_bond:.3f} "
        f"kappa={report.empath_kappa:.3f} "
        f"handoffs={report.excluded_handoffs}"
    )
    print(
        f"  bond_gate(floor={args.bond_floor}) PASS={bond_pass}  "
        f"kappa_gate(floor={args.kappa_floor}) PASS={kappa_pass}"
    )
    print(f"  crisis_handoffs_sunk={len(crisis_sink)} extreme_emotion_sunk={len(extreme_sink)}")

    pillar = CounselingDepthReport(
        extes=report,
        pillar_f_bond_floor=args.bond_floor,
        pillar_f_kappa_floor=args.kappa_floor,
    )

    out = {
        "benchmark": "phase8_f_extes_counseling_depth",
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "shim": str(args.shim),
        "limit": args.limit,
        "max_turns": args.max_turns,
        "n": report.n,
        "mean_bond": report.mean_bond,
        "empath_kappa": report.empath_kappa,
        "mean_companion_empathy": (
            sum(d.companion_empathy_score for d in report.per_dialog) / report.n
            if report.n else 0.0
        ),
        "mean_reference_empathy": (
            sum(d.reference_empathy_score for d in report.per_dialog) / report.n
            if report.n else 0.0
        ),
        "excluded_handoffs": report.excluded_handoffs,
        "bond_floor": args.bond_floor,
        "kappa_floor": args.kappa_floor,
        "bond_gate_pass": bond_pass,
        "kappa_gate_pass": kappa_pass,
        "pillar_f_extes_pass": bond_pass and kappa_pass,
        "per_dialog": [
            {
                "example_id": d.example_id,
                "bond": d.bond,
                "handoff_triggered": d.handoff_triggered,
                "companion_empathy_score": d.companion_empathy_score,
                "reference_empathy_score": d.reference_empathy_score,
                "companion_empathy_binary": d.companion_empathy_binary,
                "reference_empathy_binary": d.reference_empathy_binary,
                "n_companion_turns": len(d.companion_turns),
            }
            for d in report.per_dialog
        ],
        "crisis_sink_count": len(crisis_sink),
        "extreme_emotion_sink_count": len(extreme_sink),
        "wall_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
