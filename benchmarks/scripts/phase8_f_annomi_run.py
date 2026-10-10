"""Phase 8 Pillar F — AnnoMI counseling-depth live-fire.

For each AnnoMI dialogue, build an LLM-simulated client seeded by the
first client utterance, run a short MI-dialogue against a fresh
:class:`CompanionAgent` + full safety stack, administer WAI-SR Bond,
and score the Companion's turns for MI-fidelity (reflection + open-
question rates) using a lexical classifier against the dataset's human-
baseline rates.

Pillar F AnnoMI sub-gate (per
``docs/phases/phase-8-dataset-replay-expansion-plan.md`` §2 pillar F +
this task's partial-pass pass-criteria):
- Companion reflection rate ≥ 0.30 absolute OR ≥ 50% of AnnoMI
  human-baseline reflection rate on matched dialogues

Budget shape: 133 dialogs × ``max_turns`` × 2 LLM calls per turn + 5
WAI-SR items each ≈ ~1500 cloud calls at max_turns=6 (≈ 1.5h at
qwen-turbo cache-warm). Smoke: ``--limit 5 --max-turns 3``.

Usage::

    python benchmarks/scripts/phase8_f_annomi_run.py --limit 50 --max-turns 6
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
from psychology_base.eval.benchmarks.public_datasets.annomi import (  # noqa: E402
    iter_annomi_dialogues,
)
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_scorer import (  # noqa: E402
    CounselingDepthReport,
    make_annomi_seeker,
    score_annomi_sweep,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_CSV = REPO / ".data" / "annomi" / "raw" / "AnnoMI-full.csv"
DEFAULT_OUT = (
    REPO
    / "docs"
    / "experiments"
    / "phase-8-dataset-replay-expansion"
    / "pillar-f"
    / "annomi-counseling-depth.json"
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
    p = argparse.ArgumentParser(prog="phase8_f_annomi_run")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Dialogs to run (full AnnoMI is 133).",
    )
    p.add_argument(
        "--mi-quality",
        choices=("high", "low", "any"),
        default="any",
        help="Restrict to high- or low-quality MI dialogues.",
    )
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--mode", choices=("local", "cloud"), default="cloud")
    p.add_argument("--tier", choices=("simple", "advanced"), default="simple")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--mi-floor", type=float, default=0.30,
        help="Absolute reflection-rate floor (default 0.30).",
    )
    p.add_argument(
        "--baseline-fraction", type=float, default=0.50,
        help="Alt gate: reflection rate ≥ this fraction of AnnoMI baseline "
             "(default 0.50 = half of human-therapist baseline).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE
    if not args.csv.exists():
        raise SystemExit(
            f"AnnoMI CSV not found at {args.csv}; "
            "run `python benchmarks/scripts/provision_annomi.py` first."
        )
    mi_filter = None if args.mi_quality == "any" else args.mi_quality
    dialogues = list(
        iter_annomi_dialogues(args.csv, limit=args.limit, mi_quality_filter=mi_filter)
    )
    print(
        f"Loaded {len(dialogues)} AnnoMI dialogues from {args.csv} "
        f"(mi_quality_filter={mi_filter!r})"
    )

    router, routing_note = _build_router(args.mode, tier)
    print(f"LLM routing [tier={args.tier}]: {routing_note}")
    base_responder = make_llm_responder(router, tier=tier)
    crisis_sink: list = []
    extreme_sink: list = []

    def _system_factory():
        return CompanionSystem(top_k=6)

    def _agent_factory(system, dialogue):
        return CompanionAgent(
            user_id=f"annomi_{dialogue.transcript_id}",
            session_id="annomi",
            companion_system=system,
            llm_responder=base_responder,
            ai_identity_reminder=AIIdentityReminder(),
            crisis_escalation=CrisisEscalation(handoff_sink=crisis_sink.append),
            extreme_emotion_watcher=ExtremeEmotionWatcher(
                review_sink=extreme_sink.append,
                user_id=f"annomi_{dialogue.transcript_id}",
            ),
        )

    def _seeker_factory(dialogue):
        return make_annomi_seeker(router, dialogue, tier=tier)

    t0 = time.time()

    def _progress(idx: int, result) -> None:
        bond_s = f"{result.bond:.2f}" if result.bond is not None else "—"
        tag = " HANDOFF" if result.handoff_triggered else ""
        print(
            f"  [{idx+1}/{len(dialogues)}] {result.transcript_id} [{result.mi_quality}]: "
            f"bond={bond_s} refl={result.companion_reflection_rate:.2f} "
            f"(base={result.baseline_reflection_rate:.2f}) "
            f"openQ={result.companion_open_question_rate:.2f}{tag}"
        )

    report = score_annomi_sweep(
        dialogues,
        system_factory=_system_factory,
        agent_factory=_agent_factory,
        seeker_factory=_seeker_factory,
        max_turns_per_dialog=args.max_turns,
        base_timestamp=BASE_TS,
        progress_sink=_progress,
    )
    wall = time.time() - t0

    mi_pass = report.meets_mi_fidelity_gate(
        absolute_floor=args.mi_floor,
        baseline_fraction=args.baseline_fraction,
    )
    print(
        f"\nn={report.n} mean_bond={report.mean_bond:.3f} "
        f"refl_rate={report.mean_reflection_rate:.3f} "
        f"(base={report.mean_baseline_reflection_rate:.3f}) "
        f"openQ_rate={report.mean_open_question_rate:.3f} "
        f"(base={report.mean_baseline_open_question_rate:.3f}) "
        f"handoffs={report.excluded_handoffs}"
    )
    print(
        f"  mi_fidelity_gate(floor={args.mi_floor} or "
        f"≥{args.baseline_fraction*100:.0f}% of baseline) PASS={mi_pass}"
    )
    print(f"  crisis_handoffs_sunk={len(crisis_sink)} extreme_emotion_sunk={len(extreme_sink)}")

    pillar = CounselingDepthReport(
        annomi=report,
        pillar_f_mi_floor=args.mi_floor,
    )

    out = {
        "benchmark": "phase8_f_annomi_counseling_depth",
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "csv": str(args.csv),
        "limit": args.limit,
        "mi_quality_filter": mi_filter,
        "max_turns": args.max_turns,
        "n": report.n,
        "mean_bond": report.mean_bond,
        "mean_reflection_rate": report.mean_reflection_rate,
        "mean_baseline_reflection_rate": report.mean_baseline_reflection_rate,
        "mean_open_question_rate": report.mean_open_question_rate,
        "mean_baseline_open_question_rate": report.mean_baseline_open_question_rate,
        "excluded_handoffs": report.excluded_handoffs,
        "mi_floor": args.mi_floor,
        "baseline_fraction": args.baseline_fraction,
        "mi_fidelity_gate_pass": mi_pass,
        "pillar_f_annomi_pass": mi_pass,
        "per_dialog": [
            {
                "transcript_id": d.transcript_id,
                "mi_quality": d.mi_quality,
                "bond": d.bond,
                "handoff_triggered": d.handoff_triggered,
                "companion_reflection_rate": d.companion_reflection_rate,
                "baseline_reflection_rate": d.baseline_reflection_rate,
                "companion_open_question_rate": d.companion_open_question_rate,
                "baseline_open_question_rate": d.baseline_open_question_rate,
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
