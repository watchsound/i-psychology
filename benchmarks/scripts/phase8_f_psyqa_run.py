"""Phase 8 Pillar F — PsyQA Chinese counseling-depth LLM-judge live-fire.

For each PsyQA record, build an LLM-simulated Chinese help-seeker from
``question + description``, run a short counseling dialogue against a
fresh :class:`CompanionAgent` + full safety stack, administer WAI-SR
Bond to the simulated seeker, and score empathy via the Chinese-register
LLM judge on both the Companion's live replies and the dataset's
reference ``counselor_response``.

Pillar F PsyQA sub-gates (per
``docs/phases/phase-8-dataset-replay-expansion-plan.md`` §2 pillar F):
- mean WAI-SR Bond ≥ 3.89 (Woebot aggregate floor)
- EMPATH-κ ≥ 0.50 (Cohen's κ between judge's companion-side and
  reference-side binary labels across the dialog sample)

Budget shape: N dialogs × ``max_turns`` turns × (2 chat calls + 5 WAI-SR
items + 1 judge call for ref + ``max_turns`` judge calls for Companion).
At N=100 max_turns=4 this is ≈ 100 × (8 + 5 + 1 + 4) = 1800 cloud calls
(≈ 1.2h at qwen-turbo cache-warm).

Usage::

    python benchmarks/scripts/phase8_f_psyqa_run.py --limit 100 --max-turns 4
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
from psychology_base.eval.benchmarks.public_datasets.psyqa import (  # noqa: E402
    iter_psyqa_examples,
)
from psychology_base.eval.benchmarks.public_datasets.psyqa_counseling_depth import (  # noqa: E402
    judge_vs_lexical_kappa_psyqa,
    make_llm_empathy_judge_zh,
    make_psyqa_seeker,
    score_psyqa_sweep_llm,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = REPO / ".data" / "psyqa" / "test.jsonl"
DEFAULT_OUT = (
    REPO
    / "docs"
    / "experiments"
    / "phase-8-dataset-replay-expansion"
    / "pillar-f"
    / "psyqa-llm-judge.json"
)
BASE_TS = _dt.datetime(2026, 10, 10, 12, 0, 0, tzinfo=_dt.timezone.utc)


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
    p = argparse.ArgumentParser(prog="phase8_f_psyqa_run")
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
            "run `python benchmarks/scripts/provision_psyqa.py` first."
        )
    examples = list(iter_psyqa_examples(args.shim, limit=args.limit))
    print(f"Loaded {len(examples)} PsyQA examples from {args.shim}")

    router, routing_note = _build_router(args.mode, tier)
    print(f"LLM routing [tier={args.tier}]: {routing_note}")
    base_responder = make_llm_responder(router, tier=tier)
    crisis_sink: list = []
    extreme_sink: list = []

    def _system_factory():
        return CompanionSystem(top_k=6)

    def _agent_factory(system, example):
        return CompanionAgent(
            user_id=f"psyqa_{example.example_id}",
            session_id="psyqa",
            companion_system=system,
            llm_responder=base_responder,
            ai_identity_reminder=AIIdentityReminder(),
            crisis_escalation=CrisisEscalation(handoff_sink=crisis_sink.append),
            extreme_emotion_watcher=ExtremeEmotionWatcher(
                review_sink=extreme_sink.append,
                user_id=f"psyqa_{example.example_id}",
            ),
        )

    def _seeker_factory(example):
        return make_psyqa_seeker(router, example, tier=tier)

    judge = make_llm_empathy_judge_zh(router, tier=tier)

    t0 = time.time()

    def _progress(idx: int, result) -> None:
        bond_s = f"{result.bond:.2f}" if result.bond is not None else "—"
        tag = " HANDOFF" if result.handoff_triggered else ""
        print(
            f"  [{idx+1}/{len(examples)}] {result.example_id}: "
            f"bond={bond_s} comp_bin_rate={result.companion_empathy_binary_rate:.2f} "
            f"ref_bin={result.reference_empathy_binary}{tag}"
        )

    report = score_psyqa_sweep_llm(
        examples,
        system_factory=_system_factory,
        agent_factory=_agent_factory,
        seeker_factory=_seeker_factory,
        judge=judge,
        max_turns_per_dialog=args.max_turns,
        base_timestamp=BASE_TS,
        judge_routing=routing_note,
        progress_sink=_progress,
    )
    wall = time.time() - t0

    bond_pass = report.meets_bond_gate(args.bond_floor)
    kappa_pass = report.meets_kappa_gate(args.kappa_floor)
    diag_kappa = judge_vs_lexical_kappa_psyqa(report, examples)
    print(
        f"\nn={report.n} mean_bond={report.mean_bond:.3f} "
        f"empath_kappa={report.empath_kappa:.3f} "
        f"handoffs={report.excluded_handoffs} "
        f"unresolved={report.n_unresolved}"
    )
    print(
        f"  bond_gate(floor={args.bond_floor}) PASS={bond_pass}  "
        f"kappa_gate(floor={args.kappa_floor}) PASS={kappa_pass}"
    )
    print(
        f"  diagnostic_judge_vs_lexical_kappa={diag_kappa:.3f} "
        f"companion_binary_rate={report.mean_llm_judge_companion_empathy_binary_rate:.4f} "
        f"reference_binary_rate={report.mean_llm_judge_reference_empathy_binary_rate:.4f}"
    )
    print(
        f"  crisis_handoffs_sunk={len(crisis_sink)} extreme_emotion_sunk={len(extreme_sink)}"
    )

    out = {
        "benchmark": "phase8_f_psyqa_llm_judge",
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "shim": str(args.shim),
        "limit": args.limit,
        "max_turns": args.max_turns,
        "n": report.n,
        "mean_bond": report.mean_bond,
        "empath_kappa": report.empath_kappa,
        "diagnostic_judge_vs_lexical_kappa": diag_kappa,
        "mean_llm_judge_companion_empathy_binary_rate":
            report.mean_llm_judge_companion_empathy_binary_rate,
        "mean_llm_judge_reference_empathy_binary_rate":
            report.mean_llm_judge_reference_empathy_binary_rate,
        "excluded_handoffs": report.excluded_handoffs,
        "n_unresolved": report.n_unresolved,
        "unresolved_dialog_ids": list(report.unresolved_dialog_ids),
        "bond_floor": args.bond_floor,
        "kappa_floor": args.kappa_floor,
        "bond_gate_pass": bond_pass,
        "kappa_gate_pass": kappa_pass,
        "pillar_f_psyqa_pass": bond_pass and kappa_pass,
        "per_dialog": [
            {
                "example_id": d.example_id,
                "bond": d.bond,
                "handoff_triggered": d.handoff_triggered,
                "companion_empathy_binary_rate": d.companion_empathy_binary_rate,
                "reference_empathy_score": d.reference_empathy_score,
                "reference_empathy_binary": d.reference_empathy_binary,
                "companion_empathy_binary": d.companion_empathy_binary,
                "unresolved": d.unresolved,
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
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
