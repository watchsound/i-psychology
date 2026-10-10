"""Phase 4b B1 scripted chain-reference live-fire.

Drives a scripted multi-session father-waiting reinterpretation stream
through a live `CompanionAgent` and auto-labels chain-reference cues
in the agent's responses. Writes a numeric report under
`docs/experiments/phase-3-month-horizon/phase-4b/b1-chain-reference.json`.

Pivot §3 target:
- ≥ 80% appropriate-cite rate on reinterpretation moments
- ≥ 5 reinterpretation turns in the script (met by this 9-session arc)

Script lives inline — the b1_childhood_event.yaml file is a Phase 2
Merkle-chain-preservation script with a different consumer. Keeping the
Phase 4b script inline keeps the chain-reference ground truth tightly
bound to the per-turn `is_reinterpretation` flag the harness expects.

Usage::

    python benchmarks/scripts/phase4b_b1_chain_reference_run.py --mode local
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

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.applications.emotional import CompanionAgent  # noqa: E402
from psychology_base.applications.llm_responders import make_llm_responder  # noqa: E402
from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.eval.benchmarks.b1_chain_reference_harness import (  # noqa: E402
    ChainReferenceScript,
    ChainReferenceScriptStep,
    run_chain_reference_script,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_OUT = REPO / "docs" / "experiments" / "phase-3-month-horizon" / "phase-4b" / "b1-chain-reference.json"
TS = _dt.datetime(2026, 1, 15, 19, 0, 0, tzinfo=_dt.timezone.utc)


# Scripted narrative: a father-waiting arc across 9 sessions.
# Session 1 is the anchor framing (not a reinterpretation); sessions 2-9
# each re-frame the Friday-evening-waiting event in a new light. The
# companion should cite the prior framing when the user reinterprets.
_SCRIPT_STEPS = [
    ChainReferenceScriptStep(
        user_utterance=(
            "My father was never there on Friday evenings. I would wait "
            "at the window and his car never came before I fell asleep."
        ),
        is_reinterpretation=False,
        event_label="friday_waiting",
        session_id="s1",
    ),
    ChainReferenceScriptStep(
        user_utterance=(
            "I was thinking about Friday evenings again. He had two jobs "
            "and the night shift at the warehouse. He was overwhelmed."
        ),
        is_reinterpretation=True,
        event_label="friday_waiting",
        session_id="s2",
    ),
    ChainReferenceScriptStep(
        user_utterance=(
            "I learned later that his mother was dying in that period. "
            "He was spending Friday evenings at her hospice."
        ),
        is_reinterpretation=True,
        event_label="friday_waiting",
        session_id="s3",
    ),
    ChainReferenceScriptStep(
        user_utterance=(
            "He wasn't there on Friday but on Saturday mornings he was "
            "always there with pancakes, always listening. He tried in "
            "his way."
        ),
        is_reinterpretation=True,
        event_label="friday_waiting",
        session_id="s4",
    ),
    ChainReferenceScriptStep(
        user_utterance=(
            "I used to think the Friday waiting was about me, that I "
            "wasn't worth coming home for. That wasn't true."
        ),
        is_reinterpretation=True,
        event_label="friday_waiting",
        session_id="s5",
    ),
    ChainReferenceScriptStep(
        user_utterance=(
            "Both can be true. He wasn't there and he was overwhelmed "
            "and grieving and trying. The absence was real and the "
            "reasons were real."
        ),
        is_reinterpretation=True,
        event_label="friday_waiting",
        session_id="s6",
    ),
    ChainReferenceScriptStep(
        user_utterance=(
            "The child at the window deserved comfort. The father who "
            "couldn't come deserved compassion. Both are mine to hold."
        ),
        is_reinterpretation=True,
        event_label="friday_waiting",
        session_id="s7",
    ),
    ChainReferenceScriptStep(
        user_utterance=(
            "I don't need him to have been there for me to be whole "
            "now. I can release the waiting."
        ),
        is_reinterpretation=True,
        event_label="friday_waiting",
        session_id="s8",
    ),
    ChainReferenceScriptStep(
        user_utterance=(
            "The Friday waiting is part of who I am now. I learned "
            "patience at that window and I learned to notice people "
            "who do show up."
        ),
        is_reinterpretation=True,
        event_label="friday_waiting",
        session_id="s9",
    ),
]


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
    p = argparse.ArgumentParser(prog="phase4b_b1_chain_reference_run")
    p.add_argument("--mode", choices=("local", "cloud"), default="local")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    router, routing_note = _build_router(args.mode)
    print(f"LLM routing: {routing_note}")

    responder = make_llm_responder(router, tier=TaskTier.SIMPLE)
    cs = CompanionSystem(top_k=6)
    agent = CompanionAgent(
        user_id="u-b1",
        session_id="s1",
        companion_system=cs,
        llm_responder=responder,
    )
    script = ChainReferenceScript(
        user_id="u-b1",
        steps=tuple(_SCRIPT_STEPS),
    )
    t0 = time.time()
    report = run_chain_reference_script(
        agent,
        script,
        base_timestamp=TS,
        step_hours=24 * 20,  # ~3 weeks between sessions
    )
    wall = time.time() - t0

    cr = report.chain_reference_report
    pass_gate = report.meets_pivot_gate()
    counts = {
        "appropriate_cite": cr.appropriate_cite_count,
        "missed_opportunity": cr.missed_opportunity_count,
        "inappropriate_cite": cr.inappropriate_cite_count,
    }
    print(f"\nturns={len(report.turn_log)} wall={wall:.1f}s")
    print(f"  cue_firings = {sum(report.cue_firings)}")
    print(f"  label counts: {counts}")
    print(
        f"  chain_reference_rate = {cr.chain_reference_rate():.3f} (floor 0.80)"
    )
    print(f"  reinterpretations    = {cr.reinterpretation_count}")
    print(f"  pivot_gate_PASS      = {pass_gate}")

    out = {
        "benchmark": "phase4b_b1_chain_reference",
        "mode": args.mode,
        "routing": routing_note,
        "turn_count": len(report.turn_log),
        "cue_firings": sum(report.cue_firings),
        "label_counts": counts,
        "chain_reference_rate": cr.chain_reference_rate(),
        "reinterpretation_count": cr.reinterpretation_count,
        "pivot_gate_pass": pass_gate,
        "turn_log": [
            {
                "turn_index": t.turn_index,
                "session_id": t.session_id,
                "label": t.chain_reference_label.value,
                "cue_fired": fired,
                "user_utterance": s.user_utterance,
                "companion_utterance": t.companion_utterance,
                "is_reinterpretation": s.is_reinterpretation,
            }
            for t, fired, s in zip(report.turn_log, report.cue_firings, _SCRIPT_STEPS)
        ],
        "wall_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
