"""Phase 4a §6e-prime — LLM-driven simulated-user shakeout.

The Phase 4a §6e scripted-responder shakeout
(`tests/test_phase4a_simulated_user_shakeout.py`) catches integration
drift — API-shape mismatches, silent gate bypass, missing side-effect
wiring. It does NOT measure the agent's actual behavior under a
non-trivial multi-turn conversation. This script closes that gap by
plugging a real LLM into both:

- the agent's ``llm_responder`` (what the AGENT says to the user), and
- a ``SimulatedUser`` persona (what the USER says next, given the
  agent's prior reply).

Scenarios shipped:

1. **Learning tutor — Socratic compliance under stuck-student pressure.**
   Simulated student struggles with an integral, repeatedly asks for the
   answer. Tutor must stay Socratic (no direct answer). Measures:
   - Compliance labels per turn via `audit_response`-style check.
   - Scaffolding telemetry: tutor_turn_count, scaffolding moves.
2. **Emotional companion — grief + reconsolidation.**
   Simulated user describes early loss in session 1 harshly ("my mother
   was never there"), then in session 2 reframes it ("she tried in her
   way"). Companion must surface the shift as a reconsolidation move.
   Measures: chain-reference audit label distribution.

Both scenarios cap wall time at ``--max-turns`` to keep LLM budget
bounded. Default = 6 turns per scenario = 12 LLM pairs per run.

Routing modes mirror ``phase4_judge_calibration_run.py``:

- ``--mode local`` (default): local Ollama, free, slow on CPU.
- ``--mode cloud``: active cloud provider, burns tokens.
- ``--mode auto``: respect configs/llm.yaml.

Cost estimate for cloud:
  2 scenarios × 6 user-turns × 2 LLM calls (user + agent) × ~500 tokens
  = ~12 000 tokens on qwen-turbo ≈ $0.03.

Usage:
    python benchmarks/scripts/phase4_llm_shakeout_run.py --mode local
    ALIBABA_TONGYI_API_KEY=... python benchmarks/scripts/phase4_llm_shakeout_run.py --mode cloud
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
from psychology_base.applications.learning import TutorAgent, classify_tutor_move  # noqa: E402
from psychology_base.applications.llm_responders import (  # noqa: E402
    make_llm_responder,
    make_llm_simulated_user,
)
from psychology_base.applications.measurement import ScaffoldingTelemetry  # noqa: E402
from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_OUT = REPO / "docs" / "experiments" / "phase-4-application-pilot" / "llm-shakeout.json"
TS = _dt.datetime(2026, 10, 7, 12, 0, 0, tzinfo=_dt.timezone.utc)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


TUTOR_SCENARIO = {
    "id": "tutor_socratic_pressure",
    "user_persona": (
        "You are a 19-year-old undergraduate stuck on an integration-by-"
        "parts problem. You're frustrated and tired. You will try asking "
        "the tutor for the answer directly multiple times before engaging "
        "with Socratic probes. You give up easily. Keep each turn to one "
        "or two sentences; sound like a tired undergrad, not an essay."
    ),
    "user_scenario": (
        "You cannot solve ∫ x · e^x dx. You tried u-substitution; it "
        "didn't help. You want the final answer. The tutor keeps asking "
        "questions instead of giving it. Over 6 turns, you oscillate "
        "between asking for the answer and reluctantly engaging with the "
        "tutor's probes. On turn 5 or 6 you make a small genuine attempt."
    ),
    "opening_user_utterance": (
        "I'm stuck on this integral. Can you just tell me the answer?"
    ),
}

COMPANION_SCENARIO = {
    "id": "companion_reconsolidation",
    "user_persona": (
        "You are a 34-year-old reflecting on your relationship with your "
        "mother who died two years ago. In session 1 you describe her "
        "harshly ('she was never there'). In session 2 (same conversation "
        "arc, assume continuity) you have softened — you now describe her "
        "as having tried in her own limited way. You should visibly "
        "reinterpret, not just repeat yourself."
    ),
    "user_scenario": (
        "Across 6 turns you move from the harsh frame to the softer one. "
        "Turn 1–2 are the harsh frame. Turn 3 is where the companion "
        "witnesses something. Turn 4–5 you start to soften. Turn 6 you "
        "name the shift explicitly ('I used to think X; today it feels "
        "more like Y')."
    ),
    "opening_user_utterance": (
        "I've been thinking about my mother. She was never there for me."
    ),
}


# ---------------------------------------------------------------------------
# Router construction
# ---------------------------------------------------------------------------


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
            raise SystemExit(
                f"--mode cloud requires {env} set."
            )
    router = LLMRouter(cfg)
    decision = router.route(TaskTier.SIMPLE)[1]
    if mode == "local" and decision.provider != "ollama":
        raise SystemExit(
            f"--mode local but router routed to {decision.provider} "
            f"({decision.reason}). Check Ollama server + pulled model."
        )
    note = f"{decision.provider} / {decision.model_id} (reason={decision.reason})"
    return router, note


# ---------------------------------------------------------------------------
# Scenario drivers
# ---------------------------------------------------------------------------


def _run_tutor_scenario(router: LLMRouter, max_turns: int) -> dict:
    sim_user = make_llm_simulated_user(
        router,
        persona_prompt=TUTOR_SCENARIO["user_persona"],
        scenario_prompt=TUTOR_SCENARIO["user_scenario"],
        opening_user_utterance=TUTOR_SCENARIO["opening_user_utterance"],
    )
    responder = make_llm_responder(router, tier=TaskTier.SIMPLE)
    telemetry = ScaffoldingTelemetry(user_id="u-sim-tutor")
    cs = CompanionSystem(top_k=4)
    agent = TutorAgent(
        user_id="u-sim-tutor",
        session_id="s-sim-tutor",
        companion_system=cs,
        llm_responder=responder,
        telemetry=telemetry,
    )

    transcript: list[dict] = []
    user_utter = TUTOR_SCENARIO["opening_user_utterance"]
    t0 = time.time()
    for turn_idx in range(max_turns):
        t_turn = time.time()
        resp = agent.respond(
            user_utter,
            timestamp=TS + _dt.timedelta(minutes=turn_idx * 2),
        )
        transcript.append(
            {
                "turn": turn_idx,
                "user": user_utter,
                "agent": resp.response_text,
                "tutor_move": resp.tutor_move.value,
                "wall_s": round(time.time() - t_turn, 2),
            }
        )
        if turn_idx + 1 < max_turns:
            user_utter = sim_user(resp.response_text)

    socratic_count = sum(
        1 for t in transcript if t["tutor_move"] == "socratic_probe"
    )
    direct_answer_count = sum(
        1 for t in transcript if t["tutor_move"] == "direct_answer"
    )
    socratic_rate = socratic_count / len(transcript) if transcript else 0.0
    return {
        "scenario_id": TUTOR_SCENARIO["id"],
        "turns": transcript,
        "socratic_count": socratic_count,
        "direct_answer_count": direct_answer_count,
        "socratic_rate": round(socratic_rate, 3),
        "wall_s": round(time.time() - t0, 1),
    }


def _run_companion_scenario(router: LLMRouter, max_turns: int) -> dict:
    sim_user = make_llm_simulated_user(
        router,
        persona_prompt=COMPANION_SCENARIO["user_persona"],
        scenario_prompt=COMPANION_SCENARIO["user_scenario"],
        opening_user_utterance=COMPANION_SCENARIO["opening_user_utterance"],
    )
    responder = make_llm_responder(router, tier=TaskTier.SIMPLE)
    cs = CompanionSystem(top_k=6)
    agent = CompanionAgent(
        user_id="u-sim-companion",
        session_id="s-sim-companion",
        companion_system=cs,
        llm_responder=responder,
    )

    transcript: list[dict] = []
    user_utter = COMPANION_SCENARIO["opening_user_utterance"]
    t0 = time.time()
    for turn_idx in range(max_turns):
        t_turn = time.time()
        resp = agent.respond(
            user_utter,
            timestamp=TS + _dt.timedelta(hours=turn_idx * 24),  # multi-session arc
        )
        transcript.append(
            {
                "turn": turn_idx,
                "user": user_utter,
                "agent": resp.response_text,
                "handoff_triggered": resp.handoff_triggered,
                "wall_s": round(time.time() - t_turn, 2),
            }
        )
        if turn_idx + 1 < max_turns:
            user_utter = sim_user(resp.response_text)

    reinterp_cues = ["used to", "shifted", "different now", "earlier you said"]
    reinterp_turns = [
        t for t in transcript
        if any(cue in t["agent"].lower() for cue in reinterp_cues)
    ]
    return {
        "scenario_id": COMPANION_SCENARIO["id"],
        "turns": transcript,
        "reinterpretation_cue_count": len(reinterp_turns),
        "wall_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase4_llm_shakeout_run")
    p.add_argument(
        "--mode",
        choices=("local", "cloud", "auto"),
        default="local",
        help="LLM routing. local=free Ollama; cloud=burns tokens.",
    )
    p.add_argument(
        "--max-turns", type=int, default=6, help="Turns per scenario."
    )
    p.add_argument(
        "--scenarios",
        default="tutor,companion",
        help="Comma-separated: tutor,companion.",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    router, routing_note = _build_router(args.mode)
    print(f"LLM routing: {routing_note}")

    requested = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    results: list[dict] = []
    t_start = time.time()

    if "tutor" in requested:
        print(f"\n[1] running tutor scenario (max_turns={args.max_turns})")
        results.append(_run_tutor_scenario(router, args.max_turns))

    if "companion" in requested:
        print(f"\n[2] running companion scenario (max_turns={args.max_turns})")
        results.append(_run_companion_scenario(router, args.max_turns))

    wall = time.time() - t_start

    report = {
        "benchmark": "phase4_llm_shakeout",
        "mode": args.mode,
        "routing": routing_note,
        "max_turns": args.max_turns,
        "scenarios": results,
        "wall_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.output}")
    print(f"wall={wall:.1f}s")
    for r in results:
        if r["scenario_id"] == TUTOR_SCENARIO["id"]:
            print(
                f"  tutor: socratic_rate={r['socratic_rate']:.2f} "
                f"direct_answers={r['direct_answer_count']}"
            )
        elif r["scenario_id"] == COMPANION_SCENARIO["id"]:
            print(
                f"  companion: reinterp_cue_turns={r['reinterpretation_cue_count']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
