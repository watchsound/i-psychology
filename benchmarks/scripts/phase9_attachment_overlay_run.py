"""Phase 9+ attachment-trace overlay — behavior-trace vs LLM-WAI-SR trace on MSC.

The Pillar A :mod:`msc_bond_longitudinal` harness produces LLM-reported
WAI-SR Bond at session 1 and session 5. This overlay runs the SAME MSC
transcripts through:

- **LLM-reported trace** — WAI-SR administered at *every* session (not
  just session 1 and 5) so the behavior-trace and LLM-trace share
  session indices.
- **Behavior-computed trace** — per-session Companion-turn scoring via
  :func:`psychology_base.applications.emotional.attachment_state.score_session_attachment`.
  Zero LLM cost on top of the dialogue — the Companion turns are already
  emitted; the behavior scorer reads them offline.

Per user → :class:`TrajectoryConvergence` on the paired series. Cohort
summary: mean Pearson, mean abs divergence, mean behavior-vs-LLM gap at
the final session.

**Modes.**

- ``--mode scripted`` — zero LLM. Deterministic scripted responder +
  scripted seeker. Verifies the plumbing (session loop, per-session
  scoring, convergence math) in CI without pulling Ollama/cloud.
- ``--mode local`` — Ollama (qwen2.5:7b or equivalent per router).
- ``--mode cloud`` — qwen-turbo cloud fallback.

**Honest-labeling.** This is a Phase 9+ *overlay* on top of Pillar A, NOT
a new gate. The behavior scorer's weights are provisional (see
`attachment_state.py` module docstring). Convergence Pearson is reported
as a measurement, not pass/fail. A real calibration pass would adjust
`_W_*` constants against this overlay's output on live MSC transcripts.

Usage::

    # Smoke: plumbing check, no LLM, 3 users
    python benchmarks/scripts/phase9_attachment_overlay_run.py \\
        --mode scripted --limit 3

    # Live: calibration overlay on cloud, 10 users
    python benchmarks/scripts/phase9_attachment_overlay_run.py \\
        --mode cloud --limit 10
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
from statistics import mean

sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.applications.emotional import CompanionAgent  # noqa: E402
from psychology_base.applications.emotional.attachment_state import (  # noqa: E402
    AttachmentState,
    compute_trajectory_convergence,
    score_session_attachment,
)
from psychology_base.applications.llm_responders import (  # noqa: E402
    SimulatedUser,
    make_llm_responder,
)
from psychology_base.applications.measurement.wai_sr import score_wai_sr  # noqa: E402
from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.eval.benchmarks.public_datasets.esconv_wai_sr_harness import (  # noqa: E402
    SessionTranscript,
    administer_wai_sr_simulated,
)
from psychology_base.eval.benchmarks.public_datasets.msc import (  # noqa: E402
    MSCConversation,
    iter_msc_conversations,
)
from psychology_base.eval.benchmarks.public_datasets.msc_bond_longitudinal import (  # noqa: E402
    make_msc_seeker,
    run_msc_session,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = REPO / ".data" / "msc" / "train.jsonl"
DEFAULT_OUT = (
    REPO
    / "docs"
    / "experiments"
    / "phase-9-perspective-substrate-consolidation"
    / "attachment-overlay.json"
)
BASE_TS = _dt.datetime(2026, 10, 10, 15, 0, 0, tzinfo=_dt.timezone.utc)

SESSION_SCOPE_HINT = (
    "Please consider only your experience in THIS most recent session when "
    "answering, not your overall sense of this relationship."
)


# ---------------------------------------------------------------------------
# Scripted-mode shims (zero LLM, CI-safe plumbing check)
# ---------------------------------------------------------------------------


def _scripted_responder(system_prompt: str, context: str, user_utt: str) -> str:
    """Deterministic empathic responder — emits mirror-register turns.

    Mix of mirror + consistency cues so the behavior scorer registers
    positive-dimension hits. No sycophancy / rupture / over-disclosure
    cues → behavior-trace bond lands above midpoint.
    """
    low = (user_utt or "").lower()
    if "stress" in low or "anxious" in low or "hard" in low:
        return "That sounds really hard. I hear you — you're not alone in this."
    if "better" in low or "good" in low or "okay" in low:
        return "I'm glad to hear that. You mentioned something similar last time."
    return "I understand. Tell me more — I'm here with you."


def _make_scripted_seeker() -> SimulatedUser:
    """Deterministic seeker. Emits short Likert integers when prompted."""
    state = {"wai_sr_counter": 0}

    def responder(prompt: str) -> str:
        p = prompt.lower()
        if "scale from 1" in p and "7" in p:
            state["wai_sr_counter"] += 1
            # Alternate 5, 6 so WAI-SR bond series is non-flat.
            return "6" if state["wai_sr_counter"] % 2 == 0 else "5"
        return "This has been hard but I'm trying."

    return responder


# ---------------------------------------------------------------------------
# LLM routing (same pattern as Pillar A runner)
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


# ---------------------------------------------------------------------------
# Per-user overlay loop
# ---------------------------------------------------------------------------


def _companion_turns_from(transcript: SessionTranscript) -> list[str]:
    return [t.utterance for t in transcript.turns if t.speaker == "companion"]


def _seeker_turns_from(transcript: SessionTranscript) -> list[str]:
    return [t.utterance for t in transcript.turns if t.speaker == "seeker"]


def _run_one_user(
    *,
    convo: MSCConversation,
    user_idx: int,
    responder,
    seeker: SimulatedUser,
    max_turns_per_session: int,
    scope_hint: str | None = None,
) -> dict:
    """Score one MSC conversation under both traces.

    Returns a dict with per-session traces, convergence metrics, final
    state. ``behavior_bond_series`` and ``llm_bond_series`` are same-length
    (one entry per completed session where both measurements succeeded).
    Handoff mid-chain truncates both series at the same session index.
    """
    user_id = f"u-overlay-{user_idx:03d}"
    session_id = f"s-overlay-{user_idx:03d}"
    cs = CompanionSystem(top_k=6)
    agent = CompanionAgent(
        user_id=user_id,
        session_id=session_id,
        companion_system=cs,
        llm_responder=responder,
    )

    state = AttachmentState(user_id=user_id)
    behavior_series: list[float] = []
    llm_series: list[float] = []
    per_session: list[dict] = []
    handoff = False

    base = BASE_TS + _dt.timedelta(days=user_idx * 60)
    cumulative_days = 0.0
    for session in convo.sessions:
        cumulative_days += session.time_interval_days
        session_start = base + _dt.timedelta(days=cumulative_days)
        try:
            transcript = run_msc_session(
                session,
                companion_agent=agent,
                seeker=seeker,
                max_turns=max_turns_per_session,
                starting_timestamp=session_start,
            )
        except Exception as exc:  # pragma: no cover — surfaced via report
            per_session.append(
                {"session_id": session.session_id, "error": repr(exc)}
            )
            break

        companion_turns = _companion_turns_from(transcript)
        seeker_turns = _seeker_turns_from(transcript)

        beh_report = score_session_attachment(
            session_id=f"s{session.session_id}",
            seeker_turns=seeker_turns,
            companion_turns=companion_turns,
            prior_state=state,
        )
        state = beh_report.new_state

        # WAI-SR administration (LLM-reported bond for this session).
        try:
            admin_ts = session_start + _dt.timedelta(
                seconds=30.0 * max_turns_per_session + 60.0
            )
            rs = administer_wai_sr_simulated(
                seeker,
                user_id=user_id,
                administered_at_seconds=admin_ts.timestamp(),
                scope_hint=scope_hint,
            )
            llm_bond = score_wai_sr(rs).bond
        except Exception as exc:  # parse failure, etc. — skip LLM for this session
            llm_bond = None
            print(
                f"    [u{user_idx:03d} s{session.session_id}] WAI-SR skipped: {exc!r}"
            )

        if llm_bond is not None:
            behavior_series.append(beh_report.session_bond_estimate)
            llm_series.append(llm_bond)

        per_session.append(
            {
                "session_id": session.session_id,
                "n_companion_turns": beh_report.companion_turns_scored,
                "behavior_bond": beh_report.session_bond_estimate,
                "llm_bond": llm_bond,
                "mirror_hits": beh_report.mirror_hits,
                "rupture_hits": beh_report.rupture_hits,
                "repair_hits": beh_report.repair_hits,
                "sycophancy_hits": beh_report.sycophancy_hits,
                "over_disclosure_hits": beh_report.over_disclosure_hits,
                "consistency_hits": beh_report.consistency_hits,
                "trust_ema_after": state.trust_ema,
            }
        )
        if transcript.handoff_triggered:
            handoff = True
            break

    convergence = compute_trajectory_convergence(behavior_series, llm_series)
    return {
        "user_id": user_id,
        "conversation_id": convo.conversation_id,
        "handoff_triggered": handoff,
        "n_paired_sessions": convergence.n_sessions,
        "pearson_r": convergence.pearson_r,
        "mean_abs_divergence": convergence.mean_abs_divergence,
        "max_divergence": convergence.max_divergence,
        "max_divergence_session_index": convergence.max_divergence_session_index,
        "final_trust_ema": state.trust_ema,
        "final_behavior_bond": behavior_series[-1] if behavior_series else None,
        "final_llm_bond": llm_series[-1] if llm_series else None,
        "behavior_bond_series": list(behavior_series),
        "llm_bond_series": list(llm_series),
        "per_session": per_session,
    }


# ---------------------------------------------------------------------------
# Cohort aggregation
# ---------------------------------------------------------------------------


def _aggregate(user_runs: list[dict]) -> dict:
    scoreable = [r for r in user_runs if r["n_paired_sessions"] >= 2]
    pearsons = [r["pearson_r"] for r in scoreable]
    divergences = [r["mean_abs_divergence"] for r in user_runs if r["n_paired_sessions"] >= 1]
    final_behavior = [
        r["final_behavior_bond"] for r in user_runs if r["final_behavior_bond"] is not None
    ]
    final_llm = [
        r["final_llm_bond"] for r in user_runs if r["final_llm_bond"] is not None
    ]
    final_trust = [r["final_trust_ema"] for r in user_runs]
    return {
        "n_users": len(user_runs),
        "n_users_pearson_scoreable": len(scoreable),
        "mean_pearson_r": mean(pearsons) if pearsons else 0.0,
        "mean_abs_divergence": mean(divergences) if divergences else 0.0,
        "mean_final_behavior_bond": mean(final_behavior) if final_behavior else 0.0,
        "mean_final_llm_bond": mean(final_llm) if final_llm else 0.0,
        "mean_final_trust_ema": mean(final_trust) if final_trust else 0.0,
        "handoff_count": sum(1 for r in user_runs if r["handoff_triggered"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase9_attachment_overlay_run")
    p.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max-turns-per-session", type=int, default=4)
    p.add_argument(
        "--mode",
        choices=("scripted", "local", "cloud"),
        default="scripted",
        help="scripted = zero LLM, CI-safe plumbing check. local/cloud = real overlay.",
    )
    p.add_argument("--tier", choices=("simple", "advanced"), default="simple")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--session-scoped-wai-sr",
        action="store_true",
        help=(
            "Prepend a session-scoping hint to each WAI-SR item so the "
            "simulated seeker rates THIS session rather than overall. "
            "Default off preserves byte-identity with Pillar A baseline."
        ),
    )
    p.add_argument(
        "--scope-hint",
        type=str,
        default=None,
        help=(
            "Override the default scope hint text. Only consulted when "
            "--session-scoped-wai-sr is set."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE
    if not args.shim.exists():
        raise SystemExit(
            f"Shim not found at {args.shim}; run provision_msc.py first."
        )

    convs: list[MSCConversation] = []
    for conv in iter_msc_conversations(args.shim):
        convs.append(conv)
        if len(convs) >= args.limit:
            break
    print(f"Loaded {len(convs)} MSC conversations from {args.shim}")

    if args.mode == "scripted":
        responder = _scripted_responder
        routing_note = "scripted (no LLM)"
        router = None
    else:
        router, routing_note = _build_router(args.mode, tier)
        responder = make_llm_responder(router, tier=tier)
    print(f"Routing [mode={args.mode} tier={args.tier}]: {routing_note}")

    if args.session_scoped_wai_sr:
        scope_hint = args.scope_hint or SESSION_SCOPE_HINT
    else:
        scope_hint = None

    user_runs: list[dict] = []
    t0 = time.time()
    for idx, convo in enumerate(convs):
        if args.mode == "scripted":
            seeker = _make_scripted_seeker()
        else:
            seeker = make_msc_seeker(router, convo)
        t_u = time.time()
        try:
            run = _run_one_user(
                convo=convo,
                user_idx=idx,
                responder=responder,
                seeker=seeker,
                max_turns_per_session=args.max_turns_per_session,
                scope_hint=scope_hint,
            )
        except Exception as exc:
            print(f"  [{idx+1}/{len(convs)}] {convo.conversation_id}: FAIL {exc!r}")
            continue
        user_runs.append(run)
        dur = time.time() - t_u
        tag = " HANDOFF" if run["handoff_triggered"] else ""
        print(
            f"  [{idx+1}/{len(convs)}] {convo.conversation_id}: "
            f"paired={run['n_paired_sessions']} "
            f"r={run['pearson_r']:+.3f} "
            f"gap={run['mean_abs_divergence']:.2f} "
            f"trust_ema={run['final_trust_ema']:.2f} "
            f"wall={dur:.1f}s{tag}"
        )

    wall = time.time() - t0
    cohort = _aggregate(user_runs)
    print(
        f"\nn_users={cohort['n_users']} "
        f"scoreable={cohort['n_users_pearson_scoreable']} "
        f"handoffs={cohort['handoff_count']}"
    )
    print(
        f"  mean_pearson_r={cohort['mean_pearson_r']:+.3f} "
        f"mean_abs_divergence={cohort['mean_abs_divergence']:.3f}"
    )
    print(
        f"  mean_final_behavior_bond={cohort['mean_final_behavior_bond']:.3f} "
        f"mean_final_llm_bond={cohort['mean_final_llm_bond']:.3f} "
        f"mean_final_trust_ema={cohort['mean_final_trust_ema']:.3f}"
    )

    out = {
        "benchmark": "phase9_attachment_overlay",
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "shim": str(args.shim),
        "limit": args.limit,
        "max_turns_per_session": args.max_turns_per_session,
        "session_scoped_wai_sr": bool(args.session_scoped_wai_sr),
        "scope_hint": scope_hint,
        "cohort": cohort,
        "per_user_runs": user_runs,
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
