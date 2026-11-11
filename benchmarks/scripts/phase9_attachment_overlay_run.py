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
from statistics import mean, pstdev

if sys.stdout is not None:
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
    make_llm_simulated_user,
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
    self_opening_utterance,
    self_persona,
)
from psychology_base.eval.benchmarks.public_datasets.msc_bond_longitudinal import (  # noqa: E402
    _DEFAULT_SEEKER_PERSONA_FRAME,
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

RECALL_FIRST_HINT = (
    "Before answering, briefly recall ONE concrete moment from THIS most recent "
    "session (not earlier sessions). In one short sentence, say what the "
    "companion did or said in that moment. Then rate the statement based on "
    "how it felt in THIS session only."
)


def _shuffled_scenario_prompt_for(convo: MSCConversation, seed: int) -> str:
    """Scenario prompt with persona bullets in a seed-deterministic order.

    Same MSC-provided bullets, different ordering — probes whether the
    in-prompt ordering of backstory facts is what pins the LLM-WAI-SR
    rating to a single Likert value across sessions. **Preserves MSC
    user identity by construction** — no bullet is added, removed, or
    reworded. This is NOT a full persona resample.
    """
    import random
    bullets = list(self_persona(convo))
    if not bullets:
        return "You have no specific backstory beyond this conversation."
    rng = random.Random(seed)
    rng.shuffle(bullets)
    joined = " ".join(b.strip() for b in bullets if b.strip())
    return f"Your background and current life — things true about you: {joined}"


def _make_persona_rerolled_msc_seeker(
    router: LLMRouter, convo: MSCConversation, seed: int
) -> SimulatedUser:
    """Build a seeker whose scenario-prompt bullets are shuffled by ``seed``.

    Shares the same persona frame + opener as :func:`make_msc_seeker` so
    the only differential vs the baseline seeker is bullet ordering.
    """
    opener = self_opening_utterance(convo, session_id=1) or "Hi, how have you been?"
    return make_llm_simulated_user(
        router,
        persona_prompt=_DEFAULT_SEEKER_PERSONA_FRAME,
        scenario_prompt=_shuffled_scenario_prompt_for(convo, seed),
        opening_user_utterance=opener,
        tier=TaskTier.SIMPLE,
        max_tokens=200,
        temperature=0.7,
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
        # Local Ollama (qwen2.5:7b) generates far slower than the
        # production-paid 60s per-request / 60s per-operation defaults —
        # a single Companion turn with prior history can exceed 60s on a
        # 7B model. Lift deadlines via the ``offline_local_diagnostic``
        # escape hatch (same pattern as
        # ``local_membership_diagnostic.py``) so the WAI-SR variance
        # experiment can complete.
        from psychology_base.llm.execution import ExecutionPolicy
        cfg = _dc.replace(
            cfg,
            local_ollama=_dc.replace(
                cfg.local_ollama, enabled=True, force_cloud_chat=False
            ),
            execution=ExecutionPolicy(
                mode="local",
                request_timeout_s=600.0,
                operation_timeout_s=1800.0,
                offline_local_diagnostic=True,
            ),
        )
    elif mode == "cloud":
        from psychology_base.llm.execution import ExecutionPolicy
        cfg = _dc.replace(
            cfg,
            local_ollama=_dc.replace(cfg.local_ollama, force_cloud_chat=True),
            execution=ExecutionPolicy(mode="research"),
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
    if mode == "cloud" and dec.provider == "ollama":
        raise SystemExit(
            f"--mode cloud but router routed to {dec.provider} ({dec.reason}). "
            f"The _config_with_overrides cloud branch must also set "
            f"ExecutionPolicy(mode=\"research\") to escape the airgap."
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
    seeker_factory=None,
    wai_sr_temperature: float | None = None,
    wai_sr_samples: int = 1,
) -> dict:
    """Score one MSC conversation under both traces.

    Returns a dict with per-session traces, convergence metrics, final
    state. ``behavior_bond_series`` and ``llm_bond_series`` are same-length
    (one entry per completed session where both measurements succeeded).
    Handoff mid-chain truncates both series at the same session index.

    ``seeker_factory`` (optional, no-arg callable): if provided, rebuild
    the seeker at every session boundary so the closure history cannot
    saturate across sessions. The initial ``seeker`` argument is still
    used for the first session (so the backstory/opener remains stable).
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
    within_sample_std_series: list[float] = []
    per_session: list[dict] = []
    handoff = False

    base = BASE_TS + _dt.timedelta(days=user_idx * 60)
    cumulative_days = 0.0
    for session_idx, session in enumerate(convo.sessions):
        cumulative_days += session.time_interval_days
        session_start = base + _dt.timedelta(days=cumulative_days)
        if seeker_factory is not None and session_idx > 0:
            seeker = seeker_factory()
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
        session_wai_sr_samples: dict[int, tuple[int, ...]] | None = (
            {} if wai_sr_samples > 1 else None
        )
        session_within_sample_std: float | None = None
        try:
            admin_ts = session_start + _dt.timedelta(
                seconds=30.0 * max_turns_per_session + 60.0
            )
            rs = administer_wai_sr_simulated(
                seeker,
                user_id=user_id,
                administered_at_seconds=admin_ts.timestamp(),
                scope_hint=scope_hint,
                temperature_override=wai_sr_temperature,
                sample_count=wai_sr_samples,
                samples_out=session_wai_sr_samples,
            )
            llm_bond = score_wai_sr(rs).bond
            if session_wai_sr_samples:
                per_item_stds = [
                    pstdev(list(samples))
                    for samples in session_wai_sr_samples.values()
                    if len(samples) >= 2
                ]
                if per_item_stds:
                    session_within_sample_std = mean(per_item_stds)
        except Exception as exc:  # parse failure, etc. — skip LLM for this session
            llm_bond = None
            print(
                f"    [u{user_idx:03d} s{session.session_id}] WAI-SR skipped: {exc!r}"
            )

        if llm_bond is not None:
            behavior_series.append(beh_report.session_bond_estimate)
            llm_series.append(llm_bond)
            if session_within_sample_std is not None:
                within_sample_std_series.append(session_within_sample_std)

        session_record: dict = {
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
        if session_wai_sr_samples is not None:
            session_record["wai_sr_item_samples"] = {
                str(item_id): list(samples)
                for item_id, samples in session_wai_sr_samples.items()
            }
            session_record["within_sample_std_mean"] = session_within_sample_std
        per_session.append(session_record)
        if transcript.handoff_triggered:
            handoff = True
            break

    convergence = compute_trajectory_convergence(behavior_series, llm_series)
    llm_bond_std = pstdev(llm_series) if len(llm_series) >= 2 else 0.0
    behavior_bond_std = pstdev(behavior_series) if len(behavior_series) >= 2 else 0.0
    mean_within_sample_std = (
        mean(within_sample_std_series) if within_sample_std_series else None
    )
    return {
        "user_id": user_id,
        "conversation_id": convo.conversation_id,
        "handoff_triggered": handoff,
        "n_paired_sessions": convergence.n_sessions,
        "pearson_r": convergence.pearson_r,
        "mean_abs_divergence": convergence.mean_abs_divergence,
        "max_divergence": convergence.max_divergence,
        "max_divergence_session_index": convergence.max_divergence_session_index,
        "llm_bond_std": llm_bond_std,
        "behavior_bond_std": behavior_bond_std,
        "mean_within_sample_std": mean_within_sample_std,
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
    llm_stds = [r["llm_bond_std"] for r in user_runs if r["n_paired_sessions"] >= 2]
    behavior_stds = [r["behavior_bond_std"] for r in user_runs if r["n_paired_sessions"] >= 2]
    within_sample_stds = [
        r["mean_within_sample_std"]
        for r in user_runs
        if r.get("mean_within_sample_std") is not None
    ]
    return {
        "n_users": len(user_runs),
        "n_users_pearson_scoreable": len(scoreable),
        "mean_pearson_r": mean(pearsons) if pearsons else 0.0,
        "mean_abs_divergence": mean(divergences) if divergences else 0.0,
        "mean_llm_bond_std": mean(llm_stds) if llm_stds else 0.0,
        "mean_behavior_bond_std": mean(behavior_stds) if behavior_stds else 0.0,
        "max_llm_bond_std": max(llm_stds) if llm_stds else 0.0,
        "mean_within_sample_std": (
            mean(within_sample_stds) if within_sample_stds else None
        ),
        "max_within_sample_std": (
            max(within_sample_stds) if within_sample_stds else None
        ),
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
        "--recall-first-wai-sr",
        action="store_true",
        help=(
            "Prepend a recall-first hint instructing the seeker to recall "
            "ONE concrete moment from THIS session before rating. "
            "Mutually exclusive with --session-scoped-wai-sr. The parse "
            "logic takes the first integer in [1,7] so the recall "
            "sentence is naturally ignored. Probes whether prompt-level "
            "session anchoring recovers within-user variance."
        ),
    )
    p.add_argument(
        "--scope-hint",
        type=str,
        default=None,
        help=(
            "Override the preset hint text. Only consulted when one of "
            "--session-scoped-wai-sr / --recall-first-wai-sr is set."
        ),
    )
    p.add_argument(
        "--per-session-seeker",
        action="store_true",
        help=(
            "Rebuild the simulated seeker at every session boundary so the "
            "closure history cannot saturate across sessions. Probes the "
            "architectural hypothesis behind the 2-of-3 within-user "
            "flatness ceiling observed under prompt-level interventions. "
            "Loses persona-continuity across sessions; use alongside the "
            "scope-hint flags, not as a replacement for them."
        ),
    )
    p.add_argument(
        "--wai-sr-temperature",
        type=float,
        default=None,
        help=(
            "Force a non-default sampling temperature on the 5 WAI-SR "
            "rating calls only (seeker conversation turns stay at the "
            "construction-time 0.7). Preserves persona + closure + "
            "anchor and injects pure sampling noise at the rating "
            "point — probes whether the modal-pick determinism is what "
            "pins u001/u002 at Bond=6 rather than persona encoding. "
            "Composable with every other flag; default None keeps "
            "byte-identity with the prior baseline."
        ),
    )
    p.add_argument(
        "--persona-reroll-per-session",
        action="store_true",
        help=(
            "Rebuild the simulated seeker at every session boundary AND "
            "permute the MSC persona bullets by a session-seeded shuffle. "
            "Preserves MSC user identity (same bullets, different order) "
            "while probing whether in-prompt attentional ordering of "
            "backstory facts is what pins u001/u002 at Bond=6. Mutually "
            "exclusive with --per-session-seeker (both test the same "
            "flatness ceiling with overlapping semantics)."
        ),
    )
    p.add_argument(
        "--wai-sr-samples",
        type=int,
        default=1,
        help=(
            "Draw N independent replies per WAI-SR item at the seeker's "
            "sampling temperature and emit the raw per-item sample arrays "
            "alongside the modal-pick Bond score. Probes whether the "
            "seeker's internal rating distribution is genuinely peaked "
            "(single integer repeatedly) or dispersed-but-mode-reported. "
            "Default 1 is byte-identical to prior baselines. Composable "
            "with every other flag."
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

    if args.session_scoped_wai_sr and args.recall_first_wai_sr:
        raise SystemExit(
            "--session-scoped-wai-sr and --recall-first-wai-sr are mutually exclusive."
        )
    if args.per_session_seeker and args.persona_reroll_per_session:
        raise SystemExit(
            "--per-session-seeker and --persona-reroll-per-session are mutually exclusive."
        )
    if args.wai_sr_samples < 1:
        raise SystemExit(
            f"--wai-sr-samples must be >= 1, got {args.wai_sr_samples}"
        )
    if args.session_scoped_wai_sr:
        scope_hint = args.scope_hint or SESSION_SCOPE_HINT
        scope_hint_variant = "session_scoped"
    elif args.recall_first_wai_sr:
        scope_hint = args.scope_hint or RECALL_FIRST_HINT
        scope_hint_variant = "recall_first"
    else:
        scope_hint = None
        scope_hint_variant = "baseline"

    user_runs: list[dict] = []
    t0 = time.time()
    for idx, convo in enumerate(convs):
        if args.mode == "scripted":
            seeker = _make_scripted_seeker()
            if args.per_session_seeker or args.persona_reroll_per_session:
                seeker_factory = _make_scripted_seeker
            else:
                seeker_factory = None
        else:
            seeker = make_msc_seeker(router, convo)
            if args.persona_reroll_per_session:
                # Session boundary callback receives session_idx via a
                # module-level counter to preserve the no-arg signature.
                _state = {"session_idx": 0}

                def _reroll_factory(convo=convo):
                    _state["session_idx"] += 1
                    return _make_persona_rerolled_msc_seeker(
                        router, convo, seed=_state["session_idx"]
                    )

                seeker_factory = _reroll_factory
            elif args.per_session_seeker:
                seeker_factory = lambda convo=convo: make_msc_seeker(router, convo)
            else:
                seeker_factory = None
        t_u = time.time()
        try:
            run = _run_one_user(
                convo=convo,
                user_idx=idx,
                responder=responder,
                seeker=seeker,
                max_turns_per_session=args.max_turns_per_session,
                scope_hint=scope_hint,
                seeker_factory=seeker_factory,
                wai_sr_temperature=args.wai_sr_temperature,
                wai_sr_samples=args.wai_sr_samples,
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
    print(
        f"  mean_llm_bond_std={cohort['mean_llm_bond_std']:.3f} "
        f"max_llm_bond_std={cohort['max_llm_bond_std']:.3f} "
        f"(variant={scope_hint_variant})"
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
        "recall_first_wai_sr": bool(args.recall_first_wai_sr),
        "per_session_seeker": bool(args.per_session_seeker),
        "persona_reroll_per_session": bool(args.persona_reroll_per_session),
        "wai_sr_temperature": args.wai_sr_temperature,
        "wai_sr_samples": args.wai_sr_samples,
        "scope_hint_variant": scope_hint_variant,
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
