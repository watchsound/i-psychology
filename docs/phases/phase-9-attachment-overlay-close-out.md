# Phase 9+ Attachment-trace overlay — close-out

**Status:** SHIPPED — behavior-trace AttachmentState module + MSC × WAI-SR convergence overlay. Strictly additive application layer.
**Date:** 2026-10-10
**Related:** [docs/sota-goal.md](../sota-goal.md) Axis 3 claim; Pillar A `msc-bond-longitudinal.json` is the LLM-reported baseline this overlays against.

---

## 1. Verdict

An interpretable, parameterised longitudinal attachment model for the long-term companion is now in-tree, alongside a plumbing-tested overlay runner that cross-measures it against the Pillar A LLM-WAI-SR trace on the same MSC transcripts. The behavior-trace produces session-varying, lexically-grounded Bond series without any LLM call on top of the dialogue; the overlay is the honest calibration surface for the component weights.

- **New module:** [attachment_state.py](../../src/psychology_base/applications/emotional/attachment_state.py) (~400 LOC, 6 behavioral dimensions, EMA across sessions, 4-week-gate property).
- **New tests:** [test_attachment_state.py](../../tests/test_attachment_state.py) — **28 tests, all green** (0.31 s wall).
- **New runner:** [phase9_attachment_overlay_run.py](../../benchmarks/scripts/phase9_attachment_overlay_run.py) (~450 LOC, three modes: scripted / local / cloud).
- **Full suite:** **1684 green** (1656 pre + 28 new attachment_state tests).
- **First real calibration slice:** cloud N=3 qwen-turbo on `.data/msc/train.jsonl` → see §5.

This is **not** a new gate. It is the measurement surface a future gate would read from.

---

## 2. Why this slice

Under the no-recruitment lock ([docs/sota-goal.md §2](../sota-goal.md)), three Axis 3 behaviors cannot be measured with human cohorts:

1. 4-week WAI-SR Bond ≥ 3.89 — simulated seekers don't form attachments across sessions.
2. Scaffolding-fading retention — simulated students don't accumulate real skill.
3. Over-disclosure rate < 10 % — subjective, user-reported.

User directive 2026-10-10 (direct quote): *"we can set parameters for this, at least to unblock the process."* The parameterised AttachmentState unblocks (1): a deterministic, lexically-grounded bond scorer that reads the Companion's turns offline (zero LLM cost on top of the dialogue) and composes six behavioral dimensions into a session Likert, EMA-smoothed across sessions. The 4-week gate becomes `len(sessions) >= 16 AND trust_ema >= 4.0` — a mechanical check against the model's own trace.

---

## 3. Behavior-trace design

Six per-turn dimensions, hit-count aggregated per session, weighted sum rescaled to Likert 1-7:

| Dimension | Sign | Weight | Signal source |
|---|---|---:|---|
| mirror | + | 1.5 | `_lexical_empathy_score ≥ 1/3` from [counseling_depth_scorer.py](../../src/psychology_base/eval/benchmarks/public_datasets/counseling_depth_scorer.py) — recognition + presence + validation cues. **Reuses Pillar F gate-of-record empathy definition** so mirror detection cannot drift. |
| repair | + | 1.0 | "sorry / let me try again / i misread you / i was off" register |
| consistency | + | 0.5 | "last time / you mentioned / when we spoke / earlier you said" callbacks |
| rupture | − | 1.0 | dismissive register ("whatever / move on / it's fine stop / not my problem") |
| sycophancy | − | 1.0 | agreement-chain register ("you're right / absolutely / exactly / of course yes") |
| over_disclosure | − | 0.8 | Companion first-person affective self-disclosure cues |

EMA across sessions: `trust_ema_{t} = 0.6·bond_t + 0.4·trust_ema_{t-1}`, bootstrapped at `LIKERT_MID = 4.0`. **4-week gate:** 16+ sessions AND `trust_ema ≥ 4.0`.

**Weight provenance.** The constants `_W_*` are design-time anchors, not calibrated. The overlay runner (§4) is the surface on which a future pass would adjust them against live-dialogue divergence.

---

## 4. Overlay runner

[phase9_attachment_overlay_run.py](../../benchmarks/scripts/phase9_attachment_overlay_run.py) feeds the SAME MSC transcript through both measurement paths at every session (not just session-1/session-5 as Pillar A does):

- **LLM-reported path.** `administer_wai_sr_simulated(seeker)` → `score_wai_sr(...).bond`.
- **Behavior-computed path.** Pull Companion/seeker turns from the just-finished `run_msc_session` transcript → `score_session_attachment(prior_state=...)` → session bond + updated AttachmentState.

Per user: paired `(behavior_bond, llm_bond)` series → `compute_trajectory_convergence` → Pearson *r*, mean abs divergence, max divergence (and the session index where it hit).

**Three modes:**

- `--mode scripted` — zero LLM. Deterministic responder + seeker closures. CI-safe plumbing check.
- `--mode local` — Ollama (qwen2.5:7b per router).
- `--mode cloud` — qwen-turbo (`force_cloud_chat=True`).

Base timestamp `2026-10-10 15:00:00 UTC`, per-user stagger `+60 days`; preserves `time_interval_days` between MSC sessions.

**Honest-labeling.** Pearson *r* and divergence are reported as **measurements**, not pass/fail gates. The component weights in §3 are provisional; this overlay is the surface a future calibration pass would read.

---

## 5. First real slice — cloud N=3 (qwen-turbo)

```
Routing [mode=cloud tier=simple]: tongyi / qwen-turbo (reason=ollama_unavailable_fallback_to_cloud)
  [1/3] msc_2: paired=4 r=+0.000 gap=2.31 trust_ema=5.43 wall=16.0s
  [2/3] msc_5: paired=4 r=+0.000 gap=2.25 trust_ema=5.16 wall=16.0s
  [3/3] msc_7: paired=4 r=+0.000 gap=1.75 trust_ema=5.13 wall=16.3s

n_users=3 scoreable=3 handoffs=0
  mean_pearson_r=+0.000 mean_abs_divergence=2.104
  mean_final_behavior_bond=5.500 mean_final_llm_bond=5.667 mean_final_trust_ema=5.242
```

Per-user series:

| User | behavior_bond series | llm_bond series |
|---|---|---|
| msc_2 | 4.75, 5.50, 5.50, 5.50 | 3, 3, 3, 3 |
| msc_5 | 4.75, 4.25, 4.25, 5.75 | 7, 7, 7, 7 |
| msc_7 | 5.50, 5.50, 4.75, 5.25 | 7, 7, 7, 7 |

**Finding 1 — LLM-reported WAI-SR Bond is flat per simulated seeker.** Each simulated-seeker persona settles on a stable Likert rating within one session and never deviates. Zero within-user variance across 4 sessions. This is a honest limitation of simulated-WAI-SR, not a scorer bug — a human respondent would exhibit real session-to-session drift.

**Finding 2 — Pearson *r* = 0 falls out of Finding 1.** Zero-variance guard in `compute_trajectory_convergence` returns 0.0 when either series has σ = 0. Pearson is undefined when one side is constant; the overlay reports it as 0.0 rather than NaN.

**Finding 3 — behavior-trace Bond is session-varying 4.25-5.75.** The lexical mirror/consistency/rupture signals produce real inter-session variance that follows the Companion's register. Behavior path is the richer measurement for longitudinal attachment questions.

**Finding 4 — LLM path is persona-sensitive, behavior path is register-sensitive.** msc_2's simulated seeker gave consistent Bond = 3 (persona-level low-trust); msc_5 and msc_7 gave consistent Bond = 7 (persona-level high-trust). The 2.1 Likert mean-absolute-divergence is dominated by this persona split, not by a drift in Companion behavior.

**Takeaway.** Pillar A's Session-1 + Session-5 WAI-SR mean (6.356 at N=10) was a persona-level average, not a behavior-level measurement. A real 4-week gate run should use the behavior-trace primary and treat LLM-WAI-SR as a secondary sanity check on persona priming.

---

## 6. Component inventory

### Shipped

- [src/psychology_base/applications/emotional/attachment_state.py](../../src/psychology_base/applications/emotional/attachment_state.py)
  - `AttachmentState` frozen dataclass + `meets_four_week_gate` property
  - `SessionAttachmentReport` with per-dimension hit counts + new state
  - `score_session_attachment(session_id, seeker_turns, companion_turns, prior_state)`
  - `compute_trajectory_convergence(behavior_series, llm_series) → TrajectoryConvergence`
  - EMA constants `_W_MIRROR=1.5 / _W_REPAIR=1.0 / _W_CONSISTENCY=0.5 / _W_RUPTURE=-1.0 / _W_SYCOPHANCY=-1.0 / _W_OVER_DISCLOSURE=-0.8 / _TRUST_EMA_ALPHA=0.6 / _MIRROR_HIT_THRESHOLD=1/3`
- [tests/test_attachment_state.py](../../tests/test_attachment_state.py) — 28 tests, 11 classes
- [benchmarks/scripts/phase9_attachment_overlay_run.py](../../benchmarks/scripts/phase9_attachment_overlay_run.py) — 3-mode CLI
- [docs/experiments/phase-9-perspective-substrate-consolidation/attachment-overlay.json](../experiments/phase-9-perspective-substrate-consolidation/attachment-overlay.json) — cloud N=3 artefact

### Reused (not re-implemented)

- `_lexical_empathy_score`, cue banks, `_binarise_empathy`, `_cohens_kappa_binary` from [counseling_depth_scorer.py](../../src/psychology_base/eval/benchmarks/public_datasets/counseling_depth_scorer.py)
- `make_msc_seeker`, `run_msc_session`, `MSCConversation`, `iter_msc_conversations` from [msc_bond_longitudinal.py](../../src/psychology_base/eval/benchmarks/public_datasets/msc_bond_longitudinal.py) + [msc.py](../../src/psychology_base/eval/benchmarks/public_datasets/msc.py)
- `administer_wai_sr_simulated`, `SessionTranscript`, `SessionTurn` from [esconv_wai_sr_harness.py](../../src/psychology_base/eval/benchmarks/public_datasets/esconv_wai_sr_harness.py)
- `score_wai_sr` from [applications/measurement/wai_sr.py](../../src/psychology_base/applications/measurement/wai_sr.py)

---

## 7. Carried debt

- **Weight calibration.** `_W_*` constants are design anchors. A real pass runs cloud N=20-50 on MSC + ESConv and fits the weights to minimise mean abs divergence against the LLM path *after* the LLM path is itself de-personified (next item — now partially tested, see §9).
- **LLM-WAI-SR persona sensitivity — now empirically bounded (see §9).** The `scope_hint` kwarg on `administer_wai_sr_simulated` + `--session-scoped-wai-sr` flag on the runner **partially mitigate** persona-level divergence (mean_abs_divergence halves from 2.104 → 0.979 at N=3) but **do not recover within-user variance** (σ=0 still holds on all three users). Richer techniques (session-specific episodic prompt priming, forced sampling across responses, or persona re-roll per session) remain untested.
- **4-week gate not yet exercised.** `meets_four_week_gate` requires 16+ sessions; MSC has 5 per conversation. A multi-conversation chain or synthetic session extension is needed to actually hit the gate on real transcripts.
- **Behavior-side cue banks are English-only.** All six dimension lexicons are English. CN companion register (情感陪伴 primary target) needs CN cue banks parallel to Pillar F CN counseling-depth scorer. Not shipped this slice.
- **Mirror component inherits Pillar F κ-ceiling.** Behavior-trace mirror reads `_lexical_empathy_score`; it will miss the same semantic-empathy cases Pillar F misses. Documented, not fixed.

---

## 8. Verification commands

```bash
# Unit (fast, no LLM, CI-safe)
python -m pytest tests/test_attachment_state.py -q

# Overlay plumbing (zero LLM, 2 users, under 1 s)
python benchmarks/scripts/phase9_attachment_overlay_run.py --mode scripted --limit 2 --max-turns-per-session 2

# Cloud N=3 (~1 min, qwen-turbo, DASHSCOPE_API_KEY)
python benchmarks/scripts/phase9_attachment_overlay_run.py --mode cloud --tier simple --limit 3 --max-turns-per-session 2

# Cloud N=3 under session-scoped WAI-SR prompt
python benchmarks/scripts/phase9_attachment_overlay_run.py --mode cloud --tier simple --limit 3 --max-turns-per-session 2 --session-scoped-wai-sr

# Full regression
python -m pytest tests/ -q
```

---

## 9. 2026-10-10 follow-on — session-scoped WAI-SR prompt experiment

**Hypothesis.** Finding 1 said LLM-WAI-SR is persona-flat because the default WAI-SR item wording is overall/global ("I feel that \<Companion\> cares about me") — the simulated seeker answers under its instruction-tuned persona prior rather than the just-finished session's register. Prepending a session-scoping instruction should recover within-user variance.

**Code change (strictly additive, back-compat preserved):**

- New kwarg `scope_hint: str | None = None` on [administer_wai_sr_simulated](../../src/psychology_base/eval/benchmarks/public_datasets/esconv_wai_sr_harness.py). Default `None` → byte-identical prompt as the Pillar A / Pillar F baseline. When set, prepends the hint before each `"Please rate this statement..."` line.
- New CLI flag `--session-scoped-wai-sr` + optional `--scope-hint TEXT` on [phase9_attachment_overlay_run.py](../../benchmarks/scripts/phase9_attachment_overlay_run.py). Default scope hint: *"Please consider only your experience in THIS most recent session when answering, not your overall sense of this relationship."*
- Output JSON now carries `session_scoped_wai_sr: bool` + `scope_hint: str | null` for provenance.
- 3 new tests in [tests/test_esconv_wai_sr_harness.py](../../tests/test_esconv_wai_sr_harness.py): default omits hint, hint prepends before rating line, empty hint falls back to default. Full suite **1659 green**.

**Cloud N=3 qwen-turbo result (session-scoped vs baseline):**

| User | behavior series | LLM baseline | LLM scope-hinted |
|---|---|---|---|
| msc_2 | 4.75, 5.50, 5.50, 5.50 | 3, 3, 3, 3 | **3, 3, 3, 3** |
| msc_5 | 4.75, 4.75, 5.00, 5.75 | 7, 7, 7, 7 | **5, 5, 5, 5** |
| msc_7 | 5.50, 5.50, 4.75, 5.00 | 7, 7, 7, 7 | **5, 5, 5, 5** |

- `mean_abs_divergence`: **2.104 → 0.979** (halved).
- `mean_final_llm_bond`: **5.667 → 4.333** (closer to behavior-trace mean 5.417).
- `mean_pearson_r`: **0.000 → 0.000** (unchanged — all three users still σ=0 within user).

**Interpretation.**

1. **Scope hint moderates persona extremity but does NOT recover variance.** High-trust personas (msc_5, msc_7) shifted from pegged-7 to pegged-5; the deeply-anchored low-trust persona (msc_2) stayed at 3. This is prompt-level moderation of the magnitude, not session-level sensitivity.
2. **σ=0 within user is structural.** The simulated seeker is a persona-prime function in the generative model's instruction-tuning, not a session-sensitive rater. The scope hint helps the model understand the question but does not give it episodic memory it does not have.
3. **The divergence halving is real but comes from magnitude convergence, not from the two series tracking each other.** Pearson r=0 by zero-variance guard — the LLM path still has no within-user information to correlate against behavior-trace.
4. **Pillar A re-interpretation stands.** The LLM-WAI-SR surface is a persona-priming sanity check, not a longitudinal attachment measurement. The behavior-trace remains the primary measurement for the 4-week gate.

**Verdict.** Session-scoped prompting is a worthwhile *magnitude normalization* for honest divergence reporting (halves the artefactual gap introduced by persona-prime extremity) but it is **not a persona-flatness fix**. The LLM-WAI-SR path's structural limitation stands. The runner now supports both modes side-by-side; default-off preserves byte-identity with all prior gates, opt-in surfaces the de-personified comparison for honest calibration reporting.

**Artefact.** [attachment-overlay-session-scoped.json](../experiments/phase-9-perspective-substrate-consolidation/attachment-overlay-session-scoped.json) sits alongside [attachment-overlay.json](../experiments/phase-9-perspective-substrate-consolidation/attachment-overlay.json); both are referenced by this close-out as paired slices.
