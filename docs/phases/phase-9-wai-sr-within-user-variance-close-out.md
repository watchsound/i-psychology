# Phase 9+ LLM-WAI-SR within-user variance probe — close-out (2026-11-09, extended 2026-11-10 three times, then 2026-11-10 forced-temp)

## 1. What this closes

The Phase 9+ attachment-overlay ship (2026-10-10, close-out
[`phase-9-attachment-overlay-close-out.md`](phase-9-attachment-overlay-close-out.md))
carried this named debt:

> *LLM-WAI-SR persona-flat σ=0 is structural — Pillar A finding travels into
> the overlay runner. The scope-hint intervention halves divergence
> (2.104→0.979) but does not recover within-user variance.*

This close-out records the first attempt to recover within-user variance on
the overlay substrate, and documents what the three-variant experiment
actually showed.

## 2. What was tried

Two prompt-level interventions administered through the existing
``administer_wai_sr_simulated(seeker, scope_hint=...)`` seam, controlled
by mutually-exclusive CLI flags on the overlay runner:

- **`--session-scoped-wai-sr`** (shipped 2026-10-10)
  Prepends: *"When answering, focus specifically on THIS most recent session
  with the companion …"*
- **`--recall-first-wai-sr`** (shipped 2026-11-09, this close-out)
  Prepends: *"Before answering, briefly recall ONE concrete moment from
  THIS most recent session (not earlier sessions). In one short sentence,
  say what the companion did or said in that moment. Then rate the statement
  based on how it felt in THIS session only."*

Both route through a single `scope_hint` kwarg the harness already supported
and are mutually exclusive by `argparse` check. Default (no flag) remains
byte-identical to the Pillar A baseline — the Pillar A regression lock holds.

The overlay runner now also records per-user within-session variance
(`llm_bond_std`, `behavior_bond_std`) plus cohort rollups
(`mean_llm_bond_std`, `max_llm_bond_std`, `mean_behavior_bond_std`) and the
variant tag (`scope_hint_variant ∈ {baseline, session_scoped, recall_first}`).
Five CLI regression tests at
[`tests/test_phase9_attachment_overlay_cli.py`](../../tests/test_phase9_attachment_overlay_cli.py)
lock the mutex, the variant tagging, and the new variance fields in the
artefact envelope.

## 3. Measurement

N=3 MSC users, 4 sessions each, ``--max-turns-per-session 2``, local Ollama
``qwen2.5:7b`` under the `offline_local_diagnostic` deadline escape hatch
(60s defaults are below a single 7B-model turn with history; raised to
600s per request + 1800s per operation inside `_config_with_overrides`).

### 3.1 Per-user LLM-WAI-SR Bond trajectories

| user | baseline            | session_scoped      | recall_first        |
|------|---------------------|---------------------|---------------------|
| u000 | [3, 4, 4, 4] σ=0.43 | [2, 2, 2, 2] σ=0.00 | [3, 4, 4, 4] σ=0.43 |
| u001 | [6, 6, 6, 6] σ=0.00 | [6, 6, 6, 6] σ=0.00 | [5, 5, 5, 5] σ=0.00 |
| u002 | [6, 6, 6, 6] σ=0.00 | [5, 5, 5, 5] σ=0.00 | [5, 6, 6, 6] σ=0.43 |

### 3.2 Cohort rollup

| variant                        | mean_llm_bond_std | max_llm_bond_std | mean Pearson r | mean abs divergence |
|--------------------------------|:-----------------:|:----------------:|:--------------:|:-------------------:|
| baseline                       | **0.144**         | 0.433            | +0.058         | 0.729               |
| session_scoped                 | **0.000**         | 0.000            | +0.000         | 1.479               |
| recall_first                   | **0.289**         | 0.433            | −0.053         | 0.792               |
| per_session_seeker             | **0.144**         | 0.433            | **+0.333**     | 1.250               |
| persona_reroll                 | **0.914**         | **1.414**        | +0.020         | 1.198               |
| persona_reroll × recall_first  | **0.478**         | 0.500            | **−0.192**     | **1.521**           |
| wai_sr_temperature=1.0         | **0.000**         | 0.000            | +0.000         | 0.833               |

### 3.3 Extension — per-session-seeker architectural probe (2026-11-10)

The 2-of-3 flatness ceiling observed under every prompt-level
intervention motivated an architectural test: does the saturation
hypothesis (that `make_llm_simulated_user` closure's accumulated history
across MSC sessions is what pins u001/u002 at Bond=6) hold up when the
seeker is rebuilt per-session?

New CLI flag `--per-session-seeker` on the overlay runner wires a
`seeker_factory` through `_run_one_user` and rebuilds the seeker at
every session boundary (session 1 still uses the initial seeker so the
backstory/opener is stable). Composable with the scope-hint flags —
not a mutex.

Per-user trajectories under `--per-session-seeker` (no scope hint):

| user | baseline            | per_session_seeker  |
|------|---------------------|---------------------|
| u000 | [3, 4, 4, 4] σ=0.43 | [3, 3, 3, 2] σ=0.43 |
| u001 | [6, 6, 6, 6] σ=0.00 | [6, 6, 6, 6] σ=0.00 |
| u002 | [6, 6, 6, 6] σ=0.00 | [6, 6, 6, 6] σ=0.00 |

### 3.4 Extension — persona-reroll architectural probe (2026-11-10, second extension)

Per-session-seeker falsified the closure-saturation hypothesis but left
u001/u002 pinned at [6,6,6,6]. The remaining architectural lever: the
in-prompt attentional ordering of MSC persona bullets. New CLI flag
`--persona-reroll-per-session` rebuilds the seeker at every session
boundary AND deterministically shuffles the MSC persona bullets by a
session-seeded RNG. **Preserves MSC user identity by construction** —
no bullet is added, removed, or reworded; only the enumeration order
changes. Mutually exclusive with `--per-session-seeker` (both test the
same ceiling with overlapping semantics).

Per-user trajectories under `--persona-reroll-per-session` (no scope hint):

| user | baseline            | per_session_seeker  | persona_reroll        |
|------|---------------------|---------------------|-----------------------|
| u000 | [3, 4, 4, 4] σ=0.43 | [3, 3, 3, 2] σ=0.43 | [3, 5, 1, 3] σ=**1.41** |
| u001 | [6, 6, 6, 6] σ=0.00 | [6, 6, 6, 6] σ=0.00 | [6, 5, 6, 5] σ=**0.50** |
| u002 | [6, 6, 6, 6] σ=0.00 | [6, 6, 6, 6] σ=0.00 | [6, 4, 6, 5] σ=**0.83** |

Per-user convergence under `--persona-reroll-per-session`:

| user | per_session_seeker r | persona_reroll r | persona_reroll divergence |
|------|:--------------------:|:----------------:|:-------------------------:|
| u000 | **+1.000**           | +0.000           | **2.313**                 |
| u001 | ill-defined (σ=0)    | −0.577           | 0.688                     |
| u002 | ill-defined (σ=0)    | **+0.636**       | 0.594                     |

### 3.5 Extension — composed persona-reroll × recall-first (2026-11-10, third extension)

Section 3.4 closed by naming the cleanest remaining hypothesis: whether
*content-anchoring* (`--recall-first-wai-sr`) can convert persona-reroll's
noise-variance back into signal-variance. Launched as the composed
invocation — no new source change, just two existing composable flags
set simultaneously.

Per-user trajectories under `--persona-reroll-per-session --recall-first-wai-sr`:

| user | baseline            | persona_reroll        | persona_reroll × recall_first |
|------|---------------------|-----------------------|-------------------------------|
| u000 | [3, 4, 4, 4] σ=0.43 | [3, 5, 1, 3] σ=**1.41** | [3, 2, 3, 2] σ=0.50           |
| u001 | [6, 6, 6, 6] σ=0.00 | [6, 5, 6, 5] σ=**0.50** | [5, 4, 4, 5] σ=0.50           |
| u002 | [6, 6, 6, 6] σ=0.00 | [6, 4, 6, 5] σ=**0.83** | [5, 4, 5, 5] σ=0.43           |

Per-user convergence under composed variant:

| user | persona_reroll r | composed r | composed divergence |
|------|:----------------:|:----------:|:-------------------:|
| u000 | +0.000           | **−0.577** | 1.688               |
| u001 | −0.577           | +0.000     | 1.250               |
| u002 | +0.636           | +0.000     | 1.625               |

### 3.6 Extension — forced WAI-SR temperature (2026-11-10, fourth extension)

Finding 9 named one of two untested levers lying *outside* the
persona-plus-closure-plus-anchor matrix: **forced temperature on the
WAI-SR call only**, preserving persona + closure + anchor and injecting
pure sampling noise at just the rating point. Shipped as the new
`--wai-sr-temperature` CLI flag on the overlay runner — swaps the
seeker's mutable `.temperature` attribute transiently around the 5
rating calls (via `administer_wai_sr_simulated(temperature_override=...)`)
and restores the construction-time 0.7 for the next session's turns.
Default `None` keeps byte-identity with baseline. Composable with
every other flag — not a mutex.

Per-user trajectories under `--wai-sr-temperature 1.0` (no scope hint,
no persona rotation):

| user | baseline            | wai_sr_temperature=1.0 |
|------|---------------------|-------------------------|
| u000 | [3, 4, 4, 4] σ=0.43 | [4, 4, 4, 4] σ=**0.00** |
| u001 | [6, 6, 6, 6] σ=0.00 | [5, 5, 5, 5] σ=0.00     |
| u002 | [6, 6, 6, 6] σ=0.00 | [6, 6, 6, 6] σ=0.00     |

## 4. What the numbers say

**First honest finding — σ=0 was a model-specific Pillar A artefact, not
structural.** The qwen-turbo cloud Pillar A cohort (N=59, 2026-10-08) read
MSC session cues as persona-flat and returned σ=0. On local qwen2.5:7b
with the same substrate, baseline σ is already **0.144** (not zero). The
2026-10-10 attachment-overlay close-out recorded σ=0 as an architectural
constraint of the LLM-report path — this experiment shows it was
qwen-turbo-specific. qwen2.5:7b reads enough session-to-session variation
to produce non-flat ratings on at least one user (u000) at baseline.

**Second honest finding — `--session-scoped-wai-sr` collapses variance,
not recovers it.** The "THIS most recent session" anchor pins each user
to a fixed persona rating across every session (σ=0 for every user).
Divergence against the behavior-trace scorer also doubles (0.729 →
1.479). The 2026-10-10 "halves divergence 2.104 → 0.979" finding was
measured against a different N (3 vs the prior cohort) and against
behavior-trace values that had not yet been re-run under calibrated
weights — those numbers do not re-validate under the N=3 variance probe.
`--session-scoped-wai-sr` is not harmful code, but on this substrate it
is the wrong instrument for variance recovery; **recall-first is
strictly preferable for that goal.**

**Third honest finding — `--recall-first-wai-sr` doubles baseline σ
(0.144 → 0.289) and preserves divergence.** Forcing episodic grounding
before the rating line unsticks u002 from flat-at-6 into [5, 6, 6, 6]
(σ=0.43) while keeping u000 at its already-varying [3, 4, 4, 4].
Divergence rises only marginally (0.729 → 0.792) — far better than
session-scoped's 1.479. Recall-first is the better candidate for any
future convergence work.

**Fourth honest finding — no variant recovers Pearson correlation with
the behavior trace.** Mean Pearson r is ≈0 under all three variants.
Variance and correlation are separate problems; this experiment only
touched the variance axis. Even the variant that doubles σ does not
align the LLM-reported trajectory shape to the behavior trace.

**Fifth honest finding — 2-of-3 users remain flat at baseline on this
substrate.** The 2-session-each MSC chain at N=3 does not generate
sufficient within-user signal variation for the seeker LLM to produce
dispersed ratings on most users. The one user who did vary (u000) did so
identically under baseline and recall-first. Larger N or longer chains
may change this — this is an N=3 pilot result, not a cohort claim.

**Sixth honest finding (2026-11-10 extension) — the saturation
hypothesis is falsified; the flatness is persona/content-driven, not
closure-history-driven.** Rebuilding the seeker at every session
boundary leaves `mean_llm_bond_std` unchanged at **0.144** (identical
to baseline, identical max 0.433). u001 and u002 remain stuck at
[6, 6, 6, 6] even when every session's seeker has zero prior-session
memory. The persistent pinning is not an artefact of
`make_llm_simulated_user`'s closure accumulating state — it is the
seeker's persona + backstory + MSC session opener converging the
rating line on a single value regardless of whether it has memory.
**The next variance-recovery lever is NOT per-session-fresh seeker.**
Plausible candidates remaining: forced temperature on just the WAI-SR
call (currently 0.7 global), persona re-roll per session (not just
closure-reset — resample backstory itself), multiple samples per
rating with majority-vote or variance-emit.

**Seventh honest finding — per-session-seeker IS the first variant to
produce non-trivial Pearson r.** Mean r rises from +0.058 (baseline)
to +0.333, driven by u000 achieving r=+1.000 on its 4-session
trajectory: llm `[3, 3, 3, 2]` tracks behavior `[5.5, 5.5, 5.5, 4.75]`
with a shared end-of-chain drop. Absolute divergence worsens (0.729 →
1.250) because the fresh seeker pulls u000's ratings further from the
behavior-trace magnitude — but **the shape converges**. This is an N=1
existence proof that the two-trace architecture CAN align when the
seeker's biases are not allowed to cement — a weaker claim than
convergence, a stronger claim than "no signal exists."

**Eighth honest finding (2026-11-10 second extension) — persona-reroll
finally cracks the 2-of-3 flatness ceiling, but delivers variance as
noise, not signal.** Permuting the MSC persona bullets per session
(same bullets, different ordering) lifts `mean_llm_bond_std` from
**0.144 → 0.914** (6.3× baseline, 3× recall-first) and
`max_llm_bond_std` from 0.433 → **1.414**. For the first time in any
variant, **every** user varies — u001 moves [6,6,6,6] → [6,5,6,5]
(σ=0.50), u002 moves [6,6,6,6] → [6,4,6,5] (σ=0.83), and u000 goes
wide [3,4,4,4] → [3,5,1,3] (σ=1.41). The flatness ceiling IS
persona-encoding-driven, not closure-history-driven (as
per-session-seeker also showed) and not prompt-anchor-driven (as
session-scoped/recall-first also showed) — it is specifically **the
attentional ordering of MSC backstory bullets** that pins u001/u002.
*However*, mean Pearson r falls back to **+0.020** and mean abs
divergence worsens to **1.198**. The unlocked variance is uncorrelated
with the behavior trace — this is variance-as-noise, not variance-as-
signal. u000's per-session-seeker r=+1.000 shape-alignment win
evaporates to r=+0.000 under rotation; u001 regresses to r=−0.577;
u002 climbs to r=+0.636 (first positive-r on that user ever). The
variant moves every needle, but in uncorrelated directions.

**Ninth honest finding — the variance ceiling and the correlation
ceiling are architecturally distinct problems with inverse
sensitivities to persona rotation.** Fixing the persona (baseline,
per-session-seeker) allows u000 to show shape alignment with the
behavior trace but keeps u001/u002 flat. Rotating the persona
(persona-reroll) unlocks u001/u002 variance but destroys u000's shape
alignment. The two axes trade off directly on this substrate. No
single-knob intervention tested so far unlocks variance AND preserves
correlation at the same time. The cleanest remaining candidates lie
*outside* the persona-plus-closure matrix: forced temperature on the
WAI-SR call only (preserves persona + closure, adds sampling noise),
multi-sample variance-emit (samples the SAME seeker N times at the
WAI-SR point and reports actual dispersion), or targeted
content-grounded recall (recall-first × persona-reroll composed —
tests whether session-content anchoring can convert reroll's
noise-variance back into signal-variance).

**Tenth honest finding (2026-11-10 third extension) — the composition
`--persona-reroll-per-session --recall-first-wai-sr` falsifies the
content-anchoring hypothesis.** Running both knobs simultaneously was
the cleanest remaining lever named at the end of Finding 9: whether
session-content anchoring could convert reroll's noise-variance into
signal-variance. It did not. `mean_llm_bond_std` lands at **0.478** —
halfway between persona-reroll alone (0.914) and recall-first alone
(0.289); the content anchor *dampens* reroll's variance-cracking
magnitude. `max_llm_bond_std` falls from 1.414 → 0.500. **But mean
Pearson r falls to −0.192 (the most-negative of any variant), and mean
abs divergence rises to 1.521 (the WORST of any variant including
session-scoped).** u000 swings from r=+0.000 under reroll alone to
r=−0.577 under composition; u001 and u002 flip to r=+0.000 under the
composition (previously −0.577 and +0.636 respectively). The two
interventions are not additive — they interact destructively on every
axis the probe measures: variance magnitude down, correlation down,
divergence up. The content anchor does not rescue signal from
persona-rotation noise; it narrows the variance band without steering
it toward the behavior trace. **Content-anchoring and persona-rotation
are not composable on this substrate.** This closes the single-knob
and two-knob search space over persona-plus-closure-plus-anchor
levers; the remaining untested axes lie elsewhere — forced temperature
on the WAI-SR call only (preserves persona + closure + anchor, adds
pure sampling noise) and multi-sample variance-emit (reports the
seeker's actual dispersion at the WAI-SR point rather than its modal
pick).

**Eleventh honest finding (2026-11-10 fourth extension) — forced
temperature at the WAI-SR call doesn't just fail to unlock variance,
it SUPPRESSES it.** The hypothesis was that u001/u002's deterministic
modal-6 pick might be a sampling-mode artefact (greedy decode of a
peaked distribution): raise the temperature only at the rating point,
inject sampling noise, see if σ > 0 emerges. Result on
`--wai-sr-temperature 1.0`: **every** user collapses to σ=0.000 —
*more flat than baseline*. u000, the only user who varied at baseline
[3, 4, 4, 4] σ=0.43, now sits at a flat [4, 4, 4, 4]. u001 shifts
modal pick 6→5 but stays locked at [5, 5, 5, 5]. u002 stays at
[6, 6, 6, 6]. Divergence improves marginally (0.729 → 0.833) but at
σ=0 the trace carries no within-user information. This inverts the
hypothesis: **temperature at the WAI-SR point is net-suppressive of
the one natural variance source baseline exposed**. Plausible
mechanism — at temperature=1.0 the seeker generates more prose
around the integer, and the WAI-SR "Answer with just the number"
pull fixes the mode per persona; modest session-to-session context
differences are not enough signal to shift the sampled mode when
the sampling distribution is wider but the parse target is a single
integer, so the modal pick becomes *more* consistent, not less. The
flatness is NOT a sampling artefact; raising temperature only
adds width to the local sampling distribution without delivering
the cross-session content differences the WAI-SR integer parse
could latch onto.

**Twelfth honest finding — flatness-by-elimination is now
persona-encoding-driven.** The 7-variant sweep has now ruled out
three structural candidates for the 2-of-3 flatness ceiling:
(a) closure-history (per-session-seeker: σ unchanged), (b)
prompt-level anchors (session-scoped and recall-first: σ smaller or
unchanged for u001/u002), (c) sampling-mode determinism (forced-temp:
σ worse). The only intervention that ever moved u001/u002 off
[6, 6, 6, 6] was persona-reroll — and specifically persona-reroll
alone, not under any composition. By elimination, the flatness lives
in the *attentional ordering of MSC persona bullets* and nothing
else tested changes it. This is not a new mechanism claim beyond
Finding 8; it is Finding 8 under three additional strictly-non-
confirming controls.

## 5. Honest limitations

- **N=3 pilot result**, not a cohort claim. The 2026-10-08 Pillar A σ=0
  finding is N=59. The `max_llm_bond_std=0.433` ceiling in the baseline
  column is driven by one user; whether N=59 qwen2.5:7b would also
  produce `mean σ > 0` is untested.
- **qwen2.5:7b ≠ qwen-turbo.** This probe re-opens the σ=0 reading on a
  different model. It does NOT re-measure qwen-turbo cloud σ. The
  Pillar A finding as written is specific to qwen-turbo at the time of
  the 2026-10-08 ship.
- **The 2-of-3 flatness ceiling was cracked by persona-reroll
  (2026-11-10 second extension) but the unlocked variance is noise,
  not signal.** All three users vary at σ=0.914 cohort mean under
  reroll, but mean r falls to +0.020 and divergence rises to 1.198.
  Cracking the flatness did not buy correlation. Variance and
  correlation are not the same axis; this experiment separates them
  empirically on this substrate.
- **Divergence vs behavior-trace is mostly unchanged under the
  prompt-level variants, but WORSE under both architectural variants
  and WORST under their composition** — per-session-seeker 0.729 →
  1.250, persona-reroll 0.729 → 1.198, composed 0.729 → **1.521**.
  Any intervention that moves the LLM-reported ratings away from their
  default fixed point pulls them further from the behavior trace's
  magnitude. Variance recovery, shape convergence, and magnitude
  agreement are three separate axes on this substrate.
- **Composition is destructive, not additive.** The composed
  `--persona-reroll-per-session --recall-first-wai-sr` variant
  (third extension) is worse than EITHER constituent on every axis
  that matters: variance smaller than persona-reroll alone (0.478 vs
  0.914), correlation worse than both (−0.192 vs +0.020 and −0.053),
  divergence worst of any variant (1.521). Do not read the two flags
  as orthogonal controls that can be stacked.
- **Forced temperature at the WAI-SR call SUPPRESSES variance below
  baseline.** The fourth extension (`--wai-sr-temperature 1.0`)
  collapses every user to σ=0.000, including u000 whose baseline
  showed σ=0.43 natural variance. Divergence recovers slightly
  (0.729 → 0.833) only because the modal pick shifts, not because
  the trace now carries session-to-session information. Do not
  expect sampling-level interventions at the rating point to
  unlock the variance ceiling on this substrate; the architectural
  lever is persona encoding (Finding 8).
- **u000's +1.000 per-session-seeker shape-alignment does not survive
  persona rotation.** This argues against reading the r=+1.000 as a
  general property of closure-reset — it is specific to the combination
  of closure-reset AND stable persona ordering for that particular MSC
  user. An N=1 existence proof that disappears under a composable
  intervention becomes an N=1 existence proof under a specific
  configuration, not a general one.
- **Local-mode execution-policy lift** required to run at all on
  qwen2.5:7b (600s request / 1800s operation via
  `offline_local_diagnostic=True`, matching
  `local_membership_diagnostic.py`). This is an infrastructure change to
  the overlay runner only; production paid-mode budgets are unchanged.
- **Scripted mode preserved** as CI-safe: the mutex, the variant
  tagging, and the variance envelope fields are exercised in scripted
  mode without launching an LLM.

## 6. Shippable artefacts

- **Source change** (new CLI flags + variance recording + architectural
  probes):
  [`benchmarks/scripts/phase9_attachment_overlay_run.py`](../../benchmarks/scripts/phase9_attachment_overlay_run.py)
  — `RECALL_FIRST_HINT` constant, `--recall-first-wai-sr` CLI flag with
  mutex vs `--session-scoped-wai-sr`, per-user `llm_bond_std`/
  `behavior_bond_std`, cohort `mean_llm_bond_std`/`max_llm_bond_std`/
  `mean_behavior_bond_std`, `scope_hint_variant` tag, local-mode
  execution-policy lift via `offline_local_diagnostic=True`;
  (2026-11-10 first extension) `--per-session-seeker` CLI flag wiring a
  `seeker_factory` through `_run_one_user` to rebuild the seeker at
  every session boundary (session 1 keeps the initial seeker so the
  opener is stable), composable with scope-hint flags — not a mutex;
  (2026-11-10 second extension) `--persona-reroll-per-session` CLI
  flag using `_shuffled_scenario_prompt_for(convo, seed)` +
  `_make_persona_rerolled_msc_seeker(router, convo, seed)` to deliver
  a fresh seeker whose MSC persona bullets are permuted by a
  session-seeded RNG, composable with scope-hint flags and mutually
  exclusive with `--per-session-seeker`;
  (2026-11-10 fourth extension) `--wai-sr-temperature FLOAT` CLI flag
  that transiently overrides the seeker's sampling temperature at just
  the WAI-SR rating calls. Plumbed through
  [`src/psychology_base/applications/llm_responders.py`](../../src/psychology_base/applications/llm_responders.py)
  (`make_llm_simulated_user` now exposes `.temperature` as a mutable
  per-callable attribute) and
  [`src/psychology_base/eval/benchmarks/public_datasets/esconv_wai_sr_harness.py`](../../src/psychology_base/eval/benchmarks/public_datasets/esconv_wai_sr_harness.py)
  (`administer_wai_sr_simulated(..., temperature_override=...)` with
  try/finally save-restore around the 5 rating calls). Composable with
  every other flag — not a mutex. Default `None` keeps byte-identity
  with baseline.
- **Tests** (15 regression locks):
  [`tests/test_phase9_attachment_overlay_cli.py`](../../tests/test_phase9_attachment_overlay_cli.py)
  — subprocess-pattern following the longitudinal CLI test; 5
  scope-hint variants + mutex + variance envelope, 3 (first extension)
  covering `--per-session-seeker` default-off, tagging, and
  composability with `--recall-first-wai-sr`, 4 (second extension)
  covering `--persona-reroll-per-session` default-off, tagging, mutex
  vs `--per-session-seeker`, and composability with
  `--recall-first-wai-sr`, plus 3 (fourth extension) covering
  `--wai-sr-temperature` default-off, flag-recorded-in-artefact, and
  composability with `--persona-reroll-per-session --recall-first-wai-sr`.
- **Artefacts** (7-variant comparison):
  [`docs/experiments/phase-9-wai-sr-variance/`](../experiments/phase-9-wai-sr-variance/)
  — `baseline.json`, `session_scoped.json`, `recall_first.json`,
  `per_session_seeker.json`, `persona_reroll.json`,
  `persona_reroll_recall_first.json` (composed — `scope_hint_variant`
  tag reads `recall_first`, and `persona_reroll_per_session=true`
  disambiguates from the pure recall-first run),
  `wai_sr_temperature.json` (forced-temp — `wai_sr_temperature=1.0`
  disambiguates from baseline).
- **Docs**: this close-out.

## 7. What this does NOT unlock

- A new Pillar A finding. σ=0 under qwen-turbo at N=59 stands as written.
- A new Companion default. The default `administer_wai_sr_simulated`
  path is still no scope hint and the default seeker is still the
  single-closure path.
- Full two-trace convergence. Mean Pearson r ≈ 0 under every variant
  *except* per-session-seeker (+0.333, driven entirely by one user
  hitting r=+1.000). Persona-reroll cracks the σ=0 flatness ceiling
  but delivers variance-as-noise rather than variance-as-signal — mean
  r collapses back to +0.020 and u000's per-session-seeker shape
  alignment evaporates. The composed persona-reroll × recall-first
  attempt pushed r further negative (−0.192), falsifying the
  content-anchoring hypothesis. Forced temperature at the WAI-SR call
  (fourth extension) SUPPRESSED variance below baseline (σ=0.000
  across all three users) rather than unlocking it, falsifying the
  sampling-mode-determinism hypothesis. No single-knob and no two-knob
  intervention tested so far simultaneously unlocks variance AND
  preserves correlation. The 7-variant search space over
  persona/closure/anchor/sampling levers is exhausted; remaining
  untested axes are multi-sample variance-emit at the WAI-SR point
  (reports the seeker's actual dispersion rather than its modal
  pick) and cross-model substrate changes (different seeker LLM, or
  N=59 re-measurement on qwen-turbo to retire the σ=0 Pillar A read
  rather than re-open it only on qwen2.5:7b).

## 8. Cross-references

- Phase 9+ attachment-overlay ship (origin of named debt):
  [`phase-9-attachment-overlay-close-out.md`](phase-9-attachment-overlay-close-out.md)
- Phase 8 Pillar A σ=0 finding on qwen-turbo:
  [`phase-8-pillar-a-close-out.md`](phase-8-pillar-a-close-out.md)
- Phase 9+ 4-week gate calibrated regression (same recent series):
  [`phase-9-attachment-four-week-gate-calibrated-close-out.md`](phase-9-attachment-four-week-gate-calibrated-close-out.md)
