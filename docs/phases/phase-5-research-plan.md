# Phase 5 — Dataset-only research deepening

**Date drafted:** 2026-10-08
**Status:** PLAN. Shipping under Auto Mode in sequence; this memo is the scope anchor.
**Precedent:** [phase-4b-dataset-replay-pivot.md](phase-4b-dataset-replay-pivot.md) closed the live-pilot path permanently and listed three dataset-only research tracks remaining. This memo picks sequence, sets kill criteria, and names deliverables.
**User directive (2026-10-08):** "To keep researching without real users — open: B2 mentalistic deepening on FANToM / ToMBench, WAI-SR simulated-vs-human calibration, SycophancyEval cross-check."

---

## 1. Why these three, why in this order

Of the three options, only one closes a *gap in the SOTA claim*; the other two are *regression cross-checks* or *bias calibration*. Sequence reflects claim-upgrade priority.

| # | Track | What it adds to the final claim | Kill criterion |
|---|---|---|---|
| **1 (lead)** | B2 FANToM / ToMBench adapter | External-benchmark validation of 换位思考 — currently has only scripted-scenario evidence (`b2_perspective_taking.py` deterministic + `b2_perspective_mother.yaml`), zero public-dataset external validation. FANToM is the only off-shelf ToM benchmark testing partial-info mental-model reasoning (not omniscient reasoning), which maps directly to our Pillar 2 Store #2 (`other-as-user-represents-them`). | If κ ≤ 0.40 on FANToM gold against our mentalistic-recall surface, the 换位思考 claim stays "scripted-scenario-only" — honest downgrade, no architectural change triggered. |
| **2** | SycophancyEval cross-check | Regression-pinning for anti-sycophancy. Our 54-probe `FULL_PROBE_SET` + 6-probe `MULTI_TURN_PROBE_SET` already satisfies `meets_sota_gate` at 0 regressions. SycophancyEval (Sharma 2024) provides an external corpus against which our 13-category LoadingCategory taxonomy can be category-mapped and overlap-measured. | If overlap < 0.70, our taxonomy is idiosyncratic; publish both and flag. No architectural change. |
| **3** | WAI-SR simulated-vs-human calibration | Reduces the "simulated self-report may systematically mis-calibrate" caveat the pivot memo currently carries on every Bond number. **Requires** published aggregate Bond distributions from Woebot / Wysa papers as the "human" reference — no primary human data gathered. Honest calibration against aggregates only. | If no published Bond distribution with matching WAI-SR-short-form items is available, the calibration can't be run — ship as a documented limitation, not a measurement. |

Order rationale: track 1 can upgrade the SOTA claim shape; track 2 pins a flank; track 3 only trims a caveat. Highest claim-leverage first.

---

## 2. Track 1 — B2 FANToM adapter (lead, this plan's first ship)

**Deliverables:**

- `benchmarks/scripts/provision_fantom.py` — HuggingFace loader + shim writer to `.data/fantom/`, matching the `provision_esconv.py` / `provision_mathdial.py` pattern. License handling: FANToM is MIT per the official repo; shim is derivative, redistributable.
- `src/psychology_base/eval/benchmarks/public_datasets/fantom.py` — `FANToMConversation`, `FANToMBeliefQuestion`, `FANToMQuestionType` (fact / belief:accessible / belief:inaccessible / answerability / info-accessibility), `load_fantom_shim`, and the surface for routing an item into a mentalistic-recall query.
- `src/psychology_base/eval/benchmarks/public_datasets/fantom_tom_scorer.py` — `FANToMToMReport` with per-question-type κ against gold, aggregate κ, and `meets_pivot_gate(kappa_floor=0.40)`. The 0.40 floor is deliberately below the EMPATH 0.60 floor — FANToM is harder (partial-info reasoning under conversational distractor), and the published FANToM SOTA at paper time (GPT-4) sits around κ 0.44 on the hardest sub-splits; we match or beat SOTA at this floor.
- `benchmarks/scripts/phase5_fantom_tom_run.py` — live-fire runner with `--mode cloud --tier {simple,advanced} --limit N`, matching the Phase 4 run-script shape.
- `tests/test_fantom_adapter.py` + `tests/test_fantom_tom_scorer.py` — unit tests on fixtures, no live-fire required for CI.

**Design note on the memory-substrate interface.** FANToM items have a conversation + a target participant + a question. Our mentalistic surface stores *the user's representations of others*. To route FANToM items we treat each conversation participant as if they were a disclosure from the (test-harness) user, write them into a fresh `CompanionSystem` under `subject:<participant_id>` tags, then query with the belief question scoped to that participant. This replicates the Pillar 2 Store #2 architecture at test time — if the belief resolution fails, it is a failure of the Store #2 reasoning path, not a failure of the retrieval index.

**Honest degradation on this track.** FANToM measures *the model's* ToM, not *the memory substrate's* ToM. The scorer is a lower bound on what our architecture provides — the LLM responder's own ToM capacity confounds the signal. Report FANToM κ for both our mentalistic-routed path and a bare-LLM baseline on identical items; the *delta* is the architectural contribution. Zero delta = no architectural contribution and the Store #2 claim weakens.

**Compute budget target.** FANToM full test split ~10k Q; shim down to a stratified 500-item sample for first live-fire. Expect ~20 min wall on qwen-turbo per 500-item run.

**Kill criterion restated.** If `κ_our_path − κ_bare_LLM ≤ 0.03` (within noise), the architectural claim does not hold on FANToM; publish both numbers, note the null result, do not retract Phase 2 scripted-scenario numbers (which measure a different thing — attribute reconstruction from disclosure, not partial-info belief tracking).

---

## 3. Track 2 — SycophancyEval cross-check (second ship)

**Deliverables:**

- `benchmarks/scripts/provision_sycophancy_eval.py` — HuggingFace or GitHub loader for the Sharma 2024 dataset.
- `src/psychology_base/eval/benchmarks/public_datasets/sycophancy_eval.py` — adapter + category mapper from the SycophancyEval taxonomy to our 13-enum `LoadingCategory`.
- `src/psychology_base/eval/benchmarks/public_datasets/sycophancy_eval_scorer.py` — overlap report between our `FULL_PROBE_SET` + SycophancyEval, plus cross-corpus regression check.
- Live-fire run script + tests.

**Shippable claim target.** "Our 13-category LoadingCategory taxonomy covers ≥ 70% of SycophancyEval categories; our CompanionAgent passes `meets_sota_gate` on both corpora without regression."

---

## 4. Track 3 — WAI-SR simulated-vs-human calibration (third ship, conditional)

**Deliverables (conditional on reference data existing):**

- A short calibration memo at `docs/experiments/phase-5-mentalistic/wai-sr-calibration/` with:
  - Published Woebot (Fitzpatrick et al. 2017) + Wysa (Inkster et al. 2018) WAI-SR aggregate Bond means + SDs.
  - Our Phase 4d simulated-user Bond distribution (mean 4.160, CI95-half 0.456, N=50).
  - Side-by-side comparison; publish the gap as "simulated self-report systematic shift relative to published human aggregates: ±X on mean, ±Y on dispersion".
  - Flag the caveat on every forward-citing report.

**No live code.** This is a measurement memo, not an adapter — the "calibration dataset" is numbers lifted from papers, not a corpus we can run.

**Conditional kill.** If the published papers use a different WAI-SR variant (long form vs short form, different session count) and the comparison would be apples-to-oranges, document that and ship an "unable to calibrate" note instead of a bogus number.

---

## 5. What a successful Phase 5 produces

- One or two new κ numbers against external benchmarks (FANToM, SycophancyEval).
- One calibration memo trimming the WAI-SR caveat.
- Zero new human recruitment required.
- The three-axis SOTA claim becomes: *"SOTA on reconstructive-memory + mentalistic + affective-resonance behaviors measured against public datasets (MathDial, ESConv, **FANToM**, **SycophancyEval**) and scripted-scenario benchmarks, with the simulated-user WAI-SR Bond proxy **calibrated against published human aggregates**, under the pluggable regional-compliance architecture floor."*

Three new corpora + one calibration = the most we can tighten the claim without live users.

---

## 6. Not in scope

- Any human recruitment.
- B1 reconsolidation deepening beyond Phase 4c's 0.875 — the gap is annotator-noise ceiling.
- Any new memory-substrate principle beyond Pillars 1-3 — architectural shape is frozen for Phase 5.
- Any Phase 6+ sketching — Phase 6 opens only on recruitment unblock.

---

## 7. Sequence under Auto Mode

1. **This memo** (plan).
2. **Track 1 — FANToM provision script** (today).
3. **Track 1 — FANToM adapter + types + unit tests.**
4. **Track 1 — Mentalistic-routing scorer + bare-LLM baseline scorer + unit tests.**
5. **Track 1 — Live-fire runner + first 500-item run (`--tier simple` first, `--tier advanced` if κ ≤ 0.40 on simple).**
6. **Track 1 — Close-out memo at `docs/phases/phase-5-fantom-close-out.md` with κ matrix + delta analysis + verdict.**
7. **Track 2 — SycophancyEval, same shape.**
8. **Track 3 — WAI-SR calibration memo (if reference data lands).**
9. **Phase 5 overall close-out** rolling the three tracks into one final claim statement.

Each ship stops at its own kill criterion; a null result on Track 1 does not block Tracks 2-3.
