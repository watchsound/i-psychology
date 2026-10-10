# Phase 8 Pillar C — PANDORA + BIG5-CHAT trait inference close-out

**Status:** INCOMPLETE. Soft gate ATTEMPTED → FAIL at N=200. Hard gates BLOCKED on user ToS acceptance for PANDORA.

**Date closed:** 2026-10-08 (soft-gate half; hard-gate half pending user action)

**Pillar reference:** [docs/phases/phase-8-dataset-replay-expansion-plan.md](phase-8-dataset-replay-expansion-plan.md) §2 Pillar C.

**Artefact:** [docs/experiments/phase-8-dataset-replay-expansion/pillar-c/big5-chat-classification.json](../experiments/phase-8-dataset-replay-expansion/pillar-c/big5-chat-classification.json)

**Root-cause analysis:** [docs/experiments/phase-8-dataset-replay-expansion/pillar-c/root-cause-analysis.md](../experiments/phase-8-dataset-replay-expansion/pillar-c/root-cause-analysis.md) — 4-tier causal chain (missing A/O detectors × zero-anchored rescales × clinical-vs-roleplay lexicon mismatch × introspective-vs-embodied cue register), ceiling arithmetic (0.60 architectural cap pre-tuning), fix space (F1 A+O implicit detectors; F2 signed-anchored rescales; F3 role-play lexicon; F4 LLM semantic judge).

---

## Gate

Plan §2 Pillar C hard gate (both required):
- ≥ 3 of 5 OCEAN dims within ±1.0 Likert MAE at N=200 on PANDORA
- `_combine_estimates(log(1+N))` beats flat 50/50 averaging on all 5 dims at N≥200

Plan §2 Pillar C soft gate (optional):
- BIG5-CHAT trait-condition-forward classification accuracy ≥ 60%

## Result (soft gate half)

- `pillar_c_soft_pass = False`
- **polarity_accuracy_incl_neutral = 0.345** (gate metric)
- polarity_accuracy (signed-only, neutral excluded) = **0.683**
- `coverage = 0.505` — fraction of dialogs where any inference signal landed on the labeled dim
- `n_dialogs = 200` (20 per trait×level cell, balanced)
- `n_neutral = 99` (49.5% of dialogs — pipeline produced no signed prediction)
- `wall_s = 0.1` (lexical inference is CPU-bound and sub-second)

Per-trait polarity accuracy (incl. neutral as wrong):

| Trait | Accuracy | Confusion (predicted) |
|---|---|---|
| openness | **0.000** | all 40 neutral |
| conscientiousness | 0.500 | all 40 inferred "low" |
| extraversion | **0.750** | 17 high / 13 low / 10 neutral |
| agreeableness | **0.000** | all 40 neutral |
| neuroticism | 0.475 | 0 high / 31 low / 9 neutral |

Per-level polarity accuracy:

| Level | Accuracy |
|---|---|
| high (100 dialogs) | 0.170 |
| low (100 dialogs) | 0.520 |

## Interpretation — honest FAIL, matches plan's anticipated failure mode

The plan explicitly flagged this risk at §2 Pillar C:

> The Phase 7 B4 debt was `mean_dims_within_half_likert=2.30` at N=50 synthetic personas tuned to the lexical scorer — not evidence of real-user accuracy.

BIG5-CHAT's trait signal is encoded in surface *style* (dialogue phrasing, warmth cues, enthusiasm markers), not in the explicit self-labels Phase 7's cue bank was tuned to detect ("i'm anxious", "i love people", etc.). Three load-bearing failure modes surface:

**1. Agreeableness + Openness are structurally silent.** Phase 7's `_infer_implicit_traits` only computes implicit signals for Neuroticism / Extraversion / Conscientiousness. Agreeableness and Openness fire only on explicit cues. BIG5-CHAT's generated prose almost never embeds those exact lexical cues → both dims land neutral on 40/40 dialogs, scoring 0% polarity accuracy.

**2. Conscientiousness has a floor bias to "low".** The implicit cue (structural markers like `\n- `, bullets, enumeration) is zero on any conversational utterance. Every BIG5-CHAT dialog is a single-paragraph response → struct_rate=0 → score=LIKERT_MIN=1.0 → always predicts "low". Half the labels are "low", so the accuracy is 0.500 by coincidence of balanced data.

**3. Neuroticism and Extraversion detect polarity in opposite directions but still skew low.** Neuroticism relies on PAD-2 valence baseline + arousal rate. On single utterances, these metrics skew conservative → "low" bias (31/40 of dialogs land "low"). Extraversion's exclamation-rate + message-length proxy is more responsive, giving the 75% accuracy that is the pipeline's single success.

Not a mechanism refutation of Phase 7's inference layer: Phase 7 was designed for ≥50-utterance imports, not single-turn classification. The lexical scorer's calibration is specific to its training corpus (synthetic self-labels at B4 scale), and BIG5-CHAT single-turn is the hardest possible stress test — one utterance per profile with no accumulated signal. Failing here is useful: it quantifies the "synthetic-to-real" gap that was already carried-debt from Phase 7 §9 and that Pillar C was explicitly designed to measure.

## Hard-gate status — BLOCKED on user ToS acceptance

The two hard gates both run on PANDORA. PANDORA (Gjurković et al. 2021) requires researcher-use ToS acceptance via Takelab's dataset access form — this is an unautomatable user-side blocker. Until the user:

1. Visits the Takelab PANDORA access form,
2. Accepts the research-use terms,
3. Receives download credentials,
4. Places the data under `.data/pandora/`,

...the hard gates cannot be run. Pillar C stays INCOMPLETE on the hard-gate axis.

Carried from the plan §11:
> **Assumed PANDORA ToS can be accepted by the user.** If rejected, Pillar C ships BIG5-CHAT-only and declares the real-user side unavailable.

The soft-gate FAIL above is the "BIG5-CHAT-only" result the plan contingency anticipated. Formal pillar closure requires either PANDORA acceptance OR a plan amendment to declare the pillar not-pursuable.

## Honest reading for Phase 8 rollup

Pillar C contributes a **soft FAIL + hard INCOMPLETE** to the Phase 8 rollup. Under the plan's rollup rule ("≥ 3 of 6 pillars clear hard gates"), Pillar C currently contributes 0 to the clear count. Pillar A cleared (1 of 6); Pillar C is 0 of 6 pending PANDORA. The minimum-viable sequence A+C+D now requires:

- Pillar A: PASS ✓
- Pillar C: INCOMPLETE on hard, FAIL on soft
- Pillar D: not yet run

If Pillar D clears, the rollup is 2 of 6 — short of the ≥3 threshold. The honest option space:

1. User accepts PANDORA ToS → run hard gates → see if ≥3 clear.
2. Pillar C stays INCOMPLETE → run Pillars D/B/E/F until ≥3 clear or honest-stop.
3. Pivot to a BIG5-CHAT-only Pillar C with a redesigned soft gate that excludes agreeableness+openness or raises the fire-threshold for implicit markers.

Option 3 would require a plan amendment and is not an autonomous-mode call.

## Carried debt (feeds Phase 8 Pillar D run and Phase 9+)

- **Agreeableness + Openness implicit signals.** Phase 7 omitted implicit lexical analysers for these two dims. On any single-utterance input, both are structurally neutral. Phase 9+ work: add implicit cues (e.g., warmth-marker density + perspective-taking cues for A; novelty-vocabulary density + hypothetical-construction rate for O).
- **Phase 7 inference's "low" bias on single utterances.** PAD-2 baseline + struct_rate are zero-anchored, so single-turn inputs skew LIKERT_MIN. Phase 9+ work: either rescale the Likert window for short inputs or admit `score=None` when N<threshold instead of defaulting to LIKERT_MIN.
- **PANDORA hard-gate run.** Blocked on user Takelab ToS acceptance.
- **BIG5-CHAT soft-gate retest after Phase 9 inference-layer fix.** The 0.683 signed-only accuracy is suggestive — if the "neutral" population could be converted to signed predictions, the gate might clear. Store this as a diagnostic baseline.

## Reproduction

```bash
# 1) Shim provisioning (Apache 2.0; ~400MB HF cache; one-time)
python benchmarks/scripts/provision_big5_chat.py --limit 200 --stratify balanced

# 2) Pillar C soft gate (no LLM required, ~0.1s)
python benchmarks/scripts/phase8_c_big5_chat_run.py --limit 200
```

## Files added

- [src/psychology_base/eval/benchmarks/public_datasets/big5_chat.py](../../src/psychology_base/eval/benchmarks/public_datasets/big5_chat.py) — BIG5-CHAT adapter + types + loader.
- [src/psychology_base/eval/benchmarks/public_datasets/big5_chat_classification.py](../../src/psychology_base/eval/benchmarks/public_datasets/big5_chat_classification.py) — Pillar C soft-gate scorer: `Big5ChatClassReport`, `score_dialogs`.
- [benchmarks/scripts/provision_big5_chat.py](../../benchmarks/scripts/provision_big5_chat.py) — HF-mirror → shim JSONL (stratified `balanced` + fallback `random`).
- [benchmarks/scripts/phase8_c_big5_chat_run.py](../../benchmarks/scripts/phase8_c_big5_chat_run.py) — Pillar C CLI runner.
- [tests/test_big5_chat_adapter.py](../../tests/test_big5_chat_adapter.py) — 12 adapter tests (dataclass validation, JSONL loader, accessors).
- [tests/test_big5_chat_classification.py](../../tests/test_big5_chat_classification.py) — 12 scorer tests (lexical hits, neutral handling, aggregation, gate boundary).
