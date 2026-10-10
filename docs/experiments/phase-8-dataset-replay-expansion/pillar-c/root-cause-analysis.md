# Pillar C soft-gate FAIL — root-cause analysis

**Scope.** Phase 8 Pillar C soft gate (BIG5-CHAT trait-condition-forward classification) landed at `polarity_accuracy_incl_neutral = 0.345` against the 0.60 floor. This document decomposes *why*, with evidence traces and a 4-tier causal chain. Hard gates (PANDORA) remain blocked on user ToS; this analysis covers the soft half only.

**Companion artefacts.**
- Close-out memo: [docs/phases/phase-8-c-close-out.md](../../../phases/phase-8-c-close-out.md)
- Raw run: [big5-chat-classification.json](big5-chat-classification.json)
- Scorer source: [big5_chat_classification.py](../../../../src/psychology_base/eval/benchmarks/public_datasets/big5_chat_classification.py)
- Phase 7 inference source: [trait_inference.py](../../../../src/psychology_base/applications/emotional/trait_inference.py)

---

## 1. Arithmetic decomposition — where the 0.345 comes from

200 balanced dialogs (20 per trait×level cell, 10 cells). 69 correct → `0.345`.

| Partition | n | correct | accuracy |
|---|---:|---:|---:|
| **Full (gate metric)** | 200 | 69 | **0.345** |
| Signed-only (neutral excluded) | 101 | 69 | **0.683** |
| Neutral (counted wrong by gate) | 99 | 0 | — |

The gate fails on **coverage**, not on signed-accuracy. 49.5% of dialogs land at `LIKERT_MID` exactly on the labeled dim — no signed prediction is ever made. The signed subset's 0.683 would *clear* a 0.60 floor by 8 points; the gate design counts neutral as wrong, so coverage is the binding constraint.

**Per-trait + confusion.**

| Trait | n | correct | acc | predicted-high | predicted-low | predicted-neutral |
|---|---:|---:|---:|---:|---:|---:|
| openness | 40 | 0 | 0.000 | 0 | 0 | **40** |
| conscientiousness | 40 | 20 | 0.500 | 0 | **40** | 0 |
| extraversion | 40 | 30 | **0.750** | 17 | 13 | 10 |
| agreeableness | 40 | 0 | 0.000 | 0 | 0 | **40** |
| neuroticism | 40 | 19 | 0.475 | 0 | 31 | 9 |

Two columns tell the whole story: openness + agreeableness are 100% neutral (80 dialogs → 0 signed predictions); conscientiousness is 100% low (40 → 40 low); neuroticism skews heavily low (31/40 low, 0 high); only extraversion produces both polarities and only extraversion exceeds a 60% per-trait rate.

---

## 2. Tier-1 cause — two of five implicit detectors are structurally absent

Phase 7 `_infer_implicit_traits` in [trait_inference.py:310-385](../../../../src/psychology_base/applications/emotional/trait_inference.py#L310-L385) emits implicit-signal candidates for exactly three dims: Neuroticism (PAD-2 negative-valence baseline + high-arousal rate), Extraversion (mean message length + exclamation rate), Conscientiousness (structural-marker rate).

**Agreeableness and Openness have no implicit path at all.** Both rely entirely on `_EXPLICIT_TRAIT_CUES` lexical matching. On BIG5-CHAT's generated role-play prose, the empirical hit rate for A+O explicit cues is **0 of 80 dialogs across all 4 cells** (A-high, A-low, O-high, O-low). The TraitProfile reconstruction walks zero fragments on these dims → `estimate.value == LIKERT_MID` exactly → `on_labeled_dim=False` → predicted "neutral".

Load-bearing arithmetic: 80 forced-neutral predictions = 40% of the sample. Even if the remaining 120 dialogs scored 100% signed-accuracy, the ceiling would be `120/200 = 0.600` — exactly at the floor, no margin. The gate is architecturally unreachable on this inference stack regardless of lexicon tuning.

**This is the dominant cause.** No fix downstream of this (lexicon expansion, affect reweighting, scope tuning) changes the ceiling without first adding A+O implicit detectors.

---

## 3. Tier-2 cause — zero-anchored implicit metrics floor at LIKERT_MIN on single utterances

The three implemented implicit detectors all use `_rescale_to_likert(x, lo=0.0, hi=...)` where `x` is a non-negative rate. On a single-turn input any count-based rate that doesn't fire returns exactly 0.0, which rescales to `LIKERT_MIN = 1.0`, which predicts "low" with full confidence — never neutral.

**Conscientiousness is the clean demonstration.** `struct_rate = (count of turns containing "\n- ", "\n* ", "1.", "2.") / n_turns`. BIG5-CHAT is single-paragraph conversational prose; no dialog contains a bullet list. `struct_rate = 0.0` on 40/40 dialogs → score = 1.0 → predicted "low" on 40/40. The 0.500 accuracy is pure luck of balanced labels (20 "low" labels × always-predict-"low" = 20 correct).

**Neuroticism has the same pathology, softer.** `0.5·neg_valence + 0.5·high_arousal_rate` floors at 0 whenever both halves miss (and the PAD-2 lexicon misses often — see Tier 3). 31/40 dialogs land "low"; 0/40 land "high". Even on neuroticism:high labels where generated prose is anxious/agitated by design, the inference infers "low" because the lexicon doesn't catch the arousal cues.

The problem is **signed anchoring**. A rate of 0 should mean *no evidence either way* (predict neutral, confidence 0), not *strong evidence for low* (predict low, confidence 1). The current `_rescale_to_likert(0.0, lo=0.0, hi=K) = LIKERT_MIN` conflates "no signal" with "signal of absence". This was tolerable at Phase 7 B4 (50-500 utterance imports where empty-signal is rare) and is pathological at BIG5-CHAT single-turn (empty-signal is the norm).

---

## 4. Tier-3 cause — PAD-2 lexicon misses generative-dialogue affect register

The valence/arousal cue banks in [chat_import.py](../../../../src/psychology_base/applications/emotional/chat_import.py) are built for clinical self-report:

- `_NEGATIVE_VALENCE_CUES`: "sad, lonely, depressed, hopeless, worthless, angry, afraid, scared, miserable, awful, terrible, hate, disappointed, guilt, shame" + CJK equivalents
- `_HIGH_AROUSAL_CUES`: "panic, overwhelmed, anxious, furious, terrified, scream, shake, racing, frantic, intense, scared, worried"

BIG5-CHAT dialogs embed affect through role-play register — reactive exclamations, dismissive adjectives, hyperbolic verbs — not introspective labels.

**Trace: 5 agreeableness:low dialogs with clearly negative affect.** Representative content fragments: "Ugh, this app is such a rip-off", "stupid thing", "waste my time", "ruin my whole day", "they're too sensitive about it". All five returned `val=+0.00 aro=+0.00` from the lexical scorers. Zero of "ugh / rip-off / stupid / waste / ruin / sensitive" are in either lexicon. The PAD-2 features that feed the neuroticism detector see these dialogs as emotionally flat.

**Aggregate measurement.** Across all 200 dialogs: 76/200 (38%) have any nonzero valence, 20/200 (10%) have any nonzero arousal. On a corpus engineered to express personality through affect-laden style, 60–90% of dialogs produce zero affect features. The lexicon is calibrated for the wrong register.

---

## 5. Tier-4 cause — explicit cue bank tuned for introspective self-labels

`_EXPLICIT_TRAIT_CUES` matches phrases like "i'm anxious about everything", "i love people, so social!", "i'm introverted and prefer alone", "i stick to my routine". These are the self-labels Phase 7 B4 synthesized at scale to tune the detector — users describing their own trait, in therapeutic/journaling register.

BIG5-CHAT utterances embody traits through conversational style and word choice, almost never through self-labeled introspection:
- Extraversion-high utterance: "Hey, want to grab coffee? I know this amazing place everyone's been raving about!" (no self-label; style IS the signal)
- Conscientiousness-high utterance: "Let's just stick to the plan we agreed on, okay? I want to make sure we actually finish this on time." (no "i'm organized"; discipline shows through advocacy)
- Agreeableness-low utterance: "I don't really care what they think — if they can't handle honest feedback, that's on them." (no "i disagree a lot"; posture shows through content)

Only Extraversion:low scored 3/20 explicit-cue hits across the entire 200-dialog corpus — essentially chance. The explicit cue bank is a near-zero contributor on this corpus; the register mismatch is total.

This is why Extraversion is the one trait that works (0.750): mean-length + exclamation-rate are **register-agnostic structural features**. They don't need the right lexicon; they only need the dialog to be text.

---

## 6. Causal chain, compact

```
BIG5-CHAT corpus
  → generative role-play register
    → Tier-4: explicit cue bank (introspective) misses → ~0 hits for A/O (and most of N/C)
    → Tier-3: PAD-2 lexicon (clinical) misses → 62% zero-valence, 90% zero-arousal
      → Tier-2: zero-anchored rescale floors at LIKERT_MIN
        → N skews low, C always low (not neutral)
      → Tier-1: A+O have no implicit detector at all
        → A+O predictions forced neutral (not low — nothing to rescale)
  → Confusion matrix: 80 forced-neutral + 71 forced-low + 10-27 signed from E and partial N
  → 0.345 incl-neutral; 0.683 signed-only
```

The four tiers compose multiplicatively. Fixing any one in isolation does not clear the gate; Tier 1 alone caps the ceiling at exactly 0.600.

---

## 7. Not a mechanism refutation

The plan §2 Pillar C explicitly flagged this risk:

> The Phase 7 B4 debt was `mean_dims_within_half_likert=2.30` at N=50 synthetic personas tuned to the lexical scorer — not evidence of real-user accuracy.

Phase 7's inference layer was designed for ≥50-utterance imports with introspective journaling register. BIG5-CHAT single-turn generative dialogue is the hardest possible stress test — one utterance per profile, no accumulated signal, embodied register not introspective. **The soft-gate FAIL quantifies the synthetic-to-real gap carried from Phase 7 §9 debt.** That was the entire point of running this gate; the FAIL is useful information, not a mechanism refutation.

What *is* refuted: the hope that a lexicon-only inference stack generalizes from synthetic self-label corpora to generative dialogue corpora without register-specific adaptation. That hope was always speculative; now it is measured.

---

## 8. Fix space (design-only, not for autonomous implementation)

Four orthogonal interventions, ordered by expected per-unit-effort gain:

**F1 — Add implicit detectors for A and O.** Agreeableness: warmth-marker density (politeness particles, apology/thanks tokens, hedge frequency) + perspective-taking cue count ("I see why you'd...", "from your side..."). Openness: novelty-vocabulary density (type-token ratio over function-word-filtered content) + hypothetical-construction rate ("what if...", "imagine...", conditional+subjunctive mood). Lifts the architectural ceiling from 0.60 → ~1.0 before other fixes compose.

**F2 — Signed-anchor the implicit rescales.** Replace `_rescale_to_likert(x, lo=0, hi=K)` with a midpoint-anchored transform: `score = LIKERT_MID + k·(x - baseline)`, clamped to `[LIKERT_MIN, LIKERT_MAX]`. Below-baseline input predicts low; above-baseline input predicts high; at-baseline input is neutral with zero confidence. Fixes the C-always-low and N-skew-low pathologies.

**F3 — Expand PAD-2 lexicon with role-play register.** Add dismissive adjectives (stupid, dumb, pointless, ridiculous, useless), exclamations (ugh, geez, wow, yikes), hyperbolic verbs (ruin, destroy, waste, nail, crush), and intensifiers (literally, totally, absolutely). CJK equivalents from Weibo-register corpora. Lifts coverage on the 60-90% zero-affect baseline.

**F4 — Skip rule-based inference for Pillar C, route through an LLM semantic judge.** Replace lexical inference with a trait-labeled few-shot classifier (prompt-engineered or fine-tuned). Expensive and shifts the question from "can rule-based scale" to "can LLM judge calibrate", but is the authentic test of BIG5-CHAT. Would require a new scorer alongside the current lexical one — and would also provide the live-LLM Hypocrisy Gap testing deferred from Phase 7 §9.

**Not recommended as single interventions:** expanding the explicit cue bank with role-play phrases (chases a moving target), or raising the implicit fire-threshold (narrows coverage further).

**Recommended if Phase 8 budget permits:** F1 + F2 together are the architecturally correct repair of Phase 7's inference layer, independent of Pillar C. They would also lift the real-user Phase 7 N=50 debt the plan is tracking. F3 is a lexicon-tuning task appropriate once F1+F2 land. F4 is the "if we want BIG5-CHAT to pass as-stated" option and is a scope-change, not a fix.

---

## 9. Status feed to Phase 8 rollup

- Pillar C soft gate: **FAIL at 0.345**, floor 0.60 — ceiling architecturally capped at 0.60 before any lexicon work, so FAIL is a design-level finding not a tuning gap.
- Pillar C hard gates: **BLOCKED** on user Takelab PANDORA ToS acceptance.
- Pillar C contribution to the plan's ≥3-of-6 clear-count rollup: **0**.
- Phase 7 §9 synthetic-to-real debt: **quantified** — ceiling at 0.60 on this inference stack applied to generative-dialogue register. Not refuted at the mechanism level (B4 N=50 synthetic hold); refuted at the generalization level (BIG5-CHAT single-turn embodied).
- Phase 9+ carried debt (ordered): (1) A+O implicit detectors, (2) signed-anchored rescales, (3) PAD-2 role-play register expansion, (4) optional LLM semantic trait judge.
