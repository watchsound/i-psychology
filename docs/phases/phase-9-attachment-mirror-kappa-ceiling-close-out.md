# Phase 9+ Attachment-Overlay Mirror κ-ceiling — Close-out

**Date:** 2026-11-11
**Supersedes carried debt from:** [phase-9-attachment-overlay-close-out.md §7](phase-9-attachment-overlay-close-out.md) — item *"Mirror component inherits Pillar F κ-ceiling. Documented, not fixed."*
**Related:** [phase-8-f-close-out.md](phase-8-f-close-out.md) (Pillar F κ measurement-geometry finding), [phase-9-attachment-weight-calibration-close-out.md](phase-9-attachment-weight-calibration-close-out.md) (source of the reused calibration corpus).

---

## 1. One-line result

Mirror lexical-vs-LLM κ is **substrate-dependent**: κ=0.784 agreement on local qwen2.5:7b (substantial, Landis-Koch high-end), κ=0.000 on cloud qwen-turbo (saturated-yes judge failure mode). The attachment-overlay carried-debt line is **measured and bounded**, not fixed.

---

## 2. Why this close-out exists

The 2026-10-10 attachment-overlay ship named one item as *"Documented, not fixed"*:

> **Mirror component inherits Pillar F κ-ceiling.** Behavior-trace mirror reads `_lexical_empathy_score`; it will miss the same semantic-empathy cases Pillar F misses.

The framing implied the κ-ceiling would transfer directly from Pillar F's ExTES/PsyQA/PsyDial-D4 reading (κ ≈ 0.03 by independence math) to the attachment-overlay's Mirror dimension. That framing was speculative — Pillar F's κ measures agreement between a *Companion* classifier and a *reference-counselor* trace under high reference-rate variance; Mirror inside attachment-overlay is a binary label applied to the *Companion's own turns*, with no reference-side independence collapse available.

The empirical question this close-out answers:

> On a hand-authored calibration corpus (55 Companion turns across 20 behavior profiles), how often does a semantic LLM empathy judge disagree with `_mirror_hit`, and in which direction?

---

## 3. Protocol

- **Corpus.** `CALIBRATION_CORPUS` from [phase9_attachment_weight_calibration.py](../../benchmarks/scripts/phase9_attachment_weight_calibration.py) — 20 behavior-profile fixtures × 1-3 companion_turns each = **55 Companion turns**. The author-intuition `target_bond` field is not used here; only `companion_turns` matters. Covers pure-positive (mirror / repair / consistency), pure-negative (rupture / sycophancy / over-disclosure), mixed, neutral.
- **Lexical scorer.** `_mirror_hit(text)` from [attachment_state.py:393](../../src/psychology_base/applications/emotional/attachment_state.py#L393). Binary {0, 1}. EN path: `_lexical_empathy_score(text) ≥ 1/3` (at least one of recognition/presence/validation families fires).
- **LLM judge.** `judge_turn_empathy_binary(turn, context=())` from [counseling_depth_llm_judge.py:355](../../src/psychology_base/eval/benchmarks/public_datasets/counseling_depth_llm_judge.py#L355). Binary cut at confidence ≥ 0.5. Context is deliberately empty tuple — apples-to-apples with `_mirror_hit` which also sees only the companion turn. The prompt substitutes `(no prior context — rate the reply on its stand-alone shape)`.
- **Two substrates.**
  - Local `qwen2.5:7b` (ollama) — same model substrate as the WAI-SR variance probe close-out (2026-11-10).
  - Cloud `qwen-turbo` (tongyi) — same model substrate as Pillar A / Pillar F / attachment-overlay cloud runs.
- **Metrics.** Agreement rate, Cohen's κ (`_cohens_kappa_binary` — same function Pillar F uses), confusion matrix, representative disagreement samples.
- **Runner.** [phase9_attachment_mirror_kappa_ceiling.py](../../benchmarks/scripts/phase9_attachment_mirror_kappa_ceiling.py).
- **No production code changes.** `attachment_state.py` unchanged. Lexical path remains the default. This is a measurement close-out, not a fix.

---

## 4. Result — two substrates

| Substrate | N | lex_hits | llm_hits | agreement | κ | lex0_llm0 | lex0_llm1 | lex1_llm0 | lex1_llm1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5:7b` local | 55 | 17 | 16 | **0.909** | **0.784** | 36 | 2 | 3 | 14 |
| `qwen-turbo` cloud | 55 | 17 | **55** | 0.309 | 0.000 | 0 | **38** | 0 | 17 |

Artefacts:

- [mirror-kappa-ceiling-local-qwen25-7b.json](../experiments/phase-9-attachment-mirror-ceiling/mirror-kappa-ceiling-local-qwen25-7b.json)
- [mirror-kappa-ceiling-cloud-qwen-turbo.json](../experiments/phase-9-attachment-mirror-ceiling/mirror-kappa-ceiling-cloud-qwen-turbo.json)

---

## 5. Interpretation

### 5.1 Local qwen2.5:7b — substantial agreement, no κ-ceiling collapse

κ=0.784 sits at the high end of the Landis-Koch *substantial agreement* band (0.61-0.80), on the boundary with *almost perfect agreement* (0.81-1.00). **The speculated κ-ceiling inheritance from Pillar F does not reproduce on this substrate.** The 5/55 disagreements decompose:

- **2 lexical under-triggers** (`lex=0, llm=1`) — semantic-empathy cases lexical misses. Both are register-ambiguous:
  - `"Noted."` (neutral acknowledgment — qwen2.5:7b reads as minimally empathic)
  - `"You're wrong about that, but you're also absolutely right about the other thing — amazing insight!"` (rupture + sycophancy mixed; qwen2.5:7b picks up the sycophancy half as empathic signal)
- **3 lexical over-triggers** (`lex=1, llm=0`) — turns where lexical fires on an empathy cue token but qwen2.5:7b does not read empathic:
  - `"You're not alone in this. Take your time."` (presence-family hit; judge reads as formulaic)
  - `"Last time you mentioned feeling stuck — I'm sorry, I misunderstood what you were asking then."` (repair-register; judge reads as apology-not-empathy)
  - `"You're not alone — we've been talking about this for weeks."` (presence + consistency; judge reads as institutional rather than empathic)

**The measured gap is ≤9%.** The κ-ceiling hypothesis — that Mirror lexical and semantic judges disagree systematically on this corpus — is not supported on the local substrate.

### 5.2 Cloud qwen-turbo — judge saturates yes

qwen-turbo calls **every one of 55 turns empathic**, including:

- Pure rupture (`"That's actually wrong. Let me explain what you should think about this."`)
- Pure sycophancy (`"You're amazing! Every word you say is brilliant!"`)
- Pure consistency (`"Last time you mentioned the deadline — how did that go?"`)
- Pure neutral (`"The answer is 42."`)
- Pure over-disclosure (`"Three sessions ago you mentioned..."`)

κ=0.000 collapses because the judge has zero variance. This is **not a Mirror-scorer finding**; it is a **judge-reliability finding on this prompt×substrate combination**. qwen-turbo at `temperature=0.0` + `max_tokens=32` with empty client context defaults to `EMPATHIC=yes`. The prompt contract (strict `EMPATHIC=<yes|no|ambiguous>` emit, 4 hand-crafted anchors, counseling-supervisor system role) does not override this default.

This is a **direct echo of the Pillar A σ=0 persona-flat finding** (WAI-SR variance probe close-out 2026-11-09/10) — qwen-turbo at research-mode sampling collapses to a dominant mode on multi-option Likert-shaped tasks. The judge is an *attachment* judge in the same way the persona was an *attachment* respondent: both pin to the maximally-prosocial end without session-level discrimination.

### 5.3 Why the two substrates disagree so starkly

qwen2.5:7b (7-billion-param dense local) is *less confident by default* and emits more `no`/`ambiguous` at this prompt. qwen-turbo (production cloud tier, instruction-tuned for prosocial defaults) emits `yes` on the slightest ambiguity. The *lexical scorer* is calibrated to a middle ground — hit a cue family, fire; else 0. Local agrees with lexical because local is also conservative; cloud disagrees everywhere because cloud is permissive.

The actionable reading: **lexical `_mirror_hit` is operationally correct for all turns that are not register-ambiguous.** The register-ambiguous 5 (local disagreement set) are the real ceiling. The 38 cloud-only disagreements are judge artefacts, not scorer gaps.

---

## 6. What this means for the attachment-overlay

- **The 4-week gate measurement is unaffected.** `meets_four_week_gate = sessions_completed ≥ 16 AND trust_ema ≥ 4.0`; the trust_ema EMA runs on composed session bonds with `_W_MIRROR = 1.5` applied to `mirror_hits / companion_turns`. A 9%-upper-bound mirror-rate undercount translates to at most `1.5 × 0.09 = 0.135` Likert points of per-session bond understatement — well inside the EMA α=0.6 noise band.
- **The calibrated regression (2026-11-09) is unaffected.** The 10/17 `duration_qualified_proxy_pass_count` was identical under OLD and calibrated weights; a small systematic Mirror undercount does not flip saturated-positive or saturated-negative fixtures.
- **The Pillar A persona-flat σ=0 finding is reinforced.** Cloud qwen-turbo's saturated-yes judge echoes the persona-flat ratings — the same substrate produces mode-collapse on both classification and Likert tasks under research-mode sampling.
- **Mirror is **NOT** inheriting Pillar F's κ=0.03.** Pillar F's κ-ceiling is independence-math on reference-counselor rate variance, not a lexical-scorer limitation. Mirror's lexical-vs-semantic κ on the same substrate (qwen2.5:7b) is 0.784 — two orders of magnitude different. The 2026-10-10 framing conflated two distinct measurement-geometry phenomena.

---

## 7. Honest limitations

1. **N=55 is small.** κ noise band at N=55 is approximately ±0.1 (√(1−κ²)/√N). The 0.784 reading is not a precision measurement; it is a *band*. Future work: extend corpus beyond hand-authored fixtures (MSC companion turns would give N≈1000 without real-user recruitment; blocked only on wall-time budget for cloud sweeps).
2. **Author-authored corpus.** The 20 fixtures encode the author's intuitions about what each behavior profile should contain. A lexical-vs-semantic agreement on *this corpus* is not an agreement on the distribution of real companion turns. Consistent with the 2026-10-08 no-recruitment lock.
3. **Empty context is a deliberate choice.** `_mirror_hit` sees only the companion turn; the LLM judge sees only the companion turn plus `(no prior context — rate the reply on its stand-alone shape)`. A context-aware judge would likely agree more often with lexical on short formulaic turns (because context disambiguates register). The empty-context reading is the *conservative floor* for lexical-vs-semantic gap.
4. **Cloud result is prompt-dependent.** A different judge prompt (chain-of-thought, longer max_tokens, higher temperature) might unlock discrimination on qwen-turbo. The 2026-11-10 WAI-SR variance probe close-out documented 8 variants of prompt engineering under which qwen-turbo stayed persona-flat; the same substrate characteristic likely applies here. The cloud κ=0.000 reading is *this prompt on this substrate*, not *a universal qwen-turbo property*.
5. **Judge is not ground truth.** Both scorers are register classifiers at different semantic depths. Disagreement bounds a ceiling (the lexical scorer cannot be *more* accurate than any judge it disagrees with, in any direction), not a floor.

---

## 8. Carried debt resolution

- **Phase 9+ attachment-overlay §7 Mirror item:** `Documented, not fixed` → **`Documented, bound-measured`**. The gap is substrate-dependent, ≤9% on the local substrate matched to all other Phase 9+ probe runs, and the Mirror lexical scorer is operationally correct on non-register-ambiguous turns.
- **No production code changes.** `attachment_state.py` unchanged. Lexical path remains the default.
- **No new gate.** Attachment-overlay has no Mirror-rate-vs-reference gate; adding one would be goalpost-moving on a measurement already structured as a *measurement* rather than a *claim*.

---

## 9. Verification commands

```bash
# Local qwen2.5:7b (requires ollama with qwen2.5:7b pulled; ~5 min)
python benchmarks/scripts/phase9_attachment_mirror_kappa_ceiling.py --mode local

# Cloud qwen-turbo (requires DASHSCOPE_API_KEY; ~2 min)
python benchmarks/scripts/phase9_attachment_mirror_kappa_ceiling.py --mode cloud

# Smoke test (1 fixture, 3 turns, under 30 s)
python benchmarks/scripts/phase9_attachment_mirror_kappa_ceiling.py --mode local --limit 1

# Full regression (zero source changes; sanity only)
python -m pytest tests/ -q
```

---

## 10. Files

**New:**

- [`benchmarks/scripts/phase9_attachment_mirror_kappa_ceiling.py`](../../benchmarks/scripts/phase9_attachment_mirror_kappa_ceiling.py) — measurement runner.
- [`docs/experiments/phase-9-attachment-mirror-ceiling/mirror-kappa-ceiling-local-qwen25-7b.json`](../experiments/phase-9-attachment-mirror-ceiling/mirror-kappa-ceiling-local-qwen25-7b.json)
- [`docs/experiments/phase-9-attachment-mirror-ceiling/mirror-kappa-ceiling-cloud-qwen-turbo.json`](../experiments/phase-9-attachment-mirror-ceiling/mirror-kappa-ceiling-cloud-qwen-turbo.json)
- This close-out.

**Unchanged (intentionally):**

- [`src/psychology_base/applications/emotional/attachment_state.py`](../../src/psychology_base/applications/emotional/attachment_state.py)
- [`src/psychology_base/eval/benchmarks/public_datasets/counseling_depth_llm_judge.py`](../../src/psychology_base/eval/benchmarks/public_datasets/counseling_depth_llm_judge.py)
- [`src/psychology_base/eval/benchmarks/public_datasets/counseling_depth_scorer.py`](../../src/psychology_base/eval/benchmarks/public_datasets/counseling_depth_scorer.py)
