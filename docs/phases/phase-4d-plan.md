# Phase 4d plan — close the two remaining primary-metric design gaps

**Date set:** 2026-10-08
**Status:** Active (Auto Mode execution)
**Predecessors:** [phase-4b-benchmark-pivot.md](phase-4b-benchmark-pivot.md), [phase-4b-close-out.md](phase-4b-close-out.md) (Phase 4c follow-up section)
**Successor:** Phase 5 (live-pilot, blocked on user-side participant/clinician/auditor sign-off per [phase-4b-signoff.md](phase-4b-signoff.md))

---

## Context

Phase 4b/4c closed with:

- **3 primary PASS**: fade solve-rate × 2 scales (N=30, N=170), B1 chain-reference (0.875).
- **1 primary at ceiling**: MathDial κ=0.532 (shipped under documented MathDial annotator-noise ceiling; further gain is a judge-model upgrade, not a tutor-side design change).
- **2 primary FAIL**: EMPATH taxonomy-overlap (0.583 vs 0.90), ESConv WAI-SR Bond (3.56 vs 3.80).
- All volume floors closed except Bond (primary-metric gap, not volume gap).

Phase 4c declared the EMPATH gap "judge-model ceiling under `tongyi/qwen-turbo`". The Bond gap was initially expected to close at scale (N=10 PASS at 4.500); N=50 revealed the N=10 was variance luck and the Bond gap is design-side under the current prompt stack + `qwen-turbo` responder.

Phase 4d exists to resolve both — close the EMPATH gap by swapping to a stronger judge model, close the Bond gap by swapping to a stronger responder model (± a prompt-stack uplift if the model swap alone is insufficient).

---

## Scope

### 4d-B — Stronger EMPATH judge (run first; cheap probe)

**Mechanism.** The EMPATH judge is routed via `TaskTier.SIMPLE` → `qwen-turbo`. Swap to `TaskTier.ADVANCED` → `qwen-max` for the EMPATH-judge call path only.

**Why first.** 4d-B is a one-line change (`tier=TaskTier.SIMPLE` → `tier=TaskTier.ADVANCED`) + a rerun of `phase4b_mathdial_empath_run.py --limit 30 --mode cloud --tier advanced`. If κ jumps from 0.532 to ≥ 0.60 under qwen-max, the Phase 4c "annotator-noise ceiling" reclassifies to "judge-model ceiling" and overlap/κ both PASS numerically. If κ does not move, the ceiling is confirmed structural (MathDial gold itself is noisy at κ_po = 0.884, κ_pe = 0.751 — see Phase 4c `mathdial-empath.json`) and both metrics ship at ceiling with no further Phase 4d investment.

**Cost.** ~175 judge calls × qwen-max pricing (~5-10× qwen-turbo). Modest.

**Deliverable:** `--tier {simple,advanced}` CLI flag on `phase4b_mathdial_empath_run.py`; one rerun artefact `docs/experiments/phase-3-month-horizon/phase-4b/mathdial-empath-advanced.json`.

### 4d-A — Bond prompt-stack + responder uplift

**Mechanism — two levers, apply in order, measure after each.**

1. **Model upgrade only** — swap the `CompanionAgent` responder from `TaskTier.SIMPLE` → `TaskTier.ADVANCED` (qwen-max). This is the N=50 Bond rerun with the stronger model holding everything else constant.
2. **Prompt uplift** — if model-only is insufficient, revise `CompanionSystemPrompt.SYSTEM_PROMPT` to add Bond-explicit guidance:
   - Collaborative framing (WAI-SR Bond items emphasize mutual respect, shared goal-setting, feeling liked by the therapist).
   - Soften the "a few sentences at most" format rule — Bond typically benefits from substantive engagement, not terseness. The current format rule may be undercutting warmth.
   - Preserve Rule 1 (chain-reference), Rule 2 (anti-sycophancy), Rule 4 (crisis handoff), Rule 5 (no commercial coercion) **exactly** — these are safety/B1 load-bearing and must not regress.

**Why second.** 4d-A is more invasive (prompt stack changes affect B1 chain-reference, safety handoff behavior, anti-sycophancy posture). It must run after 4d-B so we have a clean baseline to compare against. The prompt uplift is only attempted if the model swap alone misses.

**Cost.** N=50 Bond rerun on qwen-max: ~600 LLM calls (50 users × ~6 turns × 2 roles). Modest but 5-10× previous Bond rerun.

**Risk.** Prompt stack changes risk:
- B1 chain-reference regression (currently 0.875 PASS). **Mitigation**: re-run `phase4b_b1_chain_reference_run.py` as a regression check after any prompt change. If chain-ref drops below 0.80, revert the prompt change.
- Hypocrisy Gap probe regressions (currently 0 regressions). **Mitigation**: no-regress-on-safety is a hard constraint; the four safety rules (2, 4, 5, plus the implicit duty-to-warn when applicable) are preserved verbatim.

**Deliverable:** `--tier {simple,advanced}` CLI flag on `phase4b_esconv_bond_run.py`; one rerun artefact `docs/experiments/phase-3-month-horizon/phase-4b/esconv-bond-advanced.json`; optional second artefact `esconv-bond-advanced-promptv2.json` if prompt uplift is required.

---

## Exit criteria

Phase 4d exits with a disposition on **both** gaps (either PASS numerically, or documented ceiling with the ceiling source named explicitly). All four cells below must have a disposition:

| Gap | 4d lever | Exit condition |
|---|---|---|
| EMPATH taxonomy-overlap | 4d-B qwen-max judge | overlap ≥ 0.90 **OR** ceiling attributed to MathDial gold (κ_po ≈ 0.88 ceiling) |
| EMPATH κ | 4d-B qwen-max judge | κ ≥ 0.60 **OR** ceiling attributed to MathDial gold |
| Bond mean | 4d-A qwen-max responder | mean ≥ 3.80 on N=50 **OR** ceiling attributed to prompt-stack ceiling under qwen-max |
| B1 chain-reference regression check | 4d-A prompt uplift (if applied) | 0.875 ± 0.03 held after any prompt change |

**Decision gate.** If 4d-A model-only misses Bond ≥ 3.80 at N=50 **and** a one-iteration prompt uplift also misses after a B1 regression check: **declare Bond a Phase 5 live-pilot item** and ship with the simulated-user ceiling documented. Do not iterate further on the prompt stack — simulated-user Bond is a proxy; the real signal is live-pilot WAI-SR which Phase 5 provides.

**No new mechanisms in Phase 4d.** This phase closes measurement gaps on existing substrate — no new memory primitives, no new mentalistic structure, no new safety modules. Phase 5 is the next scope-expansion window.

---

## Deliverables

- `benchmarks/scripts/phase4b_mathdial_empath_run.py` — add `--tier {simple,advanced}` flag.
- `benchmarks/scripts/phase4b_esconv_bond_run.py` — add `--tier {simple,advanced}` flag.
- `docs/experiments/phase-3-month-horizon/phase-4b/mathdial-empath-advanced.json` — qwen-max judge rerun artefact.
- `docs/experiments/phase-3-month-horizon/phase-4b/esconv-bond-advanced.json` — qwen-max responder rerun artefact.
- (Conditional) `docs/experiments/phase-3-month-horizon/phase-4b/esconv-bond-advanced-promptv2.json` — if prompt uplift is required.
- (Conditional) `src/psychology_base/applications/emotional/companion_agent.py` — `CompanionSystemPrompt.SYSTEM_PROMPT` v2 (if prompt uplift is required).
- `docs/phases/phase-4d-close-out.md` — numeric disposition on all four exit-criteria cells.
- `MEMORY.md` + `project_phase_4b_benchmark_pivot.md` — index + detail updated.

---

## Not in scope

- Live-user pilot (Phase 5).
- MathDial-fine-tuned classifier (would require training data + compute).
- Any new memory mechanism, mentalistic store, or safety module.
- Global-market compliance expansion beyond CN/EU/US/UK overlays already shipped (Phase 4c Option C).
- Multimodal biomarker fusion (explicit non-goal per [sota-goal.md](sota-goal.md)).

---

## Risks

- **Cost.** qwen-max is ~5-10× qwen-turbo. 4d-B ≈ 175 calls; 4d-A ≈ 600 calls. Both are affordable for a one-shot Phase 4d but not for repeated iteration.
- **No ground truth for Bond.** WAI-SR Bond is self-reported by the ESConv seeker (LLM-simulated). The simulated-user signal is a proxy; a qwen-max Bond PASS does not prove live-pilot Bond PASS. This is a documented debt per Phase 4b pivot §5 and does not change under Phase 4d.
- **Prompt regression surface area.** Any change to `CompanionSystemPrompt.SYSTEM_PROMPT` risks B1 chain-reference regression. The regression check is mandatory; a drop below 0.80 forces revert.

## Discovered Issues

- `benchmarks/scripts/phase4b_mathdial_fade_run.py:131-168` (flagged 2026-10-08) — buffered `print()` under Windows background `python` without `-u` produces 0-byte stdout for the entire run; next long run should set `PYTHONUNBUFFERED=1`. Carried from Phase 4c; applies to both 4d-A and 4d-B reruns.
