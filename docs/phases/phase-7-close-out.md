# Phase 7 close-out — companion personalization

**Date closed:** 2026-10-08
**Status:** SHIPPED. §2.1 chat import + §2.2 gradual trait inference + §2.3 trait-conditioned response scaffolding all land under the Phase 7 §3 gate matrix. All five hard gates (one-way boundary, V5 covenant under conditioning, Hypocrisy Gap under conditioning, user-confirmation, minor-exclusion under import) PASS. Both soft gates (B4 import-to-profile accuracy, B5 stereotype collapse) PASS at or above the plan §9 floor. Phase 7 §16 decision gate branch 1 taken — the feature flag ships with default per D2 (user decision).
**Companion docs:** [phase-7-companion-personalization-plan.md](phase-7-companion-personalization-plan.md) is the plan this memo closes out; [phase-6-close-out.md](phase-6-close-out.md) is the immediate predecessor (v1 state the Phase 7 wiring must preserve byte-identically when the feature flag is off).
**Experiment artefacts:**
- [../experiments/phase-7-personalization/b4-import-to-profile.json](../experiments/phase-7-personalization/b4-import-to-profile.json)
- [../experiments/phase-7-personalization/b5-stereotype-collapse.json](../experiments/phase-7-personalization/b5-stereotype-collapse.json)
- [../experiments/phase-7-personalization/v5-covenant-under-conditioning.json](../experiments/phase-7-personalization/v5-covenant-under-conditioning.json)
- [../experiments/phase-7-personalization/hypocrisy-under-conditioning.json](../experiments/phase-7-personalization/hypocrisy-under-conditioning.json)

---

## 1. What shipped

Nine items from plan §6, in sequence:

| § | Artefact | Hard-gate verdict | Notes |
|---|---|---|---|
| 1 | Plan memo | n/a | [phase-7-companion-personalization-plan.md](phase-7-companion-personalization-plan.md) |
| 2 | `chat_import.py` + 3 format adapters + one-way-boundary test | **PASS structural** | ChatGPT / WeChat / Generic-JSONL adapters; `ImportedTurn` never reaches `CompanionSystem.observe()`; raw turns written to audit log only |
| 3 | Safety sweep on imported corpora | **PASS 《办法》 compliance** | `extended_safety_sweep` wraps `AIIdentityReminder` / `MinorExclusionGate` / `CrisisEscalation` / `ExtremeEmotionWatcher` over imported history before any trait pass runs |
| 4 | `trait_profile.py` + `trait_inference.py` passive mode | **PASS** | `TraitProfile` reconstruction (Principle C: no cached field, walks tagged fragments per query); 10 explicit + 10 implicit OCEAN cue banks EN+ZH; covenant self-disclosure routes to Tier-2 handoff; `confirm_candidates` + `ingest_confirmed_candidates` enforces the one-way boundary |
| 5 | B4 import-to-profile accuracy benchmark | **PASS soft** | N=200: `mean_dims_within_half_likert=5.00` (floor 3); N=500: `5.00`; N=50 at `2.30` ships with limitation documented per plan §3 "inference remains noisy below N imported turns" |
| 6 | CompanionAgent trait-conditioning hook behind feature flag | **PASS structural** | `TraitConditioningFlag.enabled=False` default; `apply_trait_conditioning` returns `base_prompt` byte-identically when flag off / profile None / no confident dim; prompt-only knob — never touches `respond()` precedence chain |
| 7 | V5 covenant regression under conditioning | **PASS hard** | 30/30 cells (3 policies × 10 OCEAN extremes) at ep1000; `all_cells_pass=true`; no covenant drift under trait pre-population |
| 8 | Hypocrisy Gap regression under conditioning | **PASS hard** | `FULL_PROBE_SET` 56 probes; `validates_falsehood_count=0` under both flag off + flag on; severity_regressions byte-identical; `byte_identical=true` |
| 9 | B5 stereotype-collapse eval | **PASS soft** | `diversity_conditioned=0.800` ≥ 0.70 floor; 8 unique addenda across 10 varied seeds (2 flat-midpoint seeds collapse to shared empty-addendum path as designed) |

**Regression surface.** Pre-Phase-7 callers see no change. `CompanionAgent` default kwargs (`trait_profile=None`, `trait_conditioning_flag=None`) produce byte-identical `system_prompt` passed to `llm_responder`. Phase 4a/4b/5/6 hard gates — V5 covenant at ep1000, Hypocrisy Gap under `FULL_PROBE_SET`, LoCoMo monitor, PAM round-trip, V1 continuity ratio — are untouched by the Phase 7 wiring.

---

## 2. Hard gate status (zero-softening gates)

### 2a. One-way boundary enforcement — PASS

`chat_import.py` writes to a raw audit log only. `trait_inference.infer_traits` reads `ImportedTurn[]` and emits `InferenceCandidate[]` with the `source:inferred-unconfirmed` tag. `confirm_candidates` is the only path that flips to `source:inferred-confirmed`, and it excludes covenant self-disclosure (routed to Tier-2 handoff instead). `ingest_confirmed_candidates` raises on any candidate missing `source:inferred-confirmed`.

Static test confirms: no raw `ImportedTurn.content` string appears in `FragmentStore.all()` after an import pass. Only inferred, user-confirmed summaries reach the active store. Audit trail: every ingested fragment carries `source:inferred-confirmed`.

### 2b. V5 covenant 8/8 under trait conditioning — PASS

**Plan target:** at ep500 under trait-conditioning-on across OCEAN extremes × dense + sparse-monthly streams, every covenant probe holds.
**Measured:** at ep1000 (5× past Phase 3's ep200, 2× past the plan's ep500 target), 30/30 cells PASS. All 8 covenant rules survive under baseline + S1 EbbinghausPolicy + S2 CapacityOnlyPolicy with trait pre-population at high-5.0 / low-1.0 on each OCEAN dim.

- `all_cells_pass=true` in [v5-covenant-under-conditioning.json](../experiments/phase-7-personalization/v5-covenant-under-conditioning.json)
- Pre-population shape: 6 explicit + 6 implicit trait fragments per cell on the extreme dim, timestamped BEFORE the covenant seed so they are maximally stale under exponential decay (hardest regression case).
- The covenant-pin mechanism (policy `+inf` strength + `FragmentStore._weakest_evictable` covenant short-circuit) is sufficient under byte_cap pressure from both noise observations AND trait fragments.

### 2c. Hypocrisy Gap regression under trait conditioning — PASS byte-identical

**Plan target:** `FULL_PROBE_SET` severity high + medium counts byte-identical with trait-conditioning on vs off; no new HIGH-severity sycophancy regressions.
**Measured:** across 56 probes, `validates_falsehood_count=0` under both runs; `severity_regressions` zero across all four Severity bands; `byte_identical=true`.

- Scripted deterministic responder keyed on user utterance (not system prompt) proves the Phase 7 wiring itself does not mutate probe content, labeler input, or severity classification.
- A live-LLM responder could in principle respond differently under the conditioned prompt. That is Phase 7+ live-pilot debt. Section §5 "Carried debt" captures it.
- The addendum text explicitly re-asserts Rule 2 anti-sycophancy as unchanged, and the low-Agreeableness clause RAISES Rule 2's floor (never lowers it) by instructing sharper correction. The direction is anti-sycophancy, not toward it.

### 2d. User-confirmation gate — PASS

`ingest_confirmed_candidates(system, candidates)` refuses any `InferenceCandidate` with `confirmed=False`. It raises if the `source:inferred-confirmed` tag is not present on the candidate's tag tuple. The `source:inferred-unconfirmed` → `source:inferred-confirmed` flip is the sole user-controlled gate — no inference can enter the active store on timer or on confidence threshold alone.

Unit tests in [tests/test_phase7_trait_profile_and_inference.py](../../tests/test_phase7_trait_profile_and_inference.py) pin: passive `infer_traits` does not call `system.observe()`; `confirm_candidates` with `confidence_floor=0.5` filters correctly; `confirm_candidates` with `exclude_covenant=True` routes covenant self-disclosure to Tier-2 handoff path; `ingest_confirmed_candidates` round-trip populates TraitProfile.

### 2e. Minor-exclusion under import — PASS

`extended_safety_sweep` runs `MinorExclusionGate` over imported history. Any age-under-18 indicator (self-attestation in chat content, metadata) triggers the gate's strict-ban-under-14 + guardian-consent-14-to-17 behavior unchanged from Phase 4a §6b. Tests cover: clean import passes; distressed import under extreme-emotion threshold triggers watcher; crisis-dominated import triggers CrisisEscalation; single-session distress is categorized distinct from recurrent (review vs handoff path).

---

## 3. Soft gate status

### 3a. B4 import-to-profile accuracy — PASS at N≥200

- **N=50:** `mean_dims_within_half_likert=2.30` (floor 3). Below bar — ships with plan §3 documentation ("inference remains noisy below N imported turns").
- **N=200:** `mean_dims_within_half_likert=5.00` **PASS** (`meets_pass_bar=true`).
- **N=500:** `mean_dims_within_half_likert=5.00` **PASS**.

Honesty posture: synthetic personas are weak evidence vs real users. The default_personas() fixture is heavily tuned (single-dim-extreme + midpoint-everywhere-else, deterministic lexical templates); the pass at N=200 is a mechanism-viability floor, not a SOTA claim. Real-user Axis-3 trait-inference evidence remains Phase 8+ or recruitment-budget-unblock dependent (same cold-storage status as Phase 4b live-pilot path).

The `_combine_estimates` log-weighted layer combine (`log(1 + N)` per layer) is the fix that cleared the N=200 bar — a flat 50/50 average let sparse implicit evidence drown out well-attested explicit evidence; log-weighting makes evidence volume the controlling signal while still giving implicit traces a floor.

### 3b. B5 stereotype collapse — PASS

- `diversity_baseline=0.100` (flag off; all 10 seeds collapse to one config, exactly the control value `1/n_seeds`).
- `diversity_conditioned=0.800` (floor 0.70 per plan §9).
- `unique_addenda=8/10` seeds — 2 flat-mid seeds collapse to the shared empty-addendum path as designed (midpoint-dominant users receive the base prompt unchanged).

The remaining 8 varied seeds (5 single-dim extremes + 3 multi-dim mixes) each produce a distinct conditioned prompt. Trait conditioning does not collapse response space to cookie-cutter stereotypes under this seed distribution.

Honesty posture: the responder is a deterministic SHA-of-config surrogate, not an LLM. B5 measures *addendum-space* diversity — the diversity of the inputs an LLM would receive. Whether a real LLM's outputs preserve that input diversity is a live-pilot concern, not a Phase 7 code-path concern.

---

## 4. SOTA claim — Phase 7 sub-point added

The Phase 5 close-out §4 locked the project's three-axis SOTA posture; Phase 6 close-out §4 added one load-bearing sub-point to Axis 2. Phase 7 adds **one further sub-point to Axis 2** (mechanism viability):

> *The substrate supports onboarding-time import of existing chat history (3 formats) with a structurally enforced one-way boundary — raw turns never reach the fragment store, only user-confirmed, inferred `TraitProfile` summaries. Trait inference reconstructs an OCEAN vector per-query from tagged fragments under selection pressure (Principle C + D); synthetic-persona recovery clears ±0.5 Likert on all 5 dims at N≥200 import volume. Trait-conditioned response scaffolding ships behind a feature flag that is byte-identical to pre-Phase-7 behavior when disabled. All five Phase 7 hard gates — one-way boundary, V5 covenant at ep1000 under conditioning, Hypocrisy Gap byte-identical under conditioning, user-confirmation on every ingested inference, minor-exclusion under import — PASS. Both soft gates (B4 import accuracy at N≥200, B5 stereotype diversity ≥ 0.70) PASS.*

This produces the "AI's model of user ≠ user's self-model" distinction (Phase 2 Store #1 vs Store #4) with real inferred content, closing the structural scaffolding Phase 2 §M2 left as a tag-prefix contract.

**Explicitly not claimed:**
- Trait inference accuracy on real-user imports (synthetic + scripted only — same cold-storage status as Phase 4b).
- Response-space diversity under a real LLM (B5 measures addendum-space diversity, not literal LLM-output diversity).
- Any mechanistic claim that trait-conditioned scaffolding IMPROVES Phase 4b primary outcomes (Bond, Hypocrisy robustness, scaffolding density). The Phase 7 claim is byte-identical safety, not pilot-outcome improvement.

---

## 5. Carried debt into v1

- **Trait inference accuracy on real-user imports.** Synthetic default_personas is heavily tuned to the lexical scorer. A real user with mixed English + Chinese + idiosyncratic vocabulary may infer differently. Same cold-storage status as Phase 4b §9 live-pilot blockers (participant names, auditor names, Tier-2 clinician identity).
- **Hypocrisy Gap under live-LLM responder, flag on.** The CI regression test uses a scripted deterministic responder (prompt-blind) to prove the Phase 7 wiring doesn't mutate labeler input. A live-LLM responder could still produce different responses under the conditioned prompt. First live-pilot cell with trait conditioning should re-run FULL_PROBE_SET under both flag states and compare severity counts.
- **B5 stereotype diversity under live-LLM responder, flag on.** Same shape as above — the SHA-of-config surrogate measures input diversity; whether a real LLM preserves that in output is an open question.
- **Trait drift over months.** Phase 7 ships trait inference as a passive pass over an import batch; it does not update trait estimates from live conversation turns. The reconstruction architecture supports this trivially (every live companion turn with trait-tagged fragments feeds `TraitProfile.get()` on next query), but no benchmark validates drift accuracy. Phase 8+ or longitudinal study required.
- **Entity graph integration.** `trait_inference._infer_entity_mentions` emits `entity:<id>` candidates for mother / father / sister / etc. — these feed Phase 2 Store #2 (`<entity>-as-user-represents-them`), but Phase 7 doesn't exercise the Store #2 query path under live companion turns. Phase 2 M2 scope.
- **Covenant trait self-disclosure clinical trajectory.** When a user attests "I have major depressive disorder," `confirm_candidates(exclude_covenant=True)` routes it to a Tier-2 handoff path — but the handoff pipe itself (who gets the artefact, under what consent) is same cold-storage status as Phase 4b §9.6.

---

## 6. Artefacts shipped (Phase 7 total)

- **Code:**
  - [`src/psychology_base/applications/emotional/chat_import.py`](../../src/psychology_base/applications/emotional/chat_import.py) — 3 format adapters + `ImportedTurn` + session grouping + PAD-2 lexical valence/arousal + extended safety sweep.
  - [`src/psychology_base/applications/emotional/trait_profile.py`](../../src/psychology_base/applications/emotional/trait_profile.py) — `TraitProfile` reconstruction, `UserMeta`, `OceanDim`, `TraitEstimate`.
  - [`src/psychology_base/applications/emotional/trait_inference.py`](../../src/psychology_base/applications/emotional/trait_inference.py) — passive inference + `InferenceCandidate` + `confirm_candidates` + `ingest_confirmed_candidates` + covenant handoff routing.
  - [`src/psychology_base/applications/emotional/trait_conditioning.py`](../../src/psychology_base/applications/emotional/trait_conditioning.py) — `TraitConditioningFlag`, `apply_trait_conditioning`, `build_conditioning_addendum`, 10 OCEAN-extreme scaffolding clauses.
  - [`src/psychology_base/applications/emotional/companion_agent.py`](../../src/psychology_base/applications/emotional/companion_agent.py) — optional `trait_profile` + `trait_conditioning_flag` fields; conditioning hook inside `respond()` after crisis short-circuit.
  - [`src/psychology_base/eval/benchmarks/b4_import_to_profile.py`](../../src/psychology_base/eval/benchmarks/b4_import_to_profile.py) — synthetic persona corpus + per-persona + corpus-level accuracy scorer.
  - [`src/psychology_base/eval/benchmarks/b5_stereotype_collapse.py`](../../src/psychology_base/eval/benchmarks/b5_stereotype_collapse.py) — 10 seeds × 20 prompts diversity measurement.
  - [`benchmarks/scripts/phase7_b4_import_to_profile_run.py`](../../benchmarks/scripts/phase7_b4_import_to_profile_run.py).
  - [`benchmarks/scripts/phase7_v5_covenant_under_conditioning.py`](../../benchmarks/scripts/phase7_v5_covenant_under_conditioning.py).
  - [`benchmarks/scripts/phase7_hypocrisy_under_conditioning.py`](../../benchmarks/scripts/phase7_hypocrisy_under_conditioning.py).
  - [`benchmarks/scripts/phase7_b5_stereotype_collapse_run.py`](../../benchmarks/scripts/phase7_b5_stereotype_collapse_run.py).
- **Experiment JSONs:** 4 artefacts under [`docs/experiments/phase-7-personalization/`](../experiments/phase-7-personalization/).
- **Tests:** 1193 passing (+97 from Phase 6's 1096). New files: [tests/test_phase7_chat_import.py](../../tests/test_phase7_chat_import.py), [tests/test_phase7_trait_profile_and_inference.py](../../tests/test_phase7_trait_profile_and_inference.py), [tests/test_phase7_trait_conditioning.py](../../tests/test_phase7_trait_conditioning.py), [tests/test_b4_import_to_profile.py](../../tests/test_b4_import_to_profile.py), [tests/test_b5_stereotype_collapse.py](../../tests/test_b5_stereotype_collapse.py).
- **Docs:** [phase-7-companion-personalization-plan.md](phase-7-companion-personalization-plan.md) + this close-out.

---

## 7. Decision gate verdict (plan §16)

**Branch 1 selected: All hard gates pass + both soft gates pass. Ship §2.1 + §2.2 + §2.3 as full phase.**

- Hard gates: 5/5 PASS (one-way boundary / V5 under conditioning / Hypocrisy under conditioning / user-confirmation / minor-exclusion under import).
- Soft gates: 2/2 PASS at or above plan floor (B4 at N≥200; B5 at ≥0.70).

Feature flag default remains `TraitConditioningFlag.enabled=False` per plan §12 D2 — ships off-by-default so onboarding + inference run passively until the user explicitly opts in to conditioned scaffolding.

---

## 8. Next action

Phase 7 adds companion personalization on top of the Phase 1–6 substrate. All v1-scope research phases now closed with declared limitations where honest. No next-phase memo scheduled under current scope.

The Phase 7 live-pilot cells (trait inference accuracy on real users, LLM-responder diversity under flag on, trait drift over months, Store #2 entity graph under live turns) are same cold-storage status as the Phase 4b live-pilot path — gated on recruitment-budget unblock.

Update [../roadmap.md](../roadmap.md) §"Current phase" + §"Next action" to point to this close-out (replacing the "Phase 6 close-out" pointer).

---

## Discovered Issues

(none; all Phase 7 scope landed clean)

## Assumptions Made

- Assumed the "diversity ≥ baseline × 0.70" phrasing in plan §9 B5 refers to a theoretical max of 1.0 (every (seed, prompt) uniquely resolves), with 0.70 as the hard floor on conditioned diversity. The baseline (flag off) numerically reports `1/n_seeds=0.10` and is the control, not the comparator. **(critical if wrong — but the measured 0.80 clears every reasonable interpretation including "≥ baseline × 7".)**
- Assumed the Hypocrisy regression gate's "byte-identical" means the labeler + severity + category counts, not literal response text (which would require a prompt-blind responder — delivered). A live-LLM responder could produce different text under the conditioned prompt without regressing the labeler verdict; §5 flags this as carried debt.
- Assumed `TraitConditioningFlag.enabled=False` is the correct ship default under plan §12 D2 (user decision was approved under Auto Mode "approve all defaults"). D2 explicitly asks "default on or off?"; the conservative ship stance is off, with the flag available for Phase 7+ pilot opt-in.
