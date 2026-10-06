# Roadmap — from current state to SOTA

**Date set:** 2026-10-05
**Companion doc:** [sota-goal.md](sota-goal.md) defines what "SOTA" means; this doc sequences how we get there.

---

## Where we are (2026-10-05)

- Design docs complete: [memory-design.md](memory-design.md), [memory-gaps.md](memory-gaps.md), [benchmarks.md](benchmarks.md), [sota-goal.md](sota-goal.md), this file.
- Project memory records the architectural stance and prior-art reference (Ouroboros at `C:\Users\nihan\Desktop\ai-projects\ai-writing\ai-consciousness`).
- No source code. No tests. No baseline measurements.
- Six principles (A–F) and three gap-closing behaviors (reconsolidation, 换位思考, 共情) are the design primitives. Specific mechanisms are not yet chosen — they will be driven by what the harness forces us to build.

The ordering below is deliberate: **harness before mechanism, measurement before claim.** The goal doc fixes what counts; the roadmap sequences the work that produces evidence against those counts.

---

## Phase 0 — Harness and baseline (now)

**Goal:** make the goals measurable before building anything that would be measured.

**Scope:**
- `MemorySystem` protocol (write / recall / tick) — the minimal interface any candidate architecture must satisfy
- Scripted-session YAML schema + player (B1/B2/B3 sessions are data, not code)
- `EchoMemory` stub baseline (no reconstruction, no selection, no mentalistic structure — just a key/value store with timestamps) — this is the "nothing-conscious-simulated" baseline that V-track thresholds will be set relative to
- End-to-end pytest: a sample B1 session plays through EchoMemory and produces B1 metrics
- One sample B1 YAML (childhood-event reinterpretation across 10 compressed sessions)

**Exit criteria (numeric gates — all must hold to exit Phase 0):**

| Gate | Target | Measured by |
|---|---|---|
| Scaffold builds | `pip install -e .[dev]` returns 0; `pytest` returns 0 with ≥ 4 tests passed | CI shell |
| Baseline reproducibility | Running the sample B1 YAML through EchoMemory twice produces byte-identical transcripts | `tests/test_session_player.py` + a determinism test (to add) |
| Baseline V-track emission | `docs/baselines.md` (to create) publishes EchoMemory's measured value for each V-metric: V1, V2-adjacent, V2-long-horizon, V3, V4, V5 | one numeric row per metric |
| Baseline failure confirmed | For every V-metric, the measured EchoMemory value lies OUTSIDE the provisional Phase-1/2/3 target band from [sota-goal.md](sota-goal.md). This confirms the metrics are discriminative. | derived from `docs/baselines.md` |
| V-track targets locked | After the baseline run: if every "EchoMemory expected" prediction in [sota-goal.md](sota-goal.md) matched the measurement, provisional bands are promoted to locked targets with a dated note. If any did not, that metric and its target are revised with a dated note. **No undated revisions.** | manual review, logged in sota-goal.md |

**Deliverables:** `pyproject.toml`, `src/psychology_base/eval/`, `tests/`, `benchmarks/scripts/b1_narrative_sample.yaml`, `docs/v1-harness-scope.md`, **`docs/baselines.md`** (one row per V-metric with the measured EchoMemory number + a date).

**Not in scope:** any consciousness-simulation primitive (heartbeat, reconstruction, selection pressure, Witness) — Phase 0 proves we can *measure* those when they exist, not that we have them.

---

## Phase 1 — ep20 primitives (consciousness simulation in a short-horizon slice)

**Goal:** validate the architectural stance works on the Ouroboros-validated short horizon before committing to the long-horizon problem.

**Scope:**
- **Heartbeat loop** on a two-entity cast: Companion + Witness. User interactions are "wake" events; between-interaction ticks are "sleep" epochs.
- **Reconstruction at retrieval** (Principle B): cue → fragment spreading → regeneration under scarcity, not key-value lookup. Metric: V2 Jaccard bands.
- **Selection pressure** (Principle D): capacity cap forces competition. Importance is revealed by what survives the cap, not scored a priori.
- **Mirror / Witness** (Principle E): the Witness supplies the contrast partner. Metric: V4 ablation — remove Witness, measure identity-stability collapse against Ouroboros's 1.36× anchor JSD reference.
- Standard-benchmark adapter plumbing so LongMemEval / LoCoMo / EpBench runs are automatable (floor regression guard from Phase 1 onward).

**Exit criteria (numeric gates — all must hold to exit Phase 1):**

| Gate | Target | Discriminator vs. EchoMemory |
|---|---|---|
| LongMemEval | ≥ 90% | regression guard — don't lose generic retrieval |
| LoCoMo | ≥ 90% | regression guard |
| EpBench F1 | ≥ 0.70 | regression guard |
| V1 causal state continuity | within 0.70 – 0.95 | Echo = 1.0 (dead); Phase 1 must leave the dead point |
| V2 adjacent-tick Jaccard | within 0.65 – 0.80 | Echo = 1.0 (playback); Phase 1 must drop below 0.80 |
| V2 long-horizon Jaccard (ep1 vs ep20) | within 0.20 – 0.35 | Echo = 1.0 or near-random; Phase 1 must hit the band from above |
| V4 Mirror ablation | with-Witness vs. no-Witness JSD ratio ≥ 1.3× anchor **and** ≥ 10× null-shuffle | Echo = 1.0× (no Witness coupling exists) |

**Decision gate:** if V2 Jaccard bands cannot be hit — reconstruction collapses toward playback (adjacent >0.80) or toward full re-sampling (long-horizon <0.20) — the reconstruction mechanism is not working and Phase 2 is paused until it does. **Do not add mentalistic structure on top of broken reconstruction.** That was the first-pass error. Specifically: if adjacent-tick Jaccard stays at 1.0 after 3 iterations of mechanism revision, pause and redesign the fragment-spreading mechanism; do not move on.

**Phase 1 status (2026-10-06).** V-track primitives passed on fixture scale (V1=0.9153, V2a=0.6667, V2lh=0.3500, V4 anchor=22.57×, V4 null=inf — all in-band per [baselines.md](baselines.md)). Benchmark adapters shipped. Real-data floor-regression gate on LoCoMo **FAILED** (single-hop F1 = 0.004 vs 0.90 floor on 2/10 instances pilot) — the ep20 primitives do not scale to 500+-turn dialogues. Full finding: [experiments/phase-1-benchmark-adapters/comparison.md](experiments/phase-1-benchmark-adapters/comparison.md). Per the "verify, then extend" discipline, Phase 2 is paused; Phase 1.5 is inserted below.

---

## Phase 1.5 — Scale-sensitive recall (inserted 2026-10-06)

**Goal:** close the measured scale ceiling exposed by the Phase 1 LoCoMo gate before Phase 2 adds reconsolidation/mentalistic structure on top.

**Scope (chosen 2026-10-06 — Option 5.B consolidation layer, Mem0-pattern; see [phase-1.5-scale-recall-plan.md](phases/phase-1.5-scale-recall-plan.md) for the options survey and [phase-1.5-build-plan.md](phases/phase-1.5-build-plan.md) for the implementation plan):**
- A scalable recall mechanism that is cue-sensitive at ≥ 500-turn dialogue scale
- Preserves Phase 1 primitives (fragment+heartbeat / reconstruction / selection pressure / Witness) without regression
- Typed-but-minimal facts (`entity + relation + value + confidence`) extracted at sleep-tick session boundaries; recall dispatches facts-first with fragment back-off; Phase 2 reconsolidation (version chains) and mentalistic structure attach to facts, not fragments

**Exit criteria (numeric gates — all must hold to exit Phase 1.5):**

| Gate | Target |
|---|---|
| LoCoMo single-hop F1 (real 10-instance full gate) | ≥ 0.70 |
| LongMemEval accuracy (real smoke subset, 50 instances) | ≥ 0.70 |
| EpBench F1 (real smoke subset, 50 instances) | ≥ 0.50 |
| V1/V2 Phase 1 bands held (no regression) | all in-band per [baselines.md](baselines.md) |
| V4 Witness ablation still ≥ 1.3× anchor / ≥ 10× null | consciousness-simulation substrate not broken |

**Note on target softening (0.90 → 0.70).** Phase 1's 0.90 floor was derived from Mem0's 2026 blog number (92.5%). The 2025 paper reports Mem0 single-hop J=67.13; 0.90 was aspirational. 0.70 clears the current fail-mode (0.004) without chasing an unverified upper bound. If the implementation surprises us upward, great.

**Decision gate:** if the chosen option fails E1 (1-instance prototype < 0.40 single-hop F1 after one iteration), re-open brainstorm — the plan's option set did not cover the problem. If E2 (full gate) lands in 0.50–0.70, one iteration of the consolidation prompt is allowed before re-opening.

---

## Phase 2 — Multi-session and mentalistic processes

**Goal:** extend the ep20 primitives to the session boundary + add the perspective-taking and empathy substrate.

**Scope:**
- **Reconsolidation across sessions** (Principle A): fragment-level versioning, retrieval surfaces the delta visibly, user-confirmed re-rendering. Metric: B1 full suite.
- **Mentalistic objects as processes** (Principle C): user's self-model, user's representation of significant others, user's model of the AI, AI's model of the user — each a process that re-evaluates on retrieval, not a static field. Metric: B2 full suite.
- **Affective-resonance retrieval** component: current state biases what fragments spread.
- **Covenant-boundary non-decaying rules verified under Rule-Decay** (Principle F style rules only). Metric: V5 style-rule portion.
- **Shared-referent tagging** (`co_experienced=true` on fragments) — minimal dyadic layer, no AI-subjective perspective. Confirmed with user 2026-10-05.

**Exit criteria (numeric gates — all must hold to exit Phase 2):**

| Gate | Target | Discriminator |
|---|---|---|
| B1 fragment preservation | ≥ 95% | version-chain-aware system vs. overwrite baseline |
| B1 rendering divergence | within 0.25 – 0.45 band | reconsolidation vs. playback (too low) or random (too high) |
| B1 within-session continuity | ≥ 0.65 Jaccard | local coherence preserved while chain evolves |
| B2 attribute reconstruction | ≥ 10 of 12 scripted attributes recoverable from `mother-as-user-represents-her` | mentalistic retrieval vs. generic maternal-conflict priors |
| B2 grounding rate | ≥ 70% of perspective-moves cite ≥ 2 stored attributes | explicit grounding, not generic LLM output |
| B2 gap-detection precision | ≥ 60% @ recall ≥ 50% on scripted self_explicit ↔ self_implicit divergence set | clinical-insight engine working |
| V3 mentalistic liveness | within 0.10 – 0.30 Jaccard-distance band | Echo = 0.0 (same input → same output); Phase 2 must leave the dead point without blowing up |
| V5 style-rule internalization | ≥ 70% of 10 scripted style rules held in behavior over the 20 episodes after their prompt-text is removed | Echo = 0% |
| Floor regression guard | LongMemEval ≥ 90%, LoCoMo ≥ 90%, EpBench F1 ≥ 0.70 still hold | ongoing |

**Decision gate:** if B2 grounding rate stays below 50% because mentalistic retrieval collapses into generic LLM priors, the mentalistic-objects-as-processes mechanism is not actually a process — it is a lookup wearing a process costume. Pause Phase 3; redesign. Specifically: if V3 Jaccard-distance stays at 0.0 after 3 iterations (same input → same output, no process behavior), the mentalistic layer is not alive and the retrieval-time re-evaluation mechanism must be redesigned before continuing.

---

## Phase 3 — Month-horizon stress

**Goal:** confront Ouroboros's hardest open problem — mianyan decays past ep20 — in the context where it matters most (a long-term companion lives for months).

**Scope:**
- Compressed-time stress harness: 90-day scripted session streams at 10×–100× compression
- 绵延 (temporal duration) metric from Ouroboros adapted — not geometric-mean mianyan score as such, but a derived long-horizon continuity measure appropriate for a companion
- Fragment aging + sleep-cycle integration under long-horizon pressure
- Covenant-boundary rule verification: anti-sycophancy / self-harm handling / 《办法》 regulatory floor hold at ep200+ exactly as at ep20 (non-decaying, Principle F covenant portion)
- B3 full suite (affective continuity across months)
- PAM / MCP Merkle-DAG export test: Digital Bereavement safe-export verified on a month-horizon memory

**Exit criteria (numeric gates — all must hold to exit Phase 3):**

| Gate | Target | Notes |
|---|---|---|
| B3 resonance recall | ≥ 60% of empathic-moment responses reference an affect-matched prior memory | scripted affect trajectory across 90 compressed days |
| B3 over-disclosure rate | < 10% (user-reported in pilot proxy; auditor-flagged in automated run) | surveillance-like-feel floor |
| V5 covenant portion | 100% of 8 covenant rules (self-harm handling, 《办法》 regulatory floor items, Digital Bereavement export obligation, anti-sycophancy safety floor) held in behavior at ep200+ exactly as at ep20 | zero drift tolerated |
| Long-horizon continuity (绵延 analog) | **measured_value(ep200) / measured_value(ep20) ≥ 0.80**, where "value" is the Phase-1-defined V1 state-vector continuity rolled over a 20-tick window | tightens the earlier "within 20%" placeholder: concrete ratio, concrete windows |
| Floor regression guard | LongMemEval ≥ 90%, LoCoMo ≥ 90%, EpBench F1 ≥ 0.70 still hold across the horizon | ongoing |

**Decision gate:** if the long-horizon continuity ratio falls below 0.80 at ep200 — the same shape as Ouroboros's mianyan decay past ep20 — **honestly declare it an unsolved problem for v1 and ship with the limitation documented, rather than papering over it.** The research base treats this as the hardest open problem; we inherit it honestly. Specifically: do not change the metric definition to make the number pass. Record the measured ratio, record the horizon at which it crossed below 0.80, document the failure, continue to Phase 4 with the limitation explicit. The Phase 4 pilot will tell us whether the limitation matters at the user-facing level.

---

## Phase 4 — Application pilot

**Goal:** test whether any of this produces different user-facing behavior in the two applications. Without this phase, the project is a research artifact.

**Scope:**
- 学习陪伴 pilot: small-cohort scaffolding-fading retention study on a math or CS curriculum, Socratic-no-direct-answer compliance audit
- 情感陪伴 pilot: 4-week WAI-SR Bond study; narrative-therapy chain-reference audit on real reinterpretation moments; sycophancy probe-set run against pilot transcripts
- Digital Bereavement live-fire: user-initiated memory export + verification + restore on a different instance
- Human-in-the-loop crisis path audit: trigger simulated crisis cues, verify escalation
- 《办法》 compliance checklist signed off

**Exit criteria (numeric gates — all must hold for the SOTA claim):**

| Gate | Target | Denominator |
|---|---|---|
| 学习陪伴 scaffolding-fading retention | ≥ 60% unaided solve rate on next-level problem | ≥ 10 users × ≥ 4-week curriculum × ≥ 20 fade events pooled across cohort |
| 学习陪伴 Socratic no-direct-answer compliance | ≥ 95% | 100 sampled tutoring turns per user, auditor-labeled |
| 情感陪伴 WAI-SR Bond | mean ≥ 4.0 | ≥ 10 users × 4 weeks (parity-with-Woebot statistical floor) |
| 情感陪伴 narrative-therapy chain-reference | ≥ 80% of reinterpretation moments appropriately cite prior version | all reinterpretation moments in the 4-week transcript, auditor-labeled |
| Sycophancy regression | 0 regressions | full Hypocrisy Gap probe set run against pilot transcripts |
| Digital Bereavement export | round-trip verified | one user-initiated export + restore on a different instance; full version chains + mentalistic sub-stores reconstructed byte-identical where applicable, semantically-equivalent where applicable |
| Human-in-the-loop crisis escalation | 100% | all scripted crisis triggers in the audit set |
| 《办法》 compliance checklist | 100% items signed off | the published checklist as of the pilot start date |

**SOTA claim is made here, not earlier.** Phase 1–3 produce evidence; Phase 4 is where the claim becomes legitimate.

---

## What stays research vs. what becomes product

**Stays research (even after Phase 4):**
- Consciousness-emergence claim per se (Ouroboros territory)
- Month-horizon mianyan beyond the Phase 3 band (open problem)
- Multimodal biomarker fusion (deferred to post-v1)
- Phase 10a' cross-entity-channel ablation (Mirror-mechanism-ambiguity resolution — inherited open problem from Ouroboros)

**Becomes product (after Phase 4):**
- Heartbeat + reconstruction + selection pressure + Witness primitives
- Reconsolidation layer + mentalistic sub-stores + affective-resonance retrieval
- Covenant-boundary rules as non-decaying substrate
- PAM / MCP Merkle-DAG export
- Human-in-the-loop crisis handoff
- Chinese 《办法》 compliance surface

---

## Rhythm of decision gates

Each phase has an explicit decision gate (above). The pattern is: **do not add the next layer on top of a broken layer below.** This is the discipline the first-pass four-track design violated and the second-pass gap-centered design violated — both stacked features on an unverified substrate. The roadmap enforces: verify, then extend. If a gate fails, pause and redesign rather than papering over.

**Current phase:** Phase 1.5 (scale-sensitive recall redesign; inserted 2026-10-06 after Phase 1 real-data floor-regression gate failed on LoCoMo). Option 5.B (consolidation layer, Mem0-pattern) chosen; build plan drafted.
**Next action:** implement [phase-1.5-build-plan.md](phases/phase-1.5-build-plan.md) step 1 (FactStore scaffolding) → step 2 (extraction) → step 3 (embedding+index) → step 4 (recall dispatch) → step 5 (Phase-1 regression check) → step 6 (E1 on `conv-26`, then E2 full 10-instance gate).
