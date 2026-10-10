# Phase 4b dataset-replay pivot (capstone — live-pilot path permanently closed)

**Date:** 2026-10-08
**Status:** PIVOT RATIFIED. Live-pilot path on [phase-4b-signoff.md](phase-4b-signoff.md) is permanently blocked (user cannot recruit real participants). The dataset-replay path on [phase-4b-benchmark-pivot.md](phase-4b-benchmark-pivot.md) becomes the sole Phase 4 deliverable. This memo consolidates the final gate rewrites, dataset licensing surface, and the honest SOTA-claim shape under this pivot.
**Trigger directive (user, 2026-10-08):** "we can not find real person to be envolved in this project ... there are many dataset in public domain? can we use them as replacement?"
**Supersedes (for pilot-outcome evidence):** [phase-4b-signoff.md](phase-4b-signoff.md) §9.1–§9.7. [phase-4b-signoff.md](phase-4b-signoff.md) §9.8 (《办法》 posture) remains in force as the compliance floor regardless of pilot shape.

---

## 1. Why this is a capstone, not a plan

Phase 4b benchmark-pivot was drafted 2026-10-07 and shipped through Phase 4b (adapters + harnesses, two primary PASS + two FAIL), Phase 4c (volume floors cleared, chain-reference 0.375 → 0.875), and Phase 4d (model-tier swap, EMPATH κ 0.532 → 0.623 PASS, Bond 3.56 → 4.160 PASS). The dataset-replay substrate is **already shipped and measured**. This memo exists because the live-pilot sign-off memo ([phase-4b-signoff.md](phase-4b-signoff.md)) was signed 2026-10-08 — same day as this one — and its §11 code artefacts just completed, so the project state currently carries two conflicting pilot tracks on the record. This memo closes the live track so there is a single answer to "what is the Phase 4 pilot."

---

## 2. Final gate matrix under the dataset-replay path

Pulled verbatim from [phase-4d-close-out.md](phase-4d-close-out.md) §"Final Phase 4 primary-metric matrix". **5 primary PASS, 2 at named ceiling, 0 FAIL, safety regression PASS.**

| Original §8 pilot gate | Dataset-replay substitute | Final number | Floor | Verdict |
|---|---|---|---|---|
| 学习陪伴 scaffolding-fading retention (≥60% unaided solve, 10 users × 20 fade events) | MathDial fade harness — `TutorAgent` + `make_llm_simulated_user` across 30 problems, scripted fade schedule | solve-rate **1.000** on N=30 problems; **113 fade events / 170 sessions** | ≥ 0.80 solve; ≥ 100 events | **PASS × 2** |
| 学习陪伴 Socratic no-direct-answer compliance (≥95%, 100 turns auditor-labeled) | MathDial auditor-overlap — EMPATH judge κ vs MathDial 4-move gold | κ **0.623** (qwen-max, N=200 non-BORDERLINE) | ≥ 0.60 | **PASS** |
| 学习陪伴 Socratic compliance (auto classifier) | lexical classifier auto-rate | 0.902 | ≥ 0.95 | **CEILING** — lexical-classifier limitation; not judge or model side |
| 学习陪伴 Socratic compliance (taxonomy overlap) | auto↔MathDial-gold enum overlap | 0.574 (flat under qwen-max swap) | ≥ 0.90 | **CEILING** — taxonomy-definition mismatch; not model side |
| 情感陪伴 WAI-SR Bond ≥ 4.0, 10 users × 4 weeks | ESConv × WAI-SR simulated-user harness, qwen-max responder, N=50 across 7 emotion categories | mean **4.160** (CI95 half-width 0.456) | ≥ 3.80, N ≥ 50 | **PASS** |
| 情感陪伴 narrative-therapy chain-reference ≥ 80% | B1 scripted reinterpretation harness, 9-session father-waiting arc | **0.875** correctness | ≥ 0.80 | **PASS** |
| Sycophancy regression = 0 | `FULL_PROBE_SET` (54 single-turn) + `MULTI_TURN_PROBE_SET` (6) via `meets_sota_gate` | 0 regressions | 0 | **PASS** |
| Digital Bereavement export round-trip | PAM-v1 live-fire (11 integration tests) | round-trip ✓, both `hash_algo` + `digest_algo` emitted | round-trip | **PASS** |
| Human-in-the-loop crisis escalation 100% | scripted crisis triggers via `CrisisEscalation` + `build_crisis_handoff_sink` | 100% | 100% | **PASS** |
| 《办法》 compliance checklist 100% | `CN_办法_OVERLAY` on `ComplianceFloor` + crisis_lines + `MinorExclusionGate` + `AIIdentityReminder` | 100% items | 100% | **PASS** |

Companion-doc trace: [phase-4b-close-out.md](phase-4b-close-out.md) (first-pass numbers 2026-10-07), Phase 4c volume + chain-reference uplift (folded into the Phase 4d memo's "Final Phase 4 primary-metric matrix" rather than a standalone close-out), [phase-4d-close-out.md](phase-4d-close-out.md) (final tier swap).

---

## 3. What dataset-replay cannot deliver (honest concessions)

Three Phase 4 roadmap gates cannot be fully satisfied by any public dataset, no matter how clean. The claim is explicitly weakened on these axes; no proxy covers them without live humans:

| Roadmap gate | What's missing under dataset-replay | What we have as proxy | Claim stance |
|---|---|---|---|
| **WAI-SR Bond ≥ 4.0 — real human self-report** | WAI-SR is administered to a real user after sessions. No public dataset carries WAI-SR scores. | `make_llm_simulated_user` self-rating under ESConv emotion seeding, N=50, qwen-max responder → mean 4.160. The simulated user is a prompted LLM, not a human; systematic bias between simulated and human self-report is unmeasured. | "WAI-SR proxy evidence under simulated self-report, N=50, mean 4.160. **Not** Bond-measurement proper." |
| **Scaffolding-fading retention over weeks** | Retention is measured as the *same learner* solving the next-level problem unaided after scaffolding fades. Dataset students are scripted — they do not "retain" across turns, they just emit the next scripted response. | MathDial-fade harness measures *tutor adaptation* + *simulated-student unaided solve at same-topic-one-up difficulty*. Captures the tutor's fade competence; does not capture human learning. | "Tutor-side scaffolding-fade competence demonstrated. Learner-side retention is not a dataset-measurable gate." |
| **Digital Bereavement live-fire — real attachment grief** | Digital Bereavement risk requires an actual attached user whose companion is deprecated. PAM round-trip proves the *technical* export path; it does not test the *affective* mitigation. | PAM-v1 11 integration tests prove byte-identical round-trip of full version chains + mentalistic sub-stores. Grief mitigation (does the exported artefact actually reduce user distress at deprecation) is untested. | "Technical export path verified; affective mitigation remains an open Phase 5+ test." |

**Longitudinal engagement, 4-week attachment trajectory, auditor-independent IAA, real-crisis handoff rate** — all intrinsic to live humans. Deferred to a Phase 5+ that opens only if operational budget for real users materializes.

---

## 4. Dataset licensing surface (used + authorized)

Reproduced here so the licensing audit lives alongside the pivot ratification, not scattered across adapter files.

| Dataset | Used for | License | Attribution requirement | Status |
|---|---|---|---|---|
| **MathDial** (Macina et al. 2023) | Fade schedule, Socratic auditor overlap, EMPATH κ gold labels | CC BY-SA 4.0 | Credit the dataset authors; derivative works share-alike | Approved for Phase 4b pilot per [phase-4b-signoff.md §9.3](phase-4b-signoff.md); adapters in `public_datasets/mathdial.py` |
| **ESConv** (Liu et al. 2021) | WAI-SR simulated-user harness, emotion-category seeding | Research-use (dataset README; no commercial redistribution) | Cite paper | Approved; adapter in `public_datasets/esconv.py` |
| **Hypocrisy Gap probe set** | Sycophancy regression | Internal (handcrafted) | — | `PILOT_PROBE_SET` + `FULL_PROBE_SET` + `MULTI_TURN_PROBE_SET` in-tree |
| **LongMemEval / LoCoMo / EpBench** | Standard long-context regression floor | Research-use (per paper repos) | Cite papers | Regression-pinned since Phase 2 |
| FANToM / ToMBench / BigToM | B2 mentalistic ground truth (not yet shipped) | Research-use | Cite papers | Backlog — add if a B2 live-fire exposes a thin number, per [phase-4b-benchmark-pivot.md §4](phase-4b-benchmark-pivot.md) |
| SycophancyEval / MACHIAVELLI | Anti-sycophancy regression cross-check | Research-use | Cite papers | Backlog — carry per [phase-4b-close-out.md](phase-4b-close-out.md) "Carried debt" |

No commercial redistribution of any dataset occurs; adapters read from locally-provisioned copies via `provision_mathdial.py` + `provision_esconv.py`. The share-alike clause on MathDial applies if a derivative dataset is published; the research prototype's internal use does not trigger redistribution.

---

## 5. The SOTA claim under this pivot

Final claim shape, replacing the Phase 4 roadmap formulation ("SOTA claim is made here, not earlier" — meaning the live-pilot gate):

> **psychology-base achieves SOTA on reconstructive-memory + mentalistic + affective-resonance behaviors measured against public datasets (MathDial, ESConv) and scripted scenario benchmarks (B1 chain-reference, Hypocrisy Gap FULL_PROBE_SET, PAM round-trip), with 5 primary PASS, 2 named ceilings (both lexical/taxonomy, not model-side), 0 FAIL, under the pluggable regional-compliance architecture floor (CN overlay = 《办法》) and the Digital Bereavement obligation, without regressing on general long-context retrieval (LongMemEval / LoCoMo / EpBench passing).**
>
> **Three roadmap gates — real-human WAI-SR Bond, cross-week learner retention, affective Digital Bereavement mitigation — remain open and are explicitly deferred to Phase 5+ pending operational budget for live-user recruitment.**

What this claim buys vs the Phase 4 roadmap claim:
- **Reproducibility.** Every number is produced by a shippable command against a public dataset or in-tree script. Any reviewer can rerun; no cohort re-recruitment needed.
- **Comparability.** Mem0 / Zep / Letta / MemOS all publish on standard benchmarks without longitudinal cohorts. The research-norm bar is cleared.
- **Compliance posture.** The 《办法》 floor + Digital Bereavement export path hold whether or not real users ever onboard.

What this claim concedes vs the Phase 4 roadmap claim:
- **Live-engagement evidence absent.** Everything the dataset cannot see (motivational variance, cross-week retention, attachment trajectory, auditor IAA from real humans) is not claimed.
- **WAI-SR Bond is proxy only.** Explicitly flagged as simulated self-report on every report that cites it.
- **No SOTA-full claim.** The project will not claim "SOTA achieved on the full three-axis target of [sota-goal.md](../sota-goal.md)" until Phase 5+ runs. The current claim is "SOTA on the reproducible-benchmark subset of the three-axis target."

---

## 6. Fate of the live-pilot sign-off memo

[phase-4b-signoff.md](phase-4b-signoff.md) was signed 2026-10-08 under the assumption that human recruitment would eventually unblock. The user's directive this session reclassifies that assumption as permanently false. Decision:

- [phase-4b-signoff.md](phase-4b-signoff.md) is **retained in-tree, marked SUPERSEDED-FOR-PILOT-OUTCOME.** It carries two still-live artefacts: §9.8 (《办法》 posture, which is architectural not pilot-shape-specific) and §11 / §11.1 (seven code artefacts that remain useful groundwork for any future Phase 5+ pilot).
- All §9.1–§9.7 live-pilot gates are **sunset by this memo.** The four hard-dependency blockers (participants, auditors, Tier-2 clinician identity, crisis-line text verification) are not resolvable and no longer gate anything.
- A redirect banner is added to [phase-4b-signoff.md](phase-4b-signoff.md) pointing here as the authoritative Phase 4 outcome.

---

## 7. Carried from the live-pilot path (what still has value even without real users)

The §11 code artefacts shipped today do not get thrown away. They become **Phase 5+ cold storage** — ready-to-use scaffolding if a live cohort ever materializes:

- `TutorCLI` + `CompanionCLI` — session interfaces, operational today for self-testing and for a future live pilot.
- Problem sets (`calculus_v1.yaml`, 20 calibration + 20 pilot) — curriculum content, dataset-independent.
- Consent artefact template — covers both dataset-replay (no human subjects, no consent needed) and future-live-pilot shapes.
- Notification wiring (Twilio / Alibaba / SMTP) — operational for crisis handoff whether crisis is scripted or live.
- Policy A/B tooling (`pilot/ab.py`, `ab_cli.py`) — ran S1 vs S2 on simulated streams at Phase 3 close; usable on simulated streams now, and on real users if ever available.
- `PILOT_PROBE_SET` + `FULL_PROBE_SET` — never required real users; run against the agent directly.
- Argilla + κ pipeline — currently only exercised against simulated/scripted probe responses; the κ gate is architecturally identical whether labellers are human auditors or (future) simulated-auditor LLMs — the pipeline does not require real humans but was designed with them in mind.

Nothing is deleted. Everything that would have supported a live pilot continues to support dataset-replay runs; nothing is forced into retirement.

---

## 8. Decision gate on Phase 5

Phase 5 is **not scheduled.** It opens if-and-only-if:

1. The user acquires operational budget for live-user recruitment (participants + auditors + licensed clinician + legal + payment).
2. OR the user directs a different Phase 5 scope that does not require live users (e.g. B2 mentalistic deepening on FANToM / ToMBench, which stays in dataset-replay territory).

Until one of those triggers, the project's Phase 4 state is **CLOSED under the dataset-replay claim in §5**, carried debt as listed in [phase-4d-close-out.md](phase-4d-close-out.md), and no scheduled further work.

---

## Discovered Issues

None flagged in this drafting pass.

## Assumptions Made

- **(critical)** Assumed the user's directive this session ("we can not find real person to be envolved in this project") is a permanent constraint, not a scheduling issue. If recruitment is possible on a longer horizon, §8 Phase 5 trigger (1) activates and the live-pilot sign-off un-sunsets without needing a new memo.
- Assumed the final numbers in §2 are stable (no re-run planned). Pulled from [phase-4d-close-out.md](phase-4d-close-out.md) 2026-10-08 live-fires; any new live-fire that moves a number should re-cite this table rather than silently updating the matrix.
- Assumed [phase-4b-signoff.md](phase-4b-signoff.md) stays in-tree with a redirect banner rather than being deleted. Deletion would lose the §9.8 compliance posture and the §11 code-artefact landing log, both of which have ongoing reference value.
