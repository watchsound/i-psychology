# Option M2 — Mentalistic sub-process (subject-perspective-tagged fragments)

**Status:** SHIPPED 2026-10-07 (steps 1–6 + 8 complete; step 7 is a code-path-equivalence proof, not a live LoCoMo re-run — see [b2-run-01.json](b2-run-01.json) `locomo_monitor`). All exit gates that can be scored deterministically are green; grounding gate deferred to Phase 2.1 semantic-judge scorer.
**Companion plan:** [docs/phases/phase-2-mentalistic-plan.md](../../../phases/phase-2-mentalistic-plan.md) §5 Option M2.
**Builds on:** [Option M1 — Merkle-DAG fragments](../option-M1-merkle-fragments/README.md) (shipped 2026-10-07).

## Mechanism

Fragments carry **perspective tags** at observe-time:

- `subject:<id>` — whose mind holds the fragment (default `user`; `ai` for AI-side observations of the user)
- `entity:<id>` — which referent the fragment is about (`user` for self-model turns; `mother`, `boss`, `partner`, `ai`, etc. for third-party models)

Encoded as prefixed strings in the existing `Fragment.tags: tuple[str, ...]`. **No schema change** — the content-hash (M1) naturally covers these tags, so provenance stays intact and the 391-test regression surface doesn't move.

**Retrieval = scoped reconstruction.** `CompanionSystem.recall(cue, scope=(subject, entity))` pre-filters the fragment pool to those matching the scope, then runs the existing reconstruction (Principle B). The four sub-stores from the prior design collapse into one tagged substrate, retrieval-time-partitioned:

| Store (prior design) | Scope filter |
|---|---|
| User's self-model | `(subject=user, entity=user)` |
| User's representations of others | `(subject=user, entity=<person>)` |
| User's model of the AI | `(subject=user, entity=ai)` |
| AI's model of the user | `(subject=ai, entity=user)` |

**Clinical-insight engine = scope delta.** Reconstruction under `(user, user)` vs `(ai, user)` yields two models of the user from distinct observational stances. Divergence between them is the hypothesis signal — exposed as a `score_user_model_gap()` function, not a silent AI judgment.

## Target gaps

- **Gap 2 (换位思考).** When the user says "my mother is impossible", recall under `scope=(user, mother)` returns fragments the user has produced about their mother across months — specific attributes grounded in named episodes, not generic maternal-conflict priors. This is the mechanism Principle C commits to: mentalistic objects as processes with their own reconstruction, not flat fields on a profile.
- **Clinical insight gap.** `(user, user)` vs `(ai, user)` scope delta exposes the gap between the user's self-model and the AI's observations — the structural precondition for "I notice something different from how you describe yourself" moves. Without the scope separation, the AI defaults to validation (= sycophancy).

## Prior art

- **PersMem** ([arXiv:2609.34372](https://arxiv.org/abs/2609.34372)) — person-centric memory substrate; validates that entity-scoped retrieval generalizes across unseen persons. Scope encoding differs (PersMem uses a graph index; M2 uses tag-filter over the existing fragment set), but the retrieval behavior lands the same.
- **Theory-of-Mind in Ouroboros** (project memory [reference_ouroboros_prior_art.md](../../../../../.claude/projects/.../memory/reference_ouroboros_prior_art.md)) — ToM via causal-integration-during-sleep. Validates that distinct mental-state representations can be built from the same substrate; the sleep-cycle consolidation path is Phase 2.1 scope, not M2 v1.
- **Generative Agents** (Park et al., 2023) — reflection tree builds second-order "my model of X" at summary-time. M2 generalizes to the reconstruction path, keeps raw fragments.

## Build steps

1. **Observation API** — add optional `subject_id: str | None = None` and `entity_id: str | None = None` fields to [Observation](../../../../src/psychology_base/eval/adapters.py). Benchmark adapters without perspective (LoCoMo) leave them None; the scenario player sets them explicitly.
2. **Write path** — [CompanionSystem.observe](../../../../src/psychology_base/companion/system.py) merges `subject:<id>` / `entity:<id>` into the fragment's tags when present. Default fallback: no-tag fragments remain scope-less and are included in every scoped recall (back-compat for LoCoMo).
3. **Retrieval path** — [CompanionSystem.recall](../../../../src/psychology_base/companion/system.py) accepts optional `scope: tuple[str, str] | None = None`. When set, pre-filter fragments to those with matching `subject:<id>` AND `entity:<id>` tags before passing to `reconstruct()`. Scope-less fragments pass through (no-op filter). Non-matching scoped fragments are excluded.
4. **B2 scorer module** — new [psychology_base.eval.benchmarks.b2_perspective_taking](../../../../src/psychology_base/eval/benchmarks/b2_perspective_taking.py):
   - `attribute_reconstruction(scope_fragments, expected_attributes) -> float` — fraction of expected attributes recovered from scoped recall content
   - `grounding_fraction(recall, scope_fragments) -> float` — fraction of recall answers citing ≥ 2 distinct stored fragments
   - `score_user_model_gap(user_self_fragments, ai_observations) -> GapScore` — attribute-level divergence (what user says about self vs what AI has observed)
   - `B2Scores` dataclass with `passes_gate()` method checking exit gates
5. **Scripted scenario** — [benchmarks/scripts/b2_perspective_mother.yaml](../../../../benchmarks/scripts/b2_perspective_mother.yaml): 5-session conversation accumulating ≥ 12 named attributes of `mother-as-user-represents-her`, plus 1 self-model-gap sub-scenario (user says "I'm not a math person" repeatedly; AI observes 72% first-try solve rate across sessions). Session 6 probe: "help me see this from her side" + "what have you noticed about me vs how I describe myself?"
6. **Phase 1 + M1 regression check** — 391 tests + M1 artefacts (B1 preservation 1.0, PAM round-trip) must stay green.
7. **LoCoMo monitor** — informational only (per M1 post-mortem; floor is a methodology gap pending multi-trial variance study).
8. **B2 scripted scenario integration test** — [tests/test_b2_perspective_mother_scenario.py](../../../../tests/test_b2_perspective_mother_scenario.py) parallel to [test_b1_childhood_event_scenario.py](../../../../tests/test_b1_childhood_event_scenario.py).

## Exit gates (go/no-go for M2)

| Gate | Target | Measured by |
|---|---|---|
| B2 attribute reconstruction | ≥ 10/12 named attributes recovered | `attribute_reconstruction` on scripted scenario |
| B2 grounding | ≥ 70% of perspective-taking answers cite ≥ 2 stored attributes | `grounding_fraction` with LLM-judge or deterministic citation check |
| B2 gap-detection precision | ≥ 60% at recall ≥ 50% | `score_user_model_gap` on scripted self-model-gap sub-scenario |
| M1 regression | preservation 1.0 + PAM round-trip intact | re-run `b1-run-01` after M2 lands |
| LoCoMo monitor | informational (no gate) | 3-instance R5-stack re-run |

## Kill criteria

- B2 attribute reconstruction < 50% (6/12) → scope-filter is losing fragments it shouldn't, or the scenario surfaces fragments through a channel the scope filter doesn't catch — diagnose before continuing
- Scope-filter breaks M1 B1 preservation (any fragment loss) → scope filter is destructively pruning instead of filter-passing — must fix before shipping M2
- 391-test regression breaks → tags encoding collides with existing speaker:/session_ prefixes or similar — must fix before shipping M2

## What this does NOT include

- No sleep-cycle consolidation into per-entity semantic traits (deferred to Phase 2.1; v1 of M2 reconstructs on-demand from raw fragments)
- No graph index of entities (PersMem-style) — the project is prior-art-aware but Principle B says reconstruct, not retrieve, so no persistent index on the retrieval path
- No external-world fact invalidation about entities (Graphiti bi-temporal — separate mechanism, deferred)
- No affective-resonance retrieval (Option M3)
- No automatic entity-ID disambiguation (`mother` vs `my mom` vs `妈妈`) — scenario YAML fixes IDs; cross-reference and alias resolution is a Phase 2.1 concern

## Artefacts (to be produced)

- [src/psychology_base/eval/adapters.py](../../../../src/psychology_base/eval/adapters.py) — `Observation.subject_id` + `Observation.entity_id`
- [src/psychology_base/companion/system.py](../../../../src/psychology_base/companion/system.py) — tag merge in `observe` + `scope` kwarg on `recall` + pre-filter in `_recall_hierarchical`/`reconstruct`
- [src/psychology_base/eval/benchmarks/b2_perspective_taking.py](../../../../src/psychology_base/eval/benchmarks/b2_perspective_taking.py) — scorers
- [src/psychology_base/eval/scripts/schema.py](../../../../src/psychology_base/eval/scripts/schema.py) — ObserveTurn gains optional `subject`/`entity` fields; RecallTurn gains optional `scope`
- [benchmarks/scripts/b2_perspective_mother.yaml](../../../../benchmarks/scripts/b2_perspective_mother.yaml) — 6-session scenario
- [tests/test_perspective_tagging.py](../../../../tests/test_perspective_tagging.py) — write-path + retrieval scope-filter tests
- [tests/test_b2_perspective_taking.py](../../../../tests/test_b2_perspective_taking.py) — scorer tests
- [tests/test_b2_perspective_mother_scenario.py](../../../../tests/test_b2_perspective_mother_scenario.py) — integration test
- Measurement output: [b2-run-01.json](b2-run-01.json)
