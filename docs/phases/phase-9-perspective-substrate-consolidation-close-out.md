# Phase 9+ Perspective substrate consolidation — close-out

**Status:** SHIPPED — strictly additive, zero regression across all Phase 2–8 gates.
**Date:** 2026-10-10
**Plan reference:** [phase-9-perspective-substrate-consolidation-plan.md](phase-9-perspective-substrate-consolidation-plan.md)

---

## 1. Verdict

CLAUDE.md Principle C — *mentalistic objects are processes, not fields* — now has a first-class home in code. Scattered routing primitives that evolved across Phases 2 → 8 to clear specific gates have been consolidated into one package without changing the semantics any of those gates depended on. Dual API surface shipped: `MentalisticObject.reconstruct(system)` as the primary surface for reconstructive queries, `CompanionSystem.recall(perspective=...)` as the low-level primitive for scorer dispatch and bespoke adapters.

- **Test count:** 1563 → **1593** (+30 new tests across 3 new test modules). Full suite green.
- **Zero regression** on any Phase 2–8 gate: Phase 2 M2 scope-filter, Phase 5 Track 1 FANToM +0.791 (preserved byte-identical through `PerspectiveSpec(scope=ScopeSpec(subject="subject", entity=target_person))`), Phase 7 TraitProfile OCEAN at N=50, Phase 8 Pillar E v2 HiToM order 2 +0.376, Pillar A/C/D/E close-outs unchanged.
- **Package shipped:** [src/psychology_base/companion/perspective/](../../src/psychology_base/companion/perspective/) with `routing.py`, `objects.py`, `grammars/hitom.py`.
- **Primary surface:** `TraitProfile.reconstruct(system)` is the exemplar. Future `MentalisticObject` implementations (BigToM situation-shift tracker, multi-party trust chains, user-model-of-mother) follow the same `@classmethod reconstruct(cls, system, context, *, perspective=None)` pattern.

---

## 2. Scope delivered

| Step | What landed | Tests added |
|---|---|---:|
| 0.0 | Fixture audit — regression pins subsumed by existing tests in-tree (no new scaffolding needed) | 0 |
| 1 | `perspective/routing.py` with `PerspectiveSpec` + `ScopeSpec`; `copresence.py` is permanent shim re-exporting from the new home | 14 |
| 2 | `CompanionSystem.recall(perspective=...)` kwarg with mutex contract vs legacy `scope=`/`copresence=`; Witness-pool filter extended identically; affect-weight orthogonality pinned | 10 |
| 3 | `perspective/objects.py` with `MentalisticObject` protocol (`@runtime_checkable`); `TraitProfile` carries `PERSPECTIVE` + `reconstruct()`; `.get(dim) == .get_estimate(dim)` identity shim | 6 |
| 4 | FANToM scorer dispatches via `perspective=PerspectiveSpec(scope=ScopeSpec(subject="subject", entity=target_person))`; +0.791 preserved byte-identical | (0 — existing 37 FANToM tests regression-pin the migration) |
| 5 | HiToM prompt builders migrated to `perspective/grammars/hitom.py`; scorer imports from new home; private-name shims (`_build_hitom_mentalistic_prompt`, etc.) retained for test-suite compatibility | (0 — existing 9 HiToM tests + 17 copresence tests regression-pin the migration) |

**Total new tests:** 30 (14 + 10 + 6). Steps 4 and 5 added no new tests because the existing test suites — 37 FANToM, 9 HiToM protocol-conformance, 17 copresence — already regression-pin the migration surface.

---

## 3. Design decisions landed

### 3.1 Dual API surface (both shipped, both load-bearing)

**Primary — `MentalisticObject.reconstruct(system, context)`.** Callers that want a reconstructed mentalistic object (OCEAN profile today; BigToM shift tracker, user-model-of-mother tomorrow) reach for the object's own `reconstruct()` classmethod. The object owns its default `PERSPECTIVE`, so callers don't need to know the correct scope.

**Low-level — `CompanionSystem.recall(perspective=...)`.** Scorer dispatch paths, bespoke adapters, direct per-turn queries pass a `PerspectiveSpec` directly. This is what the FANToM scorer uses after migration.

Both surfaces coexist permanently. Collapsing to one would force scorer paths through an object they don't want (FANToM scorer doesn't want a TraitProfile; it wants a scope-asymmetric recall) and force reconstructive callers to know internal scope shapes.

### 3.2 Legacy `scope=`/`copresence=` kwargs are permanent, not deprecated

Zero out-of-tree consumers; no churn benefit to deprecation. Legacy kwargs stay as internal shims that construct a `PerspectiveSpec` and dispatch through the unified path. **Both-at-once** (`perspective=` AND legacy kwarg) raises `ValueError` — same mutex contract as Pillar B / Pillar D CLI harnesses. Metadata emits both the structured `perspective` key and the legacy `scope`/`copresence` keys when applicable.

### 3.3 ScopeSpec requires BOTH subject AND entity set

The FANToM +0.791 mechanism depends on this: FANToM ingest sets `subject_id=speaker` → fragments carry `subject:X` but no `entity:*` tag → `ScopeSpec(subject="subject", entity=target_person).apply()` AND-fails → empty recall on `BELIEF_INACCESSIBLE` → agent answers "I don't know" → matches gold. Preserving the two-sided AND-contract preserves the mechanism.

The FANToM scorer migration passes the literal string `"subject"` as the subject field (`PerspectiveSpec(scope=ScopeSpec(subject="subject", entity=question.target_person))`) because the pre-migration code called `recall(scope=("subject", target_person))` with the literal `"subject"` string — byte-identity preservation requires that exact shape, not a semantic "subject-only" extension.

### 3.4 TraitProfile `PERSPECTIVE` is declarative-semantic marker, not enforcer

`TraitProfile._candidate_fragments(dim)` uses an extended tag filter (`subject:user` tag alone + `trait:<dim>:*` tags, no two-sided AND-contract) because trait fragments carry no `entity:*` tag. The `PERSPECTIVE: ClassVar[PerspectiveSpec] = PerspectiveSpec(scope=ScopeSpec(subject="user", entity="user"))` class-attribute documents the default perspective semantically; the concrete walker is `_candidate_fragments`. Both styles are legal under the `MentalisticObject` protocol; the protocol does not force one.

### 3.5 HiToM scorer stays pure-Python witness-filter, does not route through `recall()`

Plan Step 5.3 suggested the scorer dispatch via `perspective=PerspectiveSpec(copresence=CopresenceScope(chain=target_chain))` instead of direct `copresence=` kwarg. Inspection of the actual Pillar E v2 code revealed the scorer never went through `system.recall()` at all — it uses `_mentalistic_context_for()` which parses the HiToM story grammar directly and returns pre-rendered events. The `system_factory` kwarg is declared on the signature but intentionally not invoked (see `higher_order_tom_scorer.py` comment: *"mentalistic routing moved to a pure witness filter — the memory substrate is not on the Pillar E critical path"*).

Step 5 therefore reduced to: migrate the prompt builders to `grammars/hitom.py`, keep the pure-Python witness-event rendering in place. Future scorer rewrites that want to route HiToM through `system.recall()` have `PerspectiveSpec(copresence=CopresenceScope(chain=...))` available today — the plumbing is shipped — but the current scorer's pure-Python path is faster and byte-identical to the +0.376 gate.

### 3.6 Grammars live in `perspective/grammars/`, not alongside adapters

HiToM chain scaffolding + 15-container MC prompt builders are corpus-specific grammar. They don't belong on substrate (not reusable across corpora) and don't belong in the generic scorer (dispatch logic, not grammar logic). The `perspective/grammars/` subpackage is the shared home. Future `grammars/bigtom.py`, `grammars/d4.py` land here as they're built.

### 3.7 Private-name shims preserved on the scorer

`higher_order_tom_scorer.py` exposes `_build_hitom_mentalistic_prompt = build_hitom_perspective_prompt` (and three sibling aliases) so the existing 9 Pillar E v2 protocol-conformance tests at `tests/test_higher_order_tom_scorer.py` continue to import by their original names. The authoritative builders live in `grammars/hitom.py`; the shims are three lines of alias assignment. This is the Phase 2 shim pattern (`_filter_fragments_by_scope` thin delegate) — delegate-not-dead-code.

---

## 4. Critical files touched

**Created:**
- [src/psychology_base/companion/perspective/__init__.py](../../src/psychology_base/companion/perspective/__init__.py) — public API re-exports.
- [src/psychology_base/companion/perspective/routing.py](../../src/psychology_base/companion/perspective/routing.py) — `PerspectiveSpec`, `ScopeSpec`, `filter_fragments_by_scope`, re-exports from `copresence.py`.
- [src/psychology_base/companion/perspective/objects.py](../../src/psychology_base/companion/perspective/objects.py) — `MentalisticObject` protocol (`@runtime_checkable`).
- [src/psychology_base/companion/perspective/grammars/__init__.py](../../src/psychology_base/companion/perspective/grammars/__init__.py)
- [src/psychology_base/companion/perspective/grammars/hitom.py](../../src/psychology_base/companion/perspective/grammars/hitom.py) — HiToM chain scaffold + 15-container MC prompt builders.
- [tests/test_perspective_routing.py](../../tests/test_perspective_routing.py) — 14 tests.
- [tests/test_perspective_recall_api.py](../../tests/test_perspective_recall_api.py) — 10 tests.
- [tests/test_mentalistic_objects.py](../../tests/test_mentalistic_objects.py) — 6 tests.

**Modified:**
- [src/psychology_base/companion/system.py](../../src/psychology_base/companion/system.py) — `recall(perspective=...)` kwarg + mutex + Witness-pool filter extension. `_filter_fragments_by_scope` reduced to thin delegate shim calling `perspective.routing.filter_fragments_by_scope`.
- [src/psychology_base/applications/emotional/trait_profile.py](../../src/psychology_base/applications/emotional/trait_profile.py) — `PERSPECTIVE` class-attribute + `reconstruct()` classmethod + `get_estimate()` identity alias on `get()`.
- [src/psychology_base/eval/benchmarks/public_datasets/fantom_tom_scorer.py](../../src/psychology_base/eval/benchmarks/public_datasets/fantom_tom_scorer.py) — scorer dispatches `perspective=PerspectiveSpec(scope=ScopeSpec(subject="subject", entity=question.target_person))`.
- [src/psychology_base/eval/benchmarks/public_datasets/higher_order_tom_scorer.py](../../src/psychology_base/eval/benchmarks/public_datasets/higher_order_tom_scorer.py) — imports migrated prompt builders from `grammars.hitom`; private-name shims kept for test-suite compat.

**Not modified (invariant preservation):**
- Any adapter (`locomo.py`, `fantom.py`, `big5_chat_classification.py`, etc.) — perspective-filtering is recall-time.
- `fragment.py`, `reconstruction.py`, `witness.py` — data model + reconstruct() signature unchanged.
- Phase 2/5/7/8 close-outs — this phase preserves historical gates, does not re-litigate them.
- Any `benchmarks/scripts/phase*_*_run.py` CLI — JSON schema unchanged.
- `copresence.py` module stays in-place as permanent shim re-exporting from `perspective.routing`.

---

## 5. Regression evidence

**Full suite:** 1593 passed in 13.97s (vs 1563 pre-plan). +30 new tests. Zero existing tests failed at any step.

**Byte-identity pins verified:**
- Phase 2 M2 scope-filter — `tests/test_perspective_routing.py::test_scope_spec_byte_identical_to_legacy_filter` + existing `test_companion_phase1_step*.py` scope tests.
- Phase 5 Track 1 FANToM +0.791 — `tests/test_perspective_routing.py::test_scope_spec_subject_only_fragments_dropped_under_two_sided_scope` pins the AND-contract; existing 37 FANToM tests pass under migrated scorer.
- Phase 7 TraitProfile at N=50 — `tests/test_mentalistic_objects.py::test_get_and_get_estimate_are_the_same_method_result` pins the identity; existing Phase 7 B4 fixture tests pass.
- Phase 8 Pillar E v2 HiToM order 2 — existing 9 `test_higher_order_tom_scorer.py` protocol-conformance tests pass against migrated builders; existing 17 `test_copresence.py` + 15 `test_hitom_witness.py` tests regression-pin the witness primitive.
- Witness-pool filter — `tests/test_perspective_recall_api.py` includes a test that pain-signal pool narrows under the same perspective as the main pool.

---

## 6. What this unlocks

- **BigToM situation-shift tracker as future `MentalisticObject`.** Declarative-paragraph grammar lands at `grammars/bigtom.py`; the reconstructive object implements `reconstruct()` and holds its `PERSPECTIVE` the same way `TraitProfile` does. No scorer-architecture change required.
- **Multi-scene re-entry across sessions.** When `event_time:ISO8601` tagging arrives, a new sub-spec (`EventWindowSpec`) lands as a sibling field on `PerspectiveSpec` without breaking any current caller. The composition pattern (frozen dataclass + `apply()` method + set-intersection) accommodates it.
- **User-model-of-mother / user-model-of-AI when Phase 2 B2 ships as code.** `UserRepresentationOf(subject, entity)` implements the `MentalisticObject` protocol with its own `PERSPECTIVE: ClassVar[PerspectiveSpec] = PerspectiveSpec(scope=ScopeSpec(subject="user", entity="mother"))`. No new primitive needed.
- **Scorer-local witness filters generalize.** The HiToM witness-filter pattern (parse corpus grammar → render events → render under PerspectiveSpec) is now reproducible from `grammars/<corpus>.py` without a bespoke scorer module per corpus.

---

## 7. Carried debt (post-Phase-9+)

- **HiToM scorer still bypasses `system.recall()`.** The pure-Python witness-event rendering is faster and byte-identical to the +0.376 gate, but a future rewrite that routes HiToM through `system.recall(perspective=PerspectiveSpec(copresence=CopresenceScope(chain=...)))` would exercise the memory substrate on the Pillar E critical path. The plumbing is shipped; the migration is deferred.
- **FANToM scorer passes the literal string `"subject"` as subject.** Byte-identity preservation. If a future adapter changes FANToM ingest to tag fragments with an actual subject identifier, the scorer dispatch updates accordingly. The ScopeSpec AND-contract preserves the mechanism either way.
- **BigToM declarative-paragraph grammar not yet migrated to `grammars/`.** Current BigToM path uses the full-rendered-context fallback in `_mentalistic_context_for` — no architectural win, no regression. Shipping a BigToM grammar lands when the gate motivates it.
- **TraitProfile's `_candidate_fragments` extended-filter semantics not formalized on the protocol.** `MentalisticObject.PERSPECTIVE` is declarative; the concrete walker is free-form. Two legal styles coexist (strict two-sided AND-contract vs TraitProfile's subject-tag + dim-tag). Formalizing this as a `PerspectiveSpec.extend()` sub-spec is YAGNI until a second concrete object needs the extended semantics.

---

## 8. Pointers for the next phase

If the next phase is **a new mentalistic object** (BigToM, user-model-of-X):
1. Add `grammars/<corpus>.py` for corpus-specific prompt builders.
2. Implement `MentalisticObject` on the new class: declare `PERSPECTIVE`, implement `@classmethod reconstruct(cls, system, context, *, perspective=None)`.
3. Scorer dispatches via `TheObject.reconstruct(system)` or `system.recall(perspective=TheObject.PERSPECTIVE)` depending on whether it needs the reconstructed object or raw filtered fragments.
4. Follow Phase 7 TraitProfile as the exemplar.

If the next phase is **a new sub-spec** (event-window, trust-chain, narrative-frame):
1. Add the sub-spec dataclass to `routing.py`.
2. Extend `PerspectiveSpec` with a new optional field.
3. Extend `PerspectiveSpec.apply()` to compose the new predicate as set-intersection.
4. All existing callers are unaffected (optional-None default).

**Project state:** v1 close-out extends Phase 8 → Phase 9+; no next-phase memo absent new gate authorization.
