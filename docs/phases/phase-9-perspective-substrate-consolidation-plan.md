# Plan — Phase 9+ Perspective substrate consolidation (换位思考 as first-class code)

## Context

Phase 8 locked final at 3-of-6 on 2026-10-09. Across Phases 2 → 8, mentalistic routing — the code-level expression of CLAUDE.md Principle C ("mentalistic objects are processes, not fields") — has accumulated across four files, each serving a specific gate:

- `src/psychology_base/companion/system.py::_filter_fragments_by_scope` — Phase 2 M2 primitive, AND-contracts `subject:X ∧ entity:X` on fragment tags. Pass-through on scope-less fragments. **Accidentally load-bearing for Phase 5 Track 1 FANToM +0.791**: FANToM ingests with `subject_id=speaker` → fragments carry `subject:X` but no `entity:X` → scoped recall AND-fails → empty recall on `BELIEF_INACCESSIBLE` → agent answers "I don't know" → matches gold. Scope-asymmetric mechanism is the FANToM Track 1 architectural result.
- `src/psychology_base/companion/copresence.py::CopresenceScope + filter_fragments_by_copresence` — Phase 9+ substrate witness primitive. Keep iff `{witness:A1, …, witness:AN}.issubset(fragment.tags)`; pass-through on fragments without any `witness:` tag. Shipped 2026-10-09 after Phase 8 Pillar E v2.
- `src/psychology_base/eval/benchmarks/public_datasets/higher_order_tom_scorer.py:235-338` — HiToM-specific chain scaffolding and prompt builders dispatched on `metadata.source == "hitom"`. Corpus-specific grammar; **does not belong on substrate** but currently has no shared home with other perspective-shaped primitives.
- `src/psychology_base/applications/emotional/trait_profile.py::TraitProfile` — Phase 7 OCEAN reconstruction. Walks `trait:*` tagged fragments at `.get(dim)` time; `_candidate_fragments(dim)` is Principle-C "mentalistic object as process" realized for one specific mentalistic object (user's trait profile).

**The architectural problem.** Each piece evolved to clear a specific gate. The distributed state means:
1. Discovery cost is high — a new engineer walking in has four files to read before they understand how 换位思考 is implemented.
2. New mentalistic objects (BigToM situation-shift tracker, multi-party trust chains) have no obvious home — they'd either land as another sliver or get wedged into `TraitProfile`'s shape.
3. The FANToM scope-asymmetric AND-contract is a mechanism, not a bug — but it's documented only in project memory, not as a first-class primitive. Future refactors risk regressing +0.791 by "cleaning up" the asymmetry.
4. HiToM chain scaffolding and HiToM witness-tagged ingester live separately. Their joint invariant (witness-filtered recall ↔ chain-scaffolded prompt) is enforced by a parity test, not a shared interface.

**The directive from the user (2026-10-09): "our goal is SOTA, so the solution should be complete."** Not a namespace cleanup. Not a strictly-additive migration. The complete expression of Principle C in code — one module that owns mentalistic routing primitives, one protocol that mentalistic objects implement, one API surface on `CompanionSystem.recall()`, with `MentalisticObject.reconstruct()` promoted as the primary surface callers reach for.

**Intended outcome:**
- Principle C expressed in code as `MentalisticObject` protocol + `PerspectiveSpec` composable predicate.
- Scattered pieces consolidated into one `src/psychology_base/companion/perspective/` package.
- Phase 5 Track 1 FANToM +0.791 preserved as an explicit scope-asymmetric mechanism, not accidental emergent behavior.
- Zero regression on any Phase 2–8 gate.
- `TraitProfile` is the exemplar `MentalisticObject`; future reconstructive objects follow the same pattern.

**Non-goals (explicitly out of scope for this plan):**
- Rebuilding the hierarchical fact store (`_hierarchical_fact_store` stays as-is; the fact-tier suppression logic composes with new perspective filtering as before).
- Changing cue-similarity ranking or reconstruct() semantics.
- Touching Witness entity coupling (`witness_active=True` narrowing stays composable upstream of PerspectiveSpec).
- Changing LLMRouter / scorer schemas / adapter contracts.
- Building the BigToM situation-shift grammar (this plan unblocks it as a future `MentalisticObject` implementation; grammar itself is follow-on work).
- Live-user cohort Axis 3 measurement (permanently out of scope per 2026-10-08 no-recruitment lock).
- Building `UserRepresentationOf` / `UserModelOfAI` / `UserSelfModel` concrete classes. Phase 2 B2 was planned but never shipped as code; building these ahead of a gate that exercises them is YAGNI. The protocol surface accommodates them when a gate arrives.
- Building `EventWindowSpec` / `TrustChainScope` sub-specs. No current gate exercises either; shipping untested primitives is YAGNI. The `PerspectiveSpec` composition pattern accommodates adding sub-specs when a gate arrives.

---

## Design — perspective package, MentalisticObject protocol, dual API surface

**API emphasis.** Two surfaces coexist:
- **Primary** — `MentalisticObject.reconstruct(system, context)`: callers that want a reconstructed mentalistic object (OCEAN profile, future situation-shift tracker, future user-model-of-mother) reach for the object's own `reconstruct()` classmethod. The object owns its default `PERSPECTIVE`, so the caller doesn't have to know the correct scope.
- **Low-level** — `CompanionSystem.recall(perspective=...)`: callers that need raw fragment recall under a mentalistic scope (scorer dispatch paths, direct per-turn queries, bespoke adapters) pass a `PerspectiveSpec` directly.

`TraitProfile` is the exemplar. Future mentalistic objects follow the same pattern.

### New package: `src/psychology_base/companion/perspective/`

```
src/psychology_base/companion/perspective/
├── __init__.py         # public API: PerspectiveSpec, ScopeSpec, CopresenceScope, MentalisticObject
├── routing.py          # PerspectiveSpec + ScopeSpec + CopresenceScope re-export
├── objects.py          # MentalisticObject protocol
└── grammars/
    ├── __init__.py
    └── hitom.py        # HiToM chain scaffolding (migrated from scorer)
```

### `routing.py` — composable predicates

**`PerspectiveSpec`** — the unified recall-time scope object:

```python
@dataclass(frozen=True)
class PerspectiveSpec:
    """Composable mentalistic recall scope.

    Each sub-spec is a predicate on fragments; the full spec composes
    them as set-intersection in a fixed order (scope → copresence).
    Predicates are idempotent, so order is semantically irrelevant for
    correctness, but fixed for determinism.

    All sub-specs are None by default — a bare PerspectiveSpec() is
    pass-through (identical to recall with no perspective kwarg).

    New sub-specs (event_window, trust_chain, narrative_frame, …) land
    here as follow-on fields when a gate motivates them. The composition
    pattern — frozen dataclass + explicit apply() — accommodates
    additions without breaking existing callers.
    """
    scope: Optional[ScopeSpec] = None
    copresence: Optional[CopresenceScope] = None

    def apply(self, fragments: Iterable[Fragment]) -> list[Fragment]:
        """Compose active sub-spec predicates as set-intersection."""
        pool = list(fragments)
        if self.scope is not None:
            pool = self.scope.apply(pool)
        if self.copresence is not None:
            pool = filter_fragments_by_copresence(pool, self.copresence)
        return pool
```

**`ScopeSpec`** — promoted from `system.py::_filter_fragments_by_scope`:

```python
@dataclass(frozen=True)
class ScopeSpec:
    """Possessive-scope predicate on fragment tags.

    Semantics match the Phase 2 M2 primitive EXACTLY — a fragment
    matches iff ``f"subject:{subject}"`` AND ``f"entity:{entity}"``
    are both in ``fragment.tags``. Scope-less fragments (no
    ``subject:`` and no ``entity:`` tags of any kind) pass through.

    **Scope-asymmetric semantics (load-bearing for FANToM +0.791).**
    When both subject and entity are set, the AND-contract applies and
    fragments carrying ``subject:X`` but no ``entity:*`` tag of any
    kind do NOT pass. This asymmetric behavior is the mechanism Phase 5
    Track 1 depends on; it is preserved byte-identical.
    """
    subject: Optional[str] = None
    entity: Optional[str] = None

    def apply(self, fragments: Iterable[Fragment]) -> list[Fragment]:
        ...  # exact behavior of _filter_fragments_by_scope
```

**`CopresenceScope`** — moved from `src/psychology_base/companion/copresence.py` unchanged. Public API preserved via re-export at `copresence.py` (**permanent re-export**, not grace-tick-deprecated — zero cost, zero out-of-tree consumers, no reason to churn imports).

### `objects.py` — MentalisticObject protocol

```python
@runtime_checkable
class MentalisticObject(Protocol):
    """Principle C in code: a mind's model of someone's mind is a process.

    Implementations walk tagged fragments at reconstruction time,
    aggregate under the owner's perspective, and return a reconstructed
    state. Reconstruction is pure (no caching) — invalidation is
    automatic because the fragment store IS the state.

    The ``PERSPECTIVE`` class-attribute names the default scope the
    reconstruction uses; callers can override by passing an explicit
    PerspectiveSpec to ``reconstruct()``.
    """
    PERSPECTIVE: ClassVar[PerspectiveSpec]

    @classmethod
    def reconstruct(
        cls,
        system: CompanionSystem,
        context: Context,
        *,
        perspective: Optional[PerspectiveSpec] = None,
    ) -> Self: ...
```

**Concrete implementation shipped in this plan: `TraitProfile`** (and only `TraitProfile`). Other mentalistic objects follow the same pattern when a gate motivates them.

### `CompanionSystem.recall()` — low-level API

```python
def recall(
    self,
    cue: str,
    *,
    perspective: Optional[PerspectiveSpec] = None,
    context: Optional[Context] = None,
    affect_weight: Optional[float] = None,
    # Legacy kwargs — permanent, not deprecated:
    scope: Optional[tuple[str, str]] = None,
    copresence: Optional[CopresenceScope] = None,
) -> Recall:
    ...
```

**Behavior rules:**
- `perspective is not None` → PerspectiveSpec drives all filtering; suppress hierarchical fact-store path (same reason today's `scope` and `copresence` do: fact_store carries no perspective tags, routing a mentalistic query through facts would surface the wrong referent class).
- `perspective is None` and `scope is not None` → construct `PerspectiveSpec(scope=ScopeSpec(subject, entity))` internally, dispatch. **No DeprecationWarning** — legacy kwargs stay permanent (zero out-of-tree consumers, no churn benefit).
- `perspective is None` and `copresence is not None` → construct `PerspectiveSpec(copresence=copresence)` internally, dispatch.
- Both `perspective` and legacy kwargs set → raise `ValueError` (mutex). Mirrors Pillar B / Pillar D mutex contracts.
- All None → today's unfiltered behavior unchanged.

**Metadata on returned `Recall.metadata`**: when `perspective is not None`, emit `metadata["perspective"]`-tuple-serialized provenance; when legacy kwargs used, continue to populate `scope`/`copresence` keys as today.

**Witness-pool filtering (Step 2.1b).** The Witness entity's pain-signal pool at `system.py:927` currently reads `_filter_fragments_by_scope(self.witness.store.all(), scope)`. The migration extends this identically: when `perspective is not None`, apply `perspective.apply(self.witness.store.all())` so Witness narrows under the same mentalistic scope as the main pool. This preserves the invariant that pain-signals and recall share one perspective.

**Affect-active orthogonality (Step 2.1c).** `affect_active` and `affect_weight` are reconstruction-time weightings, not fragment-pool filters. They stay orthogonal to `perspective=` — a perspective-filtered recall can still be affect-weighted at reconstruction. No composition issue; the recall dispatcher applies perspective filtering BEFORE handing to reconstruction, which then applies affect weighting as today.

### `grammars/hitom.py` — HiToM chain scaffolding

Move from `higher_order_tom_scorer.py:235-338`:
- `_hitom_choices_block`
- `_hitom_chain_scaffold`
- `_build_hitom_mentalistic_prompt` → `build_hitom_perspective_prompt`
- `_build_hitom_bare_prompt` → `build_hitom_bare_prompt`

Scorer continues to dispatch on `metadata.source == "hitom"`; it imports builders from the perspective-grammar home rather than defining them locally. **The scorer still owns the dispatch; the grammar module owns the prompt-template logic.** This keeps the "corpus-specific grammar stays corpus-local" invariant from the Phase 9+ witness primitive design.

Future `grammars/bigtom.py`, `grammars/d4.py` land here as they're built.

---

## Critical files

**Create:**
- `src/psychology_base/companion/perspective/__init__.py` — public API re-exports.
- `src/psychology_base/companion/perspective/routing.py` — `PerspectiveSpec`, `ScopeSpec`, re-exports `CopresenceScope`.
- `src/psychology_base/companion/perspective/objects.py` — `MentalisticObject` protocol.
- `src/psychology_base/companion/perspective/grammars/__init__.py`
- `src/psychology_base/companion/perspective/grammars/hitom.py` — migrated HiToM scaffolding + prompt builders.
- `tests/test_perspective_routing.py` — PerspectiveSpec composition, ScopeSpec AND-asymmetry regression, byte-identity vs `_filter_fragments_by_scope`, byte-identity vs `filter_fragments_by_copresence`.
- `tests/test_mentalistic_objects.py` — `MentalisticObject` protocol conformance, TraitProfile-via-protocol parity vs today's TraitProfile.
- `tests/test_perspective_recall_api.py` — new `perspective=` kwarg composes correctly, mutex with legacy kwargs raises ValueError, metadata round-trip, Witness-pool filtering under perspective.

**Modify:**
- `src/psychology_base/companion/system.py` — add `perspective=` kwarg to `recall()`; keep legacy `scope`/`copresence` kwargs permanently as internal shims that construct a `PerspectiveSpec`; extend Witness-pool filter at line ~927 to use `perspective.apply()` when perspective is set. `_filter_fragments_by_scope` stays as the internal implementation of `ScopeSpec.apply` (lifted into routing.py; `system.py` imports from there).
- `src/psychology_base/companion/copresence.py` — convert to permanent shim re-exporting `CopresenceScope` and `filter_fragments_by_copresence` from `perspective.routing`. No behavior change.
- `src/psychology_base/applications/emotional/trait_profile.py` — `TraitProfile` adds `PERSPECTIVE` class-attribute + `reconstruct(cls, system, context, *, perspective=None)` classmethod; existing `.get(dim)` instance method stays as backward-compat shim (see Step 3.2b).
- `src/psychology_base/eval/benchmarks/public_datasets/higher_order_tom_scorer.py` — remove inline HiToM prompt builders; import from `psychology_base.companion.perspective.grammars.hitom`. Scorer dispatch logic unchanged.
- `src/psychology_base/eval/benchmarks/public_datasets/hitom_witness.py` — docstring update noting the primitive now lives in the perspective package; actual import path for `ingest_hitom_story_as_observations` stays here (adapter-local, corpus-specific).

**Regression-pin (no behavior change expected, byte-identity tests added):**
- `tests/test_perspective_fantom_regression.py` — pin FANToM Track 1 +0.791 on `BELIEF_INACCESSIBLE` through the `perspective=` API path vs the legacy `scope=` path. Byte-identical recall sets.
- `tests/test_higher_order_tom_scorer.py` — existing 9 tests continue to pass against migrated prompt builders.
- `tests/test_copresence.py` — existing 17 tests pass against re-exported primitives.
- `tests/test_trait_inference.py` — existing tests pass against TraitProfile with new protocol surface.
- `tests/test_companion_phase1_step*.py` + any Phase 2 scope-filter tests — scope-filter semantics byte-identical.

**Do NOT modify:**
- Any adapter (`locomo.py`, `fantom.py`, `big5_chat_classification.py`, `truthful_qa.py`, etc.) — perspective-filtering is a recall-time concern; adapters build Observations unchanged.
- `fragment.py` — fragment data model unchanged.
- `reconstruction.py` — reconstruct() signature unchanged; it receives the already-filtered fragment pool as before.
- `witness.py` / Witness entity coupling — `witness_active=True` narrowing composes upstream of PerspectiveSpec, same as today's scope/copresence.
- Phase 8 close-outs, Phase 7 close-out, Phase 5 close-out — this plan does not re-litigate historical gates; it preserves them.
- Any `benchmarks/scripts/phase*_*_run.py` CLI — JSON schema unchanged; metadata key `perspective` adds alongside today's `scope`/`copresence` keys permanently.

---

## Reusable code / utilities to lean on

- **`_filter_fragments_by_scope`** at `system.py:415` — exact implementation of `ScopeSpec.apply`. Lift as-is into `perspective/routing.py`.
- **`filter_fragments_by_copresence`** at `copresence.py:62` — exact implementation of `CopresenceScope.apply` wrapped by PerspectiveSpec. Lift as-is.
- **`TraitProfile._candidate_fragments`** at `trait_profile.py:175` — the archetypal reconstruction walker. Future `MentalisticObject.reconstruct()` implementations follow this pattern.
- **`CopresenceScope` frozen-dataclass + hashability** at `copresence.py:37` — pattern for all perspective-sub-specs. Keep frozen for safe caching + hashable composition.
- **`Recall.metadata` dict** — current `scope`/`copresence` keys continue to work permanently; add `perspective` key alongside, carrying the structured spec.

**Alternative integration considered + rejected: Make `PerspectiveSpec` an ABC with virtual `apply()`.** Rejected because frozen dataclasses with explicit `apply()` methods are already as polymorphic as needed, and ABC inheritance would complicate hashability/caching.

**Alternative considered + rejected: Push HiToM prompt builders onto `MentalisticObject`.** Would collapse routing and prompting into one object. Rejected because prompt-template-building is corpus-specific grammar and must stay colocated with corpus adapter / scorer, not with the generic routing primitive.

**Alternative considered + rejected (object-owns-perspective ONLY, no `recall(perspective=...)`).** Would make `MentalisticObject.reconstruct()` the only surface and hide `perspective=` entirely. Rejected because scorer dispatch paths and bespoke adapters need the low-level recall surface — collapsing to one API forces them through an object they don't want. Shipping both lets high-level callers reach for `TraitProfile.reconstruct(system)` while low-level callers keep direct `recall(perspective=...)`.

---

## Execution order

### Step 0 — regression-pin BEFORE touching code

**0.0 (fixture audit).** Before writing any new regression pins, audit which fixtures already exist in-tree:
- Grep for existing FANToM Track 1 pin: `tests/test_fantom_architectural_delta.py` or similar — reuse if present, create fresh if not.
- Grep for Phase 7 TraitProfile at N=50 pin: `tests/test_trait_inference.py` + Phase 7 B4 fixture — reuse.
- Grep for Phase 2 B2 attribute reconstruction fixture: likely **not shipped as code** (B2 was planned, closed as architectural claim rather than implemented scripted session). If no fixture exists, drop the B2 pin — no fixture = no regression surface = YAGNI.
- Grep for Phase 6 covenant hold at ep1000 pin: `tests/test_phase6_*` — reuse.

Record which pins are reuses and which are fresh. Fresh pins only for gates whose fixtures already have ingestable data in-tree.

**0.1.** Capture current FANToM Track 1 `BELIEF_INACCESSIBLE` recall pool bytes-exactly (fragment IDs + content hashes). Pin as fixture.
**0.2.** Capture Phase 8 Pillar E v2 HiToM order 2 N=10 recall pools + mental-path answers under current API. Pin as fixture.
**0.3.** Capture Phase 7 TraitProfile OCEAN at N=50 on a saved chat corpus. Pin as deterministic fixture.
**0.4.** (Only if 0.0 found a shippable B2 fixture.) Capture Phase 2 B2 attribute reconstruction on scripted fixture.

Run `pytest tests/` — must be green at head. Record total count. Any regression after Step 0 is **blocking**; do not proceed to Step 1 until green.

### Step 1 — strictly additive: new package alongside old

1.1. Create `src/psychology_base/companion/perspective/routing.py` with `PerspectiveSpec`, `ScopeSpec`. Re-export `CopresenceScope` from `copresence.py` (which stays as permanent shim).
1.2. Unit tests `tests/test_perspective_routing.py`: composition, order-independence (equal result for scope-then-copresence vs copresence-then-scope on same inputs), each primitive's pass-through semantics, byte-identity vs legacy `_filter_fragments_by_scope` on fixture, byte-identity vs legacy `filter_fragments_by_copresence` on fixture, ScopeSpec AND-asymmetry regression.
1.3. Full suite green. **STOP if any regression.**

### Step 2 — strictly additive: unified recall API

**2.1 (main-pool integration).** Add `perspective=` kwarg to `CompanionSystem.recall()`. Internally: if `perspective is not None`, apply `perspective.apply(fragments)` after the hierarchical-suppression decision; otherwise fall through to legacy `scope`/`copresence` behavior unchanged.

**2.1b (Witness-pool integration).** Extend the Witness-pool filter at `system.py` line ~927 identically. Today reads `_filter_fragments_by_scope(self.witness.store.all(), scope)`. New behavior: when `perspective is not None`, apply `perspective.apply(self.witness.store.all())`; when legacy `scope` is used, continue today's path. Preserves invariant that pain-signals and recall share one perspective.

**2.1c (affect-active orthogonality — no code change, test only).** Add a test that `recall(cue=..., perspective=PerspectiveSpec(scope=ScopeSpec(subject="user")), affect_weight=0.7)` produces fragments that are both perspective-filtered AND affect-weighted at reconstruction. Documents the orthogonality in-code.

2.2. Add mutex check: both `perspective` and any legacy kwarg → `ValueError`.
2.3. Metadata: when `perspective is not None`, emit `metadata["perspective"] = (<spec-tuple-serialization>)`; legacy keys `scope`/`copresence` continue to populate when legacy kwargs used.
2.4. Unit tests `tests/test_perspective_recall_api.py`: new kwarg produces byte-identical fragment pool to legacy `scope=`/`copresence=` kwargs on fixture; mutex raises; metadata round-trips; Witness-pool filter test; affect-active orthogonality test.
2.5. Full suite green. **STOP if any regression.**

### Step 3 — MentalisticObject protocol + TraitProfile migration

3.1. Create `src/psychology_base/companion/perspective/objects.py`. Define `MentalisticObject` protocol with `PERSPECTIVE` class-attribute and `reconstruct()` classmethod. Mark `@runtime_checkable`.

**3.2 (TraitProfile migration — primary API).** Add to `TraitProfile`:
   - `PERSPECTIVE: ClassVar[PerspectiveSpec] = PerspectiveSpec(scope=ScopeSpec(subject="user"))`.
   - `@classmethod def reconstruct(cls, system, context, *, perspective=None) -> TraitProfile` that uses the given or default PerspectiveSpec to narrow fragments, then builds the TraitProfile from the narrowed pool.

**3.2b (backward-compat shim — explicit relationship).** Existing `TraitProfile.get(dim)` instance method stays unchanged in behavior. The identity under refactor:
```
TraitProfile(system, context).get(dim)
≡
TraitProfile.reconstruct(system, context).get_estimate(dim)
```
Both paths read the same fragments, under the same default `PERSPECTIVE`, through the same aggregation logic. Keep both — the instance-method path is terser for the trait-conditioning hot path; the classmethod path is the primary surface for new code and the exemplar for other MentalisticObjects. **No deprecation** on `.get()`.

3.3. Unit tests `tests/test_mentalistic_objects.py`:
   - `isinstance(TraitProfile, MentalisticObject)` holds at runtime via `@runtime_checkable`.
   - TraitProfile-via-protocol byte-identical OCEAN scores to pre-migration path on Phase 7 N=50 fixture.
   - `TraitProfile.reconstruct(system, context).get_estimate(dim)` == `TraitProfile(system, context).get(dim)` for all five OCEAN dims on fixture.
   - Overriding `perspective=` on `reconstruct()` narrows fragment pool as expected.
3.4. Full suite green including Step 0 regression pins. **STOP if any regression.**

### Step 4 — migrate FANToM scorer (regression-pin FIRST)

4.1. Verify Step 0.1 regression pin still passes under legacy API path.
4.2. Modify FANToM scorer to construct `PerspectiveSpec(scope=ScopeSpec(subject=target_person))` and pass as `perspective=`. **Do NOT change the subject-only / entity-None semantics.** The ScopeSpec with subject-only + no entity preserves the exact "scope-asymmetric → empty recall on INACCESSIBLE" mechanism.
4.3. Run Step 0.1 regression pin against new path. **Byte-identical fragment pool and answer set required.** +0.791 must be preserved exactly.
4.4. Run full FANToM Track 1 live-fire at the published N and tier (lightweight — same corpus, same seed, same bias). Delta must land within noise band of +0.791.
4.5. Full suite green. **STOP if +0.791 regresses.**

### Step 5 — migrate HiToM scorer (regression-pin FIRST)

5.1. Move HiToM prompt builders to `perspective/grammars/hitom.py`. Scorer imports from new path.
5.2. Verify Step 0.2 regression pin passes.
5.3. Scorer dispatch uses `perspective=PerspectiveSpec(copresence=CopresenceScope(chain=target_chain))` instead of direct `copresence=` kwarg.
5.4. Byte-identity vs Step 0.2 fixture required.
5.5. Full suite green including Pillar E v2 regression check. **STOP if +0.376 regresses.**

### Step 6 — write close-out + update memory

6.1. Write `docs/phases/phase-9-perspective-substrate-consolidation-close-out.md`:
   - Final metrics on every regression pin (byte-identical confirmed).
   - **Promote `MentalisticObject.reconstruct()` as the primary surface for reconstructive queries; `system.recall(perspective=...)` as the low-level primitive.** `TraitProfile` is the exemplar. Future reconstructive objects follow the pattern.
   - What this unlocks (BigToM situation-shift as future `MentalisticObject`, multi-scene re-entry when `event_time:` tagging arrives, user-model-of-mother when B2 ships as code).
   - Explicit note: `UserRepresentationOf` / `UserModelOfAI` / `UserSelfModel` are **not** in-tree yet; they land as MentalisticObject implementations when a gate motivates them.
6.2. Update `docs/roadmap.md` with Phase 9+ consolidation as a shipped item (not a new phase per se — this is a substrate-cleanup supporting all future phases).
6.3. Update MEMORY.md with concise entry (update existing Phase 8/9+ entry rather than add a new one, to stay under index limit).

**Stop immediately** if any step's regression pin fails. The plan does not override Phase 2–8 gates; those gates are the floor this consolidation defends.

---

## Verification plan

### Unit-level (new tests, deterministic, no LLM)

- `test_perspective_spec_composition_order_independent` — same fragment pool out regardless of sub-spec application order.
- `test_perspective_spec_passthrough_on_empty` — `PerspectiveSpec()` returns input unchanged.
- `test_perspective_spec_mutex_with_legacy_kwargs_raises`
- `test_scope_spec_subject_only_preserves_fantom_mechanism` — on FANToM-shaped fragments (subject tag only, no entity tag), `ScopeSpec(subject=X)` keeps; `ScopeSpec(subject=X, entity=Y)` drops. Pins +0.791 mechanism.
- `test_scope_spec_byte_identity_vs_legacy_filter` — on Phase 2 M2 fixture, new path produces identical kept-set to `_filter_fragments_by_scope`.
- `test_witness_pool_filtered_under_perspective` — Witness-pool `_filter_fragments_by_scope` call extends to `perspective.apply()`.
- `test_affect_weight_composes_with_perspective` — perspective-filtered + affect-weighted recall produces both effects.
- `test_mentalistic_object_protocol_runtime_checkable`
- `test_trait_profile_via_protocol_matches_legacy_api` — byte-identical OCEAN scores on Phase 7 N=50 fixture.
- `test_trait_profile_reconstruct_get_estimate_matches_get` — `.reconstruct(...).get_estimate(dim)` == `.get(dim)` for all five dims.
- `test_trait_profile_reconstruct_accepts_override_perspective` — passing explicit `perspective=` narrows fragments as expected.
- `test_hitom_grammar_byte_identical_prompt_after_migration`

### Regression pins (Step 0 captures)

- `test_fantom_track_1_belief_inaccessible_regression` — FANToM Track 1 architectural +0.791 delta pinned.
- `test_hitom_order_2_mental_delta_regression` — Pillar E v2 +0.376 pinned.
- `test_hitom_order_3_mental_delta_regression` — +0.355 pinned.
- `test_phase_7_trait_profile_ocean_at_n50_regression`
- `test_phase_7_covenant_hold_ep1000_under_conditioning_regression` — Phase 6 result unaffected (perspective consolidation touches recall path, not aging policy; smoke-test ep200 subset is sufficient).

### Integration (benchmark reruns, no gate-moving expected)

- Rerun FANToM Track 1 at published N. Delta within ±0.015 of +0.791.
- Rerun Pillar E v2 HiToM order 2 N=100. Delta within ±0.015 of +0.376.
- Rerun Phase 7 B4 at N=200. `mean_dims_within_half_likert` within ±0.1 of 5.00 (log-weighted; midpoint).
- LoCoMo 3-instance monitor F1 ≥ 0.514 (Phase 1.5 non-regression floor).

### End-to-end

- Full `pytest tests/` green. New test count = 1563 + (new unit tests across Steps 1–3) ≈ 1600+.
- No existing test broken — verified by running full suite BEFORE each Step lands, and comparing green-count diff.

### Pass criteria for the plan as a whole

- All 6 steps shipped; all regression pins green; all benchmark reruns within noise band; `MentalisticObject` protocol is the surface future mentalistic objects implement.
- `TraitProfile` demonstrates the pattern end-to-end as the exemplar.
- Phase 2–8 close-outs **unchanged** by the consolidation. The claim shape stays dual-axis + 3-of-6; this plan clarifies HOW rather than extending WHAT.

---

## Risks and open questions

- **FANToM scope-asymmetric mechanism exposed as first-class.** Making `ScopeSpec(subject=X, entity=None)` a legal and documented primitive risks future contributors "fixing" the subject-only case to also auto-require entity presence — regressing +0.791. Mitigation: regression test pin at Step 0.1 + explicit docstring at `ScopeSpec` warning that the asymmetric semantics are load-bearing.
- **`MentalisticObject.reconstruct()` signature stabilization.** Protocol signature `(system, context, *, perspective=None) -> Self` must accommodate every current and future mentalistic object. If BigToM situation-shift tracker needs an additional `narrative_frame` parameter, the protocol signature changes and all implementers follow. Mitigation: future reconstruction kwargs land on a separate `ReconstructionContext` dataclass that composes rather than widening the protocol. Noted as follow-on.
- **TraitProfile dual-API** (`.get()` instance method + `.reconstruct()` classmethod) could confuse new callers. Mitigation: close-out explicitly promotes `.reconstruct()` as the primary surface and documents `.get()` as the terse hot-path shim; no deprecation (keeping it stable costs nothing).
- **Hierarchical fact store suppression logic** currently reads `scope is not None or copresence is not None`. Must be extended to `perspective is not None`. If any PerspectiveSpec is passed, suppress the fact-tier (same reason as today). Verified in Step 2.1.
- **HiToM witness ingester** (`ingest_hitom_story_as_observations`) stays at `eval/benchmarks/public_datasets/hitom_witness.py` — adapter-local, corpus-specific grammar. This is **by design** (same reason BigToM parser belongs with the BigToM adapter); the `grammars/` sub-package in `perspective/` is for the recall-time + prompt-building side, not the ingest-time parser side. Mitigation: close-out explicitly documents the split so future contributors don't try to move the ingester.
- **Witness-pool filter extension (Step 2.1b)** must be byte-identical to today's path when legacy `scope=` is used. If any Phase 2 B2 or Phase 6 Witness pain-signal test exercises the Witness store under scope, it must pass unchanged. Mitigation: Step 2.1b lands under the same byte-identity harness as Step 2.1.
