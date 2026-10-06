# Phase 1.5 build plan — Option 5.B consolidation layer (Mem0-pattern)

**Date set:** 2026-10-06
**Decided path:** Option 5.B (consolidation layer) with typed-but-minimal facts (entity + relation + value) and the E1-prototype-before-E2-gate experiment commitment. User authorized 2026-10-06.
**Companion docs:** [phase-1.5-scale-recall-plan.md](phase-1.5-scale-recall-plan.md) is the architecture plan this build-plan implements; [comparison.md](../experiments/phase-1-benchmark-adapters/comparison.md) is the measured gate failure that triggered this phase; [roadmap.md](../roadmap.md) sequences Phase 1.5 between Phase 1 and Phase 2.
**Scope:** this plan writes code. It specifies the FactStore component, the sleep-tick extraction pass, the cue-conditioned recall dispatch, and the two gate experiments (E1 single-instance prototype, E2 full 10-instance gate). It does NOT redesign the extraction prompt mid-flight — the prompt is locked after step 2 and treated as part of the substrate.

---

## 1. Decisions carried from the architecture plan

Six open questions were listed in [phase-1.5-scale-recall-plan.md](phase-1.5-scale-recall-plan.md) §7. Resolved for this build:

| # | Open question | Decision | Rationale |
|---|---|---|---|
| 1 | Extraction cadence | **Per-session boundary**: extract on `tick()` when at least one new session's worth of fragments has accrued since the last extraction. Fallback: every N observations when no session boundary signal is available (N=20, matches Phase 1 ep20 ingestion rhythm). | Per-turn (Mem0 default) burns LLM budget on noise; per-sleep-tick is Phase-1-principle aligned. Session-boundary is the natural coherence unit for LoCoMo and for both north-star applications. |
| 2 | Fact typing | **Typed-but-minimal**: `(entity, relation, value, confidence)` four-field record. `entity` is a short string (e.g. `"user"`, `"Jennifer"`, `"user's mother"`); `relation` is a free-form predicate (e.g. `"owns"`, `"feels"`, `"gave_birth_to"`); `value` is the object. `confidence` is extractor-supplied `0.0-1.0`. | Flat strings (Mem0-style) are easier but leave Phase 2 mentalistic layer with no hook. Typed-but-minimal gives entity as the handle for `mother-as-user-represents-her` without front-running the full typed-knowledge-graph (Option 5.D) scope. |
| 3 | Fragment back-pointers | **Required** — every fact carries `source_fragment_ids: tuple[str, ...]`. On fact retrieval, callers can resolve the raw fragments for Principle B grounding. | Non-negotiable for the learning companion ("show me what we actually said about ∂") and for debugging extraction quality. |
| 4 | Embedder choice | **Local Ollama `bge-small-zh-v1.5`** for all fragment + fact + cue embeddings in ingest and recall. Cloud embedder is NOT used in this phase — the router's `embed()` path already prefers local when Ollama is ready; add a hard-fail if Ollama is unavailable (don't silently fall back to a cloud embedder we didn't budget for). | Already cached per MEMORY.md llm-architecture entry; zero marginal cost; 512-dim fits SQLite BLOB column trivially. If `bge-small-zh-v1.5` under-performs on English LoCoMo, step 3 kill criterion triggers a swap to `bge-small-en-v1.5`. |
| 5 | Recall path dispatch | **Facts first, fragments on empty**: `recall(cue)` → cosine top-K facts (K=10) → if the fact store returns < 3 facts above a cosine floor of 0.25, back off to the Phase-1 fragment-reconstruction path. The two paths are not fused; whichever path has signal answers. | Mem0's empirical finding is that extracted facts are the signal; mixing in raw-fragment noise hurts. The back-off preserves Phase 1 behavior on cold-start / under-extracted instances. |
| 6 | Fragment-vs-fact selection pressure interaction | **Independent capacities**: fragment byte-cap stays at `4096` (Phase 1 default — do not raise), fact store has its own soft cap `fact_cap=512` facts. When a fragment is purged, facts extracted from it REMAIN (facts carry the content now). When a fact is UPDATEd, prior versions are overwritten *in this phase*; Phase 2 adds the version chain. | Separates episodic decay from semantic retention — biology-shaped. Keeps the fragment byte-cap constant so Phase 1 V1/V2 bands do not shift under us (regression-guarded by step 5 gate). |

---

## 2. Build order (six steps, each with its own gate)

Build ascending in complexity. Each step must pass its gate before the next begins; a gate failure triggers one revision budget per step before escalating to the kill criteria.

| Step | What to build | Local gate |
|---|---|---|
| 1 | `FactStore` scaffolding — `Fact` dataclass, in-memory store, SQLite persistence (deferred to step 6 if build time pressures), insert / upsert / list / search-by-entity APIs | Fact round-trips in-memory; `tests/test_fact_store.py` passes on CRUD |
| 2 | Extraction pass — `facts/extraction.py` with a locked `EXTRACTION_PROMPT`, `extract_facts(fragments, prior_facts, llm_router) -> list[FactOp]` returning `ADD/UPDATE/NOOP` ops (DELETE deferred — Phase 2 reconsolidation territory) | On `conv-26` first session (session_1), extraction produces 10-40 facts; entities include at least `"Jennifer"` + `"David"` (the two speakers); no empty-value facts |
| 3 | Embedding + vector index — add `embedding: list[float]` to `Fact`, wire fact-write to `router.embed()`, implement cosine top-K search over the fact store | `search(cue, k=10)` on 50 fixture facts returns top-K ordered by cosine; Q0-vs-Q1 overlap on four distinct-gold `conv-26` single-hop questions ≤ 0.5 (half the current 1.0) |
| 4 | `CompanionSystem.recall()` dispatch — facts-first with back-off to fragment path per decision 5; `facts_used` added to `Recall.metadata`; integration test uses the real `conv-26` fixture | Direct-mode recall on 4 single-hop questions returns a reconstruction string containing the gold token on ≥ 2 of 4 (vs 0/4 today) |
| 5 | Phase-1 regression check — run existing `tests/` suite + ep20 V1/V2 baseline check with Phase 1.5 code enabled | V1, V2 adjacent, V2 long-horizon all in-band per [baselines.md](../baselines.md); V4 Witness ratio still ≥ 1.3× anchor / ≥ 10× null when fact store is empty (default off path); no regression on the 183-test suite |
| 6 | E1 prototype run on `conv-26` + E2 full gate — `scripts/run_locomo_gate.py` already exists; add `--consolidation on` flag that enables Phase 1.5 code; E1 scored on `conv-26` single-hop (154 Q); if ≥ 0.40, run E2 on all 10 instances | **E1:** `conv-26` single-hop F1 ≥ 0.40 (lift from 0.016). **E2:** full 10-instance LoCoMo single-hop F1 ≥ 0.70 |

**Instrumentation requirement.** Every fact write and every fact-store search logs to `~/.cache/psychology_base/phase-1.5/facts/<instance_id>.jsonl`. Every gate-failure debug session starts from these logs. If the logs cannot localize the mechanism in two iterations, instrumentation is wrong — fix it before further mechanism tuning. (Same discipline as Phase 1 §5 instrumentation layer.)

**Seed discipline.** Fact extraction uses the LLM router at `TaskTier.SIMPLE` with `temperature=0.0` + the router's SHA-keyed response cache. E1 is single-seed because the dominant stochasticity source is the extraction LLM; E2 is single-seed on the same grounds. The seed on the `CompanionSystem` constructor stays `0`.

**Cache discipline.** Extraction calls are cached via `LLMCache`. Re-running E1 after a non-prompt code change (e.g. recall-path tuning) does not re-pay the extraction cost — the cache keys on `(provider, model_id, messages, temperature)` which is unchanged. A prompt edit invalidates the cache for that fact store; this is the correct safety.

---

## 3. File layout

New files the implementation produces:

| File | Role |
|---|---|
| `src/psychology_base/companion/facts.py` | `Fact` dataclass + `FactStore` + `FactOp` enum (`ADD`, `UPDATE`, `NOOP`). SQLite backing optional in step 6; in-memory for E1. |
| `src/psychology_base/companion/extraction.py` | `EXTRACTION_PROMPT` (locked after step 2), `extract_facts(fragments, prior_facts, router) -> list[FactOp]`, prompt-output parser (`ADD entity | relation | value | confidence`) |
| `src/psychology_base/companion/consolidation.py` | Sleep-tick orchestrator: detects session boundary → calls `extract_facts` → applies ops to `FactStore` → stamps logs |
| `tests/test_fact_store.py` | CRUD + upsert + search-by-entity tests |
| `tests/test_extraction.py` | Prompt-contract tests using `MockProvider` with canned ADD/UPDATE/NOOP outputs |
| `tests/test_consolidation.py` | Session-boundary trigger test + idempotent re-run test |
| `tests/test_fact_search.py` | Cue-conditioned top-K test; Q0-vs-Q1 overlap assertion |
| `docs/experiments/phase-1.5-scale-recall/comparison.md` | E2 result table; parallels [phase-1-benchmark-adapters/comparison.md](../experiments/phase-1-benchmark-adapters/comparison.md) |
| `docs/experiments/phase-1.5-scale-recall/option-5B/conv-26-E1.json` | E1 artifact |
| `docs/experiments/phase-1.5-scale-recall/option-5B/full-gate-E2.json` | E2 artifact |

Files to extend:

| File | Change |
|---|---|
| `src/psychology_base/companion/system.py` | Add `consolidation_enabled: bool = False` constructor flag (default off so Phase 1 tests pass unchanged); when on, construct `FactStore` + `Consolidator`; `observe()` passes through unchanged; `recall()` dispatches facts-first then falls back; `tick()` runs consolidation when session-boundary triggered; `inspect()` reports fact counts + fact-store bytes. |
| `src/psychology_base/companion/heartbeat.py` | Add `session_boundary_tick()` to mark session boundaries explicitly from benchmark adapters that know the boundary structure (LoCoMo sessions are named `session_N`). Falls back to N-observation trigger when not called. |
| `src/psychology_base/eval/benchmarks/locomo.py` | `_ingest_history` calls `cs.heartbeat.session_boundary_tick()` between sessions (session boundary already available in the shim schema). |
| `docs/experiments/phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py` | Add `--consolidation` CLI flag that constructs `CompanionSystem(consolidation_enabled=True, ...)`. |

Files intentionally NOT modified:

- `src/psychology_base/companion/fragment.py` — byte-cap stays at 4096; selection pressure unchanged.
- `src/psychology_base/companion/edges.py` — Hebbian edges unchanged.
- `src/psychology_base/companion/reconstruction.py` — R1/R2/R7 pipeline unchanged; it only runs when the facts-first path backs off.
- `src/psychology_base/companion/witness.py` — Witness coupling unchanged; `witness_active=False` for E2 (same as E1 pilot) because the gate measures retrieval, not identity-contrast.
- `src/psychology_base/eval/benchmarks/instance.py`, `judge.py`, `reader.py` — adapters are consumers, unchanged.

---

## 4. Data shapes (contract-first)

```python
# facts.py
@dataclass(frozen=True)
class Fact:
    id: str                              # "fact-0001" sequence
    entity: str                          # "Jennifer"
    relation: str                        # "owns_boat_model"
    value: str                           # "Westerly 28"
    confidence: float                    # 0.0 .. 1.0, extractor-supplied
    source_fragment_ids: tuple[str, ...] # back-pointers for Principle B
    created_tick: int                    # heartbeat.sleep_ticks at write
    updated_tick: int                    # heartbeat.sleep_ticks at last UPDATE
    embedding: tuple[float, ...] = ()    # populated by step 3; empty in step 1-2
```

```python
# extraction.py
class FactOp(Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    NOOP = "NOOP"

@dataclass(frozen=True)
class FactOperation:
    op: FactOp
    fact: Fact                           # for ADD / UPDATE; UPDATE replaces by id
    target_fact_id: Optional[str] = None # for UPDATE
```

```text
# EXTRACTION_PROMPT (locked contract — v1)
# System:
#   You read a short conversation window and emit factual claims.
#   Output format: one line per fact.
#   Each line: ADD | entity | relation | value | confidence
#   Rules:
#     - entity: short noun phrase, the subject of the fact
#     - relation: short predicate, verb or short phrase
#     - value: short object; concrete name/number/date preferred over
#       descriptive phrase
#     - confidence: 0.0..1.0, your subjective reliability
#     - Only facts that are explicit in the conversation window
#     - Omit speculation, meta-commentary, pleasantries
#     - If a prior fact in PRIOR_FACTS contradicts a new fact,
#       emit UPDATE <prior_fact_id> | entity | relation | value | confidence
#     - If a prior fact is restated unchanged, emit NOOP <prior_fact_id>
#     - Return at most 40 lines per window
```

Format chosen for parse-stability (pipe-separated, one-per-line) over JSON — Mem0 switched to a similar structured text format after finding Qwen-SIMPLE-tier JSON mode unreliable. Judge-cache keying still works since the whole response string is hashed.

---

## 5. Per-step details

### Step 1 — FactStore scaffolding

In-memory store with id sequence counter. CRUD + `upsert(fact)` + `list_by_entity(entity)` + `all()`. No vector index yet. SQLite persistence deferred — `conv-26` has ~400 facts max estimated, in-memory is fine for E1/E2.

**Gate:** `tests/test_fact_store.py` — insert, upsert replaces by id, list_by_entity filters correctly, all() returns insertion order.

**Kill criterion:** none at this step; it's data structure plumbing.

### Step 2 — Extraction pass

`extract_facts(window: list[Fragment], prior_facts: list[Fact], router: LLMRouter) -> list[FactOperation]`:

1. Build prompt from locked template + window content (speaker-tagged) + top-10 prior facts (by `updated_tick` recency, pre-filter by entity-string presence in window).
2. Call `router.chat(..., tier=TaskTier.SIMPLE, temperature=0.0)`.
3. Parse each output line into a `FactOperation`. Lines that fail to parse are dropped with a log entry (do not crash — extraction is best-effort).
4. Return the ops; the Consolidator applies them (keeps extraction pure for testability).

**Gate:** on `conv-26` first session, extraction produces 10–40 facts; `{"Jennifer", "David"}` ⊂ `{op.fact.entity for op in ops}`; no `value=""` records.

**Kill criterion:** if the parse-success rate across all 10 LoCoMo instances is < 70% of emitted lines, the pipe-separated format is wrong for this SIMPLE-tier model — fall back to JSON mode with strict schema OR move extraction to ADVANCED tier (one-iteration budget). If still < 70%, the extraction prompt itself is wrong — redesign.

### Step 3 — Embedding + vector index

On fact-write (`upsert`), `router.embed(f"{entity} {relation} {value}")` populates `Fact.embedding`. On search, `router.embed(cue)` → cosine against every fact in the store (brute-force; `fact_cap=512` makes this O(512) per recall, cheap).

**Gate:** `tests/test_fact_search.py` runs `cs.recall(cue)` on 4 single-hop `conv-26` questions after full ingest; asserts Q0-vs-Q1 fact-set overlap ≤ 0.5. (Current: 1.0 fragment-set overlap. Target: facts actually re-rank with cue.)

**Kill criterion:** overlap stays > 0.5 → embedder is wrong. One iteration: try `bge-small-en-v1.5` since LoCoMo is English. If still > 0.5, cosine over `entity relation value` is not discriminative — add `source_fragment` sentence to the embedded string (richer semantics, more diluted but still cue-matchable).

### Step 4 — Recall dispatch

`CompanionSystem.recall(cue, context)`:

1. If `consolidation_enabled` and fact store is non-empty: embed `cue`, cosine top-K=10 facts, filter by `cosine ≥ 0.25`.
2. If ≥ 3 facts pass: compose reconstruction string as `" ".join(f"{f.entity} {f.relation} {f.value}" for f in top_facts)`, record `facts_used` in `metadata`, return.
3. Else: fall back to Phase-1 R1+R2+R7 fragment reconstruction (current `recall()` body).

The reader step in `benchmarks/reader.py` is unchanged — it already consumes `(cue, recall.content, fragments_used)`. We add fact text to `recall.content`; `fragments_used` can be populated from back-pointers (facts → their source fragments) so the RAG reader sees both.

**Gate:** on 4 single-hop `conv-26` questions, ≥ 2/4 reconstruction strings contain the gold token. Current: 0/4.

**Kill criterion:** 0/4 or 1/4 → either (a) extraction missed the gold facts (check step-2 artifacts: are the facts there at all?) or (b) cosine is not picking the right fact (check cosines of the gold fact vs the top-10 returned). If (a), the extraction prompt lost signal — re-tune within the step-2 iteration budget. If (b), the embedder is wrong — step-3 kill-criterion recovery path.

### Step 5 — Phase-1 regression check

Run `pytest tests/` with `consolidation_enabled=False` default. Run the ep20 scenario explicitly via `scripts/run_baseline.py` (per `docs/baselines.md`) and verify V1/V2/V4 all in-band.

**Gate:** 183 tests pass unchanged (Phase 1.5 adds new tests, doesn't break old ones); V1 ≥ 0.9153 ± 0.05, V2 adjacent 0.65–0.80, V2 long-horizon 0.20–0.35, V4 ratio ≥ 1.3× anchor and ≥ 10× null.

**Kill criterion:** any Phase 1 metric band missed → something in the shared code path leaked into the default-off path. Isolate before proceeding; this is a non-negotiable floor.

### Step 6 — E1 then E2

E1: `python docs/experiments/phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py --instance-limit 1 --categories single-hop --consolidation --mode rag --output docs/experiments/phase-1.5-scale-recall/option-5B/conv-26-E1.json`

- If `conv-26` single-hop F1 ≥ 0.40 → proceed to E2.
- If 0.20 ≤ F1 < 0.40 → one iteration budget. Likely culprit: cosine floor (0.25) or top-K (10). Tune one of these, re-run E1.
- If F1 < 0.20 → extraction is not surfacing gold facts. Re-open step 2.

E2: `python … run_locomo_gate.py --categories single-hop --consolidation --mode rag --output docs/experiments/phase-1.5-scale-recall/option-5B/full-gate-E2.json`

- F1 ≥ 0.70 → Phase 1.5 exits. Write `docs/experiments/phase-1.5-scale-recall/comparison.md`; update [roadmap.md](../roadmap.md).
- 0.50 ≤ F1 < 0.70 → one iteration budget on extraction prompt. The prompt is "locked after step 2" for *within-phase* stability, not against gate-failure recovery.
- F1 < 0.50 → Option 5.B is wrong for this problem. Re-open the architecture plan: Option 5.D (temporal KG) becomes the fallback.

**Cost estimate for E2.** 10 instances × ~500 turns × 1 extraction per ~20 turns × SIMPLE-tier router call (Ollama local when available) + judge calls. Rough: ~$0.50–2.00 if cloud SIMPLE is used; ~$0 for extraction if Ollama handles SIMPLE. Judge dominates (~$5–10 for 10 instances × 100 Q × GPT-4o-mini / qwen-max). Within budget.

---

## 6. What this plan does NOT decide

- **UPDATE policy details.** Step 2 emits UPDATE operations but step 1's `upsert` just replaces by id. Version chaining (Phase 2 reconsolidation substrate) is explicitly out of scope — see [phase-1.5-scale-recall-plan.md](phase-1.5-scale-recall-plan.md) §9.
- **DELETE operations.** Extraction prompt is instructed to emit ADD/UPDATE/NOOP only. DELETE is reconsolidation-adjacent; defer to Phase 2.
- **Fact-store SQLite persistence.** In-memory is enough for E1/E2 (one process per instance). Persistence can land after Phase 1.5 exits if later phases need it.
- **BM25 / cross-encoder reranker.** Mem0 docs are explicit that BM25 is a boost, not a recall expander — add only if the cosine-only path shows a long tail of close-but-missed facts. Deferred.
- **ADVANCED-tier extraction fallback.** Only triggered by step-2 kill criterion if SIMPLE-tier parse rate is too low. Not a planned path.
- **LongMemEval / EpBench runs.** Phase 1.5 exit criteria include smoke runs on these ([phase-1.5-scale-recall-plan.md](phase-1.5-scale-recall-plan.md) §8). They are run after E2 passes; writing the runners is out of scope for this build plan (adapters already exist, only the `--consolidation` flag + smoke scripts are new, deferred to post-E2 completion).

---

## 7. Risks

- **Extraction cost dominates if SIMPLE-tier falls through to cloud.** Mitigation: router already prefers Ollama when probe passes; this plan hard-fails extraction when Ollama is unavailable (see decision 4), so a silent cost blowup is impossible.
- **Fact-cap=512 saturation at 500-turn dialogue scale.** If a single `conv-26`-sized dialogue produces > 512 facts, we need an eviction policy for the fact store before Phase 2. Mitigation: step-2 gate measures actual fact count per session; if trending above 50/session (×20 sessions = 1000), bump `fact_cap` to 1024 or add LRU eviction — but this is a kill-criterion-triggered path, not planned.
- **Embedder quality on LoCoMo English content.** `bge-small-zh-v1.5` is Chinese-primary; English retrieval may be degraded. Mitigation: step-3 kill criterion triggers swap to `bge-small-en-v1.5` (also small, also local-runnable).
- **Session-boundary detection fragility.** The LoCoMo shim has clean `session_N` boundaries; LongMemEval has them too. Real deployed companions won't have explicit markers. Mitigation: fallback N-observation trigger (N=20) handles the no-boundary case; application-side session detection is Phase 2 scope.
- **Facts contradicting Phase 1 principle A (living process).** Facts in this phase are overwrite-on-UPDATE. This looks like a "store, not process" step backward. Mitigation: Phase 2 adds the version chain on top; the architecture plan §9 declares this explicitly. Phase 1.5 is deliberately a one-step-sideways to clear the scale ceiling, not a step toward or away from consciousness-simulation.
- **Phase 2 scope leak.** Fact typing nudges toward mentalistic structure. If step 2 implementation grows typed relations (e.g. `relation` constrained to a vocabulary), it is scope leak — Phase 2 designs the vocabulary against measured need, not during this build.

---

## 8. Verification of this plan

Internal checks before implementation:

1. **Each decision from [phase-1.5-scale-recall-plan.md](phase-1.5-scale-recall-plan.md) §7 is resolved with a rationale.** ✓
2. **Each step has a numeric gate and a kill criterion.** ✓
3. **Phase 1 regression is a hard gate (step 5), not an afterthought.** ✓
4. **Phase 2 pre-payment is explicit** (typed facts + back-pointers = mentalistic hook + Principle B grounding; UPDATE semantics = reconsolidation hook). ✓
5. **E1 before E2 saves cost** (per the "experiment commitment" decision authorized 2026-10-06). ✓
6. **The locked-after-step-2 prompt discipline matches the Phase 1 reader-prompt discipline** ([phase-1-benchmark-adapters-plan.md](phase-1-benchmark-adapters-plan.md) §8 "reader-prompt as hidden parameter"). ✓

End-to-end validation (actually hitting E2 F1 ≥ 0.70) is the output of implementation, not of this plan.
