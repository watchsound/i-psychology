# Phase 1 benchmark adapters — plan (LongMemEval / LoCoMo / EpBench)

**Date set:** 2026-10-05
**Companion docs:** [benchmarks.md](../benchmarks.md) §120-130 (standard SOTA floor), [roadmap.md](../roadmap.md) §61-63 (Phase 1 exit criteria), [phase-1-plan.md](phase-1-plan.md) (Option C substrate the adapters evaluate).
**Process:** six-step phase-planning directive — PDF research → online research → OSS study → brainstorm → options → goal cross-check — same discipline as [phase-1-plan.md](phase-1-plan.md).
**Scope:** this plan covers *adapter plumbing* only — dataset loading, SUT invocation, scoring. The adapters evaluate the Phase 1 Option C `CompanionSystem` built in step 4; they do not modify it. The gate is a don't-regress floor, not a leaderboard chase.

---

## 1. Research findings (what the adapters must contend with)

### 1.1 From the reference PDFs (foundational framing)

- **《AI Agent Memory Framework Analysis》, 《智能体记忆架构研究》 — benchmark taxonomy.** The PDFs name LongMemEval, LoCoMo, EpBench as the three load-bearing long-term-memory benchmarks and classify each by what capability it probes: LongMemEval = long-horizon factual recall across many sessions; LoCoMo = multi-session consistency / coreference / temporal reasoning; EpBench = temporal ordering and episodic-event retrieval. The PDFs frame these as regression guards, not acceptance gates, which matches our [benchmarks.md](../benchmarks.md) framing.
- **Reconstruction stance consequence.** The PDFs assume a retrieve-and-answer surface. Our Option C substrate reconstructs rather than retrieves — the adapter cannot assume `recall(question)` returns an answer string that is already an answer; it may return a *rebuilt fragment composition* that needs a downstream reader step to answer the benchmark's question. This is the single biggest adapter-design decision (addressed in §2.5).
- **Mem0 94.4% / 92.5% is the published reference.** Our floor is 90% / 90% (parity acceptable, not SOTA on this axis — intentional per [sota-goal.md](../sota-goal.md)).

### 1.2 From online research (benchmark state 2026-Q3)

- **LongMemEval** (Wu et al., arXiv 2404.13940) — 500 questions × 50-session histories per instance. Five question types: single-session recall, multi-session reasoning, temporal reasoning, knowledge update, abstention. Evaluation is `gpt-4o-as-judge` on short-form answers against gold answers. The 94.4% Mem0 number is on the "oracle session" split (session labeled); the "full history" split is harder and more honest.
- **LoCoMo** (Maharana et al., arXiv 2402.17753) — 10 ultra-long dialogues (~300 turns, ~9k tokens) with 2000+ questions across five categories (single-hop, multi-hop, open-domain, temporal, adversarial). Evaluation combines F1 (short-form) + BLEU + judge for open-ended. Mem0's 92.5% is single-hop accuracy; aggregate is lower (~68% in the Mem0 paper).
- **EpBench** (Huang et al., arXiv 2501.13121) — ~2000 synthetic story-event sequences with questions about **what-when-where-who** quadruples. Scored by F1 over quadruple elements. Published top-LLM scores < 0.36 F1; our 0.70 floor is aggressive-but-reachable with a working memory substrate.
- **Dataset availability.** LongMemEval is on HuggingFace Datasets (`xiaowu0162/longmemeval`); LoCoMo is on GitHub (`snap-research/locomo`); EpBench is on GitHub (`DT-AI/EpBench`). All three are public, text-only, English (LongMemEval has a Chinese split that we can defer).
- **Judge-model dependency.** All three benchmarks have an LLM-judge step for free-form answers. Judge choice (GPT-4o vs Claude Sonnet vs open-weights) can swing scores ±5pt. The adapter needs a configurable judge, and the chosen judge needs to be the one Mem0 reported with to make the parity claim honest.

### 1.3 From OSS study (how shipped systems adapt to these benchmarks)

| Project | How it loads data | SUT surface | Scoring | Lessons for us |
|---|---|---|---|---|
| **Mem0** | Loads LongMemEval JSON directly; feeds each turn as a `memory.add(user_message, agent_message)` pair, then calls `memory.search(question)` → feeds top-k into GPT-4o as context → judges output | retrieve-and-synthesize | separate judge script; releases per-question logs | Judge is a separate process — adapter can be dumb about scoring. **Mem0's turn-ingestion maps directly to our `observe()`.** |
| **Letta** | Pre-ingests entire conversation into archival memory via `insert_passage`; at Q time, agent uses tool call to search | tool-use agent | LMM judge | Insertion-first then query is the common pattern; our substrate also prefers this (observe → recall). |
| **Zep** | Ingests messages into dual-time graph via `memory.add`; at Q time calls `memory.search(session_id, question)` | retrieve | F1 + judge | Zep's number is lower (71.2%) because the search is schema-constrained — a cautionary tale about adapter choices leaking into scores. |
| **Generative Agents (ref impl)** | — (no benchmark adapter shipped) | — | — | None — community-written adapters exist but are not canonical. |

**Operational constraints the shipped code reveals:**
- **Ingestion-time cost dominates.** LongMemEval has 500 instances × ~50 sessions × ~20 turns ≈ 500k observe calls. If `observe()` is O(n) in fragment count (Hebbian edge updates in our system are), total ingestion is quadratic. Adapter must support batched / parallelized runs per instance (instances are independent).
- **Judge cost dominates evaluation.** 500 questions × GPT-4o judge call ≈ $5-15 per run depending on prompt length. Need a judge cache keyed on `(question, gold, candidate)` so re-runs after a code change don't re-pay for identical judgments.
- **Instance-level isolation is non-negotiable.** Each benchmark instance must get a fresh `CompanionSystem()` — leaking fragments across instances inflates recall trivially (this is the "memory of the memory" risk flagged in [benchmarks.md](../benchmarks.md) §148).

### 1.4 The adapter problem restated

Three benchmarks, one SUT, three isolated concerns:
- **Format**: each benchmark has a different JSON schema (LongMemEval nested session/turn, LoCoMo flat per-dialogue, EpBench event-list + QA-list). Adapter needs a per-benchmark loader → common `ScriptedScenario`-shaped intermediate representation.
- **SUT invocation**: common for all three — `observe()` per turn, `recall(question)` at question time. One decision point: when `recall()` returns a fragment composition rather than an answer string, do we add a downstream LLM-reader step inside the adapter, or do we require the SUT to produce answerable text?
- **Scoring**: per-benchmark judge + metric. Cacheable, parallelizable, orthogonal to the loader + SUT.

The three concerns are cleanly separable, which maps directly to Option 2 below.

---

## 2. Brainstorm (unfiltered)

### 2.1 Dataset loading

- **L1** — Direct HuggingFace + GitHub pulls at harness-run time, no vendoring.
- **L2** — Vendor a frozen snapshot into `reference-docs/benchmarks/` for reproducibility.
- **L3** — Hybrid: lazy download on first run, local cache under `~/.cache/psychology_base/benchmarks/` with hash pinning.
- **L4** — Per-benchmark loader class, each converting to a common `BenchmarkInstance` dataclass. Decouples schema differences from the harness.
- **L5** — Translate to our existing `ScriptedScenario` schema (reuse `SessionPlayer`). Zero new ingestion code.

### 2.2 SUT invocation surface

- **I1** — Direct: `observe()` per turn, `recall(question)` at Q time, return `recall.content` as the candidate answer.
- **I2** — Direct + reader: `recall(question)` returns fragment composition; a small LLM reader step composes an answer from `fragments_used` + `content`.
- **I3** — Retrieval-only: pass `fragments_used` to a canned prompt template → external LLM composes answer (closest to Mem0's pattern).
- **I4** — Tool-use agent wrapping: wrap `CompanionSystem` as a tool the benchmark's QA agent calls; most framework-faithful, most expensive.
- **I5** — Dual-mode: adapter supports both I1 (direct substrate answer, cheap) and I3 (retrieval-augmented, expensive but comparable to published numbers). Report both.

### 2.3 Scoring

- **S1** — Vendor each benchmark's reference scoring script verbatim, call it from the adapter.
- **S2** — Reimplement scoring from each benchmark's paper spec. (High risk of divergence; cheap only in prompt tokens.)
- **S3** — Judge-model-based scoring with a configurable judge; cache judgments on `(question, gold, candidate)` SHA.
- **S4** — Hybrid: exact-match + F1 for short-form answers (deterministic, no judge needed), judge only for free-form / open-domain splits.

### 2.4 Harness orchestration

- **O1** — CLI script per benchmark (`scripts/run_longmemeval.py`, etc.). Simple, explicit, easy to debug.
- **O2** — Unified CLI with `--benchmark {longmemeval,locomo,epbench}` flag.
- **O3** — pytest-driven: each benchmark is a parametrized test that runs on CI. (Expensive — judge calls on CI not viable.)
- **O4** — Hybrid: pytest-driven **small-sample smoke** (10 instances, no judge, exact-match only) on CI + CLI script for full-suite judge runs.

### 2.5 The reconstruction-answer bridge

This is the one novel question for our substrate. Shipped systems retrieve raw text and let an LLM compose an answer. Our `recall.content` is a reconstructed composition — closer to an answer than to a retrieval list, but not formatted for QA.

- **B1** — Train / prompt an answer-head: a small LLM reader takes `(question, recall.content, fragments_used)` and returns a short-form answer. Standard RAG-style.
- **B2** — Bypass the reader: assume `recall.content` IS the answer and let the judge decide. Honest to the reconstruction stance; likely under-performs on narrow factual QA.
- **B3** — Hybrid-mode reporting: run both B1 and B2, report both numbers. The gap between them is itself diagnostic — if B2 is competitive, the substrate is answering directly; if B1 is much higher, the reader is doing the work, not the memory.

---

## 3. Options (compressed to 3 shippable designs)

### Option α — Minimal faithful adapters (vendored data, deterministic scoring where possible)

- **Loading:** L2 + L4 — vendor frozen snapshots, per-benchmark loader class → `BenchmarkInstance`.
- **SUT invocation:** I1 (direct) only.
- **Scoring:** S4 — exact-match / F1 for short-form, judge only where unavoidable; S1 fallback for benchmark-specific tricky scoring (EpBench quadruple F1).
- **Orchestration:** O1 — one CLI per benchmark.
- **Bridge:** B2 — `recall.content` is the answer.
- **Risks:** B2 likely under-reports on narrow factual QA; vendoring adds repo weight (~50MB).
- **Pass evidence:** meets 90/90/0.70 floor on at least one of the three.
- **Kill criterion:** if all three benchmarks miss the floor by > 20pt in B2 mode, the direct substrate is not QA-shaped; promote to Option γ before further tuning.

### Option β — Published-parity adapters (RAG-mode, judge-faithful)

- **Loading:** L3 — lazy download + local cache + hash pinning.
- **SUT invocation:** I3 — retrieve-augment: Companion returns fragments + reconstruction; external LLM composes final answer. This is the invocation pattern Mem0/Zep use.
- **Scoring:** S1 for each benchmark — vendor reference scoring scripts.
- **Orchestration:** O2 — unified CLI, O4 — pytest smoke subset.
- **Bridge:** B1 — explicit reader step.
- **Risks:** judge cost ($5-15/run × iteration count); heaviest implementation; results heavily depend on reader-prompt engineering → risks grading the reader, not the substrate.
- **Pass evidence:** parity numbers against Mem0 published scores.
- **Kill criterion:** reader-prompt tuning moves scores more than substrate changes do → reader is dominating. Fall back to Option γ dual-reporting.

### Option γ — Dual-mode adapters (direct + RAG, both reported)

- **Loading:** L3 (lazy cache) + L4 (per-benchmark loader → `BenchmarkInstance`).
- **SUT invocation:** I5 — both I1 (direct) and I3 (RAG with reader) runs; harness reports both numbers side-by-side.
- **Scoring:** S4 for short-form (deterministic), S1 for benchmark-specific (EpBench F1), S3 for free-form with judge cache.
- **Orchestration:** O4 — pytest smoke on CI (direct-mode, deterministic subset only), CLI for full runs.
- **Bridge:** B3 — both reported, gap is the diagnostic signal.
- **Risks:** two surfaces to maintain; result tables are twice as wide. Mitigated by making I3 strictly derivative of I1 (same recall path, different final-composition step).
- **Pass evidence:** either mode hits the 90/90/0.70 floor; both reported so the reader's contribution is visible. If direct-mode fails and RAG-mode passes, the Phase 1 gate is formally met (floor is a don't-regress threshold, not an architectural purity statement) but the gap is flagged as a Phase 2 attention item.
- **Kill criterion:** if direct-mode and RAG-mode scores are within 2pt across all three benchmarks, the reader is a no-op — drop to Option α to save complexity.

---

## 4. Goal cross-check (against the two north-star applications)

Each option is tested against the question: *do its results tell us something our applications care about?*

| Option | 学习陪伴 relevance | 情感陪伴 relevance | Verdict |
|---|---|---|---|
| **α — Minimal direct** | Shows whether the substrate's reconstructions carry *answerable* content — relevant for Socratic probes where the AI must know the concept state. | Shows whether reconstructions carry answer-grade continuity across sessions — relevant for "remember when we…" moments. BUT: companion-dialogue is less QA-shaped than tutoring, so B2 under-reporting may be unfair here. | **Partial.** Signals cleanly for 学习陪伴; under-represents 情感陪伴. |
| **β — RAG-mode parity** | Guards against regression on the benchmark SOTA, so Phase 2 can be built on a known-good substrate. BUT: tells us little about whether the substrate itself is 学习陪伴-grade — reader does the final step. | Same floor-guard argument. Same reader-dominance caveat. | **Floor-guard only; no application-signal.** |
| **γ — Dual-mode** | Direct-mode is the application-signal (does the substrate carry answer-grade reconstructions). RAG-mode is the floor-guard. The gap is a diagnostic. | Same decomposition — direct-mode is honest-to-companion, RAG-mode is honest-to-floor. | **Yes, both goals addressed, gap is diagnostic.** |

**Chosen: Option γ.** Dual-mode reporting is the only option that both guards against regression on the published floor AND surfaces an application-relevant signal (direct-mode performance). The complexity cost is bounded because I3 is strictly a post-step on top of I1 — no second observe path, no second recall path.

**Held in reserve: Option α.** If early implementation reveals that the RAG-mode reader is doing nothing substantive (dual-mode numbers within 2pt), drop to α and save the complexity. This is a one-week-into-implementation decision, not a plan-time decision.

**Dropped: Option β.** Floor-guard-only is insufficient for a Phase 1 that is otherwise fully application-anchored. If we only want parity numbers, we are grading someone else's design, not ours.

---

## 5. The chosen design (Option γ) — build order

Four build steps, each with its own gate. Benchmarks ordered by implementation cost ascending (EpBench's schema is cleanest, LoCoMo's is messiest).

| Step | What to build | Local gate |
|---|---|---|
| 1 | `BenchmarkInstance` dataclass + loader-protocol + judgment cache + CLI skeleton (unified `--benchmark` flag) | Empty-run on all three benchmarks loads instances without errors; cache round-trips a judgment |
| 2 | EpBench loader + direct-mode (I1) scoring + reference quadruple-F1 scorer (S1) | EpBench F1 ≥ 0.70 on a 50-instance smoke subset in direct mode. Full run deferred to step 4. |
| 3 | LongMemEval loader + RAG-mode (I3) reader-step + judge integration (S3 with cache) | LongMemEval ≥ 90% on a 50-instance smoke subset in RAG mode; direct-mode number also reported (no gate on direct-mode yet) |
| 4 | LoCoMo loader + full-suite CLI runs + pytest smoke tests + dual-mode result table | LoCoMo single-hop ≥ 90% (parity); all three benchmarks' full-run numbers on the dual-mode result table; pytest smoke (direct-mode, deterministic subset only, no judge) passes on CI |

**Judge discipline.** Judgments are cached on `(benchmark, question_id, gold, candidate_sha)` keys under `~/.cache/psychology_base/judgments/`. Cache survives code changes; invalidated only when gold or benchmark data changes. Judge model is **configurable but pinned per report** — the result table records which judge produced which numbers.

**Instance isolation (non-negotiable).** Each benchmark instance gets `CompanionSystem()` from scratch. The harness MUST assert `cs.inspect()["fragment_count"] == 0` at instance start; this guards against the memory-of-memory contamination risk flagged in [benchmarks.md](../benchmarks.md) §148.

**Seed discipline.** The 3-seed median rule from [phase-1-plan.md](phase-1-plan.md) §5 applies here only for the direct-mode gate check — RAG-mode numbers are reported single-seed because the external reader LLM is the dominant stochasticity source. Seeds are logged either way.

**Vendoring vs. caching.** Lazy-download-and-cache (L3) rather than vendor. Repo weight stays flat; hash pinning in `benchmarks.lock.toml` guards against upstream data shift mid-project. If a benchmark's hosting disappears (small risk for recent arXiv-adjacent datasets), we vendor the snapshot at that point, not preemptively.

---

## 6. What this plan does NOT decide

Deliberately left for the implementation phase or deferred:

- **Judge model choice.** Will be pinned at implementation time per benchmark; Mem0 uses GPT-4o, which is the published-parity choice but not necessarily the one we have cheapest access to. First-week decision.
- **Reader prompt template** (I3). Not written here; will be derived from Mem0's released prompts where available, otherwise written to minimum-complexity (question + reconstruction → short answer).
- **Chinese-split handling** for LongMemEval. Deferred — the English split is what Mem0's published number is on, and parity on the English split is the Phase 1 target.
- **Longitudinal / multi-run averaging infrastructure.** Phase 1 reports point-estimates with seed-logs; averaging across runs is a Phase 2 concern once we're iterating on substrate rather than adapter.
- **CI budget for the pytest smoke subset.** First-week spike will measure actual runtime; if direct-mode no-judge smoke exceeds ~2 min on a 50-instance subset, drop to 20-instance or move to nightly.
- **Dual-mode disagreement triage protocol.** If direct-mode and RAG-mode give substantively different Phase 1 verdicts (one passes, one fails), the response is a short investigation, not a procedural rule. Writing the rule now would be premature.

---

## 7. Critical files

New artifacts the implementation will produce (not written in this plan):

- `src/psychology_base/eval/benchmarks/__init__.py` — package marker.
- `src/psychology_base/eval/benchmarks/instance.py` — `BenchmarkInstance` dataclass + loader protocol.
- `src/psychology_base/eval/benchmarks/longmemeval.py`, `locomo.py`, `epbench.py` — per-benchmark loaders + scorers.
- `src/psychology_base/eval/benchmarks/reader.py` — I3 RAG reader-step (single, shared across benchmarks).
- `src/psychology_base/eval/benchmarks/judge.py` — judge integration + cache.
- `src/psychology_base/eval/scripts/run_benchmark.py` — unified CLI (`--benchmark {longmemeval,locomo,epbench} --mode {direct,rag,both}`).
- `tests/test_benchmark_adapters.py` — direct-mode deterministic smoke subset.
- `benchmarks.lock.toml` — hash-pinned dataset versions.

Files to extend (not rewrite):
- [docs/baselines.md](../baselines.md) — add a benchmark-floor section once numbers land.
- [docs/roadmap.md](../roadmap.md) §61-63 — annotate with the measured numbers as they come in.

No modifications to `CompanionSystem`, `reconstruction.py`, or `witness.py` — the adapters are consumers, not substrate changes.

---

## 8. Risks and open questions

- **Judge drift.** GPT-4o version shifts can move numbers. Mitigation: pin model version in config; record it in the result table.
- **Reader-prompt as hidden parameter.** I3's reader prompt is the thing that most moves RAG-mode numbers. Risk: tuning the reader to pass the floor rather than tuning the substrate. Mitigation: lock the reader prompt in step 3 and don't change it between runs; variation beyond that lock-point is treated as substrate iteration, not adapter iteration.
- **Benchmark overfitting.** Running the same 500-question LongMemEval instance set over and over while tuning substrate parameters is itself a form of overfitting. Mitigation: hold out 100 instances as a sealed-until-Phase-1-exit split; final gate check runs on sealed split only.
- **Instance-count vs. iteration budget.** 500 instances × dual-mode × 3 seeds × judge calls will not be affordable per iteration. Development workflow: 50-instance smoke on every substrate change, full-run only at explicit gates.
- **Dependency: LLM API access.** The project does not currently require an LLM API key. Benchmark adapters introduce that dependency for RAG-mode and judge-mode. Mitigation: direct-mode + deterministic-scoring subset works with zero external API, so CI smoke tests do not need LLM access.

---

## 9. Verification how-to (end-to-end)

Since no code is produced in *this* plan, verification is of the architectural plan itself:

1. **Internal consistency.** Options α/β/γ cover the three natural stances (purity / parity / dual). ✓
2. **Honest attribution.** Dual-mode reporting makes it visible when the reader rather than the substrate is doing the work — directly addresses the hidden-parameter risk. ✓
3. **Alignment with [benchmarks.md](../benchmarks.md).** Three benchmarks match the standard-SOTA regression floor table §120-130 verbatim; thresholds match. ✓
4. **Alignment with [phase-1-plan.md](phase-1-plan.md) discipline.** Seed discipline, instance isolation, and gate-per-step structure carried forward. ✓
5. **Cost realism.** Judge-cost + instance-cost estimated in §1.3 and §8; smoke/full-run split in §5 addresses the iteration budget directly. ✓

End-to-end validation (actually running the three benchmarks on step-4 `CompanionSystem` and hitting the floor) is deferred to the implementation step that follows this plan.
