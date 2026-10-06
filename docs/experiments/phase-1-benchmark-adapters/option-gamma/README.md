# Option γ — Dual-mode adapter

**Status:** pending — no measurements yet.
**Spec:** [phase-1-benchmark-adapters-plan.md §3 Option γ](../../../phases/phase-1-benchmark-adapters-plan.md).

Dual-mode: both direct (α-equivalent — `recall.content` as answer) and RAG (reader LLM composes final answer from `recall.content` + `fragments_used`). Both numbers reported side-by-side; the gap between them is diagnostic (if γ-RAG ≫ α-direct, the reader is doing the work, not the substrate).

Reader prompt is **locked** at step 3 and does not vary across runs; prompt-level iteration counts as substrate iteration, not adapter iteration.

## When numbers land

Each run adds a row-grouped artifact under this directory. Minimum fields per run:

- Benchmark + split (as Option α)
- **Both** direct-mode and RAG-mode numbers (same instance set, same seeds for direct; single-seed for RAG with judge-model + version pinned)
- Judge model + version used + cache hit-rate
- Reader prompt SHA
- Floor target pass/fail per mode
- SUT config + git SHA
- Timestamp

## Kill criterion

Per the plan: if direct-mode and RAG-mode scores are within 2pt across all three benchmarks after step 4 full-run, the reader is a no-op — drop to Option α to save complexity. Record in `comparison.md`, keep this directory for backtrack reference.
