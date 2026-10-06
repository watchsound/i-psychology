# Option α — Minimal direct adapter

**Status (2026-10-05):** plumbing landed for EpBench; real-data gate validation blocked on Figshare dataset download.
**Spec:** [phase-1-benchmark-adapters-plan.md §3 Option α](../../../phases/phase-1-benchmark-adapters-plan.md).

Direct substrate answer only: `recall(question).content` is treated as the candidate answer. Deterministic scoring where possible (set-F1 for list-structured answers; vendored scorer for benchmark-specific tricky scoring). No LLM reader step, no RAG re-composition. Judge used only where unavoidable (free-form LoCoMo splits).

## Measurements landed

- [epbench-fixture-smoke.json](epbench-fixture-smoke.json) — 2026-10-05, synthetic fixture (`tests/fixtures/epbench_mini/`, 2 instances / 5 questions), aggregate mean F1 = **0.0**. **Expected per plan** — direct-mode `recall.content` is a reconstructed token bag, not a comma-separated list of entity/place names, so set-F1 against list-structured gold is zero by construction. This is the signal that confirms the plan §3 Option α risk ("likely under-performs on narrow factual QA") and the step-3 RAG reader is the path to the floor on this benchmark.

## Real-data integration path (EpBench)

Real-data runs require the EpBench dataset bundle:

1. Download the bundle from https://doi.org/10.6084/m9.figshare.28244480.
2. Extract to a local directory. Inside, each universe (e.g. `Udefault_Sdefault_seed0`) contains `df_qa.parquet` (questions) + `books/*/book.json` (events) + metadata parquets.
3. Write a `parquet → epbench_shim.jsonl` converter (one-off; see schema in [src/psychology_base/eval/benchmarks/epbench.py](../../../../src/psychology_base/eval/benchmarks/epbench.py) `_ShimEvent` / `_ShimQuestion`). Converter reads: event's `(t, s, e, c)` indices + joined paragraph text → shim event; `df_qa` row → shim question (`question`, `correct_answer`, `retrieval_type`, `n_items_correct_answer`).
4. Point the loader at the shim via `--data-root <path>` (the CLI resolves `root / "epbench_shim.jsonl"`) or by instantiating `EpBenchLoader(shim_path=...)` directly.

The converter is NOT yet in-repo. It's one of the two blockers to running the real-data step-2 gate. The other blocker is the direct-mode F1-floor finding above — even with real data, Option α cannot hit F1 ≥ 0.70 on list-structured EpBench questions. Step 3 (RAG mode) is the actual path to the floor; this directory records α's honest floor-contribution.

## Plan-wording issue surfaced

The step-2 gate as written in [phase-1-benchmark-adapters-plan.md §5](../../../phases/phase-1-benchmark-adapters-plan.md#5) says *"EpBench F1 ≥ 0.70 on a 50-instance smoke subset in direct mode."* The fixture smoke result shows **direct mode cannot hit this gate** on EpBench by construction. The gate should be revised to:

- *Step 2:* EpBench loader + direct-mode plumbing works + direct-mode baseline recorded (regardless of magnitude). ✓ met.
- *Step 3:* EpBench F1 ≥ 0.70 in Option γ RAG mode on real data. (unmet — awaits RAG step.)

Flagged for triage; not fixing the plan unilaterally.

## Kill criterion

Per the plan: if all three benchmarks miss the floor by > 20pt in direct mode, promote to Option γ before further tuning. The EpBench finding is a strong early signal that this will trigger on list-structured benchmarks; LongMemEval and LoCoMo short-form may behave differently. Record the kill in `comparison.md`, do not delete this directory.

## What each future numeric artifact must record

- Benchmark (LongMemEval / LoCoMo / EpBench)
- Split (full / smoke-50 / sealed-100 / fixture-mini)
- Measured metric + seeds (3-seed median for real runs; single-seed noted for fixture smokes)
- Floor target per [roadmap.md §61-63](../../../roadmap.md) — pass/fail
- SUT git SHA + CompanionSystem config (`pain_gain`, `companion_only_suppress`, `byte_cap`, `witness_active`)
- Timestamp
