# Phase 2 — ceiling breakers

Phase 1.5-prime capped at **F1≈0.47** on conv-26 single-hop under judge-mode
across all three structural pivots (P1 judge redefinition / P2 embedder
ensemble / P3 cross-encoder rerank). The 0.70 gate did not fall out of that
architecture. Phase 2 attacks the ceiling with four candidate levers, each
targeting a different failure mode diagnosed in
[../phase-1.5-prime-structural-pivot/comparison.md](../phase-1.5-prime-structural-pivot/comparison.md).

| Option | Lever | Target failure | Prior-art cue |
|---|---|---|---|
| **R1** | Rerank over *all* fragments (bypass top-150 cosine pool) | 5 Bucket A questions where gold is cosine-ranked past 150 — rerank never sees them | Cross-encoder recall should scale with pool size; the top-150 cap was a cost gate, not a quality gate |
| **R2** | BM25 + semantic hybrid recall | Proper nouns (Sweden, Adoption, Single) that paraphrase-cue embeddings bury but exact-token BM25 finds trivially | Standard dense+sparse hybrid; most Mem0/Letta reference stacks run both |
| **R3** | Reader prompt redesign | Reader refuses "not stated" on q11 (Sweden), q40 (beach count), q60 (clarinet) when context *does* contain the answer | Phase 1.5 v6 showed prompt pressure trades Bucket C against Bucket B; need a more surgical rewrite than v6's blanket tightening |
| **R4** | Extractor swap qwen-turbo → qwen-max | Fact-store quality bound by SIMPLE-tier extraction; 5–8% of gold-bearing turns extract wrong (observed in `e1-audit-conv26-v4-promptTune.jsonl`) | Phase 1.5 extraction prompt is strong but qwen-turbo mis-attributes relations under high-density turns |

**Running order.** R1 first — directly addresses the diagnosed P3 failure mode
and reuses the cross-encoder adapter already shipped. R2–R4 require larger
new scaffolding (BM25 index / prompt-rewrite A/B / reconsolidation at ADVANCED
tier). Each option's directory holds its own `README.md`, result JSON/log,
and (where relevant) a per-row diagnosis.

**Verdict table.** See [comparison.md](comparison.md) (populated as options run).

**Gate.** Phase 1.5 target was single-hop F1 ≥ 0.70 under judge-mode on
conv-26. Phase 2 keeps the same gate — the architectural ceiling of 0.47
leaves ~0.23 of F1 to recover before E2 (10-instance full gate) becomes
economically worth running.
