# Option M3 — Affective-resonance retrieval (Emotional-RAG dual-vector)

**Status:** SHIPPED 2026-10-07 — B3 affect-match @ K=3 = 1.000 (gate ≥ 0.50), resonance recall = 1.000 (gate ≥ 0.60); 449 tests green under default `affect_weight=0.0`; measurement at [b3-run-01.json](b3-run-01.json).
**Companion plan:** [docs/phases/phase-2-mentalistic-plan.md](../../../phases/phase-2-mentalistic-plan.md) §5 Option M3.
**Builds on:** [Option M1 — Merkle-DAG fragments](../option-M1-merkle-fragments/README.md) (shipped) + [Option M2 — perspective-tagged fragments](../option-M2-perspective-tagged/README.md) (shipped).

## Mechanism

Fragments already carry an affect dict (`{valence, arousal}` populated by every B1/B2 scenario — see e.g. [b2_perspective_mother.yaml](../../../../benchmarks/scripts/b2_perspective_mother.yaml) `affect: { valence: -0.5, arousal: 0.4 }`). M3 **lights up the retrieval side**: the reconstruction score gains an **affective-resonance** term that lifts fragments whose stored affect matches the current cue-side affect.

- **Dual-vector per fragment** = (semantic tokens for Jaccard/embedding, `(valence, arousal)` PAD-2 scalar for affect). Zero schema change — Fragment.affect is already a dict; M3 standardizes the `valence` + `arousal` keys and adds the retrieval-time term.
- **Cue-side affect** comes from `context` (scenario YAML sets it; application layer infers it from user messages via `LLMRouter.chat` on SIMPLE tier). No biometric input — PAD scalars from text only, per Chinese 《办法》 floor.
- **Affect-resonance score** = `1 − euclidean_distance_in_pad / sqrt(8)` (normalized to [0, 1] since PAD axes are [-1, 1]). **Additively** composed with Jaccard-relevance inside `reconstruct()`: `score = raw × context_weight + affect_weight × pair_affect_score(frag.affect, cue_affect)`. Matches the plan's `γ·relevance + δ·affective-resonance` formula; multiplicative was the first-draft shape but cannot rescue a zero-relevance affect-matched fragment (the "thin semantic overlap, strong affective overlap" case this mechanism is specifically for). Recency and importance stay unchanged.

**Big-Five Neuroticism head** is deferred to Phase 2.1 — PAD-2 is the compact, interpretable, computable-from-stored-dicts path. Upgrade path: swap `_cue_affect(context)` + `_pair_affect(fragment, cue_affect)` for a Neuroticism-head call when it ships.

## Target gaps

- **Gap 3 (共情).** User enters a session distressed. Without affect resonance, recall surfaces the semantically-most-relevant fragments regardless of affect — a user talking about work stress sees the "I got the promotion" fragment because "work" tokens match. With M3, high-arousal-negative cues preferentially surface prior high-arousal-negative fragments; the AI can echo "last time this came up you also said you weren't sleeping — is that happening again?" rather than default-mirror.
- **Reconsolidation × affect (interaction with M1).** When affect drifts across a chain (B1 scenario: anchor valence −0.7 → leaf valence +0.4), M3 lets the AI pull up the version whose affect matches the current session, not just the latest-written version. The AI can honor where the user is today without discarding earlier versions.
- **Clinical-insight × affect (interaction with M2).** `(ai, user)` scope + affect filter together surface the AI's observations of past states matching the current one — the substrate for "last session's affect looked like this one; want to compare notes?" moves.

## Prior art

- **Emotional RAG** ([arXiv:2410.23041](https://arxiv.org/abs/2410.23041), reference impl [BAI-LAB/EmotionalRAG](https://github.com/BAI-LAB/EmotionalRAG)) — dual-vector per memory (semantic + emotion); emotion vec drives retrieval re-rank. Validates the mechanism on dialogue-memory benchmarks. We adopt the retrieval-score term but keep reconstruction (Principle B) rather than Emotional-RAG's retrieve-then-rerank shape.
- **Generative Agents** (Park et al., 2023) — importance scoring is a weighted retrieval term; M3's affect-resonance term lands as a sibling to the existing recency+importance+relevance mix.
- **Appraisal-theory affect models** (OCC, PAD) — PAD-2 (valence × arousal) is the standard low-dim affect vector; dominance axis dropped for v1 since it's rarely stable across short texts.

## Build steps

1. **Affect standardization** — document in [adapters.py](../../../../src/psychology_base/eval/adapters.py) that `Fragment.affect` SHOULD carry float `valence ∈ [-1, 1]` + `arousal ∈ [-1, 1]` keys when known. Fragments without them participate in retrieval with a scope-less-equivalent default (affect-resonance term = 0, no penalty). No code-change — this is a conventions-doc update.
2. **Affect resonance helper** — new [psychology_base.companion.affect_resonance](../../../../src/psychology_base/companion/affect_resonance.py): `pair_affect_score(frag_affect, cue_affect) -> float ∈ [0, 1]`, `cue_affect_from_context(context) -> dict | None`.
3. **Reconstruction score integration** — [reconstruct()](../../../../src/psychology_base/companion/reconstruction.py) gains an `affect_weight: float = 0.0` kwarg. When > 0 and both `cue_affect` + `fragment.affect` are populated, add `affect_weight × pair_affect_score(...)` to the per-fragment score. Default `affect_weight=0.0` preserves M1+M2 behavior byte-for-byte.
4. **Recall wiring** — [CompanionSystem.recall](../../../../src/psychology_base/companion/system.py) accepts optional `affect_weight: float | None = None`; when None, uses `self.affect_weight` (new init kwarg, default `0.0`). Scope and affect compose: scoped recall with affect weighting pre-filters by scope then re-ranks by affect-weighted Jaccard.
5. **B3 scorer module** — new [psychology_base.eval.benchmarks.b3_affect_resonance](../../../../src/psychology_base/eval/benchmarks/b3_affect_resonance.py):
   - `resonance_recall(recalls, affect_tagged_fragments, cue_affects) -> float` — fraction of recalls whose top-K fragments carry affect within target distance of the cue affect
   - `affect_match_at_k(recall, cue_affect, k) -> float` — per-recall diagnostic
   - `over_disclosure_rate(recalls, ground_truth) -> float` — informational for v1 (needs user feedback)
   - `B3Scores` dataclass with `passes_gate()`
6. **Scripted scenario** — [benchmarks/scripts/b3_affect_resonance.yaml](../../../../benchmarks/scripts/b3_affect_resonance.yaml): 5-session scenario where the user has three prior high-arousal-negative episodes (sleep loss, work pressure, breakup aftermath) and three low-arousal-positive episodes (quiet weekends). Session 6 cue is high-arousal-negative; M3 must surface the three negative-arousal fragments ahead of the positive-arousal ones even when semantic overlap is thinner.
7. **Phase 1 + M1 + M2 regression** — 409 tests + B1 preservation 1.0 + B2 attribute reconstruction 12/12 must stay green with `affect_weight=0.0` default.
8. **LoCoMo monitor** — informational. Default `affect_weight=0.0` keeps the LoCoMo code path byte-identical to M2 (which is byte-identical to M1 under scope=None). Re-running only needed if we A/B M3's affect-on path vs the default-off baseline, which belongs under B3 not LoCoMo.
9. **B3 scripted scenario integration test** — [tests/test_b3_affect_resonance_scenario.py](../../../../tests/test_b3_affect_resonance_scenario.py) parallel to [test_b2_perspective_mother_scenario.py](../../../../tests/test_b2_perspective_mother_scenario.py).

## Exit gates (go/no-go for M3)

| Gate | Target | Measured by |
|---|---|---|
| B3 resonance recall | ≥ 60% of affect-cued recalls surface ≥ 1 affect-matched prior fragment in top-K | `resonance_recall` on scripted scenario |
| B3 affect-match @ K=3 | ≥ 50% of top-3 fragments within PAD-distance ≤ 0.4 of cue affect | `affect_match_at_k` on scripted scenario |
| Over-disclosure rate | < 10% (informational for v1) | user-reported; v1 lands as placeholder |
| M1 + M2 regression | B1 preservation 1.0 + B2 attribute reconstruction 12/12 + 409 tests | re-run `b1-run-01` + `b2-run-01` with `affect_weight=0.0` |
| LoCoMo monitor | informational (no gate; byte-identical to M2 under default) | optional 3-instance R5 run |

## Kill criteria

- B3 resonance recall < 30% → the affect vector is not actually separating positive-vs-negative-arousal fragments at retrieval time; diagnose by inspecting per-pair scores before continuing
- Enabling `affect_weight > 0` degrades the B2 scope-filtered pool (e.g. drops attribute reconstruction below 10/12) → affect term is swamping relevance; cap the weight or add a floor on Jaccard before applying affect
- Enabling `affect_weight > 0` breaks the 409-test regression under default-off → the gated code path has leaked into the default; must fix before shipping M3

## What this does NOT include

- No automatic affect extraction from user utterances inside the companion loop — v1 reads affect from `Observation.affect` (set by the scenario YAML or the application layer). An `LLMRouter.chat_batch`-driven affect extractor is a Phase 2.1 component
- No Big-Five Neuroticism head (deferred to Phase 2.1 per the mechanism note)
- No PAD-3 (dominance axis) — unstable in short texts, drop for v1
- No multimodal affect inputs (voice F0, FACS, keystroke) — 《办法》 floor plus the applications are text-first
- No consolidation of fragments by affect cluster — the per-fragment affect vector feeds retrieval, not a new sub-store (Principle B: process, not store)
- No "mood regression" semantic-therapy move — M3 provides the substrate; the companion-side policy for when to use it is application-layer

## Artefacts (to be produced)

- [src/psychology_base/companion/affect_resonance.py](../../../../src/psychology_base/companion/affect_resonance.py) — `pair_affect_score`, `cue_affect_from_context`
- [src/psychology_base/companion/reconstruction.py](../../../../src/psychology_base/companion/reconstruction.py) — `affect_weight` kwarg + per-fragment affect-boost term
- [src/psychology_base/companion/system.py](../../../../src/psychology_base/companion/system.py) — `affect_weight` init kwarg; `recall(affect_weight=...)` kwarg; metadata key `affect_weight`
- [src/psychology_base/eval/benchmarks/b3_affect_resonance.py](../../../../src/psychology_base/eval/benchmarks/b3_affect_resonance.py) — scorers
- [benchmarks/scripts/b3_affect_resonance.yaml](../../../../benchmarks/scripts/b3_affect_resonance.yaml) — scenario
- [tests/test_affect_resonance.py](../../../../tests/test_affect_resonance.py) — pair_affect + reconstruct-weighting unit tests
- [tests/test_b3_affect_resonance.py](../../../../tests/test_b3_affect_resonance.py) — scorer tests
- [tests/test_b3_affect_resonance_scenario.py](../../../../tests/test_b3_affect_resonance_scenario.py) — integration test
- Measurement output: [b3-run-01.json](b3-run-01.json)
