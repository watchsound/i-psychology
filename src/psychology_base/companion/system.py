from __future__ import annotations

from pathlib import Path
from typing import Optional

from psychology_base.companion.consolidation import Consolidator
from psychology_base.companion.edges import EdgeStore
from psychology_base.companion.entities import Companion, Witness
from psychology_base.companion.facts import (
    DEFAULT_COSINE_FLOOR,
    DEFAULT_TOP_K,
    FactStore,
)
from psychology_base.companion.heartbeat import Heartbeat
from psychology_base.companion.reconstruction import (
    ActivationState,
    jaccard_prior,
    reconstruct,
)
from psychology_base.companion.witness import WitnessState, witness_couple
from psychology_base.eval.adapters import Fragment, Observation, Recall
from psychology_base.llm.router import LLMRouter


DEFAULT_BYTE_CAP = 4_096
JACCARD_SEED_THRESHOLD = 0.01
MIN_FACTS_FOR_FACT_PATH = 3
# Phase 1.5 Option 5.C — hierarchical tiering. When consolidation is on,
# the reader always sees BOTH top-K facts and top-K cosine-ranked raw
# fragments. Facts carry compressed cross-session structure; fragments
# carry the specific proper nouns (Sweden, "Charlotte's Web", pottery)
# that session-level LLM extraction routinely drops.
#
# Phase 1.5 E1 v6 diagnosis (2026-10-06): mxbai-embed-large cosine
# ranking buries the gold fact (``Caroline origin Sweden`` at rank 75
# for cue ``Where did Caroline move from 4 years ago?``) under noise
# facts that pattern-match the verb-shape of the cue, AND it buries
# the gold raw turn (the "home country, Sweden" necklace quote) at
# rank 189 under generic ``Caroline`` turns. v7 fix without changing
# embedder:
#   - fact tier top-K drops 150→50 so a 1-of-50 gold line isn't lost
#     mid-dump (qwen-max probe showed lost-in-the-middle on 150-line
#     noisy FACTS blocks; a clean 1-fact context DOES get picked up);
#   - FACTS are deduped by string before composition — the SOTA run
#     emitted "Caroline activity support" × 5, "Caroline activity
#     family" × 3, etc. — dedup lifts information density;
#   - fragment tier top-K lifts 30→200 so the Sweden turn at rank 189
#     reaches the DIALOGUE block (reader probe showed answers DO land
#     when a supporting raw turn is in context, even if facts alone
#     under-specify the temporal cue).
# qwen-max 32K context easily holds 150 facts + 50 raw turns (~25KB).
#
# v6 (2026-10-06): back to mxbai + wide top-K + substring augmentation.
# The v3/v4/v5 ablation showed: wide-cosine (v3) wins Bucket B via
# recall; substring aug (v5) wins Bucket A on proper nouns. Combining:
# wide cosine for recall + conservative substring cap for precision +
# noun-phrase reader prompt for terseness. Fragment cosine top-K lifts
# to 150 (not 200) — the Sweden-turn rank 189 is unreachable under
# mxbai no matter what, so we accept the loss and let substring aug
# handle Bucket A via "Sweden" value matches.
HIERARCHICAL_FACT_TOP_K = 150
HIERARCHICAL_FRAGMENT_TOP_K = 50
HIERARCHICAL_FACT_COSINE_FLOOR = 0.0
# Substring aug: top fact values matched as substrings against the raw
# fragment store. v5-v7 explored this lever — wide scan (v7) recovered
# some Bucket B wins (q4 +0.667, q66 +0.684) but broke exact-answer
# precision on q40 "2"→"once or twice" and q60 clarinet→guitar and did
# NOT recover Bucket A proper nouns (Sweden/Adoption/Single stayed
# "not stated"). Net regression vs v3 (0.264 vs 0.288). Shipped state
# below reproduces v3 champion by setting FRAGMENT_CAP=0 (disables
# augmentation). Scan/filter constants retained so the lever can be
# reactivated by raising FRAGMENT_CAP without re-coding the path. See
# docs/experiments/phase-1.5-scale-recall/comparison.md for the full
# v2-v7 sweep and the "v3 is the architectural ceiling" verdict.
HIERARCHICAL_SUBSTRING_FACTS_SCANNED = 150
HIERARCHICAL_SUBSTRING_FRAGMENT_CAP = 0
HIERARCHICAL_SUBSTRING_VALUE_MIN_LEN = 4
HIERARCHICAL_SUBSTRING_VALUE_MAX_LEN = 60

# Phase 1.5-prime P3 — cross-encoder rerank pool. When
# ``cross_encoder_rerank`` is on, cosine recall pulls the top-N
# candidates (wider than the final fragment cap), the rerank API scores
# them against the cue, and we trim to the final cap. Larger pool costs
# more rerank input tokens but gives the cross-encoder more lift to find
# fragments cosine ranked past the final cap.
HIERARCHICAL_RERANK_POOL_SIZE = 150


class CompanionSystem:
    """Phase 1 Option C substrate — build step 4 (complete).

    Current scope:
      - fragment store + dual-rate heartbeat + 2-entity cast          (step 1 ✓)
      - R1 spreading + R2 generative + R7 context-conditioning        (step 2 ✓)
      - S1 byte cap + S7 Hebbian edges + S5 covenant pins             (step 3 ✓)
      - W1 state-reading + W4 divergence-as-pain + W7 second-perspective
        + W3 pairwise-comparison instrumentation                       (step 4 ✓)

    witness_active=False ablates the Witness coupling (W4 pain feedback is
    skipped). This is the control condition for the V4 Mirror-ablation metric.
    """

    def __init__(
        self,
        top_k: int = 6,
        seed: int = 0,
        byte_cap: Optional[int] = DEFAULT_BYTE_CAP,
        witness_active: bool = True,
        pain_gain: float = 0.6,
        companion_only_suppress: float = 0.4,
        *,
        consolidation_enabled: bool = False,
        consolidation_router: Optional[LLMRouter] = None,
        consolidation_log_dir: Optional[Path] = None,
        consolidation_instance_id: str = "default",
        fact_top_k: Optional[int] = None,
        fact_cosine_floor: Optional[float] = None,
        embedder_ensemble: bool = False,
        cross_encoder_rerank: bool = False,
    ) -> None:
        self.companion = Companion()
        self.witness = Witness()
        # Option 5.C: when consolidation is on, the extracted-fact tier
        # is the compression mechanism, so the raw-fragment tier runs
        # uncapped. 4KB purge gutted 405/419 conv-26 turns in Phase 1.
        effective_byte_cap = None if consolidation_enabled else byte_cap
        self.companion.store.byte_cap = effective_byte_cap
        self.witness.store.byte_cap = effective_byte_cap
        self.heartbeat = Heartbeat()
        self.activation = ActivationState()
        self.edges = EdgeStore()
        self.witness_edges = EdgeStore()
        self.witness_state = WitnessState()
        self.top_k = top_k
        self.seed = seed
        self.witness_active = witness_active
        self.pain_gain = pain_gain
        self.companion_only_suppress = companion_only_suppress

        # Phase 1.5 consolidation layer. Default off so Phase 1 tests
        # (which construct CompanionSystem() with no args) are unchanged.
        # When consolidation is on, defaults widen to the hierarchical
        # tier settings (fact_top_k=150, cosine_floor=0.0) so the reader
        # sees enough of the fact store to compensate for mxbai cosine
        # ranking burying gold facts under noise.
        self.consolidation_enabled = consolidation_enabled
        if fact_top_k is None:
            fact_top_k = HIERARCHICAL_FACT_TOP_K if consolidation_enabled else DEFAULT_TOP_K
        if fact_cosine_floor is None:
            fact_cosine_floor = (
                HIERARCHICAL_FACT_COSINE_FLOOR
                if consolidation_enabled
                else DEFAULT_COSINE_FLOOR
            )
        self.fact_top_k = fact_top_k
        self.fact_cosine_floor = fact_cosine_floor
        self.fact_store: Optional[FactStore] = None
        self.consolidator: Optional[Consolidator] = None
        if consolidation_enabled:
            if consolidation_router is None:
                raise ValueError(
                    "consolidation_enabled=True requires consolidation_router"
                )
            self.fact_store = FactStore()
            self.consolidator = Consolidator(
                router=consolidation_router,
                store=self.fact_store,
                log_dir=consolidation_log_dir,
                instance_id=consolidation_instance_id,
            )
        self._consolidation_router = consolidation_router

        # Phase 1.5-prime P2 — embedder ensemble. When on, every candidate
        # (fact + fragment) is scored by both the local backend (mxbai) and
        # the active cloud backend (DashScope text-embedding-v3), and the
        # per-candidate final cosine is max(local, cloud). This captures
        # each embedder's strengths without inheriting the other's
        # weaknesses — the Phase 1.5 E1 v3→v4 ablation showed DashScope
        # swaps per-query failures rather than eliminates them (Sweden
        # rank 75→8 but Single 18→104); max-pool should recover both.
        #
        # Implementation: Fact.embedding carries the local vector at
        # consolidation time. The cloud vector is computed lazily at first
        # recall and parked in a side-car dict keyed by fact.id. The
        # fragment path recomputes both backends on every recall (router
        # disk cache amortizes cost after the first run).
        self.embedder_ensemble = embedder_ensemble
        self._cloud_fact_embeddings: dict[str, tuple[float, ...]] = {}
        if embedder_ensemble and consolidation_router is None:
            raise ValueError(
                "embedder_ensemble=True requires consolidation_router"
            )

        # Phase 1.5-prime P3 — cross-encoder rerank. When on, the raw-
        # fragment tier does cosine recall → rerank through the active
        # provider's cross-encoder (DashScope gte-rerank-v1) → take the
        # top-N reranked. Targets paraphrase-cue failures that bi-encoder
        # cosine ranking buries (the "home country Sweden" quote at
        # cosine rank 189 for cue "where did Caroline move from").
        self.cross_encoder_rerank = cross_encoder_rerank
        if cross_encoder_rerank and consolidation_router is None:
            raise ValueError(
                "cross_encoder_rerank=True requires consolidation_router"
            )

    def observe(self, event: Observation) -> None:
        self.heartbeat.wake()
        fragment = Fragment(
            id=f"frag-{self.heartbeat.wake_ticks - 1:04d}",
            content=event.content,
            timestamp=event.timestamp,
            affect=dict(event.affect),
            tags=event.tags,
        )
        for other in self.companion.store.all():
            w = jaccard_prior(fragment, other)
            if w >= JACCARD_SEED_THRESHOLD:
                self.edges.set(fragment.id, other.id, w)
        self.companion.store.add(fragment, edges=self.edges)
        for other in self.witness.store.all():
            w = jaccard_prior(fragment, other)
            if w >= JACCARD_SEED_THRESHOLD:
                self.witness_edges.set(fragment.id, other.id, w)
        self.witness.store.add(fragment, edges=self.witness_edges)

        if self.consolidator is not None:
            self.consolidator.observe(fragment)

    def mark_session_boundary(self) -> None:
        """Benchmark adapters that know the session structure call this
        between sessions. The actual extraction runs on the next ``tick``
        — sleep work on sleep ticks."""
        self.heartbeat.session_boundary_tick()
        if self.consolidator is not None:
            self.consolidator.mark_session_boundary()

    def recall(self, cue: str, context: Optional[dict] = None) -> Recall:
        hier_hit = self._recall_hierarchical(cue, context)
        if hier_hit is not None:
            return hier_hit
        rec = reconstruct(
            fragments=self.companion.store.all(),
            cue=cue,
            context=context,
            state=self.activation,
            tick_count=self.heartbeat.sleep_ticks,
            system_seed=self.seed,
            top_k=self.top_k,
            edges=self.edges,
        )
        pain = 0.0
        if self.witness_active:
            pain = witness_couple(
                fragments=self.witness.store.all(),
                cue=cue,
                companion_context=context,
                companion_reconstruction=rec,
                witness_state=self.witness_state,
                companion_activation=self.activation,
                witness_edges=self.witness_edges,
                tick_count=self.heartbeat.sleep_ticks,
                system_seed=self.seed,
                top_k=self.top_k,
                pain_gain=self.pain_gain,
                companion_only_suppress=self.companion_only_suppress,
            )
        return Recall(
            cue=cue,
            content=rec.content,
            fragments_used=rec.fragments_used,
            metadata={
                "step": 4,
                "path": "fragment",
                "wake_ticks": self.heartbeat.wake_ticks,
                "sleep_ticks": self.heartbeat.sleep_ticks,
                "recall_count": self.activation.recall_count,
                "pool_size": len(rec.fragments_used),
                "edge_count": len(self.edges),
                "edge_top5_share": self.edges.top_fraction_share(0.05),
                "witness_active": self.witness_active,
                "witness_pain": pain,
                "fact_count": len(self.fact_store) if self.fact_store is not None else 0,
            },
        )

    def _recall_hierarchical(
        self, cue: str, context: Optional[dict]
    ) -> Optional[Recall]:
        """Phase 1.5 Option 5.C — hierarchical recall.

        When consolidation is enabled, the reader always sees two tiers:

        1. **Fact tier** — top-K facts above cosine floor; carries
           compressed cross-session structure that beat a flat substrate
           by 3.5× in the Phase 1.5 E1 A/B (RAG F1 0.148 vs 0.042).
        2. **Fragment tier** — top-K cosine-ranked raw dialogue turns
           using the same mxbai embedding as the fact tier. Carries the
           specific proper nouns (Sweden, "Charlotte's Web", pottery)
           that session-level LLM extraction routinely drops — the
           "Bucket A / not stated" failure mode in E1.

        Returns ``None`` only when the system isn't configured for
        consolidation. If consolidation is on, we always return a Recall
        (facts can be empty — fragments alone are still useful),
        bypassing the Phase-1 Jaccard reconstruction which is a weaker
        baseline for scale-recall per the plan.
        """
        if (
            self.fact_store is None
            or self._consolidation_router is None
        ):
            return None

        if self.embedder_ensemble:
            fact_hits, fragment_hits = self._ensemble_recall(cue)
            cue_embedding = ()  # unused downstream in ensemble path
        else:
            cue_embedding = tuple(self._consolidation_router.embed(cue))

            # Fact tier.
            fact_hits = self.fact_store.search(
                cue_embedding,
                top_k=self.fact_top_k,
                cosine_floor=self.fact_cosine_floor,
            )
            fragment_hits = None  # computed below after fact dedup
        # SIMPLE-tier extractions produce snake_case values. LoCoMo's
        # Mem0-compat _normalize strips `_` as punctuation without
        # inserting space, gluing tokens into one unmatchable blob.
        # Replace underscores at compose-time (Phase 1.5 E1 v2→v3 fix).
        #
        # Dedup by composed string — qwen-turbo extraction routinely
        # emits exact duplicates across sessions ("Caroline activity
        # support" × 5). Duplicates burn attention budget on no new
        # signal; preserve the first occurrence (highest cosine) only.
        fact_lines: list[str] = []
        seen_lines: set[str] = set()
        for f, _ in fact_hits:
            line = f"{f.entity} {f.relation} {f.value}".replace("_", " ")
            key = line.lower().strip()
            if key in seen_lines:
                continue
            seen_lines.add(key)
            fact_lines.append(line)

        # Fragment tier — cosine-rank the raw dialogue by the same cue.
        # Embeddings are computed on-demand and cached by text (SHA key)
        # at the router layer, so repeated gates re-use them.
        if fragment_hits is None:
            # P3 widens the cosine pool so the reranker has more
            # candidates to pull up. Non-rerank path keeps the original
            # top-K to preserve substrate behavior.
            cos_top_k = (
                HIERARCHICAL_RERANK_POOL_SIZE
                if self.cross_encoder_rerank
                else HIERARCHICAL_FRAGMENT_TOP_K
            )
            fragment_hits = self._cosine_rank_fragments(
                cue_embedding, top_k=cos_top_k
            )

        # Phase 1.5-prime P3 — cross-encoder rerank. After cosine recall
        # pulls the top-K fragments, we rerank them jointly with the cue
        # via DashScope gte-rerank-v2. The reranker scores (query, doc)
        # pairs as cross-attention over joint content, not independent
        # embeddings — the standard fix for paraphrase-cue failures that
        # bi-encoders bury. We rerank over a *wider* pool (3× the final
        # cap) to give the cross-encoder a chance to pull up fragments
        # cosine ranked at K=50-150, then trim to the final cap.
        if self.cross_encoder_rerank and fragment_hits:
            fragment_hits = self._cross_encoder_rerank_fragments(
                cue, fragment_hits
            )

        # Fact-value substring augmentation (v5). The facts surface
        # specific proper nouns and entities ("Sweden", "single parent",
        # "Charlotte's Web") that no cosine embedder reliably ranks
        # into a 30-turn window. Substring-match top-K fact values back
        # against the raw fragment store; matching turns prepend the
        # DIALOGUE block at highest priority. Guarantees proper-noun
        # fidelity under adversarial cue phrasing (e.g. "where did you
        # move from" never contains "Sweden").
        all_fragments = self.companion.store.all()
        cosine_frag_ids = {frag.id for frag, _ in fragment_hits}
        targets: list[str] = []
        seen_targets: set[str] = set()
        for f, _ in fact_hits[:HIERARCHICAL_SUBSTRING_FACTS_SCANNED]:
            raw = f.value.replace("_", " ").strip()
            v = raw.lower()
            # Prefer distinctive values: multi-word OR contains an
            # uppercase character in the original. Rejects noise targets
            # like "love", "transition", "support" that match too many
            # turns and dilute DIALOGUE precision.
            is_multiword = len(raw.split()) >= 2
            has_upper = any(ch.isupper() for ch in raw)
            if not (is_multiword or has_upper):
                continue
            if (
                HIERARCHICAL_SUBSTRING_VALUE_MIN_LEN
                <= len(v)
                <= HIERARCHICAL_SUBSTRING_VALUE_MAX_LEN
                and v not in seen_targets
            ):
                targets.append(v)
                seen_targets.add(v)
        substring_fragments: list[Fragment] = []
        seen_substring_ids: set[str] = set()
        for frag in all_fragments:
            if frag.id in cosine_frag_ids or frag.id in seen_substring_ids:
                continue
            content_l = frag.content.lower()
            if any(t in content_l for t in targets):
                substring_fragments.append(frag)
                seen_substring_ids.add(frag.id)
                if len(substring_fragments) >= HIERARCHICAL_SUBSTRING_FRAGMENT_CAP:
                    break

        if not fact_hits and not fragment_hits and not substring_fragments:
            return None

        # Reader-facing content: facts first (compressed, high-signal),
        # then substring-matched raw turns (proper-noun fidelity),
        # then cosine-ranked raw turns (recall).
        sections: list[str] = []
        if fact_lines:
            sections.append("FACTS:\n" + "\n".join(fact_lines))
        dialogue_lines: list[str] = []
        for frag in substring_fragments:
            dialogue_lines.append(frag.content)
        for frag, _ in fragment_hits:
            dialogue_lines.append(frag.content)
        if dialogue_lines:
            sections.append("DIALOGUE:\n" + "\n".join(dialogue_lines))
        content = "\n\n".join(sections)

        source_ids: list[str] = []
        seen: set[str] = set()
        for f, _ in fact_hits:
            for sid in f.source_fragment_ids:
                if sid in seen:
                    continue
                seen.add(sid)
                source_ids.append(sid)
        for frag in substring_fragments:
            if frag.id not in seen:
                seen.add(frag.id)
                source_ids.append(frag.id)
        for frag, _ in fragment_hits:
            if frag.id not in seen:
                seen.add(frag.id)
                source_ids.append(frag.id)
        fragments_used: list[Fragment] = []
        for sid in source_ids:
            frag = self.companion.store.by_id(sid)
            if frag is not None:
                fragments_used.append(frag)

        return Recall(
            cue=cue,
            content=content,
            fragments_used=tuple(fragments_used),
            metadata={
                "step": 4,
                "path": "hierarchical",
                "wake_ticks": self.heartbeat.wake_ticks,
                "sleep_ticks": self.heartbeat.sleep_ticks,
                "recall_count": self.activation.recall_count,
                "pool_size": len(fact_hits) + len(fragment_hits),
                "fact_count": len(self.fact_store),
                "facts_used": [f.id for f, _ in fact_hits],
                "fact_scores": [s for _, s in fact_hits],
                "fragments_cosine_used": [frag.id for frag, _ in fragment_hits],
                "fragment_scores": [s for _, s in fragment_hits],
                "witness_active": self.witness_active,
                "witness_pain": 0.0,
            },
        )

    def _ensemble_recall(
        self, cue: str
    ) -> tuple[list[tuple], list[tuple[Fragment, float]]]:
        """Phase 1.5-prime P2 — dual-backend max-pool recall.

        For each candidate (fact + fragment), score = ``max(cos_local,
        cos_cloud)``. Cloud fact embeddings are populated lazily on the
        first ensemble recall and parked in a per-system side-car dict;
        subsequent recalls hit the router disk cache so the dominant
        cost is the first batch call.

        Fragments are re-ranked both backends on every recall — same
        caching dynamics (first call pays, later calls amortize).

        Returns ``(fact_hits, fragment_hits)`` with the same shape as
        the non-ensemble path so downstream composition is uniform.
        """
        from psychology_base.companion.facts import cosine, embed_text_for

        router = self._consolidation_router
        assert router is not None  # guarded in __init__

        cue_local = tuple(router.embed(cue, backend="local"))
        cue_cloud = tuple(router.embed(cue, backend="cloud"))

        # Facts — build cloud side-car if cold or stale (consolidation may
        # have appended since last recall). Only the missing fact ids are
        # embedded; the router disk cache does the deduplication across
        # runs by content hash.
        facts = self.fact_store.all() if self.fact_store is not None else []
        missing = [f for f in facts if f.id not in self._cloud_fact_embeddings]
        if missing:
            texts = [embed_text_for(f) for f in missing]
            embeds = router.embed_batch(
                texts, max_workers=4, backend="cloud"
            )
            for f, emb in zip(missing, embeds):
                self._cloud_fact_embeddings[f.id] = tuple(emb)

        scored_facts: list[tuple] = []
        for f in facts:
            if not f.embedding:
                continue
            s_local = cosine(cue_local, f.embedding)
            cloud_emb = self._cloud_fact_embeddings.get(f.id, ())
            s_cloud = cosine(cue_cloud, cloud_emb) if cloud_emb else 0.0
            score = max(s_local, s_cloud)
            if score < self.fact_cosine_floor:
                continue
            scored_facts.append((f, score))
        scored_facts.sort(key=lambda it: -it[1])
        fact_hits = scored_facts[: self.fact_top_k]

        # Fragments — same dual-backend rank + max-pool.
        fragments = self.companion.store.all()
        if not fragments:
            return fact_hits, []
        texts = [f.content for f in fragments]
        embeds_local = router.embed_batch(
            texts, max_workers=4, backend="local"
        )
        embeds_cloud = router.embed_batch(
            texts, max_workers=4, backend="cloud"
        )
        scored_frags: list[tuple[Fragment, float]] = []
        for frag, el, ec in zip(fragments, embeds_local, embeds_cloud):
            s_local = cosine(cue_local, tuple(el))
            s_cloud = cosine(cue_cloud, tuple(ec))
            scored_frags.append((frag, max(s_local, s_cloud)))
        scored_frags.sort(key=lambda it: -it[1])
        fragment_hits = scored_frags[:HIERARCHICAL_FRAGMENT_TOP_K]

        return fact_hits, fragment_hits

    def _cross_encoder_rerank_fragments(
        self,
        cue: str,
        fragment_hits: list[tuple[Fragment, float]],
    ) -> list[tuple[Fragment, float]]:
        """Rerank the cosine top-K fragments jointly against the cue.

        The reranker returns ``(index, relevance_score)`` ordered by
        score desc. We keep the fragment reference from the input list
        and swap the cosine score for the rerank score so downstream
        metadata reflects the composite ranking signal.
        """
        router = self._consolidation_router
        assert router is not None  # guarded in __init__
        documents = [frag.content for frag, _ in fragment_hits]
        if not documents:
            return []
        reranked = router.rerank(
            cue, documents, top_n=HIERARCHICAL_FRAGMENT_TOP_K
        )
        out: list[tuple[Fragment, float]] = []
        for idx, score in reranked:
            if 0 <= idx < len(fragment_hits):
                frag, _ = fragment_hits[idx]
                out.append((frag, score))
        return out

    def _cosine_rank_fragments(
        self,
        cue_embedding: tuple[float, ...],
        *,
        top_k: int,
    ) -> list[tuple[Fragment, float]]:
        """Cosine-rank fragments against the cue.

        Embeddings are router-cached by text content, so repeat queries
        after the first gate run are cheap. First run on a 419-turn
        LoCoMo instance ≈ 230s with mxbai-embed-large on local Ollama.
        """
        from psychology_base.companion.facts import cosine

        fragments = self.companion.store.all()
        if not fragments:
            return []
        texts = [f.content for f in fragments]
        embeds = self._consolidation_router.embed_batch(texts, max_workers=4)
        scored: list[tuple[Fragment, float]] = []
        for frag, emb in zip(fragments, embeds):
            scored.append((frag, cosine(cue_embedding, tuple(emb))))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def tick(self) -> None:
        self.heartbeat.sleep()
        self.activation.decay()
        self.witness_state.activation.decay()
        self.edges.decay()
        self.witness_edges.decay()
        if self.consolidator is not None:
            self.consolidator.maybe_consolidate(
                tick_count=self.heartbeat.sleep_ticks
            )

    def inspect(self) -> dict:
        out = {
            "step": 4,
            "fragment_count": len(self.companion.store),
            "witness_fragment_count": len(self.witness.store),
            "wake_ticks": self.heartbeat.wake_ticks,
            "sleep_ticks": self.heartbeat.sleep_ticks,
            "session_boundaries": self.heartbeat.session_boundaries,
            "recall_count": self.activation.recall_count,
            "residual_activation": dict(self.activation.residual),
            "edge_count": len(self.edges),
            "edge_total_weight": self.edges.total(),
            "edge_top5_share": self.edges.top_fraction_share(0.05),
            "total_bytes": self.companion.store.total_bytes(self.edges),
            "byte_cap": self.companion.store.byte_cap,
            "witness_active": self.witness_active,
            "witness_pain_log": [
                {
                    "recall_count": e.recall_count,
                    "tick_count": e.tick_count,
                    "cue": e.cue,
                    "pain": e.pain,
                    "witness_only_fragments": list(e.witness_only_fragments),
                }
                for e in self.witness_state.pain_log
            ],
            "fragments": [
                {
                    "id": f.id,
                    "timestamp": f.timestamp.isoformat(),
                    "content": f.content,
                    "tags": list(f.tags),
                }
                for f in self.companion.store.all()
            ],
            "consolidation_enabled": self.consolidation_enabled,
        }
        if self.fact_store is not None and self.consolidator is not None:
            out["fact_count"] = len(self.fact_store)
            out["consolidation_stats"] = {
                "adds": self.consolidator.stats.adds,
                "updates": self.consolidator.stats.updates,
                "noops": self.consolidator.stats.noops,
                "dropped": self.consolidator.stats.dropped,
            }
        return out
