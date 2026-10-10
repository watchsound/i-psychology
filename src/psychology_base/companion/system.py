from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from psychology_base.companion.aging import AgingPolicy
from psychology_base.companion.consolidation import Consolidator
from psychology_base.companion.copresence import (
    CopresenceScope,
    filter_fragments_by_copresence,
)
from psychology_base.companion.perspective.routing import (
    PerspectiveSpec,
    ScopeSpec,
)
from psychology_base.companion.edges import EdgeStore
from psychology_base.companion.entities import Companion, Witness
from psychology_base.companion.fragment import is_covenant
from psychology_base.companion.entity_graph import (
    EntityGraph,
    build_entity_graph,
    score_fragments as entity_graph_score_fragments,
)
from psychology_base.companion.facts import (
    DEFAULT_COSINE_FLOOR,
    DEFAULT_TOP_K,
    FactStore,
)
from psychology_base.companion.heartbeat import Heartbeat
from psychology_base.companion.affect_resonance import cue_affect_from_context
from psychology_base.companion.reconstruction import (
    ActivationState,
    jaccard_prior,
    reconstruct,
    spread_activation,
)
from psychology_base.companion.witness import WitnessState, witness_couple
from psychology_base.eval.adapters import (
    Fragment,
    Observation,
    Recall,
    make_fragment,
)
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


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

# Phase 2 R1 — upper bound on the rerank-all pool. DashScope
# ``gte-rerank-v2`` caps per-call document count at 500. If the fragment
# store exceeds this cap, we fall back to cosine prefilter at this value
# instead of rejecting the call. conv-26 has 419 fragments so this does
# not bind on the current benchmark; it keeps the rerank-all path safe
# on larger instances.
HIERARCHICAL_RERANK_ALL_MAX_POOL = 500

# Phase 2 R2 — BM25 + semantic hybrid recall. Reciprocal Rank Fusion
# constant. Standard literature value is 60; smaller k emphasizes top
# ranks more sharply, larger k flattens the tail. The RRF formula is
# ``score(frag) = sum_channel 1/(k + rank_channel)`` with rank starting
# at 1. A fragment appearing in only one channel gets a single term;
# fragments in both channels sum.
HIERARCHICAL_BM25_RRF_K = 60

# Phase 2 R8 — within-session fragment ranking. Default top-M sessions
# to pull all turns from into the within-session cross-encoder pool.
# R6 diagnostic showed 5 of 7 persistent single-hop broken rows have
# their evidence session reach cosine top-50 but the specific gold turn
# ranks past top-150 within its session. R8-c addresses this by: group
# cosine top-150 by session_id → pick top-M sessions by max-cosine →
# pull ALL turns from those M sessions out of the fragment store →
# cross-encoder rerank over that scope → trim to top-50. M=3 covers
# typical single-hop evidence (single session) with headroom; M=5
# trades more distractor budget for multi-hop. Pilot M=3.
HIERARCHICAL_WITHIN_SESSION_TOP_M = 3

# Phase 2b Step A — fact-to-turn lift. When on, the top-N facts in the
# fact tier contribute their ``source_fragment_ids`` back-pointers as
# directly-lifted dialogue fragments, prepended to the DIALOGUE block
# ahead of cosine-ranked and substring-matched turns. The back-pointer
# is the strongest available evidence that a fragment matters to a
# cue-matching fact: the extractor literally produced the fact FROM
# that turn. Default off until the 3-instance gate measures lift.
HIERARCHICAL_FACT_SOURCE_LIFT_TOP_FACTS = 15
HIERARCHICAL_FACT_SOURCE_LIFT_FRAGMENT_CAP = 30

# Phase 2b Step F — full-session reader context. When on, after
# cross-encoder rerank identifies the top-1 reranked fragment, the
# fragment tier is REPLACED with every turn of that fragment's
# session, ordered chronologically. The reader is thus guaranteed to
# see the specific gold turn IF the right session is picked — the
# direct attack on `session_present_gold_absent` (12/17 of 3-instance
# R5 incorrect) which is a within-session ranking failure, not a
# session-selection failure. Trade-off: context length grows from
# ~50 turns (reranked top-K) to ~session-size (typically 20-100
# turns on LoCoMo). qwen-max 32K context handles this easily.
HIERARCHICAL_FULL_SESSION_READER_CAP_TURNS = 300

# Phase 2b Step E — localize-filter LLM stage. A SIMPLE-tier LLM pass
# inserted between cross-encoder rerank and the reader with a recall-
# biased prompt: "which fragments COULD plausibly contribute to
# answering Q — err on inclusion." EMem ablation reports ~4.7 LoCoMo
# points for this stage alone. Direct attack on the `gold_in_pool`
# bucket (reader-precision failures on fragments the rerank already
# surfaced): the recall-biased filter re-sorts the pool to make
# plausible gold-adjacent turns visible to the reader even when the
# cross-encoder's precision-biased score pushed them past the final
# cap.
#
# Design:
# - Preserves rerank score ordering for kept rows; drops nothing more
#   than the LLM explicitly excludes. Never drops below MIN_KEEP as
#   a safety floor against degenerate LLM output.
# - Numbered-index response schema (compact, parse-stable). Content
#   truncated per fragment so prompt fits comfortably in qwen-turbo's
#   32K context even with the ADDITIVE Step F pool.
HIERARCHICAL_LOCALIZE_FILTER_MIN_KEEP = 5
HIERARCHICAL_LOCALIZE_FILTER_MAX_INPUT = 80
HIERARCHICAL_LOCALIZE_FILTER_CONTENT_CHARS = 400
HIERARCHICAL_LOCALIZE_FILTER_MAX_TOKENS = 512

# Phase 2b Step H — reconstructive assembly. When on, the existing
# R1 spreading-activation over the Jaccard-seeded ``EdgeStore`` is
# used to score fragments; the top-N by activation become a
# "reconstructive_boost" tier appended to the fragment pool, dedup'd
# against cosine/substring/source_lift/entity_boost. This wires the
# reconstructive-memory primitives (``companion/reconstruction.py``)
# into the LoCoMo eval path per the thesis directive — we re-measure
# F1 using a structural recall channel instead of pure retrieve-then-
# compose. R2 coalescence is NOT applied to the reader output (that
# would collapse the answer to a sparse token-bag unsuitable for
# F1); R1 provides the ranking signal only. Compatible with every
# existing tier flag.
HIERARCHICAL_RECONSTRUCTIVE_CAP = 20
HIERARCHICAL_RECONSTRUCTIVE_HOPS = 2
HIERARCHICAL_RECONSTRUCTIVE_HOP_DECAY = 0.5

# Phase 2b Step C — entity-graph fragment lift. When on, after
# cross-encoder rerank (and Steps F / E if active), a fact-derived
# entity graph is probed with the cue. Cue entities → 1-hop graph
# neighbors → fragments mentioning any expanded entity get appended
# to the fragment pool as a bounded top-N "entity_boost" tier. The
# tier is DEDUPED against cosine_frag_ids, substring_fragments, and
# source_lift_fragments so no double emission. Attacks
# ``session_present_gold_absent`` without requiring per-fact
# provenance (independent of Step A/A').
HIERARCHICAL_ENTITY_GRAPH_CAP = 20
HIERARCHICAL_ENTITY_GRAPH_SEED_CAP = 10
HIERARCHICAL_ENTITY_GRAPH_MIN_HITS = 1

# Phase 2b Step B — hybrid fact retrieval. Fuses three channels over
# the fact store via Reciprocal Rank Fusion with k=60:
#   1. Dense cosine over `embed_text_for(fact)` (existing signal).
#   2. BM25 over the same canonical fact text (literal-token recall for
#      proper-noun / date / named-item matches cosine buries).
#   3. Entity-overlap: cue-token overlap with `fact.entity` tokens;
#      specifically targets "named-person / named-concept" cues
#      ("tell me about Caroline") where the entity *is* the signal.
# RRF is rank-based so no channel-weight calibration is needed; a fact
# strong on any one channel surfaces. Floor applied after RRF against
# the *dense* score (not RRF) to preserve the semantics of
# `fact_cosine_floor` as a quality gate rather than a rank gate.
HIERARCHICAL_FACT_HYBRID_RRF_K = 60
# Entity-overlap tokenizer is identical to BM25 tokenizer; keeps
# "Caroline" ↔ "caroline" match deterministic. Entities that fall
# below a minimum token length are dropped from the overlap channel
# because single-char entities ("I") blow up noise.
HIERARCHICAL_FACT_ENTITY_TOKEN_MIN_LEN = 2

# Phase 2 R8-b — fact-density tie-breaker. Applied AFTER cross-encoder
# rerank (either regular or within-session) as a conservative re-sort.
# Each fragment gets a density score = count of (proper-noun tokens +
# 4-digit years + ISO-ish dates + bare integers). The post-rerank
# ordering only swaps adjacent pairs whose rerank-score |Δ| ≤ threshold
# AND whose density |Δ| ≥ 2. Threshold is small (0.03) so we never
# override the cross-encoder's confident calls — only break near-ties
# in favor of the fact-dense fragment, which is what the R6 ceiling
# rows (q71 Becoming Nicole, q75 "3", q76 "19 October 2023") need.
HIERARCHICAL_FACT_DENSITY_SCORE_DELTA = 0.03
HIERARCHICAL_FACT_DENSITY_MIN_DIFF = 2
# A proper-noun candidate token is a word-shape with at least one
# interior lowercase letter (so "Sweden" matches but "I" / "LOL" don't).
# "4-digit year" matches 1900–2099 literal years in the LoCoMo era.
# ISO-ish date: YYYY-MM-DD or "DD Month YYYY" style.
# Bare integer: standalone 1–3 digit integer not inside a larger token.
_FACT_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Za-z]+)*\b")
_FACT_SENTENCE_START_RE = re.compile(r"(?:^|[.!?]\s+|\n\s*)")
# Common English words that get titlecased only because they start a
# sentence — pronouns, determiners, high-frequency verbs, question
# words, prepositions, conjunctions, weekdays/months-as-month-names
# are deliberately excluded (months are legitimate signal when they
# appear mid-sentence or inside a DMY date). Sentence-initial matches
# whose lowercased form is in this set are dropped as proper-noun
# candidates; sentence-initial matches NOT in this set (e.g. "Sweden
# is nice.") remain counted.
_SENTENCE_START_COMMON_WORDS = frozenset(
    {
        "a", "an", "the",
        "he", "she", "it", "we", "you", "they",
        "his", "her", "its", "my", "your", "our", "their",
        "this", "that", "these", "those",
        "and", "but", "or", "so", "yet", "nor",
        "because", "when", "while", "if", "then", "though", "although",
        "in", "on", "at", "to", "from", "with", "by", "for", "of", "about",
        "after", "before", "during", "since", "until", "over", "under",
        "what", "where", "when", "why", "how", "who", "which", "whose",
        "moved", "visited", "went", "came", "walked", "ran", "drove",
        "said", "told", "asked", "answered", "replied",
        "thought", "felt", "knew", "wondered", "wanted", "needed",
        "loved", "liked", "hated", "missed", "remembered", "forgot",
        "found", "lost", "got", "had", "took", "gave", "made", "did",
        "was", "were", "is", "am", "are", "been", "being",
        "will", "would", "should", "could", "can", "may", "might", "must",
        "yesterday", "today", "tomorrow", "later", "soon", "finally",
        "something", "everything", "nothing", "someone", "everyone",
        "last", "next", "every", "each", "many", "most", "some", "any",
        "hello", "hi", "hey", "oh", "ah", "well", "yes", "no",
        "okay", "thanks", "please", "sorry", "right", "sure", "maybe",
        "actually", "honestly", "really", "just", "only", "always",
        "never", "sometimes", "often", "once", "twice",
    }
)
_FACT_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_FACT_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_FACT_DMY_RE = re.compile(
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
_FACT_BARE_INT_RE = re.compile(r"(?<![\w.])\d{1,3}(?![\w.])")


def _collect_sentence_start_offsets(text: str) -> set[int]:
    """Byte offsets in ``text`` that begin a new sentence.

    Start of string, start of newline-separated lines, and positions
    following sentence-terminating punctuation (. ! ?) plus whitespace
    all count. Used to decide whether a titlecased match is likely
    only capitalized due to position rather than being a proper noun.
    """
    offsets = {0}
    for m in _FACT_SENTENCE_START_RE.finditer(text):
        offsets.add(m.end())
    return offsets


def _count_proper_nouns(text: str) -> int:
    """Count proper-noun candidates with sentence-start filtering.

    A titlecased token at a sentence-start position is dropped if its
    lowercased form is in ``_SENTENCE_START_COMMON_WORDS`` — this
    catches "Moved", "Her", "The", etc. without penalizing legitimate
    sentence-initial proper nouns like "Sweden is where I lived".
    """
    starts = _collect_sentence_start_offsets(text)
    count = 0
    for m in _FACT_PROPER_NOUN_RE.finditer(text):
        if (
            m.start() in starts
            and m.group(0).lower() in _SENTENCE_START_COMMON_WORDS
        ):
            continue
        count += 1
    return count

# BM25 tokenization — lowercase, split on non-word chars. Deliberately
# no stopword removal or stemming: dialogue stopwords are rare, and the
# gold phrases we care about (proper nouns, dates, specific objects)
# survive any sane filter. Keeping tokenization simple also keeps the
# cue tokenization identical to the fragment tokenization — the critical
# property for BM25 to find proper-noun matches.
_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase + split on non-word chars. Preserves digits (for dates
    like ``2023`` and quantifiers like ``2``)."""
    return [m.group(0).lower() for m in _BM25_TOKEN_RE.finditer(text)]


def _fact_density_score(text: str) -> int:
    """Phase 2 R8-b — fragment-side fact-density signal.

    Count proper-noun tokens, 4-digit years, ISO/DMY dates, and bare
    integers. The score is deliberately simple (unweighted sum) because
    R8-b uses it as a tie-breaker, not a primary ranking signal — the
    cross-encoder owns primary ranking. Higher = more "fact-dense".
    """
    proper_nouns = _count_proper_nouns(text)
    years = len(_FACT_YEAR_RE.findall(text))
    iso_dates = len(_FACT_ISO_DATE_RE.findall(text))
    dmy_dates = len(_FACT_DMY_RE.findall(text))
    bare_ints = len(_FACT_BARE_INT_RE.findall(text))
    return proper_nouns + years + iso_dates + dmy_dates + bare_ints


_LOCALIZE_INDEX_RE = re.compile(r"\d+")


def _parse_localize_filter_response(
    response: str, *, max_index: int
) -> list[int]:
    """Parse the comma-separated integer-index list produced by Step E.

    Tolerant of common LLM deviations: brackets, markdown bullets,
    multi-line output, trailing prose. Any integer in [0, max_index] is
    kept; dupes are deduplicated preserving first-seen order.
    """
    if not response or max_index < 0:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for match in _LOCALIZE_INDEX_RE.findall(response):
        try:
            idx = int(match)
        except ValueError:
            continue
        if 0 <= idx <= max_index and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _apply_fact_density_tie_breaker(
    reranked: list[tuple[Fragment, float]],
    *,
    score_delta: float = HIERARCHICAL_FACT_DENSITY_SCORE_DELTA,
    min_density_diff: int = HIERARCHICAL_FACT_DENSITY_MIN_DIFF,
) -> list[tuple[Fragment, float]]:
    """Conservative post-rerank re-sort by fact-density on near-ties.

    Walk the reranked list in order. If adjacent pair ``(a, b)`` satisfies
    ``|a.score - b.score| <= score_delta`` AND
    ``density(b) - density(a) >= min_density_diff``, swap them. Only
    adjacent swaps — this is intentional: the cross-encoder's confident
    ordering (gaps > score_delta) is preserved; only near-ties where
    the lower-ranked fragment is substantially more fact-dense get
    promoted. One pass; not a full bubble sort.
    """
    if len(reranked) <= 1:
        return reranked
    out = list(reranked)
    i = 0
    while i < len(out) - 1:
        a_frag, a_score = out[i]
        b_frag, b_score = out[i + 1]
        if abs(a_score - b_score) <= score_delta:
            a_density = _fact_density_score(a_frag.content)
            b_density = _fact_density_score(b_frag.content)
            if b_density - a_density >= min_density_diff:
                out[i], out[i + 1] = out[i + 1], out[i]
        i += 1
    return out


def _filter_fragments_by_scope(
    fragments,
    scope: tuple[str, str],
):
    """M2 scope filter — pre-pool fragments by perspective tags.

    Thin re-export shim: the primitive now lives in
    :mod:`psychology_base.companion.perspective.routing` so the whole
    mentalistic-routing surface sits in one package. Byte-identical
    behavior is preserved — the FANToM Track 1 +0.791 scope-asymmetric
    mechanism depends on this.
    """
    from psychology_base.companion.perspective.routing import (
        filter_fragments_by_scope,
    )

    return filter_fragments_by_scope(fragments, scope)


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
        cross_encoder_rerank_all: bool = False,
        bm25_hybrid: bool = False,
        within_session_rerank: bool = False,
        within_session_top_m_sessions: int = HIERARCHICAL_WITHIN_SESSION_TOP_M,
        fact_density_tie_breaker: bool = False,
        fact_source_turn_lift: bool = False,
        fact_source_lift_top_facts: int = HIERARCHICAL_FACT_SOURCE_LIFT_TOP_FACTS,
        fact_source_lift_fragment_cap: int = HIERARCHICAL_FACT_SOURCE_LIFT_FRAGMENT_CAP,
        fact_hybrid_retrieval: bool = False,
        full_session_reader: bool = False,
        full_session_reader_cap_turns: int = HIERARCHICAL_FULL_SESSION_READER_CAP_TURNS,
        localize_filter: bool = False,
        localize_filter_min_keep: int = HIERARCHICAL_LOCALIZE_FILTER_MIN_KEEP,
        localize_filter_max_input: int = HIERARCHICAL_LOCALIZE_FILTER_MAX_INPUT,
        reader_evidence_rerank_first: bool = False,
        entity_graph_retrieval: bool = False,
        entity_graph_cap: int = HIERARCHICAL_ENTITY_GRAPH_CAP,
        entity_graph_seed_cap: int = HIERARCHICAL_ENTITY_GRAPH_SEED_CAP,
        entity_graph_min_hits: int = HIERARCHICAL_ENTITY_GRAPH_MIN_HITS,
        reconstructive_retrieval: bool = False,
        reconstructive_cap: int = HIERARCHICAL_RECONSTRUCTIVE_CAP,
        reconstructive_hops: int = HIERARCHICAL_RECONSTRUCTIVE_HOPS,
        reconstructive_hop_decay: float = HIERARCHICAL_RECONSTRUCTIVE_HOP_DECAY,
        affect_weight: float = 0.0,
        aging_policy: Optional[AgingPolicy] = None,
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

        # Phase 2 R1 — rerank over all fragments. Phase 1.5-prime P3
        # pre-trimmed to a top-150 cosine pool to cap rerank-input cost;
        # per-row diagnosis showed 5 Bucket A failures where gold fragment
        # was cosine-ranked past 150 so rerank never saw it. Setting this
        # flag passes ALL fragments to the reranker (bounded by provider
        # per-call document cap — DashScope gte-rerank-v2 = 500).
        self.cross_encoder_rerank_all = cross_encoder_rerank_all
        if cross_encoder_rerank_all and not cross_encoder_rerank:
            raise ValueError(
                "cross_encoder_rerank_all=True requires cross_encoder_rerank=True"
            )

        # Phase 2 R2 — BM25 + semantic hybrid recall. When on, the
        # fragment-pool fill switches from pure cosine to RRF-fused
        # BM25 + cosine. BM25 is literal-token recall; cosine is
        # paraphrase-tolerant. The two channels catch complementary
        # failures: proper-noun gold (Sweden, Adoption) that cosine
        # buries ranks at the top of BM25, and paraphrased gold that
        # BM25 misses ranks high on cosine. RRF with k=60 fuses both.
        # Lazy index — built on first recall when fragment store is
        # populated; invalidated whenever observe() mutates the store.
        self.bm25_hybrid = bm25_hybrid
        self._bm25_index = None
        self._bm25_fragments: list = []

        # Phase 2 R8-c — within-session rerank. Two-stage: cosine selects
        # top-M sessions by max-cosine; cross-encoder reranks over ALL
        # turns of those M sessions (not just the ones in cosine top-150)
        # to lift the specific gold turn out of within-session filler.
        # Requires cross_encoder_rerank (we reuse the DashScope reranker).
        self.within_session_rerank = within_session_rerank
        self.within_session_top_m_sessions = within_session_top_m_sessions
        if within_session_rerank and not cross_encoder_rerank:
            raise ValueError(
                "within_session_rerank=True requires cross_encoder_rerank=True"
            )
        if within_session_rerank and cross_encoder_rerank_all:
            raise ValueError(
                "within_session_rerank is mutually exclusive with "
                "cross_encoder_rerank_all (both widen the rerank pool "
                "via different strategies)"
            )

        # Phase 2 R8-b — fact-density tie-breaker. Applied after whichever
        # cross-encoder path ran (regular or within-session). Conservative:
        # only swaps adjacent near-ties where the lower-ranked fragment is
        # substantially more fact-dense. No-op without cross_encoder_rerank
        # — on the cosine-only path the primary signal is already shaped
        # by cosine, so a density boost would be re-ranking against the
        # wrong primary.
        self.fact_density_tie_breaker = fact_density_tie_breaker
        if fact_density_tie_breaker and not cross_encoder_rerank:
            raise ValueError(
                "fact_density_tie_breaker=True requires cross_encoder_rerank=True"
            )

        # Phase 2b Step A — fact-to-turn lift. Requires a fact store;
        # pulls source_fragment_ids off the top-N fact hits.
        self.fact_source_turn_lift = fact_source_turn_lift
        self.fact_source_lift_top_facts = fact_source_lift_top_facts
        self.fact_source_lift_fragment_cap = fact_source_lift_fragment_cap
        if fact_source_turn_lift and not consolidation_enabled:
            raise ValueError(
                "fact_source_turn_lift=True requires consolidation_enabled=True"
            )

        # Phase 2b Step B — hybrid retrieval over facts (BM25 + cosine +
        # entity-overlap, RRF-fused). Lazy BM25 index built on first
        # recall; rebuilt whenever the fact count changes (facts are
        # append-only in Phase 1.5 so length-change is sufficient).
        self.fact_hybrid_retrieval = fact_hybrid_retrieval
        self._fact_bm25_index = None
        self._fact_bm25_facts: list = []
        if fact_hybrid_retrieval and not consolidation_enabled:
            raise ValueError(
                "fact_hybrid_retrieval=True requires consolidation_enabled=True"
            )

        # Phase 2b Step F — full-session reader context. Replaces
        # ``fragment_hits`` with every turn of the winning session,
        # chronologically ordered. Needs the cross-encoder path to
        # pick the winning session; mutex with within-session rerank
        # (both widen the fragment pool via different strategies and
        # stacking produces no coherent semantics).
        self.full_session_reader = full_session_reader
        self.full_session_reader_cap_turns = full_session_reader_cap_turns
        if full_session_reader and not cross_encoder_rerank:
            raise ValueError(
                "full_session_reader=True requires cross_encoder_rerank=True"
            )
        if full_session_reader and within_session_rerank:
            raise ValueError(
                "full_session_reader is mutually exclusive with "
                "within_session_rerank (both widen the fragment pool via "
                "different strategies)"
            )

        # Phase 2b Step E — localize-filter LLM stage. Runs between
        # cross-encoder rerank and (optionally) full-session reader. The
        # consolidation router carries the SIMPLE tier; this flag has no
        # effect unless rerank already produced a non-empty pool to
        # filter. Requires consolidation_router so there's a SIMPLE-tier
        # endpoint to call.
        self.localize_filter = localize_filter
        self.localize_filter_min_keep = localize_filter_min_keep
        self.localize_filter_max_input = localize_filter_max_input
        if localize_filter and consolidation_router is None:
            raise ValueError(
                "localize_filter=True requires consolidation_router "
                "(the SIMPLE-tier endpoint used by the filter)"
            )
        if localize_filter and not cross_encoder_rerank:
            raise ValueError(
                "localize_filter=True requires cross_encoder_rerank=True "
                "(filter runs on the reranked pool)"
            )

        # Phase 2b reader-side ordering fix. The legacy ordering puts
        # facts' window-wide `source_fragment_ids` first in
        # `fragments_used` (which dominates `_format_fragments(limit=50)`
        # so the reader's formatted-list is noise) AND DIALOGUE puts the
        # cross-encoder-reranked top-K last (so the highest-precision
        # evidence lands in the attention-weakest region of the context).
        # When this flag is True, both `fragments_used` and DIALOGUE are
        # reordered to put reranked `fragment_hits` first, then substring
        # fragments, then source_lift, then facts' window-sources (dedup
        # preserved). Opt-in because R5 shipped behavior must not change
        # under a flag flip. Requires `cross_encoder_rerank=True` to be
        # meaningful (no reranked tier to promote otherwise).
        self.reader_evidence_rerank_first = reader_evidence_rerank_first

        # Phase 2b Step C — entity-graph fragment lift. Lazy graph built
        # on first recall from the fact store; rebuilt when fact count
        # changes (facts are append-only in Phase 1.5 so a length-change
        # check is sufficient — same pattern as Step B's fact BM25).
        # Requires consolidation (the fact store is the graph source).
        self.entity_graph_retrieval = entity_graph_retrieval
        self.entity_graph_cap = entity_graph_cap
        self.entity_graph_seed_cap = entity_graph_seed_cap
        self.entity_graph_min_hits = entity_graph_min_hits
        self._entity_graph: Optional[EntityGraph] = None
        self._entity_graph_fact_count: int = -1
        if entity_graph_retrieval and not consolidation_enabled:
            raise ValueError(
                "entity_graph_retrieval=True requires consolidation_enabled=True"
            )

        # Phase 2b Step H — reconstructive assembly. Uses R1 spreading
        # activation over the Jaccard-seeded EdgeStore as a structural
        # recall channel orthogonal to cosine/BM25/rerank. The
        # activation profile is computed per-cue (no cache); 2-hop
        # spread over the full fragment set is O(|F|^2 · hops) which
        # on LoCoMo (|F| ≈ 400-900) is ~500ms-1s per cue. Compatible
        # with any upstream retrieval tier.
        self.reconstructive_retrieval = reconstructive_retrieval
        self.reconstructive_cap = reconstructive_cap
        self.reconstructive_hops = reconstructive_hops
        self.reconstructive_hop_decay = reconstructive_hop_decay
        # M3 — affect-resonance retrieval weight. 0.0 keeps M1+M2
        # behaviour byte-identical; non-zero boosts fragments whose
        # stored affect matches the cue-side affect (read from
        # context["affect"]). See companion/affect_resonance.py.
        self.affect_weight = affect_weight

        # Phase 3 E1 — optional aging policy (S1 Decay / S2 Capacity /
        # S3 Metabolism / S4 Tiered). Default None keeps Phase 2 behaviour
        # byte-identical. When set, the FragmentStore's eviction strength
        # key is replaced by the policy's retention value (eviction
        # becomes importance × decay(age) under S1; covenant pins remain
        # protected through COVENANT_TAG regardless of policy).
        self.aging_policy = aging_policy
        if aging_policy is not None:
            self.companion.store.strength_fn = aging_policy.strength
            self.companion.store.evicted_callback = aging_policy.on_evict
            self.witness.store.strength_fn = aging_policy.strength
            self.witness.store.evicted_callback = aging_policy.on_evict

    def observe(self, event: Observation) -> None:
        self.heartbeat.wake()
        # M1 step 2 — content-addressed id via BLAKE2b hash of
        # (content, timestamp, affect, tags, parent_ids). Fresh
        # observations have empty parent_ids; re-renderings go through
        # ``rerender`` which chains back via parent_ids. The DAG shape
        # lives on fragments; retrieval remains reconstruction per
        # memory-design Principle B.
        #
        # M2 step 2 — perspective tags. subject_id / entity_id on the
        # observation become ``subject:<id>`` / ``entity:<id>`` tag
        # prefixes merged into fragment.tags. No schema change — the
        # content-hash naturally covers them (they live inside the
        # existing tags tuple), so provenance stays intact.
        tags = tuple(event.tags)
        if event.subject_id is not None:
            tags = tags + (f"subject:{event.subject_id}",)
        if event.entity_id is not None:
            tags = tags + (f"entity:{event.entity_id}",)
        fragment = make_fragment(
            content=event.content,
            timestamp=event.timestamp,
            affect=dict(event.affect),
            tags=tags,
            parent_ids=(),
        )
        self._ingest_fragment(fragment)

    def rerender(
        self,
        referent_id: str,
        new_content: str,
        *,
        timestamp,
        affect: Optional[dict] = None,
        tags: tuple[str, ...] = (),
    ) -> Fragment:
        """Append a new rendering of an existing referent.

        Creates a child fragment whose ``parent_ids`` chains back to
        ``referent_id``. Fragment is added to the companion + witness
        stores and the BM25 index is invalidated, mirroring ``observe``.
        No retrieval-time behaviour changes — the chain is write-side
        substrate (Principle B).
        """
        self.heartbeat.wake()
        fragment = make_fragment(
            content=new_content,
            timestamp=timestamp,
            affect=affect or {},
            tags=tags,
            parent_ids=(referent_id,),
        )
        self._ingest_fragment(fragment)
        return fragment

    def _ingest_fragment(self, fragment: Fragment) -> None:
        # Phase 3 E1 — register with aging policy BEFORE the store adds
        # it, so if this ingest triggers eviction the new fragment's
        # own strength is already known to the policy (otherwise the
        # policy's `strength` falls through to the 0.0 default and the
        # newly-arrived fragment becomes the eviction target).
        if self.aging_policy is not None:
            self.aging_policy.on_observe(
                fragment.id, is_covenant=is_covenant(fragment)
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

        # BM25 index is cue-independent but corpus-dependent; invalidate
        # on any store mutation so the next hybrid recall rebuilds.
        self._bm25_index = None

        if self.consolidator is not None:
            self.consolidator.observe(fragment)

    def mark_session_boundary(self) -> None:
        """Benchmark adapters that know the session structure call this
        between sessions. The actual extraction runs on the next ``tick``
        — sleep work on sleep ticks."""
        self.heartbeat.session_boundary_tick()
        if self.consolidator is not None:
            self.consolidator.mark_session_boundary()

    def recall(
        self,
        cue: str,
        context: Optional[dict] = None,
        *,
        perspective: Optional[PerspectiveSpec] = None,
        scope: Optional[tuple[str, str]] = None,
        copresence: Optional[CopresenceScope] = None,
        affect_weight: Optional[float] = None,
    ) -> Recall:
        # M2 step 3 — scope-filtered recall. When ``scope=(subject, entity)``
        # is set, pre-filter the fragment pool to those tagged
        # ``subject:<id>`` AND ``entity:<id>`` (matching M2 observe-time
        # tag merge). Scope-less fragments (no ``subject:``/``entity:``
        # prefix at all) pass through — back-compat for LoCoMo and any
        # pre-M2 scenario. Non-matching scoped fragments are excluded.
        #
        # Scoped recalls bypass the hierarchical path: the fact_store
        # carries no perspective tags, so there is no filter to apply
        # there in v1. Routing through the pure reconstruct() path is
        # the clean mentalistic-process behavior per Principle C.
        #
        # M3 — affect resonance. ``affect_weight`` (per-recall) overrides
        # ``self.affect_weight`` (system default) when provided. The
        # cue-side affect is read from ``context["affect"]``. Non-zero
        # affect weight also routes recalls through reconstruct() (the
        # hierarchical path's fact tier doesn't carry per-fragment PAD).
        # Phase 9+ perspective substrate consolidation: ``perspective=``
        # is the unified mentalistic recall-scope kwarg. Legacy ``scope=``
        # and ``copresence=`` are permanent back-compat shims that
        # construct an equivalent ``PerspectiveSpec`` internally. Mutex:
        # cannot combine ``perspective`` with any legacy kwarg (mirrors
        # the Pillar B / Pillar D responder mutex contracts).
        if perspective is not None and (
            scope is not None or copresence is not None
        ):
            raise ValueError(
                "recall() cannot mix 'perspective' with legacy 'scope'/"
                "'copresence' kwargs — pass one shape only"
            )
        if perspective is None and (
            scope is not None or copresence is not None
        ):
            perspective = PerspectiveSpec(
                scope=(
                    ScopeSpec(subject=scope[0], entity=scope[1])
                    if scope is not None
                    else None
                ),
                copresence=copresence,
            )
        perspective_active = (
            perspective is not None and not perspective.is_empty()
        )

        effective_affect_weight = (
            affect_weight if affect_weight is not None else self.affect_weight
        )
        cue_affect = cue_affect_from_context(context)
        affect_active = effective_affect_weight > 0.0 and cue_affect is not None
        # ``perspective_active`` suppresses the hierarchical path for the
        # same reason legacy scope/copresence did — the fact_store carries
        # no perspective tags, so a mentalistic query routed through facts
        # would surface the wrong referent class. ``affect_active`` is
        # orthogonal (reconstruction-time weighting), but still suppresses
        # the fact tier because the fact store has no per-fragment PAD.
        if not perspective_active and not affect_active:
            hier_hit = self._recall_hierarchical(cue, context)
            if hier_hit is not None:
                return hier_hit
            fragments = self.companion.store.all()
        elif perspective_active:
            fragments = perspective.apply(self.companion.store.all())
        else:
            fragments = self.companion.store.all()
        rec = reconstruct(
            fragments=fragments,
            cue=cue,
            context=context,
            state=self.activation,
            tick_count=self.heartbeat.sleep_ticks,
            system_seed=self.seed,
            top_k=self.top_k,
            edges=self.edges,
            cue_affect=cue_affect,
            affect_weight=effective_affect_weight,
        )
        # Phase 3 E1 — reconsolidate on recall. Every fragment that
        # entered the top-K pool gets its retention curve reset + boosted
        # so repeatedly-recalled fragments survive month-horizon decay.
        if self.aging_policy is not None:
            for f in rec.fragments_used:
                self.aging_policy.on_recall(f.id)
        pain = 0.0
        if self.witness_active:
            # Phase 9+ Step 2.1b — Witness-pool filter uses the same
            # perspective predicate as the main pool so pain-signals
            # and recall share one mentalistic scope.
            witness_pool = (
                perspective.apply(self.witness.store.all())
                if perspective_active
                else self.witness.store.all()
            )
            pain = witness_couple(
                fragments=witness_pool,
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
                "scope": (
                    (perspective.scope.subject, perspective.scope.entity)
                    if perspective is not None and perspective.scope is not None
                    else None
                ),
                "copresence": (
                    perspective.copresence.chain
                    if perspective is not None
                    and perspective.copresence is not None
                    else None
                ),
                "perspective": perspective,
                "affect_weight": effective_affect_weight,
                "cue_affect": cue_affect,
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

            # Fact tier. Step B swaps the pure-cosine search for a
            # three-channel hybrid (BM25 + cosine + entity-overlap, RRF)
            # when ``fact_hybrid_retrieval`` is on; otherwise defaults to
            # the shipped cosine-only path.
            if self.fact_hybrid_retrieval:
                fact_hits = self._hybrid_rank_facts(
                    cue,
                    cue_embedding,
                    top_k=self.fact_top_k,
                    cosine_floor=self.fact_cosine_floor,
                )
            else:
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
            # candidates to pull up. R1 widens it to every fragment in
            # the store (bounded by the provider's per-call document cap
            # at the rerank layer). Non-rerank path keeps the original
            # top-K to preserve substrate behavior.
            if self.cross_encoder_rerank_all:
                cos_top_k = min(
                    len(self.companion.store.all()),
                    HIERARCHICAL_RERANK_ALL_MAX_POOL,
                )
            elif self.cross_encoder_rerank:
                cos_top_k = HIERARCHICAL_RERANK_POOL_SIZE
            else:
                cos_top_k = HIERARCHICAL_FRAGMENT_TOP_K

            # Phase 2 R2 — BM25 + semantic hybrid. RRF-fuse the top-K
            # from BM25 (literal-token recall) and cosine (semantic
            # recall) into a single pool. Keeps the same cos_top_k
            # pool-size semantics so downstream rerank/reader paths are
            # unchanged.
            if self.bm25_hybrid:
                fragment_hits = self._hybrid_rank_fragments(
                    cue, cue_embedding, top_k=cos_top_k
                )
            else:
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
        #
        # Phase 2 R8-c — within-session rerank widens the rerank input
        # pool from the cosine top-150 to ALL turns in the top-M
        # sessions. The session set is picked by max-cosine on top-150;
        # the per-turn ranking inside those sessions is then delegated
        # entirely to the cross-encoder. This routes around the failure
        # mode where the right session reaches cosine top-50 but the
        # specific gold turn ranks past 150 within that session.
        if self.cross_encoder_rerank and fragment_hits:
            if self.within_session_rerank:
                fragment_hits = self._within_session_rerank_fragments(
                    cue, fragment_hits
                )
            else:
                fragment_hits = self._cross_encoder_rerank_fragments(
                    cue, fragment_hits
                )
            if self.fact_density_tie_breaker and fragment_hits:
                fragment_hits = _apply_fact_density_tie_breaker(fragment_hits)

        # Phase 2b Step F — full-session reader context. The top-1
        # reranked fragment's session_id is the winning session; every
        # turn of that session (not just the reranked top-50) becomes
        # the DIALOGUE pool, chronologically ordered. Direct attack on
        # `session_present_gold_absent`: if the right session is picked,
        # the gold turn is guaranteed to be visible to the reader even
        # when it ranks past top-150 within session.
        if self.full_session_reader and fragment_hits:
            fragment_hits = self._full_session_fragments(fragment_hits)

        # Phase 2b Step E — localize-filter LLM stage. Runs AFTER
        # cross-encoder rerank AND after Step F's full-session union
        # (if active). This is the EMem pattern: widen the pool first
        # (rerank + session expansion), THEN prune it with a recall-
        # biased LLM filter. The filter sees the full winning-session
        # context plus cross-session rerank hits and selects a focused
        # subset for the reader. Attacks reader-side failures on
        # Step F's ADDITIVE output (gold visible but drowned in 80-500
        # turn pools).
        if self.localize_filter and fragment_hits:
            fragment_hits = self._localize_filter_fragments(cue, fragment_hits)

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

        # Phase 2b Step A — fact-to-turn lift. For each of the top-N
        # fact hits, lift the fragments named in ``source_fragment_ids``
        # directly from the fragment store. The back-pointer is literal
        # evidence that the extractor used this turn to build the fact,
        # so when the fact matches the cue, surfacing its source turn
        # is more precise than cosine or substring. Dedupe against both
        # cosine-ranked and substring-lifted turns so we never double-
        # emit a fragment. Prepended to DIALOGUE at highest priority.
        source_lift_fragments: list[Fragment] = []
        if self.fact_source_turn_lift and fact_hits:
            fragment_by_id = {frag.id: frag for frag in all_fragments}
            seen_lift_ids = (
                cosine_frag_ids | {frag.id for frag in substring_fragments}
            )
            for f, _ in fact_hits[: self.fact_source_lift_top_facts]:
                if len(source_lift_fragments) >= self.fact_source_lift_fragment_cap:
                    break
                for sid in f.source_fragment_ids:
                    if sid in seen_lift_ids:
                        continue
                    frag = fragment_by_id.get(sid)
                    if frag is None:
                        continue
                    source_lift_fragments.append(frag)
                    seen_lift_ids.add(sid)
                    if len(source_lift_fragments) >= self.fact_source_lift_fragment_cap:
                        break

        # Phase 2b Step C — entity-graph fragment lift. Build/refresh
        # the graph if the flag is on and the fact count moved. Score
        # fragments by expanded-entity mentions; take top-N as a
        # dedupe-safe "entity_boost" tier appended into DIALOGUE
        # /``fragments_used`` AFTER source_lift / substring. Keeps the
        # existing high-precision tiers at the top of the pool while
        # adding entity-structural recall for the
        # ``session_present_gold_absent`` bucket.
        entity_boost_fragments: list[Fragment] = []
        if self.entity_graph_retrieval and self.fact_store is not None:
            fact_count = len(self.fact_store)
            if (
                self._entity_graph is None
                or self._entity_graph_fact_count != fact_count
            ):
                self._entity_graph = build_entity_graph(self.fact_store.all())
                self._entity_graph_fact_count = fact_count
            if len(self._entity_graph) > 0:
                seen_boost_ids = (
                    cosine_frag_ids
                    | {f.id for f in substring_fragments}
                    | {f.id for f in source_lift_fragments}
                )
                scored = entity_graph_score_fragments(
                    self._entity_graph,
                    cue,
                    all_fragments,
                    cap_per_seed=self.entity_graph_seed_cap,
                    min_hits=self.entity_graph_min_hits,
                )
                for frag, _hits in scored:
                    if frag.id in seen_boost_ids:
                        continue
                    entity_boost_fragments.append(frag)
                    seen_boost_ids.add(frag.id)
                    if len(entity_boost_fragments) >= self.entity_graph_cap:
                        break

        # Phase 2b Step H — reconstructive assembly. R1 spreading
        # activation over the Jaccard-seeded ``self.edges`` produces an
        # activation profile keyed by fragment id. Rank fragments by
        # activation (desc, zero-filter), dedup against every existing
        # tier, and take top-N as a "reconstructive_boost" tier. This is
        # the thesis-aligned recall path — a structural signal built
        # from co-occurrence, not from embedder/BM25/entity-overlap
        # semantics. We re-use ``ActivationState`` so cue-adjacent
        # recalls get residual continuity (Principle A), but DO NOT
        # mutate ``self.activation`` here because the shipped flat
        # fallback path (``recall()``'s ``reconstruct(..., state=self.activation)``)
        # also mutates it — double-mutation inflates recall_count and
        # breaks V-metric invariants. We use a one-shot local state.
        reconstructive_fragments: list[Fragment] = []
        if self.reconstructive_retrieval and all_fragments:
            local_state = ActivationState()
            activation = spread_activation(
                fragments=all_fragments,
                cue=cue,
                state=local_state,
                hops=self.reconstructive_hops,
                hop_decay=self.reconstructive_hop_decay,
                edges=self.edges,
            )
            seen_recon_ids = (
                cosine_frag_ids
                | {f.id for f in substring_fragments}
                | {f.id for f in source_lift_fragments}
                | {f.id for f in entity_boost_fragments}
            )
            ranked_recon = sorted(
                (
                    (f, activation.get(f.id, 0.0))
                    for f in all_fragments
                    if activation.get(f.id, 0.0) > 0.0
                ),
                key=lambda it: -it[1],
            )
            for frag, _score in ranked_recon:
                if frag.id in seen_recon_ids:
                    continue
                reconstructive_fragments.append(frag)
                seen_recon_ids.add(frag.id)
                if len(reconstructive_fragments) >= self.reconstructive_cap:
                    break

        if (
            not fact_hits
            and not fragment_hits
            and not substring_fragments
            and not source_lift_fragments
            and not entity_boost_fragments
            and not reconstructive_fragments
        ):
            return None

        # Reader-facing content ordering. Two modes:
        #   legacy — source_lift → substring → fragment_hits (reranked at BOTTOM
        #   of DIALOGUE; and in `fragments_used`, facts' window-wide
        #   `source_fragment_ids` come FIRST, dominating the top-50 the reader's
        #   formatted-list shows).
        #   rerank-first — fragment_hits → substring → source_lift (reranked at
        #   TOP of DIALOGUE; and in `fragments_used`, reranked top-K comes first
        #   so `_format_fragments(limit=50)` shows the highest-precision
        #   evidence instead of noise).
        sections: list[str] = []
        if fact_lines:
            sections.append("FACTS:\n" + "\n".join(fact_lines))
        dialogue_lines: list[str] = []
        if self.reader_evidence_rerank_first:
            for frag, _ in fragment_hits:
                dialogue_lines.append(frag.content)
            for frag in substring_fragments:
                dialogue_lines.append(frag.content)
            for frag in source_lift_fragments:
                dialogue_lines.append(frag.content)
            for frag in entity_boost_fragments:
                dialogue_lines.append(frag.content)
            for frag in reconstructive_fragments:
                dialogue_lines.append(frag.content)
        else:
            for frag in source_lift_fragments:
                dialogue_lines.append(frag.content)
            for frag in substring_fragments:
                dialogue_lines.append(frag.content)
            for frag in entity_boost_fragments:
                dialogue_lines.append(frag.content)
            for frag in reconstructive_fragments:
                dialogue_lines.append(frag.content)
            for frag, _ in fragment_hits:
                dialogue_lines.append(frag.content)
        if dialogue_lines:
            sections.append("DIALOGUE:\n" + "\n".join(dialogue_lines))
        content = "\n\n".join(sections)

        source_ids: list[str] = []
        seen: set[str] = set()
        if self.reader_evidence_rerank_first:
            for frag, _ in fragment_hits:
                if frag.id not in seen:
                    seen.add(frag.id)
                    source_ids.append(frag.id)
            for frag in substring_fragments:
                if frag.id not in seen:
                    seen.add(frag.id)
                    source_ids.append(frag.id)
            for frag in source_lift_fragments:
                if frag.id not in seen:
                    seen.add(frag.id)
                    source_ids.append(frag.id)
            for frag in entity_boost_fragments:
                if frag.id not in seen:
                    seen.add(frag.id)
                    source_ids.append(frag.id)
            for frag in reconstructive_fragments:
                if frag.id not in seen:
                    seen.add(frag.id)
                    source_ids.append(frag.id)
            for f, _ in fact_hits:
                for sid in f.source_fragment_ids:
                    if sid in seen:
                        continue
                    seen.add(sid)
                    source_ids.append(sid)
        else:
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
            for frag in entity_boost_fragments:
                if frag.id not in seen:
                    seen.add(frag.id)
                    source_ids.append(frag.id)
            for frag in reconstructive_fragments:
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

    def _within_session_rerank_fragments(
        self,
        cue: str,
        cosine_hits: list[tuple[Fragment, float]],
    ) -> list[tuple[Fragment, float]]:
        """Phase 2 R8-c — two-stage: session selection then within-session rerank.

        Stage 1 (session selection): group the cosine top-150 by
        ``session_id`` (``Fragment.tags[0]``). The max cosine score of
        any fragment in a session is that session's score. Pick the
        top-M sessions by that score.

        Stage 2 (within-session rerank): pull ALL fragments whose
        ``tags[0]`` is one of the picked session ids from the full
        fragment store — not just the ones that happened to rank in
        cosine top-150. Cross-encoder rerank that scoped pool against
        the cue; return the top-50.

        The mechanism is intentionally narrow: it only widens the per-
        session reach for sessions already shown to be relevant by the
        cue, so the cross-encoder's distractor budget stays bounded by
        M · (turns-per-session) rather than growing to the full store
        (which was R1's failure mode).

        Fragments with no session_id tag (``tags`` empty or
        ``tags[0] == ""``) are passed through as a single "unsessioned"
        bucket so they can still compete.
        """
        router = self._consolidation_router
        assert router is not None  # guarded in __init__

        def _session_id_of(frag: Fragment) -> str:
            if frag.tags and frag.tags[0]:
                return frag.tags[0]
            return ""

        session_max: dict[str, float] = {}
        session_order: list[str] = []
        for frag, score in cosine_hits:
            sid = _session_id_of(frag)
            if sid not in session_max:
                session_max[sid] = score
                session_order.append(sid)
            elif score > session_max[sid]:
                session_max[sid] = score

        if not session_max:
            return []

        ranked_sessions = sorted(
            session_order, key=lambda s: -session_max[s]
        )
        top_m = ranked_sessions[: self.within_session_top_m_sessions]
        picked_set = set(top_m)

        pool: list[Fragment] = [
            frag
            for frag in self.companion.store.all()
            if _session_id_of(frag) in picked_set
        ]
        if not pool:
            return []

        documents = [frag.content for frag in pool]
        reranked = router.rerank(
            cue, documents, top_n=HIERARCHICAL_FRAGMENT_TOP_K
        )
        out: list[tuple[Fragment, float]] = []
        for idx, score in reranked:
            if 0 <= idx < len(pool):
                out.append((pool[idx], score))
        return out

    def _full_session_fragments(
        self,
        reranked_hits: list[tuple[Fragment, float]],
    ) -> list[tuple[Fragment, float]]:
        """Phase 2b Step F — ADDITIVE full-session union.

        2026-10-07 variant: an earlier design REPLACED fragment_hits
        with the winning session's turn set (top-1 session). That
        regressed by 0.047 F1 across 3 instances: top-1 session was
        often not the right session, so the replacement dropped
        correct cross-session evidence and the ``session_present_gold_absent``
        bucket did not migrate.

        The additive variant keeps every reranked top-K fragment AND
        unions in every un-reranked turn of the winning session. The
        reader sees strictly MORE context than R5: all original
        cross-session rerank hits plus the full within-session
        chronology around the top-1 anchor. Guarantees are:
          - every R5 top-K fragment survives → no evidence loss;
          - the winning session's gold turn is visible even if it
            ranked past top-150 within session → direct attack on
            `session_present_gold_absent`.

        Order: reranked fragments first (preserving the cross-encoder's
        confident ranking signal to prime the reader), then the
        remaining winning-session turns in chronological order. Score
        is passed through from rerank for ranked turns; session-only
        turns get score 0.0 as a signal to downstream metadata that
        they're in by session membership, not primary ranking.
        """
        if not reranked_hits:
            return []

        def _session_id_of(frag: Fragment) -> str:
            if frag.tags and frag.tags[0]:
                return frag.tags[0]
            return ""

        top_frag, _ = reranked_hits[0]
        winning_sid = _session_id_of(top_frag)

        reranked_ids = {frag.id for frag, _ in reranked_hits}

        session_extras = [
            frag
            for frag in self.companion.store.all()
            if _session_id_of(frag) == winning_sid
            and frag.id not in reranked_ids
        ]
        session_extras.sort(key=lambda f: f.timestamp)

        # Combined cap: reranked hits stay as-is; session-extra budget
        # fills to the overall cap. For a typical 50-turn reranked +
        # 50-turn session, this lands at ~100 turns, well under the
        # 300-turn cap and well under qwen-max 32K context.
        budget = max(0, self.full_session_reader_cap_turns - len(reranked_hits))
        if budget < len(session_extras):
            session_extras = session_extras[:budget]

        return list(reranked_hits) + [(frag, 0.0) for frag in session_extras]

    def _localize_filter_fragments(
        self,
        cue: str,
        reranked_hits: list[tuple[Fragment, float]],
    ) -> list[tuple[Fragment, float]]:
        """Phase 2b Step E — recall-biased LLM filter.

        Takes the reranked top-K pool and asks a SIMPLE-tier LLM which
        rows COULD plausibly contribute to answering the cue. Prompt
        is explicitly recall-biased ("err on inclusion"). Output is a
        subset of the input, preserving original rerank-score order
        for kept rows.

        Why recall-biased, not precision-biased: the cross-encoder
        already provides precision. The gap this step attacks is
        reader-precision failure on gold-adjacent fragments the
        cross-encoder confidently scored but the reader then
        over-filtered via its own precision prior ("not stated",
        "cannot determine"). A recall-biased filter re-primes the
        pool composition — same budget, better signal.

        Safety floor: if the LLM returns fewer than
        ``localize_filter_min_keep`` indexes, the top-N by rerank
        score are kept regardless. Covers both degenerate LLM output
        and the "LLM says 'nothing matches'" false-negative case.
        """
        if not reranked_hits:
            return []
        if self._consolidation_router is None:
            return reranked_hits

        # Cap input so prompt fits comfortably in SIMPLE-tier context.
        # The LLM's attention on an ordered list degrades past ~80
        # items anyway, so this is a quality bound, not just a cost
        # bound.
        pool = reranked_hits[: self.localize_filter_max_input]

        numbered = "\n".join(
            f"[{i}] {frag.content[: HIERARCHICAL_LOCALIZE_FILTER_CONTENT_CHARS]}"
            for i, (frag, _) in enumerate(pool)
        )
        system = (
            "You are a RECALL-biased relevance filter for a dialogue "
            "memory system. You receive numbered dialogue fragments and "
            "a question. Your job: list every fragment index that COULD "
            "plausibly contribute to answering the question — including "
            "fragments that only indirectly or partially support an "
            "answer. ERR ON THE SIDE OF INCLUSION. When in doubt, keep "
            "the fragment.\n\n"
            "Output: a single line of comma-separated integer indexes "
            "(e.g. 0,2,3,5,7). No prose, no explanation, no brackets, "
            "no extra lines. Indexes must be from the input only."
        )
        user = (
            f"Question: {cue}\n\n"
            f"Fragments:\n{numbered}\n\n"
            "Return the comma-separated indexes now."
        )

        try:
            completion = self._consolidation_router.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                ],
                tier=TaskTier.SIMPLE,
                max_tokens=HIERARCHICAL_LOCALIZE_FILTER_MAX_TOKENS,
                temperature=0.0,
            )
            kept_indexes = _parse_localize_filter_response(
                completion.content, max_index=len(pool) - 1
            )
        except Exception:
            # Any error (parsing, provider, timeout) → fall back to the
            # unfiltered pool rather than silently dropping evidence.
            return reranked_hits

        kept = [pool[i] for i in sorted(kept_indexes)]

        # Safety floor against degenerate output.
        if len(kept) < self.localize_filter_min_keep:
            seen = {id(pair) for pair in kept}
            for pair in pool:
                if len(kept) >= self.localize_filter_min_keep:
                    break
                if id(pair) in seen:
                    continue
                kept.append(pair)
                seen.add(id(pair))

        # Preserve rerank order (not the LLM's output order) so the
        # downstream Step F top-1 remains the highest-rerank-score
        # kept row, not whichever row the LLM listed first.
        kept.sort(
            key=lambda pair: next(
                (i for i, (frag, _) in enumerate(pool) if frag.id == pair[0].id),
                len(pool),
            )
        )

        # Tail of reranked_hits past max_input stays as-is (never seen
        # by the filter — can't filter what you didn't show it).
        tail = reranked_hits[self.localize_filter_max_input :]
        return kept + tail

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

    def _build_bm25_index(self) -> None:
        """Build (or rebuild) the BM25 index over current fragments.

        Called lazily on first hybrid recall. Any fragment-store mutation
        (observe/purge) invalidates the index by setting
        ``_bm25_index=None`` so the next recall rebuilds.
        """
        from rank_bm25 import BM25Okapi

        fragments = self.companion.store.all()
        self._bm25_fragments = fragments
        if not fragments:
            self._bm25_index = None
            return
        tokenized = [_bm25_tokenize(f.content) for f in fragments]
        self._bm25_index = BM25Okapi(tokenized)

    def _hybrid_rank_fragments(
        self,
        cue: str,
        cue_embedding: tuple[float, ...],
        *,
        top_k: int,
    ) -> list[tuple[Fragment, float]]:
        """RRF-fuse BM25 and cosine ranked lists over all fragments.

        BM25 scores literal-token overlap (strong on proper nouns and
        dates that paraphrase-cue embeddings bury). Cosine scores
        semantic similarity (strong on paraphrased content BM25 would
        miss on vocabulary mismatch). Reciprocal Rank Fusion merges the
        two rank lists with k=60:

            rrf(frag) = 1/(k + bm25_rank) + 1/(k + cosine_rank)

        Returns the top-K fragments by RRF score. Score field in the
        output tuple is the RRF score, not the raw channel score.
        """
        from psychology_base.companion.facts import cosine as cosine_sim

        fragments = self.companion.store.all()
        if not fragments:
            return []

        if self._bm25_index is None or self._bm25_fragments is not fragments:
            self._build_bm25_index()
        assert self._bm25_index is not None  # guarded by empty check above

        bm25_scores = self._bm25_index.get_scores(_bm25_tokenize(cue))
        bm25_order = sorted(
            range(len(fragments)),
            key=lambda i: -bm25_scores[i],
        )
        bm25_rank = {idx: rank + 1 for rank, idx in enumerate(bm25_order)}

        texts = [f.content for f in fragments]
        embeds = self._consolidation_router.embed_batch(texts, max_workers=4)
        cos_scores = [cosine_sim(cue_embedding, tuple(e)) for e in embeds]
        cos_order = sorted(
            range(len(fragments)),
            key=lambda i: -cos_scores[i],
        )
        cos_rank = {idx: rank + 1 for rank, idx in enumerate(cos_order)}

        k_rrf = HIERARCHICAL_BM25_RRF_K
        rrf: list[tuple[Fragment, float]] = []
        for i, frag in enumerate(fragments):
            s = 1.0 / (k_rrf + bm25_rank[i]) + 1.0 / (k_rrf + cos_rank[i])
            rrf.append((frag, s))
        rrf.sort(key=lambda it: -it[1])
        return rrf[:top_k]

    def _build_fact_bm25_index(self) -> None:
        """Build (or rebuild) the BM25 index over current facts.

        Called lazily on first hybrid-fact recall. Facts are append-only
        in Phase 1.5 so a length-change check is sufficient invalidation.
        Index is parked on the system; the per-recall ranking method
        reads it. On empty store leaves ``_fact_bm25_index=None`` and
        callers short-circuit.
        """
        from rank_bm25 import BM25Okapi
        from psychology_base.companion.facts import embed_text_for

        if self.fact_store is None:
            self._fact_bm25_index = None
            self._fact_bm25_facts = []
            return
        facts = self.fact_store.all()
        self._fact_bm25_facts = facts
        if not facts:
            self._fact_bm25_index = None
            return
        tokenized = [_bm25_tokenize(embed_text_for(f)) for f in facts]
        self._fact_bm25_index = BM25Okapi(tokenized)

    def _hybrid_rank_facts(
        self,
        cue: str,
        cue_embedding: tuple[float, ...],
        *,
        top_k: int,
        cosine_floor: float,
    ) -> list[tuple]:
        """RRF-fuse three channels over the fact store.

        Channels:
          1. **BM25** over ``embed_text_for(fact)`` ("entity relation
             value"). Literal-token recall for proper-noun / date /
             named-item cues cosine buries.
          2. **Cosine** over existing fact embeddings. Paraphrase-tolerant
             semantic recall (shipped signal).
          3. **Entity-overlap**: cue-token overlap with ``fact.entity``
             tokens. Boosts named-person / named-concept cues where the
             entity *is* the answer.

        RRF formula: ``score(fact) = sum_channel 1/(k + rank_channel)``
        with k=60. Facts missing from a channel (no overlap, no embedding)
        contribute 0 for that channel.

        Returns ``[(fact, rrf_score)]`` ordered by RRF desc, trimmed to
        top_k. ``cosine_floor`` is applied against the raw dense score,
        not RRF, to preserve its semantics as a quality gate — a fact
        that only reaches top_k via BM25 but with cosine 0.02 is kept
        if the floor is 0.0, dropped if the floor is 0.1. On the shipped
        hierarchical config ``cosine_floor=0.0`` so nothing is dropped.
        """
        from psychology_base.companion.facts import cosine as cosine_sim
        from psychology_base.companion.facts import embed_text_for

        assert self.fact_store is not None  # guarded by caller
        facts = self.fact_store.all()
        if not facts or top_k <= 0:
            return []

        if (
            self._fact_bm25_index is None
            or len(self._fact_bm25_facts) != len(facts)
        ):
            self._build_fact_bm25_index()
        # Build may still have left index as None on empty store, though
        # we filtered above; guard anyway for the None-return narrowing.
        if self._fact_bm25_index is None:
            return []

        cue_tokens = _bm25_tokenize(cue)
        bm25_scores = self._fact_bm25_index.get_scores(cue_tokens)
        bm25_order = sorted(
            range(len(facts)),
            key=lambda i: -bm25_scores[i],
        )
        bm25_rank = {idx: rank + 1 for rank, idx in enumerate(bm25_order)}

        # Cosine channel over fact embeddings (facts without embeddings
        # score 0 and sort to the tail — same contract as FactStore.search).
        cos_scores = [
            cosine_sim(cue_embedding, f.embedding) if f.embedding else 0.0
            for f in facts
        ]
        cos_order = sorted(
            range(len(facts)),
            key=lambda i: -cos_scores[i],
        )
        cos_rank = {idx: rank + 1 for rank, idx in enumerate(cos_order)}

        # Entity-overlap channel: count cue tokens present in the
        # fact.entity token set. We drop tokens shorter than the min
        # length so single-char junk ("I") doesn't dominate; a count of
        # zero means the fact is at the tail of this channel.
        cue_token_set = {
            t for t in cue_tokens
            if len(t) >= HIERARCHICAL_FACT_ENTITY_TOKEN_MIN_LEN
        }
        ent_scores: list[float] = []
        for f in facts:
            ent_tokens = {
                t for t in _bm25_tokenize(f.entity)
                if len(t) >= HIERARCHICAL_FACT_ENTITY_TOKEN_MIN_LEN
            }
            ent_scores.append(float(len(cue_token_set & ent_tokens)))
        ent_order = sorted(
            range(len(facts)),
            key=lambda i: -ent_scores[i],
        )
        ent_rank = {idx: rank + 1 for rank, idx in enumerate(ent_order)}

        k_rrf = HIERARCHICAL_FACT_HYBRID_RRF_K
        rrf: list[tuple] = []
        for i, f in enumerate(facts):
            s = (
                1.0 / (k_rrf + bm25_rank[i])
                + 1.0 / (k_rrf + cos_rank[i])
                + 1.0 / (k_rrf + ent_rank[i])
            )
            # Dense-floor gate: drop facts whose cosine falls below floor
            # even if they'd otherwise reach top_k via BM25/entity. On
            # hierarchical defaults (floor=0.0) this is a no-op.
            if cos_scores[i] < cosine_floor:
                continue
            rrf.append((f, s))
        rrf.sort(key=lambda it: -it[1])
        return rrf[:top_k]

    def tick(self) -> None:
        self.heartbeat.sleep()
        self.activation.decay()
        self.witness_state.activation.decay()
        self.edges.decay()
        self.witness_edges.decay()
        if self.aging_policy is not None:
            self.aging_policy.on_tick()
        if self.consolidator is not None:
            self.consolidator.maybe_consolidate(
                tick_count=self.heartbeat.sleep_ticks
            )

    def advance_virtual_time(self, ticks: int) -> None:
        """Phase 3 compression primitive — fast-forward N sleep ticks.

        Each virtual tick runs the full ``tick()`` body (activation +
        edge decay + optional consolidation). No LLM calls per tick; the
        cost is O(|edges| + |fragments|) per step. Used by the
        compression harness to span virtual days / weeks between
        scripted anchor turns at 10×–100× compression ratios.

        A loop over ``tick()`` is semantically identical; this method
        exists so scenarios can express compression intent explicitly
        (``AdvanceTurn(ticks=240)`` reads as "10 virtual days at 1 tick
        per hour" rather than 240 unexplained ``TickTurn``s).
        """
        if ticks < 0:
            raise ValueError(f"advance_virtual_time requires ticks >= 0, got {ticks}")
        for _ in range(ticks):
            self.tick()

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
