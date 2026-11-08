"""Step 2 reconstruction: R1 spreading + R2 generative + R7 context-conditioning.

The three mechanisms interact as a pipeline, cue → activation → eligible pool →
reconstructed utterance. Each is deterministic given a seed; the state-advancing
side-effect of recall (Principle A) lives in the activation residual, which
decays between calls and is partially refreshed each cue.

Deliberately NOT in step 2:
    - Edge decay from capacity pressure                  (step 3: S7 Hebbian + S1 cap)
    - Witness coupling into the reconstruction params    (step 4: W4 pain)
    - LLM-in-the-loop coalescence                        (post-pilot)
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Optional

from psychology_base.companion.affect_resonance import pair_affect_score
from psychology_base.companion.edges import EdgeStore
from psychology_base.eval.adapters import Fragment
from psychology_base.companion.turn_provenance import graph_tags, evidence_key, activation_tie_key, local_neighbors, event_key
from psychology_base.companion.navigation import NavigationHints, compose_seed, snapshot_identity, validate_hints
from psychology_base.i18n import detect_lang, tokenize as i18n_tokenize


_TOKEN_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]+")
_HAN_PAIR_RE = re.compile(r"(?=([\u3400-\u4dbf\u4e00-\u9fff]{2}))")
RECALL_SELECTION_POLICY = "bounded-spread-distinct-evidence-novel-cue-v6-han-zero-seed"
_CONTEXT_SELECTION_VERSIONS = {"identity": RECALL_SELECTION_POLICY, "scaled_focus": "scaled-focus-v1"}


def context_selection_identity(name: str) -> dict:
    """Validate the instance option and return detached semantic metadata."""
    if not isinstance(name, str) or name not in _CONTEXT_SELECTION_VERSIONS:
        raise ValueError("fragment_context_selection must be 'identity' or 'scaled_focus'")
    return {"name": name, "version": _CONTEXT_SELECTION_VERSIONS[name]}


def tokens(text: str, cjk_tokenizer: str = "unicode") -> list[str]:
    """Lowercased alpha / CJK token stream. Shared across R1 edges + R2 coalescence.

    ``cjk_tokenizer="jieba"`` opts into finer-grained Chinese segmentation
    for CJK content only — zh text routes through ``i18n.tokenize`` with
    jieba, splitting "我最近感到很焦虑" into word tokens so a shorter cue
    ("焦虑") can match through shared tokens. Non-CJK text (including
    English) keeps the legacy letter-run path to preserve byte-identity
    with the pre-i18n reconstruction behavior — ``_TOKEN_RE.findall("it's")``
    returns ``["it", "s"]`` which ``i18n.tokenize("it's", "en")`` does not,
    so a naive migration would drift the English pipeline.
    """
    if cjk_tokenizer == "jieba" and detect_lang(text) == "zh":
        return [t.lower() for t in i18n_tokenize(text, "zh", tokenizer="jieba")]
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# R1 — spreading activation over a fragment graph
# ---------------------------------------------------------------------------


@dataclass
class ActivationState:
    """Residual activation per fragment id, decayed between recalls so that
    adjacent recalls on the same cue see slightly different states (Principle A).
    Decay + refresh are the step-2 state-change signal; Hebbian edge updates
    arrive in step 3.
    """

    residual: dict[str, float] = field(default_factory=dict)
    residual_decay: float = 0.6  # multiplier applied before each recall
    recall_count: int = 0
    # Saturated Hebbian spread + max-based writeback compound residuals
    # geometrically (observed ×10/recall at step-4 tuning, ×1.4/recall at
    # defaults). Without a cap we overflow float64 at ~r200 and V4 collapses
    # because every cue-overlapping fragment ties at inf. Normalise after
    # decay — uniform rescale preserves ratios, so top-k rankings and all
    # V-metrics are unaffected.
    normalize_threshold: float = 1e6
    normalize_target: float = 1.0

    def decay(self) -> None:
        self.residual = {fid: v * self.residual_decay for fid, v in self.residual.items()}
        if not self.residual:
            return
        max_val = max(self.residual.values())
        if max_val > self.normalize_threshold:
            scale = self.normalize_target / max_val
            self.residual = {fid: v * scale for fid, v in self.residual.items()}

    def set(self, fid: str, value: float) -> None:
        """Accumulative-max writeback: residual grows if new activation is higher
        than previous, but a quiet recall doesn't shrink it (that's what decay
        is for). This is the step-2 writeback contract; step-4 pain feedback
        uses `add` / `multiply` to break out of the max envelope on purpose."""
        self.residual[fid] = max(self.residual.get(fid, 0.0), value)

    def add(self, fid: str, delta: float) -> None:
        """Direct residual bump (step-4 W4 pain feedback). Unlike `set`, this
        can push a residual above its max writeback — the Witness's second
        perspective is supposed to override the Companion's self-reinforcing
        selection."""
        self.residual[fid] = max(0.0, self.residual.get(fid, 0.0) + delta)

    def multiply(self, fid: str, factor: float) -> None:
        """Direct residual scale (step-4 W4 pain feedback). Used to suppress
        companion-only fragments on pain, so the pool actually shifts rather
        than only growing."""
        self.residual[fid] = max(0.0, self.residual.get(fid, 0.0) * factor)

    def get(self, fid: str) -> float:
        return self.residual.get(fid, 0.0)


def _graph_prior_tags(fragment: Fragment) -> set[str]:
    """Tag view for initial graph similarity; stored provenance is unchanged."""
    return graph_tags(fragment)


def jaccard_prior(a: Fragment, b: Fragment, cjk_tokenizer: str = "unicode") -> float:
    """Token-overlap Jaccard + tag-overlap Jaccard, averaged. The initial edge
    weight stored on observe; Hebbian co-activation adjusts it from there."""
    at = set(tokens(a.content, cjk_tokenizer))
    bt = set(tokens(b.content, cjk_tokenizer))
    tok_j = len(at & bt) / len(at | bt) if (at | bt) else 0.0
    aa, bb = _graph_prior_tags(a), _graph_prior_tags(b)
    tag_j = len(aa & bb) / len(aa | bb) if (aa | bb) else 0.0
    return 0.6 * tok_j + 0.4 * tag_j


def effective_edge_weight(
    a: Fragment,
    b: Fragment,
    edges: Optional[EdgeStore],
    cjk_tokenizer: str = "unicode",
) -> float:
    """Edge weight used by spreading activation. When an EdgeStore is wired
    (step 3+) its learned weight replaces the Jaccard prior — initial Jaccard
    is seeded at observe time, so a non-zero stored weight always exists for
    pairs that would have had one under step 2."""
    if edges is not None:
        return edges.get(a.id, b.id)
    return jaccard_prior(a, b, cjk_tokenizer)


def _han_pair_positions(words: set[str]) -> dict[str, set[tuple[str, int]]]:
    """Subterms within existing tokens; never join tokens or punctuation."""
    positions: dict[str, set[tuple[str, int]]] = {}
    for word in words:
        for match in _HAN_PAIR_RE.finditer(word):
            positions.setdefault(match[1], set()).add((word, match.start()))
    return positions


def _disjoint_pair_positions(left: set[tuple[str, int]], right: set[tuple[str, int]]) -> bool:
    return any(a != b or abs(i - j) >= 2 for a, i in left for b, j in right)


def _cue_activation(fragment_tokens: list[tuple[str, set[str]]], cue_tokens: set[str]) -> dict[str, float]:
    """Exact seeds first; rescue zero overlap with two independent Han subterms.

    The fallback is pool-wide: any exact seed preserves the previous scores.
    Two distinct pairs must occupy non-overlapping positions on both sides;
    a repeated pair or one three-character span is insufficient. A supported
    fragment gets one vote, so expansion cannot inflate the fixed cue budget.
    These are retrieval hints, not evidence that a remembered claim is true.
    """
    matches = {fid: float(len(cue_tokens & words)) for fid, words in fragment_tokens}
    if any(matches.values()):
        return matches
    cue_positions = _han_pair_positions(cue_tokens)
    if len(cue_positions) < 2:
        return matches
    for fid, words in fragment_tokens:
        positions = _han_pair_positions(words)
        for left, right in combinations(cue_positions.keys() & positions.keys(), 2):
            if (_disjoint_pair_positions(cue_positions[left], cue_positions[right])
                    and _disjoint_pair_positions(positions[left], positions[right])):
                matches[fid] = 1.0
                break
    return matches


@dataclass
class RecallTokens:
    cue: set[str]
    fragments: list[tuple[str, set[str]]]
    focus: Optional[set[str]] = None
    ordered_cue: Optional[list[str]] = None
    ordered_fragments: Optional[list[tuple[str, list[str]]]] = None
    ordered_focus: Optional[list[str]] = None
    identity: Optional[str] = None


def validate_focus_cue(focus_cue: Optional[str]) -> None:
    if focus_cue is not None and not isinstance(focus_cue, str):
        raise ValueError("focus_cue must be a string or None")


def recall_tokens(fragments: list[Fragment], cue: str, cjk_tokenizer: str,
                  *, focus_cue: Optional[str] = None, retain_order: bool = False) -> RecallTokens:
    """One lexical snapshot shared by spreading and contextual selection."""
    validate_focus_cue(focus_cue)
    if retain_order:
        cue_stream = tokens(cue, cjk_tokenizer)
        focus_stream = (None if focus_cue is None else
                        cue_stream if focus_cue == cue else tokens(focus_cue, cjk_tokenizer))
        streams = [(f.id, tokens(f.content, cjk_tokenizer)) for f in fragments]
        return RecallTokens(set(cue_stream), [(fid, set(words)) for fid, words in streams],
                            None if focus_stream is None else set(focus_stream),
                            cue_stream, streams, focus_stream,
                            snapshot_identity(fragments, cue, focus_cue, cjk_tokenizer))
    cue_words = set(tokens(cue, cjk_tokenizer))
    focus_words = None if focus_cue is None else (
        cue_words if focus_cue == cue else set(tokens(focus_cue, cjk_tokenizer)))
    return RecallTokens(cue_words, [(f.id, set(tokens(f.content, cjk_tokenizer))) for f in fragments], focus_words)


def _selection_scores(fragments: list[Fragment], weighted: dict[str, float],
                      snapshot: RecallTokens) -> dict[str, float]:
    """Production identity policy; research may bind a reviewed alternative."""
    return weighted


_IDENTITY_SELECTION_SCORES = _selection_scores


def cue_context_scores(fragments: list[Fragment], weighted: dict[str, float],
                       snapshot: RecallTokens) -> dict[str, float]:
    """Experimental bounded one-hop selection, never raw-state writeback.

    Direct cue seeds support either immediate neighbor by at most half the
    strongest seed. Max support avoids degree accumulation; no recursive
    expansion. Uniform normalization conserves the original selection mass.
    This function is not enabled by the production reconstruction policy.
    """
    return _neighbor_support_scores(fragments, weighted, snapshot, snapshot.cue)


def focused_context_scores(fragments: list[Fragment], weighted: dict[str, float],
                           snapshot: RecallTokens) -> dict[str, float]:
    """Research policy: validated context around the supplied active question."""
    return _neighbor_support_scores(fragments, weighted, snapshot,
        snapshot.cue if snapshot.focus is None else snapshot.focus)


def scaled_focus_scores(fragments: list[Fragment], weighted: dict[str, float],
                        snapshot: RecallTokens, *, _neighbors=None) -> dict[str, float]:
    """Opt-in selection with support in the activation profile's units."""
    return _neighbor_support_scores(fragments, weighted, snapshot,
        snapshot.cue if snapshot.focus is None else snapshot.focus, scale_with_mass=True,
        neighbor_index=_neighbors)


def _neighbor_support_scores(fragments: list[Fragment], weighted: dict[str, float],
                             snapshot: RecallTokens, cue_tokens: set[str],
                             *, scale_with_mass: bool = False, neighbor_index=None) -> dict[str, float]:
    """One owner for bounded max-neighbor support and mass conservation."""
    matches = _cue_activation(snapshot.fragments, cue_tokens)
    total = sum(matches.values())
    mass = sum(weighted.values())
    if total <= 0 or mass <= 0:
        return weighted
    support_mass = mass if scale_with_mass else 1.0
    proposed = dict(weighted)
    if neighbor_index is None:
        neighbor_index = local_neighbors(fragments)
    for fid, neighbors in neighbor_index.items():
        support = 0.5 * max((matches.get(other, 0.0) / total for other in neighbors), default=0.0) * support_mass
        proposed[fid] = max(proposed.get(fid, 0.0), support)
    if proposed == weighted:
        return weighted
    scale = mass / sum(proposed.values())
    return {fid: value * scale for fid, value in proposed.items()}


def query_information(fragment_tokens: list[tuple[str, set[str]]], query_words: set[str]):
    """Shared occurrence-frequency/IDF arithmetic; caller owns its query/filter."""
    frequencies = Counter(word for _, words in fragment_tokens for word in words if word in query_words)
    weights = {word: 1.0 + math.log((len(fragment_tokens) + 1) / (count + 1))
               for word, count in frequencies.items()}
    return frequencies, weights


def _residual_gate(seed: dict[str, float], residual: dict[str, float],
                   fragment_tokens: list[tuple[str, set[str]]], cue_tokens: set[str],
                   tie_keys: Optional[dict[str, str]] = None) -> float:
    """Carry state across continuations; suppress predominantly new cue focus.

    Only supported cue terms occurring in less than half of the eligible pool
    can signal a switch. Compare their smoothed-IDF information with the six
    most active live residual fragments (ties use legacy IDs or their
    occurrence-metadata-free projections).
    Unsupported/ambiguous cues retain
    state. This gates one recall's contribution, never deletes stored state.
    """
    if not any(value > 0 for value in seed.values()) or not residual:
        return 1.0
    frequencies, information = query_information(fragment_tokens, cue_tokens)
    distinctive = {word for word, count in frequencies.items() if count * 2 < len(fragment_tokens)}
    if not distinctive:
        return 1.0
    recent = set(sorted(residual, key=lambda fid: (-residual[fid], (tie_keys or {}).get(fid, fid)))[:6])
    recent_words = set().union(*(words for fid, words in fragment_tokens if fid in recent))
    weights = {word: information[word] for word in distinctive}
    total = sum(weights[word] for word in sorted(weights))
    familiar = sum(weights[word] for word in sorted(distinctive & recent_words))
    return 1.0 if familiar >= total * 0.5 else 0.0


def spread_activation(
    fragments: list[Fragment],
    cue: str,
    state: ActivationState,
    hops: int = 2,
    hop_decay: float = 0.5,
    seed_boost: float = 1.0,
    edges: Optional[EdgeStore] = None,
    cjk_tokenizer: str = "unicode",
    *,
    _token_snapshot: Optional[RecallTokens] = None,
    _navigation_hints: Optional[NavigationHints] = None,
    _navigation_weight: float = 0.25,
) -> dict[str, float]:
    """R1 — one call ⇒ one activation profile across fragments.

    A cue supplies a fixed activation budget shared by its matching fragments.
    Cue-conditioned residual activation contributes at most one budget. Each hop distributes
    (rather than copies) its source mass over weighted neighbors. This prevents
    graph size and repeated recalls from manufacturing unbounded activation.
    """
    snapshot = _token_snapshot if _token_snapshot is not None else recall_tokens(fragments, cue, cjk_tokenizer)
    cue_tokens = snapshot.cue
    live_ids = {f.id for f in fragments}
    residual = {fid: value for fid, value in state.residual.items()
                if fid in live_ids and value > 0.0}
    residual_total = sum(residual.values())
    fragment_tokens = snapshot.fragments
    matches = _cue_activation(fragment_tokens, cue_tokens)
    total_overlap = sum(matches.values())
    seed = {fid: seed_boost * overlap / total_overlap for fid, overlap in matches.items() if overlap}
    if _navigation_hints is not None:
        identity = snapshot.identity or snapshot_identity(fragments, cue, cjk_tokenizer=cjk_tokenizer)
        validate_hints(_navigation_hints, fragments, identity, _navigation_weight)
        seed = compose_seed(seed, _navigation_hints, _navigation_weight, seed_boost)
    tie_keys = {f.id: activation_tie_key(f) for f in fragments if f.id in residual}
    residual_scale = _residual_gate(seed, residual, fragment_tokens, cue_tokens, tie_keys) / max(1.0, residual_total)
    activation: dict[str, float] = {fid: value * residual_scale for fid, value in residual.items()}

    # Seed: cue-token overlap
    for fid, value in seed.items():
        activation[fid] = activation.get(fid, 0.0) + value

    # Spread
    neighbors = {}
    current = dict(activation)
    for _ in range(hops):
        delta: dict[str, float] = {}
        for src in fragments:
            src_act = current.get(src.id, 0.0)
            if src_act <= 0.0:
                continue
            if src.id not in neighbors:
                # A miss must not scan every edge; cache only sources reached
                # during these hops. Normalization and neighbor order are unchanged.
                outgoing = [(dst.id, effective_edge_weight(src, dst, edges, cjk_tokenizer))
                            for dst in fragments if dst.id != src.id]
                outgoing = [(fid, weight) for fid, weight in outgoing if weight > 0.0]
                total_weight = sum(weight for _, weight in outgoing)
                neighbors[src.id] = [(fid, weight / total_weight) for fid, weight in outgoing]
            for dst_id, weight in neighbors[src.id]:
                delta[dst_id] = delta.get(dst_id, 0.0) + src_act * weight * hop_decay
        for fid, d in delta.items():
            activation[fid] = activation.get(fid, 0.0) + d
        current = {fid: delta.get(fid, 0.0) for fid in {f.id for f in fragments}}

    return activation


# ---------------------------------------------------------------------------
# R7 — context-conditioning on the eligible pool
# ---------------------------------------------------------------------------


def context_weight(fragment: Fragment, context: Optional[dict]) -> float:
    """Multiplier applied to activation based on context dict.

    Step-2 rule: context values are matched against fragment tags; a hit multiplies
    activation by 1.5, a miss by 1.0 (no penalty — context is a boost, not a filter,
    so cue-matched fragments stay eligible even in off-context).

    This is the minimum R7 that makes V3 (mentalistic liveness) non-degenerate in
    Phase 2; context={"mode": "self"} vs. context={"mode": "other"} will produce
    different activation profiles when fragments carry those tags.
    """
    if not context:
        return 1.0
    hits = sum(1 for v in context.values() if isinstance(v, str) and v in fragment.tags)
    return 1.0 + 0.5 * hits


# ---------------------------------------------------------------------------
# R2 — generative coalescence into a reconstructed utterance
# ---------------------------------------------------------------------------


def _recall_salt(
    cue: str,
    context: Optional[dict],
    recall_count: int,
    tick_count: int,
    system_seed: int = 0,
) -> str:
    """Per-recall salt string used to produce independent per-token hashes. Varies with:
      - recall_count   → adjacent recalls on same cue+context differ   (V2 adjacent)
      - tick_count     → recalls far apart in time differ             (V2 long-horizon, V1)
      - context        → same cue under different context differs      (V3 foundation)
      - cue            → unrelated cues don't collide
      - system_seed    → 3-seed median discipline (plan §5)
    """
    return f"{system_seed}|{cue}|{sorted((context or {}).items())}|{recall_count}|{tick_count}"


def _token_keep_frac(salt: str, token: str) -> float:
    """Hash-based uniform [0, 1) per (salt, token). Independent across salts even
    on tiny token pools, where a sequential RNG stream can correlate decisions."""
    h = hashlib.sha256(f"{salt}|{token}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _tick_decayed_keep_prob(
    tick_count: int,
    base: float = 0.80,
    floor: float = 0.30,
    half_life: float = 3.0,
) -> float:
    """Fresh recalls retain ~base tokens; recalls after many ticks retain only ~floor.
    Primary driver of V2 long-horizon drift and V1 state continuity: the memory
    'fades' with elapsed ticks while the cue token (floor-regression guard)
    remains anchored.

    Band targets at the current parameterization (independent hash sampling,
    cue-token guard ignored for the back-of-envelope):
        tick 0  → p=0.80 → V2 adjacent Jaccard ≈ 0.67       (band 0.65 – 0.80 ✓)
        tick 20 → p≈0.31 → V2 long-horizon Jaccard ≈ 0.28   (band 0.20 – 0.35 ✓)
    """
    decay = 0.5 ** (tick_count / half_life)
    return floor + (base - floor) * decay


def coalesce(
    fragments_ranked: list[tuple[Fragment, float]],
    cue: str,
    context: Optional[dict],
    recall_count: int,
    tick_count: int,
    system_seed: int = 0,
    top_k: int = 6,
    cjk_tokenizer: str = "unicode",
) -> str:
    """R2 — stitch the top-k activated fragments into a reconstruction.

    Per-token hash-threshold with tick-decayed retention. Fresh recalls keep most
    tokens; aged recalls keep few. Salt differs per recall so adjacent recalls on
    the same tick still differ (V2 adjacent band). Cue token is always included.
    """
    pool = fragments_ranked[:top_k]
    if not pool:
        return ""

    token_order: list[str] = []
    token_weight: dict[str, float] = {}
    for frag, act in pool:
        for t in tokens(frag.content, cjk_tokenizer):
            if t not in token_weight:
                token_order.append(t)
                token_weight[t] = 0.0
            token_weight[t] += act

    salt = _recall_salt(cue, context, recall_count, tick_count, system_seed)
    keep_prob = _tick_decayed_keep_prob(tick_count)
    kept: list[str] = []
    for t in token_order:
        if _token_keep_frac(salt, t) < keep_prob:
            kept.append(t)

    # Floor-regression guard: always include the cue token if any fragment has it.
    cue_toks = [t for t in tokens(cue, cjk_tokenizer) if t in token_weight]
    if cue_toks and not any(t in cue_toks for t in kept):
        kept.insert(0, cue_toks[0])

    return " ".join(kept)


# ---------------------------------------------------------------------------
# Pipeline — compose R1 + R7 + R2 and update ActivationState
# ---------------------------------------------------------------------------


@dataclass
class Reconstruction:
    content: str
    fragments_used: tuple[Fragment, ...]
    activation_profile: dict[str, float]
    selection_profile: dict[str, float] = field(default_factory=dict)
    context_selection: Optional[dict] = None
    navigation: Optional[dict] = None
    selection_trace: Optional[dict] = None


def _distinct_ranked_fragments(
    ranked: list[tuple[Fragment, float]],
) -> list[tuple[Fragment, float]]:
    """Spend the reconstruction budget once per identical piece of evidence.

    Repeated turns remain separate, content-addressed observations in storage.
    At recall time their highest-ranked representative occupies one slot, so
    repetition cannot fill the entire context with the same words. Session and
    timestamp locate an occurrence; owner, speaker, perspective, affect, other
    semantic tags and version parents distinguish evidence and remain in the
    grouping key. No answer labels, anchor IDs or preferred speaker are used.
    """
    seen = set()
    distinct = []
    for fragment, score in ranked:
        key = evidence_key(fragment)
        if key not in seen:
            seen.add(key)
            distinct.append((fragment, score))
    return distinct


@dataclass(frozen=True)
class ResearchSelection:
    """Shared opt-in selection/budget contract; no serving default enables it.

    Delivery reuses legacy or occurrence admission. Other modes remain explicit
    research options, with their existing policy and consolidation restrictions.

    The caller supplies the existing application's renderer, keeping byte
    accounting identical to delivery without importing application code here.
    A required-occurrence callback closes a bounded query's target set before
    admission. Its receipt concerns this eligible pool, not complete history.
    """

    mode: str
    render_context: Callable[[list[Fragment]], str]
    max_bytes: int = 6000
    byte_counter_factory: Optional[Callable[[], Callable[[list[Fragment]], int]]] = None
    required_occurrences: Optional[Callable[[list[Fragment]], tuple[str, ...]]] = None

    def __post_init__(self):
        if self.mode not in ("legacy", "occurrences", "two_anchors", "three_anchors", *_ALLOCATION_MODES):
            raise ValueError("Unknown research selection mode")
        if not callable(self.render_context):
            raise ValueError("Research selection requires the actual context renderer")
        if type(self.max_bytes) is not int or self.max_bytes < 0:
            raise ValueError("Research selection max_bytes must be a nonnegative integer")
        if self.byte_counter_factory is not None and not callable(self.byte_counter_factory):
            raise ValueError("Research selection byte counter must be a factory")
        if self.required_occurrences is not None and (
                self.mode != 'occurrences' or not callable(self.required_occurrences)):
            raise ValueError('Required evidence needs occurrence admission and a callable')


_UTILITY_MODES = ("focus_occurrences", "utility_occurrences", "utility_spans")
_ALLOCATION_MODES = ("query_occurrences", "span_allocation", "query_span_allocation", *_UTILITY_MODES)
_ALLOCATION_MAX_EXPANSIONS = 9216


def validate_research_selection(research, policy, top_k, *, consolidation_enabled=False):
    """One option contract for the system, low-level recall and study cloning."""
    if research is None:
        return
    if not isinstance(research, ResearchSelection):
        raise ValueError("research_selection requires ResearchSelection")
    if type(top_k) is not int or top_k < 0:
        raise ValueError("Research selection top_k must be a nonnegative integer")
    if consolidation_enabled or (research.mode != "legacy" and policy not in (None, "identity")):
        raise ValueError("Research selection requires fragment recall and the common identity profile")


class _SelectionBudget:
    """One feasibility owner; serializer-specific optimization stays with its caller."""

    def __init__(self, research, top_k):
        self.research, self.top_k = research, top_k
        self.measure = (research.byte_counter_factory() if research.byte_counter_factory else
                        lambda fs: len(research.render_context(fs).encode("utf-8")))

    def size(self, fragments):
        size = self.measure(fragments)
        if type(size) is not int or size < 0:
            raise ValueError("Context byte counter returned an invalid size")
        return size

    def fits(self, fragments):
        return len(fragments) <= self.top_k and self.size(fragments) <= self.research.max_bytes


def _feasible_occurrences(ranked, budget, *, role="baseline"):
    """Stable ranked admission shared by allocation baselines and focus control."""
    selected, skips = [], []
    for fragment, score in ranked:
        if len(selected) >= budget.top_k:
            break
        if budget.fits([f for f, _ in selected] + [fragment]):
            selected.append((fragment, score))
        else:
            skips.append({"id": fragment.id, "reason": "byte_limit", "role": role})
    return selected, skips


def _allocate_occurrences(fragments, scores, ranked, positive, research, budget, snapshot):
    """Bounded set search over original occurrences; no state updates or labels."""
    if snapshot is None:
        raise ValueError("Allocation requires the existing recall token snapshot")
    by_id = {f.id: f for f in fragments}
    if len(by_id) != len(fragments):
        raise ValueError("Allocation requires unique eligible occurrence IDs")
    utility = research.mode in _UTILITY_MODES
    if utility and any(not math.isfinite(scores.get(f.id, 0.0)) for f in fragments):
        raise ValueError("Utility requires finite original scores")
    baseline, skips = _feasible_occurrences(positive, budget)
    _, weights = query_information(snapshot.fragments, snapshot.cue if snapshot.focus is None else snapshot.focus)
    words = sorted(weights)
    query_bits = {fid: sum(1 << i for i, word in enumerate(words) if word in terms)
                  for fid, terms in snapshot.fragments}
    direct = {fid: sum(weights[word] for word in words if word in terms)
              for fid, terms in snapshot.fragments}
    neighbors = local_neighbors(fragments) if utility and weights else None
    focus, focus_bound, query_bound = {}, 0.0, sum(weights[word] for word in words)
    if utility:
        hint = scaled_focus_scores(fragments, scores, snapshot, _neighbors=neighbors)
        if any(not math.isfinite(value) for value in hint.values()):
            raise ValueError("Utility requires finite focus scores")
        focus = {f.id: max(0.0, hint.get(f.id, 0.0)) for f in fragments}
        largest = set(sorted(focus, key=focus.__getitem__, reverse=True)[:budget.top_k])
        focus_bound = sum(focus[f.id] for f in fragments if f.id in largest)
        if not math.isfinite(focus_bound) or not math.isfinite(query_bound):
            raise ValueError("Utility requires finite normalization bounds")
    anchors = positive[:12] + sorted(positive, key=lambda pair: direct[pair[0].id], reverse=True)[:12]
    anchor_ids = {f.id for f, _ in anchors}
    anchor_ids = [f.id for f, _ in ranked if f.id in anchor_ids]
    index = {f.id: i for i, f in enumerate(fragments)}
    bit = {fid: 1 << i for fid, i in index.items()}
    baseline_ids = [f.id for f, _ in baseline]
    baseline_mask = sum(bit[fid] for fid in baseline_ids)
    spans, seen_spans = [], set()
    if research.mode not in ("query_occurrences", "focus_occurrences", "utility_occurrences") and weights:
        if neighbors is None:
            neighbors = local_neighbors(fragments)
        for fid in anchor_ids:
            key = event_key(by_id[fid])
            if key is None:
                continue
            near = {fid, *neighbors.get(fid, ())}
            near.update(other for adjacent in tuple(near) for other in neighbors.get(adjacent, ()))
            positions = {event_key(by_id[other])[-1]: other for other in near if event_key(by_id[other]) is not None}
            candidates = []
            for start in range(key[-1] - 2, key[-1] + 1):
                ids = tuple(positions.get(start + offset) for offset in range(3))
                if (all(ids) and all(b in neighbors.get(a, ()) for a, b in zip(ids, ids[1:]))
                        and any(direct[other] > 0 for other in ids)):
                    candidates.append(ids)
            if not candidates:
                candidates = [tuple(sorted((fid, other), key=lambda oid: event_key(by_id[oid])[-1]))
                              for other in neighbors.get(fid, ()) if direct[fid] > 0 or direct[other] > 0]
            for ids in candidates:
                if ids not in seen_spans:
                    seen_spans.add(ids)
                    spans.append({"anchor": fid, "ids": list(ids), "mask": sum(bit[other] for other in ids),
                                  "group": key[:-1]})
    options = list(dict.fromkeys([bit[fid] for fid in anchor_ids] + [s["mask"] for s in spans]))
    if len(options) > 96:
        raise ValueError("Allocation option bound exceeded")

    # Compile recall-local lookup arrays once. A candidate contains at most
    # top_k members; walking its set bits avoids scanning the whole inventory.
    # Keep ascending input order for A and sorted-token order for Q: changing
    # floating-point summation order can change a later lexicographic tie.
    input_scores = [scores.get(f.id, 0.0) for f in fragments]
    input_focus = [focus[f.id] for f in fragments] if utility else []
    input_queries = [query_bits[f.id] for f in fragments]
    query_weights = [(1 << i, weights[word]) for i, word in enumerate(words)]
    rank_position = [0] * len(fragments)
    for position, (f, _) in enumerate(ranked):
        rank_position[index[f.id]] = position
    group_bits = {}
    compiled_spans = []
    for span in spans:
        group = group_bits.setdefault(span["group"], 1 << len(group_bits))
        compiled_spans.append((span["mask"], group, tuple(index[fid] for fid in span["ids"])))
    baseline_members = [index[fid] for fid in baseline_ids]

    def evaluate(mask):
        members, covered, remaining = [], 0, mask
        while remaining:
            lowest = remaining & -remaining
            member = lowest.bit_length() - 1
            members.append(member)
            covered |= input_queries[member]
            remaining ^= lowest
        q = sum(weight for token_bit, weight in query_weights if covered & token_bit)
        a = sum(input_scores[member] for member in members)
        ordered, seen, groups = [], 0, 0
        for span_mask, group, span_members in compiled_spans:
            if span_mask & mask == span_mask:
                groups |= group
                for member in span_members:
                    member_bit = 1 << member
                    if not seen & member_bit:
                        ordered.append(member)
                        seen |= member_bit
        if mask == baseline_mask:
            ordered = baseline_members
        else:
            ordered.extend(member for member in sorted(members, key=rank_position.__getitem__)
                           if not seen & (1 << member))
        d = min(2, groups.bit_count())
        objective = ((q, a) if research.mode == "query_occurrences" else
                     (d, a) if research.mode == "span_allocation" else (q, d, a))
        if utility:
            f = sum(input_focus[member] for member in members) / focus_bound if focus_bound else 0.0
            normalized_q = q / query_bound if query_bound else 0.0
            objective = (f if research.mode == "focus_occurrences" else (f + normalized_q) / 2, a)
        key = (objective, mask == baseline_mask, tuple(-member for member in ordered))
        return ordered, key

    best, expansions, rounds, fallback = baseline_mask, 0, 0, None
    focus_members = None
    if not fragments:
        fallback = "empty_inventory"
    elif utility and budget.top_k == 0:
        fallback = "zero_fragment_budget"
    elif not weights:
        fallback = "no_query_support"
    elif utility and focus_bound <= 0:
        fallback = "no_positive_focus_mass"
    elif not positive:
        fallback = "no_positive_anchor"
    elif research.mode == "focus_occurrences":
        focused = sorted(((f, focus[f.id]) for f in fragments if focus[f.id] > 0),
                         key=lambda pair: pair[1], reverse=True)
        admitted, focus_skips = _feasible_occurrences(focused, budget, role="focus")
        skips.extend(focus_skips)
        best = sum(bit[f.id] for f, _ in admitted)
        # With no spans the ordinary raw-rank emission would erase the control's
        # focus order. Budget feasibility and delivery must use the same order.
        focus_members = [index[f.id] for f, _ in admitted]
    else:
        beam = [0]
        _, best_key = evaluate(best)
        for _ in range(6):
            heap, retained = [], set()
            for state in beam:
                for option in options:
                    if expansions >= _ALLOCATION_MAX_EXPANSIONS:
                        fallback = "expansion_limit"
                        break
                    expansions += 1
                    candidate = state | option
                    if candidate == state or candidate.bit_count() > budget.top_k or candidate in retained:
                        continue
                    members, key = evaluate(candidate)
                    if not budget.fits([fragments[member] for member in members]):
                        continue
                    if key > best_key:
                        best, best_key = candidate, key
                    if len(heap) < 16:
                        heapq.heappush(heap, (key, candidate))
                        retained.add(candidate)
                    elif key > heap[0][0]:
                        _, removed = heapq.heapreplace(heap, (key, candidate))
                        retained.remove(removed)
                        retained.add(candidate)
                if fallback:
                    break
            rounds += 1
            if fallback or not heap:
                break
            beam = [mask for _, mask in sorted(heap, reverse=True)]
        if fallback:
            best = baseline_mask
    selected_members, selected_key = evaluate(best)
    if focus_members is not None:
        selected_members = focus_members
    selected = [(fragments[member], input_scores[member]) for member in selected_members]
    actual_bytes = len(research.render_context([f for f, _ in selected]).encode("utf-8"))
    if actual_bytes != budget.size([f for f, _ in selected]) or actual_bytes > research.max_bytes:
        raise ValueError("Context byte counter differs from actual delivery")
    trace = {"version": "ordered-allocation-v3" if utility else "ordered-allocation-v2", "mode": research.mode,
             "selected": [{"id": f.id, "score": score, "reason": "focus_rank" if focus_members is not None else
                          "baseline" if fallback or best == baseline_mask else "allocated"}
                          for f, score in selected], "links": [], "skips": skips,
             "spans": [{"anchor": s["anchor"], "ids": s["ids"]} for s in spans if s["mask"] & best == s["mask"]],
             "proposed_spans": [{"anchor": s["anchor"], "ids": s["ids"]} for s in spans],
             "baseline": baseline_ids, "replaced": [fid for fid in baseline_ids if not bit[fid] & best],
             "candidate_ids": anchor_ids, "query_weights": {word: weights[word] for word in words},
             "objective": list(selected_key[0]), "fallback": fallback,
             "search": {"expansions": expansions, "rounds": rounds, "options": len(options), "approximate": True},
             "rendered_bytes": actual_bytes, "max_bytes": research.max_bytes}
    if utility:
        covered = 0
        for member in selected_members:
            covered |= input_queries[member]
        f = sum(input_focus[member] for member in sorted(selected_members)) / focus_bound if focus_bound else 0.0
        q = sum(weight for token_bit, weight in query_weights if covered & token_bit) / query_bound if query_bound else 0.0
        visible = set(anchor_ids) | set(baseline_ids) | {f.id for f, _ in selected} | largest
        visible.update(fid for span in spans for fid in span["ids"])
        trace["utility"] = {"version": "focus-query-v1", "focus_bound": focus_bound,
            "query_bound": query_bound, "F": f, "Q": q, "U": (f + q) / 2,
            "focus_bound_ids": [f.id for f in fragments if f.id in largest],
            "focus_scores": [{"id": f.id, "value": focus[f.id]} for f in fragments if f.id in visible]}
    return selected, trace


def ordered_selection(fragments: list[Fragment], scores: dict[str, float], top_k: int,
                      research: Optional[ResearchSelection] = None, *, snapshot: Optional[RecallTokens] = None):
    """One ranking/budget owner for reconstruction, provenance and reinforcement."""
    ranked = sorted(((f, scores.get(f.id, 0.0)) for f in fragments),
                    key=lambda pair: pair[1], reverse=True)
    positive = [(f, score) for f, score in ranked if score > 0.0]
    if research is None:
        return _distinct_ranked_fragments(positive)[:top_k], None
    budget = _SelectionBudget(research, top_k)
    if research.mode in _ALLOCATION_MODES:
        return _allocate_occurrences(fragments, scores, ranked, positive, research, budget, snapshot)

    chosen: dict[str, tuple[Fragment, float]] = {}
    reasons, links, skips = {}, [], []

    def admit(fragment, reason):
        if fragment.id in chosen:
            return True
        if len(chosen) >= top_k:
            return False
        candidate = [f for f, _ in chosen.values()] + [fragment]
        if not budget.fits(candidate):
            skips.append({"id": fragment.id, "reason": "byte_limit", "role": reason})
            return False
        chosen[fragment.id] = (fragment, scores.get(fragment.id, 0.0))
        reasons[fragment.id] = reason
        return True

    requirement = None
    if research.required_occurrences is not None:
        by_id = {f.id: f for f in fragments}
        ids = research.required_occurrences(fragments)
        if (len(by_id) != len(fragments) or type(ids) is not tuple
                or any(type(i) is not str or i not in by_id for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Required evidence must identify unique eligible occurrences')
        wanted = set(ids)
        required = [f for f, _ in ranked if f.id in wanted]
        complete = budget.fits(required)
        requirement = dict(ids=[f.id for f in required], complete=complete,
            reason=None if complete else 'record_limit' if len(required) > top_k else 'byte_limit',
            candidate_count=len(fragments),
            frontier_sha256=hashlib.sha256(json.dumps(sorted(by_id), separators=(',', ':')).encode()).hexdigest())
        if complete:
            # Feasibility is a property of the complete serialized set; a
            # serializer need not be monotonic on its intermediate prefixes.
            for fragment in required:
                chosen[fragment.id] = (fragment, scores.get(fragment.id, 0.0))
                reasons[fragment.id] = 'required_evidence'

    anchors = []
    if research.mode in ("two_anchors", "three_anchors"):
        count = 2 if research.mode == "two_anchors" else 3
        for fragment, _ in positive:
            if fragment.id not in chosen and admit(fragment, "anchor"):
                anchors.append(fragment)
                if len(anchors) == count:
                    break
        neighbors = local_neighbors(fragments)
        rank_index = {f.id: i for i, (f, _) in enumerate(ranked)}
        by_id = {f.id: f for f in fragments}
        for anchor in anchors:
            candidates = sorted(neighbors.get(anchor.id, ()), key=rank_index.__getitem__)
            # Only the best immediate neighbor is proposed; no second-hop or
            # fallback search manufactures another event connection.
            if candidates and admit(by_id[candidates[0]], "context"):
                links.append({"anchor": anchor.id, "context": candidates[0]})
    fill = (_distinct_ranked_fragments(positive) if research.mode == "legacy" else positive)
    for fragment, _ in fill:
        if len(chosen) >= top_k:
            break
        admit(fragment, "ranked_fill")

    ordered_ids = []
    for anchor in anchors:
        ordered_ids.append(anchor.id)
        ordered_ids.extend(link["context"] for link in links if link["anchor"] == anchor.id)
    ordered_ids.extend(chosen)
    ordered_ids = list(dict.fromkeys(ordered_ids))
    selected = [chosen[fid] for fid in ordered_ids]
    trace = {"version": "ordered-occurrences-v1", "mode": research.mode,
             "selected": [{"id": f.id, "score": score, "reason": reasons[f.id]}
                          for f, score in selected], "links": links, "skips": skips,
             "rendered_bytes": len(research.render_context([f for f, _ in selected]).encode("utf-8")),
             "max_bytes": research.max_bytes}
    if requirement is not None:
        trace['required_occurrences'] = requirement
    return selected, trace


def reconstruct(
    fragments: list[Fragment],
    cue: str,
    context: Optional[dict],
    state: ActivationState,
    tick_count: int,
    system_seed: int = 0,
    top_k: int = 6,
    edges: Optional[EdgeStore] = None,
    hebbian_delta: float = 0.05,
    *,
    cue_affect: Optional[dict] = None,
    affect_weight: float = 0.0,
    cjk_tokenizer: str = "unicode",
    focus_cue: Optional[str] = None,
    fragment_context_selection: Optional[str] = None,
    research_selection: Optional[ResearchSelection] = None,
    _navigation_hints: Optional[NavigationHints] = None,
    _navigation_weight: float = 0.25,
    _token_snapshot: Optional[RecallTokens] = None,
) -> Reconstruction:
    """Full pipeline. State-advancing: decays residual, computes fresh activation
    (R1 — using Hebbian edges when an EdgeStore is wired), applies context
    weighting (R7), coalesces top-k (R2), writes activation back into residual,
    and bumps edges between all co-activated top-k pairs (S7 Hebbian). Saturation
    guard is applied after bumping so hot edges can't monopolise the budget.

    M3 affect resonance: when ``affect_weight > 0`` AND ``cue_affect`` is
    provided, ``affect_weight × pair_affect_score(fragment.affect, cue_affect)``
    is **added** to each fragment's score. Additive composition matches the
    plan's ``γ·relevance + δ·affective-resonance`` formula and lets an
    affect-matched fragment surface even when its cue-token overlap is
    zero — a multiplicative boost cannot rescue a zero-relevance fragment,
    which was wrong for the "thin semantic overlap, strong affective
    overlap" case M3 is specifically for. Fragments without numeric
    valence+arousal get a 0.0 pair score → no boost (not a penalty).
    Default ``affect_weight=0.0`` leaves M1+M2 behavior byte-identical.
    """
    validate_focus_cue(focus_cue)
    validate_research_selection(research_selection, fragment_context_selection, top_k)
    if research_selection is not None:
        if (research_selection.mode != "legacy" and
                (fragment_context_selection not in (None, "identity") or
                 _selection_scores is not _IDENTITY_SELECTION_SCORES)):
            raise ValueError("Occurrence research selection requires the common identity profile")
    if fragment_context_selection is None:
        scorer = _selection_scores
        selection_identity = (context_selection_identity("identity")
            if scorer is _IDENTITY_SELECTION_SCORES else
            {"name": f"research:{scorer.__name__}", "version": None})
    else:
        selection_identity = context_selection_identity(fragment_context_selection)
        if _selection_scores is not _IDENTITY_SELECTION_SCORES:
            raise ValueError("Runtime context selection cannot override a bound research selector")
        scorer = scaled_focus_scores if fragment_context_selection == "scaled_focus" else _selection_scores
    snapshot = _token_snapshot
    if _navigation_hints is not None:
        identity = snapshot_identity(fragments, cue, focus_cue, cjk_tokenizer)
        validate_hints(_navigation_hints, fragments, identity, _navigation_weight)
        if snapshot is not None and snapshot.identity != identity:
            raise ValueError("Navigation token snapshot differs from the eligible input")
        if snapshot is None:
            snapshot = recall_tokens(fragments, cue, cjk_tokenizer,
                                     focus_cue=focus_cue, retain_order=True)
    elif snapshot is not None:
        raise ValueError("An explicit reconstruction token snapshot requires navigation hints")
    state.decay()
    if snapshot is None:
        snapshot = recall_tokens(fragments, cue, cjk_tokenizer,
                                **({"focus_cue": focus_cue} if focus_cue is not None else {}))
    raw = spread_activation(
        fragments, cue, state, edges=edges, cjk_tokenizer=cjk_tokenizer, _token_snapshot=snapshot,
        **({"_navigation_hints": _navigation_hints, "_navigation_weight": _navigation_weight}
           if _navigation_hints is not None else {})
    )
    weighted: dict[str, float] = {}
    for f in fragments:
        score = raw.get(f.id, 0.0) * context_weight(f, context)
        if affect_weight > 0.0 and cue_affect is not None:
            score += affect_weight * pair_affect_score(f.affect, cue_affect)
        weighted[f.id] = score
    selection = scorer(fragments, weighted, snapshot)
    ranked, selection_trace = ordered_selection(fragments, selection, top_k, research_selection, snapshot=snapshot)
    content = coalesce(
        ranked, cue, context, state.recall_count, tick_count,
        system_seed=system_seed, top_k=len(ranked), cjk_tokenizer=cjk_tokenizer,
    )

    # Write back residual: current recall's activation becomes next recall's seed.
    for fid, act in weighted.items():
        if act > 0.0:
            state.set(fid, act)
    state.recall_count += 1

    # S7 Hebbian: top-k co-activated pairs reinforce their edges.
    if edges is not None and hebbian_delta > 0.0:
        pool = [f for f, _ in ranked]
        for i, a in enumerate(pool):
            for b in pool[i + 1 :]:
                edges.bump(a.id, b.id, hebbian_delta)
        edges.apply_saturation_guard()

    return Reconstruction(
        content=content,
        fragments_used=tuple(f for f, _ in ranked),
        activation_profile=weighted,
        selection_profile=selection,
        context_selection=selection_identity,
        selection_trace=selection_trace,
        **({"navigation": {"policy": json.loads(_navigation_hints.policy),
                            "snapshot": _navigation_hints.snapshot,
                            "reason": _navigation_hints.reason,
                            "hinted_occurrences": len(_navigation_hints.weights),
                            "accounted_bytes": _navigation_hints.accounted_bytes,
                            "mixture": _navigation_weight}}
           if _navigation_hints is not None else {}),
    )
