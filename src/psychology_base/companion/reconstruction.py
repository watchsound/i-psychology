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
import re
from dataclasses import dataclass, field
from typing import Optional

from psychology_base.companion.edges import EdgeStore
from psychology_base.eval.adapters import Fragment


_TOKEN_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]+")


def tokens(text: str) -> list[str]:
    """Lowercased alpha / CJK token stream. Shared across R1 edges + R2 coalescence."""
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


def jaccard_prior(a: Fragment, b: Fragment) -> float:
    """Token-overlap Jaccard + tag-overlap Jaccard, averaged. The initial edge
    weight stored on observe; Hebbian co-activation adjusts it from there."""
    at, bt = set(tokens(a.content)), set(tokens(b.content))
    tok_j = len(at & bt) / len(at | bt) if (at | bt) else 0.0
    aa, bb = set(a.tags), set(b.tags)
    tag_j = len(aa & bb) / len(aa | bb) if (aa | bb) else 0.0
    return 0.6 * tok_j + 0.4 * tag_j


def effective_edge_weight(
    a: Fragment, b: Fragment, edges: Optional[EdgeStore]
) -> float:
    """Edge weight used by spreading activation. When an EdgeStore is wired
    (step 3+) its learned weight replaces the Jaccard prior — initial Jaccard
    is seeded at observe time, so a non-zero stored weight always exists for
    pairs that would have had one under step 2."""
    if edges is not None:
        return edges.get(a.id, b.id)
    return jaccard_prior(a, b)


def spread_activation(
    fragments: list[Fragment],
    cue: str,
    state: ActivationState,
    hops: int = 2,
    hop_decay: float = 0.5,
    seed_boost: float = 1.0,
    edges: Optional[EdgeStore] = None,
) -> dict[str, float]:
    """R1 — one call ⇒ one activation profile across fragments.

    Seed activation: fragments whose content token-overlaps the cue get +seed_boost.
    Then propagate for `hops` hops, each hop multiplying by hop_decay and distributing
    to neighbors weighted by effective_edge_weight. Residual state from prior
    recalls is included.
    """
    cue_tokens = set(tokens(cue))
    activation: dict[str, float] = dict(state.residual)

    # Seed: cue-token overlap
    for f in fragments:
        if cue_tokens & set(tokens(f.content)):
            activation[f.id] = activation.get(f.id, 0.0) + seed_boost

    # Spread
    current = dict(activation)
    for _ in range(hops):
        delta: dict[str, float] = {}
        for src in fragments:
            src_act = current.get(src.id, 0.0)
            if src_act <= 0.0:
                continue
            for dst in fragments:
                if dst.id == src.id:
                    continue
                w = effective_edge_weight(src, dst, edges)
                if w <= 0.0:
                    continue
                delta[dst.id] = delta.get(dst.id, 0.0) + src_act * w * hop_decay
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
        for t in tokens(frag.content):
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
    cue_toks = [t for t in tokens(cue) if t in token_weight]
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
) -> Reconstruction:
    """Full pipeline. State-advancing: decays residual, computes fresh activation
    (R1 — using Hebbian edges when an EdgeStore is wired), applies context
    weighting (R7), coalesces top-k (R2), writes activation back into residual,
    and bumps edges between all co-activated top-k pairs (S7 Hebbian). Saturation
    guard is applied after bumping so hot edges can't monopolise the budget."""
    state.decay()
    raw = spread_activation(fragments, cue, state, edges=edges)
    weighted = {
        f.id: raw.get(f.id, 0.0) * context_weight(f, context)
        for f in fragments
    }
    ranked = sorted(
        ((f, weighted.get(f.id, 0.0)) for f in fragments),
        key=lambda pair: pair[1],
        reverse=True,
    )
    ranked = [(f, a) for f, a in ranked if a > 0.0]
    content = coalesce(
        ranked, cue, context, state.recall_count, tick_count,
        system_seed=system_seed, top_k=top_k,
    )

    # Write back residual: current recall's activation becomes next recall's seed.
    for fid, act in weighted.items():
        if act > 0.0:
            state.set(fid, act)
    state.recall_count += 1

    # S7 Hebbian: top-k co-activated pairs reinforce their edges.
    if edges is not None and hebbian_delta > 0.0:
        pool = [f for f, _ in ranked[:top_k]]
        for i, a in enumerate(pool):
            for b in pool[i + 1 :]:
                edges.bump(a.id, b.id, hebbian_delta)
        edges.apply_saturation_guard()

    return Reconstruction(
        content=content,
        fragments_used=tuple(f for f, _ in ranked[:top_k]),
        activation_profile=weighted,
    )
