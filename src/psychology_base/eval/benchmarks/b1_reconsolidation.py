"""B1 — reconsolidation benchmark scorers.

Measures the three deterministic sub-metrics that make up the B1 exit
gate ([docs/experiments/phase-2-mentalistic/option-M1-merkle-fragments/
README.md](../../../../docs/experiments/phase-2-mentalistic/option-M1-merkle-fragments/README.md)):

1. **Fragment preservation.** Fraction of originally-observed
   fragments still reachable in the final DAG. For an append-only
   Merkle-DAG this is proof-by-construction = 1.0 unless the store
   drops fragments (eviction, bug). Target ≥ 0.95.
2. **Rendering divergence.** Mean pairwise token-Jaccard distance
   between adjacent nodes in a chain. Captures the ideal signal of
   *this is a re-rendering, not a duplicate, not a different story*.
   Target band 0.25–0.45 — below 0.25 the rerendering is a near-dup
   (not a reinterpretation); above 0.45 the chain has broken referent
   coherence (contradiction, not reconsolidation). Token-Jaccard is a
   deterministic stand-in for embedding distance; a semantic LLM-judge
   variant is tracked separately.
3. **Within-session Jaccard.** Mean pairwise token-Jaccard across
   fragments observed within the same session. Captures local
   coherence — fragments in a session should talk about related
   things. Target ≥ 0.65.

The LLM-judge variant for *reframing vs confabulation* is defined in
[docs/benchmarks.md](../../../../docs/benchmarks.md) and will land as a
separate scorer that calls the LLMRouter; this module stays
LLM-free so unit tests are deterministic and fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from psychology_base.companion.reconstruction import tokens
from psychology_base.eval.adapters import Fragment


def _token_jaccard(a: str, b: str) -> float:
    at, bt = set(tokens(a)), set(tokens(b))
    if not (at | bt):
        return 0.0
    return len(at & bt) / len(at | bt)


def fragment_preservation(
    original_ids: list[str], final_fragments: list[Fragment]
) -> float:
    """Fraction of originally-observed fragment ids still present in
    the final fragment set. In an append-only DAG with no eviction this
    is 1.0 by construction; a value < 1.0 is diagnostic — eviction
    fired, a mutation path exists, or a bug silently dropped nodes."""
    if not original_ids:
        return 1.0
    final_ids = {f.id for f in final_fragments}
    kept = sum(1 for oid in original_ids if oid in final_ids)
    return kept / len(original_ids)


def rendering_divergence_for_chain(chain: list[Fragment]) -> float:
    """Mean token-Jaccard DISTANCE (1 - J) across adjacent pairs in a
    single chain, ordered root → leaf. A chain of length 1 has no
    adjacent pairs and returns 0.0 (no re-rendering happened yet;
    callers should filter length-1 chains before aggregating)."""
    if len(chain) < 2:
        return 0.0
    diffs = []
    for i in range(1, len(chain)):
        j = _token_jaccard(chain[i - 1].content, chain[i].content)
        diffs.append(1.0 - j)
    return sum(diffs) / len(diffs)


def rendering_divergence(chains: list[list[Fragment]]) -> float:
    """Macro-average rendering divergence across all chains with at
    least one re-rendering (length ≥ 2). Chains of length 1 do not
    contribute (nothing to compare)."""
    scored = [c for c in chains if len(c) >= 2]
    if not scored:
        return 0.0
    return sum(rendering_divergence_for_chain(c) for c in scored) / len(scored)


def within_session_jaccard(sessions: list[list[Fragment]]) -> float:
    """Mean pairwise token-Jaccard across fragments in the same
    session, macro-averaged over sessions. Sessions with fewer than
    two fragments do not contribute."""
    scores: list[float] = []
    for frags in sessions:
        if len(frags) < 2:
            continue
        pair_scores: list[float] = []
        for i in range(len(frags)):
            for j in range(i + 1, len(frags)):
                pair_scores.append(
                    _token_jaccard(frags[i].content, frags[j].content)
                )
        if pair_scores:
            scores.append(sum(pair_scores) / len(pair_scores))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def group_chains_by_root(fragments: list[Fragment]) -> list[list[Fragment]]:
    """Walk ``parent_ids`` to reconstruct chains rooted at fragments
    with empty ``parent_ids``. A fragment with multiple parents is
    attached to its first-parent chain (the first-parent-only policy
    keeps the metric well-defined for the v1 one-parent-per-rerender
    convention; multi-parent nodes are a Phase 2.1+ concern)."""
    by_id = {f.id: f for f in fragments}
    children: dict[str, list[str]] = {f.id: [] for f in fragments}
    for f in fragments:
        if f.parent_ids:
            parent = f.parent_ids[0]
            if parent in children:
                children[parent].append(f.id)

    chains: list[list[Fragment]] = []
    for f in fragments:
        if f.parent_ids:
            continue
        # DFS from root; take the first child at each branch so each
        # root yields exactly one chain (a tree with branching becomes
        # multiple chains in future work).
        chain = [f]
        current = f.id
        while children[current]:
            nxt = children[current][0]
            chain.append(by_id[nxt])
            current = nxt
        chains.append(chain)
    return chains


@dataclass
class B1Scores:
    fragment_preservation: float
    rendering_divergence: float
    within_session_jaccard: float

    def passes_gate(self) -> bool:
        return (
            self.fragment_preservation >= 0.95
            and 0.25 <= self.rendering_divergence <= 0.45
            and self.within_session_jaccard >= 0.65
        )


def score_b1(
    original_ids: list[str],
    final_fragments: list[Fragment],
    sessions: Mapping[int, list[Fragment]],
) -> B1Scores:
    """Compute the three B1 sub-metrics. ``sessions`` is a mapping of
    session index → fragments observed in that session; the mapping is
    produced by the scenario runner that drives the companion."""
    chains = group_chains_by_root(final_fragments)
    return B1Scores(
        fragment_preservation=fragment_preservation(original_ids, final_fragments),
        rendering_divergence=rendering_divergence(chains),
        within_session_jaccard=within_session_jaccard(list(sessions.values())),
    )
