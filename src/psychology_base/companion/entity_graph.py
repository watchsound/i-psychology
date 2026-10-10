"""Phase 2b Step C — fact-derived entity graph.

Builds an in-memory undirected graph over the fact store. Nodes are
normalized entity strings; edges connect ``fact.entity`` to ``fact.value``
with the relation name as a label. Appearing-together-in-the-same-fact
is the primary edge signal; weight is the number of distinct facts
linking a given (entity, value) pair.

The graph is used at recall time to **widen the fragment pool by
entity-expansion**: cue → entities named in cue (or similar via
tokenization) → 1-hop graph neighbors → fragments mentioning any of
those expanded entities get a bounded top-N lift into DIALOGUE.

Why this attacks ``session_present_gold_absent``: the gold session is
already visible (reranked top-K selects the right session) but the gold
*turn* is past the reader's 50-row cut-off. Entity-expansion surfaces
turns that mention cue-relevant entities regardless of cosine/paraphrase
rank — if gold mentions a cue entity (common, since questions typically
contain the subject of the answer), the expansion lifts it.

Phase 2b scope decisions:
- Nodes are lowercased, whitespace-collapsed strings. No stemming / no
  stopword drop here — entity values are often multi-word proper nouns
  ("Charlotte's Web", "home country") where stemming distorts.
- Edges are **undirected** and **unlabeled** at the graph-structure
  level. Relation labels are stored but not used for expansion scoring
  in step 1 — scope creep risk against the 11-row target bucket.
- No multi-hop expansion in step 1. 2-hop blows up fan-out (one
  "Caroline" node can reach the entire conv-26 fact store via
  frequency-1 edges). 1-hop is already enough to cover
  entity-mentions-in-same-session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def _normalize(entity: str) -> str:
    """Lowercase + collapse whitespace. Keeps hyphens/apostrophes.

    Done once at graph-build time and once per cue-entity probe, so the
    membership check is a pure dict lookup.
    """
    return " ".join(entity.strip().lower().split())


def _tokens(text: str) -> list[str]:
    """Same shape as system._bm25_tokenize but kept local so the module
    is standalone. Lowercased alphanumeric runs."""
    return [m.group(0).lower() for m in _ENTITY_TOKEN_RE.finditer(text)]


@dataclass
class EntityGraph:
    """In-memory undirected graph over fact-derived entities.

    ``_adj[node] -> {neighbor: weight}`` carries edge weights (count of
    distinct facts linking the pair). ``_nodes`` is the full entity
    vocabulary used for cue-side membership lookup. Multi-word entities
    are stored intact (so "home country" stays one node) AND each of
    their constituent tokens is indexed into ``_token_to_nodes`` so a
    cue mentioning "country" can still reach the "home country" node.
    """

    _adj: dict[str, dict[str, int]] = field(default_factory=dict)
    _nodes: set[str] = field(default_factory=set)
    _token_to_nodes: dict[str, set[str]] = field(default_factory=dict)
    _fact_count: int = 0

    def __len__(self) -> int:
        return len(self._nodes)

    def _add_node(self, node: str) -> None:
        if node in self._nodes:
            return
        self._nodes.add(node)
        self._adj.setdefault(node, {})
        for tok in _tokens(node):
            if len(tok) < 2:
                continue
            self._token_to_nodes.setdefault(tok, set()).add(node)

    def _add_edge(self, a: str, b: str) -> None:
        if a == b:
            return
        self._adj[a][b] = self._adj[a].get(b, 0) + 1
        self._adj[b][a] = self._adj[b].get(a, 0) + 1

    def ingest_fact(self, entity: str, value: str) -> None:
        """Add a single fact's endpoints as nodes + their undirected edge.

        Both endpoints are normalized; empty or whitespace-only strings
        are skipped (defensive — extractor should never emit these but
        has in the past on malformed sessions).
        """
        a = _normalize(entity)
        b = _normalize(value)
        if not a or not b:
            return
        self._add_node(a)
        self._add_node(b)
        self._add_edge(a, b)
        self._fact_count += 1

    def entities_from_cue(self, cue: str) -> set[str]:
        """Return graph nodes relevant to the cue.

        Two match paths:
          1. **Whole-phrase match**: for every node in the graph whose
             exact normalized text appears as a substring of the
             normalized cue, include it. Handles multi-word entities
             ("home country" in "what was her home country").
          2. **Token match**: cue tokens look up into
             ``_token_to_nodes`` — any node whose constituent tokens
             intersect the cue gets included. Handles the single-token
             case ("Caroline" in "where did Caroline go").

        Both paths are needed because cue tokenization alone would
        reach "home" and "country" but not reconstruct the whole-phrase
        node "home country" without the substring check.
        """
        cue_norm = _normalize(cue)
        matched: set[str] = set()
        if not cue_norm:
            return matched
        # Whole-phrase substring pass. O(|nodes|) per cue; acceptable
        # for 1-10K-node graphs on LoCoMo-scale instances.
        for node in self._nodes:
            if " " not in node:
                # single-token node — handled by the token pass below
                continue
            if node in cue_norm:
                matched.add(node)
        # Token pass. Deliberately caps length ≥ 2 (single-char tokens
        # like "a"/"i" would blow up fan-out and never carry signal).
        for tok in _tokens(cue):
            if len(tok) < 2:
                continue
            for node in self._token_to_nodes.get(tok, ()):
                matched.add(node)
        return matched

    def expand(
        self, seeds: Iterable[str], *, cap_per_seed: int = 10
    ) -> set[str]:
        """1-hop graph expansion over the seed set.

        For each seed, include the seed itself plus up to ``cap_per_seed``
        highest-weight neighbors. Returning a set deduplicates across
        seeds. ``cap_per_seed`` is a precision guard against hub nodes
        (a frequently-named entity like "Caroline" can neighbor dozens
        of low-weight values; without the cap the expanded set covers
        the full fact store).
        """
        out: set[str] = set()
        for seed in seeds:
            s = _normalize(seed)
            if s not in self._adj:
                continue
            out.add(s)
            neighbors = sorted(
                self._adj[s].items(), key=lambda kv: -kv[1]
            )[:cap_per_seed]
            for n, _w in neighbors:
                out.add(n)
        return out

    def neighbors(self, node: str) -> dict[str, int]:
        """Raw neighbor dict for ``node`` (debug / introspection)."""
        return dict(self._adj.get(_normalize(node), {}))


def build_entity_graph(facts: Iterable) -> EntityGraph:
    """Construct an EntityGraph from an iterable of ``Fact`` records.

    Pure function so tests can build a graph from fixtures without
    standing up a full CompanionSystem. The caller owns the resulting
    graph; mutation after build is supported but discouraged (prefer
    rebuild on fact-store growth, matching the Phase 2b Step B
    length-change rebuild pattern).
    """
    graph = EntityGraph()
    for f in facts:
        graph.ingest_fact(f.entity, f.value)
    return graph


def score_fragments(
    graph: EntityGraph,
    cue: str,
    fragments: list,
    *,
    cap_per_seed: int = 10,
    min_hits: int = 1,
) -> list[tuple[object, int]]:
    """Score each fragment by distinct expanded-entity mentions.

    Returns ``[(fragment, hit_count)]`` sorted by hit count descending,
    stable within ties (preserves input order). Fragments with
    ``hit_count < min_hits`` are dropped.

    ``cap_per_seed`` throttles graph fan-out — see ``EntityGraph.expand``.
    ``min_hits=1`` keeps any fragment mentioning at least one expanded
    entity; raising this to 2 or 3 narrows to multi-entity turns which
    is the natural precision/recall knob at the call site.
    """
    seeds = graph.entities_from_cue(cue)
    if not seeds:
        return []
    expanded = graph.expand(seeds, cap_per_seed=cap_per_seed)
    if not expanded:
        return []
    # Pre-normalize matcher: for multi-token entities use substring
    # match on fragment text; for single-token use token-set membership
    # (keeps precision — "Rome" shouldn't match "chrome").
    multi: list[str] = [e for e in expanded if " " in e]
    single: set[str] = {e for e in expanded if " " not in e}
    scored: list[tuple[object, int]] = []
    for frag in fragments:
        content_norm = _normalize(frag.content)
        content_tokens = set(_tokens(frag.content))
        hits = 0
        matched_entities: set[str] = set()
        for e in multi:
            if e in content_norm and e not in matched_entities:
                hits += 1
                matched_entities.add(e)
        for tok in content_tokens:
            if tok in single and tok not in matched_entities:
                hits += 1
                matched_entities.add(tok)
        if hits >= min_hits:
            scored.append((frag, hits))
    scored.sort(key=lambda it: -it[1])
    return scored


__all__ = [
    "EntityGraph",
    "build_entity_graph",
    "score_fragments",
]
