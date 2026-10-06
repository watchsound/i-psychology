"""Phase 1.5 FactStore — typed-but-minimal consolidation layer.

Per [phase-1.5-build-plan.md](../../../docs/phases/phase-1.5-build-plan.md)
decisions 2+3+6:

- Facts carry `(entity, relation, value, confidence)` + required source
  fragment back-pointers (Principle B grounding).
- UPDATE replaces by id (no version chain in this phase; Phase 2 adds).
- Independent capacity: `fact_cap=512` soft bound; no eviction policy in
  step 1 (step-2 gate measures per-session count so we see saturation
  before it bites).
- Step 1 is in-memory only; SQLite persistence deferred.
- Embeddings populated in step 3; step 1 stores empty `()`.

Step 1 surface:
  - ``FactStore.add(fact)`` — append with new id if ``fact.id == ""``, else insert
  - ``FactStore.upsert(fact)`` — replace by id, or add if id not present
  - ``FactStore.by_id`` / ``by_entity`` / ``all``
  - ``FactStore.next_id()`` — monotonic id generator (``fact-0001`` ...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional


FACT_ID_PREFIX = "fact-"
DEFAULT_FACT_CAP = 512
DEFAULT_COSINE_FLOOR = 0.25
DEFAULT_TOP_K = 10


def embed_text_for(fact: "Fact") -> str:
    """Canonical string an embedder consumes for a fact.

    Keeping this in one place makes the embedded content auditable: the
    recall-side ``search`` embeds the cue separately, and the symmetry
    between ingest and query is where cue-sensitive retrieval lives.
    """
    return f"{fact.entity} {fact.relation} {fact.value}".strip()


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity. Zero-length vectors return 0.0 (no signal)."""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"Dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class FactOp(Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    NOOP = "NOOP"


@dataclass(frozen=True)
class Fact:
    """Typed-but-minimal fact record.

    Phase 1.5 keeps the type system shallow — ``entity`` and ``relation``
    are free-form strings. Phase 2 may constrain them against a mentalistic
    vocabulary; doing so here would be scope-leak per the build plan §7.
    """

    id: str
    entity: str
    relation: str
    value: str
    confidence: float
    source_fragment_ids: tuple[str, ...]
    created_tick: int
    updated_tick: int
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True)
class FactOperation:
    """Output of ``extract_facts`` — the Consolidator applies these to the
    store. Keeping extraction pure (returns ops, doesn't mutate) makes
    step-2 testable with ``MockProvider``."""

    op: FactOp
    fact: Optional[Fact] = None            # for ADD / UPDATE
    target_fact_id: Optional[str] = None   # for UPDATE / NOOP


@dataclass
class FactStore:
    """In-memory fact container with monotonic id sequencing.

    ``fact_cap`` is advisory for step 1 — exceeding it logs via caller;
    eviction policy is deferred until step-2 measurement shows it's
    needed. Phase 2 reconsolidation will replace this surface with a
    versioned store, so investing in LRU here is premature.
    """

    _facts: list[Fact] = field(default_factory=list)
    _by_id: dict[str, int] = field(default_factory=dict)
    _seq: int = 0
    fact_cap: int = DEFAULT_FACT_CAP

    def next_id(self) -> str:
        self._seq += 1
        return f"{FACT_ID_PREFIX}{self._seq:04d}"

    def add(self, fact: Fact) -> Fact:
        """Append a fact. If ``fact.id`` is empty, assign the next id.
        Raises on duplicate id (caller should use ``upsert`` for that case)."""
        if fact.id == "":
            fact = replace(fact, id=self.next_id())
        if fact.id in self._by_id:
            raise ValueError(f"Fact id already present: {fact.id!r}. Use upsert().")
        self._by_id[fact.id] = len(self._facts)
        self._facts.append(fact)
        return fact

    def upsert(self, fact: Fact) -> Fact:
        """Add if id absent; replace by id otherwise. Empty id → treat as add.

        UPDATE semantics in this phase: full overwrite. Phase 2 will swap
        this for an append-to-version-chain surface.
        """
        if fact.id == "" or fact.id not in self._by_id:
            return self.add(fact)
        idx = self._by_id[fact.id]
        self._facts[idx] = fact
        return fact

    def by_id(self, fact_id: str) -> Optional[Fact]:
        idx = self._by_id.get(fact_id)
        if idx is None:
            return None
        return self._facts[idx]

    def by_entity(self, entity: str) -> list[Fact]:
        """Case-sensitive entity match. Preserves insertion order."""
        return [f for f in self._facts if f.entity == entity]

    def all(self) -> list[Fact]:
        return list(self._facts)

    def search(
        self,
        cue_embedding: tuple[float, ...],
        *,
        top_k: int = DEFAULT_TOP_K,
        cosine_floor: float = DEFAULT_COSINE_FLOOR,
    ) -> list[tuple[Fact, float]]:
        """Return ``(fact, cosine_score)`` ordered by score desc.

        Facts without embeddings are skipped silently — recall should
        never crash because the embedder is cold on a subset of the store.
        Scores below ``cosine_floor`` are filtered out *after* top-K
        selection, so a thin set is a signal to the recall-dispatch
        back-off path (see ``CompanionSystem.recall``).
        """
        if not cue_embedding or top_k <= 0:
            return []
        scored: list[tuple[Fact, float]] = []
        for f in self._facts:
            if not f.embedding:
                continue
            scored.append((f, cosine(cue_embedding, f.embedding)))
        scored.sort(key=lambda it: it[1], reverse=True)
        return [(f, s) for f, s in scored[:top_k] if s >= cosine_floor]

    def __len__(self) -> int:
        return len(self._facts)
