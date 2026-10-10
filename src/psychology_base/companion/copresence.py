"""Co-presence witness filter — Principle C mentalistic routing primitive.

Given a chain of agents ``(A1, A2, ..., AN)``, filter fragments to those
every chain member witnessed. "Witnessed" is encoded as the presence of
``witness:<agent>`` tags on each fragment at ingest time.

Semantics:
  - ``chain=()``       — order 0, world truth, pass-through (every event)
  - ``chain=(A,)``     — order 1, events A witnessed
  - ``chain=(A,B)``    — order 2, events both A and B co-witnessed
  - ``chain=(A,B,C)``  — order 3, events the whole chain co-witnessed

Design rationale. The heavy lifting — parsing events, computing who was
in which room at which step — belongs in the corpus-specific adapter
(HiToM container-world grammar; BigToM narrative paragraphs; FANToM
conversational turns). What belongs on the substrate is the recall-time
intersection: given fragments pre-tagged with their witness set, surface
the ones every chain agent was present for. This is Principle C,
"mentalistic objects are processes," operationalised as a tag-based
witness-set intersection — the pure analogue of
``_filter_fragments_by_scope`` for mentalistic (not possessive) scoping.

Fragments without any ``witness:`` tag at all are treated as un-sited
and pass through — back-compat for LoCoMo / FANToM / BigToM where
witness semantics aren't modeled on the fragment. Only fragments that
HAVE witness tags are subject to the intersection check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from psychology_base.eval.adapters import Fragment


@dataclass(frozen=True)
class CopresenceScope:
    """A mentalistic recall scope: the chain of nested-belief agents.

    ``chain=()``       — order 0, world truth, pass-through
    ``chain=(A,)``     — order 1, what A witnessed
    ``chain=(A,B)``    — order 2, what A thinks B knows
    ``chain=(A,B,C)``  — order 3, what A thinks B thinks C knows

    Frozen so instances are hashable and safe to cache / share across
    recalls. Chain tuple order is semantically irrelevant for the
    co-presence intersection (set-equal), but the dataclass field
    preserves document order for provenance / logging.
    """

    chain: tuple[str, ...] = ()


def _has_any_witness_tag(fragment: Fragment) -> bool:
    for t in fragment.tags:
        if t.startswith("witness:"):
            return True
    return False


def filter_fragments_by_copresence(
    fragments: Iterable[Fragment],
    scope: CopresenceScope,
) -> list[Fragment]:
    """Return the subset of ``fragments`` every chain agent witnessed.

    Pass-through rules:
      - ``scope.chain == ()`` → every fragment (world-truth / order 0)
      - fragment with no ``witness:`` tags → pass-through (un-sited)
    Filter rule:
      - fragment with ``witness:`` tags → keep iff every chain agent
        has a corresponding ``witness:<agent>`` tag on the fragment
    """

    if not scope.chain:
        return list(fragments)
    required = {f"witness:{a}" for a in scope.chain}
    kept: list[Fragment] = []
    for f in fragments:
        if not _has_any_witness_tag(f):
            kept.append(f)
            continue
        tag_set = set(f.tags)
        if required.issubset(tag_set):
            kept.append(f)
    return kept


__all__ = [
    "CopresenceScope",
    "filter_fragments_by_copresence",
]
