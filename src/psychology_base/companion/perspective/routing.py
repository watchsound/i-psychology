"""Composable mentalistic-recall predicates.

``PerspectiveSpec`` composes two sub-specs — ``ScopeSpec`` (possessive,
Phase 2 M2) and ``CopresenceScope`` (witness-chain, Phase 9+) — as
set-intersection filtering over a fragment pool. Each sub-spec is a
frozen dataclass carrying its own ``apply(fragments) -> list[Fragment]``
predicate. New sub-specs (event-window, trust-chain, narrative-frame)
land here as follow-on fields when a gate motivates them.

**Byte-identity guarantee.** ``ScopeSpec(subject, entity).apply(...)``
produces the identical kept-set as the legacy
``_filter_fragments_by_scope(fragments, (subject, entity))`` call, and
``PerspectiveSpec(copresence=cs).apply(...)`` is identical to
``filter_fragments_by_copresence(fragments, cs)``. The migration
preserves every Phase 2–8 gate's behavior — the consolidation is
re-expression, not re-semantics.

**Scope-asymmetric semantics (load-bearing for Phase 5 Track 1 FANToM
+0.791).** FANToM ingests observations with ``subject_id=speaker`` →
fragments carry ``subject:<speaker>`` but no ``entity:*`` tag. The
scorer calls ``recall(scope=("subject", target_person))`` → the AND
filter wants ``subject:subject`` AND ``entity:<target_person>``, which
no fragment carries → empty recall pool on ``BELIEF_INACCESSIBLE`` →
agent answers "I don't know" → matches gold. This asymmetric mechanism
is the FANToM architectural result; preserving it is a non-negotiable
invariant across every refactor touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from psychology_base.companion.copresence import (
    CopresenceScope,
    filter_fragments_by_copresence,
)
from psychology_base.eval.adapters import Fragment


@dataclass(frozen=True)
class ScopeSpec:
    """Possessive-scope predicate on fragment tags.

    A fragment matches iff ``f"subject:{subject}"`` AND
    ``f"entity:{entity}"`` are both in ``fragment.tags``. Fragments that
    carry NO ``subject:``/``entity:`` tags at all are treated as
    scope-less and pass through — back-compat for LoCoMo and any
    pre-M2 scenario. Scoped fragments that do not match the requested
    scope are excluded.

    This is a frozen hashable record so instances compose safely into
    ``PerspectiveSpec`` and share across recall calls.
    """

    subject: str
    entity: str

    def apply(self, fragments: Iterable[Fragment]) -> list[Fragment]:
        return filter_fragments_by_scope(fragments, (self.subject, self.entity))


def filter_fragments_by_scope(
    fragments: Iterable[Fragment],
    scope: tuple[str, str],
) -> list[Fragment]:
    """Byte-identical replacement for ``system._filter_fragments_by_scope``.

    Lifted here so the perspective package owns the primitive; the
    ``system.py`` original delegates to this function during the grace
    tick. ``ScopeSpec.apply`` is the structured call-site; this
    function is the tuple-shaped shim kept for legacy callers.
    """

    subject, entity = scope
    want_subject = f"subject:{subject}"
    want_entity = f"entity:{entity}"
    kept: list[Fragment] = []
    for f in fragments:
        has_any_scope = any(
            t.startswith("subject:") or t.startswith("entity:")
            for t in f.tags
        )
        if not has_any_scope:
            kept.append(f)
            continue
        if want_subject in f.tags and want_entity in f.tags:
            kept.append(f)
    return kept


@dataclass(frozen=True)
class PerspectiveSpec:
    """Composable mentalistic recall scope.

    Each sub-spec is a predicate on fragments; the full spec composes
    them as set-intersection in a fixed order (scope → copresence).
    Predicates are idempotent, so order is semantically irrelevant for
    correctness, but fixed for determinism.

    All sub-specs are None by default — a bare ``PerspectiveSpec()``
    is pass-through (identical to recall with no perspective kwarg).
    """

    scope: Optional[ScopeSpec] = None
    copresence: Optional[CopresenceScope] = None

    def apply(self, fragments: Iterable[Fragment]) -> list[Fragment]:
        pool: Iterable[Fragment] = fragments
        if self.scope is not None:
            pool = self.scope.apply(pool)
        if self.copresence is not None:
            pool = filter_fragments_by_copresence(pool, self.copresence)
        return list(pool)

    def is_empty(self) -> bool:
        """True iff no sub-spec is active (pass-through semantics)."""
        return self.scope is None and self.copresence is None


__all__ = [
    "CopresenceScope",
    "PerspectiveSpec",
    "ScopeSpec",
    "filter_fragments_by_copresence",
    "filter_fragments_by_scope",
]
