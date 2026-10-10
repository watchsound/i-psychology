"""Perspective substrate — Principle C ("mentalistic objects are processes").

One package owning the composable predicates and the protocol that any
mentalistic object (OCEAN TraitProfile today; BigToM situation-shift
tracker, user-model-of-mother, multi-party trust chains tomorrow)
implements. Two API surfaces coexist:

- **Primary** — ``MentalisticObject.reconstruct(system, context)``: the
  object owns its default ``PERSPECTIVE``, so callers don't have to know
  the correct scope. ``TraitProfile`` is the exemplar.
- **Low-level** — ``CompanionSystem.recall(perspective=PerspectiveSpec(...))``:
  scorer dispatch paths, bespoke adapters, direct per-turn queries.

See ``docs/phases/phase-9-perspective-substrate-consolidation-plan.md``
for the full consolidation story.
"""

from __future__ import annotations

from psychology_base.companion.perspective.objects import MentalisticObject
from psychology_base.companion.perspective.routing import (
    CopresenceScope,
    PerspectiveSpec,
    ScopeSpec,
    filter_fragments_by_copresence,
    filter_fragments_by_scope,
)

__all__ = [
    "CopresenceScope",
    "MentalisticObject",
    "PerspectiveSpec",
    "ScopeSpec",
    "filter_fragments_by_copresence",
    "filter_fragments_by_scope",
]
