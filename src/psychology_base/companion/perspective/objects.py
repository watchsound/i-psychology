"""MentalisticObject protocol — Principle C in code.

A mind's model of someone's mind is a *process*, not a field. Each
implementation walks tagged fragments at reconstruction time, aggregates
under the owner's perspective, and returns a reconstructed state.
Reconstruction is pure (no caching) — invalidation is automatic because
the fragment store IS the state.

The protocol has two load-bearing conventions:

1. **``PERSPECTIVE`` class-attribute.** Names the default scope the
   reconstruction uses. Callers can override by passing an explicit
   ``PerspectiveSpec`` to :meth:`reconstruct`. The attribute is
   declarative: implementations may read it to filter fragments via
   :meth:`PerspectiveSpec.apply`, or they may use an extended filter
   (e.g., :class:`TraitProfile` matches ``subject:user`` tag alone
   without the two-sided AND-contract) and treat ``PERSPECTIVE`` as a
   semantic marker rather than an enforcer. Both styles are legal; the
   protocol does not force one.

2. **``reconstruct(cls, system, context, *, perspective=None) -> Self``
   classmethod.** The primary public surface. Callers that want a
   reconstructed mentalistic object should reach for this instead of
   calling ``system.recall(perspective=...)`` directly — the object
   owns its default scope, so the caller doesn't have to know it.

:class:`TraitProfile` is the exemplar. Future reconstructive objects
(BigToM situation-shift tracker, user-model-of-mother, multi-party
trust-chain beliefs) follow the same pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional, Protocol, runtime_checkable

from psychology_base.companion.perspective.routing import PerspectiveSpec

if TYPE_CHECKING:
    from psychology_base.companion.system import CompanionSystem


@runtime_checkable
class MentalisticObject(Protocol):
    """Principle-C protocol: a reconstructive mentalistic object.

    Implementations:

    - Declare a ``PERSPECTIVE: ClassVar[PerspectiveSpec]`` naming the
      default scope.
    - Expose ``@classmethod reconstruct(cls, system, context, *,
      perspective=None) -> Self`` that returns the reconstructed object.
      Implementations may use the given ``perspective`` or fall back to
      ``cls.PERSPECTIVE``.
    - Hold NO cached state — every reconstruction walks the fragment
      store fresh.

    ``@runtime_checkable`` so ``isinstance(obj, MentalisticObject)``
    works at runtime for protocol-conformance tests.
    """

    PERSPECTIVE: ClassVar[PerspectiveSpec]

    @classmethod
    def reconstruct(
        cls,
        system: "CompanionSystem",
        context: Optional[dict] = None,
        *,
        perspective: Optional[PerspectiveSpec] = None,
    ) -> "MentalisticObject": ...


__all__ = ["MentalisticObject"]
