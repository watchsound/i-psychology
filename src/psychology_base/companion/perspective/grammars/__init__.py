"""Corpus-specific grammar grammars for mentalistic routing.

Each module under this package owns the shape-sensitive prompt builders
and context renderers for ONE benchmark corpus whose mentalistic routing
needs more than a generic witness-chain filter. The scorer dispatches on
``item.conversation.metadata["source"]``; builders here receive already-
filtered mentalistic context (produced by the routing layer) and project
it into the corpus's standard evaluation prompt shape.

Current corpora with dedicated grammars:
- ``hitom`` — HiToM 15-container multiple-choice protocol with nested-
  belief chain scaffolding (orders 1-4). The co-presence filter runs at
  the routing layer via ``CopresenceScope``; builders here only shape the
  prompt, not the context selection.

New corpora with corpus-specific prompt shapes (BigToM with its own
situation-shift grammar, future HiToM++, etc.) land as sibling modules
without touching the generic scorer loop.
"""

from __future__ import annotations

__all__: list[str] = []
