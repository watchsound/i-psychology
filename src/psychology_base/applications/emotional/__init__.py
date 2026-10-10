"""情感陪伴 — long-term emotional companion.

The CompanionAgent is a composition over `CompanionSystem` (memory
substrate — observe/recall + Witness) + the full 《办法》 safety surface
+ Chain-reference audit hook. It is the north-star application #2 per
CLAUDE.md.

Package name is `emotional` (not `companion`) to disambiguate from the
substrate-level `psychology_base.companion` package.
"""

from psychology_base.applications.emotional.companion_agent import (
    CompanionAgent,
    CompanionResponse,
    CompanionSystemPrompt,
)

__all__ = [
    "CompanionAgent",
    "CompanionResponse",
    "CompanionSystemPrompt",
]
