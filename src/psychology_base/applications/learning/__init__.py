"""学习陪伴 — resilient learning tutor.

The TutorAgent is a composition over `CompanionSystem` (memory substrate)
+ safety modules + scaffolding-telemetry recorder. It is the north-star
application #1 per CLAUDE.md.
"""

from psychology_base.applications.learning.tutor_agent import (
    TutorAgent,
    TutorResponse,
    TutorSystemPrompt,
    classify_tutor_move,
)

__all__ = [
    "TutorAgent",
    "TutorResponse",
    "TutorSystemPrompt",
    "classify_tutor_move",
]
