from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class Fragment:
    id: str
    content: str
    timestamp: datetime
    affect: dict = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    speaker: str
    content: str
    affect: dict = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Recall:
    cue: str
    content: str
    fragments_used: tuple[Fragment, ...] = ()
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class MemorySystem(Protocol):
    """Interface every candidate memory architecture must satisfy to be scored.

    Verbs are chosen to leave room for a reconstruction-based system — observe/recall
    rather than set/get. tick() is first-class so consciousness-sim systems can run
    sleep epochs between user interactions; store-shaped baselines no-op.
    inspect() is optional (returns None for black-box systems) so fragment-level
    metrics skip rather than lie when the information is unavailable.
    """

    def observe(self, event: Observation) -> None: ...
    def recall(self, cue: str, context: Optional[dict] = None) -> Recall: ...
    def tick(self) -> None: ...
    def inspect(self) -> Optional[dict]: ...
