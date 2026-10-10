from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field


class _TurnBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at: datetime


class ObserveTurn(_TurnBase):
    type: Literal["observe"]
    speaker: str = "user"
    content: str
    affect: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    # Optional label — a later ``rerender`` turn uses this to reference
    # which prior fragment is being re-rendered. Scenarios without
    # rerender turns can omit it entirely.
    label: Optional[str] = None
    # M2 perspective tagging — subject = whose mind holds the fragment
    # (default: user). entity = which referent the fragment is about
    # (default: user). Both are encoded as ``subject:<id>`` /
    # ``entity:<id>`` tag prefixes in the resulting fragment.
    subject: Optional[str] = None
    entity: Optional[str] = None


class RecallTurn(_TurnBase):
    type: Literal["recall"]
    cue: str
    context: dict = Field(default_factory=dict)
    # Metric-side expectations; player does not act on them, metrics do.
    expect: Optional[dict] = None
    # M2 scope filter — pre-pool fragments to tag-matching
    # (subject, entity) pair before reconstruction. ``null``/absent =
    # scope-less recall (back-compat).
    scope: Optional[list[str]] = None


class TickTurn(_TurnBase):
    type: Literal["tick"]


class AdvanceTurn(_TurnBase):
    """Phase 3 compression primitive — advance ``ticks`` sleep ticks.

    Equivalent to N sequential ``TickTurn``s. Explicit about compression
    intent (``ticks: 240`` reads as "10 virtual days at 1 tick/hour");
    cheaper to write in YAML than 240 ``TickTurn`` entries.
    """

    type: Literal["advance"]
    ticks: int = Field(ge=0)


class RerenderTurn(_TurnBase):
    """A chained re-rendering of a prior observation.

    ``parent_label`` must match the ``label`` of an earlier ObserveTurn.
    Produces a new Fragment whose ``parent_ids == (prior.id,)``. This
    is the M1 reconsolidation primitive expressed in scripted scenarios.
    """

    type: Literal["rerender"]
    parent_label: str
    speaker: str = "user"
    content: str
    affect: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    # Optional label so later rerender turns can chain to this
    # rerendering (not just back to the original anchor). Enables
    # the linear-chain model where each reframe references the
    # most-recent rendering.
    label: Optional[str] = None


Turn: TypeAlias = Annotated[
    Union[ObserveTurn, RecallTurn, TickTurn, RerenderTurn, AdvanceTurn],
    Field(discriminator="type"),
]


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str = ""
    turns: list[Turn]


class ScriptedScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    benchmark: Literal["B1", "B2", "B3", "V5", "continuity"]
    description: str = ""
    sessions: list[Session]
