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


class RecallTurn(_TurnBase):
    type: Literal["recall"]
    cue: str
    context: dict = Field(default_factory=dict)
    # Metric-side expectations; player does not act on them, metrics do.
    expect: Optional[dict] = None


class TickTurn(_TurnBase):
    type: Literal["tick"]


Turn: TypeAlias = Annotated[
    Union[ObserveTurn, RecallTurn, TickTurn],
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
    benchmark: Literal["B1", "B2", "B3"]
    description: str = ""
    sessions: list[Session]
