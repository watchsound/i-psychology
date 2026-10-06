from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from psychology_base.eval.adapters import MemorySystem, Observation
from psychology_base.eval.scripts.schema import (
    ObserveTurn,
    RecallTurn,
    ScriptedScenario,
    TickTurn,
)


@dataclass
class RecallRecord:
    session_id: str
    cue: str
    at: Any
    content: str
    fragments_used: tuple
    expect: dict | None


class SessionPlayer:
    def __init__(self, memory: MemorySystem) -> None:
        self.memory = memory
        self.transcript: list[RecallRecord] = []

    def play(self, scenario: ScriptedScenario) -> list[RecallRecord]:
        for session in scenario.sessions:
            for turn in session.turns:
                if isinstance(turn, ObserveTurn):
                    self.memory.observe(
                        Observation(
                            timestamp=turn.at,
                            speaker=turn.speaker,
                            content=turn.content,
                            affect=dict(turn.affect),
                            tags=tuple(turn.tags),
                        )
                    )
                elif isinstance(turn, RecallTurn):
                    recall = self.memory.recall(turn.cue, dict(turn.context))
                    self.transcript.append(
                        RecallRecord(
                            session_id=session.id,
                            cue=turn.cue,
                            at=turn.at,
                            content=recall.content,
                            fragments_used=recall.fragments_used,
                            expect=turn.expect,
                        )
                    )
                elif isinstance(turn, TickTurn):
                    self.memory.tick()
                else:
                    raise TypeError(f"Unknown turn type: {type(turn).__name__}")
        return self.transcript


def load_scenario(path: Path | str) -> ScriptedScenario:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ScriptedScenario.model_validate(raw)
