from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from psychology_base.eval.adapters import (
    Fragment,
    MemorySystem,
    Observation,
    make_fragment,
)
from psychology_base.eval.scripts.schema import (
    AdvanceTurn,
    ObserveTurn,
    RecallTurn,
    RerenderTurn,
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
    # Pass-through of the memory system's recall.metadata so transcripts
    # can be audited after the fact — which scope was applied, pool
    # size, witness pain, fact-tier hit count, etc. Carries whatever
    # the memory system chose to expose; readers that only care about
    # ``content`` can ignore it (M2 added ``scope`` here).
    metadata: dict = field(default_factory=dict)


class SessionPlayer:
    def __init__(self, memory: MemorySystem) -> None:
        self.memory = memory
        self.transcript: list[RecallRecord] = []
        # label → fragment id. Populated when an ObserveTurn/RerenderTurn
        # carries a label; consumed by RerenderTurn's parent_label.
        self.labels: dict[str, str] = {}
        # session_id → list of fragments observed in that session. Used
        # by B1 within-session Jaccard. CompanionSystem fragments carry
        # content-addressed ids so the id lookup after observe/rerender
        # is unambiguous even across identical content.
        self.sessions: dict[str, list[Fragment]] = {}
        # Observation IDs ordered by scenario — ``fragment_preservation``
        # consumes this to check the final-DAG survival of all originals.
        self.original_ids: list[str] = []

    def play(self, scenario: ScriptedScenario) -> list[RecallRecord]:
        for session in scenario.sessions:
            frags_this_session: list[Fragment] = []
            for turn in session.turns:
                if isinstance(turn, ObserveTurn):
                    fid = self._observe(turn)
                    if fid is not None:
                        self._track_fragment(session.id, fid, frags_this_session)
                        if turn.label:
                            self.labels[turn.label] = fid
                elif isinstance(turn, RerenderTurn):
                    fid = self._rerender(turn)
                    if fid is not None:
                        self._track_fragment(session.id, fid, frags_this_session)
                        if turn.label:
                            self.labels[turn.label] = fid
                elif isinstance(turn, RecallTurn):
                    recall_kwargs: dict[str, Any] = {}
                    if turn.scope is not None:
                        if len(turn.scope) != 2:
                            raise ValueError(
                                "RecallTurn.scope must be a two-element "
                                "[subject, entity] list"
                            )
                        recall_kwargs["scope"] = (
                            turn.scope[0],
                            turn.scope[1],
                        )
                    recall = self.memory.recall(
                        turn.cue, dict(turn.context), **recall_kwargs
                    )
                    self.transcript.append(
                        RecallRecord(
                            session_id=session.id,
                            cue=turn.cue,
                            at=turn.at,
                            content=recall.content,
                            fragments_used=recall.fragments_used,
                            expect=turn.expect,
                            metadata=dict(recall.metadata),
                        )
                    )
                elif isinstance(turn, TickTurn):
                    self.memory.tick()
                elif isinstance(turn, AdvanceTurn):
                    advance_fn = getattr(self.memory, "advance_virtual_time", None)
                    if callable(advance_fn):
                        advance_fn(turn.ticks)
                    else:
                        for _ in range(turn.ticks):
                            self.memory.tick()
                else:
                    raise TypeError(f"Unknown turn type: {type(turn).__name__}")
            if frags_this_session:
                self.sessions[session.id] = frags_this_session
        return self.transcript

    def _observe(self, turn: ObserveTurn) -> Optional[str]:
        self.memory.observe(
            Observation(
                timestamp=turn.at,
                speaker=turn.speaker,
                content=turn.content,
                affect=dict(turn.affect),
                tags=tuple(turn.tags),
                subject_id=turn.subject,
                entity_id=turn.entity,
            )
        )
        # Content-addressed id lets us recover the just-added fragment
        # without peeking at the memory system's internals. The player
        # must mirror the observe-side tag-merge rule (M2 step 2) so
        # the hash it computes matches the hash the store actually
        # produced.
        merged_tags = tuple(turn.tags)
        if turn.subject is not None:
            merged_tags = merged_tags + (f"subject:{turn.subject}",)
        if turn.entity is not None:
            merged_tags = merged_tags + (f"entity:{turn.entity}",)
        fid = make_fragment(
            content=turn.content,
            timestamp=turn.at,
            affect=dict(turn.affect),
            tags=merged_tags,
            parent_ids=(),
        ).id
        self.original_ids.append(fid)
        return fid

    def _rerender(self, turn: RerenderTurn) -> Optional[str]:
        parent_id = self.labels.get(turn.parent_label)
        if parent_id is None:
            raise ValueError(
                f"RerenderTurn references unknown label {turn.parent_label!r}"
            )
        # Rerender is a CompanionSystem affordance; non-companion memories
        # (EchoMemory, stubs) treat the turn as a plain observe.
        rerender_fn = getattr(self.memory, "rerender", None)
        if callable(rerender_fn):
            child = rerender_fn(
                referent_id=parent_id,
                new_content=turn.content,
                timestamp=turn.at,
                affect=dict(turn.affect),
                tags=tuple(turn.tags),
            )
            return child.id
        # Fallback: observe without chaining. B1 scorers will see a flat
        # (unchained) fragment, which is the expected baseline shape.
        self.memory.observe(
            Observation(
                timestamp=turn.at,
                speaker=turn.speaker,
                content=turn.content,
                affect=dict(turn.affect),
                tags=tuple(turn.tags),
            )
        )
        return make_fragment(
            content=turn.content,
            timestamp=turn.at,
            affect=dict(turn.affect),
            tags=tuple(turn.tags),
            parent_ids=(),
        ).id

    def _track_fragment(
        self, session_id: str, fid: str, bucket: list[Fragment]
    ) -> None:
        """Resolve the id to a Fragment via the memory system's own store
        when available. Falls back to a shallow lookup via ``inspect``
        for baselines that don't expose a store attribute."""
        store = getattr(getattr(self.memory, "companion", None), "store", None)
        if store is not None:
            frag = store.by_id(fid)
            if frag is not None:
                bucket.append(frag)


def load_scenario(path: Path | str) -> ScriptedScenario:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ScriptedScenario.model_validate(raw)
