"""HiToM witness interpreter — Pillar E mentalistic routing (Option A).

**Design.** HiToM stories are symbolic container-world puzzles with a
tiny grammar:

- ``N A, B, C and D entered the ROOM.``  — N agents enter a room
- ``N A exited the ROOM.``                — one agent exits
- ``N A moved the OBJECT to the TARGET.`` — one agent relocates object
- ``N The OBJECT is in the LOCATION.``    — declarative initial state
- Distractor forms (``N A likes/dislikes X``, ``N A made no movements``,
  ``N A saw a Y``, ``N A lost his Z``) — unparsed; semantically inert
  but witnessed by whoever is in the room

Answering an order-N ToM question requires **witness filtering**: given
an agent chain ``(A1, A2, ..., AN)``, surface only the events where
EVERY agent in the chain was in the same room. That is the subset of
events ``A1`` could use to infer ``A2``'s belief about ``A3``'s belief
about ... ``AN``'s belief about the object.

**Why this is the correct abstraction (not the memory substrate's scope filter).**
The memory substrate's ``_filter_fragments_by_scope`` AND-contracts on
``subject:X`` AND ``entity:X`` tags; HiToM has no entity-level
observations, so scoped recall returns empty. Unscoped top-K cue
similarity picks the WRONG fragments (entry/exit over object moves)
because cue similarity isn't a ToM primitive. The correct mentalistic
routing for symbolic container-world ToM is **witness-at-time
intersection** — a stateful computation over parsed events, not a
tag-based pool filter.

This module is scorer-local because HiToM's grammar is corpus-specific.
A general-purpose witness primitive on the memory substrate (modeling
``location, participants, event_time`` on ``Observation``) is deferred
to Phase 9+; see Discovered Issues at the Pillar E close-out.

**Semantics.**
- Agent is "present" in a room from the step of their ``entered`` event
  through the step of their ``exited`` event (inclusive — they witness
  their own exit).
- Declarative events (``The X is in the Y``, preferences, non-events)
  are included if all chain agents are in the same room at that step.
- Order 0 (world truth, no chain) passes all events through.
- Order 1 (single agent) returns events that agent witnessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from psychology_base.eval.adapters import Observation


_STEP_PREFIX = re.compile(r"^(\d+)\s+")
_ENTERED = re.compile(
    r"^\d+\s+(?P<agents>.+?)\s+entered\s+the\s+(?P<room>[\w_]+)\.?\s*$"
)
_EXITED = re.compile(
    r"^\d+\s+(?P<agent>[A-Z][a-z]+)\s+exited\s+the\s+(?P<room>[\w_]+)\.?\s*$"
)
_MOVED = re.compile(
    r"^\d+\s+(?P<agent>[A-Z][a-z]+)\s+moved\s+the\s+(?P<object>[\w_]+)\s+to\s+the\s+(?P<target>[\w_]+)\.?\s*$"
)
_DECLARATIVE = re.compile(
    r"^\d+\s+The\s+(?P<object>[\w_]+)\s+is\s+in\s+the\s+(?P<location>[\w_]+)\.?\s*$"
)


@dataclass(frozen=True)
class HiToMEvent:
    """One parsed HiToM story line.

    ``agents`` is the tuple of agents performing the action (empty for
    declaratives and unparsed distractors). ``room`` is the room the
    action happens in (set for entered/exited; None for moves and
    declaratives since those happen in the agent's current room).
    """

    step: int
    raw: str
    event_type: str  # entered | exited | moved | declarative | unknown
    agents: tuple[str, ...]
    room: Optional[str]
    object: Optional[str]
    target: Optional[str]


def _parse_agent_list(fragment: str) -> tuple[str, ...]:
    """Parse an "A, B, C and D" / "A and B" / "A" agent list."""

    cleaned = fragment.replace(" and ", ", ")
    parts = [p.strip() for p in cleaned.split(",")]
    return tuple(p for p in parts if p and p[0].isupper())


def _parse_line(line: str) -> Optional[HiToMEvent]:
    """Parse a single story line into a :class:`HiToMEvent`.

    Returns None if the line has no step prefix (whitespace / junk).
    Unknown-form distractors (``N Elizabeth dislikes the tangerine``,
    ``N Isabella made no movements...``) are retained as ``unknown``
    events so witness filtering can include them when all chain agents
    are in the room (they are perceivable utterances).
    """

    m_step = _STEP_PREFIX.match(line)
    if not m_step:
        return None
    step = int(m_step.group(1))

    m = _ENTERED.match(line)
    if m:
        return HiToMEvent(
            step=step,
            raw=line,
            event_type="entered",
            agents=_parse_agent_list(m.group("agents")),
            room=m.group("room"),
            object=None,
            target=None,
        )
    m = _EXITED.match(line)
    if m:
        return HiToMEvent(
            step=step,
            raw=line,
            event_type="exited",
            agents=(m.group("agent"),),
            room=m.group("room"),
            object=None,
            target=None,
        )
    m = _MOVED.match(line)
    if m:
        return HiToMEvent(
            step=step,
            raw=line,
            event_type="moved",
            agents=(m.group("agent"),),
            room=None,
            object=m.group("object"),
            target=m.group("target"),
        )
    m = _DECLARATIVE.match(line)
    if m:
        return HiToMEvent(
            step=step,
            raw=line,
            event_type="declarative",
            agents=(),
            room=None,
            object=m.group("object"),
            target=m.group("location"),
        )
    return HiToMEvent(
        step=step,
        raw=line,
        event_type="unknown",
        agents=(),
        room=None,
        object=None,
        target=None,
    )


def parse_hitom_story(story: str) -> list[HiToMEvent]:
    """Parse a HiToM story (multi-line string) into ordered events.

    Blank lines and non-numbered lines are skipped. Events are returned
    in document order; the step prefix is preserved on ``raw`` for the
    LLM to see when the context is rendered.
    """

    events: list[HiToMEvent] = []
    for raw_line in story.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        ev = _parse_line(line)
        if ev is not None:
            events.append(ev)
    return events


def compute_presence(
    events: list[HiToMEvent],
) -> dict[str, dict[int, Optional[str]]]:
    """Return ``agent → {step → room_or_None}`` presence timeline.

    An agent is "present" in a room from the step of their ``entered``
    event through the step of their ``exited`` event, inclusive on both
    endpoints. Before entry and after exit the room is ``None``. Agents
    may re-enter a different room later (multi-scene stories).
    """

    all_agents: set[str] = set()
    for ev in events:
        for a in ev.agents:
            all_agents.add(a)

    presence: dict[str, dict[int, Optional[str]]] = {a: {} for a in all_agents}
    current: dict[str, Optional[str]] = {a: None for a in all_agents}

    sorted_events = sorted(events, key=lambda e: e.step)
    for ev in sorted_events:
        if ev.event_type == "entered":
            for a in ev.agents:
                current[a] = ev.room

        # Snapshot BEFORE applying any exit — the agent still witnesses
        # the step of their own exit (they are leaving at that moment).
        for a in all_agents:
            presence[a][ev.step] = current[a]

        if ev.event_type == "exited":
            for a in ev.agents:
                current[a] = None

    return presence


def events_witnessed_by_chain(
    events: list[HiToMEvent],
    chain: tuple[str, ...],
) -> list[HiToMEvent]:
    """Return events every agent in ``chain`` witnessed in the same room.

    Semantics: for each event's step, every agent in ``chain`` must be
    present in the SAME room. An empty chain (order 0 — world truth)
    returns every event. Chain-of-one (order 1) returns events the
    single agent witnessed.
    """

    if not chain:
        return list(events)

    presence = compute_presence(events)
    out: list[HiToMEvent] = []
    for ev in events:
        rooms: list[str] = []
        all_present = True
        for a in chain:
            room = presence.get(a, {}).get(ev.step)
            if room is None:
                all_present = False
                break
            rooms.append(room)
        if all_present and len(set(rooms)) == 1:
            out.append(ev)
    return out


def render_witnessed_events(events: list[HiToMEvent]) -> str:
    """Render filtered events as a newline-joined context block.

    The ``raw`` field preserves the HiToM step prefix so the LLM can see
    the temporal order explicitly (critical for ToM reasoning: who moved
    the object WHEN relative to who exited WHEN).
    """

    return "\n".join(ev.raw for ev in events)


def ingest_hitom_story_as_observations(
    story: str,
    *,
    base_time: Optional[datetime] = None,
    step_delta: Optional[timedelta] = None,
) -> list[Observation]:
    r"""Convert a HiToM story to witness-tagged :class:`Observation`\ s.

    Each event becomes one :class:`Observation` whose ``tags`` carry its
    co-presence set as ``witness:<agent>`` entries — one per agent in
    the room at that step, computed by :func:`compute_presence`. Also
    emits ``event_step:<N>`` for provenance and (when the event is
    located) ``location:<room>``.

    This is the write-side pairing of the substrate's
    :class:`CopresenceScope` recall-side filter: once ingested, a recall
    with ``copresence=CopresenceScope(chain=(A,B))`` surfaces exactly
    the events A and B co-witnessed.

    Design: ``speaker`` is set to the first acting agent (or ``"world"``
    for declaratives / unknown-form distractors). ``subject_id`` /
    ``entity_id`` are intentionally left unset — HiToM is a container
    world, not a conversational scope, and setting them would re-trigger
    the ``_filter_fragments_by_scope`` AND-contract bug that killed
    pre-fix Pillar E recall. Witness semantics live purely in the
    ``witness:*`` tag namespace.
    """

    t0 = base_time or datetime(2026, 1, 1, 12, 0, 0)
    delta = step_delta or timedelta(seconds=1)

    events = parse_hitom_story(story)
    presence = compute_presence(events)

    observations: list[Observation] = []
    for ev in events:
        agents_present = sorted(
            a for a, timeline in presence.items() if timeline.get(ev.step)
        )
        rooms_at_step = {
            presence[a][ev.step] for a in agents_present
        }
        tags: list[str] = [f"event_step:{ev.step}"]
        if len(rooms_at_step) == 1:
            (room,) = rooms_at_step
            if room is not None:
                tags.append(f"location:{room}")
        elif ev.event_type == "entered" and ev.room is not None:
            tags.append(f"location:{ev.room}")
        for a in agents_present:
            tags.append(f"witness:{a}")

        speaker = ev.agents[0] if ev.agents else "world"
        observations.append(
            Observation(
                timestamp=t0 + delta * ev.step,
                speaker=speaker,
                content=ev.raw,
                tags=tuple(tags),
            )
        )
    return observations


__all__ = [
    "HiToMEvent",
    "compute_presence",
    "events_witnessed_by_chain",
    "ingest_hitom_story_as_observations",
    "parse_hitom_story",
    "render_witnessed_events",
]
