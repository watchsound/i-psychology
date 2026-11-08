"""Portable event provenance in hash-covered tags, without an event store.

Readers preserve opaque tags. Only one canonical, internally consistent v1
descriptor can establish a raw event or a local co-occurrence relationship.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import re
from typing import Iterable
from uuid import UUID, uuid4

from psychology_base.eval.adapters import Fragment, content_hash


NAMESPACE = "pb-turn:"
PREFIX = NAMESPACE + "v1:"
MAX_DESCRIPTOR_BYTES = 256
ACTORS = frozenset(("user", "companion", "tutor"))


@dataclass(frozen=True)
class TurnDescriptor:
    kind: str
    group: str
    position: int
    actor: str | None

    def __post_init__(self):
        if self.kind not in ("exchange", "source-sequence"):
            raise ValueError("Unknown turn kind")
        if type(self.position) is not int or not 0 <= self.position < 2**63:
            raise ValueError("Turn position must be a bounded nonnegative integer")
        if self.actor is not None and (not isinstance(self.actor, str) or self.actor not in ACTORS):
            raise ValueError("Unknown turn actor")
        if not isinstance(self.group, str):
            raise ValueError("Turn group must be a string")
        if self.kind == "exchange":
            try:
                valid_group = str(UUID(self.group)) == self.group
            except ValueError:
                valid_group = False
            if (not valid_group or self.position not in (0, 1)
                    or self.actor not in ({"user"} if self.position == 0 else {"companion", "tutor"})):
                raise ValueError("Invalid exchange descriptor")
        elif not re.fullmatch(r"[0-9a-f]{64}", self.group):
            raise ValueError("Invalid source-sequence group")

    def encode(self) -> str:
        tag = PREFIX + json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if len(tag.encode("utf-8")) > MAX_DESCRIPTOR_BYTES:
            raise ValueError("Turn descriptor exceeds byte limit")
        return tag


def new_exchange_id() -> str:
    return str(uuid4())


def exchange_tag(group: str, actor: str, position: int) -> str:
    return TurnDescriptor("exchange", group, position, actor).encode()


def sequence_tag(source_identity: str, dialogue_id: str, position: int) -> str:
    """Identity is the pinned upstream source, not a cue, label or window."""
    identity = json.dumps([source_identity, dialogue_id], ensure_ascii=False, separators=(",", ":"))
    group = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return TurnDescriptor("source-sequence", group, position, None).encode()


@dataclass(frozen=True)
class ParsedTurn:
    status: str
    descriptor: TurnDescriptor | None = None


def parse_turn(tags: tuple[str, ...]) -> ParsedTurn:
    """Only bounded descriptor bytes are cached; never owners or opaque payloads."""
    candidates = [tag for tag in tags if tag.startswith(NAMESPACE)]
    if not candidates:
        return ParsedTurn("missing")
    if len(candidates) != 1:
        return ParsedTurn("ambiguous")
    tag = candidates[0]
    if len(tag.encode("utf-8")) > MAX_DESCRIPTOR_BYTES:
        return ParsedTurn("malformed")
    if not tag.startswith(PREFIX):
        return ParsedTurn("unsupported")
    parsed = _parse_descriptor(tag)
    if parsed.descriptor is None:
        return parsed
    speakers = {t[len("speaker:"):] for t in tags if t.startswith("speaker:")}
    # Source sequences leave actor unspecified; a single source speaker label
    # supplies evidence without claiming that speaker is the current user.
    if speakers and (len(speakers) != 1 or "" in speakers
                     or (parsed.descriptor.actor is not None and parsed.descriptor.actor not in speakers)):
        return ParsedTurn("ambiguous")
    return parsed


@lru_cache(maxsize=4096)
def _parse_descriptor(tag: str) -> ParsedTurn:
    try:
        payload = json.loads(tag[len(PREFIX):])
        if not isinstance(payload, dict) or set(payload) != {"kind", "group", "position", "actor"}:
            return ParsedTurn("malformed")
        descriptor = TurnDescriptor(**payload)
        if descriptor.encode() != tag:
            return ParsedTurn("malformed")
    except (TypeError, ValueError):
        return ParsedTurn("malformed")
    return ParsedTurn("valid", descriptor)


def graph_tags(fragment: Fragment) -> set[str]:
    return {tag for tag in fragment.tags if not tag.startswith(NAMESPACE)}


def evidence_tags(fragment: Fragment) -> tuple[str, ...]:
    tags = [tag for tag in fragment.tags if not tag.startswith(NAMESPACE)]
    parsed = parse_turn(fragment.tags)
    if parsed.descriptor is not None and parsed.descriptor.actor is not None:
        speaker = f"speaker:{parsed.descriptor.actor}"
        if speaker not in tags:
            tags.append(speaker)
    return tuple(sorted(tags))


def evidence_key(fragment: Fragment) -> tuple:
    """Existing recall equivalence, shared by selection and derived cueing."""
    semantic_tags = tuple(sorted(
        tag for tag in evidence_tags(fragment)
        if not tag.startswith("session:") and not re.fullmatch(r"session_\d+", tag)))
    return (fragment.content, semantic_tags,
            json.dumps(fragment.affect, sort_keys=True, ensure_ascii=False), fragment.parent_ids)


def activation_tie_key(fragment: Fragment) -> str:
    """Legacy keys stay exact; occurrence UUIDs cannot select residual topics."""
    if not any(tag.startswith(NAMESPACE) for tag in fragment.tags):
        return fragment.id
    tags = tuple(tag for tag in fragment.tags if not tag.startswith(NAMESPACE))
    parsed = parse_turn(fragment.tags)
    if parsed.descriptor and parsed.descriptor.actor and not any(t.startswith("speaker:") for t in tags):
        tags += (f"speaker:{parsed.descriptor.actor}",)
    return content_hash(fragment.content, fragment.timestamp, fragment.affect, tags, fragment.parent_ids)


def single_tag_value(tags: tuple[str, ...], prefix: str) -> str | None:
    """Read one nonempty metadata value; duplicate equal tags are unambiguous."""
    values = {tag[len(prefix):] for tag in tags if tag.startswith(prefix)}
    return next(iter(values)) if len(values) == 1 and "" not in values else None


def event_key(fragment: Fragment) -> tuple | None:
    parsed = parse_turn(fragment.tags)
    owner, session = single_tag_value(fragment.tags, "user:"), single_tag_value(fragment.tags, "session:")
    if fragment.parent_ids or parsed.descriptor is None or owner is None or session is None:
        return None
    d = parsed.descriptor
    return owner, session, d.kind, d.group, d.position


def _perspective_scope(fragment: Fragment) -> tuple[str, ...] | None:
    if any(len({tag for tag in fragment.tags if tag.startswith(prefix)}) > 1
           for prefix in ("subject:", "entity:")):
        return None
    return tuple(sorted({tag for tag in fragment.tags
                         if tag.startswith(("subject:", "entity:", "witness:"))}))


def source_order(fragment: Fragment) -> dict | None:
    """Project known raw turn order; opaque groups convey no event chronology."""
    key, scope = event_key(fragment), _perspective_scope(fragment)
    if key is None or scope is None:
        return None
    identity = json.dumps(["pb-source-order-v1", key[:-1], scope],
                          ensure_ascii=False, separators=(",", ":"))
    group = hashlib.blake2b(identity.encode("utf-8"), digest_size=16).hexdigest()
    return {"group": group, "position": key[-1]}


def validate_source_order(value) -> None:
    """Validate projected coordinates without treating metadata as proof."""
    if value is None:
        return
    if (not isinstance(value, dict) or set(value) != {"group", "position"}
            or not isinstance(value["group"], str) or not re.fullmatch(r"[0-9a-f]{32}", value["group"])
            or type(value["position"]) is not int or not 0 <= value["position"] < 2**63):
        raise ValueError("Invalid source-order coordinate")


def _source_groups(coordinates):
    groups: dict[str, list[tuple[int, int]]] = {}
    for row, coordinate in enumerate(coordinates):
        validate_source_order(coordinate)
        if coordinate is not None:
            groups.setdefault(coordinate['group'], []).append((row, coordinate['position']))
    return groups


def _conflict_rows(entries):
    counts = Counter(position for _, position in entries)
    return tuple(row for row, position in entries if counts[position] > 1)


def source_conflicts(coordinates: Iterable[dict | None]) -> dict[str, tuple[int, ...]]:
    """Linear conflict witnesses, using the same scoped groups as ordering."""
    return {group: rows for group, entries in _source_groups(coordinates).items()
            if (rows := _conflict_rows(entries))}


def source_relations(coordinates: Iterable[dict | None]) -> list[dict[str, list[int]]]:
    """Project strict reporting order among delivered rows, never event time.

    A duplicated position invalidates its entire scoped group. This is a
    conservative view even when an input repeats the identical occurrence.
    Missing coordinates and different groups stay incomparable. Work is O(N²)
    in the delivered bundle (applications supply at most six rows), not the
    memory store; neither fragments nor coordinates are modified.
    """
    coordinates = list(coordinates)
    result = [{'earlier': [], 'later': []} for _ in coordinates]
    for entries in _source_groups(coordinates).values():
        if _conflict_rows(entries):
            continue
        for row, position in entries:
            result[row] = {
                'earlier': [other for other, p in entries if p < position],
                'later': [other for other, p in entries if p > position],
            }
    return result


def source_reading_view(coordinates: Iterable[dict | None]) -> list[dict]:
    """Arrange a partial reporting sequence without changing recall selection.

    Blocks follow first occurrence in recall, with no order between blocks.
    Only distinct positions in one scoped source can order a block. A conflict
    invalidates the whole block; each unknown coordinate remains independent.
    recall_row refers to the original selected occurrence, including repeats.
    """
    coordinates = list(coordinates)
    groups = _source_groups(coordinates)
    seen, result = set(), []
    block = 0
    for row, coordinate in enumerate(coordinates):
        if coordinate is None:
            entries, reliable = [(row, None)], False
        else:
            group = coordinate['group']
            if group in seen:
                continue
            seen.add(group)
            entries = groups[group]
            reliable = not _conflict_rows(entries)
            if reliable:
                entries = sorted(entries, key=lambda entry: entry[1])
        for recall_row, _ in entries:
            result.append(dict(recall_row=recall_row, source_reading=dict(
                group=block, order='reporting_sequence' if reliable else 'unresolved',
                between_groups='unresolved')))
        block += 1
    return result


def validate_source_reading(records: list[dict]) -> None:
    """Recompute the complete reading view from its delivered coordinates."""
    present = [bool({'recall_row', 'source_reading'} & record.keys()) for record in records]
    if not any(present):
        return
    if (not all(present) or any(not {'recall_row', 'source_order', 'source_reading'} <= r.keys()
                              or 'source_relations' in r or 'row' in r for r in records)):
        raise ValueError('Incomplete source-reading view')
    indices = [r['recall_row'] for r in records]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(records))):
        raise ValueError('Invalid source-reading occurrence references')
    coordinates = [None] * len(records)
    for record in records:
        coordinates[record['recall_row']] = record['source_order']
    expected = source_reading_view(coordinates)
    for record, view in zip(records, expected):
        value = record['source_reading']
        if (not isinstance(value, dict) or set(value) != {'group', 'order', 'between_groups'}
                or type(value['group']) is not int
                or {key: record[key] for key in view} != view):
            raise ValueError('Invalid source-reading view')


def validate_source_relations(records: list[dict]) -> None:
    """Check the complete derived view against delivered coordinates only."""
    present = [bool({'row', 'source_relations'} & record.keys()) for record in records]
    if not any(present):
        return
    if not all(present) or any(not {'row', 'source_order', 'source_relations'} <= record.keys() for record in records):
        raise ValueError('Incomplete source-relations view')
    expected = source_relations(record['source_order'] for record in records)
    for index, (record, relation) in enumerate(zip(records, expected)):
        value = record['source_relations']
        if (type(record['row']) is not int or record['row'] != index
                or not isinstance(value, dict) or set(value) != {'earlier', 'later'}
                or any(type(value[k]) is not list or any(type(row) is not int for row in value[k])
                       for k in ('earlier', 'later')) or value != relation):
            raise ValueError('Invalid source-relations view')


def validate_events(fragments: Iterable[Fragment]) -> None:
    """Preflight one intake set (including both live stores) before mutation."""
    events: dict[tuple, Fragment] = {}
    identities: dict[str, Fragment] = {}
    for fragment in fragments:
        old = identities.setdefault(fragment.id, fragment)
        if old != fragment:
            raise ValueError(f"Fragment ID conflicts with existing payload: {fragment.id}")
        key = event_key(fragment)
        if key is not None and events.setdefault(key, fragment) != fragment:
            raise ValueError("Turn event position conflicts with existing payload")


def locally_associated(a: Fragment, b: Fragment) -> bool:
    ka, kb = event_key(a), event_key(b)
    if ka is None or kb is None or ka[:-1] != kb[:-1] or abs(ka[-1] - kb[-1]) != 1:
        return False
    # Equal provenance is conservative: association never bridges minds/rooms.
    scope = _perspective_scope(a)
    return scope is not None and scope == _perspective_scope(b)


def local_neighbors(fragments: Iterable[Fragment]) -> dict[str, tuple[str, ...]]:
    """Index only unambiguous immediate events in this already eligible pool.

    Temporary O(N) index; no inferred positions, persistent registry or pair
    scan. Conflicting event positions fail closed even for callers that have
    not passed ingestion's conflict preflight.
    """
    events: dict[tuple, Fragment | None] = {}
    for fragment in fragments:
        key = event_key(fragment)
        if key is not None:
            if key in events and events[key] != fragment:
                events[key] = None
            else:
                events[key] = fragment
    neighbors = {}
    for key, fragment in events.items():
        if fragment is None:
            continue
        candidates = (events.get((*key[:-1], key[-1] + offset)) for offset in (-1, 1))
        neighbors[fragment.id] = tuple(other.id for other in candidates
            if other is not None and locally_associated(fragment, other))
    return neighbors
