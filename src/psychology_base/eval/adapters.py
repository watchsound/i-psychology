from __future__ import annotations

import hashlib
import json
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
    parent_ids: tuple[str, ...] = ()


def content_hash(
    content: str,
    timestamp: datetime,
    affect: Optional[dict] = None,
    tags: tuple[str, ...] = (),
    parent_ids: tuple[str, ...] = (),
) -> str:
    """Deterministic content-addressed id for a Fragment.

    Covers every field that distinguishes a rendering of a referent:
    content, time, affect, tags, and chain parents. BLAKE2b-128 is a
    stand-in for BLAKE3 (hashlib built-in, zero new deps); the hash
    algorithm is a swap point if PAM round-trip ever needs byte-identity
    with the arXiv:2605.11032 spec.
    """
    payload = json.dumps(
        {
            "content": content,
            "timestamp": timestamp.isoformat(),
            "affect": dict(sorted((affect or {}).items())),
            "tags": list(tags),
            "parent_ids": list(parent_ids),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def make_fragment(
    content: str,
    timestamp: datetime,
    affect: Optional[dict] = None,
    tags: tuple[str, ...] = (),
    parent_ids: tuple[str, ...] = (),
) -> Fragment:
    """Build a Fragment whose id is its content-addressed hash.

    Preferred constructor on the write path. Direct ``Fragment(id=...)``
    construction with caller-supplied ids remains supported for legacy
    tests and for migration of pre-existing fragments.
    """
    aff = dict(affect) if affect else {}
    fid = content_hash(content, timestamp, aff, tags, parent_ids)
    return Fragment(
        id=fid,
        content=content,
        timestamp=timestamp,
        affect=aff,
        tags=tags,
        parent_ids=parent_ids,
    )


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    speaker: str
    content: str
    affect: dict = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    subject_id: Optional[str] = None
    entity_id: Optional[str] = None


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
