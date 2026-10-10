"""Portable Agent Memory export / import for Merkle-DAG fragments.

Mechanism — M1 step 4 in [docs/experiments/phase-2-mentalistic/
option-M1-merkle-fragments/README.md](../../../docs/experiments/phase-2-mentalistic/option-M1-merkle-fragments/README.md).

Serializes an append-only fragment set as a content-addressed JSON
artifact. Round-trip property: ``pam_import(pam_export(fragments))``
yields a fragment set byte-identical to the input under
``content_hash`` — the Portable Agent Memory
([arXiv:2605.11032](https://arxiv.org/html/2605.11032v1)) + Digital
Bereavement export obligation becomes a graph walk over the fragment
list, with ids provably reproducible from content.

Scope. v1 serializes the Fragment layer only. EdgeStore / FactStore /
witness state are substrate-derivable from fragments and are deferred
to a later module (they are not part of the PAM contract — PAM is
content, not retrieval index). Hash algorithm is BLAKE2b-128 as a
hashlib-built-in stand-in for BLAKE3; the field ``hash_algo`` in the
export preserves the swap point if PAM-arXiv-spec byte-identity is
ever needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from psychology_base.eval.adapters import Fragment, content_hash


PAM_FORMAT = "pam-v1"
PAM_HASH_ALGO = "blake2b-128"
# agent-memory-mcp spec uses `digest_algo`; we emit both keys and accept
# either on read so the artifact round-trips either way. Interop bridge
# per Phase 4 plan §2.3.
PAM_DIGEST_ALGO_KEY = "digest_algo"


def _fragment_to_dict(f: Fragment) -> dict:
    return {
        "id": f.id,
        "content": f.content,
        "timestamp": f.timestamp.isoformat(),
        "affect": dict(sorted(f.affect.items())),
        "tags": list(f.tags),
        "parent_ids": list(f.parent_ids),
    }


def _fragment_from_dict(d: dict) -> Fragment:
    return Fragment(
        id=d["id"],
        content=d["content"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        affect=dict(d.get("affect") or {}),
        tags=tuple(d.get("tags") or ()),
        parent_ids=tuple(d.get("parent_ids") or ()),
    )


def pam_export(fragments: Iterable[Fragment]) -> dict:
    """Serialize a fragment set as a PAM artifact.

    Each fragment is emitted with its stored id intact; callers are
    expected to use ``make_fragment`` on the write path so that ids
    are already content-addressed. Fragments produced by legacy
    construction paths (``Fragment(id="echo-0", ...)``) will round-trip
    their caller-supplied id — ``validate_content_addressing`` is where
    we check the id-equals-hash invariant.

    The artifact carries BOTH ``hash_algo`` (legacy psychology-base key)
    and ``digest_algo`` (agent-memory-mcp spec key) with the same value;
    this is the interop bridge per Phase 4 plan §2.3.
    """
    frag_list = list(fragments)
    return {
        "format": PAM_FORMAT,
        "hash_algo": PAM_HASH_ALGO,
        PAM_DIGEST_ALGO_KEY: PAM_HASH_ALGO,
        "fragments": [_fragment_to_dict(f) for f in frag_list],
    }


def pam_import(data: dict) -> list[Fragment]:
    """Rehydrate the fragment list from a PAM artifact.

    Accepts either ``hash_algo`` (legacy) or ``digest_algo`` (agent-
    memory-mcp spec) as the algorithm manifest key; if both are present
    they must agree. Raises ``ValueError`` if the format or algo header
    does not match this v1 contract.
    """
    fmt = data.get("format")
    if fmt != PAM_FORMAT:
        raise ValueError(f"unsupported PAM format: {fmt!r} (expected {PAM_FORMAT!r})")
    legacy = data.get("hash_algo")
    spec = data.get(PAM_DIGEST_ALGO_KEY)
    if legacy is None and spec is None:
        raise ValueError(
            "PAM artifact missing both 'hash_algo' and 'digest_algo' — "
            "no algorithm manifest key present"
        )
    if legacy is not None and spec is not None and legacy != spec:
        raise ValueError(
            f"PAM manifest keys disagree: hash_algo={legacy!r} vs "
            f"digest_algo={spec!r}"
        )
    algo = legacy if legacy is not None else spec
    if algo != PAM_HASH_ALGO:
        raise ValueError(
            f"unsupported hash algorithm: {algo!r} (expected {PAM_HASH_ALGO!r})"
        )
    return [_fragment_from_dict(d) for d in data.get("fragments", [])]


def validate_content_addressing(fragments: Iterable[Fragment]) -> list[str]:
    """Return ids of fragments whose stored id does NOT equal
    ``content_hash`` of their payload. Empty list == strict content-
    addressing holds (every fragment is Merkle-provable from its
    content). Non-empty list == fragments whose ids were caller-
    supplied (e.g. legacy tick-indexed ``frag-NNNN`` ids or EchoMemory
    ``echo-N`` ids); these round-trip under PAM but do not satisfy the
    strict-DAG byte-identity invariant.
    """
    offenders: list[str] = []
    for f in fragments:
        expected = content_hash(
            f.content, f.timestamp, f.affect, f.tags, f.parent_ids
        )
        if f.id != expected:
            offenders.append(f.id)
    return offenders
