"""M1 step 4 — PAM export / import round-trip.

Round-trip property: ``pam_import(pam_export(fragments))`` produces a
fragment list byte-identical to the input (same ids, same ordering,
same payload). Content-addressing validation: fragments built via
``make_fragment`` must satisfy ``validate_content_addressing`` → []
(strict DAG byte-identity).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from psychology_base.companion.pam_export import (
    PAM_FORMAT,
    PAM_HASH_ALGO,
    pam_export,
    pam_import,
    validate_content_addressing,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import (
    Fragment,
    Observation,
    make_fragment,
)


TS = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def _obs(content: str, offset_minutes: int = 0, tags=()):
    return Observation(
        timestamp=TS + timedelta(minutes=offset_minutes),
        speaker="user",
        content=content,
        tags=tags,
    )


def test_export_schema_headers():
    data = pam_export([])
    assert data["format"] == PAM_FORMAT
    assert data["hash_algo"] == PAM_HASH_ALGO
    assert data["fragments"] == []


def test_roundtrip_preserves_fragment_payload():
    original = [
        make_fragment("father was never there", TS),
        make_fragment("father was overwhelmed", TS + timedelta(days=90)),
    ]
    rehydrated = pam_import(pam_export(original))
    assert len(rehydrated) == len(original)
    for a, b in zip(original, rehydrated):
        assert a.id == b.id
        assert a.content == b.content
        assert a.timestamp == b.timestamp
        assert a.affect == b.affect
        assert a.tags == b.tags
        assert a.parent_ids == b.parent_ids


def test_roundtrip_preserves_chain_structure():
    """The DAG shape — parent_ids chaining — must survive export+import."""
    root = make_fragment("v1 framing", TS)
    rev = make_fragment(
        "v2 framing", TS + timedelta(days=30), parent_ids=(root.id,)
    )
    rehydrated = pam_import(pam_export([root, rev]))
    assert rehydrated[0].parent_ids == ()
    assert rehydrated[1].parent_ids == (root.id,)
    assert rehydrated[1].parent_ids[0] == rehydrated[0].id


def test_roundtrip_is_json_serializable():
    """The export artifact must be JSON-serializable without custom
    encoders — this is what makes the Merkle-DAG portable."""
    frags = [make_fragment("alpha", TS, affect={"valence": 0.3}, tags=("x",))]
    exported = pam_export(frags)
    blob = json.dumps(exported)
    parsed = json.loads(blob)
    rehydrated = pam_import(parsed)
    assert rehydrated[0].id == frags[0].id


def test_validate_content_addressing_passes_for_make_fragment():
    frags = [
        make_fragment("alpha", TS),
        make_fragment("beta", TS + timedelta(minutes=1)),
    ]
    assert validate_content_addressing(frags) == []


def test_validate_content_addressing_catches_legacy_ids():
    """Legacy Fragment(id='frag-0001', ...) construction leaves a non-
    content-addressed id. ``validate_content_addressing`` surfaces
    these so callers know which fragments won't satisfy strict DAG
    byte-identity (they still round-trip under PAM — the stored id is
    preserved — but a stricter DAG verifier would reject them)."""
    f = Fragment(id="frag-0001", content="legacy", timestamp=TS)
    offenders = validate_content_addressing([f])
    assert offenders == ["frag-0001"]


def test_import_rejects_wrong_format():
    with pytest.raises(ValueError, match="unsupported PAM format"):
        pam_import({"format": "pam-v99", "hash_algo": PAM_HASH_ALGO, "fragments": []})


def test_import_rejects_wrong_hash_algo():
    with pytest.raises(ValueError, match="unsupported hash algorithm"):
        pam_import({"format": PAM_FORMAT, "hash_algo": "sha256", "fragments": []})


def test_full_companion_roundtrip():
    """End-to-end: run a scenario through CompanionSystem (fresh observe
    + rerender), export the fragment set, import it back, and verify
    the fragment set is byte-identical under content-addressing."""
    sys = CompanionSystem()
    sys.observe(_obs("my father was never there", tags=("father",)))
    original_id = sys.companion.store.all()[0].id
    sys.rerender(
        referent_id=original_id,
        new_content="my father was overwhelmed and tried in his way",
        timestamp=TS + timedelta(days=90),
        tags=("father", "reinterpretation"),
    )
    sys.observe(_obs("orthogonal event", offset_minutes=5))

    original_frags = sys.companion.store.all()
    exported = pam_export(original_frags)
    rehydrated = pam_import(exported)

    assert len(rehydrated) == len(original_frags)
    for a, b in zip(original_frags, rehydrated):
        assert a.id == b.id
        assert a.content == b.content
        assert a.parent_ids == b.parent_ids
    # Strict DAG byte-identity: every fragment on the write path was
    # produced via make_fragment, so every id must equal its content hash.
    assert validate_content_addressing(rehydrated) == []
