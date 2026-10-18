"""Offline, provenance-aware attachment trajectories and bounded checkpoints.

Duration thresholds and fixed replies are software controls. Synthetic inputs
and lexical scores provide no model-quality or human-attachment validation.
PAM restores content; only fixed-transcript attachment replay is exact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator

from psychology_base.applications.emotional import attachment_state as attachment
from psychology_base.applications.emotional.companion_agent import CompanionAgent, LLMResponder, _render_context_block
from psychology_base.applications.safety.ai_identity_reminder import AIIdentityReminder
from psychology_base.applications.safety.crisis_escalation import CrisisEscalation
from psychology_base.applications.safety.extreme_emotion import ExtremeEmotionWatcher, SessionAffect
from psychology_base.applications.safety.minor_exclusion import (
    AgeAttestation, ApplicationSurface, MinorExclusionGate,
)
from psychology_base.companion.pam_export import pam_export, pam_import, validate_content_addressing
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import make_fragment
from psychology_base.eval.benchmarks.public_datasets.msc import MSCConversation, MSCSession, MSCTurn
from psychology_base.eval.benchmarks.public_datasets.msc_bond_longitudinal import run_msc_session


BASE_TIME = datetime(2026, 10, 18, tzinfo=timezone.utc)
BYTE_CAP = 1_000_000  # This experiment measures continuity, not capacity pressure.
REPO = Path(__file__).resolve().parents[4]
DEFAULT_CATALOG = REPO / "benchmarks/fixtures/attachment_longitudinal/profiles.json"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scorer_configuration() -> dict:
    return {
        "weights": {name: getattr(attachment, name) for name in (
            "_W_MIRROR", "_W_REPAIR", "_W_CONSISTENCY", "_W_RUPTURE",
            "_W_SYCOPHANCY", "_W_OVER_DISCLOSURE",
        )},
        "ema_alpha": attachment._TRUST_EMA_ALPHA,
        "scorer_source_sha256": hashlib.sha256(Path(attachment.__file__).read_bytes()).hexdigest(),
        "top_k": 6, "byte_cap": BYTE_CAP, "tokenizer": "unicode",
        "consolidation_enabled": False, "mode": "scripted",
    }


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class FixtureAffect(Contract):
    """Explicit test input, never inferred from a scripted dialogue."""

    valence: float = Field(ge=-1, le=1, strict=True)
    arousal: float = Field(ge=-1, le=1, strict=True)


class ContinuityAnchor(Contract):
    session_id: StrictStr
    user_turn: StrictInt = Field(ge=1)


class TrajectorySession(Contract):
    session_id: StrictStr
    user_id: StrictStr
    index: StrictInt = Field(ge=1)
    block_id: StrictStr = Field(min_length=1)
    starts_at: datetime
    ends_at: datetime
    provenance: Literal["msc_opener_scripted_dialogue", "synthetic_continuation"]
    source_conversation_id: StrictStr | None = None
    source_session_id: StrictInt | None = Field(default=None, ge=1)
    source_gap_days: float | None = Field(default=None, ge=0)
    source_elapsed_days: float | None = Field(default=None, ge=0)
    seeker_turns: tuple[StrictStr, ...] = Field(min_length=1)
    companion_turns: tuple[StrictStr, ...]
    seeker_provenance: tuple[Literal["msc_source_opener", "scripted"], ...]
    companion_provenance: Literal["scripted"] = "scripted"
    affect: FixtureAffect | None = None

    @field_validator("starts_at", "ends_at")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def coherent(self) -> TrajectorySession:
        if self.ends_at < self.starts_at + timedelta(seconds=30 * max(1, len(self.companion_turns))):
            raise ValueError("session duration cannot contain its scheduled turns")
        if self.companion_turns and len(self.companion_turns) != len(self.seeker_turns):
            raise ValueError("scripted seeker/companion turn counts must match")
        if len(self.seeker_provenance) != len(self.seeker_turns):
            raise ValueError("every seeker turn needs provenance")
        if any(not text.strip() for text in self.seeker_turns + self.companion_turns):
            raise ValueError("turn text cannot be empty; use zero companion turns for an empty session")
        source_values = (self.source_conversation_id, self.source_session_id,
                         self.source_gap_days, self.source_elapsed_days)
        if self.provenance == "msc_opener_scripted_dialogue":
            if any(value is None for value in source_values):
                raise ValueError("MSC sessions need source identity and original timing")
            if self.seeker_provenance != ("msc_source_opener",) + ("scripted",) * (len(self.seeker_turns) - 1):
                raise ValueError("only the first seeker turn may be source-derived")
        elif any(value is not None for value in source_values) or "msc_source_opener" in self.seeker_provenance:
            raise ValueError("synthetic sessions cannot claim source provenance")
        return self


class TrajectoryManifest(Contract):
    schema_version: Literal["attachment-longitudinal-v2"] = "attachment-longitudinal-v2"
    trajectory_id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    user_id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    language: Literal["en", "zh"]
    fixture_version: StrictStr = Field(min_length=1)
    fixture_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_file_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_conversation_ids: tuple[StrictStr, ...] = ()
    persona: tuple[StrictStr, ...]
    sentinel: StrictStr = Field(pattern=r"^[a-z0-9]+$")
    continuity_anchor: ContinuityAnchor
    seed: StrictInt = 0
    age_years: StrictInt = Field(default=30, ge=0)
    mode: Literal["scripted"] = "scripted"
    sessions: tuple[TrajectorySession, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_chain(self) -> TrajectoryManifest:
        seen_sources: set[tuple[str, int]] = set()
        source_ids: set[str] = set()
        last_source_index = 0
        source_base = None
        source_elapsed = 0.0
        synthetic_seen = False
        finished_blocks: set[str] = set()
        for pos, session in enumerate(self.sessions, 1):
            if session.index != pos or session.session_id != f"{self.trajectory_id}-s{pos:04d}":
                raise ValueError("session IDs and indices must form a unique ordered trajectory")
            if session.user_id != self.user_id:
                raise ValueError("mixed-user trajectory")
            if pos > 1:
                previous = self.sessions[pos - 2]
                if session.starts_at <= previous.ends_at:
                    raise ValueError("sessions must be increasing and non-overlapping")
                if previous.block_id != session.block_id:
                    finished_blocks.add(previous.block_id)
                    if session.block_id in finished_blocks:
                        raise ValueError("blocks must be contiguous")
            if session.provenance == "msc_opener_scripted_dialogue":
                if synthetic_seen:
                    raise ValueError("source sessions must precede synthetic continuations")
                key = (session.source_conversation_id, session.source_session_id)
                if key in seen_sources or session.source_session_id <= last_source_index:
                    raise ValueError("duplicate or unordered source session")
                seen_sources.add(key)
                source_ids.add(session.source_conversation_id)
                last_source_index = session.source_session_id
                source_elapsed += session.source_gap_days
                if not math.isclose(source_elapsed, session.source_elapsed_days, abs_tol=1e-9):
                    raise ValueError("source cumulative timing differs from original gaps")
                if source_base is None:
                    source_base = session.starts_at - timedelta(days=source_elapsed)
                if session.starts_at != source_base + timedelta(days=source_elapsed):
                    raise ValueError("source session timing was relabeled")
            else:
                synthetic_seen = True
        if len(source_ids) > 1:
            raise ValueError("unrelated source identities cannot be merged")
        if source_ids != set(self.source_conversation_ids) or len(self.source_conversation_ids) != len(source_ids):
            raise ValueError("source identity manifest mismatch")
        if bool(source_ids) != bool(self.source_file_sha256):
            raise ValueError("source-backed manifests require a file hash")
        anchor = next((s for s in self.sessions if s.session_id == self.continuity_anchor.session_id), None)
        if anchor is None or self.continuity_anchor.user_turn > len(anchor.companion_turns):
            raise ValueError("continuity anchor must identify an observed user turn")
        return self

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))

    @property
    def anchor_fragment(self):
        spec = next(s for s in self.sessions if s.session_id == self.continuity_anchor.session_id)
        turn = self.continuity_anchor.user_turn
        return _turn_fragments(self, spec, spec.seeker_turns[turn - 1], spec.companion_turns[turn - 1], turn)[0]


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("scenarios"), list) or not isinstance(value.get("languages"), dict):
        raise ValueError("catalog must contain a scenario list and language profiles")
    names = [r.get("name") for r in value["scenarios"] if isinstance(r, dict)]
    if len(names) != len(value["scenarios"]) or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError("catalog scenario names must be unique strings")
    return value


def build_trajectory(
    name: str, *, catalog: dict | None = None,
    source: MSCConversation | None = None, source_file_sha256: str | None = None,
) -> TrajectoryManifest:
    """Build one identity, never splice arbitrary MSC people into one chain."""
    catalog = load_catalog() if catalog is None else catalog
    recipe = next((r for r in catalog["scenarios"] if r["name"] == name), None)
    if recipe is None:
        raise ValueError(f"unknown scenario: {name}")
    lang = recipe["language"]
    profile = catalog["languages"][lang]
    phases = recipe["phases"]
    if not phases or any(type(p.get("count")) is not int or p["count"] < 1
                         or p.get("profile") not in profile["replies"] for p in phases):
        raise ValueError("scenario phases need positive integer counts and known reply profiles")
    trajectory_id = name if source is None else f"{name}-msc-{digest(source.conversation_id)[:12]}"
    user_id = f"user-{trajectory_id}"
    sentinel = "continuity" + digest(user_id)[:16]
    identity = profile["identity"].format(sentinel=sentinel)
    followup = profile["followup"].format(sentinel=sentinel)
    if source is not None:
        identity = profile["source_identity"].format(
            sentinel=sentinel, persona=" ".join(source.personas[0]) or "We are continuing the same conversation.",
        )
        followup = profile["source_followup"].format(sentinel=sentinel)
    labels = [phase["profile"] for phase in recipe["phases"] for _ in range(phase["count"])]
    count = len(labels)
    for key in ("empty_at", "handoff_at"):
        at = recipe.get(key)
        if at is not None and (type(at) is not int or not 1 <= at <= count):
            raise ValueError(f"{key} must identify a scheduled session")
    if recipe.get("empty_at") is not None and recipe.get("empty_at") == recipe.get("handoff_at"):
        raise ValueError("empty and handoff controls cannot share a session")
    source_sessions = tuple(sorted(source.sessions, key=lambda s: s.session_id)) if source else ()
    if source is not None and (not source_sessions or len(source_sessions) >= count):
        raise ValueError("source needs at least one session and room for continuations")
    offsets: list[float] = []
    cumulative = 0.0
    for session in source_sessions:
        cumulative += session.time_interval_days
        offsets.append(cumulative)
    if offsets:
        end = max(float(recipe["duration_days"]) + offsets[0], offsets[-1] + 1)
        offsets += [offsets[-1] + (end - offsets[-1]) * i / (count - len(source_sessions))
                    for i in range(1, count - len(source_sessions) + 1)]
    else:
        offsets = [float(recipe["duration_days"]) * i / max(1, count - 1) for i in range(count)]
    sessions = []
    for i, (offset, label) in enumerate(zip(offsets, labels), 1):
        original = source_sessions[i - 1] if i <= len(source_sessions) else None
        opener = next((t.utterance for t in original.turns if t.speaker_idx == 0), None) if original else None
        if original and not opener:
            raise ValueError("source session has no source-speaker opening")
        seeker_turns = (opener or followup, identity if i == 1 else followup)
        if i == recipe.get("handoff_at"):
            if original:
                raise ValueError("a source opener cannot be replaced by a synthetic crisis probe")
            seeker_turns = (profile["crisis_probe"], followup)
        replies = (profile["replies"][label],) * 2
        if i == recipe.get("empty_at"):
            replies = ()
        start = BASE_TIME + timedelta(days=offset)
        sessions.append(TrajectorySession(
            session_id=f"{trajectory_id}-s{i:04d}", user_id=user_id, index=i,
            block_id=f"{trajectory_id}-block{(i - 1) // 4 + 1}",
            starts_at=start, ends_at=start + timedelta(minutes=2),
            provenance="msc_opener_scripted_dialogue" if original else "synthetic_continuation",
            source_conversation_id=source.conversation_id if original else None,
            source_session_id=original.session_id if original else None,
            source_gap_days=original.time_interval_days if original else None,
            source_elapsed_days=offset if original else None,
            seeker_turns=seeker_turns, companion_turns=replies,
            seeker_provenance=("msc_source_opener" if original else "scripted", "scripted"),
            affect=recipe.get("affect"),
        ))
    return TrajectoryManifest(
        trajectory_id=trajectory_id, user_id=user_id, language=lang,
        fixture_version=catalog["fixture_version"], fixture_sha256=digest(catalog),
        source_file_sha256=source_file_sha256,
        source_conversation_ids=(source.conversation_id,) if source else (),
        persona=source.personas[0] if source else tuple(profile["persona"]),
        sentinel=sentinel, sessions=tuple(sessions),
        continuity_anchor=ContinuityAnchor(session_id=sessions[0].session_id, user_turn=2),
    )


class _AuditedSystem(CompanionSystem):
    """Observe real recall output without altering selection or its state."""

    def recall(self, *args, **kwargs):
        recalled = super().recall(*args, **kwargs)
        self.last_context_fragments = recalled.fragments_used[:6]
        return recalled


def _new_system(manifest: TrajectoryManifest) -> CompanionSystem:
    return _AuditedSystem(top_k=6, seed=manifest.seed, byte_cap=BYTE_CAP, cjk_tokenizer="unicode")


class _FixtureAgent(CompanionAgent):
    """Pass an explicitly synthetic age attestation through the real gate."""

    fixture_age: int = 30

    def respond(self, user_utterance: str, **kwargs):
        stamp = kwargs["timestamp"].timestamp()
        return super().respond(user_utterance, attestation=AgeAttestation(
            age_years=self.fixture_age, attested_at=stamp, attestation_source="synthetic_fixture",
        ), **kwargs)


def _score(state: attachment.AttachmentState, spec: TrajectorySession, seekers, companions):
    return attachment.score_session_attachment(
        session_id=spec.session_id, seeker_turns=seekers, companion_turns=companions, prior_state=state,
    )


def _record(spec: TrajectorySession, status: str, seekers=(), companions=(), contexts=(),
            context_fragments=(), **extra) -> dict:
    return {
        "session_id": spec.session_id, "index": spec.index, "block_id": spec.block_id,
        "user_id": spec.user_id, "starts_at": spec.starts_at.isoformat(), "ends_at": spec.ends_at.isoformat(),
        "provenance": spec.provenance, "source_conversation_id": spec.source_conversation_id,
        "source_session_id": spec.source_session_id, "source_gap_days": spec.source_gap_days,
        "source_elapsed_days": spec.source_elapsed_days, "seeker_provenance": list(spec.seeker_provenance),
        "companion_provenance": spec.companion_provenance,
        "affect": spec.affect.model_dump() if spec.affect else None,
        "affect_provenance": "explicit_fixture_input" if spec.affect else "not_measured",
        "status": status, "seeker_turns": list(seekers), "companion_turns": list(companions),
        "contexts": list(contexts), "context_fragments": list(context_fragments), **extra,
    }


def _replay(manifest: TrajectoryManifest, ledger: list[dict]) -> attachment.AttachmentState:
    """Validate the completed prefix and reconstruct its attachment state."""
    if not isinstance(ledger, list) or len(ledger) > len(manifest.sessions):
        raise ValueError("invalid checkpoint ledger")
    state = attachment.AttachmentState(user_id=manifest.user_id)
    for spec, record in zip(manifest.sessions, ledger):
        if not isinstance(record, dict) or record.get("status") not in ("completed", "empty"):
            raise ValueError("checkpoint contains an unaccepted session")
        empty = not spec.companion_turns
        if record.get("status") != ("empty" if empty else "completed"):
            raise ValueError("checkpoint completion mismatch")
        contexts = record.get("contexts")
        if not isinstance(contexts, list) or any(not isinstance(c, str) for c in contexts):
            raise ValueError("invalid checkpoint contexts")
        if len(contexts) != len(spec.companion_turns):
            raise ValueError("checkpoint turn count mismatch")
        context_fragments = record.get("context_fragments")
        _validate_contexts(manifest, spec, ledger[:spec.index - 1], contexts, context_fragments)
        expected = _record(spec, "empty" if empty else "completed",
                           () if empty else spec.seeker_turns, spec.companion_turns, contexts, context_fragments)
        if not empty:
            scored = _score(state, spec, spec.seeker_turns, spec.companion_turns)
            state = scored.new_state
            expected.update(scoring={k: v for k, v in asdict(scored).items() if k != "new_state"},
                            attachment_state=asdict(state))
            # JSON serialization turns the history tuple into a list.
        if canonical_json(record) != canonical_json(expected):
            raise ValueError("checkpoint ledger differs from its manifest or scorer")
    return state


def _turn_fragments(manifest, spec, user, reply, turn):
    stamp = spec.starts_at + timedelta(seconds=30 * turn)
    return [make_fragment(text, stamp, tags=(
        f"user:{manifest.user_id}", f"session:{spec.session_id}", f"speaker:{speaker}",
    )) for speaker, text in (("user", user), ("companion", reply))]


def _expected_fragments(manifest: TrajectoryManifest, ledger: list[dict]) -> list:
    fragments = []
    for spec, record in zip(manifest.sessions, ledger):
        if record["status"] != "completed":
            continue
        for turn, (user, reply) in enumerate(zip(record["seeker_turns"], record["companion_turns"]), 1):
            fragments.extend(_turn_fragments(manifest, spec, user, reply, turn))
    return fragments


def _validate_contexts(manifest, spec, prefix, contexts, snapshots):
    if not isinstance(snapshots, list) or len(snapshots) != len(contexts):
        raise ValueError("checkpoint context audit count mismatch")
    allowed = {f.id: f for f in _expected_fragments(manifest, prefix)}
    for turn, (context, snapshot) in enumerate(zip(contexts, snapshots), 1):
        if not isinstance(snapshot, list) or len(snapshot) > 6:
            raise ValueError("invalid context fragment audit")
        fragments = pam_import({**pam_export(()), "fragments": snapshot})
        if (len({f.id for f in fragments}) != len(fragments)
                or validate_content_addressing(fragments)
                or any(f.id not in allowed or f != allowed[f.id] for f in fragments)
                or pam_export(fragments)["fragments"] != snapshot
                or _render_context_block(fragments) != context):
            raise ValueError("context audit contains unknown, future, or mislabeled memory")
        if spec.companion_turns:
            for fragment in _turn_fragments(manifest, spec, spec.seeker_turns[turn - 1],
                                             spec.companion_turns[turn - 1], turn):
                allowed[fragment.id] = fragment


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def save_checkpoint(path: Path, manifest: TrajectoryManifest, state, ledger: list[dict], system) -> None:
    replayed = _replay(manifest, ledger)
    if replayed != state:
        raise ValueError("checkpoint attachment state does not match accepted sessions")
    artifact = pam_export(system.companion.store.all())
    if digest(artifact) != digest(pam_export(_expected_fragments(manifest, ledger))):
        raise ValueError("checkpoint memory differs from accepted transcript content")
    memory_hash = digest(artifact)
    memory_name = f"memory-{memory_hash}.json"
    atomic_write_json(path.with_name(memory_name), artifact)
    payload = {
        "schema_version": "attachment-checkpoint-v2", "user_id": manifest.user_id,
        "manifest_sha256": manifest.sha256, "source_file_sha256": manifest.source_file_sha256,
        "configuration": scorer_configuration(), "state": asdict(state), "ledger": ledger,
        "next_index": len(ledger) + 1,
        "simulated_clock": ledger[-1]["ends_at"] if ledger else None,
        "memory_file": memory_name, "memory_sha256": memory_hash,
    }
    atomic_write_json(path, {"payload": payload, "sha256": digest(payload)})


def load_checkpoint(path: Path, manifest: TrajectoryManifest):
    """Validate before ingesting; restore content, not a complete runtime."""
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "sha256"}:
        raise ValueError("invalid checkpoint envelope")
    payload = envelope["payload"]
    if digest(payload) != envelope["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    expected_keys = {"schema_version", "user_id", "manifest_sha256", "source_file_sha256", "configuration",
                     "state", "ledger", "next_index", "simulated_clock", "memory_file", "memory_sha256"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("invalid checkpoint fields")
    if (payload["schema_version"] != "attachment-checkpoint-v2" or payload["user_id"] != manifest.user_id
            or payload["manifest_sha256"] != manifest.sha256
            or payload["source_file_sha256"] != manifest.source_file_sha256
            or payload["configuration"] != scorer_configuration()):
        raise ValueError("checkpoint identity, manifest, or configuration mismatch")
    state = _replay(manifest, payload["ledger"])
    state_data = payload["state"]
    if not isinstance(state_data, dict) or set(state_data) != {f.name for f in fields(state)}:
        raise ValueError("invalid attachment state fields")
    # Reject boolean counts and non-numeric/coerced state rather than relying
    # on the attachment dataclass's deliberately minimal validation.
    for key, value in state_data.items():
        if key.startswith("cumulative_") or key == "sessions_completed":
            if type(value) is not int or value < 0:
                raise ValueError("invalid attachment count")
    if canonical_json(state_data) != canonical_json(asdict(state)):
        raise ValueError("checkpoint attachment state mismatch")
    ledger = payload["ledger"]
    if type(payload["next_index"]) is not int or payload["next_index"] != len(ledger) + 1:
        raise ValueError("checkpoint next index mismatch")
    if payload["simulated_clock"] != (ledger[-1]["ends_at"] if ledger else None):
        raise ValueError("checkpoint simulated clock mismatch")
    name = payload["memory_file"]
    if not isinstance(name, str) or not re.fullmatch(r"memory-[0-9a-f]{64}\.json", name):
        raise ValueError("invalid checkpoint memory filename")
    artifact = json.loads(path.with_name(name).read_text(encoding="utf-8"))
    if digest(artifact) != payload["memory_sha256"] or name != f"memory-{digest(artifact)}.json":
        raise ValueError("checkpoint memory checksum mismatch")
    if digest(artifact) != digest(pam_export(_expected_fragments(manifest, ledger))):
        raise ValueError("checkpoint memory does not match transcript ledger")
    restored = pam_import(artifact)
    if validate_content_addressing(restored):
        raise ValueError("checkpoint contains corrupt memory content addresses")
    system = _new_system(manifest)
    for fragment in restored:
        system._ingest_fragment(fragment)
    return state, ledger, system


def memory_round_trip_checks(system: CompanionSystem, cue: str, user_id: str) -> dict:
    original = system.companion.store.all()
    artifact = pam_export(original)
    imported = pam_import(artifact)
    restored = CompanionSystem(top_k=system.top_k, seed=system.seed,
                               byte_cap=system.companion.store.byte_cap, cjk_tokenizer=system.cjk_tokenizer)
    for fragment in imported:
        restored._ingest_fragment(fragment)
    ids = {f.id for f in imported}
    recalled = restored.recall(cue).fragments_used if imported else ()
    return {
        "fragment_count": len(imported), "fragment_set_preserved": imported == original,
        "content_addressing_valid": not validate_content_addressing(imported),
        "parent_chains_preserved": all(p in ids for f in imported for p in f.parent_ids),
        "user_tags_isolated": all(
            f"user:{user_id}" in f.tags and all(t == f"user:{user_id}" for t in f.tags if t.startswith("user:"))
            for f in imported
        ),
        "sentinel_recall_after_restore": any(cue in f.content for f in recalled),
        "runtime_equivalence_claimed": False,
    }


@dataclass
class TrajectoryResult:
    manifest: TrajectoryManifest
    state: attachment.AttachmentState
    ledger: list[dict]
    system: CompanionSystem
    stop_reason: str | None
    incidents: list[dict]
    review_flags: list[dict]
    response_source: str = "fixed_script"

    def report(self) -> dict:
        completed = [r for r in self.ledger if r["status"] == "completed"]
        elapsed = ((datetime.fromisoformat(completed[-1]["starts_at"])
                    - datetime.fromisoformat(completed[0]["starts_at"])).total_seconds() / 86400
                   if len(completed) >= 2 else 0.0)
        complete = len(completed) == len(self.manifest.sessions) and self.stop_reason is None
        transitions = []
        for spec in self.manifest.sessions[1:]:
            previous = self.manifest.sessions[spec.index - 2]
            if previous.block_id == spec.block_id:
                continue
            record = next((r for r in self.ledger if r["session_id"] == spec.session_id), None)
            anchor = self.manifest.anchor_fragment
            observed = bool(record and record["status"] == "completed" and any(
                snapshot == pam_export((anchor,))["fragments"][0]
                for fragments in record["context_fragments"] for snapshot in fragments
            ))
            transitions.append({"session_id": spec.session_id, "attempted": record is not None,
                                "original_user_fact_in_context": observed,
                                "anchor_fragment_id": anchor.id})
        return {
            "schema_version": "attachment-trajectory-report-v2",
            "trajectory_id": self.manifest.trajectory_id, "user_id": self.manifest.user_id,
            "manifest_sha256": self.manifest.sha256, "configuration": scorer_configuration(),
            "evidence_class": "scripted_software_control" if self.response_source == "fixed_script" else "unverified_injected_callable",
            "response_source": self.response_source,
            "response_uses_recalled_context": False if self.response_source == "fixed_script" else None,
            "model_response_quality_measured": False, "attachment_validity_established": False,
            "simulated_time_only": True, "memory_capacity_pressure_exercised": False,
            "planned_sessions": len(self.manifest.sessions), "attempted_sessions": len(self.ledger),
            "completed_sessions": len(completed),
            "excluded_sessions": sum(r["status"] in ("empty", "handoff") for r in self.ledger),
            "failed_sessions": sum(r["status"] == "error" for r in self.ledger),
            "unattempted_sessions": len(self.manifest.sessions) - len(self.ledger),
            "elapsed_simulated_days": elapsed, "trajectory_complete": complete,
            "stop_reason": self.stop_reason, "legacy_gate_pass": self.state.meets_four_week_gate,
            "duration_qualified_proxy_pass": bool(complete and elapsed >= 28 and self.state.meets_four_week_gate),
            "neutral_control_passes_legacy_gate": bool(self.state.meets_four_week_gate and all(
                self.state.__dict__[f"cumulative_{dim}_turns"] == 0
                for dim in ("mirror", "rupture", "repair", "sycophancy", "over_disclosure", "consistency")
            )),
            "attachment_state": asdict(self.state), "block_transitions": transitions,
            "continuity_probe_pass": bool(complete and transitions and all(
                t["original_user_fact_in_context"] for t in transitions
            )),
            "handoff_incidents": self.incidents, "extreme_emotion_review_flags": self.review_flags,
            "per_session": self.ledger,
            "memory_checks": memory_round_trip_checks(self.system, self.manifest.sentinel, self.manifest.user_id),
            "llm_wai_sr": {"measured": False, "correlation": None,
                           "reason": "offline scripted run; persona-flat LLM ratings are not a calibration target"},
            "limitations": ["synthetic targets are not clinical ground truth",
                            "neutral sessions at midpoint can pass both proxy thresholds",
                            "single-turn over-disclosure residual remains algorithm debt",
                            "PAM is a content export, not a complete runtime snapshot",
                            "fixed replies ignore memory and do not measure companion response quality",
                            "calendar gaps do not exercise elapsed-time aging",
                            "the one-megabyte capacity is an explicit continuity-test setting"],
        }


def run_trajectory(
    manifest: TrajectoryManifest, *, checkpoint_path: Path | None = None,
    resume: bool = False, stop_after: int | None = None,
    responder_factory: Callable[[TrajectorySession], LLMResponder] | None = None,
    checkpoint_every: int = 8,
) -> TrajectoryResult:
    """Replay one identity's scripts; external callables have unverified provenance."""
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after must be a positive completed-session count")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if responder_factory is not None and checkpoint_path is not None:
        raise ValueError("checkpoints require the manifest's fixed scripted responses")
    if resume:
        if checkpoint_path is None:
            raise ValueError("resume needs a checkpoint path")
        state, ledger, system = load_checkpoint(checkpoint_path, manifest)
    else:
        if checkpoint_path is not None and checkpoint_path.exists():
            raise ValueError("checkpoint already exists; use resume or a new output directory")
        state, ledger, system = attachment.AttachmentState(user_id=manifest.user_id), [], _new_system(manifest)
    if stop_after is not None and stop_after < state.sessions_completed:
        raise ValueError("stop_after precedes the restored checkpoint")
    incidents: list[dict] = []
    reviews: list[dict] = []
    def new_watcher():
        return ExtremeEmotionWatcher(review_sink=lambda flag: reviews.append(asdict(flag)), user_id=manifest.user_id)

    watcher = new_watcher()
    for spec, record in zip(manifest.sessions, ledger):
        if record["status"] == "empty" or spec.affect is None:
            watcher = new_watcher()  # Unknown affect breaks an observed consecutive streak.
        else:
            watcher.record(SessionAffect(spec.session_id, spec.ends_at.timestamp(),
                                         spec.affect.valence, spec.affect.arousal))
    result = TrajectoryResult(manifest, state, ledger, system, None, incidents, reviews,
                              "injected_callable" if responder_factory else "fixed_script")
    for spec in manifest.sessions[len(ledger):]:
        if stop_after is not None and result.state.sessions_completed >= stop_after:
            result.stop_reason = "paused_at_checkpoint"
            break
        if not spec.companion_turns:
            result.ledger.append(_record(spec, "empty"))
            watcher = new_watcher()
            continue
        contexts: list[str] = []
        context_fragments: list[list[dict]] = []
        replies = iter(spec.companion_turns)
        injected = None

        def responder(prompt, context, utterance):
            contexts.append(context)
            context_fragments.append(pam_export(system.last_context_fragments)["fragments"])
            return injected(prompt, context, utterance) if injected else next(replies)

        agent = _FixtureAgent(
            user_id=manifest.user_id, session_id=spec.session_id, companion_system=system,
            llm_responder=responder, ai_identity_reminder=AIIdentityReminder(),
            minor_exclusion_gate=MinorExclusionGate(surface=ApplicationSurface.EMOTIONAL_COMPANION),
            crisis_escalation=CrisisEscalation(handoff_sink=lambda incident: incidents.append(asdict(incident))),
            extreme_emotion_watcher=watcher,
        )
        agent.fixture_age = manifest.age_years
        seeker_replies = iter(spec.seeker_turns[1:])
        prior_state = result.state
        prior_review_count = len(reviews)
        try:
            injected = responder_factory(spec) if responder_factory else None
            transcript = run_msc_session(
                MSCSession(spec.index, 0, (MSCTurn(0, spec.seeker_turns[0]),)), agent,
                lambda _: next(seeker_replies, ""), max_turns=len(spec.companion_turns),
                starting_timestamp=spec.starts_at,
            )
            seekers = [t.utterance for t in transcript.turns if t.speaker == "seeker"]
            companions = [t.utterance for t in transcript.turns if t.speaker == "companion"]
            if transcript.handoff_triggered:
                result.ledger.append(_record(spec, "handoff", seekers, companions, contexts, context_fragments))
                result.stop_reason = "handoff"
                break
            if (len(companions) != len(spec.companion_turns) or len(seekers) != len(spec.seeker_turns)
                    or any(not isinstance(text, str) or not text.strip() for text in companions)):
                raise ValueError("responder produced an empty response")
            scored = _score(result.state, spec, seekers, companions)
            if spec.affect is not None:
                agent.record_session_affect(spec.affect.valence, spec.affect.arousal, timestamp=spec.ends_at)
            else:
                watcher = new_watcher()
            agent.mark_session_boundary()
            result.state = scored.new_state
            result.ledger.append(_record(
                spec, "completed", seekers, companions, contexts, context_fragments,
                scoring={k: v for k, v in asdict(scored).items() if k != "new_state"},
                attachment_state=asdict(result.state),
            ))
            if checkpoint_path is not None and (
                result.state.sessions_completed % checkpoint_every == 0
                or result.state.sessions_completed == stop_after or spec.index == len(manifest.sessions)
            ):
                save_checkpoint(checkpoint_path, manifest, result.state, result.ledger, system)
        except Exception as exc:
            # A failed session is counted and stops the chain. The last atomic
            # checkpoint still denotes the preceding accepted boundary.
            if result.ledger and result.ledger[-1]["session_id"] == spec.session_id:
                result.ledger.pop()
            result.state = prior_state
            del reviews[prior_review_count:]
            result.ledger.append(_record(spec, "error", contexts=contexts, context_fragments=context_fragments,
                                         error=f"{type(exc).__name__}: {exc}"))
            result.stop_reason = "execution_error"
            break
    return result


def isolation_checks(results: list[TrajectoryResult]) -> dict:
    if len({r.manifest.user_id for r in results}) != len(results):
        raise ValueError("cohort identities must be unique")
    if len({r.manifest.sentinel for r in results}) != len(results):
        raise ValueError("cohort sentinels must be unique")
    leaks = []
    for result in results:
        for fragment in result.system.companion.store.all():
            if f"user:{result.manifest.user_id}" not in fragment.tags or any(
                tag.startswith("user:") and tag != f"user:{result.manifest.user_id}" for tag in fragment.tags
            ):
                leaks.append({"user_id": result.manifest.user_id, "fragment_id": fragment.id,
                              "reason": "foreign or missing ownership tag"})
        content = canonical_json(pam_export(result.system.companion.store.all())) + canonical_json(result.ledger)
        for other in results:
            if other is not result and other.manifest.sentinel in content:
                leaks.append({"user_id": result.manifest.user_id, "foreign_user_id": other.manifest.user_id})
    return {"users_checked": len(results), "passed": not leaks, "leaks": leaks}
