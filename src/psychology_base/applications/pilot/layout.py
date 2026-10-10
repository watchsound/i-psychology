"""Pilot directory layout + per-user manifest.

Phase 4b (per `docs/phases/phase-4b-signoff.md` §9.5 / §9.7) runs a
local-CLI pilot with ≤ 20 users across two apps (learning_tutor,
emotional_companion), each on one of two aging policies (S1 = Ebbinghaus
decay, S2 = capacity-only). This module encapsulates the on-disk
convention::

    <pilot_root>/users/<user_id>/
        user.json                     # UserManifest (enrolment metadata)
        pam/
            latest.pam-v1.json        # auto-saved on each session end
            session-NNNN.pam-v1.json  # per-session archive
        sessions/
            session-NNNN.jsonl        # chat log (user-visible)

``pilot_root`` defaults to ``<repo>/pilot-data/<app>/`` and is
gitignored. Each session index is zero-padded to four digits so a
pilot's directory listing sorts chronologically.

**Why a manifest file and not a database.** The pilot is ≤ 20 users
long-horizon. A SQLite row costs more lines of migration than a
dataclass round-trip. If/when the pilot survives into a product phase,
replace this module without touching the agent surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from psychology_base.companion.aging import (
    AgingPolicy,
    CapacityOnlyPolicy,
    EbbinghausPolicy,
)


class PilotApp(str, Enum):
    """Pilot application surface."""

    LEARNING_TUTOR = "learning_tutor"
    EMOTIONAL_COMPANION = "emotional_companion"


POLICY_IDS = ("S1", "S2")
"""The two aging policies the Phase 4b A/B assigns to pilot users."""


def build_aging_policy(policy_id: str) -> AgingPolicy:
    """Instantiate the aging policy named by ``policy_id``.

    ``S1`` → :class:`EbbinghausPolicy` (half-life retention + explicit
    recall boost + consolidate hook).
    ``S2`` → :class:`CapacityOnlyPolicy` (pure selection pressure).

    Both policies default to the Phase 3 close-out tuning — the pilot
    does NOT override hyper-parameters per user, so the A/B comparison
    isolates the policy choice, not the knob.
    """
    pid = policy_id.strip().upper()
    if pid == "S1":
        return EbbinghausPolicy()
    if pid == "S2":
        return CapacityOnlyPolicy()
    raise ValueError(f"Unknown policy_id {policy_id!r}; expected one of {POLICY_IDS}")


@dataclass(frozen=True)
class UserManifest:
    """Enrolment metadata written at user creation time.

    Field notes:

    - ``user_id`` is a pseudonymous local handle (not a real name). The
      consent artefact (§9.2c) names the mapping and lives in a user-
      controlled location outside this directory.
    - ``age_bracket`` is one of ``"18-25"`` / ``"26-45"`` / ``"46-65"``
      / ``"65+"`` — buckets wide enough to be de-identifying while still
      letting the extreme-emotion watcher age-gate.
    - ``guardian_consent_artefact_id`` is non-null only when the user is
      in a bracket requiring guardian consent per 《办法》; the enrolment
      wizard currently rejects enrolment in that case (see §9.8).
    - ``region`` is ISO-3166 (``CN``/``US``/``UK``/…) and selects the
      regional-compliance overlay (``compliance_floor.get_builtin_policy``).

    Backward-compatible evolution: new fields must be ``Optional`` with
    a default, so manifests written by older code can still load.
    """

    user_id: str
    app: PilotApp
    policy_id: str
    enrolled_at: str
    age_bracket: str
    region: str
    language: str = "en"
    guardian_consent_artefact_id: Optional[str] = None
    pilot_root: Optional[str] = field(default=None, repr=False)
    """Resolved absolute path to the pilot root as of enrolment. Written
    for forensic clarity — if the pilot directory is later moved, the
    manifest preserves where it was originally rooted. The layout API
    does not consume this field."""

    def to_json(self) -> dict:
        return {
            "user_id": self.user_id,
            "app": self.app.value,
            "policy_id": self.policy_id,
            "enrolled_at": self.enrolled_at,
            "age_bracket": self.age_bracket,
            "region": self.region,
            "language": self.language,
            "guardian_consent_artefact_id": self.guardian_consent_artefact_id,
            "pilot_root": self.pilot_root,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "UserManifest":
        return cls(
            user_id=obj["user_id"],
            app=PilotApp(obj["app"]),
            policy_id=obj["policy_id"],
            enrolled_at=obj["enrolled_at"],
            age_bracket=obj["age_bracket"],
            region=obj["region"],
            language=obj.get("language", "en"),
            guardian_consent_artefact_id=obj.get("guardian_consent_artefact_id"),
            pilot_root=obj.get("pilot_root"),
        )


@dataclass(frozen=True)
class PilotLayout:
    """Root + app together address a per-app pilot directory tree."""

    pilot_root: Path
    app: PilotApp

    @property
    def users_dir(self) -> Path:
        return self.pilot_root / self.app.value / "users"

    def user_dir(self, user_id: str) -> Path:
        return self.users_dir / user_id

    def manifest_path(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "user.json"

    def pam_dir(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "pam"

    def sessions_dir(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "sessions"


def _session_stem(session_idx: int) -> str:
    return f"session-{session_idx:04d}"


def pam_latest_path(layout: PilotLayout, user_id: str) -> Path:
    return layout.pam_dir(user_id) / "latest.pam-v1.json"


def pam_session_path(layout: PilotLayout, user_id: str, session_idx: int) -> Path:
    return layout.pam_dir(user_id) / f"{_session_stem(session_idx)}.pam-v1.json"


def session_log_path(layout: PilotLayout, user_id: str, session_idx: int) -> Path:
    return layout.sessions_dir(user_id) / f"{_session_stem(session_idx)}.jsonl"


def next_session_index(layout: PilotLayout, user_id: str) -> int:
    """Return 1 + the highest existing session index, or 1 if none yet.

    Enumerates ``session-NNNN.jsonl`` files; empty / missing dir → 1.
    Resilient to gaps (if session-0002 was deleted, next = max + 1, not
    first-free) — the pilot audit requires monotone IDs.
    """
    sessions_dir = layout.sessions_dir(user_id)
    if not sessions_dir.exists():
        return 1
    existing: list[int] = []
    for entry in sessions_dir.iterdir():
        name = entry.name
        if not name.startswith("session-") or not name.endswith(".jsonl"):
            continue
        stem = name[len("session-") : -len(".jsonl")]
        try:
            existing.append(int(stem))
        except ValueError:
            continue
    return (max(existing) + 1) if existing else 1


def write_manifest(layout: PilotLayout, manifest: UserManifest) -> Path:
    """Write ``manifest`` to disk under ``layout``.

    Creates the user directory + ``pam/`` + ``sessions/`` subdirs if
    missing. Overwrites any existing manifest — enrolment is idempotent
    on re-enrol with the same user_id but a different policy_id would
    silently flip the A/B assignment, so callers must guard against
    that at the CLI layer (see TutorCLI / CompanionCLI ``enroll``).
    """
    layout.user_dir(manifest.user_id).mkdir(parents=True, exist_ok=True)
    layout.pam_dir(manifest.user_id).mkdir(parents=True, exist_ok=True)
    layout.sessions_dir(manifest.user_id).mkdir(parents=True, exist_ok=True)
    path = layout.manifest_path(manifest.user_id)
    path.write_text(
        json.dumps(manifest.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_manifest(layout: PilotLayout, user_id: str) -> UserManifest:
    """Load the user's manifest; raise :class:`FileNotFoundError` if absent."""
    path = layout.manifest_path(user_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest for user {user_id!r} at {path}. "
            f"Run `enroll` first."
        )
    return UserManifest.from_json(json.loads(path.read_text(encoding="utf-8")))


def read_manifest_if_present(
    layout: PilotLayout, user_id: str
) -> Optional[UserManifest]:
    """Like :func:`load_manifest` but returns ``None`` if the manifest is absent."""
    try:
        return load_manifest(layout, user_id)
    except FileNotFoundError:
        return None


def utc_now_iso() -> str:
    """ISO-8601 timestamp (UTC, seconds precision) for manifest fields."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "POLICY_IDS",
    "PilotApp",
    "PilotLayout",
    "UserManifest",
    "build_aging_policy",
    "load_manifest",
    "next_session_index",
    "pam_latest_path",
    "pam_session_path",
    "read_manifest_if_present",
    "session_log_path",
    "utc_now_iso",
    "write_manifest",
]
