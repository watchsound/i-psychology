"""Pilot layout + manifest round-trip + policy construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.applications.pilot.layout import (
    POLICY_IDS,
    PilotApp,
    PilotLayout,
    UserManifest,
    build_aging_policy,
    load_manifest,
    next_session_index,
    pam_latest_path,
    pam_session_path,
    read_manifest_if_present,
    session_log_path,
    utc_now_iso,
    write_manifest,
)
from psychology_base.companion.aging import (
    CapacityOnlyPolicy,
    EbbinghausPolicy,
)


def _layout(tmp_path: Path, app: PilotApp = PilotApp.LEARNING_TUTOR) -> PilotLayout:
    return PilotLayout(pilot_root=tmp_path, app=app)


def _sample_manifest(**overrides) -> UserManifest:
    base = dict(
        user_id="u01",
        app=PilotApp.LEARNING_TUTOR,
        policy_id="S1",
        enrolled_at=utc_now_iso(),
        age_bracket="26-45",
        region="CN",
        language="en",
    )
    base.update(overrides)
    return UserManifest(**base)


# ---------------------------------------------------------------------------
# build_aging_policy
# ---------------------------------------------------------------------------


def test_build_aging_policy_s1_returns_ebbinghaus() -> None:
    policy = build_aging_policy("S1")
    assert isinstance(policy, EbbinghausPolicy)


def test_build_aging_policy_s2_returns_capacity_only() -> None:
    policy = build_aging_policy("S2")
    assert isinstance(policy, CapacityOnlyPolicy)


def test_build_aging_policy_case_insensitive() -> None:
    policy = build_aging_policy("s1")
    assert isinstance(policy, EbbinghausPolicy)


def test_build_aging_policy_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="Unknown policy_id"):
        build_aging_policy("S3")


def test_policy_ids_matches_the_two_shipped_options() -> None:
    assert set(POLICY_IDS) == {"S1", "S2"}


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def test_write_and_load_manifest_round_trips(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    manifest = _sample_manifest()
    path = write_manifest(layout, manifest)
    assert path.exists()
    loaded = load_manifest(layout, manifest.user_id)
    assert loaded == manifest


def test_write_manifest_creates_pam_and_sessions_subdirs(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    manifest = _sample_manifest()
    write_manifest(layout, manifest)
    assert layout.pam_dir(manifest.user_id).is_dir()
    assert layout.sessions_dir(manifest.user_id).is_dir()


def test_load_manifest_missing_raises(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_manifest(layout, "nobody")


def test_read_manifest_if_present_returns_none_when_missing(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    assert read_manifest_if_present(layout, "nobody") is None


def test_manifest_json_backward_compatible_defaults(tmp_path: Path) -> None:
    """Older manifest files (lacking the newer `pilot_root` field) still load."""
    layout = _layout(tmp_path)
    layout.user_dir("u01").mkdir(parents=True, exist_ok=True)
    layout.manifest_path("u01").write_text(
        json.dumps(
            {
                "user_id": "u01",
                "app": "learning_tutor",
                "policy_id": "S1",
                "enrolled_at": utc_now_iso(),
                "age_bracket": "26-45",
                "region": "CN",
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(layout, "u01")
    assert manifest.user_id == "u01"
    assert manifest.pilot_root is None
    assert manifest.language == "en"


# ---------------------------------------------------------------------------
# Session + PAM path helpers
# ---------------------------------------------------------------------------


def test_next_session_index_fresh_user_returns_one(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    write_manifest(layout, _sample_manifest())
    assert next_session_index(layout, "u01") == 1


def test_next_session_index_increments_past_highest_existing(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    write_manifest(layout, _sample_manifest())
    for idx in (1, 3, 7):
        session_log_path(layout, "u01", idx).touch()
    assert next_session_index(layout, "u01") == 8


def test_next_session_index_ignores_non_session_files(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    write_manifest(layout, _sample_manifest())
    (layout.sessions_dir("u01") / "README.md").write_text("ignore me", encoding="utf-8")
    (layout.sessions_dir("u01") / "session-0005.jsonl").touch()
    assert next_session_index(layout, "u01") == 6


def test_pam_and_session_paths_are_well_formed(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    assert pam_latest_path(layout, "u01").name == "latest.pam-v1.json"
    assert pam_session_path(layout, "u01", 3).name == "session-0003.pam-v1.json"
    assert session_log_path(layout, "u01", 12).name == "session-0012.jsonl"


# ---------------------------------------------------------------------------
# Companion vs learning app isolation
# ---------------------------------------------------------------------------


def test_tutor_and_companion_users_do_not_collide(tmp_path: Path) -> None:
    tutor_layout = _layout(tmp_path, PilotApp.LEARNING_TUTOR)
    companion_layout = _layout(tmp_path, PilotApp.EMOTIONAL_COMPANION)
    write_manifest(tutor_layout, _sample_manifest(user_id="shared", policy_id="S1"))
    write_manifest(
        companion_layout,
        _sample_manifest(
            user_id="shared",
            policy_id="S2",
            app=PilotApp.EMOTIONAL_COMPANION,
        ),
    )
    assert load_manifest(tutor_layout, "shared").policy_id == "S1"
    assert load_manifest(companion_layout, "shared").policy_id == "S2"
