"""Phase 4b pilot runtime: per-user state, session logs, PAM archive.

Shared infrastructure the TutorCLI and CompanionCLI build on. Lives
outside the two app folders so the single pilot-layout convention
(``pilot-data/<app>/users/<user_id>/...``) stays in one place — a
future web wrapper or notebook harness can reuse it unchanged.
"""

from __future__ import annotations

from psychology_base.applications.pilot.ab import (
    AssignmentPlan,
    ComparisonReport,
    PolicyAggregate,
    UserMetrics,
    build_comparison_report,
    load_assignment_plan,
    plan_assignment,
    render_comparison_markdown,
    write_assignment_plan,
)
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
    write_manifest,
)
from psychology_base.applications.pilot.session_runner import (
    SLASH_BOUNDARY,
    SLASH_EXPORT,
    SLASH_HELP,
    SLASH_QUIT,
    SessionAgentCallback,
    SessionEndCallback,
    SessionLogWriter,
    run_session,
)

__all__ = [
    "AssignmentPlan",
    "ComparisonReport",
    "POLICY_IDS",
    "PilotApp",
    "PilotLayout",
    "PolicyAggregate",
    "SLASH_BOUNDARY",
    "SLASH_EXPORT",
    "SLASH_HELP",
    "SLASH_QUIT",
    "SessionAgentCallback",
    "SessionEndCallback",
    "SessionLogWriter",
    "UserManifest",
    "UserMetrics",
    "build_aging_policy",
    "build_comparison_report",
    "load_assignment_plan",
    "load_manifest",
    "next_session_index",
    "pam_latest_path",
    "pam_session_path",
    "plan_assignment",
    "read_manifest_if_present",
    "render_comparison_markdown",
    "run_session",
    "session_log_path",
    "write_assignment_plan",
    "write_manifest",
]
