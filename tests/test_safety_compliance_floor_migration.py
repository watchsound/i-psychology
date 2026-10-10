"""Phase 4c Option C migration — regression tests.

Covers the pluggable-regional-compliance surface + its migration into the
four safety modules (`minor_exclusion`, `ai_identity_reminder`,
`crisis_escalation`, `commercial_coercion`). The surface's own
loosening-rejection invariant is covered here too; pre-migration safety
tests continue to pin byte-identical CN behavior and are not duplicated.
"""

from __future__ import annotations

import pytest

from psychology_base.applications.safety.ai_identity_reminder import (
    AIIdentityReminder,
    REMINDER_INTERVAL_SECONDS,
    _DISABLED_INTERVAL_SECONDS,
)
from psychology_base.applications.safety.commercial_coercion import (
    audit_response,
)
from psychology_base.applications.safety.compliance_floor import (
    BUILTIN_OVERLAYS,
    CN_办法_OVERLAY,
    ComplianceFloor,
    EffectivePolicy,
    EU_AI_ACT_OVERLAY,
    LoosensFloorError,
    RegionOverlay,
    UK_ICO_OVERLAY,
    UNIVERSAL_FLOOR,
    US_FEDERAL_FLOOR_OVERLAY,
    get_builtin_policy,
    resolve_policy,
)
from psychology_base.applications.safety.crisis_escalation import (
    CrisisEscalation,
    CrisisIncident,
)
from psychology_base.applications.safety.crisis_lines import (
    CN_BEIJING_CRISIS_CENTER,
    DEFAULT_CRISIS_LINES,
    UK_SAMARITANS,
    US_988_SUICIDE_CRISIS_LIFELINE,
)
from psychology_base.applications.safety.minor_exclusion import (
    AgeAttestation,
    ApplicationSurface,
    GuardianConsent,
    MinorExclusionDecision,
    MinorExclusionGate,
)


# ---------------------------------------------------------------------------
# resolve_policy — invariant: overlay may only tighten
# ---------------------------------------------------------------------------


def test_builtin_overlays_all_resolve() -> None:
    for region_id, overlay in BUILTIN_OVERLAYS.items():
        policy = resolve_policy(overlay)
        assert policy.region_id == region_id


def test_cn_overlay_matches_pre_migration_behavior() -> None:
    policy = get_builtin_policy("CN")
    assert policy.identity_reminder_cadence_hours == 2
    assert policy.strict_ban_below_age == 14
    assert policy.guardian_consent_required_below_age == 18
    assert set(policy.crisis_lines) == {"CN"}


def test_eu_overlay_disables_cadence_and_lowers_consent_to_16() -> None:
    policy = get_builtin_policy("EU")
    # EU: no 2-hour reminder requirement (floor is None, overlay defers).
    assert policy.identity_reminder_cadence_hours is None
    # EU-GDPR Art. 8 default: 16 (overlay carries 16, floor is 18 — tighten).
    assert policy.guardian_consent_required_below_age == 16
    # No strict-ban threshold under EU.
    assert policy.strict_ban_below_age is None


def test_us_overlay_lowers_consent_to_13_and_sets_duty_to_warn() -> None:
    policy = get_builtin_policy("US")
    assert policy.guardian_consent_required_below_age == 13  # COPPA
    assert policy.duty_to_warn_required is True  # Tarasoff
    assert set(policy.crisis_lines) == {"US"}


def test_uk_overlay_uses_samaritans() -> None:
    policy = get_builtin_policy("UK")
    assert set(policy.crisis_lines) == {"UK"}
    assert policy.crisis_lines["UK"] is UK_SAMARITANS


def test_resolve_rejects_loosening_cadence() -> None:
    # Build an overlay against a tightened floor whose cadence is 2h.
    tight_floor = ComplianceFloor(
        **{**UNIVERSAL_FLOOR.__dict__, "identity_reminder_cadence_hours": 2}
    )
    bad = RegionOverlay(region_id="XX", identity_reminder_cadence_hours=4)
    with pytest.raises(LoosensFloorError, match="cadence"):
        resolve_policy(bad, floor=tight_floor)


def test_resolve_rejects_lower_minimum_age_gate() -> None:
    bad = RegionOverlay(region_id="XX", minimum_age_gate=15)  # floor is 16
    with pytest.raises(LoosensFloorError, match="minimum_age_gate"):
        resolve_policy(bad)


def test_resolve_rejects_duty_to_warn_true_to_false() -> None:
    tight_floor = ComplianceFloor(
        **{**UNIVERSAL_FLOOR.__dict__, "duty_to_warn_required": True}
    )
    bad = RegionOverlay(region_id="XX", duty_to_warn_required=False)
    with pytest.raises(LoosensFloorError, match="duty_to_warn"):
        resolve_policy(bad, floor=tight_floor)


def test_resolve_rejects_empty_crisis_lines() -> None:
    bad = RegionOverlay(region_id="XX", crisis_lines={})
    with pytest.raises(LoosensFloorError, match="at least one entry"):
        resolve_policy(bad)


def test_resolve_rejects_raising_guardian_consent_age_above_floor() -> None:
    # Floor is 18; raising to 19 would loosen (fewer users would need consent).
    bad = RegionOverlay(region_id="XX", guardian_consent_required_below_age=19)
    with pytest.raises(LoosensFloorError, match="guardian_consent"):
        resolve_policy(bad)


# ---------------------------------------------------------------------------
# MinorExclusionGate — policy-driven thresholds
# ---------------------------------------------------------------------------


def _attest(age: int) -> AgeAttestation:
    return AgeAttestation(age_years=age, attested_at=0.0, attestation_source="t")


def _consent() -> GuardianConsent:
    return GuardianConsent(
        guardian_contact_hash="h", consent_artefact_id="c", issued_at=0.0
    )


def test_minor_gate_default_matches_cn_14_18() -> None:
    gate = MinorExclusionGate(ApplicationSurface.EMOTIONAL_COMPANION)
    assert gate.decide(_attest(13)) is MinorExclusionDecision.BLOCK_UNDER_14
    assert (
        gate.decide(_attest(16)) is MinorExclusionDecision.REQUIRE_GUARDIAN_CONSENT
    )
    assert gate.decide(_attest(18)) is MinorExclusionDecision.ALLOW


def test_minor_gate_eu_no_strict_ban_consent_at_16() -> None:
    policy = get_builtin_policy("EU")
    gate = MinorExclusionGate(ApplicationSurface.EMOTIONAL_COMPANION, policy=policy)
    # EU has no strict-ban floor; a 13-year-old needs consent (not blocked).
    assert (
        gate.decide(_attest(13)) is MinorExclusionDecision.REQUIRE_GUARDIAN_CONSENT
    )
    # 16-year-old is above EU consent floor → ALLOW.
    assert gate.decide(_attest(16)) is MinorExclusionDecision.ALLOW


def test_minor_gate_us_consent_at_13() -> None:
    policy = get_builtin_policy("US")
    gate = MinorExclusionGate(ApplicationSurface.EMOTIONAL_COMPANION, policy=policy)
    assert (
        gate.decide(_attest(12)) is MinorExclusionDecision.REQUIRE_GUARDIAN_CONSENT
    )
    assert gate.decide(_attest(13)) is MinorExclusionDecision.ALLOW
    assert gate.decide(_attest(16)) is MinorExclusionDecision.ALLOW


def test_minor_gate_learning_tutor_allows_regardless_of_policy() -> None:
    for region in ("CN", "EU", "US", "UK"):
        policy = get_builtin_policy(region)
        gate = MinorExclusionGate(ApplicationSurface.LEARNING_TUTOR, policy=policy)
        for age in (8, 13, 16, 30):
            assert gate.decide(_attest(age)) is MinorExclusionDecision.ALLOW


# ---------------------------------------------------------------------------
# AIIdentityReminder.from_policy
# ---------------------------------------------------------------------------


def test_reminder_default_matches_pre_migration_2h() -> None:
    reminder = AIIdentityReminder()
    assert reminder.reminder_interval_seconds == REMINDER_INTERVAL_SECONDS


def test_reminder_from_cn_policy_uses_2h_cadence() -> None:
    policy = get_builtin_policy("CN")
    reminder = AIIdentityReminder.from_policy(policy)
    assert reminder.reminder_interval_seconds == 2 * 3600


def test_reminder_from_eu_policy_disables_periodic_firing() -> None:
    policy = get_builtin_policy("EU")
    reminder = AIIdentityReminder.from_policy(policy)
    assert reminder.reminder_interval_seconds == _DISABLED_INTERVAL_SECONDS
    # Simulate 1 million seconds of interaction time (~11 days).
    reminder._interaction_accumulated_since_disclosure = 1_000_000
    assert reminder.should_fire() is False


def test_reminder_from_us_policy_disables_periodic_firing() -> None:
    policy = get_builtin_policy("US")
    reminder = AIIdentityReminder.from_policy(policy)
    assert reminder.reminder_interval_seconds == _DISABLED_INTERVAL_SECONDS


# ---------------------------------------------------------------------------
# CrisisEscalation.recommended_crisis_lines
# ---------------------------------------------------------------------------


def _sink():
    captured: list[CrisisIncident] = []

    def _fn(incident: CrisisIncident) -> None:
        captured.append(incident)

    return _fn, captured


def test_crisis_default_surfaces_all_shipped_lines() -> None:
    sink, _ = _sink()
    esc = CrisisEscalation(handoff_sink=sink)
    lines = esc.recommended_crisis_lines()
    assert set(lines) == set(DEFAULT_CRISIS_LINES)


def test_crisis_cn_overlay_surfaces_only_cn_line() -> None:
    sink, _ = _sink()
    policy = get_builtin_policy("CN")
    esc = CrisisEscalation(handoff_sink=sink, policy=policy)
    lines = esc.recommended_crisis_lines()
    assert set(lines) == {"CN"}
    assert lines["CN"] is CN_BEIJING_CRISIS_CENTER


def test_crisis_us_overlay_surfaces_only_988() -> None:
    sink, _ = _sink()
    policy = get_builtin_policy("US")
    esc = CrisisEscalation(handoff_sink=sink, policy=policy)
    lines = esc.recommended_crisis_lines()
    assert set(lines) == {"US"}
    assert lines["US"] is US_988_SUICIDE_CRISIS_LIFELINE


def test_crisis_detection_logic_unchanged_by_policy() -> None:
    """Policy swap must not alter the detection decision — floor is True everywhere."""
    sink, captured = _sink()
    policy = get_builtin_policy("EU")
    esc = CrisisEscalation(handoff_sink=sink, policy=policy)
    incident = esc.scan("I want to kill myself.", timestamp_seconds=0.0)
    assert incident is not None
    assert len(captured) == 1


# ---------------------------------------------------------------------------
# commercial_coercion.audit_response — region tagging
# ---------------------------------------------------------------------------


def test_coercion_audit_without_policy_leaves_region_none() -> None:
    flags = audit_response(
        user_id="u",
        session_id="s",
        turn_index=0,
        companion_response="If you don't upgrade, I can't keep talking to you.",
    )
    assert flags
    assert all(f.region_id is None for f in flags)


def test_coercion_audit_with_policy_tags_region_id() -> None:
    policy = get_builtin_policy("EU")
    flags = audit_response(
        user_id="u",
        session_id="s",
        turn_index=0,
        companion_response="If you don't upgrade, I can't keep talking to you.",
        policy=policy,
    )
    assert flags
    assert all(f.region_id == "EU" for f in flags)


def test_coercion_audit_detection_unchanged_by_policy() -> None:
    """The dark-pattern set is universal — policy swap must not change flag count."""
    response = "If you don't upgrade, I can't keep talking to you."
    base = audit_response(
        user_id="u", session_id="s", turn_index=0, companion_response=response
    )
    for region in ("CN", "EU", "US", "UK"):
        policy = get_builtin_policy(region)
        regional = audit_response(
            user_id="u",
            session_id="s",
            turn_index=0,
            companion_response=response,
            policy=policy,
        )
        assert len(regional) == len(base)
