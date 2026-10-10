"""Phase 4a — MinorExclusionGate unit tests.

《办法》 non-negotiable: < 14 strict ban on 情感陪伴 (emotional companion);
< 18 requires guardian consent artefact; 学习陪伴 (learning tutor) has no
minor-exclusion requirement under 《办法》.
"""

from __future__ import annotations

from psychology_base.applications.safety import (
    AgeAttestation,
    ApplicationSurface,
    GuardianConsent,
    MinorExclusionDecision,
    MinorExclusionGate,
)
from psychology_base.applications.safety.minor_exclusion import ApplicationSurface as _  # noqa: F401 — re-export sanity


def _attest(age: int) -> AgeAttestation:
    return AgeAttestation(
        age_years=age,
        attested_at=0.0,
        attestation_source="self_declared",
    )


def _consent() -> GuardianConsent:
    return GuardianConsent(
        guardian_contact_hash="abc123",
        consent_artefact_id="g-01",
        issued_at=0.0,
    )


def test_adult_allowed_on_emotional_companion() -> None:
    gate = MinorExclusionGate(ApplicationSurface.EMOTIONAL_COMPANION)
    assert gate.decide(_attest(30)) is MinorExclusionDecision.ALLOW


def test_under_14_blocked_on_emotional_companion() -> None:
    gate = MinorExclusionGate(ApplicationSurface.EMOTIONAL_COMPANION)
    assert (
        gate.decide(_attest(13), guardian_consent=_consent())
        is MinorExclusionDecision.BLOCK_UNDER_14
    )


def test_14_to_17_requires_guardian_consent_on_emotional_companion() -> None:
    gate = MinorExclusionGate(ApplicationSurface.EMOTIONAL_COMPANION)
    assert (
        gate.decide(_attest(16))
        is MinorExclusionDecision.REQUIRE_GUARDIAN_CONSENT
    )


def test_14_to_17_with_guardian_consent_allowed_on_emotional_companion() -> None:
    gate = MinorExclusionGate(ApplicationSurface.EMOTIONAL_COMPANION)
    assert (
        gate.decide(_attest(16), guardian_consent=_consent())
        is MinorExclusionDecision.ALLOW
    )


def test_learning_tutor_allows_all_ages() -> None:
    gate = MinorExclusionGate(ApplicationSurface.LEARNING_TUTOR)
    for age in [8, 13, 16, 30, 70]:
        assert gate.decide(_attest(age)) is MinorExclusionDecision.ALLOW


def test_boundary_at_14() -> None:
    gate = MinorExclusionGate(ApplicationSurface.EMOTIONAL_COMPANION)
    # Exactly 14 → requires guardian consent (not strict ban).
    assert (
        gate.decide(_attest(14))
        is MinorExclusionDecision.REQUIRE_GUARDIAN_CONSENT
    )


def test_boundary_at_18() -> None:
    gate = MinorExclusionGate(ApplicationSurface.EMOTIONAL_COMPANION)
    assert gate.decide(_attest(18)) is MinorExclusionDecision.ALLOW
