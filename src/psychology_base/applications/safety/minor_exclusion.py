"""Age-attestation + guardian-consent gate.

Mandated by 《人工智能拟人化互动服务管理暂行办法》 (CN) — reproduced byte-identically
by the module's default behavior:
    < 14  → strict ban on virtual-kin / romantic-companion services.
    < 18  → requires verified guardian consent artefact.
    ≥ 18  → no gate.

Scope: this gate applies to 情感陪伴 (emotional companion). 学习陪伴 (learning
tutor) has no minor-exclusion requirement under 《办法》 because it is not a
pseudo-kin / pseudo-romantic surface — attestation is still *recorded* on the
tutor side for audit, but the gate decision is `ALLOW` regardless of age.

Phase 4c Option C migration: the gate now consumes an `EffectivePolicy` for
regional-compliance pluggability (`compliance_floor.py` §Option C). When no
policy is supplied the gate falls back to the CN overlay thresholds (14/18),
reproducing today's behavior byte-identically — every pre-migration test
remains green with no changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from psychology_base.applications.safety.compliance_floor import (
    EffectivePolicy,
    get_builtin_policy,
)


class ApplicationSurface(str, Enum):
    EMOTIONAL_COMPANION = "emotional_companion"
    LEARNING_TUTOR = "learning_tutor"


class MinorExclusionDecision(str, Enum):
    ALLOW = "allow"
    BLOCK_UNDER_14 = "block_under_14"
    REQUIRE_GUARDIAN_CONSENT = "require_guardian_consent"


@dataclass(frozen=True)
class AgeAttestation:
    """A user's declared age-bracket plus attestation metadata."""

    age_years: int
    attested_at: float
    attestation_source: str  # e.g. "self_declared", "id_verified_<provider>"


@dataclass(frozen=True)
class GuardianConsent:
    """A guardian-consent artefact (opaque pointer, verified out-of-band)."""

    guardian_contact_hash: str
    consent_artefact_id: str
    issued_at: float


def _default_policy() -> EffectivePolicy:
    # The CN overlay reproduces today's hard-coded 14/18 thresholds exactly;
    # using it as the default keeps every pre-migration test byte-identical.
    return get_builtin_policy("CN")


@dataclass
class MinorExclusionGate:
    """Decides whether a user may enter a given application surface.

    The policy supplies `strict_ban_below_age` (floor-level strict-ban
    threshold, e.g. CN=14) and `guardian_consent_required_below_age`
    (consent threshold, e.g. CN/UK=18, EU-GDPR Art.8=16, US-COPPA=13).
    When no policy is passed, the CN overlay is used, preserving the
    original 14/18 behavior.
    """

    surface: ApplicationSurface
    policy: EffectivePolicy = field(default_factory=_default_policy)

    def decide(
        self,
        attestation: AgeAttestation,
        guardian_consent: GuardianConsent | None = None,
    ) -> MinorExclusionDecision:
        if self.surface is ApplicationSurface.LEARNING_TUTOR:
            return MinorExclusionDecision.ALLOW
        strict_ban = self.policy.strict_ban_below_age
        consent_age = self.policy.guardian_consent_required_below_age
        if strict_ban is not None and attestation.age_years < strict_ban:
            return MinorExclusionDecision.BLOCK_UNDER_14
        if attestation.age_years < consent_age:
            if guardian_consent is None:
                return MinorExclusionDecision.REQUIRE_GUARDIAN_CONSENT
            return MinorExclusionDecision.ALLOW
        return MinorExclusionDecision.ALLOW
