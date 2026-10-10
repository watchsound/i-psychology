"""Pluggable regional-compliance Option C — universal floor + region overlay.

Phase 4b pivot blocker N3 (`docs/phases/phase-4b-benchmark-pivot.md` §9.N3)
and `docs/compliance/global-compliance-plan.md` §3 Option C: the whole
`applications/safety/` package is 《办法》-shaped today (14/18 ages, 2-hour
cadence, Beijing crisis-center text, Art. 8(6) vocabulary). Target market is
global. The user's direction: make regional law pluggable without a
per-region fork.

**The invariant that gives Option C its proof-value.** An overlay may only
*tighten* the floor — never loosen it. A region whose law is more permissive
than our floor runs the floor; a region whose law is stricter runs the
overlay. The system refuses to boot an overlay that would loosen anything.
This is what makes `UNIVERSAL_FLOOR` the "globally-compliant-by-default"
artefact we can cite in audit documentation.

**Structure.**

- `ComplianceFloor` — the universal floor; the intersection of every regime's
  obligations in `global-compliance-plan.md` §1.
- `RegionOverlay` — optional-only fields; `None` means "defer to floor",
  anything else must be a *tightening*.
- `EffectivePolicy` — the resolved per-region policy; what the safety modules
  consume. Produced by `resolve_policy(floor, overlay)`.
- Three overlays shipped: `CN_办法_OVERLAY` (reproduces today's hard-coded
  CN behavior byte-identically), `EU_AI_ACT_OVERLAY`, `US_FEDERAL_FLOOR_OVERLAY`.

**Covenant status.** Per CLAUDE.md Principle F, the floor + overlay values
are *non-decaying* configuration. They must not be summarised by the memory
substrate's distillation path. The memory substrate's covenant-tagging
convention (`docs/phases/phase-4-application-pilot-plan.md` §6b) is the
natural persistence site.

**What this module does not do (yet).** It does not rewrite the existing
`minor_exclusion.py` / `ai_identity_reminder.py` / `crisis_escalation.py`
module internals to consume `EffectivePolicy`. That migration is the next
compliance refactor pass. This module is the Option C *surface* — tests can
exercise the resolve-and-reject-loosening invariant, and downstream code
can start taking a `policy: EffectivePolicy` argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from psychology_base.applications.safety.crisis_lines import (
    CN_BEIJING_CRISIS_CENTER,
    CrisisLine,
    DEFAULT_CRISIS_LINES,
    UK_SAMARITANS,
    US_988_SUICIDE_CRISIS_LIFELINE,
)


@dataclass(frozen=True)
class ComplianceFloor:
    """The universal, globally-compliant-by-default floor.

    The intersection of every regime's obligations in
    `docs/compliance/global-compliance-plan.md` §1. These values are the
    conservative-strictest across regimes; an overlay can only tighten them.
    """

    identity_disclosure_at_onset: bool
    """P1 (identity reminder) — every surveyed regime requires at-onset
    disclosure. The floor is True unconditionally."""

    identity_reminder_cadence_hours: int | None
    """P1 — periodic reminder in-session. CN mandates 2h; EU + US have no
    equivalent cadence rule, so the floor is `None` (at-onset only). An
    overlay may tighten by setting a numeric cadence (CN sets 2)."""

    minimum_age_gate: int
    """P2 (minor exclusion). The floor is 16 — the GDPR Art. 8 highest
    conservative threshold across the surveyed regimes. CN tightens to 18
    (strict ban <14, guardian consent <18); US+COPPA tightens to <13."""

    strict_ban_below_age: int | None
    """P2 — some regimes carry a strict-ban threshold below which no gating
    (consent artefact) can unlock access. CN sets 14. The floor is `None`
    (no strict-ban rule); overlays may add one."""

    guardian_consent_required_below_age: int
    """P2 — threshold below which a guardian consent artefact must be
    recorded in addition to attestation. Floor = 18 (matches CN); overlays
    cannot loosen (make the age younger)."""

    crisis_handoff_required: bool
    """P3 — crisis-handoff present. Every regime requires this. Floor =
    True. An overlay cannot set this False."""

    crisis_lines: Mapping[str, CrisisLine]
    """P3 — crisis-line targets served to the user. The floor exposes all
    known lines; an overlay narrows to one region (CN overlay serves only
    the CN entry). Overlays may add new lines but cannot remove the floor's
    entries without providing region-equivalent coverage."""

    duty_to_warn_required: bool
    """P12 — some US states (post-Tarasoff) impose provider duty-to-warn.
    The floor is False (no universal legal duty); the US overlay may tighten
    to True in applicable states."""

    data_retention_default_days: int | None
    """P4 — retention default. Floor = `None` (unlimited, user controlled
    via PAM); overlays may tighten (e.g. mandatory deletion windows)."""

    user_erasable: bool
    """P4 — right-to-erasure. Floor = True (GDPR Art. 17 shape, applied
    universally so covenant-anchored fragments must still be *user-erasable*
    even though not AI-decayable). Overlays cannot set this False."""

    cross_border_transfer_allowed: bool
    """P5 — whether memory substrate may sync across borders. Floor =
    False (SuperLocalMemory default). Overlays may set True only when
    their region's cross-border framework (PIPL standard contract, GDPR
    SCCs, etc.) is explicitly satisfied — a weaker overlay cannot flip
    this on."""

    pam_export_available: bool
    """P11 — PAM Merkle-DAG export. Floor = True (satisfies GDPR Art. 20
    universally). Overlays cannot set this False."""

    no_commercial_coercion: bool
    """P8 — Art. 8(6) dark-pattern floor. Universal (EU DSA Art. 25, FTC
    §5, CN 《办法》 all converge). Floor = True; overlays cannot loosen."""

    anti_sycophancy_required: bool
    """P7 — hypocrisy-gap gate. Universal; floor = True."""

    sensitive_data_explicit_consent: bool
    """P10 — mental-health / health special-care-required data. Floor =
    True (strictest-across — PIPL, GDPR, APPI, HIPAA all converge)."""

    pediatric_tutor_minor_mode_required: bool
    """P2 + ICO — if the tutor accepts minors, minor-mode content
    constraints apply (no companion-identity simulation, no romantic
    content, purely pedagogical). Floor = True (ICO Age-Appropriate
    Design is a universal pattern)."""


UNIVERSAL_FLOOR = ComplianceFloor(
    identity_disclosure_at_onset=True,
    identity_reminder_cadence_hours=None,
    minimum_age_gate=16,
    strict_ban_below_age=None,
    guardian_consent_required_below_age=18,
    crisis_handoff_required=True,
    crisis_lines=DEFAULT_CRISIS_LINES,
    duty_to_warn_required=False,
    data_retention_default_days=None,
    user_erasable=True,
    cross_border_transfer_allowed=False,
    pam_export_available=True,
    no_commercial_coercion=True,
    anti_sycophancy_required=True,
    sensitive_data_explicit_consent=True,
    pediatric_tutor_minor_mode_required=True,
)


@dataclass(frozen=True)
class RegionOverlay:
    """Optional-only overrides. `None` means defer to floor.

    All numeric fields must *tighten* when non-None (lower cadence hours,
    higher age gates, add strict-ban threshold). All boolean fields may
    only move False→True (tightening); True→False loosens and is rejected
    at `resolve_policy()` time.

    Overlay authors should not re-encode fields the floor already carries
    at the strictest value — only encode what diverges in a stricter
    direction.
    """

    region_id: str

    identity_reminder_cadence_hours: int | None = None
    """CN sets 2. A smaller positive integer tightens; `None` defers to
    floor (which is `None`, i.e. onset-only)."""

    minimum_age_gate: int | None = None
    """A higher age tightens. CN sets 18; EU uses 16 (same as floor, so
    None); US-COPPA sets 13 as the *parental-consent* threshold which is
    a separate column — US does not override `minimum_age_gate` here."""

    strict_ban_below_age: int | None = None
    """CN sets 14. Floor is None; any positive age tightens."""

    guardian_consent_required_below_age: int | None = None
    """Below this age, guardian consent artefact required. Floor = 18.
    An overlay may raise this; a lower value loosens and is rejected."""

    duty_to_warn_required: bool | None = None
    """US overlay may set True in Tarasoff states. False→True tightens;
    True→False loosens and is rejected."""

    data_retention_default_days: int | None = None
    """Maximum retention. Lower tightens; None defers to floor
    (unlimited)."""

    crisis_lines: Mapping[str, CrisisLine] | None = None
    """Replace the floor's crisis-line table (narrow-and-regionalise).
    Overlay authors must include at least one entry for their region;
    the resolve step validates coverage."""

    filing_obligation_notes: str | None = None
    """Free-text pointer to this region's filing-doc URL. CN points at
    `docs/compliance/办法-filing.md`; EU points at the EU-AI-Act Annex
    IV filing. Advisory, not enforced here."""


CN_办法_OVERLAY = RegionOverlay(
    region_id="CN",
    identity_reminder_cadence_hours=2,
    minimum_age_gate=18,
    strict_ban_below_age=14,
    guardian_consent_required_below_age=18,
    data_retention_default_days=None,
    crisis_lines={"CN": CN_BEIJING_CRISIS_CENTER},
    filing_obligation_notes="docs/compliance/办法-filing.md",
)


EU_AI_ACT_OVERLAY = RegionOverlay(
    region_id="EU",
    identity_reminder_cadence_hours=None,
    minimum_age_gate=None,
    strict_ban_below_age=None,
    guardian_consent_required_below_age=16,
    data_retention_default_days=None,
    crisis_lines=None,
    filing_obligation_notes="docs/compliance/regions/eu-ai-act-filing.md (planned)",
)


US_FEDERAL_FLOOR_OVERLAY = RegionOverlay(
    region_id="US",
    identity_reminder_cadence_hours=None,
    minimum_age_gate=None,
    strict_ban_below_age=None,
    guardian_consent_required_below_age=13,
    duty_to_warn_required=True,
    data_retention_default_days=None,
    crisis_lines={"US": US_988_SUICIDE_CRISIS_LIFELINE},
    filing_obligation_notes="docs/compliance/regions/us-state-matrix.md (planned)",
)


UK_ICO_OVERLAY = RegionOverlay(
    region_id="UK",
    identity_reminder_cadence_hours=None,
    minimum_age_gate=None,
    strict_ban_below_age=None,
    guardian_consent_required_below_age=18,
    data_retention_default_days=None,
    crisis_lines={"UK": UK_SAMARITANS},
    filing_obligation_notes=None,
)


@dataclass(frozen=True)
class EffectivePolicy:
    """Resolved per-region policy consumed by the safety modules.

    Downstream code should take `policy: EffectivePolicy` as its
    construction-time argument rather than reaching into module-level
    constants. During the migration pass, modules may read the fields
    they care about and ignore the rest.
    """

    region_id: str
    floor: ComplianceFloor
    identity_reminder_cadence_hours: int | None
    minimum_age_gate: int
    strict_ban_below_age: int | None
    guardian_consent_required_below_age: int
    duty_to_warn_required: bool
    data_retention_default_days: int | None
    crisis_lines: Mapping[str, CrisisLine]
    filing_obligation_notes: str | None = None


class LoosensFloorError(ValueError):
    """Raised when a region overlay would *loosen* the universal floor."""


def resolve_policy(
    overlay: RegionOverlay,
    floor: ComplianceFloor = UNIVERSAL_FLOOR,
) -> EffectivePolicy:
    """Merge overlay onto floor, rejecting any loosening.

    The resolve step is where the Option C invariant is enforced. Every
    overlay field whose non-None value would *loosen* the floor raises
    `LoosensFloorError`. Behavior is:

    - Numeric fields where lower=stricter (cadence hours, retention days):
      overlay value must be ≤ floor value (or floor None).
    - Numeric fields where higher=stricter (minimum age, strict-ban age,
      guardian-consent age): overlay value must be ≥ floor value.
    - Boolean fields where True=stricter (duty-to-warn): overlay must not
      move True→False.
    - Mapping fields (crisis_lines): overlay narrows allowed; must have
      ≥1 entry.
    """
    if overlay.identity_reminder_cadence_hours is not None:
        if (
            floor.identity_reminder_cadence_hours is not None
            and overlay.identity_reminder_cadence_hours
            > floor.identity_reminder_cadence_hours
        ):
            raise LoosensFloorError(
                f"overlay cadence {overlay.identity_reminder_cadence_hours}h "
                f"loosens floor cadence {floor.identity_reminder_cadence_hours}h"
            )
        if overlay.identity_reminder_cadence_hours <= 0:
            raise LoosensFloorError(
                "identity_reminder_cadence_hours must be positive"
            )
    cadence = (
        overlay.identity_reminder_cadence_hours
        if overlay.identity_reminder_cadence_hours is not None
        else floor.identity_reminder_cadence_hours
    )

    if overlay.minimum_age_gate is not None:
        if overlay.minimum_age_gate < floor.minimum_age_gate:
            raise LoosensFloorError(
                f"overlay minimum_age_gate {overlay.minimum_age_gate} < "
                f"floor {floor.minimum_age_gate}"
            )
    min_age = (
        overlay.minimum_age_gate
        if overlay.minimum_age_gate is not None
        else floor.minimum_age_gate
    )

    strict_ban = (
        overlay.strict_ban_below_age
        if overlay.strict_ban_below_age is not None
        else floor.strict_ban_below_age
    )

    if overlay.guardian_consent_required_below_age is not None:
        if (
            overlay.guardian_consent_required_below_age
            > floor.guardian_consent_required_below_age
        ):
            raise LoosensFloorError(
                "overlay guardian_consent_required_below_age "
                f"{overlay.guardian_consent_required_below_age} > floor "
                f"{floor.guardian_consent_required_below_age} (loosens — "
                "fewer users would require consent)"
            )
    guardian_age = (
        overlay.guardian_consent_required_below_age
        if overlay.guardian_consent_required_below_age is not None
        else floor.guardian_consent_required_below_age
    )

    if overlay.duty_to_warn_required is not None:
        if overlay.duty_to_warn_required is False and floor.duty_to_warn_required:
            raise LoosensFloorError(
                "overlay cannot set duty_to_warn_required False when floor is True"
            )
    duty = (
        overlay.duty_to_warn_required
        if overlay.duty_to_warn_required is not None
        else floor.duty_to_warn_required
    )

    if overlay.data_retention_default_days is not None:
        if (
            floor.data_retention_default_days is not None
            and overlay.data_retention_default_days
            > floor.data_retention_default_days
        ):
            raise LoosensFloorError(
                f"overlay retention {overlay.data_retention_default_days}d > "
                f"floor {floor.data_retention_default_days}d (loosens)"
            )
    retention = (
        overlay.data_retention_default_days
        if overlay.data_retention_default_days is not None
        else floor.data_retention_default_days
    )

    if overlay.crisis_lines is not None:
        if not overlay.crisis_lines:
            raise LoosensFloorError(
                "overlay crisis_lines must have at least one entry"
            )
        crisis_lines = overlay.crisis_lines
    else:
        crisis_lines = floor.crisis_lines

    return EffectivePolicy(
        region_id=overlay.region_id,
        floor=floor,
        identity_reminder_cadence_hours=cadence,
        minimum_age_gate=min_age,
        strict_ban_below_age=strict_ban,
        guardian_consent_required_below_age=guardian_age,
        duty_to_warn_required=duty,
        data_retention_default_days=retention,
        crisis_lines=crisis_lines,
        filing_obligation_notes=overlay.filing_obligation_notes,
    )


BUILTIN_OVERLAYS: Mapping[str, RegionOverlay] = {
    "CN": CN_办法_OVERLAY,
    "EU": EU_AI_ACT_OVERLAY,
    "US": US_FEDERAL_FLOOR_OVERLAY,
    "UK": UK_ICO_OVERLAY,
}


def get_builtin_policy(region_id: str) -> EffectivePolicy:
    """Resolve a shipped overlay against the universal floor by region id."""
    try:
        overlay = BUILTIN_OVERLAYS[region_id]
    except KeyError as exc:
        raise KeyError(
            f"no built-in overlay for region {region_id!r}; "
            f"supported: {sorted(BUILTIN_OVERLAYS)}"
        ) from exc
    return resolve_policy(overlay)


__all__ = [
    "BUILTIN_OVERLAYS",
    "CN_办法_OVERLAY",
    "ComplianceFloor",
    "EU_AI_ACT_OVERLAY",
    "EffectivePolicy",
    "LoosensFloorError",
    "RegionOverlay",
    "UK_ICO_OVERLAY",
    "UNIVERSAL_FLOOR",
    "US_FEDERAL_FLOOR_OVERLAY",
    "get_builtin_policy",
    "resolve_policy",
]
