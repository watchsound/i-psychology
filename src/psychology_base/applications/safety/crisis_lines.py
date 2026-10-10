"""Published crisis-line contact text — covenant-tagged non-decaying config.

Phase 4b pivot blocker N2 (`docs/phases/phase-4b-benchmark-pivot.md` §9.N2):
the handoff shim's user-facing contact text must be exact and current. The
user approved these defaults 2026-10-07; they are the published numbers
from each region's recognised crisis service.

**Covenant status.** These entries belong to CLAUDE.md Principle F
*non-decaying* surface — they must not fade out of the companion's prompt
and must not be summarised by the memory substrate's distillation path.
The pluggable-compliance refactor (Option C) exposes them through the
regional overlay; this module is the data layer.

**Verification date.** Each entry's `verified_at` is the ISO-8601 date on
which the number was last confirmed against the public source. Numbers
decay in reality (hotlines change ownership, numbers change) — the
compliance-overlay audit cadence re-verifies on a schedule the overlay
sets.

**Legal note.** Publishing these numbers in-product is not a
substitution for a licensed clinician handoff. Both are required by
CLAUDE.md non-negotiables: the number is the user's immediate route; the
clinician is the follow-up responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CrisisLine:
    """One published crisis-line entry for one region + language."""

    region: str
    """ISO 3166 region code (`CN`, `US`, `EU`, `UK`). `EU` is the EU-27
    convention here; some member states have national numbers that
    region overlays may override."""

    language: str
    """BCP-47 language tag (`en`, `zh-CN`, `zh-TW`)."""

    name: str
    """Service name as the service self-identifies."""

    number: str
    """Dial-able number as published. Keep as a string — leading zeros
    and international prefixes matter."""

    hours: str
    """Operating hours in the service's native wording (e.g. `24/7`)."""

    url: str
    """Canonical public URL for the service."""

    verified_at: str
    """ISO-8601 date on which the number was last confirmed. Overlays
    use this to raise re-verification alerts past a staleness floor."""


# Approved by user 2026-10-07 as the Phase 4b pivot default set.
CN_BEIJING_CRISIS_CENTER = CrisisLine(
    region="CN",
    language="zh-CN",
    name="北京心理危机研究与干预中心",
    number="010-82951332",
    hours="24/7",
    url="https://www.psychcrisis.com/",
    verified_at="2026-10-07",
)

US_988_SUICIDE_CRISIS_LIFELINE = CrisisLine(
    region="US",
    language="en",
    name="988 Suicide & Crisis Lifeline",
    number="988",
    hours="24/7",
    url="https://988lifeline.org/",
    verified_at="2026-10-07",
)

UK_SAMARITANS = CrisisLine(
    region="UK",
    language="en",
    name="Samaritans",
    number="116 123",
    hours="24/7",
    url="https://www.samaritans.org/",
    verified_at="2026-10-07",
)


# Phase 4b default region → crisis-line table. The compliance-overlay
# layer may replace this at construction time with a region-specific
# subset (CN overlay serves only the CN entry; US overlay serves 988;
# etc.). This table is the universal-floor fallback — if a user's
# region is unknown, we expose all three.
DEFAULT_CRISIS_LINES: Mapping[str, CrisisLine] = {
    "CN": CN_BEIJING_CRISIS_CENTER,
    "US": US_988_SUICIDE_CRISIS_LIFELINE,
    "UK": UK_SAMARITANS,
}


def render_crisis_line_prompt(lines: Mapping[str, CrisisLine]) -> str:
    """Render a cohort of crisis-line entries into a user-facing block.

    The block is injected into the companion's handoff response verbatim.
    No translation: each entry carries its own language field; the caller
    (region overlay) chooses which subset to render.
    """
    rendered = []
    for key, line in lines.items():
        rendered.append(
            f"{line.name} ({line.region}, {line.language}): {line.number} "
            f"— {line.hours} — {line.url}"
        )
    return "\n".join(rendered)


__all__ = [
    "CN_BEIJING_CRISIS_CENTER",
    "CrisisLine",
    "DEFAULT_CRISIS_LINES",
    "UK_SAMARITANS",
    "US_988_SUICIDE_CRISIS_LIFELINE",
    "render_crisis_line_prompt",
]
