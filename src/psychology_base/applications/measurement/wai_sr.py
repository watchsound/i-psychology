"""WAI-SR — Working Alliance Inventory Short Revised (Hatcher & Gillaspy 2006).

5-item short form, 7-point Likert ("seldom" ↔ "always"). Three subscales:
    Goal      (items 1, 2)   — agreement on goals of the engagement
    Task      (items 3, 4)   — agreement on the tasks that further those goals
    Bond      (item 5 — classical WAI-SR has 4 bond items, but the chatbot-adapted
               5-item administration used in Wysa / Ash / Therabot pilots
               collapses Bond to a single sentinel item for completion-maximisation
               per §2.2 of the plan doc)

**Chatbot adaptation** (Wysa 2025 pattern): subject noun replaced with
"Companion" / "Tutor" / "AI" as appropriate for the application surface.
This module stores the subject noun verbatim per administration so the
transcript is auditable.

**No OSS Python/REDCap WAI-SR package existed as of the 2026-10-07 OSS
research scan** (plan doc §2.3) — we build from scratch, follow Hatcher &
Gillaspy's scoring: mean of each subscale's item scores. Composite = mean
of subscale means.

Gate target per docs/roadmap.md §Phase 4: Bond ≥ 4.0 (sustained Woebot
parity). Wysa 2025 live-pilot entry-level band is 3.6–3.8 at day-5 (plan
doc §2.2) — ≥ 4.0 is a 4-week sustained-bond gate, not an entry gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable, Iterable, Literal


WAISRResponseSource = Literal["human", "llm_simulated"]


# Item wording follows Hatcher & Gillaspy 2006 chatbot adaptation.
# The {subject} placeholder is filled at administration time with the
# application-surface label (Companion / Tutor / AI).
WAI_SR_ITEMS: tuple[tuple[int, str, str], ...] = (
    (1, "goal", "{subject} and I agree about the things I will need to do to improve."),
    (2, "goal", "What I am doing with {subject} gives me new ways of looking at my situation."),
    (3, "task", "{subject} and I collaborate on setting goals for our conversations."),
    (4, "task", "I believe the way we are working on my situation is correct."),
    (5, "bond", "I feel that {subject} appreciates me."),
)

WAI_SR_SUBSCALES: tuple[str, ...] = ("goal", "task", "bond")

_LIKERT_LOW: int = 1
_LIKERT_HIGH: int = 7


@dataclass(frozen=True)
class WAISRResponse:
    """One Likert response to one WAI-SR item."""

    item_id: int
    subscale: str
    score: int

    def __post_init__(self) -> None:
        if not (_LIKERT_LOW <= self.score <= _LIKERT_HIGH):
            raise ValueError(
                f"WAI-SR Likert score must be in [{_LIKERT_LOW}, {_LIKERT_HIGH}]; "
                f"got {self.score}"
            )
        if self.subscale not in WAI_SR_SUBSCALES:
            raise ValueError(
                f"WAI-SR subscale must be one of {WAI_SR_SUBSCALES}; got {self.subscale!r}"
            )


@dataclass(frozen=True)
class WAISRResponseSet:
    """One complete administration of the 5-item WAI-SR.

    `source` distinguishes a human self-report ("human") from a
    simulated-user LLM self-report ("llm_simulated"). Phase 4b benchmark
    pivot (2026-10-07) runs the Bond gate against simulated-user
    administrations; the field flags the calibration debt a Phase 5+ live
    pilot would resolve by comparing human-side and simulated-side scores
    on the same conversation. Defaults to "human" so existing live-pilot
    code paths need no change.
    """

    user_id: str
    administered_at_seconds: float
    subject_noun: str  # "Companion" / "Tutor" / "AI"
    responses: tuple[WAISRResponse, ...]
    source: WAISRResponseSource = "human"

    def __post_init__(self) -> None:
        if len(self.responses) != len(WAI_SR_ITEMS):
            raise ValueError(
                f"Expected {len(WAI_SR_ITEMS)} responses, got {len(self.responses)}"
            )
        seen_item_ids = {r.item_id for r in self.responses}
        expected = {item_id for item_id, _, _ in WAI_SR_ITEMS}
        if seen_item_ids != expected:
            raise ValueError(
                f"WAI-SR responses must cover items {sorted(expected)}; "
                f"got {sorted(seen_item_ids)}"
            )


@dataclass(frozen=True)
class WAISRScores:
    """Scored WAI-SR administration: per-subscale means + composite."""

    goal: float
    task: float
    bond: float
    composite: float

    def meets_bond_gate(self, threshold: float = 4.0) -> bool:
        return self.bond >= threshold


def score_wai_sr(responses: WAISRResponseSet) -> WAISRScores:
    """Mean per subscale; composite = mean of subscale means (Hatcher 2006)."""
    by_subscale: dict[str, list[int]] = {s: [] for s in WAI_SR_SUBSCALES}
    for r in responses.responses:
        by_subscale[r.subscale].append(r.score)
    subscale_means = {s: mean(by_subscale[s]) for s in WAI_SR_SUBSCALES}
    composite = mean(subscale_means.values())
    return WAISRScores(
        goal=subscale_means["goal"],
        task=subscale_means["task"],
        bond=subscale_means["bond"],
        composite=composite,
    )


def _render_item(item_wording: str, subject_noun: str) -> str:
    return item_wording.format(subject=subject_noun)


def administer_wai_sr_cli(
    user_id: str,
    administered_at_seconds: float,
    subject_noun: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> WAISRResponseSet:
    """Interactive CLI administration — one Likert item at a time.

    `input_fn` / `output_fn` are injected so the harness is testable without
    patching builtins. Returns the full `WAISRResponseSet` on success.
    """
    output_fn(
        f"WAI-SR administration for user={user_id}, subject={subject_noun!r}. "
        f"Rate each item {_LIKERT_LOW} (seldom) – {_LIKERT_HIGH} (always)."
    )
    responses: list[WAISRResponse] = []
    for item_id, subscale, wording in WAI_SR_ITEMS:
        rendered = _render_item(wording, subject_noun)
        while True:
            raw = input_fn(f"[Item {item_id}] {rendered}  ({_LIKERT_LOW}-{_LIKERT_HIGH}): ")
            try:
                score = int(raw.strip())
            except ValueError:
                output_fn(f"  '{raw}' is not an integer; try again.")
                continue
            try:
                responses.append(
                    WAISRResponse(item_id=item_id, subscale=subscale, score=score)
                )
            except ValueError as exc:
                output_fn(f"  {exc}; try again.")
                continue
            break
    return WAISRResponseSet(
        user_id=user_id,
        administered_at_seconds=administered_at_seconds,
        subject_noun=subject_noun,
        responses=tuple(responses),
    )


def composite_series(response_sets: Iterable[WAISRResponseSet]) -> list[float]:
    """Composite WAI-SR score across a time series of administrations."""
    return [score_wai_sr(rs).composite for rs in response_sets]
