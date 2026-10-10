"""Problem-set schema for pilot curricula.

A `ProblemSet` is a {calibration pool, pilot pool} pair. Calibration
problems are drawn from a flat pool at intake (per
`docs/phases/phase-4b-signoff.md` §9.3: 5 unaided per user, one per
week topic + 1 novel transfer). Pilot problems are structured into a
4-week × 5-per-week curriculum with per-week scaffold variants (full →
partial → focus → unaided; final week pairs integration with Week 1
derivatives for reconsolidation).

The schema is MathDial-compatible on the content fields (`problem`,
`ground_truth`, `common_wrong`) and layers scaffolding + reconsolidation
onto that core. See `calculus_v1.yaml` for the Phase 4b default.

Problems are fresh-authored, NOT MathDial derivatives — the signoff §9.3
flagged CC BY-SA 4.0 share-alike as the reason to keep the pilot
problem set license-independent of the MathDial dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class WeekTopic(str, Enum):
    """Phase 4b calculus curriculum topics, aligned with §9.3."""

    POWER_RULE = "power_rule"
    CHAIN_RULE = "chain_rule"
    MIXED_RULES = "mixed_rules"
    INTEGRATION = "integration"
    NOVEL_TRANSFER = "novel_transfer"


class ScaffoldLevel(str, Enum):
    """Scaffold intensity per the §9.3 fade schedule.

    - ``full``: tutor walks through every step with worked artefacts
      before releasing the pen.
    - ``partial``: tutor opens with a Socratic probe + one worked step
      if the learner stalls, then releases.
    - ``focus``: tutor offers focus-attention cues only; no worked
      steps unless the learner explicitly surrenders.
    - ``unaided``: tutor offers no scaffolds; learner solves cold.
    - ``reconsolidation_pair``: paired with a prior-week problem; tutor
      elicits the connection ("you computed X earlier — now compute Y;
      what do you notice?") per Pillar 1 reconsolidation.
    """

    FULL = "full"
    PARTIAL = "partial"
    FOCUS = "focus"
    UNAIDED = "unaided"
    RECONSOLIDATION_PAIR = "reconsolidation_pair"


@dataclass(frozen=True)
class CalibrationProblem:
    """One pre-pilot calibration problem (unaided baseline).

    The intake flow solves 5 of these (sampled to cover each
    `WeekTopic` plus a novel transfer) and records a per-user unaided
    baseline. Scaffolding-fading retention is scored against this
    per-user baseline, not against cohort mean.
    """

    id: str
    topic: WeekTopic
    difficulty: int                       # 1 (easiest) through 5 (hardest)
    problem: str
    ground_truth: str
    ground_truth_steps: tuple[str, ...] = ()
    common_wrong: tuple[str, ...] = ()
    notes: Optional[str] = None


@dataclass(frozen=True)
class PilotProblem:
    """One 4-week pilot problem.

    `scaffold_hints` are the ordered hint ladder the tutor delivers
    at ``full``/``partial`` scaffold levels; at ``focus`` only the
    first hint fires as a focus-attention cue; at ``unaided`` the
    hint ladder is withheld unless the learner surrenders.

    `reconsolidation_links` names the id(s) of earlier-week problems
    this problem revisits. Only populated when
    `scaffold_level == RECONSOLIDATION_PAIR` (Week 4 derivatives ↔
    integration pairing per §9.3).
    """

    id: str
    week: int                             # 1..4
    position_in_week: int                 # 1..5
    scaffold_level: ScaffoldLevel
    topic: WeekTopic
    difficulty: int
    problem: str
    ground_truth: str
    ground_truth_steps: tuple[str, ...] = ()
    common_wrong: tuple[str, ...] = ()
    scaffold_hints: tuple[str, ...] = ()
    reconsolidation_links: tuple[str, ...] = ()
    notes: Optional[str] = None


@dataclass(frozen=True)
class ProblemSet:
    """One versioned problem-set (calibration pool + 4-week pilot pool)."""

    id: str
    description: str
    license: str
    provenance: dict
    calibration: tuple[CalibrationProblem, ...]
    pilot: tuple[PilotProblem, ...]

    def calibration_by_topic(self, topic: WeekTopic) -> tuple[CalibrationProblem, ...]:
        return tuple(p for p in self.calibration if p.topic is topic)

    def pilot_by_week(self, week: int) -> tuple[PilotProblem, ...]:
        return tuple(p for p in self.pilot if p.week == week)

    def pilot_by_id(self, problem_id: str) -> PilotProblem:
        for p in self.pilot:
            if p.id == problem_id:
                return p
        raise KeyError(problem_id)
