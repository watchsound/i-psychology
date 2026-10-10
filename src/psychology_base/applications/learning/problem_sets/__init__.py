"""Problem sets for 学习陪伴 (resilient learning tutor) pilots.

See `calculus_v1.yaml` for the Phase 4b pilot problem set (20 calibration
+ 20 pilot with per-week scaffold variants, per
`docs/phases/phase-4b-signoff.md` §9.3 and §11 item 2).
"""

from psychology_base.applications.learning.problem_sets.loader import (
    DEFAULT_CALCULUS_V1_PATH,
    load_problem_set,
)
from psychology_base.applications.learning.problem_sets.schema import (
    CalibrationProblem,
    PilotProblem,
    ProblemSet,
    ScaffoldLevel,
    WeekTopic,
)

__all__ = [
    "CalibrationProblem",
    "DEFAULT_CALCULUS_V1_PATH",
    "PilotProblem",
    "ProblemSet",
    "ScaffoldLevel",
    "WeekTopic",
    "load_problem_set",
]
