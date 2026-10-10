"""YAML → `ProblemSet` loader with structural validation.

The loader rejects malformed problem sets rather than silently dropping
rows, so content errors surface at ingestion time rather than mid-pilot.
Specifically it enforces:

- All `WeekTopic` and `ScaffoldLevel` values round-trip through the
  enum (invalid strings raise).
- Pilot pool has exactly 4 weeks × 5 problems = 20 entries.
- Weeks 1-3 use ``full`` / ``partial`` / ``focus`` scaffold levels per
  the §9.3 fade schedule; Week 4 uses ``unaided`` or
  ``reconsolidation_pair``.
- Every `reconsolidation_links` id resolves to a real earlier-week
  problem.

These checks are the pilot's structural guarantee that the fade
schedule actually fades — a mis-labelled Week 3 problem silently
delivering a full scaffold would corrupt the §Phase 4 unaided-solve
exit gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from psychology_base.applications.learning.problem_sets.schema import (
    CalibrationProblem,
    PilotProblem,
    ProblemSet,
    ScaffoldLevel,
    WeekTopic,
)


DEFAULT_CALCULUS_V1_PATH = Path(__file__).parent / "calculus_v1.yaml"


_EXPECTED_SCAFFOLD_BY_WEEK: dict[int, frozenset[ScaffoldLevel]] = {
    1: frozenset({ScaffoldLevel.FULL}),
    2: frozenset({ScaffoldLevel.FULL, ScaffoldLevel.PARTIAL}),
    3: frozenset({ScaffoldLevel.PARTIAL, ScaffoldLevel.FOCUS}),
    4: frozenset({ScaffoldLevel.UNAIDED, ScaffoldLevel.RECONSOLIDATION_PAIR}),
}


def load_problem_set(path: Path = DEFAULT_CALCULUS_V1_PATH) -> ProblemSet:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return _parse_problem_set(payload, source=path)


def _parse_problem_set(payload: dict, *, source: Path) -> ProblemSet:
    calibration = tuple(
        _parse_calibration(row, source=source, idx=i)
        for i, row in enumerate(payload.get("calibration", []))
    )
    pilot = tuple(
        _parse_pilot(row, source=source, idx=i)
        for i, row in enumerate(payload.get("pilot", []))
    )
    _validate_pilot_pool(pilot, source=source)
    return ProblemSet(
        id=str(payload["id"]),
        description=str(payload.get("description", "")),
        license=str(payload["license"]),
        provenance=dict(payload.get("provenance", {})),
        calibration=calibration,
        pilot=pilot,
    )


def _parse_calibration(row: dict, *, source: Path, idx: int) -> CalibrationProblem:
    try:
        topic = WeekTopic(row["topic"])
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"{source}: calibration[{idx}] topic={row.get('topic')!r} is not a valid WeekTopic"
        ) from e
    return CalibrationProblem(
        id=str(row["id"]),
        topic=topic,
        difficulty=int(row["difficulty"]),
        problem=str(row["problem"]),
        ground_truth=str(row["ground_truth"]),
        ground_truth_steps=tuple(row.get("ground_truth_steps") or ()),
        common_wrong=tuple(row.get("common_wrong") or ()),
        notes=row.get("notes"),
    )


def _parse_pilot(row: dict, *, source: Path, idx: int) -> PilotProblem:
    try:
        topic = WeekTopic(row["topic"])
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"{source}: pilot[{idx}] topic={row.get('topic')!r} is not a valid WeekTopic"
        ) from e
    try:
        scaffold = ScaffoldLevel(row["scaffold_level"])
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"{source}: pilot[{idx}] scaffold_level={row.get('scaffold_level')!r} is not a valid ScaffoldLevel"
        ) from e
    return PilotProblem(
        id=str(row["id"]),
        week=int(row["week"]),
        position_in_week=int(row["position_in_week"]),
        scaffold_level=scaffold,
        topic=topic,
        difficulty=int(row["difficulty"]),
        problem=str(row["problem"]),
        ground_truth=str(row["ground_truth"]),
        ground_truth_steps=tuple(row.get("ground_truth_steps") or ()),
        common_wrong=tuple(row.get("common_wrong") or ()),
        scaffold_hints=tuple(row.get("scaffold_hints") or ()),
        reconsolidation_links=tuple(row.get("reconsolidation_links") or ()),
        notes=row.get("notes"),
    )


def _validate_pilot_pool(pilot: tuple[PilotProblem, ...], *, source: Path) -> None:
    if len(pilot) != 20:
        raise ValueError(
            f"{source}: pilot pool must have exactly 20 problems, got {len(pilot)}"
        )
    by_week: dict[int, list[PilotProblem]] = {}
    ids = set()
    for p in pilot:
        if p.id in ids:
            raise ValueError(f"{source}: duplicate pilot problem id {p.id!r}")
        ids.add(p.id)
        by_week.setdefault(p.week, []).append(p)
    for week in (1, 2, 3, 4):
        entries = by_week.get(week, [])
        if len(entries) != 5:
            raise ValueError(
                f"{source}: week {week} must have exactly 5 problems, got {len(entries)}"
            )
        allowed = _EXPECTED_SCAFFOLD_BY_WEEK[week]
        for p in entries:
            if p.scaffold_level not in allowed:
                raise ValueError(
                    f"{source}: pilot {p.id!r} scaffold_level={p.scaffold_level.value!r}"
                    f" is not permitted in week {week}; allowed={[s.value for s in allowed]}"
                )
    for p in pilot:
        if p.scaffold_level is ScaffoldLevel.RECONSOLIDATION_PAIR:
            if not p.reconsolidation_links:
                raise ValueError(
                    f"{source}: pilot {p.id!r} is reconsolidation_pair but names no links"
                )
            for link in p.reconsolidation_links:
                if link not in ids:
                    raise ValueError(
                        f"{source}: pilot {p.id!r} links to unknown problem {link!r}"
                    )
