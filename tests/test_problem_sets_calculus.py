"""Problem-set loader + calculus-v1 structural checks.

Live-fire-adjacent: these tests load the real `calculus_v1.yaml` and
assert the §9.3 fade schedule, topic distribution, reconsolidation
pairing, and license fields are all intact. A content regression (e.g.
someone edits the YAML and breaks the Week 3 fade) breaks these tests
before it reaches the pilot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from psychology_base.applications.learning.problem_sets import (
    DEFAULT_CALCULUS_V1_PATH,
    ScaffoldLevel,
    WeekTopic,
    load_problem_set,
)
from psychology_base.applications.learning.problem_sets.loader import (
    _parse_problem_set,
)


# ---------------------------------------------------------------------------
# calculus-v1 structural shape
# ---------------------------------------------------------------------------


def test_calculus_v1_loads_without_error() -> None:
    ps = load_problem_set()
    assert ps.id == "calculus-v1"
    assert "CC0" in ps.license, "License must be CC0 to avoid CC BY-SA propagation."


def test_calibration_pool_has_20_problems_evenly_distributed() -> None:
    ps = load_problem_set()
    assert len(ps.calibration) == 20
    counts = {t: len(ps.calibration_by_topic(t)) for t in WeekTopic}
    assert counts == {
        WeekTopic.POWER_RULE: 4,
        WeekTopic.CHAIN_RULE: 4,
        WeekTopic.MIXED_RULES: 4,
        WeekTopic.INTEGRATION: 4,
        WeekTopic.NOVEL_TRANSFER: 4,
    }


def test_pilot_pool_structure_matches_signoff_93() -> None:
    ps = load_problem_set()
    assert len(ps.pilot) == 20
    for week in (1, 2, 3, 4):
        assert len(ps.pilot_by_week(week)) == 5

    # Week 1: all FULL.
    assert all(p.scaffold_level is ScaffoldLevel.FULL for p in ps.pilot_by_week(1))

    # Week 2: positions 1,2 FULL; 3,4,5 PARTIAL.
    w2 = {p.position_in_week: p.scaffold_level for p in ps.pilot_by_week(2)}
    assert w2[1] is ScaffoldLevel.FULL
    assert w2[2] is ScaffoldLevel.FULL
    assert w2[3] is ScaffoldLevel.PARTIAL
    assert w2[4] is ScaffoldLevel.PARTIAL
    assert w2[5] is ScaffoldLevel.PARTIAL

    # Week 3: positions 1,2 PARTIAL; 3,4,5 FOCUS.
    w3 = {p.position_in_week: p.scaffold_level for p in ps.pilot_by_week(3)}
    assert w3[1] is ScaffoldLevel.PARTIAL
    assert w3[2] is ScaffoldLevel.PARTIAL
    assert w3[3] is ScaffoldLevel.FOCUS
    assert w3[4] is ScaffoldLevel.FOCUS
    assert w3[5] is ScaffoldLevel.FOCUS

    # Week 4: positions 1,2,3 UNAIDED; 4,5 RECONSOLIDATION_PAIR.
    w4 = {p.position_in_week: p.scaffold_level for p in ps.pilot_by_week(4)}
    assert w4[1] is ScaffoldLevel.UNAIDED
    assert w4[2] is ScaffoldLevel.UNAIDED
    assert w4[3] is ScaffoldLevel.UNAIDED
    assert w4[4] is ScaffoldLevel.RECONSOLIDATION_PAIR
    assert w4[5] is ScaffoldLevel.RECONSOLIDATION_PAIR


def test_week_topics_match_signoff() -> None:
    ps = load_problem_set()
    assert all(p.topic is WeekTopic.POWER_RULE for p in ps.pilot_by_week(1))
    assert all(p.topic is WeekTopic.CHAIN_RULE for p in ps.pilot_by_week(2))
    assert all(p.topic is WeekTopic.MIXED_RULES for p in ps.pilot_by_week(3))
    assert all(p.topic is WeekTopic.INTEGRATION for p in ps.pilot_by_week(4))


def test_reconsolidation_pairs_link_to_week1() -> None:
    ps = load_problem_set()
    pairs = [p for p in ps.pilot if p.scaffold_level is ScaffoldLevel.RECONSOLIDATION_PAIR]
    assert len(pairs) == 2
    for pair in pairs:
        assert pair.reconsolidation_links, f"{pair.id} missing links"
        for link_id in pair.reconsolidation_links:
            linked = ps.pilot_by_id(link_id)
            assert linked.week == 1, (
                f"{pair.id} links to {link_id} which is in week {linked.week}, not 1"
            )


def test_every_problem_has_ground_truth_and_steps() -> None:
    ps = load_problem_set()
    for p in ps.calibration + ps.pilot:
        assert p.ground_truth.strip(), f"{p.id} missing ground_truth"
        assert p.ground_truth_steps, f"{p.id} missing ground_truth_steps"


def test_pilot_problems_have_at_least_one_scaffold_hint() -> None:
    """Scaffold hints are the ladder the tutor delivers. UNAIDED problems
    still carry a surrender-recovery hint so the tutor has material if the
    learner explicitly asks for help."""
    ps = load_problem_set()
    for p in ps.pilot:
        assert p.scaffold_hints, f"{p.id} has no scaffold_hints"


# ---------------------------------------------------------------------------
# Loader rejects malformed input
# ---------------------------------------------------------------------------


def _minimal_valid_payload() -> dict:
    """Lift the real pilot pool shape, which is what structural validation
    guards. Calibration can be empty — validation only enforces pilot-pool
    structure."""
    pilot = []
    for week in (1, 2, 3, 4):
        for pos in (1, 2, 3, 4, 5):
            scaffold = {
                1: "full",
                2: "full" if pos <= 2 else "partial",
                3: "partial" if pos <= 2 else "focus",
                4: "unaided" if pos <= 3 else "reconsolidation_pair",
            }[week]
            topic = {1: "power_rule", 2: "chain_rule", 3: "mixed_rules", 4: "integration"}[week]
            entry = {
                "id": f"w{week}p{pos}",
                "week": week,
                "position_in_week": pos,
                "scaffold_level": scaffold,
                "topic": topic,
                "difficulty": 1,
                "problem": "placeholder",
                "ground_truth": "placeholder",
                "ground_truth_steps": ["step"],
                "scaffold_hints": ["hint"],
            }
            if scaffold == "reconsolidation_pair":
                entry["reconsolidation_links"] = ["w1p1"]
            pilot.append(entry)
    return {
        "id": "test",
        "license": "CC0",
        "provenance": {},
        "calibration": [],
        "pilot": pilot,
    }


def test_loader_rejects_wrong_pilot_pool_size(tmp_path: Path) -> None:
    bad = _minimal_valid_payload()
    bad["pilot"].pop()  # 19, not 20
    with pytest.raises(ValueError, match="exactly 20"):
        _parse_problem_set(bad, source=tmp_path / "fake.yaml")


def test_loader_rejects_wrong_week_count(tmp_path: Path) -> None:
    bad = _minimal_valid_payload()
    bad["pilot"][0]["week"] = 2  # week 1 now has 4; week 2 has 6
    with pytest.raises(ValueError, match="week 1 must have exactly 5"):
        _parse_problem_set(bad, source=tmp_path / "fake.yaml")


def test_loader_rejects_wrong_scaffold_for_week(tmp_path: Path) -> None:
    bad = _minimal_valid_payload()
    bad["pilot"][0]["scaffold_level"] = "focus"  # week 1 cannot be focus
    with pytest.raises(ValueError, match="not permitted in week 1"):
        _parse_problem_set(bad, source=tmp_path / "fake.yaml")


def test_loader_rejects_reconsolidation_without_links(tmp_path: Path) -> None:
    bad = _minimal_valid_payload()
    # Clear the links on w4p4 / w4p5.
    for entry in bad["pilot"]:
        if entry["scaffold_level"] == "reconsolidation_pair":
            entry["reconsolidation_links"] = []
    with pytest.raises(ValueError, match="names no links"):
        _parse_problem_set(bad, source=tmp_path / "fake.yaml")


def test_loader_rejects_unknown_reconsolidation_link(tmp_path: Path) -> None:
    bad = _minimal_valid_payload()
    for entry in bad["pilot"]:
        if entry["scaffold_level"] == "reconsolidation_pair":
            entry["reconsolidation_links"] = ["ghost-id"]
    with pytest.raises(ValueError, match="unknown problem"):
        _parse_problem_set(bad, source=tmp_path / "fake.yaml")


def test_loader_rejects_invalid_topic(tmp_path: Path) -> None:
    bad = _minimal_valid_payload()
    bad["pilot"][0]["topic"] = "not_a_topic"
    with pytest.raises(ValueError, match="not a valid WeekTopic"):
        _parse_problem_set(bad, source=tmp_path / "fake.yaml")


def test_loader_rejects_duplicate_pilot_id(tmp_path: Path) -> None:
    bad = _minimal_valid_payload()
    bad["pilot"][1]["id"] = bad["pilot"][0]["id"]
    with pytest.raises(ValueError, match="duplicate pilot problem id"):
        _parse_problem_set(bad, source=tmp_path / "fake.yaml")


# ---------------------------------------------------------------------------
# Round-trip sanity on the default path
# ---------------------------------------------------------------------------


def test_default_calculus_v1_path_exists() -> None:
    assert DEFAULT_CALCULUS_V1_PATH.exists(), (
        f"calculus_v1.yaml missing at {DEFAULT_CALCULUS_V1_PATH}"
    )
