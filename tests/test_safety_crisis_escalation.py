"""Phase 4a — CrisisEscalation unit tests.

Non-negotiable: zero false negatives on the explicit-ideation + imminent-danger
cue banks. False positives are a cost; false negatives are a safety failure.
"""

from __future__ import annotations

from psychology_base.applications.safety import (
    CrisisEscalation,
    CrisisIncident,
    CrisisSeverity,
)


def _collector() -> tuple[list[CrisisIncident], CrisisEscalation]:
    log: list[CrisisIncident] = []
    detector = CrisisEscalation(handoff_sink=log.append)
    return log, detector


def test_benign_utterance_does_not_fire() -> None:
    log, detector = _collector()
    assert detector.scan("I had a nice walk today.", 0.0) is None
    assert log == []


def test_explicit_ideation_en_fires() -> None:
    log, detector = _collector()
    inc = detector.scan("Sometimes I just want to die.", 1.0)
    assert inc is not None
    assert inc.severity is CrisisSeverity.EXPLICIT_IDEATION
    assert len(log) == 1


def test_explicit_ideation_zh_fires() -> None:
    log, detector = _collector()
    inc = detector.scan("最近真的太累了，我想死。", 2.0)
    assert inc is not None
    assert inc.severity is CrisisSeverity.EXPLICIT_IDEATION
    assert len(log) == 1


def test_imminent_danger_en_fires_at_highest_severity() -> None:
    log, detector = _collector()
    inc = detector.scan("Tonight I will end it.", 3.0)
    assert inc is not None
    assert inc.severity is CrisisSeverity.IMMINENT_DANGER
    assert len(log) == 1


def test_imminent_danger_zh_fires() -> None:
    log, detector = _collector()
    inc = detector.scan("今晚我就了结一切。", 4.0)
    assert inc is not None
    assert inc.severity is CrisisSeverity.IMMINENT_DANGER


def test_ambiguous_fires_at_ambiguous_severity() -> None:
    log, detector = _collector()
    inc = detector.scan("I feel like I can't go on anymore.", 5.0)
    assert inc is not None
    assert inc.severity is CrisisSeverity.AMBIGUOUS


def test_higher_severity_wins_when_multiple_cues_present() -> None:
    log, detector = _collector()
    inc = detector.scan("I can't go on. Tonight I will end it.", 6.0)
    assert inc is not None
    assert inc.severity is CrisisSeverity.IMMINENT_DANGER


def test_case_insensitive() -> None:
    log, detector = _collector()
    inc = detector.scan("I WANT TO DIE.", 7.0)
    assert inc is not None
    assert inc.severity is CrisisSeverity.EXPLICIT_IDEATION


def test_handoff_sink_called_once_per_incident() -> None:
    log, detector = _collector()
    detector.scan("I want to die.", 1.0)
    detector.scan("Having a decent afternoon.", 2.0)
    detector.scan("Suicide is on my mind.", 3.0)
    assert len(log) == 2
    assert all(isinstance(x, CrisisIncident) for x in log)


def test_incident_log_mirrors_sink() -> None:
    log, detector = _collector()
    detector.scan("I want to die.", 1.0)
    detector.scan("Tonight I will end it.", 2.0)
    assert len(detector.incident_log()) == 2
    assert detector.incident_log()[-1].severity is CrisisSeverity.IMMINENT_DANGER
