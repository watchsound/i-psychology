"""Phase 4a — ExtremeEmotionWatcher unit tests.

《办法》-adjacent: beyond acute crisis cues, prolonged
negative-valence-high-arousal patterns across ≥ 3 sessions warrant a flag.
"""

from __future__ import annotations

from psychology_base.applications.safety import (
    ExtremeEmotionFlag,
    ExtremeEmotionWatcher,
    SessionAffect,
)


def _collector() -> tuple[list[ExtremeEmotionFlag], ExtremeEmotionWatcher]:
    log: list[ExtremeEmotionFlag] = []
    watcher = ExtremeEmotionWatcher(review_sink=log.append, user_id="u1")
    return log, watcher


def _affect(session_id: str, valence: float, arousal: float) -> SessionAffect:
    return SessionAffect(
        session_id=session_id,
        timestamp_seconds=0.0,
        valence=valence,
        arousal=arousal,
    )


def test_single_session_in_quadrant_does_not_fire() -> None:
    log, watcher = _collector()
    assert watcher.record(_affect("s1", -0.6, 0.5)) is None
    assert watcher.current_streak_length() == 1
    assert log == []


def test_two_sessions_in_quadrant_do_not_fire() -> None:
    log, watcher = _collector()
    watcher.record(_affect("s1", -0.6, 0.5))
    assert watcher.record(_affect("s2", -0.7, 0.6)) is None
    assert log == []


def test_three_sessions_in_quadrant_fire() -> None:
    log, watcher = _collector()
    watcher.record(_affect("s1", -0.6, 0.5))
    watcher.record(_affect("s2", -0.7, 0.6))
    flag = watcher.record(_affect("s3", -0.5, 0.8))
    assert flag is not None
    assert flag.consecutive_sessions == 3
    assert flag.first_session_id == "s1"
    assert flag.last_session_id == "s3"
    assert flag.user_id == "u1"
    assert len(log) == 1


def test_streak_breaks_on_out_of_quadrant_session() -> None:
    log, watcher = _collector()
    watcher.record(_affect("s1", -0.6, 0.5))
    watcher.record(_affect("s2", -0.7, 0.6))
    # Neutral session breaks the streak.
    assert watcher.record(_affect("s3", 0.1, 0.2)) is None
    assert watcher.current_streak_length() == 0
    # Need three NEW consecutive sessions in-quadrant to fire.
    watcher.record(_affect("s4", -0.6, 0.5))
    watcher.record(_affect("s5", -0.7, 0.6))
    assert watcher.record(_affect("s6", -0.5, 0.8)) is not None
    assert len(log) == 1


def test_does_not_fire_on_low_arousal_depression() -> None:
    # Low-arousal (apathetic/withdrawn) depression is outside the
    # "distressed + agitated" quadrant this watcher targets.
    log, watcher = _collector()
    watcher.record(_affect("s1", -0.6, -0.2))
    watcher.record(_affect("s2", -0.7, -0.3))
    assert watcher.record(_affect("s3", -0.5, -0.1)) is None
    assert log == []


def test_does_not_fire_on_positive_valence() -> None:
    log, watcher = _collector()
    watcher.record(_affect("s1", 0.6, 0.7))
    watcher.record(_affect("s2", 0.7, 0.6))
    assert watcher.record(_affect("s3", 0.5, 0.8)) is None


def test_boundary_at_valence_ceiling() -> None:
    log, watcher = _collector()
    # valence = -0.4 exactly is NOT below ceiling -0.4.
    watcher.record(_affect("s1", -0.4, 0.5))
    assert watcher.current_streak_length() == 0


def test_configurable_threshold() -> None:
    log: list[ExtremeEmotionFlag] = []
    watcher = ExtremeEmotionWatcher(
        review_sink=log.append,
        threshold_sessions=2,
        user_id="u2",
    )
    watcher.record(_affect("s1", -0.6, 0.5))
    flag = watcher.record(_affect("s2", -0.7, 0.6))
    assert flag is not None
    assert flag.consecutive_sessions == 2


def test_does_not_double_fire_on_same_last_session() -> None:
    log, watcher = _collector()
    watcher.record(_affect("s1", -0.6, 0.5))
    watcher.record(_affect("s2", -0.7, 0.6))
    watcher.record(_affect("s3", -0.5, 0.8))
    assert len(log) == 1
    # Re-recording the same s3 does not re-fire.
    watcher.record(_affect("s3", -0.5, 0.8))
    assert len(log) == 1


def test_does_not_re_fire_on_next_in_streak_session() -> None:
    log, watcher = _collector()
    watcher.record(_affect("s1", -0.6, 0.5))
    watcher.record(_affect("s2", -0.7, 0.6))
    watcher.record(_affect("s3", -0.5, 0.8))  # fires
    watcher.record(_affect("s4", -0.6, 0.7))  # still in quadrant → no re-fire
    watcher.record(_affect("s5", -0.5, 0.6))
    assert len(log) == 1


def test_re_fires_after_break_and_new_streak() -> None:
    log, watcher = _collector()
    watcher.record(_affect("s1", -0.6, 0.5))
    watcher.record(_affect("s2", -0.7, 0.6))
    watcher.record(_affect("s3", -0.5, 0.8))  # fires
    watcher.record(_affect("s4", 0.1, 0.0))  # breaks streak
    watcher.record(_affect("s5", -0.6, 0.5))
    watcher.record(_affect("s6", -0.7, 0.6))
    watcher.record(_affect("s7", -0.5, 0.8))  # fires again
    assert len(log) == 2
