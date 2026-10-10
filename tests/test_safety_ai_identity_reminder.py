"""Phase 4a — AIIdentityReminder unit tests.

《办法》 non-negotiable: prolonged sessions surface AI-identity reminders every
2 h of interaction time. Idle gaps do not advance the clock; explicit
disclosure resets the counter.
"""

from __future__ import annotations

from psychology_base.applications.safety import AIIdentityReminder


def test_does_not_fire_without_user_activity() -> None:
    reminder = AIIdentityReminder()
    assert reminder.should_fire() is False
    assert reminder.interaction_seconds_since_disclosure() == 0.0


def test_fires_after_two_hours_of_continuous_interaction() -> None:
    reminder = AIIdentityReminder()
    start = 0.0
    reminder.record_user_turn(start)
    # One turn every 60 s for just over 2 h.
    for i in range(1, 121):
        reminder.record_user_turn(start + i * 60.0)
    assert reminder.should_fire() is True


def test_does_not_fire_at_two_hours_minus_one_second() -> None:
    reminder = AIIdentityReminder(idle_threshold_seconds=120)
    reminder.record_user_turn(0.0)
    reminder.record_user_turn(2 * 60 * 60 - 1)  # single 7199s gap > idle threshold
    # The 7199s gap exceeds idle_threshold_seconds=120, so it is NOT counted
    # as interaction time. The reminder should NOT fire.
    assert reminder.should_fire() is False


def test_idle_gap_does_not_advance_the_clock() -> None:
    reminder = AIIdentityReminder(idle_threshold_seconds=60)
    # 10 turns 30s apart — accumulates 9 * 30 = 270s.
    for i in range(10):
        reminder.record_user_turn(i * 30.0)
    assert reminder.interaction_seconds_since_disclosure() == 270.0

    # Then a 5-minute gap — idle_threshold=60s, so gap is NOT counted.
    reminder.record_user_turn(10 * 30.0 + 5 * 60.0)
    assert reminder.interaction_seconds_since_disclosure() == 270.0

    # Resume regular cadence — the next 30s gap IS counted.
    reminder.record_user_turn(10 * 30.0 + 5 * 60.0 + 30.0)
    assert reminder.interaction_seconds_since_disclosure() == 300.0


def test_disclosure_resets_counter() -> None:
    reminder = AIIdentityReminder()
    reminder.record_user_turn(0.0)
    for i in range(1, 121):
        reminder.record_user_turn(i * 60.0)
    assert reminder.should_fire() is True

    reminder.record_identity_disclosure(121 * 60.0)
    assert reminder.should_fire() is False
    assert reminder.interaction_seconds_since_disclosure() == 0.0


def test_seconds_until_next_reminder_counts_down() -> None:
    reminder = AIIdentityReminder()
    assert reminder.seconds_until_next_reminder() == 2 * 60 * 60

    reminder.record_user_turn(0.0)
    # Add 1 h of interaction (60 turns, 60 s apart).
    for i in range(1, 61):
        reminder.record_user_turn(i * 60.0)
    # 60 min accumulated → 60 min remaining.
    assert reminder.seconds_until_next_reminder() == 60 * 60.0


def test_configurable_interval() -> None:
    reminder = AIIdentityReminder(reminder_interval_seconds=30 * 60)
    for i in range(31):
        reminder.record_user_turn(i * 60.0)
    assert reminder.should_fire() is True
