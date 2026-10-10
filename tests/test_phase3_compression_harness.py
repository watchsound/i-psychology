"""Phase 3 E0.8a — compression harness unit tests.

Covers the two primitives the month-horizon stress test needs to span
virtual days/weeks between scripted anchor turns:

- ``CompanionSystem.advance_virtual_time(ticks)`` — fast-forwards N sleep
  ticks (semantically identical to a ``for _ in range(ticks): tick()``
  loop, but makes compression intent explicit).
- ``AdvanceTurn`` scripted-scenario type — the YAML-level primitive that
  ``SessionPlayer`` dispatches to ``advance_virtual_time`` on companion
  memories and falls back to repeated ``tick()`` on baselines (EchoMemory)
  that don't expose the method.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.baselines import EchoMemory
from psychology_base.eval.scripts.schema import (
    AdvanceTurn,
    ScriptedScenario,
    Session,
)
from psychology_base.eval.session_player import SessionPlayer


def _at(hour: int = 9) -> datetime:
    return datetime(2026, 7, 1, hour, 0, tzinfo=timezone.utc)


def test_advance_virtual_time_increments_sleep_ticks():
    sys = CompanionSystem(byte_cap=None, witness_active=False)
    before = sys.heartbeat.sleep_ticks
    sys.advance_virtual_time(5)
    assert sys.heartbeat.sleep_ticks == before + 5


def test_advance_virtual_time_zero_is_noop():
    sys = CompanionSystem(byte_cap=None, witness_active=False)
    before = sys.heartbeat.sleep_ticks
    sys.advance_virtual_time(0)
    assert sys.heartbeat.sleep_ticks == before


def test_advance_virtual_time_rejects_negative():
    sys = CompanionSystem(byte_cap=None, witness_active=False)
    with pytest.raises(ValueError):
        sys.advance_virtual_time(-1)


def test_advance_virtual_time_matches_tick_loop():
    """Compression primitive is a semantic no-op vs an explicit tick loop
    on sleep_ticks — the only distinction is scenario legibility."""
    a = CompanionSystem(byte_cap=None, witness_active=False)
    b = CompanionSystem(byte_cap=None, witness_active=False)
    a.advance_virtual_time(10)
    for _ in range(10):
        b.tick()
    assert a.heartbeat.sleep_ticks == b.heartbeat.sleep_ticks


def _advance_scenario(ticks: int) -> ScriptedScenario:
    return ScriptedScenario(
        id="phase3-compression-harness",
        benchmark="continuity",
        description="E0.8a compression primitive round-trip",
        sessions=[
            Session(
                id="s1",
                description="single advance turn",
                turns=[AdvanceTurn(type="advance", at=_at(), ticks=ticks)],
            )
        ],
    )


def test_advance_turn_dispatches_to_companion_method():
    sys = CompanionSystem(byte_cap=None, witness_active=False)
    before = sys.heartbeat.sleep_ticks
    player = SessionPlayer(sys)
    player.play(_advance_scenario(7))
    assert sys.heartbeat.sleep_ticks == before + 7


def test_advance_turn_falls_back_to_tick_on_baseline():
    """EchoMemory has ``tick()`` but no ``advance_virtual_time`` — the
    player must fan out to N ``tick()`` calls so B1/B2/B3-style scorers
    that compose compression + baseline reader still run."""
    memory = EchoMemory()
    tick_calls = {"n": 0}
    original_tick = memory.tick

    def counting_tick() -> None:
        tick_calls["n"] += 1
        return original_tick()

    memory.tick = counting_tick  # type: ignore[method-assign]
    player = SessionPlayer(memory)
    player.play(_advance_scenario(4))
    assert tick_calls["n"] == 4


def test_advance_turn_schema_rejects_negative_ticks():
    with pytest.raises(ValueError):
        AdvanceTurn(type="advance", at=_at(), ticks=-1)
