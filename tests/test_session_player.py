from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from psychology_base.eval.adapters import Observation
from psychology_base.eval.baselines import EchoMemory
from psychology_base.eval.scripts.schema import ScriptedScenario
from psychology_base.eval.session_player import SessionPlayer, load_scenario


SAMPLE_YAML = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "scripts"
    / "b1_narrative_sample.yaml"
)


def test_sample_yaml_parses_as_b1_scenario():
    scenario = load_scenario(SAMPLE_YAML)
    assert isinstance(scenario, ScriptedScenario)
    assert scenario.benchmark == "B1"
    assert len(scenario.sessions) == 3
    assert [s.id for s in scenario.sessions] == ["s1", "s2", "s3"]


def test_echo_memory_end_to_end():
    scenario = load_scenario(SAMPLE_YAML)
    memory = EchoMemory()
    player = SessionPlayer(memory)
    transcript = player.play(scenario)

    # Three recall turns across s1, s2, s3.
    assert len(transcript) == 3

    # s1 recall sees only the harsh framing.
    assert "never there" in transcript[0].content.lower()

    # s2 recall sees both — EchoMemory returns most recent (reinterpretation framing).
    assert "overwhelmed" in transcript[1].content.lower()

    # s3 recall against the probe cue: EchoMemory still returns the most recent
    # observation containing "father" — the reinterpretation. The baseline cannot
    # produce a rendering that carries both framings; that's the point.
    assert "overwhelmed" in transcript[2].content.lower()

    # Fragment-count-based metrics can inspect the baseline.
    inspection = memory.inspect()
    assert inspection["fragment_count"] == 2


def test_echo_memory_handles_unmatched_cue():
    memory = EchoMemory()
    memory.observe(
        Observation(
            timestamp=datetime(2026, 1, 15, 19, 0, tzinfo=timezone.utc),
            speaker="user",
            content="A sentence about something else entirely.",
        )
    )
    recall = memory.recall("father")
    assert recall.content == ""
    assert recall.metadata["match_count"] == 0
    assert recall.fragments_used == ()


def test_tick_is_a_noop_for_echo():
    memory = EchoMemory()
    memory.tick()  # must not raise
    assert memory.inspect()["fragment_count"] == 0
