"""Phase 1 build step 1 gate — substrate plumbing.

Per docs/phases/phase-1-plan.md §5, step 1 gate is:
  - Fragments observable + inspectable
  - Tick advances state
  - measure_v_metrics runs without errors

V-metric band assertions are deliberately NOT here; those are the step-2 (V1/V2)
and step-4 (V4) gates. Step 1 is pure plumbing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psychology_base.companion import CompanionSystem
from psychology_base.eval.adapters import MemorySystem, Observation
from psychology_base.eval.v_metrics import measure_v_metrics


def _obs(content: str, offset_days: int = 0, tags: tuple[str, ...] = ()) -> Observation:
    base = datetime(2026, 1, 15, 19, 0, tzinfo=timezone.utc)
    return Observation(
        timestamp=base + timedelta(days=offset_days),
        speaker="user",
        content=content,
        tags=tags,
    )


def test_satisfies_memory_system_protocol():
    assert isinstance(CompanionSystem(), MemorySystem)


def test_observe_adds_fragment_and_wakes_heartbeat():
    sys = CompanionSystem()
    sys.observe(_obs("my father was never there", tags=("father",)))
    state = sys.inspect()
    assert state["fragment_count"] == 1
    assert state["witness_fragment_count"] == 1
    assert state["wake_ticks"] == 1
    assert state["sleep_ticks"] == 0


def test_tick_advances_sleep_counter_without_touching_wake():
    sys = CompanionSystem()
    sys.tick()
    sys.tick()
    state = sys.inspect()
    assert state["sleep_ticks"] == 2
    assert state["wake_ticks"] == 0


def test_fragments_are_inspectable_with_id_content_tags():
    sys = CompanionSystem()
    sys.observe(_obs("my father was overwhelmed", tags=("father", "reinterpretation")))
    state = sys.inspect()
    assert len(state["fragments"]) == 1
    f = state["fragments"][0]
    # M1 step 2 — fragment ids are content-addressed BLAKE2b-128 hex
    # digests (32 chars). Pre-M1 tick-indexed "frag-NNNN" format is gone.
    assert isinstance(f["id"], str)
    assert len(f["id"]) == 32
    assert all(c in "0123456789abcdef" for c in f["id"])
    assert f["content"] == "my father was overwhelmed"
    assert "father" in f["tags"]
    assert "reinterpretation" in f["tags"]


def test_measure_v_metrics_runs_end_to_end():
    """The V-metric harness must run without errors against the new substrate.
    Band assertions are step-2/step-4 gates; here we only check the harness drives
    the Protocol correctly and returns numeric values in [0, 1]."""
    scenario = [
        _obs("my father was never there on Friday evenings", tags=("father",)),
        _obs(
            "I think my father was overwhelmed, trying in his way",
            offset_days=90,
            tags=("father", "reinterpretation"),
        ),
    ]
    m = measure_v_metrics(
        memory_factory=CompanionSystem,
        probe_cue="father",
        scenario_observations=scenario,
    )
    assert 0.0 <= m.v1_state_continuity <= 1.0
    assert 0.0 <= m.v2_adjacent_jaccard <= 1.0
    assert 0.0 <= m.v2_longhorizon_jaccard <= 1.0
    assert 0.0 <= m.v3_mentalistic_liveness <= 1.0
    # V4 + V5 remain N/A — Witness coupling and rule mechanism arrive in steps 4 / phase 2.
    assert m.v4_mirror_ablation is None
    assert m.v5_rule_internalization is None


def test_recall_metadata_exposes_step_and_tick_counters():
    """Recall metadata must expose the step number and heartbeat counters so
    later debugging can confirm which mechanism answered. The step number tracks
    the plan: step 1 = substring, step 2+ = reconstruction."""
    sys = CompanionSystem()
    sys.observe(_obs("my father was never there", tags=("father",)))
    sys.tick()
    r = sys.recall("father")
    assert r.metadata["step"] >= 1
    assert r.metadata["wake_ticks"] == 1
    assert r.metadata["sleep_ticks"] == 1
    assert "father" in r.content.lower()
