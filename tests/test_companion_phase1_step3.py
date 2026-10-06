"""Phase 1 build step 3 gate — S1 cap + S7 Hebbian + S5 covenant pins.

Per docs/phases/phase-1-plan.md §5, step 3 gate is:
  - V2 long-horizon in band (0.20 – 0.35)
  - Top-5% edges hold < 50% of edge-weight budget at ep20 (saturation guard working)

Additional behavioural checks exercised here (foundation for later gates, not
blocking step-3 exit):
  - Eviction kicks in when byte cap breached; LRU-by-weakness ordering.
  - Covenant-tagged fragments immune from eviction (S5 pin).
  - Hebbian bump reinforces co-activated pairs across repeated recalls.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

from psychology_base.companion import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.v_metrics import measure_v_metrics


BASE = datetime(2026, 1, 15, 19, 0, tzinfo=timezone.utc)


def _scenario() -> list[Observation]:
    """Same father/sister/father scenario used at step 2 — the step-3 gate is
    whether the new selection-pressure layer leaves V2 long-horizon in band."""
    return [
        Observation(
            timestamp=BASE,
            speaker="user",
            content="my father was never there on Friday evenings always working late",
            tags=("father", "childhood", "self"),
        ),
        Observation(
            timestamp=BASE + timedelta(days=30),
            speaker="user",
            content="my sister called me yesterday about the holiday plans",
            tags=("sister",),
        ),
        Observation(
            timestamp=BASE + timedelta(days=90),
            speaker="user",
            content="I think my father was overwhelmed trying in his way",
            tags=("father", "childhood", "reinterpretation", "other"),
        ),
    ]


def _median_v2_longhorizon() -> float:
    v2l = []
    for seed in (0, 1, 2):
        m = measure_v_metrics(
            memory_factory=lambda s=seed: CompanionSystem(seed=s),
            probe_cue="father",
            scenario_observations=_scenario(),
        )
        v2l.append(m.v2_longhorizon_jaccard)
    return statistics.median(v2l)


def test_v2_longhorizon_in_band_under_step3_selection_pressure():
    """V2 long-horizon band 0.20 – 0.35. Hebbian reinforcement would collapse
    this toward 1.0 if the edge decay / saturation guard weren't tuned; step 3
    must preserve the step-2 drift behaviour while adding selection pressure."""
    v2l = _median_v2_longhorizon()
    assert 0.20 <= v2l <= 0.35, f"V2 long-horizon out of band at step 3: {v2l}"


def test_saturation_guard_keeps_top5_share_under_half_at_ep20():
    """Top 5% of edges must hold < 50% of total edge-weight budget after 20
    observe/recall/tick cycles. Without the saturation guard, Hebbian bumps
    on a hot pair would monopolise the budget; this is the Ouroboros-inherited
    open problem the step-3 design explicitly closes."""
    sys = CompanionSystem(seed=0, byte_cap=20_000)
    for i in range(20):
        content = (
            "my father was overwhelmed on Friday evenings"
            if i % 2 == 0
            else "my mother worried about father on holidays"
        )
        sys.observe(Observation(
            timestamp=BASE + timedelta(days=i * 5),
            speaker="user",
            content=content,
            tags=("father",),
        ))
        sys.recall("father")
        sys.tick()
    state = sys.inspect()
    assert state["edge_top5_share"] < 0.5, (
        f"saturation guard failed: top-5% share = {state['edge_top5_share']:.3f}"
    )


def test_byte_cap_evicts_weakest_non_covenant_fragments():
    """When total bytes exceed the cap, weakest non-covenant fragments are
    evicted (LRU-by-incident-edge-weight, ties → oldest). The covenant-tagged
    fragment must survive even when it is the oldest."""
    sys = CompanionSystem(seed=0, byte_cap=500)
    for i in range(15):
        tags = ("covenant", "rule") if i == 0 else ("event",)
        sys.observe(Observation(
            timestamp=BASE + timedelta(days=i * 2),
            speaker="user",
            content=f"event {i} content words words words words words",
            tags=tags,
        ))
    state = sys.inspect()
    kept_ids = [f["id"] for f in state["fragments"]]
    assert "frag-0000" in kept_ids, "S5 covenant pin failed — covenant fragment was evicted"
    assert state["fragment_count"] < 15, "S1 cap failed — no eviction happened"
    assert state["total_bytes"] <= 500, (
        f"S1 cap failed — bytes {state['total_bytes']} exceeded cap"
    )


def test_hebbian_edge_grows_on_repeated_co_activation():
    """Repeated recall of the same cue should strengthen the edge between
    co-activated fragments. The signed-direction check: edge weight after
    5 recalls > edge weight after 1 recall for the top pair."""
    sys = CompanionSystem(seed=0, byte_cap=20_000)
    for obs in _scenario():
        sys.observe(obs)
    sys.recall("father")
    edges_after_1 = max(sys.edges.weights.values()) if sys.edges.weights else 0.0
    for _ in range(5):
        sys.recall("father")
    edges_after_6 = max(sys.edges.weights.values()) if sys.edges.weights else 0.0
    assert edges_after_6 > edges_after_1, (
        f"S7 Hebbian failed — top edge did not grow: {edges_after_1} -> {edges_after_6}"
    )


def test_recall_metadata_exposes_edge_telemetry():
    """Step-3 metadata must expose edge count + top-5% share so W3 instrumentation
    at step 4 can log the saturation signal without needing extra plumbing."""
    sys = CompanionSystem(seed=0)
    for obs in _scenario():
        sys.observe(obs)
    r = sys.recall("father")
    assert r.metadata["step"] >= 3
    assert "edge_count" in r.metadata
    assert "edge_top5_share" in r.metadata
