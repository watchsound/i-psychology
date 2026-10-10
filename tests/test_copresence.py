"""Unit tests for the Phase 9+ substrate witness primitive.

Covers:
- :class:`CopresenceScope` + :func:`filter_fragments_by_copresence`
- :func:`CompanionSystem.recall(..., copresence=...)` wiring
- HiToM-adapter parity: ingest_hitom_story_as_observations + substrate
  co-presence filter produces the SAME witnessed-event set as the
  scorer-local :func:`events_witnessed_by_chain` on the Phase 8 Pillar E
  worked example.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from psychology_base.companion.copresence import (
    CopresenceScope,
    filter_fragments_by_copresence,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation, make_fragment
from psychology_base.eval.benchmarks.public_datasets.hitom_witness import (
    events_witnessed_by_chain,
    ingest_hitom_story_as_observations,
    parse_hitom_story,
)


# ---------------------------------------------------------------------------
# CopresenceScope + filter_fragments_by_copresence — unit tests
# ---------------------------------------------------------------------------


def _frag(content: str, tags: tuple[str, ...]):
    return make_fragment(
        content=content,
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        tags=tags,
    )


def test_copresence_scope_order_zero_passes_everything():
    frags = [
        _frag("x", ("witness:Alice",)),
        _frag("y", ("witness:Bob",)),
        _frag("z", ()),
    ]
    kept = filter_fragments_by_copresence(frags, CopresenceScope(chain=()))
    assert kept == frags


def test_copresence_scope_order_one_filters_to_single_agent():
    a = _frag("alice-saw", ("witness:Alice",))
    b = _frag("bob-saw", ("witness:Bob",))
    both = _frag("both-saw", ("witness:Alice", "witness:Bob"))
    kept = filter_fragments_by_copresence(
        [a, b, both], CopresenceScope(chain=("Alice",))
    )
    assert kept == [a, both]


def test_copresence_scope_order_two_requires_both_witnesses():
    a = _frag("alice-only", ("witness:Alice",))
    both = _frag("both", ("witness:Alice", "witness:Bob"))
    all3 = _frag("all3", ("witness:Alice", "witness:Bob", "witness:Carol"))
    kept = filter_fragments_by_copresence(
        [a, both, all3], CopresenceScope(chain=("Alice", "Bob"))
    )
    assert kept == [both, all3]


def test_copresence_scope_fragments_without_witness_tags_passthrough():
    """Un-sited fragments pass through — back-compat with LoCoMo/FANToM."""
    sited = _frag("sited", ("witness:Alice",))
    unsited = _frag("unsited", ("subject:user",))
    bare = _frag("bare", ())
    kept = filter_fragments_by_copresence(
        [sited, unsited, bare], CopresenceScope(chain=("Bob",))
    )
    # sited drops (no witness:Bob); unsited + bare pass through
    assert kept == [unsited, bare]


def test_copresence_scope_chain_order_irrelevant():
    both = _frag("both", ("witness:Alice", "witness:Bob"))
    k_ab = filter_fragments_by_copresence(
        [both], CopresenceScope(chain=("Alice", "Bob"))
    )
    k_ba = filter_fragments_by_copresence(
        [both], CopresenceScope(chain=("Bob", "Alice"))
    )
    assert k_ab == k_ba == [both]


def test_copresence_scope_frozen_hashable():
    s1 = CopresenceScope(chain=("Alice", "Bob"))
    s2 = CopresenceScope(chain=("Alice", "Bob"))
    # Frozen dataclasses are hashable; equal values hash equal
    assert hash(s1) == hash(s2)
    assert s1 == s2


# ---------------------------------------------------------------------------
# CompanionSystem.recall(copresence=...) wiring
# ---------------------------------------------------------------------------


def test_recall_copresence_filters_fragment_pool():
    system = CompanionSystem(seed=0, witness_active=False)
    t = datetime(2026, 1, 1, 12, 0, 0)
    system.observe(
        Observation(
            timestamp=t,
            speaker="Alice",
            content="Alice entered the kitchen.",
            tags=("witness:Alice",),
        )
    )
    system.observe(
        Observation(
            timestamp=t,
            speaker="Bob",
            content="Bob entered the garden.",
            tags=("witness:Bob",),
        )
    )
    recall = system.recall(
        cue="Who was in the kitchen?",
        copresence=CopresenceScope(chain=("Alice",)),
    )
    joined = recall.content
    assert "kitchen" in joined
    assert "garden" not in joined


def test_recall_copresence_metadata_round_trips():
    system = CompanionSystem(seed=0, witness_active=False)
    recall = system.recall(
        cue="x",
        copresence=CopresenceScope(chain=("Alice", "Bob")),
    )
    assert recall.metadata["copresence"] == ("Alice", "Bob")


def test_recall_without_copresence_metadata_is_none():
    system = CompanionSystem(seed=0, witness_active=False)
    recall = system.recall(cue="x")
    assert recall.metadata["copresence"] is None


def test_recall_copresence_order_zero_passes_all_through_filter():
    """Order-0 chain is pass-through at the filter step — any fragments
    reconstruct selects downstream come from the full pool, not an
    empty one. Verified by checking the filter directly rather than
    through reconstruct's cue-similarity ranking."""
    system = CompanionSystem(seed=0, witness_active=False)
    t = datetime(2026, 1, 1, 12, 0, 0)
    system.observe(
        Observation(
            timestamp=t,
            speaker="Alice",
            content="A saw X.",
            tags=("witness:Alice",),
        )
    )
    system.observe(
        Observation(
            timestamp=t,
            speaker="Bob",
            content="B saw Y.",
            tags=("witness:Bob",),
        )
    )
    kept = filter_fragments_by_copresence(
        system.companion.store.all(), CopresenceScope(chain=())
    )
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# HiToM adapter parity — substrate matches scorer-local witness filter
# ---------------------------------------------------------------------------


_WORKED_STORY = """1 Charlotte, Avery and Elizabeth entered the kitchen.
2 The lettuce is in the fridge.
3 Charlotte moved the lettuce to the drawer.
4 Avery exited the kitchen.
5 Charlotte moved the lettuce to the pantry.
"""


def test_ingest_hitom_story_emits_witness_tags():
    obs = ingest_hitom_story_as_observations(_WORKED_STORY)
    assert len(obs) == 5
    first = obs[0]
    # Step 1 — all three enter kitchen, so all three are witnesses
    assert "witness:Charlotte" in first.tags
    assert "witness:Avery" in first.tags
    assert "witness:Elizabeth" in first.tags
    assert "location:kitchen" in first.tags
    assert "event_step:1" in first.tags


def test_ingest_hitom_story_exit_snapshot_before_apply():
    """Avery exited step 4 → still witness at step 4, not step 5."""
    obs = ingest_hitom_story_as_observations(_WORKED_STORY)
    step4 = next(o for o in obs if "event_step:4" in o.tags)
    step5 = next(o for o in obs if "event_step:5" in o.tags)
    assert "witness:Avery" in step4.tags
    assert "witness:Avery" not in step5.tags


def test_substrate_filter_matches_local_witness_on_worked_example():
    """Parity: substrate copresence filter == scorer-local witness filter.

    This is the core promotion correctness check: for the same chain,
    both paths must select the SAME set of events (indexed by step
    prefix on the raw line). If this ever drifts, the substrate is no
    longer a drop-in for the scorer-local path.
    """
    events = parse_hitom_story(_WORKED_STORY)
    chain = ("Charlotte", "Avery")

    local_filtered = events_witnessed_by_chain(events, chain)
    local_steps = {ev.step for ev in local_filtered}

    system = CompanionSystem(seed=0, witness_active=False)
    for ob in ingest_hitom_story_as_observations(_WORKED_STORY):
        system.observe(ob)
    substrate_filtered = filter_fragments_by_copresence(
        system.companion.store.all(),
        CopresenceScope(chain=chain),
    )
    substrate_steps = set()
    for f in substrate_filtered:
        for t in f.tags:
            if t.startswith("event_step:"):
                substrate_steps.add(int(t.split(":", 1)[1]))

    assert substrate_steps == local_steps, (
        f"parity drift: substrate={sorted(substrate_steps)} "
        f"local={sorted(local_steps)}"
    )


def test_substrate_filter_matches_local_witness_order_three():
    """Order-3 chain through a multi-scene story — witness intersection
    must still agree with local filter."""
    events = parse_hitom_story(_WORKED_STORY)
    chain = ("Charlotte", "Avery", "Elizabeth")

    local_filtered = events_witnessed_by_chain(events, chain)
    local_steps = {ev.step for ev in local_filtered}

    obs = ingest_hitom_story_as_observations(_WORKED_STORY)
    substrate_filtered = filter_fragments_by_copresence(
        obs,  # Observations have tags too; filter_fragments_by_copresence
        CopresenceScope(chain=chain),
    ) if False else None  # placeholder — see note below

    # NOTE: filter_fragments_by_copresence takes Fragments. We ingest to
    # a system to get Fragments. This path mirrors the "scorer migrates
    # to substrate" shape.
    system = CompanionSystem(seed=0, witness_active=False)
    for ob in obs:
        system.observe(ob)
    substrate_filtered = filter_fragments_by_copresence(
        system.companion.store.all(),
        CopresenceScope(chain=chain),
    )
    substrate_steps = set()
    for f in substrate_filtered:
        for t in f.tags:
            if t.startswith("event_step:"):
                substrate_steps.add(int(t.split(":", 1)[1]))

    assert substrate_steps == local_steps


def test_ingest_hitom_declarative_present_only_to_in_room_agents():
    """``The lettuce is in the fridge.`` at step 2 — all three in kitchen,
    so all three witness it."""
    obs = ingest_hitom_story_as_observations(_WORKED_STORY)
    step2 = next(o for o in obs if "event_step:2" in o.tags)
    assert "witness:Charlotte" in step2.tags
    assert "witness:Avery" in step2.tags
    assert "witness:Elizabeth" in step2.tags


def test_ingest_hitom_story_speaker_is_first_actor():
    obs = ingest_hitom_story_as_observations(_WORKED_STORY)
    step3 = next(o for o in obs if "event_step:3" in o.tags)
    # "Charlotte moved the lettuce..."
    assert step3.speaker == "Charlotte"


def test_ingest_hitom_story_declarative_speaker_is_world():
    obs = ingest_hitom_story_as_observations(_WORKED_STORY)
    step2 = next(o for o in obs if "event_step:2" in o.tags)
    # "The lettuce is in the fridge." — no actor
    assert step2.speaker == "world"
