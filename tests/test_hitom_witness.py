"""Unit tests for the HiToM witness interpreter (Pillar E Option A).

Pins:

- Grammar coverage: entered (multi/single), exited, moved, declarative,
  unknown-form distractors.
- Presence timeline: enters add to room, exits take effect AFTER the
  exit step (agent witnesses their own exit).
- Chain witness filter: order 0 passes everything; order 1 is single-agent
  presence; order N≥2 is intersection in a shared room.
- Multi-scene handling: an agent who re-enters a new room at step K is
  present in that new room from step K onward.
- Full worked example (3-order real HiToM story) checked end-to-end.
"""

from __future__ import annotations

from psychology_base.eval.benchmarks.public_datasets.hitom_witness import (
    HiToMEvent,
    compute_presence,
    events_witnessed_by_chain,
    parse_hitom_story,
    render_witnessed_events,
)


# ---------------------------------------------------------------------------
# Grammar coverage
# ---------------------------------------------------------------------------


def test_parse_entered_multi_agent_and_single() -> None:
    story = (
        "1 Avery, Charlotte, Isabella, Elizabeth and Owen entered the living_room.\n"
        "2 Charlotte entered the garden."
    )
    events = parse_hitom_story(story)
    assert events[0].event_type == "entered"
    assert events[0].agents == ("Avery", "Charlotte", "Isabella", "Elizabeth", "Owen")
    assert events[0].room == "living_room"
    assert events[1].event_type == "entered"
    assert events[1].agents == ("Charlotte",)
    assert events[1].room == "garden"


def test_parse_exited() -> None:
    events = parse_hitom_story("5 Avery exited the living_room.")
    assert events[0].event_type == "exited"
    assert events[0].agents == ("Avery",)
    assert events[0].room == "living_room"


def test_parse_moved() -> None:
    events = parse_hitom_story("3 Avery moved the lettuce to the green_bathtub.")
    assert events[0].event_type == "moved"
    assert events[0].agents == ("Avery",)
    assert events[0].object == "lettuce"
    assert events[0].target == "green_bathtub"


def test_parse_declarative() -> None:
    events = parse_hitom_story("2 The lettuce is in the green_drawer.")
    assert events[0].event_type == "declarative"
    assert events[0].agents == ()
    assert events[0].object == "lettuce"
    assert events[0].target == "green_drawer"


def test_parse_unknown_distractors_preserved() -> None:
    """Preferences / no-movement / lost-items don't match a known form
    but must still appear as unknown events — they are perceivable by
    agents in the room, so witness filtering may still include them."""

    story = (
        "4 Elizabeth dislikes the tangerine.\n"
        "9 Isabella made no movements and stayed in the living_room for 1 minute.\n"
        "10 Jacob lost his watch."
    )
    events = parse_hitom_story(story)
    assert [ev.event_type for ev in events] == ["unknown", "unknown", "unknown"]
    assert [ev.step for ev in events] == [4, 9, 10]


def test_parse_skips_blank_and_unnumbered_lines() -> None:
    events = parse_hitom_story("\n\n   \n***\n1 Avery entered the room.\n")
    assert len(events) == 1
    assert events[0].agents == ("Avery",)


# ---------------------------------------------------------------------------
# Presence timeline
# ---------------------------------------------------------------------------


def test_presence_enter_adds_at_same_step() -> None:
    events = parse_hitom_story(
        "1 Avery entered the room.\n"
        "2 The thing is in the box."
    )
    presence = compute_presence(events)
    assert presence["Avery"][1] == "room"
    assert presence["Avery"][2] == "room"


def test_presence_exit_includes_own_exit_step() -> None:
    """Agent witnesses their own exit event (they are leaving AT that
    step). Only steps STRICTLY AFTER the exit are hidden."""

    events = parse_hitom_story(
        "1 Avery entered the room.\n"
        "2 The thing is in the box.\n"
        "3 Avery exited the room.\n"
        "4 Charlotte entered the room."
    )
    presence = compute_presence(events)
    assert presence["Avery"][1] == "room"
    assert presence["Avery"][2] == "room"
    assert presence["Avery"][3] == "room"  # Avery's own exit step
    assert presence["Avery"][4] is None


def test_presence_multi_scene_re_entry() -> None:
    events = parse_hitom_story(
        "1 Avery entered the living_room.\n"
        "2 Avery exited the living_room.\n"
        "3 Charlotte entered the kitchen.\n"
        "4 Avery entered the kitchen."
    )
    presence = compute_presence(events)
    assert presence["Avery"][1] == "living_room"
    assert presence["Avery"][2] == "living_room"
    assert presence["Avery"][3] is None
    assert presence["Avery"][4] == "kitchen"


# ---------------------------------------------------------------------------
# Chain witness filter
# ---------------------------------------------------------------------------


def test_order_0_empty_chain_passes_everything() -> None:
    events = parse_hitom_story(
        "1 Avery entered the room.\n"
        "2 The lettuce is in the drawer.\n"
        "3 Avery exited the room."
    )
    witnessed = events_witnessed_by_chain(events, ())
    assert [ev.step for ev in witnessed] == [1, 2, 3]


def test_order_1_single_agent_filters_to_their_witness_set() -> None:
    events = parse_hitom_story(
        "1 Avery entered the room.\n"
        "2 The lettuce is in the drawer.\n"
        "3 Avery exited the room.\n"
        "4 Charlotte moved the lettuce to the shelf."
    )
    witnessed = events_witnessed_by_chain(events, ("Avery",))
    assert [ev.step for ev in witnessed] == [1, 2, 3]  # Avery didn't see step 4


def test_order_2_chain_intersection_cuts_at_earliest_exit() -> None:
    """Classic HiToM order-2 shape: Charlotte, Avery both in the room
    up through Avery's exit; later moves hidden from both of them."""

    events = parse_hitom_story(
        "1 Avery, Charlotte entered the living_room.\n"
        "2 The lettuce is in the green_drawer.\n"
        "3 Avery moved the lettuce to the green_bathtub.\n"
        "4 Avery exited the living_room.\n"
        "5 Charlotte moved the lettuce to the blue_pantry.\n"
        "6 Charlotte exited the living_room."
    )
    witnessed = events_witnessed_by_chain(events, ("Charlotte", "Avery"))
    assert [ev.step for ev in witnessed] == [1, 2, 3, 4]


def test_chain_filter_excludes_events_in_different_rooms() -> None:
    """If chain agents are in different rooms at the same step, the
    event is NOT witnessed-together."""

    events = parse_hitom_story(
        "1 Avery entered the living_room.\n"
        "2 Charlotte entered the kitchen.\n"
        "3 The lettuce is in the drawer."
    )
    witnessed = events_witnessed_by_chain(events, ("Avery", "Charlotte"))
    assert witnessed == []  # same step, different rooms


def test_chain_filter_includes_distractor_events_when_all_present() -> None:
    """Preferences / no-movement lines are perceivable utterances; if
    all chain agents are in the room, they must appear in the context."""

    events = parse_hitom_story(
        "1 Avery, Charlotte entered the living_room.\n"
        "2 Elizabeth dislikes the tangerine.\n"
        "3 Charlotte exited the living_room."
    )
    witnessed = events_witnessed_by_chain(events, ("Charlotte", "Avery"))
    assert [ev.step for ev in witnessed] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_preserves_step_prefixes() -> None:
    """The LLM must see explicit step numbers so temporal ordering of
    moves vs exits is unambiguous — critical for ToM reasoning."""

    events = parse_hitom_story(
        "1 Avery entered the room.\n"
        "3 Avery exited the room."
    )
    rendered = render_witnessed_events(events)
    assert "1 Avery entered the room." in rendered
    assert "3 Avery exited the room." in rendered


# ---------------------------------------------------------------------------
# Full worked example — real HiToM order-2 story end to end
# ---------------------------------------------------------------------------


def test_real_hitom_order_2_charlotte_avery_lettuce_scenario() -> None:
    """REGRESSION PIN: 15-line HiToM order-2 scenario.

    Chain (Charlotte, Avery) — "where does Charlotte think Avery
    thinks the lettuce is?"
    Gold = green_bathtub (what Charlotte saw Avery do before Avery
    exited; Charlotte doesn't know Charlotte then moved it elsewhere
    FROM Avery's perspective, because Avery exited first).

    Witness-filtered events must include:
      step 1 (joint entry), step 2 (declarative), step 3 (Avery's
      move to green_bathtub), step 4 (Elizabeth distractor — both
      present), step 5 (Avery exits — both witness the exit step).

    Must EXCLUDE: step 6 onward (Charlotte's own move, Owen's move,
    etc. — all after Avery exits, so Charlotte can't model them as
    Avery-visible).
    """

    story = (
        "1 Avery, Charlotte, Isabella, Elizabeth and Owen entered the living_room.\n"
        "2 The lettuce is in the green_drawer.\n"
        "3 Avery moved the lettuce to the green_bathtub.\n"
        "4 Elizabeth dislikes the tangerine.\n"
        "5 Avery exited the living_room.\n"
        "6 Charlotte moved the lettuce to the blue_pantry.\n"
        "7 Owen likes the green_envelope.\n"
        "8 Charlotte exited the living_room.\n"
        "9 Isabella made no movements and stayed in the living_room for 1 minute.\n"
        "10 Isabella exited the living_room.\n"
        "11 Owen dislikes the green_pepper.\n"
        "12 Elizabeth made no movements and stayed in the living_room for 1 minute.\n"
        "13 Elizabeth exited the living_room.\n"
        "14 Owen moved the lettuce to the green_drawer.\n"
        "15 Owen exited the living_room."
    )
    events = parse_hitom_story(story)
    witnessed = events_witnessed_by_chain(events, ("Charlotte", "Avery"))
    witnessed_steps = [ev.step for ev in witnessed]
    assert witnessed_steps == [1, 2, 3, 4, 5]
    # The critical move Avery made (step 3 → green_bathtub) is present
    step_3 = next(ev for ev in witnessed if ev.step == 3)
    assert step_3.target == "green_bathtub"
    # The confounding move Charlotte makes AFTER Avery exits is excluded
    assert 6 not in witnessed_steps
