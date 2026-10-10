"""Unit tests for `applications.llm_responders` wire-shape.

Verifies that `make_llm_responder` and `make_llm_simulated_user`
compose correctly with `LLMRouter` + `MockProvider`. No real LLM
calls — this is a shape test that lives alongside the §6e scripted
shakeout and the §6e-prime LLM-driven shakeout benchmark.
"""

from __future__ import annotations

from pathlib import Path

from psychology_base.applications.llm_responders import (
    make_llm_responder,
    make_llm_simulated_user,
)
from psychology_base.llm import LLMCache, LLMConfig, LLMRouter, LocalOllamaConfig
from psychology_base.llm.config import ProviderConfig
from psychology_base.llm.providers import MockProvider
from psychology_base.llm.types import ChatMessage, TaskTier


def _router(tmp_path: Path, responder):
    cfg = LLMConfig(
        active_provider="mock",
        providers={
            "mock": ProviderConfig(
                name="mock",
                simple_model="mock-simple",
                advanced_model="mock-advanced",
            )
        },
        local_ollama=LocalOllamaConfig(enabled=False),
    )
    mock = MockProvider(name="mock", responder=responder)
    router = LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )
    return router, mock


def test_responder_passes_system_prompt_and_payload(tmp_path: Path) -> None:
    captured: list[list[ChatMessage]] = []

    def responder(messages, tier):
        captured.append(list(messages))
        return "What have you tried so far?"

    router, _ = _router(tmp_path, responder)
    responder_fn = make_llm_responder(router, tier=TaskTier.SIMPLE)

    reply = responder_fn(
        "You are a tutor. Stay Socratic.",
        "Prior: student attempted u-substitution and failed.",
        "I'm stuck.",
    )

    assert reply == "What have you tried so far?"
    assert len(captured) == 1
    msgs = captured[0]
    assert msgs[0].role == "system"
    assert "Stay Socratic" in msgs[0].content
    assert msgs[1].role == "user"
    assert "[CONTEXT]" in msgs[1].content
    assert "u-substitution" in msgs[1].content
    assert "[USER]" in msgs[1].content
    assert "I'm stuck." in msgs[1].content


def test_responder_omits_context_block_when_empty(tmp_path: Path) -> None:
    captured: list[list[ChatMessage]] = []

    def responder(messages, tier):
        captured.append(list(messages))
        return "hi"

    router, _ = _router(tmp_path, responder)
    responder_fn = make_llm_responder(router)
    responder_fn("sys", "", "hello")

    user_msg = captured[0][1].content
    assert user_msg == "hello"
    assert "[CONTEXT]" not in user_msg


def test_responder_routes_at_specified_tier(tmp_path: Path) -> None:
    router, mock = _router(tmp_path, lambda m, t: "ok")
    responder_fn = make_llm_responder(router, tier=TaskTier.ADVANCED)
    responder_fn("sys", "ctx", "utter")
    assert mock.calls[0][1] is TaskTier.ADVANCED


def test_simulated_user_accumulates_history(tmp_path: Path) -> None:
    """Simulated user must see prior agent turns so it can reinterpret
    (chain-reference audit depends on this). Role-mapping is inverted:
    agent turns are `user` from the sim-LLM's view, sim turns are
    `assistant`."""
    turns = ["I'm scared.", "Earlier you said it was fine; something shifted?"]
    idx = {"n": 0}

    def responder(messages, tier):
        reply = turns[idx["n"]]
        idx["n"] += 1
        return reply

    router, mock = _router(tmp_path, responder)
    sim = make_llm_simulated_user(
        router,
        persona_prompt="You are a 28-year-old grieving a parent.",
        scenario_prompt="You start withdrawn, then open up over 2 turns.",
        opening_user_utterance="Hi.",
    )

    u1 = sim("Hello.")
    u2 = sim("That sounds really heavy. What's on your mind?")

    assert u1 == "I'm scared."
    assert u2 == "Earlier you said it was fine; something shifted?"
    # Second call: history includes both agent turns (as `user`) and the
    # first sim turn (as `assistant`). Opening lives in the system prompt,
    # not in the chat.
    msgs_turn_2 = mock.calls[1][0]
    roles_contents = [(m.role, m.content) for m in msgs_turn_2]
    assert ("user", "Hello.") in roles_contents
    assert ("assistant", "I'm scared.") in roles_contents
    assert ("user", "That sounds really heavy. What's on your mind?") in roles_contents
    # Opening is NOT re-injected as a chat turn (it's in system prompt).
    assert ("user", "Hi.") not in roles_contents
    assert ("assistant", "Hi.") not in roles_contents


def test_simulated_user_first_call_starts_with_agent_as_user(tmp_path: Path) -> None:
    """Regression — qwen2.5:7b returned empty with `finish_reason=stop`
    when the sim-LLM saw [system, user(opening), assistant(agent)] (it
    treated the next generation as a continuation of the agent's turn).

    Fix: role-reverse. Agent reply is the first `user` turn; the opening
    is anchored in the system prompt as "you already said …"."""
    captured: list[list[ChatMessage]] = []

    def responder(messages, tier):
        captured.append(list(messages))
        return "yeah I guess"

    router, _ = _router(tmp_path, responder)
    sim = make_llm_simulated_user(
        router,
        persona_prompt="persona",
        scenario_prompt="scenario",
        opening_user_utterance="I'm stuck.",
    )
    sim("What have you tried?")

    msgs = captured[0]
    # system, then the agent reply as the first user turn.
    assert msgs[0].role == "system"
    assert "I'm stuck." in msgs[0].content  # opening is in system prompt
    assert len(msgs) == 2
    assert msgs[1].role == "user"
    assert msgs[1].content == "What have you tried?"


def test_simulated_user_system_prompt_combines_persona_and_scenario(tmp_path: Path) -> None:
    captured: list[list[ChatMessage]] = []

    def responder(messages, tier):
        captured.append(list(messages))
        return "I don't want to say."

    router, _ = _router(tmp_path, responder)
    sim = make_llm_simulated_user(
        router,
        persona_prompt="PERSONA: adolescent, withdrawn.",
        scenario_prompt="SCENARIO: has not slept in 3 days.",
        opening_user_utterance="Hey.",
    )
    sim("How are you?")

    system = captured[0][0]
    assert system.role == "system"
    assert "PERSONA: adolescent" in system.content
    assert "SCENARIO: has not slept" in system.content
    assert "STUDENT / USER" in system.content
    assert "Hey." in system.content  # opening anchored into system prompt
