"""Phase 4a §6e-prime — LLM-driven responder adapters.

The agent wrappers (:class:`TutorAgent`, :class:`CompanionAgent`) take
an injected ``llm_responder`` with signature
``(system_prompt, context_block, user_utterance) -> str``. The Phase
4a §6e shakeout wires this with stub closures that return fixed strings
— good for integration-shape testing, insufficient for anything that
measures *behavior*. This module wires the same injection point to a
real :class:`LLMRouter`.

Two factories:

- :func:`make_llm_responder` — turns the three-arg signature into one
  system + user chat call. Routes through the project's cache so
  benchmark re-runs over identical inputs are free.

- :func:`make_llm_simulated_user` — the complement: an LLM playing a
  persona on the *user* side of the dialogue, consuming the agent's
  prior response and emitting the next user utterance. Lets a pilot
  team run end-to-end multi-turn probes without a human in the loop.
  Not a stand-in for a human — simulated users miss the long-tail
  surprise that pilot evaluation exists to find. Used for shakeout
  only.

Both are thin — no retry / backoff / token accounting. The project's
cache + the router's probe handle 90% of operational concerns; add
features here only when a benchmark surfaces a missing one.
"""

from __future__ import annotations

from typing import Callable, Optional

from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


LLMResponder = Callable[[str, str, str], str]
"""Mirror of the agent-side type alias; kept local so this module does
not import from the agents (would create a cycle)."""


SimulatedUser = Callable[[str], str]
"""Signature: (previous_agent_reply) -> next_user_utterance."""


def make_llm_responder(
    router: LLMRouter,
    *,
    tier: TaskTier = TaskTier.SIMPLE,
    max_tokens: Optional[int] = 400,
    temperature: Optional[float] = 0.3,
) -> LLMResponder:
    """Build an LLMResponder that routes through ``router``.

    The three-arg signature is rendered as a two-message chat (system +
    user). The CONTEXT block and USER utterance share one user turn so
    the cache key stays stable across identical reconstructions —
    putting them in two turns would create spurious cache misses when
    one changes.

    ``max_tokens`` defaults to 400: companion responses are short by
    design (narrative-therapy move + witnessing), tutor Socratic probes
    are shorter still. Pass ``None`` to let the provider decide.

    ``temperature=0.3`` matches the shakeout mode where we want *some*
    behavioral variance so the chain-reference audit sees different
    phrasings across trials, but not so much that compliance labels
    (SOCRATIC_PROBE / VALIDATES_FALSEHOOD / etc.) flip randomly.
    """

    def _responder(
        system_prompt: str, context_block: str, user_utterance: str
    ) -> str:
        if context_block:
            user_payload = (
                f"[CONTEXT]\n{context_block}\n\n[USER]\n{user_utterance}"
            )
        else:
            user_payload = user_utterance
        completion = router.chat(
            [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_payload),
            ],
            tier=tier,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = getattr(completion, "content", "") or getattr(
            completion, "text", ""
        )
        return text.strip()

    return _responder


def make_llm_simulated_user(
    router: LLMRouter,
    *,
    persona_prompt: str,
    scenario_prompt: str,
    opening_user_utterance: str,
    tier: TaskTier = TaskTier.SIMPLE,
    max_tokens: Optional[int] = 200,
    temperature: Optional[float] = 0.7,
) -> SimulatedUser:
    """Build a SimulatedUser that plays ``persona_prompt`` through
    ``scenario_prompt``.

    **Role-reversal is load-bearing.** The sim LLM is playing the
    *student / help-seeker* side of the dialogue; the agent (tutor /
    companion) is the *other party*. Chat templates are trained to
    generate as the ``assistant`` role, so from the sim LLM's internal
    view:

    - agent turns → ``user`` role (what is said *to* the sim)
    - sim turns → ``assistant`` role (what the sim says back)

    Feeding the chat with sim-side turns as ``user`` and agent-side
    turns as ``assistant`` (the naive mapping) causes qwen2.5:7b and
    other instruct-tuned models to either (a) return empty with
    ``finish_reason=stop`` (interpreting the next generation as "nothing
    left to say for the assistant") or (b) continue the agent's turn
    verbatim as a prefix. Both were observed in Phase 4a §6e-prime
    first-run shakeout against local Ollama.

    ``opening_user_utterance`` is the sim's *first* line (what it says
    before the agent has spoken). It is anchored into the system prompt
    as "you already said …" rather than injected as a chat turn — so
    the chat history starts cleanly with the agent's reply as the first
    ``user`` turn, which chat templates expect.

    State is held in a closure (``history`` list) so the simulated user
    can reference earlier turns — important for the chain-reference
    audit (reinterpretation requires memory of the first framing).

    ``temperature=0.7`` because the simulated user *should* vary — the
    agent side is what we're measuring, and uniform user utterances
    collapse the test surface.

    Returns a callable; the closure keeps state. Build a fresh one per
    scenario; do not reuse across scenarios (persona bleed).
    """
    system_prompt = (
        f"{persona_prompt}\n\n{scenario_prompt}\n\n"
        f'You are the STUDENT / USER in this dialogue. The other party '
        f'(tutor or companion) speaks to you as the "user" role; your '
        f'replies are the "assistant" role in the chat below. You '
        f'already opened the conversation by saying: '
        f'"{opening_user_utterance}"\n\n'
        f"When the other party replies, give your next line in character "
        f"— one or two short sentences in natural voice."
    )
    history: list[tuple[str, str]] = []

    def _simulated_user(previous_agent_reply: str) -> str:
        history.append(("user", previous_agent_reply))
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt)
        ]
        for role, text in history:
            messages.append(ChatMessage(role=role, content=text))
        completion = router.chat(
            messages,
            tier=tier,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = getattr(completion, "content", "") or getattr(
            completion, "text", ""
        )
        utterance = text.strip()
        history.append(("assistant", utterance))
        return utterance

    return _simulated_user


__all__ = [
    "LLMResponder",
    "SimulatedUser",
    "make_llm_responder",
    "make_llm_simulated_user",
]
