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

Both are thin text adapters. Execution permission, reservations, usage and
retries belong to the router; the serving adapter also exposes its turn scope
so both agents can start it before recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


LLMResponder = Callable[[str, str, str], str]
"""Mirror of the agent-side type alias; kept local so this module does
not import from the agents (would create a cycle)."""


SimulatedUser = Callable[[str], str]
"""Signature: (previous_agent_reply) -> next_user_utterance."""

_CONTEXT_PREFIX = '[CONTEXT]\n'
_USER_SEPARATOR = '\n\n[USER]\n'


@dataclass(frozen=True)
class RoutedResponder:
    """The existing three-text adapter with an explicit whole-turn boundary."""
    router: LLMRouter
    tier: TaskTier = TaskTier.SIMPLE
    max_tokens: Optional[int] = 400
    temperature: Optional[float] = 0.3

    @property
    def _has_turn_scope(self):
        return callable(getattr(self.router, "conversation_turn", None))

    def conversation_turn(self):
        # Historical packet builders also use this text adapter with a
        # chat-only message capture. Execution scopes belong to the router.
        return self.router.conversation_turn() if self._has_turn_scope else nullcontext()

    def __call__(self, system_prompt: str, context_block: str, user_utterance: str) -> str:
        response = (self.router.operation("response_generation", inference_role="response")
                    if self._has_turn_scope else nullcontext())
        with self.conversation_turn(), response:
            completion = self.router.chat(
                responder_messages(system_prompt, context_block, user_utterance),
                tier=self.tier, max_tokens=self.max_tokens, temperature=self.temperature)
            text = getattr(completion, "content", "") or getattr(completion, "text", "")
            return text.strip()


def conversation_turn(method):
    """Share one scope across recall, response and observation in both agents.

    Custom injected callables retain their contract. Their owners can wrap a
    whole agent call in router.conversation_turn() when using runtime inference.
    """
    @wraps(method)
    def bounded(self, *args, **kwargs):
        if isinstance(self.llm_responder, RoutedResponder):
            with self.llm_responder.conversation_turn():
                return method(self, *args, **kwargs)
        return method(self, *args, **kwargs)
    return bounded


def responder_messages(system_prompt: str, context_block: str, user_utterance: str) -> list[ChatMessage]:
    """The single message-composition owner for delivery and input accounting."""
    content = (_CONTEXT_PREFIX + context_block + _USER_SEPARATOR + user_utterance
               if context_block else user_utterance)
    return [ChatMessage(role='system', content=system_prompt), ChatMessage(role='user', content=content)]


def request_context_selection(system_prompt, user_utterance, *, application, format_version,
                              max_input, max_context_bytes=6000, mode='legacy',membership_session=None):
    """Reserve actual prompt/query/framing bytes for the existing selector.

    Uses the router's conservative plain-message bound, not a token estimate.
    Empty-context framing differs; an empty-only fit is valid and selects none.
    No fragments, instructions or queries are truncated.
    ``max_input=None`` retains only the context cap for the local reporting path.
    """
    from functools import partial
    from psychology_base.applications.context import context_byte_counter, render_context_block
    from psychology_base.companion.reconstruction import ResearchSelection
    from psychology_base.llm.accounting import request_input_bound
    from psychology_base.llm.execution import ExecutionDeferred
    if ((max_input is not None and (type(max_input) is not int or max_input <= 0))
            or type(max_context_bytes) is not int or max_context_bytes < 0):
        raise ValueError('Request input and context limits require bounded integers')
    if mode not in ('legacy','occurrences'):
        raise ValueError('Delivery selection supports legacy or occurrence evidence')
    empty_bound = request_input_bound(responder_messages(system_prompt, '', user_utterance))
    if max_input is not None and empty_bound > max_input:
        raise ExecutionDeferred('Prompt and query exceed the input limit even without memory')
    framing = len((_CONTEXT_PREFIX + _USER_SEPARATOR).encode('utf-8'))
    available = (max_context_bytes if max_input is None else
                 max(0, min(max_context_bytes, max_input - empty_bound - framing)))
    renderer = partial(render_context_block, assistant_speaker=application, format_version=format_version)
    counter = partial(context_byte_counter, assistant_speaker=application, format_version=format_version)
    requirements = None
    if mode == 'occurrences':
        from psychology_base.applications.reporting_order import reporting_order_intent, reporting_required_occurrences
        if reporting_order_intent(application, user_utterance) is not None:
            requirements = partial(reporting_required_occurrences, application, user_utterance)
            if membership_session is not None:
                from psychology_base.applications.reporting_order import ReportingMembershipSession
                if (not isinstance(membership_session,ReportingMembershipSession)
                        or membership_session.application!=application or membership_session.query!=user_utterance):
                    raise ValueError('Selection membership session belongs to another request')
                requirements=membership_session.required_occurrences
    return ResearchSelection(mode, renderer, available, counter, required_occurrences=requirements)


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
    are shorter still. ``None`` is supported only outside bounded serving.

    ``temperature=0.3`` matches the shakeout mode where we want *some*
    behavioral variance so the chain-reference audit sees different
    phrasings across trials, but not so much that compliance labels
    (SOCRATIC_PROBE / VALIDATES_FALSEHOOD / etc.) flip randomly.
    """

    return RoutedResponder(router, tier, max_tokens, temperature)


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
            temperature=_simulated_user.temperature,
        )
        text = getattr(completion, "content", "") or getattr(
            completion, "text", ""
        )
        utterance = text.strip()
        history.append(("assistant", utterance))
        return utterance

    # Mutable per-callable attribute so a caller can transiently override
    # sampling temperature (e.g. WAI-SR forced-temp probe) and restore the
    # construction-time value — default reads are byte-identical.
    _simulated_user.temperature = temperature  # type: ignore[attr-defined]
    return _simulated_user


__all__ = [
    "LLMResponder",
    "SimulatedUser",
    "make_llm_responder",
    "make_llm_simulated_user",
]
