"""Unit tests for the PATIENT-Ψ simulated-patient LLM agent factory.

Covers plan §4.3:

- System-prompt renders without KeyError for mixed/empty core-belief
  families — bulleted lines are produced only for non-empty families.
- Register directive is applied; unknown register raises ValueError
  with the valid-set in the message.
- Role-inversion chat template:
  therapist turns → ``user``, patient turns → ``assistant``. Unknown
  speaker raises ValueError.
- Empty-output fallback returns :data:`_EMPTY_FALLBACK` so the scorer
  loop never stalls on a NaN or empty utterance.
- Router ``chat`` signature compat: agent passes ``seed`` through, and
  retries without ``seed`` on ``TypeError`` (older signature).
"""

from __future__ import annotations

from typing import Sequence

import pytest

from psychology_base.eval.benchmarks.public_datasets.patient_psi import (
    CognitiveModel,
    PatientPsiProfile,
)
from psychology_base.eval.benchmarks.public_datasets.patient_psi_agent import (
    SPEAKER_PATIENT,
    SPEAKER_THERAPIST,
    _EMPTY_FALLBACK,
    _REGISTER_DIRECTIVES,
    _history_to_messages,
    _render_cognitive_model_block,
    _render_patient_system_prompt,
    make_patient_psi_agent,
)
from psychology_base.llm.types import ChatMessage, Completion, TaskTier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _profile(
    *,
    helpless: tuple[str, ...] = ("I am incompetent.",),
    unlovable: tuple[str, ...] = (),
    worthless: tuple[str, ...] = (),
    intermediate: str = "If I avoid challenges, I'll be okay.",
    coping: str = "Avoids asking for help.",
    situation: str = "Thinking about bills.",
    auto_thought: str = "What if I run out of money?",
    emotion: tuple[str, ...] = ("anxious",),
    behavior: str = "Continues to sit on couch.",
    history: str = "Mom works two jobs.",
) -> PatientPsiProfile:
    return PatientPsiProfile(
        profile_id="1-1",
        name="Abe",
        style_types=("plain",),
        history=history,
        cognitive_model=CognitiveModel(
            core_beliefs_helpless=helpless,
            core_beliefs_unlovable=unlovable,
            core_beliefs_worthless=worthless,
            intermediate_belief=intermediate,
            coping_strategies=coping,
            situation=situation,
            auto_thought=auto_thought,
            emotion=emotion,
            behavior=behavior,
        ),
        suicidal_ideation_flag=False,
    )


class _StubRouter:
    """Minimal LLMRouter stub returning a scripted reply.

    ``accept_seed`` toggles the ``TypeError`` branch — the agent should
    retry without ``seed`` when the underlying router.chat does not
    accept it (older signature)."""

    def __init__(self, reply: str, *, accept_seed: bool = True) -> None:
        self._reply = reply
        self._accept_seed = accept_seed
        self.calls: list[dict] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tier: TaskTier = TaskTier.SIMPLE,
        max_tokens=None,
        temperature=None,
        seed=None,
    ) -> Completion:
        if seed is not None and not self._accept_seed:
            raise TypeError("router.chat() got an unexpected keyword argument 'seed'")
        self.calls.append(
            {
                "messages": list(messages),
                "tier": tier,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
            }
        )
        return Completion(
            content=self._reply,
            provider="stub",
            model_id="stub-7b",
            tier=tier,
        )


# ---------------------------------------------------------------------------
# System-prompt rendering
# ---------------------------------------------------------------------------


def test_render_cognitive_model_block_skips_empty_families() -> None:
    cm = CognitiveModel(
        core_beliefs_helpless=("I am incompetent.",),
        core_beliefs_unlovable=(),
        core_beliefs_worthless=(),
        intermediate_belief="If X then Y.",
        coping_strategies="Avoids asking for help.",
        situation="",
        auto_thought="",
        emotion=(),
        behavior="",
    )
    profile = _profile()
    block = _render_cognitive_model_block(profile)
    assert "helplessness" in block
    assert "unlovability" not in block
    assert "worthlessness" not in block
    assert "Intermediate beliefs" in block
    assert "Coping strategies" in block


def test_render_patient_system_prompt_default_register_plain() -> None:
    prompt = _render_patient_system_prompt(_profile())
    assert "role-playing a cognitive-behavioral-therapy patient" in prompt
    assert "What if I run out of money?" in prompt
    assert "Thinking about bills." in prompt
    assert _REGISTER_DIRECTIVES["plain"] in prompt


def test_render_patient_system_prompt_applies_register() -> None:
    prompt_upset = _render_patient_system_prompt(_profile(), register="upset")
    prompt_reserved = _render_patient_system_prompt(_profile(), register="reserved")
    assert _REGISTER_DIRECTIVES["upset"] in prompt_upset
    assert _REGISTER_DIRECTIVES["reserved"] in prompt_reserved
    assert _REGISTER_DIRECTIVES["upset"] not in prompt_reserved


def test_render_patient_system_prompt_handles_all_empty_beliefs() -> None:
    """Regression: at least one shim row could ship fully-empty belief
    families (profiles sourced from synthetic generators). The template
    must not emit stray bullet lines or raise on empty tuples."""

    prompt = _render_patient_system_prompt(
        _profile(helpless=(), unlovable=(), worthless=())
    )
    assert "helplessness" not in prompt
    assert "unlovability" not in prompt
    assert "worthlessness" not in prompt


def test_render_patient_system_prompt_handles_empty_emotion() -> None:
    prompt = _render_patient_system_prompt(_profile(emotion=()))
    assert "unclear" in prompt


# ---------------------------------------------------------------------------
# History → messages (role-inversion)
# ---------------------------------------------------------------------------


def test_history_to_messages_empty_history_yields_system_only() -> None:
    msgs = _history_to_messages("SYSTEM", [])
    assert len(msgs) == 1
    assert msgs[0].role == "system"
    assert msgs[0].content == "SYSTEM"


def test_history_to_messages_role_inversion() -> None:
    """REGRESSION PIN: the patient-side LLM is role-playing the patient,
    so therapist turns map to ``user`` role (what is said *to* the
    patient) and patient turns map to ``assistant`` (what the patient
    has already said). Naive mapping emits empty on qwen2.5:7b."""

    history = [
        (SPEAKER_THERAPIST, "What brings you in today?"),
        (SPEAKER_PATIENT, "I feel stuck."),
        (SPEAKER_THERAPIST, "Tell me more."),
    ]
    msgs = _history_to_messages("SYS", history)
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1].content == "What brings you in today?"
    assert msgs[2].content == "I feel stuck."


def test_history_to_messages_raises_on_unknown_speaker() -> None:
    with pytest.raises(ValueError, match="unknown speaker"):
        _history_to_messages("SYS", [("narrator", "The scene opens.")])


# ---------------------------------------------------------------------------
# make_patient_psi_agent — end-to-end with stub router
# ---------------------------------------------------------------------------


def test_make_patient_psi_agent_rejects_unknown_register() -> None:
    router = _StubRouter("I don't know.")
    with pytest.raises(ValueError, match="unknown register"):
        make_patient_psi_agent(_profile(), router=router, register="screaming")


def test_make_patient_psi_agent_returns_stripped_reply() -> None:
    router = _StubRouter("  I feel really anxious today.  ")
    agent = make_patient_psi_agent(_profile(), router=router)
    utterance = agent([])
    assert utterance == "I feel really anxious today."


def test_make_patient_psi_agent_empty_output_fallback() -> None:
    router = _StubRouter("   \n   ")  # whitespace-only → treated as empty
    agent = make_patient_psi_agent(_profile(), router=router)
    utterance = agent([])
    assert utterance == _EMPTY_FALLBACK


def test_make_patient_psi_agent_passes_seed_when_supported() -> None:
    router = _StubRouter("hello", accept_seed=True)
    agent = make_patient_psi_agent(_profile(), router=router, seed=123)
    agent([])
    assert router.calls[0]["seed"] == 123


def test_make_patient_psi_agent_retries_without_seed_on_type_error() -> None:
    """REGRESSION PIN: older ``LLMRouter.chat`` signatures do not
    accept ``seed``. The factory must catch ``TypeError`` and retry
    without the kwarg so the agent degrades gracefully."""

    router = _StubRouter("hello", accept_seed=False)
    agent = make_patient_psi_agent(_profile(), router=router, seed=123)
    utterance = agent([])
    assert utterance == "hello"
    # One retry means `router.calls` records only the successful call;
    # the TypeError-raising call didn't reach the append.
    assert len(router.calls) == 1
    assert "seed" not in router.calls[0] or router.calls[0]["seed"] is None


def test_make_patient_psi_agent_threads_history() -> None:
    router = _StubRouter("Sure, I can tell you more.")
    agent = make_patient_psi_agent(_profile(), router=router)
    history = [
        (SPEAKER_THERAPIST, "What brings you in today?"),
        (SPEAKER_PATIENT, "I feel stuck."),
        (SPEAKER_THERAPIST, "Tell me more."),
    ]
    agent(history)
    call = router.calls[0]
    msgs = call["messages"]
    assert msgs[0].role == "system"
    assert [m.role for m in msgs[1:]] == ["user", "assistant", "user"]
    assert msgs[-1].content == "Tell me more."
