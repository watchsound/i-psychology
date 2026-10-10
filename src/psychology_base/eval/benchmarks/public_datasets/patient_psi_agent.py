"""PATIENT-Ψ simulated-patient LLM agent factory.

**Plan §4.3** — the simulated-patient side of a Pillar B 10-turn
therapeutic dialogue. The companion (``CompanionAgent`` wrap or
``make_llm_responder`` bare baseline) plays the therapist; this agent
plays the patient whose Beck cognitive model is encoded in a
:class:`PatientPsiProfile`.

**Role-inversion** (same posture as :func:`make_msc_seeker` and
:func:`make_llm_simulated_user` — load-bearing). The sim LLM is
playing the *patient* side of the dialogue; from the LLM's internal
view:

- therapist (companion / bare) turns → ``user`` role
- patient (sim) turns → ``assistant`` role

Naive mapping (patient as user, therapist as assistant) causes
qwen2.5:7b and other instruct-tuned models to either emit empty
completions or echo the therapist's turn as a prefix. Both observed
in Phase 4a §6e-prime shakeout.

**System sidecar.** The patient-prompt reproduces the PATIENT-Ψ paper's
"you are a patient with the following cognitive model" scaffold
(§Appendix of Wang et al. 2024). Core beliefs, intermediate beliefs,
coping strategies, and the Beck quadruple (situation / auto_thought
/ emotion / behavior) are embedded as structured facts the patient
voices during therapy. Style register ("plain" / "verbose" / "tangent"
/ "upset" / "reserved" / "pleasing") is appended as a register
directive — defaults to ``"plain"`` for the Pillar B shakeout; the
scorer may rotate styles for a soft register-sensitivity probe.

**Determinism.** ``temperature=0.0 seed=0`` by default per plan §6
risk mitigation; the cache-pinned :class:`LLMRouter` makes re-runs
byte-identical across cold starts.

**Signature.**

    make_patient_psi_agent(profile, *, router, ...) ->
        callable(opening_turn, [(speaker, text), ...]) -> str

The returned agent is stateless with respect to history — each call
rebuilds the chat from the full history. Build one agent per profile;
reusing across profiles leaks cognitive-model state.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from psychology_base.eval.benchmarks.public_datasets.patient_psi import (
    PatientPsiProfile,
)
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


SPEAKER_THERAPIST: str = "therapist"
SPEAKER_PATIENT: str = "patient"


PatientPsiAgent = Callable[[Sequence[tuple[str, str]]], str]
"""Signature: history → next_patient_utterance.

``history`` is an ordered list of ``(speaker, text)`` pairs where
``speaker in {"therapist", "patient"}``. The agent emits the next
``"patient"`` utterance given this history.
"""


_PATIENT_PROMPT_TEMPLATE = (
    "You are role-playing a cognitive-behavioral-therapy patient. "
    "You are NOT the therapist. Speak as the patient would, in first "
    "person, one or two short sentences per turn. Do NOT narrate, "
    "do NOT describe yourself in third person, do NOT offer coping "
    "advice — you are the one seeking help.\n"
    "\n"
    "Your cognitive model (Beck framework):\n"
    "{cognitive_model_block}"
    "\n"
    "Current session context:\n"
    "- Situation: {situation}\n"
    "- Your automatic thought: \"{auto_thought}\"\n"
    "- Your emotion right now: {emotion}\n"
    "- Your observable behavior: {behavior}\n"
    "\n"
    "History (relevant background):\n"
    "{history}\n"
    "\n"
    "Register directive: {register_block}\n"
    "\n"
    "Voice your automatic thoughts and emotions authentically. When "
    "the therapist asks, respond as this patient would — including "
    "defending maladaptive coping strategies, discounting positive "
    "evidence, or escalating distortions, consistent with the cognitive "
    "model above. If the therapist offers a reframing, respond "
    "realistically — a real patient does not immediately accept new "
    "framings. Keep each turn short."
)


_REGISTER_DIRECTIVES: dict[str, str] = {
    "plain": "Speak plainly and directly, matter-of-fact.",
    "verbose": "Speak at length; elaborate with examples and tangents.",
    "tangent": "Drift from the therapist's question onto related worries "
    "before circling back.",
    "upset": "Convey distress openly — your voice is tense, your sentences "
    "may be short and frustrated.",
    "reserved": "Speak minimally; hesitate before answering, keep responses "
    "brief and guarded.",
    "pleasing": "Try to say what you think the therapist wants to hear; "
    "defer to their framing even when it does not fit your experience.",
}


def _render_cognitive_model_block(profile: PatientPsiProfile) -> str:
    """Project the structured cognitive model into a bulleted block.

    Only non-empty core-belief families are listed to avoid wasting
    system-prompt tokens on empty "unlovable_belief: []" rows.
    """

    lines: list[str] = []
    cm = profile.cognitive_model

    if cm.core_beliefs_helpless:
        lines.append(
            "- Core beliefs (helplessness): "
            + "; ".join(cm.core_beliefs_helpless)
        )
    if cm.core_beliefs_unlovable:
        lines.append(
            "- Core beliefs (unlovability): "
            + "; ".join(cm.core_beliefs_unlovable)
        )
    if cm.core_beliefs_worthless:
        lines.append(
            "- Core beliefs (worthlessness): "
            + "; ".join(cm.core_beliefs_worthless)
        )

    if cm.intermediate_belief:
        lines.append(f"- Intermediate beliefs: {cm.intermediate_belief}")
    if cm.coping_strategies:
        lines.append(f"- Coping strategies: {cm.coping_strategies}")

    return "\n".join(lines) + "\n"


def _render_patient_system_prompt(
    profile: PatientPsiProfile, *, register: str = "plain"
) -> str:
    """Compose the system-prompt sidecar for one profile.

    ``register`` must be a key of :data:`_REGISTER_DIRECTIVES`;
    defaults to ``"plain"``. The function is pure (no I/O, no LLM
    call) so it can be tested without network.
    """

    register_block = _REGISTER_DIRECTIVES.get(
        register, _REGISTER_DIRECTIVES["plain"]
    )
    cm = profile.cognitive_model

    return _PATIENT_PROMPT_TEMPLATE.format(
        cognitive_model_block=_render_cognitive_model_block(profile),
        situation=cm.situation,
        auto_thought=cm.auto_thought,
        emotion="; ".join(cm.emotion) if cm.emotion else "unclear",
        behavior=cm.behavior,
        history=profile.history,
        register_block=register_block,
    )


def _history_to_messages(
    system_prompt: str, history: Sequence[tuple[str, str]]
) -> list[ChatMessage]:
    """Render history to a ChatMessage list under role-inversion.

    - therapist turn → ``user`` role (what is said *to* the patient)
    - patient turn → ``assistant`` role (what the patient has already said)

    An empty history yields just the system message. The LLM then
    generates the patient's first turn in response to no therapist
    utterance — which is the "patient opens the session" scenario
    PATIENT-Ψ uses by default.
    """

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt)
    ]
    for speaker, text in history:
        if speaker == SPEAKER_THERAPIST:
            messages.append(ChatMessage(role="user", content=text))
        elif speaker == SPEAKER_PATIENT:
            messages.append(ChatMessage(role="assistant", content=text))
        else:
            raise ValueError(
                f"patient_psi_agent: unknown speaker {speaker!r}; "
                f"expected one of {{{SPEAKER_THERAPIST!r}, {SPEAKER_PATIENT!r}}}"
            )
    return messages


_EMPTY_FALLBACK: str = (
    "I don't know what to say right now."
)


def make_patient_psi_agent(
    profile: PatientPsiProfile,
    *,
    router: LLMRouter,
    tier: TaskTier = TaskTier.SIMPLE,
    register: str = "plain",
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = 150,
    seed: Optional[int] = 0,
) -> PatientPsiAgent:
    """Build a simulated-patient agent for ``profile``.

    ``register`` picks one of the PATIENT-Ψ style tags; defaults to
    ``"plain"``. ``temperature=0.0 seed=0`` for byte-identical re-runs
    under the cache-pinned router. ``max_tokens=150`` keeps patient
    turns short — long patient monologues over-count turns against
    the gate denominator and change the therapeutic surface the
    companion is probed on.

    The returned callable has signature
    ``history -> next_patient_utterance``. It is stateless; the caller
    must thread ``history`` through turn-by-turn.

    **Empty-output fallback.** If the LLM returns an empty string
    (role-mapping edge case, provider timeout, content-filter) the
    agent returns :data:`_EMPTY_FALLBACK` so the scorer loop does not
    stall or NaN-propagate. The scorer counts empty fallbacks for the
    close-out and treats a run with any fallback as a plumbing warning.
    """

    if register not in _REGISTER_DIRECTIVES:
        raise ValueError(
            f"patient_psi_agent: unknown register {register!r}; "
            f"expected one of {sorted(_REGISTER_DIRECTIVES)}"
        )

    system_prompt = _render_patient_system_prompt(profile, register=register)

    def _patient(history: Sequence[tuple[str, str]]) -> str:
        messages = _history_to_messages(system_prompt, history)
        try:
            completion = router.chat(
                messages,
                tier=tier,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
        except TypeError:
            # Older router.chat signatures may not accept ``seed``;
            # retry without it so the agent degrades gracefully.
            completion = router.chat(
                messages,
                tier=tier,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        text = (
            getattr(completion, "content", "")
            or getattr(completion, "text", "")
        )
        utterance = text.strip()
        if not utterance:
            return _EMPTY_FALLBACK
        return utterance

    return _patient


__all__ = [
    "PatientPsiAgent",
    "SPEAKER_PATIENT",
    "SPEAKER_THERAPIST",
    "make_patient_psi_agent",
]
