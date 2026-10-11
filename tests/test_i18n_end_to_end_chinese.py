"""End-to-end Chinese-trace integration test for the Phase 9+ i18n stack.

The four slices shipped on the i18n close-out — extraction fact
directive, response directive, reader directive, judge directive —
each have their own unit + integration contract. This test proves the
*composed* stack behaves correctly on a single Chinese user utterance
flowing through:

    user utterance (zh)
      → CompanionAgent.respond
         → CompanionSystem.recall  (round-trip Chinese tokenization)
         → response_language_directive(utterance)  (zh → "Chinese" directive)
         → LLMResponder captures the composed system prompt
      → CompanionSystem.observe(user + companion)

    extraction window (zh)
      → extract_facts(window, [], router)
         → fact_language_directive(window)         (zh → "Chinese" directive)
         → MockProvider captures system prompt
      → scripted ADD lines carrying zh entity / relation / value

The invariant locked here is the stack-level claim behind the
Phase 9+ promise: "a Chinese user drives Chinese-faithful memory +
Chinese-faithful response in one trace, byte-identical to the English
pipeline when the utterance is English". The test asserts both halves
in one file so a future regression that silently drops the directive
on one leg of the pipeline is caught by a single assertion failure.
"""

from __future__ import annotations

from datetime import datetime, timezone

from psychology_base.applications.emotional import (
    CompanionAgent,
    CompanionSystemPrompt,
)
from psychology_base.companion.extraction import extract_facts
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Fragment
from psychology_base.llm import (
    CacheConfig,
    LLMConfig,
    LLMRouter,
    LocalOllamaConfig,
    ProviderConfig,
)
from psychology_base.llm.providers import MockProvider


TS = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)


def _router_capturing(capture: list, *, response_text: str) -> LLMRouter:
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
        cache=CacheConfig(enabled=False),
    )

    def responder(messages, tier):
        capture.append(list(messages))
        return response_text

    return LLMRouter(cfg, providers={"mock": MockProvider(name="mock", responder=responder)})


def test_chinese_end_to_end_response_and_extraction() -> None:
    """One Chinese utterance drives both the response path and the
    extraction path, both carrying a Chinese directive, with the memory
    substrate round-tripping the Chinese content intact."""

    # ---------- A) response path ----------
    captured_response: list[tuple[str, str, str]] = []

    def responder(system_prompt: str, context_block: str, user_utterance: str) -> str:
        captured_response.append((system_prompt, context_block, user_utterance))
        return "我听到了你的感受——跟我多说一点好吗?"

    cs = CompanionSystem(top_k=4)
    agent = CompanionAgent(
        user_id="u1",
        session_id="s1",
        companion_system=cs,
        llm_responder=responder,
    )
    agent.respond("我最近感到很焦虑,工作压力很大。", timestamp=TS)

    assert len(captured_response) == 1
    composed_sys, _ctx, user_utt = captured_response[0]
    # Response directive wiring: system prompt is the locked
    # CompanionSystemPrompt body + an appended LANGUAGE: ... Chinese
    # clause. The locked body stays byte-identical (no translation).
    assert CompanionSystemPrompt.SYSTEM_PROMPT in composed_sys
    assert "LANGUAGE:" in composed_sys
    assert "Chinese" in composed_sys
    # The utterance survives the responder arg intact (no translation,
    # no mojibake).
    assert user_utt == "我最近感到很焦虑,工作压力很大。"

    # Memory substrate observed 2 fragments (user + companion).
    frags = list(cs.companion.store.all())
    assert len(frags) == 2
    user_frag = next(f for f in frags if "speaker:user" in f.tags)
    comp_frag = next(f for f in frags if "speaker:companion" in f.tags)
    # Round-trip: Chinese content survives storage. No UTF-8 corruption.
    assert "焦虑" in user_frag.content
    assert "工作压力" in user_frag.content
    assert "感受" in comp_frag.content

    # ---------- B) recall round-trip on a CJK cue (full-clause match) ----------
    # The default tokenizer splits CJK on punctuation only (`[A-Za-z\u4e00-\u9fff]+`),
    # so "焦虑" by itself does not match the full-clause token
    # "我最近感到很焦虑" — a known honest limit flagged in the Phase 9+
    # i18n close-out ("mono-sentence CJK one token"; jieba is a deferred
    # tokenizer: knob). Use the full-clause cue to hit the dispatch path.
    recall = cs.recall("我最近感到很焦虑")
    recalled_texts = [f.content for f in recall.fragments_used]
    assert any("焦虑" in t for t in recalled_texts)

    # ---------- C) extraction path on the same Chinese window ----------
    scripted_add = (
        "ADD | 用户 | 情绪 | 焦虑 | 0.9\n"
        "ADD | 用户 | 压力源 | 工作 | 0.8\n"
    )
    captured_extraction: list = []
    router = _router_capturing(captured_extraction, response_text=scripted_add)

    window = (
        Fragment(
            id="frag-0001",
            content="我最近感到很焦虑,工作压力很大。",
            timestamp=TS,
            tags=("speaker:user",),
        ),
    )
    ops = extract_facts(window, prior_facts=[], router=router, tick_count=0)

    assert len(captured_extraction) == 1
    msgs = captured_extraction[0]
    system_msg = next(m for m in msgs if m.role == "system")
    # Extraction fact directive wiring: system prompt is the locked
    # EXTRACTION_SYSTEM_PROMPT + an appended LANGUAGE: ... Chinese
    # clause pinning the ADD / UPDATE / NOOP op tokens in English.
    assert "LANGUAGE:" in system_msg.content
    assert "Chinese" in system_msg.content
    assert "ADD / UPDATE / NOOP" in system_msg.content

    # The scripted response round-trips: two Chinese fact ops emerge.
    assert len(ops) == 2
    entities = {op.fact.entity for op in ops}
    assert "用户" in entities
    values = {op.fact.value for op in ops}
    assert "焦虑" in values
    assert "工作" in values


def test_english_end_to_end_byte_identical_control() -> None:
    """Byte-identical control: an English utterance through the same
    composed stack produces NO LANGUAGE: directive on either leg. This
    is the English pipeline regression lock the whole i18n foundation
    was designed to protect — ensures the end-to-end composition does
    not accidentally fire on English because of an upstream feature
    leaking through."""

    captured_response: list[tuple[str, str, str]] = []

    def responder(system_prompt: str, context_block: str, user_utterance: str) -> str:
        captured_response.append((system_prompt, context_block, user_utterance))
        return "I hear you — tell me more."

    cs = CompanionSystem(top_k=4)
    agent = CompanionAgent(
        user_id="u1",
        session_id="s1",
        companion_system=cs,
        llm_responder=responder,
    )
    agent.respond("I've been feeling anxious about work lately.", timestamp=TS)

    assert len(captured_response) == 1
    composed_sys, _ctx, _utt = captured_response[0]
    # No directive leaked into the English response path. The locked
    # system prompt is byte-identical to the pre-i18n composition.
    assert "LANGUAGE:" not in composed_sys
    assert composed_sys == CompanionSystemPrompt.SYSTEM_PROMPT

    # Extraction leg — same control.
    scripted_add = "ADD | user | emotion | anxious | 0.9\n"
    captured_extraction: list = []
    router = _router_capturing(captured_extraction, response_text=scripted_add)
    window = (
        Fragment(
            id="frag-0001",
            content="I've been feeling anxious about work lately.",
            timestamp=TS,
            tags=("speaker:user",),
        ),
    )
    ops = extract_facts(window, prior_facts=[], router=router, tick_count=0)

    assert len(captured_extraction) == 1
    system_msg = next(m for m in captured_extraction[0] if m.role == "system")
    assert "LANGUAGE:" not in system_msg.content
    assert len(ops) == 1
    assert ops[0].fact.entity == "user"
    assert ops[0].fact.value == "anxious"
