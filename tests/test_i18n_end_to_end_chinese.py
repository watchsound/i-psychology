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

from psychology_base.llm.execution import ExecutionPolicy

import json
import pytest

from datetime import datetime, timezone

from psychology_base.applications.emotional import (
    CompanionAgent,
    CompanionSystemPrompt,
)
from psychology_base.applications.context import with_runtime_ambiguity_policy
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
        execution=ExecutionPolicy(mode="research"),
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
    # tokenizer: knob — see test_jieba_knob_live_shortens_cjk_cue below for
    # the knob-on recovery). Use the full-clause cue to hit the dispatch path.
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
    # English adds no language directive; both languages receive the
    # shared runtime ambiguity policy.
    assert "LANGUAGE:" not in composed_sys
    assert composed_sys == with_runtime_ambiguity_policy(CompanionSystemPrompt.SYSTEM_PROMPT, assistant_speaker='companion')

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


# ---------- Phase 9+ i18n follow-on: jieba live-wiring at recall path ----------


def test_jieba_knob_live_shortens_cjk_cue() -> None:
    """`CompanionSystem(cjk_tokenizer="jieba")` lifts the "mono-sentence
    CJK one-token" honest limit documented above.

    Baseline (``cjk_tokenizer="unicode"``, the shipped default): the cue
    "焦虑" and the stored clause "我最近感到很焦虑" tokenise under
    `[A-Za-z\\u4e00-\\u9fff]+` to one token each with no shared element,
    so R1 spreading activation seeds zero and the fragment is NOT
    returned. Knob on: jieba segments the stored clause into word
    tokens including "焦虑", R1 seed overlaps, and the fragment is
    retrieved. English pipeline must remain unaffected on either knob
    position (regression lock).
    """
    import pytest

    pytest.importorskip("jieba")

    # Baseline — unicode default: short cue does NOT hit the stored clause.
    cs_default = CompanionSystem(top_k=4)
    agent_default = CompanionAgent(
        user_id="u1",
        session_id="s1",
        companion_system=cs_default,
        llm_responder=lambda sp, cb, uu: "ok",
    )
    agent_default.respond("我最近感到很焦虑,工作压力很大。", timestamp=TS)
    baseline_recall = cs_default.recall("焦虑")
    baseline_texts = [f.content for f in baseline_recall.fragments_used]
    assert not any("焦虑" in t for t in baseline_texts), (
        "unicode-tokenizer baseline must NOT retrieve on short CJK cue; "
        "if this starts passing the honest-limit documentation is stale."
    )

    # Knob on — jieba: same short cue DOES hit via shared word tokens.
    cs_jieba = CompanionSystem(top_k=4, cjk_tokenizer="jieba")
    agent_jieba = CompanionAgent(
        user_id="u1",
        session_id="s1",
        companion_system=cs_jieba,
        llm_responder=lambda sp, cb, uu: "ok",
    )
    agent_jieba.respond("我最近感到很焦虑,工作压力很大。", timestamp=TS)
    jieba_recall = cs_jieba.recall("焦虑")
    jieba_texts = [f.content for f in jieba_recall.fragments_used]
    assert any("焦虑" in t for t in jieba_texts), (
        "jieba knob should segment the stored clause into words so '焦虑' "
        "matches via shared tokens — the whole point of the knob."
    )


def test_jieba_knob_english_recall_byte_identical(chinese_cases, monkeypatch) -> None:
    """Threading ``cjk_tokenizer="jieba"`` through a system that only
    ever sees English utterances must leave the recall path
    byte-identical to the default. Jieba routes zh-only; English
    content keeps the legacy letter-run tokenizer, so the pool and
    ranking must be unchanged."""
    import pytest

    pytest.importorskip("jieba")
    from psychology_base.applications.emotional import companion_agent as agent_module
    # Hold occurrence metadata fixed across the two tokenizer treatments.
    # These are real UUIDs; memory, observations and recall remain the real path.
    groups = [agent_module.new_exchange_id() for _ in chinese_cases["english"]]

    def _run(tokenizer: str):
        from datetime import timedelta
        receipts = iter(groups)
        monkeypatch.setattr(agent_module, "new_exchange_id", lambda: next(receipts))

        cs = CompanionSystem(top_k=4, cjk_tokenizer=tokenizer)
        agent = CompanionAgent(
            user_id="u1",
            session_id="s1",
            companion_system=cs,
            llm_responder=lambda sp, cb, uu: "I hear you.",
        )
        for index, text in enumerate(chinese_cases["english"]):
            agent.respond(text, timestamp=TS + timedelta(minutes=index))
        trace = []
        for cue in ("sister's dog", "anxious work", "guide-dog program"):
            recall = cs.recall(cue)
            state = cs.inspect()
            assert state.pop("tokenizer_identity")["backend"] == tokenizer
            trace.append((recall, state))
        return trace

    assert _run("unicode") == _run("jieba")


@pytest.mark.parametrize("case_id", ["anxiety", "guide_dog", "mixed_script"])
@pytest.mark.parametrize("include_source", [True, False], ids=["source_present", "source_absent"])
@pytest.mark.parametrize("application", ["companion", "tutor"])
def test_natural_chinese_source_in_application_context(chinese_cases, case_id, include_source, application):
    from dataclasses import replace
    from datetime import timedelta
    from psychology_base.applications.learning.tutor_agent import TutorAgent
    from psychology_base.applications.measurement.scaffolding_telemetry import ScaffoldingTelemetry
    from psychology_base.llm.config import I18nConfig, load_config

    pytest.importorskip("jieba")
    case = next(case for case in chinese_cases["recall"] if case["id"] == case_id)
    contexts = []
    actual_recalls = []

    class ObservedSystem(CompanionSystem):
        """Record real return values; production recall runs without substitution."""

        def recall(self, *args, **kwargs):
            result = super().recall(*args, **kwargs)
            actual_recalls.append(result)
            return result

    def capture_context(system_prompt, context, utterance):
        contexts.append(context)
        return "收到。"

    cfg = replace(load_config(), i18n=I18nConfig(tokenizer="jieba"))
    system = ObservedSystem.from_llm_config(cfg, top_k=6)
    agent_type = CompanionAgent if application == "companion" else TutorAgent
    extras = {} if application == "companion" else {"telemetry": ScaffoldingTelemetry(user_id="u1")}

    def make_agent(session_id):
        return agent_type(user_id="u1", session_id=session_id, companion_system=system,
                          llm_responder=capture_context, **extras)

    agent = make_agent("s1")
    if include_source:
        agent.respond(case["observation"], timestamp=TS)
        original = next(f for f in system.companion.store.all()
                        if f.content == case["observation"])
        assert "user:u1" in original.tags
        if application == "companion":
            assert "speaker:user" in original.tags
        assert original.parent_ids == ()
    else:
        # Nonempty, stateful memory with lexical overlap is not source evidence.
        agent.respond(chinese_cases["edge_pair"][0], timestamp=TS)
    system.mark_session_boundary()
    agent = make_agent("s2")
    agent.respond(case["cue"], timestamp=TS + timedelta(days=7))
    assert contexts[-1]
    assert (case["observation"] in contexts[-1]) is include_source
    assert (case["supporting_span"] in contexts[-1]) is include_source
    selected = actual_recalls[-1].fragments_used
    assert 0 < len(selected) <= 6
    assert (case["observation"] in {fragment.content for fragment in selected}) is include_source
    assert all(system.companion.store.by_id(fragment.id) == fragment for fragment in selected)
    if include_source:
        assert original.id in {fragment.id for fragment in selected}
        assert system.companion.store.by_id(original.id) == original
        if application == "companion":
            assert {'speaker': 'user', 'session': 's1', 'text': original.content} in [
                {k: v for k, v in row.items() if k != 'source_order'}
                for row in map(json.loads, contexts[-1].splitlines())]


def test_chinese_changed_fact_keeps_distinct_provenance(chinese_cases):
    from psychology_base.eval.adapters import Observation, content_hash

    pytest.importorskip("jieba")
    system = CompanionSystem(cjk_tokenizer="jieba")
    for index, text in enumerate(chinese_cases["changed_facts"]):
        system.observe(Observation(timestamp=TS, speaker="user", content=text,
                                   tags=("user:u1", f"session:s{index + 1}")))
    originals = system.companion.store.all()
    rec = system.recall("姐姐的导盲犬")
    assert len({f.id for f in originals}) == 2
    assert {f.content for f in rec.fragments_used} == set(chinese_cases["changed_facts"])
    for fragment in originals:
        assert fragment.id == content_hash(fragment.content, fragment.timestamp, fragment.affect,
                                           fragment.tags, fragment.parent_ids)
        assert system.companion.store.by_id(fragment.id) == fragment
