"""Phase 7 §2.1 — chat import parsers + safety sweep + one-way boundary.

Three sections:
- One-way boundary enforcement: static scan of chat_import.py bans
  imports of `psychology_base.companion.*` substrate modules and bans
  calls to `.observe(` / `make_fragment(` / `CompanionSystem(`.
- Parsers: ChatGPT JSON / WhatsApp text / generic JSONL.
- Safety sweep: user-only scanning, fail-closed on explicit ideation.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from psychology_base.applications.emotional.chat_import import (
    DEFAULT_IMPORTS_ROOT,
    ExtendedSafetyReport,
    ImportConsent,
    ImportFormat,
    ImportedTurn,
    _assert_one_way_boundary,
    aggregate_session_affect,
    extended_safety_sweep,
    group_turns_into_sessions,
    import_chat_history,
    parse_chatgpt_json,
    parse_jsonl,
    parse_whatsapp_text,
    safety_sweep,
    score_arousal_lexical,
    score_valence_lexical,
    write_raw_audit_log,
)


CHAT_IMPORT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "psychology_base"
    / "applications"
    / "emotional"
    / "chat_import.py"
)

TS = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


def _consent() -> ImportConsent:
    return ImportConsent(
        user_id="u1",
        consent_text_version="phase7-v1",
        attested_at=TS,
        scope="prior-companion",
    )


# ---------------------------------------------------------------------------
# One-way boundary
# ---------------------------------------------------------------------------


def test_one_way_boundary_import_scan() -> None:
    _assert_one_way_boundary()


def test_chat_import_source_bans_companion_system_import() -> None:
    tree = ast.parse(CHAT_IMPORT_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "psychology_base.companion.system",
        "psychology_base.companion.fragment",
        "psychology_base.companion.reconstruction",
    }
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
    assert not (imports & forbidden), (
        f"chat_import.py has forbidden imports: {imports & forbidden}. "
        "Phase 7 one-way boundary: raw imports must never become fragments."
    )


def test_chat_import_source_bans_observe_and_make_fragment_calls() -> None:
    text = CHAT_IMPORT_PATH.read_text(encoding="utf-8")
    # The _assert_one_way_boundary function references the names
    # 'make_fragment' and 'CompanionSystem' in its documentation /
    # forbidden set strings, so we scan for the call-pattern (open paren)
    # not the bare symbol.
    forbidden_call_patterns = [
        ".observe(",
        "make_fragment(",
        "CompanionSystem(",
    ]
    for pattern in forbidden_call_patterns:
        assert pattern not in text, (
            f"chat_import.py contains forbidden call pattern {pattern!r}. "
            "Phase 7 one-way boundary: imports feed inference only."
        )


# ---------------------------------------------------------------------------
# ChatGPT JSON parser
# ---------------------------------------------------------------------------


def _chatgpt_conversation(turns: list[tuple[str, str, float]]) -> dict:
    """Build a minimal ChatGPT-shape conversation.

    turns is a list of (role, text, create_time).
    """
    mapping: dict = {}
    for idx, (role, text, ct) in enumerate(turns):
        mapping[f"n{idx}"] = {
            "id": f"n{idx}",
            "message": {
                "author": {"role": role},
                "content": {"parts": [text]},
                "create_time": ct,
            },
        }
    return {"conversation_id": "conv-abc", "mapping": mapping}


def test_parse_chatgpt_json_basic() -> None:
    conv = _chatgpt_conversation([
        ("user", "I'm not a math person", 1_700_000_000.0),
        ("assistant", "That's a story worth examining.", 1_700_000_010.0),
        ("user", "my mother always said so", 1_700_000_020.0),
    ])
    turns = parse_chatgpt_json(conv)
    assert len(turns) == 3
    assert turns[0].speaker == "user"
    assert turns[0].content == "I'm not a math person"
    assert turns[1].speaker == "assistant"
    assert turns[2].speaker == "user"
    assert all(t.source_format is ImportFormat.CHATGPT_JSON for t in turns)
    assert turns[0].timestamp.tzinfo is not None


def test_parse_chatgpt_json_skips_system_and_empty() -> None:
    conv = _chatgpt_conversation([
        ("system", "you are helpful", 1_700_000_000.0),
        ("user", "", 1_700_000_010.0),
        ("user", "actual content", 1_700_000_020.0),
    ])
    turns = parse_chatgpt_json(conv)
    assert len(turns) == 1
    assert turns[0].content == "actual content"


def test_parse_chatgpt_json_sorts_by_create_time() -> None:
    conv = _chatgpt_conversation([
        ("user", "second", 1_700_000_020.0),
        ("user", "first", 1_700_000_010.0),
    ])
    turns = parse_chatgpt_json(conv)
    assert [t.content for t in turns] == ["first", "second"]


def test_parse_chatgpt_json_accepts_list_of_conversations() -> None:
    conv_a = _chatgpt_conversation([("user", "a", 1_700_000_000.0)])
    conv_b = _chatgpt_conversation([("user", "b", 1_700_000_100.0)])
    conv_a["conversation_id"] = "A"
    conv_b["conversation_id"] = "B"
    turns = parse_chatgpt_json([conv_a, conv_b])
    assert len(turns) == 2
    assert turns[0].source_metadata["conversation_id"] == "A"
    assert turns[1].source_metadata["conversation_id"] == "B"


# ---------------------------------------------------------------------------
# WhatsApp parser
# ---------------------------------------------------------------------------


WHATSAPP_SAMPLE = """\
[2026/10/01, 09:15:00] Mei: Hey, how was your day?
[2026/10/01, 09:16:12] Lin: Rough. Mom called again.
[2026/10/01, 09:16:30] Lin: She wanted to know when I'm getting married.
[2026/10/01, 09:18:00] Mei: Oh no, that again.
"""


def test_parse_whatsapp_text_basic() -> None:
    turns = parse_whatsapp_text(WHATSAPP_SAMPLE, user_name="Lin")
    assert len(turns) == 4
    assert turns[0].speaker == "other:Mei"
    assert turns[1].speaker == "user"
    assert turns[2].speaker == "user"
    assert turns[3].speaker == "other:Mei"
    assert turns[0].content.startswith("Hey, how")
    assert all(t.source_format is ImportFormat.WHATSAPP_TEXT for t in turns)


def test_parse_whatsapp_text_handles_continuation_lines() -> None:
    text = (
        "[2026/10/01, 09:15:00] Mei: First line.\n"
        "Continuation on second line.\n"
        "[2026/10/01, 09:16:00] Mei: New turn.\n"
    )
    turns = parse_whatsapp_text(text, user_name="Lin")
    assert len(turns) == 2
    assert "Continuation on second line." in turns[0].content
    assert turns[1].content == "New turn."


def test_parse_whatsapp_text_accepts_dmy_date_format() -> None:
    text = "01/10/2026, 09:15 - Lin: hello\n"
    turns = parse_whatsapp_text(text, user_name="Lin")
    assert len(turns) == 1
    assert turns[0].timestamp.year == 2026
    assert turns[0].timestamp.month == 10
    assert turns[0].timestamp.day == 1


def test_parse_whatsapp_text_handles_am_pm() -> None:
    text = "[2026/10/01, 02:30:00 PM] Lin: afternoon\n"
    turns = parse_whatsapp_text(text, user_name="Lin")
    assert len(turns) == 1
    assert turns[0].timestamp.hour == 14


# ---------------------------------------------------------------------------
# JSONL parser
# ---------------------------------------------------------------------------


def test_parse_jsonl_basic() -> None:
    lines = [
        json.dumps({
            "speaker": "user",
            "content": "I feel overwhelmed",
            "timestamp": "2026-10-01T09:00:00+00:00",
        }),
        json.dumps({
            "speaker": "assistant",
            "content": "Tell me more.",
            "timestamp": "2026-10-01T09:00:30+00:00",
            "metadata": {"model": "gpt-4"},
        }),
    ]
    turns = parse_jsonl("\n".join(lines))
    assert len(turns) == 2
    assert turns[0].speaker == "user"
    assert turns[1].source_metadata["model"] == "gpt-4"


def test_parse_jsonl_ignores_blank_lines() -> None:
    text = "\n\n" + json.dumps({
        "speaker": "user",
        "content": "x",
        "timestamp": "2026-10-01T09:00:00",
    }) + "\n\n"
    turns = parse_jsonl(text)
    assert len(turns) == 1


def test_parse_jsonl_adds_utc_when_tz_missing() -> None:
    line = json.dumps({
        "speaker": "user",
        "content": "x",
        "timestamp": "2026-10-01T09:00:00",
    })
    turns = parse_jsonl(line)
    assert turns[0].timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Safety sweep
# ---------------------------------------------------------------------------


def _make_turn(speaker: str, content: str) -> ImportedTurn:
    return ImportedTurn(
        speaker=speaker,
        content=content,
        timestamp=TS,
        source_format=ImportFormat.GENERIC_JSONL,
    )


def test_safety_sweep_clean_import_no_handoff() -> None:
    turns = [
        _make_turn("user", "I had a nice day"),
        _make_turn("assistant", "That's good to hear"),
        _make_turn("user", "went for a walk with mom"),
    ]
    report = safety_sweep(turns)
    assert report.incidents == ()
    assert report.requires_handoff is False
    assert report.user_turns_scanned == 2


def test_safety_sweep_explicit_ideation_triggers_handoff() -> None:
    turns = [
        _make_turn("user", "sometimes I think I want to die"),
        _make_turn("user", "but I don't act on it"),
    ]
    report = safety_sweep(turns)
    assert report.requires_handoff is True
    assert len(report.incidents) == 1
    assert "want to die" in report.incidents[0].matched_cue
    assert "Tier-2 handoff" in report.tier_handoff_rationale


def test_safety_sweep_imminent_danger_triggers_handoff() -> None:
    turns = [
        _make_turn("user", "tonight i will end everything"),
    ]
    report = safety_sweep(turns)
    assert report.requires_handoff is True
    assert report.incidents[0].severity == "imminent_danger"


def test_safety_sweep_ambiguous_cue_does_not_force_handoff() -> None:
    turns = [
        _make_turn("user", "I just can't go on with this project"),
    ]
    report = safety_sweep(turns)
    # Ambiguous matches ARE flagged in incidents, but do not force handoff.
    assert any(inc.severity == "ambiguous" for inc in report.incidents)
    assert report.requires_handoff is False


def test_safety_sweep_only_scans_user_turns() -> None:
    turns = [
        _make_turn("assistant", "want to die is a valid feeling"),
        _make_turn("other:Mei", "suicide rates are rising"),
        _make_turn("user", "I'm fine really"),
    ]
    report = safety_sweep(turns)
    # Despite the forbidden phrases in assistant / other turns, we don't scan them.
    assert report.incidents == ()
    assert report.user_turns_scanned == 1


def test_safety_sweep_zh_explicit_cue_triggers_handoff() -> None:
    turns = [
        _make_turn("user", "我真的活不下去了"),
    ]
    report = safety_sweep(turns)
    assert report.requires_handoff is True
    assert any(inc.severity == "explicit_ideation" for inc in report.incidents)


def test_safety_sweep_zh_ambiguous_cue() -> None:
    turns = [
        _make_turn("user", "最近真的没意思"),
    ]
    report = safety_sweep(turns)
    assert any(inc.severity == "ambiguous" for inc in report.incidents)
    assert report.requires_handoff is False


# ---------------------------------------------------------------------------
# Raw audit log + one-call pipeline
# ---------------------------------------------------------------------------


def test_write_raw_audit_log_round_trip(tmp_path: Path) -> None:
    turns = [
        _make_turn("user", "hello"),
        _make_turn("assistant", "hi there"),
    ]
    path = write_raw_audit_log(
        turns,
        user_id="u1",
        consent=_consent(),
        out_root=tmp_path,
    )
    assert path.exists()
    assert path.parent == tmp_path / "u1" / "raw"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["user_id"] == "u1"
    assert data["turn_count"] == 2
    assert data["consent"]["scope"] == "prior-companion"
    assert len(data["turns"]) == 2


def test_import_chat_history_jsonl_end_to_end(tmp_path: Path) -> None:
    jsonl = json.dumps({
        "speaker": "user",
        "content": "I'm not a math person",
        "timestamp": "2026-10-01T09:00:00+00:00",
    }) + "\n" + json.dumps({
        "speaker": "assistant",
        "content": "That's a story worth examining",
        "timestamp": "2026-10-01T09:00:30+00:00",
    })
    result = import_chat_history(
        source_format=ImportFormat.GENERIC_JSONL,
        raw_text_or_data=jsonl,
        user_id="u1",
        consent=_consent(),
        out_root=tmp_path,
    )
    assert len(result.turns) == 2
    assert result.safety_report.requires_handoff is False
    assert result.audit_log_path.exists()


def test_import_chat_history_whatsapp_requires_user_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="user_name"):
        import_chat_history(
            source_format=ImportFormat.WHATSAPP_TEXT,
            raw_text_or_data=WHATSAPP_SAMPLE,
            user_id="u1",
            consent=_consent(),
            out_root=tmp_path,
        )


def test_import_chat_history_whatsapp_end_to_end(tmp_path: Path) -> None:
    result = import_chat_history(
        source_format=ImportFormat.WHATSAPP_TEXT,
        raw_text_or_data=WHATSAPP_SAMPLE,
        user_id="u1",
        consent=_consent(),
        user_name="Lin",
        out_root=tmp_path,
    )
    assert len(result.turns) == 4
    # Lin is the user — two Lin turns, two "other:Mei"
    user_count = sum(1 for t in result.turns if t.speaker == "user")
    assert user_count == 2
    assert result.safety_report.requires_handoff is False


def test_import_chat_history_chatgpt_rejects_raw_text(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="parsed dict"):
        import_chat_history(
            source_format=ImportFormat.CHATGPT_JSON,
            raw_text_or_data="not json",
            user_id="u1",
            consent=_consent(),
            out_root=tmp_path,
        )


def test_import_chat_history_flags_handoff_on_imported_ideation(tmp_path: Path) -> None:
    jsonl = json.dumps({
        "speaker": "user",
        "content": "I used to want to die every day",
        "timestamp": "2026-10-01T09:00:00+00:00",
    })
    result = import_chat_history(
        source_format=ImportFormat.GENERIC_JSONL,
        raw_text_or_data=jsonl,
        user_id="u1",
        consent=_consent(),
        out_root=tmp_path,
    )
    assert result.safety_report.requires_handoff is True
    # Audit log is still written — handoff is a gate on *inference*, not on *audit retention*.
    assert result.audit_log_path.exists()


# ---------------------------------------------------------------------------
# Lexical PAD-2 scorers
# ---------------------------------------------------------------------------


def test_score_valence_positive_en() -> None:
    assert score_valence_lexical("I feel happy and grateful") > 0.5


def test_score_valence_negative_en() -> None:
    assert score_valence_lexical("I feel sad and worthless") < -0.5


def test_score_valence_positive_zh() -> None:
    assert score_valence_lexical("今天很开心, 感觉平静") > 0.5


def test_score_valence_negative_zh() -> None:
    assert score_valence_lexical("我感觉孤独, 很绝望") < -0.5


def test_score_valence_neutral_when_no_cues() -> None:
    assert score_valence_lexical("I went to the store at noon") == 0.0


def test_score_valence_mixed_sentiment() -> None:
    # One positive + one negative cue → zero.
    v = score_valence_lexical("happy but also sad")
    assert -0.1 <= v <= 0.1


def test_score_arousal_high_en() -> None:
    assert score_arousal_lexical("I'm anxious and overwhelmed") > 0.5


def test_score_arousal_low_en() -> None:
    assert score_arousal_lexical("I feel numb and exhausted") < -0.5


def test_score_arousal_high_zh() -> None:
    assert score_arousal_lexical("我焦虑到崩溃") > 0.5


def test_score_arousal_neutral_when_no_cues() -> None:
    assert score_arousal_lexical("the project is on track") == 0.0


# ---------------------------------------------------------------------------
# F3 — role-play register PAD-2 cue coverage (BIG5-CHAT generative prose)
# ---------------------------------------------------------------------------


def test_pad2_role_play_negative_valence_fires() -> None:
    # "rip-off" + "ruin" + "waste" are role-play-register negatives
    # none of which existed in the clinical-register base lexicon.
    v = score_valence_lexical("ugh, this app is a rip-off and a waste")
    assert v < -0.3


def test_pad2_role_play_positive_valence_fires() -> None:
    assert score_valence_lexical("nailed it, that was awesome!") > 0.3


def test_pad2_role_play_low_arousal_fires() -> None:
    assert score_arousal_lexical("meh, nothing special today") < -0.3


def test_pad2_role_play_high_arousal_fires() -> None:
    # "wow" + "insane" are role-play-register arousal markers.
    assert score_arousal_lexical("wow, that is insane") > 0.3


def test_pad2_role_play_zh_register_fires() -> None:
    # Chinese role-play register additions.
    assert score_valence_lexical("今天好棒, 牛!") > 0.3
    assert score_valence_lexical("太烂了, 真烦人") < -0.3


def test_hostility_tokens_do_not_affect_pad2_valence() -> None:
    # Disjoint-concept invariant: "stupid" / "ugh" / "whatever" are
    # hostility (trait_inference), NOT PAD-2 negative-valence. Score
    # should be 0.0, not negative.
    assert score_valence_lexical("ugh, that's stupid, whatever") == 0.0


# ---------------------------------------------------------------------------
# Session grouping
# ---------------------------------------------------------------------------


def _ts(hour: int, day: int = 1) -> datetime:
    return datetime(2026, 10, day, hour, 0, 0, tzinfo=timezone.utc)


def test_group_turns_single_session_within_gap() -> None:
    turns = [
        ImportedTurn("user", "a", _ts(9), ImportFormat.GENERIC_JSONL),
        ImportedTurn("user", "b", _ts(10), ImportFormat.GENERIC_JSONL),
        ImportedTurn("user", "c", _ts(11), ImportFormat.GENERIC_JSONL),
    ]
    sessions = group_turns_into_sessions(turns, gap_minutes=180)
    assert len(sessions) == 1
    assert len(sessions[0]) == 3


def test_group_turns_splits_on_gap() -> None:
    turns = [
        ImportedTurn("user", "a", _ts(9, day=1), ImportFormat.GENERIC_JSONL),
        ImportedTurn("user", "b", _ts(9, day=2), ImportFormat.GENERIC_JSONL),
        ImportedTurn("user", "c", _ts(10, day=2), ImportFormat.GENERIC_JSONL),
    ]
    sessions = group_turns_into_sessions(turns, gap_minutes=180)
    assert len(sessions) == 2
    assert len(sessions[0]) == 1
    assert len(sessions[1]) == 2


def test_group_turns_sorts_by_timestamp() -> None:
    turns = [
        ImportedTurn("user", "c", _ts(11), ImportFormat.GENERIC_JSONL),
        ImportedTurn("user", "a", _ts(9), ImportFormat.GENERIC_JSONL),
        ImportedTurn("user", "b", _ts(10), ImportFormat.GENERIC_JSONL),
    ]
    sessions = group_turns_into_sessions(turns, gap_minutes=180)
    assert len(sessions) == 1
    assert [t.content for t in sessions[0]] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Session affect aggregation
# ---------------------------------------------------------------------------


def test_aggregate_session_affect_user_only() -> None:
    session = [
        ImportedTurn("user", "I'm anxious and overwhelmed", _ts(9), ImportFormat.GENERIC_JSONL),
        ImportedTurn("assistant", "I'm here with you", _ts(9), ImportFormat.GENERIC_JSONL),
        ImportedTurn("user", "I feel scared and panic", _ts(10), ImportFormat.GENERIC_JSONL),
    ]
    affect = aggregate_session_affect(session, session_id="s1")
    assert affect.valence < -0.3
    assert affect.arousal > 0.3


def test_aggregate_session_affect_neutral_on_empty_user_turns() -> None:
    session = [
        ImportedTurn("assistant", "hi", _ts(9), ImportFormat.GENERIC_JSONL),
    ]
    affect = aggregate_session_affect(session, session_id="s1")
    assert affect.valence == 0.0
    assert affect.arousal == 0.0


# ---------------------------------------------------------------------------
# Extended safety sweep
# ---------------------------------------------------------------------------


def _session_turns(day: int, cue_pair: tuple[str, str]) -> list[ImportedTurn]:
    """Build two consecutive user turns on `day` loaded with cue_pair."""
    return [
        ImportedTurn("user", cue_pair[0], _ts(9, day=day), ImportFormat.GENERIC_JSONL),
        ImportedTurn("user", cue_pair[1], _ts(10, day=day), ImportFormat.GENERIC_JSONL),
    ]


def test_extended_safety_sweep_clean_import_no_flag() -> None:
    turns = (
        _session_turns(1, ("I had a nice walk", "feeling calm"))
        + _session_turns(2, ("good day at work", "proud of what I did"))
        + _session_turns(3, ("grateful for mom", "happy"))
    )
    report = extended_safety_sweep(turns)
    assert isinstance(report, ExtendedSafetyReport)
    assert report.requires_handoff is False
    assert report.requires_review is False
    assert report.extreme_emotion_flag is None
    assert len(report.session_affects) == 3


def test_extended_safety_sweep_fires_on_three_distressed_sessions() -> None:
    # Three consecutive sessions in the negative-valence, high-arousal quadrant.
    turns = (
        _session_turns(1, (
            "I feel scared and overwhelmed",
            "I'm anxious and lonely",
        ))
        + _session_turns(2, (
            "panic again today, feeling hopeless",
            "awful and terrified",
        ))
        + _session_turns(3, (
            "frantic, miserable, afraid",
            "I'm shaking, feel worthless",
        ))
    )
    report = extended_safety_sweep(turns)
    assert report.extreme_emotion_flag is not None
    assert report.extreme_emotion_flag.consecutive_sessions >= 3
    assert report.requires_handoff is False  # no explicit-ideation cues
    assert report.requires_review is True
    assert "consecutive imported sessions" in report.rationale


def test_extended_safety_sweep_crisis_dominates() -> None:
    turns = [
        ImportedTurn(
            "user",
            "I used to want to die every day",
            _ts(9),
            ImportFormat.GENERIC_JSONL,
        ),
    ]
    report = extended_safety_sweep(turns)
    assert report.requires_handoff is True
    assert report.requires_review is True
    assert "Tier-2 handoff" in report.rationale


def test_extended_safety_sweep_single_distressed_session_no_flag() -> None:
    turns = _session_turns(1, (
        "I feel scared and anxious",
        "panicked and overwhelmed",
    ))
    report = extended_safety_sweep(turns)
    assert report.extreme_emotion_flag is None
    assert report.requires_review is False


def test_extended_safety_sweep_review_vs_handoff_distinction() -> None:
    # Extreme-emotion only — review=True but handoff=False, so trait inference
    # is not blocked but the operator is nudged to Tier-1 before continuing.
    turns = (
        _session_turns(1, ("sad hopeless and panic", "lonely and anxious"))
        + _session_turns(2, ("hopeless and afraid, furious", "miserable and overwhelmed"))
        + _session_turns(3, ("lonely afraid anxious", "awful and shaking"))
    )
    report = extended_safety_sweep(turns)
    assert report.requires_handoff is False
    assert report.requires_review is True
