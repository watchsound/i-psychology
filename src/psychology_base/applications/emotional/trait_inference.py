"""Phase 7 §2.2 — trait inference from imported chat history (passive mode).

The one-way boundary this module enforces:

    chat_import.py     → ImportedTurn[]  (raw turns; never ingested)
    trait_inference.py → InferenceCandidate[]  (one per fragment candidate)
    confirmation gate  → user confirms / rejects / corrects
    confirmed-only     → CompanionSystem.observe()  (ingest with
                         ``source:inferred-confirmed`` tag)

``infer_traits`` is passive — it reads ImportedTurn[] and emits
candidates but DOES NOT call ``CompanionSystem.observe()``. Callers
must route through ``ingest_confirmed_candidates`` which refuses to
ingest anything lacking a ``confirmed`` flag.

LLM-backed layer is Phase 7+ carried debt. The v1 inference pass is a
lexical heuristic over the EN+ZH cue banks already defined in
``chat_import`` plus message-length / punctuation / diurnal / affective
baseline statistics. Mechanism-viability floor only, not SOTA claim —
B4 benchmark measures real accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Iterable, Optional

from psychology_base.applications.emotional.chat_import import (
    ImportedTurn,
    _count_cues,
    score_arousal_lexical,
    score_valence_lexical,
)
from psychology_base.applications.emotional.trait_profile import (
    LIKERT_MAX,
    LIKERT_MID,
    LIKERT_MIN,
    OceanDim,
    UserMeta,
)
from psychology_base.companion.fragment import COVENANT_TAG
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation


SOURCE_INFERRED_UNCONFIRMED = "source:inferred-unconfirmed"
SOURCE_INFERRED_CONFIRMED = "source:inferred-confirmed"


@dataclass(frozen=True)
class InferenceCandidate:
    """One pre-confirmation candidate for ingestion.

    ``kind`` names the structural role: ``trait``, ``entity``,
    ``self_model_statement``, ``affective_baseline``.
    ``summary`` is the human-readable surface for the confirmation
    gate ("I'm picking up that you have a sister named Lin — is that
    right?").
    ``tags`` are the exact tags that would land on the Observation if
    confirmed. ``confirmed`` is set by the confirmation gate; the
    ingest helper refuses False.
    """

    kind: str
    summary: str
    content: str
    tags: tuple[str, ...]
    timestamp: datetime
    affect: dict = field(default_factory=dict)
    subject_id: Optional[str] = "user"
    entity_id: Optional[str] = None
    confidence: float = 0.0
    source_turn_indices: tuple[int, ...] = ()
    confirmed: bool = False

    def with_confirmation(self, confirmed: bool) -> "InferenceCandidate":
        return InferenceCandidate(
            kind=self.kind,
            summary=self.summary,
            content=self.content,
            tags=self.tags,
            timestamp=self.timestamp,
            affect=dict(self.affect),
            subject_id=self.subject_id,
            entity_id=self.entity_id,
            confidence=self.confidence,
            source_turn_indices=self.source_turn_indices,
            confirmed=confirmed,
        )


# ---------------------------------------------------------------------------
# Lexical cue banks for explicit trait self-labels (v1 heuristic)
# ---------------------------------------------------------------------------


_EXPLICIT_TRAIT_CUES: dict[OceanDim, dict[str, tuple[str, ...]]] = {
    OceanDim.OPENNESS: {
        "high": (
            "i'm curious", "i love learning", "i'm creative", "artsy",
            "我喜欢学习", "我很好奇", "有创造力",
        ),
        "low": (
            "i stick to", "i don't like change", "not creative",
            "我不喜欢变化", "没有想象力",
        ),
    },
    OceanDim.CONSCIENTIOUSNESS: {
        "high": (
            "i plan", "i'm organized", "disciplined",
            "我很有条理", "我做事很有计划",
        ),
        "low": (
            "i'm messy", "i procrastinate", "disorganized",
            "我拖延", "我做事没有计划",
        ),
    },
    OceanDim.EXTRAVERSION: {
        "high": (
            "i love people", "i'm outgoing", "social",
            "我喜欢热闹", "我是外向的",
        ),
        "low": (
            "i'm introverted", "i prefer alone", "quiet",
            "我是内向的", "我喜欢安静",
        ),
    },
    OceanDim.AGREEABLENESS: {
        "high": (
            "i avoid conflict", "i'm trusting", "i'm patient",
            "我不喜欢冲突", "我很信任别人",
        ),
        "low": (
            "i push back", "i don't trust", "i argue",
            "我不信任人", "我喜欢争辩",
        ),
    },
    OceanDim.NEUROTICISM: {
        "high": (
            "i worry a lot", "i'm anxious", "i overthink", "i'm stressed",
            "我很焦虑", "我总是担心", "我很紧张",
        ),
        "low": (
            "i'm calm", "i don't stress", "chill",
            "我很平静", "我不焦虑",
        ),
    },
}


_COVENANT_SELF_DISCLOSURE_CUES: tuple[str, ...] = (
    "diagnosed with", "i have major depress", "i live with",
    "in recovery from", "clinical anxiety", "ptsd",
    "被诊断", "我有重度抑郁", "康复中", "成瘾",
)


# Agreeableness implicit lexicon (F1). Warmth = cooperative / grateful /
# other-directed care. Hostility = INTERPERSONAL DISMISSIVES ONLY —
# tokens addressed-at-other, not generic negative affect. The split is
# conceptually disjoint from PAD-2 _NEGATIVE_VALENCE_CUES so no word
# double-counts on agreeableness and neuroticism signals.
_WARMTH_CUES: tuple[str, ...] = (
    "please", "thank", "thanks", "sorry", "appreciate", "kindly",
    "no worries", "i understand", "i hear you", "that makes sense",
    "fair enough", "take care", "glad to help", "of course",
    "谢谢", "请", "抱歉", "辛苦", "理解", "没关系", "麻烦你",
)

_HOSTILITY_CUES: tuple[str, ...] = (
    "stupid", "idiot", "ridiculous", "dumb", "ugh", "whatever",
    "pathetic", "shut up", "get lost", "nonsense",
    "笨", "蠢", "无语", "呵呵", "滚", "闭嘴",
)


# Openness implicit lexicon (F1). Hypothetical / exploratory
# construction markers signal openness to experience — "what if",
# subjunctive, imaginative framing.
_HYPOTHETICAL_CUES: tuple[str, ...] = (
    "what if", "imagine", "suppose", "i wonder", "could be",
    "maybe we could", "assuming", "hypothetically", "picture this",
    "如果", "假设", "想象一下", "也许", "可能", "万一",
)


# Low-openness lexicon (post-v2 Discovered Issue #1). Routine /
# "stick to what works" / concreteness markers are the empirical
# anti-openness surface — BIG5-CHAT low-O dialogs recur these without
# any high-O signal to subtract. Disjoint from PAD-2 (not affect)
# and from _HOSTILITY_CUES (not interpersonal). Second-pass cues
# ("not gonna change", "leave things as they are", "fine the way",
# "what we always do", "not into fancy") harvested from BIG5-CHAT
# low-O sample analysis after v3 confusion matrix showed 37/40 O
# dialogs still predicting high.
_ROUTINE_CUES: tuple[str, ...] = (
    "stick to", "stick with", "the usual", "same as always",
    "same old", "like usual", "no need to change", "nothing new",
    "keep it simple", "nothing fancy", "straightforward",
    "not gonna change", "not going to change",
    "leave things as they are", "what we always do",
    "not into fancy", "fine the way",
    "按部就班", "一如既往", "老样子", "没什么新", "简单点",
)


# Content-word stopword set for TTR novelty scoring. Short list kept
# inline — only strips the highest-frequency function words that
# otherwise flatten TTR on short conversational utterances.
_CONTENT_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "so", "if", "to", "of",
    "in", "on", "at", "for", "with", "by", "is", "am", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "will", "would", "can", "could", "should", "may", "might", "just",
    "not", "no", "yes", "ok", "okay",
})


def _rate_cue_presence(
    user_turns: list[tuple[int, ImportedTurn]],
    cues: tuple[str, ...],
) -> float:
    """Fraction of turns containing ≥1 cue. Reuses ``_count_cues``."""
    if not user_turns:
        return 0.0
    hits = sum(1 for _, t in user_turns if _count_cues(t.content, cues) > 0)
    return hits / len(user_turns)


def _ttr_content_words(user_turns: list[tuple[int, ImportedTurn]]) -> float:
    """Type-token ratio over stopword-filtered content tokens.

    Short-sample floor at TTR 0.5 (midpoint) when fewer than four
    content tokens survive filtering — on single-utterance inputs TTR
    is dominated by sample size, not vocabulary richness.
    """
    tokens: list[str] = []
    for _, t in user_turns:
        for raw in t.content.lower().split():
            token = "".join(ch for ch in raw if ch.isalnum())
            if not token or token in _CONTENT_STOPWORDS:
                continue
            tokens.append(token)
    if len(tokens) < 4:
        return 0.5
    return len(set(tokens)) / len(tokens)


# ---------------------------------------------------------------------------
# Public API — passive inference pass
# ---------------------------------------------------------------------------


def infer_traits(
    turns: Iterable[ImportedTurn],
    *,
    now: Optional[datetime] = None,
) -> tuple[InferenceCandidate, ...]:
    """Passive inference pass over imported turns.

    Produces one InferenceCandidate per trait signal + entity mention +
    self-model statement + affective baseline. **Does NOT call
    ``CompanionSystem.observe()``.** Caller must route via
    ``ingest_confirmed_candidates``.
    """
    now = now or datetime.now(timezone.utc)
    user_turns = [
        (i, t) for i, t in enumerate(turns) if t.speaker == "user"
    ]
    candidates: list[InferenceCandidate] = []

    candidates.extend(_infer_explicit_traits(user_turns, now=now))
    candidates.extend(_infer_implicit_traits(user_turns, now=now))
    candidates.extend(_infer_covenant_self_disclosure(user_turns, now=now))
    candidates.extend(_infer_entity_mentions(user_turns, now=now))
    candidates.extend(_infer_affective_baseline(user_turns, now=now))

    return tuple(candidates)


def ingest_confirmed_candidates(
    system: CompanionSystem,
    candidates: Iterable[InferenceCandidate],
) -> int:
    """Ingest ONLY candidates with ``confirmed=True``.

    Returns the count ingested. Raises nothing for unconfirmed
    candidates — they are silently skipped. This is the Phase 7 §7
    structural invariant #2 enforcement point: every ingested fragment
    carries ``source:inferred-confirmed``.
    """
    count = 0
    for cand in candidates:
        if not cand.confirmed:
            continue
        if SOURCE_INFERRED_CONFIRMED not in cand.tags:
            raise ValueError(
                "Confirmed candidate missing source:inferred-confirmed tag — "
                "ingest refused. Phase 7 §7 invariant #2 violated."
            )
        obs = Observation(
            timestamp=cand.timestamp,
            speaker="user",
            content=cand.content,
            affect=dict(cand.affect),
            tags=cand.tags,
            subject_id=cand.subject_id,
            entity_id=cand.entity_id,
        )
        system.observe(obs)
        count += 1
    return count


def summarize_meta(candidates: Iterable[InferenceCandidate]) -> UserMeta:
    """Build a ``UserMeta`` summary from inference candidates (no ingestion)."""
    cand_list = list(candidates)
    entities = tuple(
        sorted({c.entity_id for c in cand_list if c.kind == "entity" and c.entity_id})
    )
    self_statements = tuple(
        c.content for c in cand_list if c.kind == "self_model_statement"
    )
    baseline_pairs = [
        (c.affect.get("valence", 0.0), c.affect.get("arousal", 0.0))
        for c in cand_list
        if c.kind == "affective_baseline"
    ]
    baseline = (
        (mean(v for v, _ in baseline_pairs), mean(a for _, a in baseline_pairs))
        if baseline_pairs
        else (0.0, 0.0)
    )
    return UserMeta(
        significant_others=entities,
        self_model_statements=self_statements,
        affective_baseline=baseline,
        communication_style={},
    )


# ---------------------------------------------------------------------------
# Explicit trait extraction — self-label cue matching
# ---------------------------------------------------------------------------


def _infer_explicit_traits(
    user_turns: list[tuple[int, ImportedTurn]],
    *,
    now: datetime,
) -> list[InferenceCandidate]:
    out: list[InferenceCandidate] = []
    for idx, turn in user_turns:
        lower = turn.content.lower()
        for dim, cue_map in _EXPLICIT_TRAIT_CUES.items():
            for pole, cues in cue_map.items():
                if not any(c in lower for c in cues):
                    continue
                score = LIKERT_MAX if pole == "high" else LIKERT_MIN
                tags = (
                    f"trait:{dim.value}",
                    f"trait:{dim.value}:explicit",
                    f"trait:{dim.value}:score:{score:.1f}",
                    SOURCE_INFERRED_UNCONFIRMED,
                )
                out.append(
                    InferenceCandidate(
                        kind="trait",
                        summary=(
                            f"OCEAN {dim.value}: {pole} "
                            f"(explicit self-statement) — {turn.content!r}"
                        ),
                        content=turn.content,
                        tags=tags,
                        timestamp=turn.timestamp,
                        confidence=0.7,
                        source_turn_indices=(idx,),
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Implicit trait extraction — behavioral statistics
# ---------------------------------------------------------------------------


def _mean_or_zero(xs: list[float]) -> float:
    return mean(xs) if xs else 0.0


def _rescale_signed(x: float, *, baseline: float, half_width: float) -> float:
    """Map ``x`` to Likert via signed deviation from ``baseline``.

    Below-baseline → below midpoint; above-baseline → above midpoint;
    at-baseline → ``LIKERT_MID``. ``half_width`` is the deviation that
    saturates to ``LIKERT_MIN`` or ``LIKERT_MAX``. Fixes the zero-anchor
    pathology where "no signal" was indistinguishable from "signal of
    absence" — see Phase 8 Pillar C root-cause analysis Tier 2.
    """
    if half_width <= 0:
        return LIKERT_MID
    norm = max(-1.0, min(1.0, (x - baseline) / half_width))
    return LIKERT_MID + norm * (LIKERT_MAX - LIKERT_MID)


# Conscientiousness-negative cue bank — filler / disfluency tokens. Low-C
# needs a positive signal, not only "absence of structure". Disjoint from
# PAD-2 and from _HOSTILITY_CUES (these are self-register markers, not
# interpersonal dismissals or affect).
_DISFLUENCY_CUES: tuple[str, ...] = (
    "like", "um", "uh", "idk", "whatever", "sorta", "kinda", "dunno",
    "呃", "嗯", "就是", "反正",
)


def _infer_implicit_traits(
    user_turns: list[tuple[int, ImportedTurn]],
    *,
    now: datetime,
) -> list[InferenceCandidate]:
    if not user_turns:
        return []

    out: list[InferenceCandidate] = []
    contents = [t.content for _, t in user_turns]
    lengths = [len(c) for c in contents]
    valences = [score_valence_lexical(c) for c in contents]
    arousals = [score_arousal_lexical(c) for c in contents]

    # Neuroticism — negative valence baseline + high-arousal rate.
    # Zero-coverage fallback: if neither PAD-2 lexicon fires on any turn,
    # emit LIKERT_MID rather than rescaling a false-zero into "slight low".
    neg_valence = max(0.0, -_mean_or_zero(valences))  # ∈ [0, 1]
    high_arousal_rate = sum(1 for a in arousals if a > 0.3) / len(arousals)
    neurotic_coverage = any(v != 0.0 for v in valences) or any(a != 0.0 for a in arousals)
    if not neurotic_coverage:
        neurotic_signal = LIKERT_MID
    else:
        neurotic_signal = _rescale_signed(
            0.5 * neg_valence + 0.5 * high_arousal_rate,
            baseline=0.05,
            half_width=0.30,
        )
    out.append(
        _implicit_candidate(
            dim=OceanDim.NEUROTICISM,
            score=neurotic_signal,
            summary=(
                f"Implicit neuroticism signal from PAD-2 baseline — "
                f"neg_valence={neg_valence:.2f}, "
                f"high_arousal_rate={high_arousal_rate:.2f}"
            ),
            user_turns=user_turns,
        )
    )

    # Extraversion — exclamation-rate is the empirical separator on
    # BIG5-CHAT (high-E≈1.95 excl/dialog, low-E≈0.00). Discovered Issue
    # #2 fix: discard the mean-length component (both high- and low-E
    # BIG5-CHAT dialogs are 200+ chars, so length saturated the baseline
    # on tone-measured low-E prose). mean_len retained in summary only
    # for diagnostic transparency.
    mean_len = _mean_or_zero([float(x) for x in lengths])
    exclam_rate = sum(c.count("!") for c in contents) / max(1, len(contents))
    extraversion_signal = _rescale_signed(
        min(1.0, exclam_rate),
        baseline=0.1,
        half_width=0.4,
    )
    out.append(
        _implicit_candidate(
            dim=OceanDim.EXTRAVERSION,
            score=extraversion_signal,
            summary=(
                f"Implicit extraversion signal — mean_length={mean_len:.1f}, "
                f"exclamation_rate={exclam_rate:.2f}"
            ),
            user_turns=user_turns,
        )
    )

    # Conscientiousness — list / bullet / enumeration markers (high-C)
    # minus filler / disfluency rate (low-C). Signed combination so a
    # single-utterance input with neither signal lands at midpoint, not
    # at LIKERT_MIN.
    struct_markers = sum(
        1 for c in contents if any(m in c for m in ("\n- ", "\n* ", "1.", "2."))
    )
    struct_rate = struct_markers / len(contents)
    disfluency_hits = sum(1 for c in contents if _count_cues(c, _DISFLUENCY_CUES) > 0)
    disfluency_rate = disfluency_hits / len(contents)
    conscientious_signal = _rescale_signed(
        struct_rate - 0.5 * disfluency_rate,
        baseline=0.0,
        half_width=0.3,
    )
    out.append(
        _implicit_candidate(
            dim=OceanDim.CONSCIENTIOUSNESS,
            score=conscientious_signal,
            summary=(
                f"Implicit conscientiousness signal — "
                f"structural_marker_rate={struct_rate:.2f}, "
                f"disfluency_rate={disfluency_rate:.2f}"
            ),
            user_turns=user_turns,
        )
    )

    # Agreeableness — warmth-cue rate (high-A) minus hostility-cue rate
    # (low-A). Lexicons disjoint from PAD-2 so no word double-counts
    # against neuroticism.
    warmth_rate = _rate_cue_presence(user_turns, _WARMTH_CUES)
    hostility_rate = _rate_cue_presence(user_turns, _HOSTILITY_CUES)
    agreeableness_signal = _rescale_signed(
        warmth_rate - hostility_rate,
        baseline=0.0,
        half_width=0.3,
    )
    out.append(
        _implicit_candidate(
            dim=OceanDim.AGREEABLENESS,
            score=agreeableness_signal,
            summary=(
                f"Implicit agreeableness signal — "
                f"warmth_rate={warmth_rate:.2f}, "
                f"hostility_rate={hostility_rate:.2f}"
            ),
            user_turns=user_turns,
        )
    )

    # Openness — hypothetical / exploratory construction rate + novelty
    # vocabulary (TTR offset from 0.5 conversational baseline) MINUS
    # routine-marker rate (post-v2 Discovered Issue #1 fix). Routine
    # cues give an explicit anti-openness surface so low-O dialogs that
    # lack hypothetical cues and have moderate TTR can still land below
    # midpoint.
    hypothetical_rate = _rate_cue_presence(user_turns, _HYPOTHETICAL_CUES)
    routine_rate = _rate_cue_presence(user_turns, _ROUTINE_CUES)
    ttr = _ttr_content_words(user_turns)
    openness_signal = _rescale_signed(
        0.5 * hypothetical_rate + 0.5 * (ttr - 0.5) - routine_rate,
        baseline=0.0,
        half_width=0.3,
    )
    out.append(
        _implicit_candidate(
            dim=OceanDim.OPENNESS,
            score=openness_signal,
            summary=(
                f"Implicit openness signal — "
                f"hypothetical_rate={hypothetical_rate:.2f}, "
                f"routine_rate={routine_rate:.2f}, "
                f"ttr={ttr:.2f}"
            ),
            user_turns=user_turns,
        )
    )

    return out


def _implicit_candidate(
    *,
    dim: OceanDim,
    score: float,
    summary: str,
    user_turns: list[tuple[int, ImportedTurn]],
) -> InferenceCandidate:
    tags = (
        f"trait:{dim.value}",
        f"trait:{dim.value}:implicit",
        f"trait:{dim.value}:score:{score:.2f}",
        SOURCE_INFERRED_UNCONFIRMED,
    )
    last_ts = user_turns[-1][1].timestamp
    return InferenceCandidate(
        kind="trait",
        summary=summary,
        content=f"Behavioral-marker-aggregate score={score:.2f} for {dim.value}",
        tags=tags,
        timestamp=last_ts,
        confidence=min(1.0, len(user_turns) / 50.0),
        source_turn_indices=tuple(i for i, _ in user_turns),
    )


# ---------------------------------------------------------------------------
# Covenant-tagged self-disclosure (routed to Tier-2 handoff, not auto-ingested)
# ---------------------------------------------------------------------------


def _infer_covenant_self_disclosure(
    user_turns: list[tuple[int, ImportedTurn]],
    *,
    now: datetime,
) -> list[InferenceCandidate]:
    """Phase 7 §11 risk-mitigation: crisis-adjacent self-disclosure
    produces covenant-tagged candidates that MUST be routed through
    Tier-2 handoff before any confirmation. The candidate exists so the
    confirmation gate can surface it; it is NEVER auto-confirmed."""
    out: list[InferenceCandidate] = []
    for idx, turn in user_turns:
        lower = turn.content.lower()
        if not any(c in lower for c in _COVENANT_SELF_DISCLOSURE_CUES):
            continue
        tags = (
            "self_model:covenant",
            COVENANT_TAG,
            SOURCE_INFERRED_UNCONFIRMED,
            "requires:tier2-handoff",
        )
        out.append(
            InferenceCandidate(
                kind="self_model_statement",
                summary=(
                    "Covenant-tagged self-disclosure — "
                    f"{turn.content!r}. Route via Tier-2 handoff before confirmation."
                ),
                content=turn.content,
                tags=tags,
                timestamp=turn.timestamp,
                confidence=0.9,
                source_turn_indices=(idx,),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Entity extraction — significant-other mentions
# ---------------------------------------------------------------------------


_RELATION_CUES: dict[str, tuple[str, ...]] = {
    "mother": ("my mother", "my mom", "妈妈", "我妈"),
    "father": ("my father", "my dad", "爸爸", "我爸"),
    "sister": ("my sister", "姐姐", "妹妹"),
    "brother": ("my brother", "哥哥", "弟弟"),
    "partner": ("my partner", "my boyfriend", "my girlfriend", "我男朋友", "我女朋友"),
    "spouse": ("my husband", "my wife", "我老公", "我老婆"),
    "boss": ("my boss", "my manager", "我老板"),
    "friend": ("my friend", "我朋友"),
}


def _infer_entity_mentions(
    user_turns: list[tuple[int, ImportedTurn]],
    *,
    now: datetime,
) -> list[InferenceCandidate]:
    seen: dict[str, list[int]] = {}
    for idx, turn in user_turns:
        lower = turn.content.lower()
        for relation, cues in _RELATION_CUES.items():
            if any(c in lower for c in cues):
                seen.setdefault(relation, []).append(idx)
    out: list[InferenceCandidate] = []
    for relation, indices in seen.items():
        if not indices:
            continue
        last_turn = user_turns[indices[-1] if indices[-1] < len(user_turns) else -1][1]
        tags = (
            f"entity:{relation}",
            f"relation:{relation}",
            SOURCE_INFERRED_UNCONFIRMED,
        )
        out.append(
            InferenceCandidate(
                kind="entity",
                summary=(
                    f"Significant-other candidate: {relation} "
                    f"(mentioned {len(indices)}× in import)"
                ),
                content=(
                    f"User mentions {relation} {len(indices)} times across "
                    f"imported history."
                ),
                tags=tags,
                timestamp=last_turn.timestamp,
                entity_id=relation,
                confidence=min(1.0, len(indices) / 5.0),
                source_turn_indices=tuple(indices),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Affective baseline — Phase 2 M3 PAD-2 starting point
# ---------------------------------------------------------------------------


def _infer_affective_baseline(
    user_turns: list[tuple[int, ImportedTurn]],
    *,
    now: datetime,
) -> list[InferenceCandidate]:
    if not user_turns:
        return []
    valences = [score_valence_lexical(t.content) for _, t in user_turns]
    arousals = [score_arousal_lexical(t.content) for _, t in user_turns]
    v = mean(valences)
    a = mean(arousals)
    last_ts = user_turns[-1][1].timestamp
    tags = (
        "affect:baseline",
        f"affect:baseline:valence:{v:.2f}",
        f"affect:baseline:arousal:{a:.2f}",
        SOURCE_INFERRED_UNCONFIRMED,
    )
    return [
        InferenceCandidate(
            kind="affective_baseline",
            summary=(
                f"PAD-2 baseline from {len(user_turns)} user turns — "
                f"valence={v:.2f}, arousal={a:.2f}"
            ),
            content=(
                f"Affective baseline across imported history: "
                f"valence={v:.2f}, arousal={a:.2f}."
            ),
            tags=tags,
            timestamp=last_ts,
            affect={"valence": v, "arousal": a},
            confidence=min(1.0, len(user_turns) / 30.0),
            source_turn_indices=tuple(i for i, _ in user_turns),
        )
    ]


# ---------------------------------------------------------------------------
# Confirmation-gate helper (programmatic default; UI copy is D3 user-side)
# ---------------------------------------------------------------------------


def confirm_candidates(
    candidates: Iterable[InferenceCandidate],
    *,
    confidence_floor: float = 0.5,
    exclude_covenant: bool = True,
) -> tuple[InferenceCandidate, ...]:
    """Programmatic high-confidence-only confirmation (default per §12 D3).

    ``exclude_covenant`` excludes covenant-tagged candidates from
    auto-confirmation — those MUST route through Tier-2 handoff + user
    affirmative decision per §11. Returns candidates with their
    ``confirmed`` flag flipped (True) and the
    ``source:inferred-unconfirmed`` tag replaced with
    ``source:inferred-confirmed``.
    """
    out: list[InferenceCandidate] = []
    for cand in candidates:
        if exclude_covenant and COVENANT_TAG in cand.tags:
            continue
        if "requires:tier2-handoff" in cand.tags:
            continue
        if cand.confidence < confidence_floor:
            continue
        new_tags = tuple(
            SOURCE_INFERRED_CONFIRMED if t == SOURCE_INFERRED_UNCONFIRMED else t
            for t in cand.tags
        )
        if SOURCE_INFERRED_CONFIRMED not in new_tags:
            new_tags = new_tags + (SOURCE_INFERRED_CONFIRMED,)
        out.append(
            InferenceCandidate(
                kind=cand.kind,
                summary=cand.summary,
                content=cand.content,
                tags=new_tags,
                timestamp=cand.timestamp,
                affect=dict(cand.affect),
                subject_id=cand.subject_id,
                entity_id=cand.entity_id,
                confidence=cand.confidence,
                source_turn_indices=cand.source_turn_indices,
                confirmed=True,
            )
        )
    return tuple(out)


__all__ = [
    "InferenceCandidate",
    "SOURCE_INFERRED_CONFIRMED",
    "SOURCE_INFERRED_UNCONFIRMED",
    "infer_traits",
    "ingest_confirmed_candidates",
    "summarize_meta",
    "confirm_candidates",
]
