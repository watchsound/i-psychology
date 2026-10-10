"""LoCoMo adapter — ultra-long two-speaker dialogues + category-tagged QA.

**Native format reverse-engineered from the LoCoMo repo**
(snap-research/locomo, arXiv 2402.17753). Load-bearing observations:

- A LoCoMo instance is one *dialogue* between two synthetic human speakers
  (e.g. "Jennifer" + "David"), split into many sessions. Each session has
  a date; each turn has a speaker + text. Full dialogues are ~300 turns
  and ~9k tokens.
- Each instance ships 100-300 QA pairs across five categories:
  single-hop, multi-hop, open-domain, temporal-reasoning, adversarial.
  The ``category`` field distinguishes them and drives scoring choice.
- Single-hop questions have short-form gold answers (names, dates,
  single-phrase facts). F1 against gold is the published scoring for this
  subset — Mem0's 92.5% is single-hop F1.
- Open-domain / multi-hop questions have longer gold answers and require
  judge-based scoring.
- Adversarial questions have a special ``adversarial_answer`` field that
  is the gold *refusal* ("cannot be determined from the dialogue") — an
  abstention test like LongMemEval's abstention category.

**Dialogue, not Q&A.** Unlike LongMemEval the "history" is not a user-AI
chat; it is two humans talking to each other, and the system is a third
party observing. We feed each turn as ``speaker=<name>``,
``content=<text>``, so the SUT sees "Jennifer: I bought a sailboat" rather
than ``role=user``. This is what the plan §1.3 OSS table calls out as the
"schema-constrained search" risk — getting the speaker-tagging right
preserves the SUT's ability to attribute facts to the right person at
recall time.

**Format shim.** The real repo ships `locomo10.json` as a JSON array of 10
objects. We ingest a JSONL shim one-instance-per-line to match EpBench /
LongMemEval conventions (keeps memory flat, enables streaming). The
``convert_locomo`` script performs the real → shim transform.

JSONL shim schema (one object per line)::

    {
      "instance_id": "...",
      "sessions": [
        {
          "session_id": "session_1",
          "date": "YYYY-MM-DDTHH:MM:SS",     # optional
          "turns": [
            {"speaker": "Jennifer", "content": "..."},
            {"speaker": "David", "content": "..."},
            ...
          ]
        },
        ...
      ],
      "questions": [
        {
          "id": "...",
          "question": "...",
          "answer": "...",                       # short-form gold
          "adversarial_answer": "...",           # optional; only for
                                                 #   category='adversarial'
          "category": "single-hop",              # one of the five
          "evidence": ["session_3"]              # optional; session ids
                                                 #   from which the answer
                                                 #   derives
        },
        ...
      ]
    }

**Dual-mode runners.** Direct-mode scores ``recall.content`` against the
gold via F1 (for short-form single-hop) or judge (for free-form); RAG-mode
runs the recall through the shared reader step and scores the reader's
output the same way. Deterministic-F1 subset is what the step-4 pytest
smoke test exercises.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Optional

from psychology_base.eval.benchmarks.instance import (
    BenchmarkInstance,
    BenchmarkQuestion,
    BenchmarkTurn,
)


# ---------- categories ----------

# LoCoMo's five categories. The deterministic-F1 subset is single-hop
# only; the others go through the judge. ``adversarial`` is special — the
# gold is a refusal, and the ``adversarial_answer`` field (if present) is
# the actual gold string; the normal ``answer`` for adversarial rows is
# the trap answer the SUT must NOT return.
SINGLE_HOP = "single-hop"
MULTI_HOP = "multi-hop"
OPEN_DOMAIN = "open-domain"
TEMPORAL = "temporal-reasoning"
ADVERSARIAL = "adversarial"

CATEGORIES = (SINGLE_HOP, MULTI_HOP, OPEN_DOMAIN, TEMPORAL, ADVERSARIAL)
DETERMINISTIC_CATEGORIES = (SINGLE_HOP,)


# ---------- short-form F1 (Mem0 / SQuAD convention) ----------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """SQuAD-style normalization — lowercase, strip punctuation, strip
    English articles, collapse whitespace. Mem0's scoring code uses this
    same path, so our single-hop F1 numbers are commensurable with their
    published 92.5%."""
    if text is None:
        return ""
    s = text.lower()
    s = _ARTICLE_RE.sub(" ", s)
    s = s.translate(_PUNCT_TABLE)
    s = _WS_RE.sub(" ", s).strip()
    return s


def token_f1(gold: str, candidate: str) -> float:
    """Token-level F1 after normalization. Returns 1.0 when both normalize
    to empty, 0.0 when exactly one does."""
    g = _normalize(gold).split()
    c = _normalize(candidate).split()
    if not g and not c:
        return 1.0
    if not g or not c:
        return 0.0
    g_counts: dict[str, int] = {}
    for t in g:
        g_counts[t] = g_counts.get(t, 0) + 1
    overlap = 0
    for t in c:
        if g_counts.get(t, 0) > 0:
            overlap += 1
            g_counts[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(c)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


# ---------- shim parsing ----------


@dataclass(frozen=True)
class _ShimTurn:
    speaker: str
    content: str


@dataclass(frozen=True)
class _ShimSession:
    session_id: str
    date: datetime
    turns: tuple[_ShimTurn, ...]


def _parse_shim_timestamp(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime(2000, 1, 1)


def _parse_sessions(raw: dict) -> tuple[_ShimSession, ...]:
    sessions_raw = raw.get("sessions") or []
    out: list[_ShimSession] = []
    for idx, session in enumerate(sessions_raw):
        sid = str(session.get("session_id", f"session-{idx}"))
        date = _parse_shim_timestamp(str(session.get("date", "")))
        turns = tuple(
            _ShimTurn(
                speaker=str(t.get("speaker", "")),
                content=str(t.get("content", "")),
            )
            for t in session.get("turns", [])
        )
        out.append(_ShimSession(session_id=sid, date=date, turns=turns))
    return tuple(out)


def _sessions_to_turns(sessions: tuple[_ShimSession, ...]) -> tuple[BenchmarkTurn, ...]:
    out: list[BenchmarkTurn] = []
    for session in sessions:
        for i, turn in enumerate(session.turns):
            out.append(
                BenchmarkTurn(
                    timestamp=session.date + timedelta(minutes=i),
                    speaker=turn.speaker,
                    content=turn.content,
                    session_id=session.session_id,
                )
            )
    return tuple(out)


def _parse_questions(raw: dict) -> tuple[BenchmarkQuestion, ...]:
    out: list[BenchmarkQuestion] = []
    for q in raw.get("questions") or []:
        category = str(q.get("category", "")).strip() or SINGLE_HOP
        # For adversarial rows, the "gold" is the refusal string if
        # supplied; otherwise fall back to a canonical refusal so the F1
        # scorer still has something to compare.
        if category == ADVERSARIAL:
            gold = str(q.get("adversarial_answer") or "cannot be determined")
        else:
            gold = str(q.get("answer", ""))
        out.append(
            BenchmarkQuestion(
                id=str(q["id"]),
                cue=str(q["question"]),
                gold=gold,
                question_type=category,
                metadata={
                    "evidence": tuple(str(e) for e in q.get("evidence") or ()),
                    "raw_answer": str(q.get("answer", "")),
                    "adversarial_answer": str(q.get("adversarial_answer", "")),
                },
            )
        )
    return tuple(out)


# ---------- loader ----------


@dataclass
class LoCoMoLoader:
    """Streams LoCoMo instances from a JSONL shim file."""

    name: str = "locomo"
    shim_path: Optional[Path] = None

    def iter_instances(self, root: Optional[Path] = None) -> Iterator[BenchmarkInstance]:
        path = self.shim_path or (
            root / "locomo_shim.jsonl" if root else None
        )
        if path is None or not Path(path).exists():
            return
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                sessions = _parse_sessions(raw)
                history = _sessions_to_turns(sessions)
                questions = _parse_questions(raw)
                yield BenchmarkInstance(
                    name="locomo",
                    instance_id=str(raw["instance_id"]),
                    history=history,
                    questions=questions,
                    metadata={
                        "num_sessions": len(sessions),
                        "num_turns": len(history),
                        "num_questions": len(questions),
                    },
                )


# ---------- result types ----------


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    gold: str
    candidate: str
    score: float
    verdict: str = ""
    category: str = ""
    rationale: str = ""
    # Phase 2 R6: optional per-row recall diagnostics (evidence-session
    # overlap, gold-substring presence, gold-token coverage) so that
    # stacked-ablation comparisons can classify incorrect rows as
    # reader-side (gold-in-pool, reader mis-extracts) vs recall-side
    # (gold-absent-from-pool). Populated by run_rag_mode / run_direct_mode
    # when the instance carries LoCoMo evidence metadata; None otherwise
    # so the field is additive and callers without diagnostics keep
    # working.
    recall_diagnostics: Optional[dict] = None


@dataclass
class InstanceResult:
    instance_id: str
    per_question: list[QuestionResult] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.per_question:
            return 0.0
        return sum(r.score for r in self.per_question) / len(self.per_question)

    def mean_by_category(self) -> dict[str, float]:
        """Per-category mean. Empty categories omitted. Lets single-hop
        parity vs. aggregate get surfaced separately in the dual-mode
        result table — single-hop is the Mem0 92.5% comparison point;
        aggregate is the honest overall number."""
        buckets: dict[str, list[float]] = {}
        for r in self.per_question:
            buckets.setdefault(r.category or SINGLE_HOP, []).append(r.score)
        return {cat: sum(xs) / len(xs) for cat, xs in buckets.items() if xs}


# ---------- Phase 2 R6: recall diagnostics ----------


_LOCOMO_TURN_EVIDENCE_RE = re.compile(r"^[Dd](\d+):(\d+)$")


def _normalize_evidence_sessions(evidence: Iterable[str]) -> tuple[str, ...]:
    """LoCoMo native evidence is per-turn (``D9:2`` = dialog 9 turn 2).
    Our fragment tags carry ``session_<N>``. Map the former to the latter
    for the session-overlap check. Non-matching strings pass through so
    other schemas (future benchmarks) are not broken."""
    out: list[str] = []
    seen: set[str] = set()
    for e in evidence:
        if not e:
            continue
        m = _LOCOMO_TURN_EVIDENCE_RE.match(e)
        if m:
            key = f"session_{int(m.group(1))}"
        else:
            key = e
        if key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)


def _recall_diagnostics(
    *,
    question: BenchmarkQuestion,
    fragments: Iterable,
) -> dict:
    """Classify a question's recall as reader-side vs recall-side bottleneck.

    Returns a dict with:
      - ``evidence_sessions``: tuple of session IDs the LoCoMo authors
        marked as the gold source (from ``question.metadata['evidence']``)
      - ``top_k_fragment_count``: number of fragments the reader will see
      - ``top_k_sessions``: sorted unique session IDs among those fragments
        (sourced from ``Fragment.tags`` which carries the originating
        ``session_id`` under the LoCoMo adapter's ingest contract)
      - ``evidence_session_in_top_k``: True iff any top-K fragment comes
        from an evidence session — session-level recall@K signal
      - ``gold_substring_in_top_k``: True iff the normalized gold string
        appears as a substring of the normalized concat of top-K fragment
        texts — lexical "the reader literally had the answer in-pool" check
      - ``gold_token_coverage``: fraction of unique normalized gold tokens
        that appear as whole-token matches anywhere in top-K text — a
        softer coverage score for multi-token golds like "19 October 2023"

    Reader-side failure = gold_substring_in_top_k True (reader saw the
    answer, output something else). Recall-side failure =
    evidence_session_in_top_k False AND gold_substring_in_top_k False
    (the pool genuinely never had the answer). Mixed / imprecise = evidence
    session present but gold string absent (pool from right session but
    specific turn not surfaced)."""
    frags = list(fragments)
    evidence_raw = tuple(question.metadata.get("evidence") or ())
    evidence_sessions = _normalize_evidence_sessions(evidence_raw)
    top_k_sessions: list[str] = []
    seen: set[str] = set()
    for f in frags:
        tags = getattr(f, "tags", ()) or ()
        for t in tags:
            if t and t not in seen:
                seen.add(t)
                top_k_sessions.append(t)
    top_k_sessions.sort()

    evidence_set = set(evidence_sessions)
    evidence_session_in_top_k = bool(evidence_set & seen) if evidence_set else False

    text_norm = _normalize(" ".join(getattr(f, "content", "") for f in frags))
    gold_norm = _normalize(question.gold)
    gold_substring_in_top_k = bool(gold_norm) and gold_norm in text_norm

    gold_tokens = tuple({t for t in gold_norm.split() if t})
    pool_tokens = set(text_norm.split())
    gold_token_coverage = (
        sum(1 for t in gold_tokens if t in pool_tokens) / len(gold_tokens)
        if gold_tokens
        else 0.0
    )

    return {
        "evidence_sessions": evidence_sessions,
        "top_k_fragment_count": len(frags),
        "top_k_sessions": tuple(top_k_sessions),
        "evidence_session_in_top_k": evidence_session_in_top_k,
        "gold_substring_in_top_k": gold_substring_in_top_k,
        "gold_token_coverage": round(gold_token_coverage, 3),
    }


# ---------- runners ----------


def _ingest_history(system, instance: BenchmarkInstance) -> None:
    from psychology_base.eval.adapters import Observation

    probe = getattr(system, "inspect", None)
    if callable(probe):
        info = probe()
        if info is not None and info.get("fragment_count", 0) != 0:
            raise RuntimeError(
                f"Instance isolation violated: fragment_count="
                f"{info.get('fragment_count')} at instance start"
            )
    mark_boundary = getattr(system, "mark_session_boundary", None)
    tick = getattr(system, "tick", None)
    prev_session: Optional[str] = None
    for turn in instance.history:
        # Phase 1.5: emit a session boundary + a sleep tick when the
        # session_id transitions. The boundary signals the Consolidator
        # that the previous window is coherent; the tick triggers
        # extraction on that window before the next session's turns
        # start arriving. Systems without consolidation ignore both calls
        # (``mark_session_boundary`` is a counter; ``tick`` is already
        # part of the MemorySystem protocol).
        if (
            turn.session_id
            and prev_session is not None
            and turn.session_id != prev_session
        ):
            if callable(mark_boundary):
                mark_boundary()
            if callable(tick):
                tick()
        system.observe(
            Observation(
                timestamp=turn.timestamp,
                speaker=turn.speaker,
                content=turn.content,
                affect={},
                tags=(turn.session_id,) if turn.session_id else (),
            )
        )
        if turn.session_id:
            prev_session = turn.session_id
    # Flush the final session's window so facts are available at recall.
    if callable(mark_boundary):
        mark_boundary()
    if callable(tick):
        tick()


def _score(
    *,
    category: str,
    gold: str,
    candidate: str,
    judge,
    judgment_cache,
    benchmark: str,
    instance_id: str,
    question_id: str,
    question: str,
    force_judge_categories: Optional[Iterable[str]] = None,
):
    """Pick deterministic F1 vs. judge based on category.

    Single-hop uses token F1 (deterministic, matches Mem0 convention).
    Everything else routes through the judge cache. The adversarial
    category is still judged — a strict F1 against the refusal string
    would false-fail on acceptable paraphrases like "the dialogue doesn't
    say".

    ``force_judge_categories`` overrides the F1 route for the named
    categories. Phase 1.5-prime P1 uses this to re-score single-hop
    through the judge: paraphrase-correct answers that score 0.0 under
    token F1 (e.g. gold "Roast marshmallows, tell stories" vs reader
    "exploring, hiking") may be judge-correct, closing part of the
    0.288→0.70 gap without changing the substrate.
    """
    forced = set(force_judge_categories or ())
    if category in DETERMINISTIC_CATEGORIES and category not in forced:
        score = token_f1(gold, candidate)
        verdict = "f1-correct" if score >= 0.5 else "f1-incorrect"
        return score, verdict, f"token_f1={score:.3f}"
    judgment = judgment_cache.judge_cached(
        client=judge,
        benchmark=benchmark,
        instance_id=instance_id,
        question_id=question_id,
        question=question,
        gold=gold,
        candidate=candidate,
        question_type=category,
    )
    return judgment.score, judgment.verdict, judgment.rationale


def run_direct_mode(
    instance: BenchmarkInstance,
    *,
    system_factory,
    judge,
    judgment_cache,
    force_judge_categories: Optional[Iterable[str]] = None,
) -> InstanceResult:
    """Observe each turn, call ``recall(question)``, score raw
    ``recall.content`` by category: F1 for single-hop, judge for the rest.
    """
    system = system_factory()
    _ingest_history(system, instance)
    result = InstanceResult(instance_id=instance.instance_id)
    for q in instance.questions:
        recall = system.recall(q.cue, {})
        score, verdict, rationale = _score(
            category=q.question_type,
            gold=q.gold,
            candidate=recall.content,
            judge=judge,
            judgment_cache=judgment_cache,
            benchmark="locomo",
            instance_id=instance.instance_id,
            question_id=q.id,
            question=q.cue,
            force_judge_categories=force_judge_categories,
        )
        diagnostics = _recall_diagnostics(question=q, fragments=recall.fragments_used)
        result.per_question.append(
            QuestionResult(
                question_id=q.id,
                gold=q.gold,
                candidate=recall.content,
                score=score,
                verdict=verdict,
                category=q.question_type,
                rationale=rationale,
                recall_diagnostics=diagnostics,
            )
        )
    return result


def run_rag_mode(
    instance: BenchmarkInstance,
    *,
    system_factory,
    reader,
    judge,
    judgment_cache,
    force_judge_categories: Optional[Iterable[str]] = None,
) -> InstanceResult:
    """Observe each turn, call ``recall(question)``, pass through the
    shared reader step, score the reader's output by category."""
    system = system_factory()
    _ingest_history(system, instance)
    result = InstanceResult(instance_id=instance.instance_id)
    for q in instance.questions:
        recall = system.recall(q.cue, {})
        answer = reader.answer(
            question=q.cue,
            recall_content=recall.content,
            fragments=recall.fragments_used,
            question_type=q.question_type,
        )
        score, verdict, rationale = _score(
            category=q.question_type,
            gold=q.gold,
            candidate=answer,
            judge=judge,
            judgment_cache=judgment_cache,
            benchmark="locomo",
            instance_id=instance.instance_id,
            question_id=q.id,
            question=q.cue,
            force_judge_categories=force_judge_categories,
        )
        diagnostics = _recall_diagnostics(question=q, fragments=recall.fragments_used)
        result.per_question.append(
            QuestionResult(
                question_id=q.id,
                gold=q.gold,
                candidate=answer,
                score=score,
                verdict=verdict,
                category=q.question_type,
                rationale=rationale,
                recall_diagnostics=diagnostics,
            )
        )
    return result
