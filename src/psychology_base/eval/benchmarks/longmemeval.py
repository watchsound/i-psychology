"""LongMemEval adapter — multi-session user histories + judge-scored answers.

**Native format reverse-engineered from the LongMemEval repo**
(xiaowu0162/LongMemEval, arXiv 2404.13940). Load-bearing observations:

- A LongMemEval instance is one *user history*: a list of chat sessions, each
  session a list of `{role, content}` turns. One question is asked after the
  entire history is observed. Five question types: single-session-user,
  single-session-assistant, multi-session, temporal-reasoning, knowledge-update,
  abstention.
- The gold answer is short-form free text (one line to one sentence). Scoring
  is `gpt-4o-as-judge` against the gold — no exact-match, no F1.
- The published Mem0 94.4% is on the "oracle session" split; the "full history"
  split is harder. Our Phase-1 floor target is 90% on either split, so loaders
  support both. The `metadata` on each `BenchmarkInstance` carries
  `oracle_session_ids` when the split exposes them, so a Phase-2 run can
  restrict ingestion.

**Format shim.** The real repo ships instances as a single large JSON file
(~hundreds of MB). We ingest a JSONL shim one-instance-per-line to keep
memory flat and support streaming. A `json → jsonl` converter is a thin
side artifact (not written here — flagged alongside the EpBench
parquet→jsonl converter as the real-data integration task).

JSONL shim schema (one object per line):
    {
      "question_id": "...",
      "question_type": "multi-session",
      "question": "...",
      "answer": "...",
      "question_date": "YYYY-MM-DDTHH:MM:SS",   # optional, used for the
                                                  # cue if the question is
                                                  # temporal
      "haystack_sessions": [                     # list of sessions
        [{"role":"user","content":"..."},
         {"role":"assistant","content":"..."},
         ...],
        ...
      ],
      "haystack_session_ids": ["s0", "s1", ...], # 1:1 with haystack_sessions
      "haystack_dates": ["YYYY-MM-DDTHH:MM:SS", ...],  # 1:1 with sessions
      "answer_session_ids": ["s7", "s12"]        # oracle labels; optional
    }

**Direct-mode runner** (Option α path) and **RAG-mode runner** (Option γ):
provided here for the benchmark; the RAG reader is in `reader.py` and the
judge wiring is in `judge.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

from psychology_base.eval.benchmarks.instance import (
    BenchmarkInstance,
    BenchmarkQuestion,
    BenchmarkTurn,
)


# ---------- timestamp parse ----------


def _parse_shim_timestamp(ts: str) -> datetime:
    """Best-effort ISO parse. Falls back to a stable epoch so the harness
    never crashes on exotic dates; the SUT isn't told what the string means."""
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime(2000, 1, 1)


# ---------- shim parsing ----------


@dataclass(frozen=True)
class _ShimTurn:
    role: str
    content: str


@dataclass(frozen=True)
class _ShimSession:
    session_id: str
    date: datetime
    turns: tuple[_ShimTurn, ...]


def _parse_sessions(raw: dict) -> tuple[_ShimSession, ...]:
    sessions_raw = raw.get("haystack_sessions") or []
    ids_raw = raw.get("haystack_session_ids") or []
    dates_raw = raw.get("haystack_dates") or []
    out: list[_ShimSession] = []
    for idx, session in enumerate(sessions_raw):
        sid = str(ids_raw[idx]) if idx < len(ids_raw) else f"session-{idx}"
        date = (
            _parse_shim_timestamp(dates_raw[idx])
            if idx < len(dates_raw)
            else datetime(2000, 1, 1)
        )
        turns = tuple(
            _ShimTurn(
                role=str(t.get("role", "user")),
                content=str(t.get("content", "")),
            )
            for t in session
        )
        out.append(_ShimSession(session_id=sid, date=date, turns=turns))
    return tuple(out)


def _sessions_to_turns(sessions: tuple[_ShimSession, ...]) -> tuple[BenchmarkTurn, ...]:
    """Flatten sessions → turns. Within a session, turn N is observed one
    minute after turn N-1 so the SUT's temporal substrate sees a plausible
    sub-session cadence; sessions themselves are anchored at
    ``haystack_dates[i]``.
    """
    out: list[BenchmarkTurn] = []
    for session in sessions:
        for i, turn in enumerate(session.turns):
            out.append(
                BenchmarkTurn(
                    timestamp=session.date + timedelta(minutes=i),
                    speaker=turn.role,
                    content=turn.content,
                    session_id=session.session_id,
                )
            )
    return tuple(out)


# ---------- loader ----------


@dataclass
class LongMemEvalLoader:
    """Streams LongMemEval instances from a JSONL shim file.

    One instance per line. See module docstring for the schema.
    """

    name: str = "longmemeval"
    shim_path: Optional[Path] = None

    def iter_instances(self, root: Optional[Path] = None) -> Iterator[BenchmarkInstance]:
        path = self.shim_path or (
            root / "longmemeval_shim.jsonl" if root else None
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
                question = BenchmarkQuestion(
                    id=str(raw["question_id"]),
                    cue=str(raw["question"]),
                    gold=str(raw.get("answer", "")),
                    question_type=str(raw.get("question_type", "")),
                    metadata={
                        "question_date": str(raw.get("question_date", "")),
                        "answer_session_ids": tuple(
                            str(x) for x in raw.get("answer_session_ids") or ()
                        ),
                    },
                )
                yield BenchmarkInstance(
                    name="longmemeval",
                    instance_id=str(raw["question_id"]),
                    history=history,
                    questions=(question,),
                    metadata={
                        "num_sessions": len(sessions),
                        "num_turns": len(history),
                    },
                )


# ---------- direct-mode + RAG-mode runners ----------


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    gold: str
    candidate: str
    score: float
    verdict: str = ""
    question_type: str = ""
    rationale: str = ""


@dataclass
class InstanceResult:
    instance_id: str
    per_question: list[QuestionResult] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.per_question:
            return 0.0
        return sum(r.score for r in self.per_question) / len(self.per_question)


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
    for turn in instance.history:
        system.observe(
            Observation(
                timestamp=turn.timestamp,
                speaker=turn.speaker,
                content=turn.content,
                affect={},
                tags=(turn.session_id,) if turn.session_id else (),
            )
        )


def run_direct_mode(
    instance: BenchmarkInstance,
    *,
    system_factory,
    judge,
    judgment_cache,
) -> InstanceResult:
    """Observe each turn, call ``recall(question)``, judge the raw
    ``recall.content`` against the gold.

    This is Option γ direct-mode: the substrate's own reconstruction is the
    answer. Expected to under-report on narrow factual QA; the gap vs.
    RAG-mode is the diagnostic signal (plan §3 Option γ "pass evidence" rule).
    """
    system = system_factory()
    _ingest_history(system, instance)
    result = InstanceResult(instance_id=instance.instance_id)
    for q in instance.questions:
        recall = system.recall(q.cue, {})
        judgment = judgment_cache.judge_cached(
            client=judge,
            benchmark="longmemeval",
            instance_id=instance.instance_id,
            question_id=q.id,
            question=q.cue,
            gold=q.gold,
            candidate=recall.content,
            question_type=q.question_type,
        )
        result.per_question.append(
            QuestionResult(
                question_id=q.id,
                gold=q.gold,
                candidate=recall.content,
                score=judgment.score,
                verdict=judgment.verdict,
                question_type=q.question_type,
                rationale=judgment.rationale,
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
) -> InstanceResult:
    """Observe each turn, call ``recall(question)``, pass
    ``(question, recall.content, fragments_used)`` through the reader to get
    a short-form answer, then judge the reader's answer against the gold.

    This is Option γ RAG-mode. The reader is the strict Mem0-pattern post-step
    on top of our ``recall()`` surface — same observe path, same recall path,
    different final composition. The reader prompt is locked by step 3 so
    tuning after this point counts as substrate iteration, not adapter
    iteration (plan §8 risks: reader-prompt as hidden parameter).
    """
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
        judgment = judgment_cache.judge_cached(
            client=judge,
            benchmark="longmemeval",
            instance_id=instance.instance_id,
            question_id=q.id,
            question=q.cue,
            gold=q.gold,
            candidate=answer,
            question_type=q.question_type,
        )
        result.per_question.append(
            QuestionResult(
                question_id=q.id,
                gold=q.gold,
                candidate=answer,
                score=judgment.score,
                verdict=judgment.verdict,
                question_type=q.question_type,
                rationale=judgment.rationale,
            )
        )
    return result
