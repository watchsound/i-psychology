"""EpBench adapter — events + list-recall questions + F1 scoring.

**Native format reverse-engineered from the EpBench repo** (ahstat/episodic-memory-benchmark,
ICLR 2025). Load-bearing observations:

- An EpBench instance is a *book*: a sequence of chapters, each ultimately
  realized as a paragraph. Each chapter has a 4-tuple of (temporal, spatial,
  entity, content) indices into the universe's dimension pools.
- A question's `correct_answer` is a formatted list string ("item1, item2,
  and item3"); `n_items_correct_answer` gives the gold count; `retrieval_type`
  names the question template family (e.g. "Full event details").
- Scoring is **simple recall F1** over the parsed item sets.

**Format shim.** The real repo ships questions in `df_qa.parquet`. We ingest a
JSONL shim one-instance-per-line to avoid dragging pyarrow into the main
dependency tree; a `parquet -> jsonl` converter is a thin side artifact (not
yet written — flagged as the real-data integration task in
docs/experiments/phase-1-benchmark-adapters/option-alpha/README.md).

**Direct-mode runner** (Option α path): for each instance, observe each event's
content string as a `BenchmarkTurn`, then at each question call
`cs.recall(question.cue)` and score `recall.content` against the gold list via
F1. 3-seed median per [docs/phases/phase-1-plan.md §5] seed discipline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from psychology_base.eval.benchmarks.instance import (
    BenchmarkInstance,
    BenchmarkQuestion,
    BenchmarkTurn,
)


# ---------- answer-string parsing ----------

# EpBench formats gold as "item1, item2, and item3" (Oxford comma) or
# "item1 and item2" for pairs or just "item1" for singletons. We parse all three.
_AND_SEP = re.compile(r",?\s+and\s+", flags=re.IGNORECASE)


def parse_answer_items(answer: str) -> tuple[str, ...]:
    """Parse a formatted-list answer string into an ordered tuple of items.

    Normalization is deliberately mild — EpBench's gold strings carry mixed
    case (place names, entity names) and we lower-case + strip to merge
    trivial variants without crushing meaningful tokens.
    """
    if not answer:
        return ()
    # Strip wrapping quotes some generators add.
    s = answer.strip().strip('"').strip("'")
    if not s:
        return ()
    # Split on "and" first (handles the final conjunction), then commas.
    parts: list[str] = []
    for chunk in _AND_SEP.split(s):
        parts.extend(chunk.split(","))
    cleaned = tuple(p.strip().lower() for p in parts if p.strip())
    return cleaned


def set_f1(gold: tuple[str, ...], candidate: tuple[str, ...]) -> float:
    """Simple-recall F1 over item sets. Returns 0.0 when either side is empty
    and the sides disagree; 1.0 when both empty (vacuous match)."""
    gold_set = set(gold)
    cand_set = set(candidate)
    if not gold_set and not cand_set:
        return 1.0
    if not gold_set or not cand_set:
        return 0.0
    tp = len(gold_set & cand_set)
    if tp == 0:
        return 0.0
    precision = tp / len(cand_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


# ---------- JSONL shim schema ----------


@dataclass(frozen=True)
class _ShimEvent:
    index: int
    timestamp: str  # ISO8601 or EpBench-native string
    place: str
    entities: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class _ShimQuestion:
    id: str
    question: str
    correct_answer: str
    retrieval_type: str = ""
    n_items: int = 0


def _parse_shim_timestamp(ts: str) -> datetime:
    """Best-effort timestamp parse. EpBench universes vary (modern dates in
    news-style, fictional dates in sci-fi); we fall back to a stable epoch when
    the string isn't ISO-parseable, so the harness never crashes on exotic
    date formats — the SUT isn't told what the string means anyway."""
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime(2000, 1, 1)


def _event_to_turn(event: _ShimEvent) -> BenchmarkTurn:
    # EpBench chapters are narrator-perspective paragraphs. We prepend the
    # structured scaffold (place + entities) to the content so a reconstructive
    # memory substrate has token overlap to key on at recall time.
    scaffold = f"[place={event.place}; entities={', '.join(event.entities)}] "
    return BenchmarkTurn(
        timestamp=_parse_shim_timestamp(event.timestamp),
        speaker="narrator",
        content=scaffold + event.content,
        session_id=f"chapter-{event.index}",
    )


def _question_to_record(q: _ShimQuestion) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=q.id,
        cue=q.question,
        gold=q.correct_answer,
        question_type=q.retrieval_type,
        metadata={"n_items": q.n_items},
    )


# ---------- loader ----------


@dataclass
class EpBenchLoader:
    """Streams EpBench instances from a JSONL shim file.

    Each line in the shim is one instance:
        {"instance_id": "...", "events": [...], "questions": [...]}

    where each event has {index, timestamp, place, entities, content} and
    each question has {id, question, correct_answer, retrieval_type, n_items}.
    """

    name: str = "epbench"
    shim_path: Optional[Path] = None

    def iter_instances(self, root: Optional[Path] = None) -> Iterator[BenchmarkInstance]:
        path = self.shim_path or (root / "epbench_shim.jsonl" if root else None)
        if path is None or not Path(path).exists():
            return
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                events = [
                    _ShimEvent(
                        index=int(e["index"]),
                        timestamp=str(e["timestamp"]),
                        place=str(e["place"]),
                        entities=tuple(str(x) for x in e.get("entities", ())),
                        content=str(e["content"]),
                    )
                    for e in raw.get("events", [])
                ]
                questions = [
                    _ShimQuestion(
                        id=str(q["id"]),
                        question=str(q["question"]),
                        correct_answer=str(q["correct_answer"]),
                        retrieval_type=str(q.get("retrieval_type", "")),
                        n_items=int(q.get("n_items", 0)),
                    )
                    for q in raw.get("questions", [])
                ]
                yield BenchmarkInstance(
                    name="epbench",
                    instance_id=str(raw["instance_id"]),
                    history=tuple(_event_to_turn(e) for e in events),
                    questions=tuple(_question_to_record(q) for q in questions),
                    metadata={"source_universe": raw.get("universe", "unknown")},
                )


# ---------- direct-mode runner + scorer ----------


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    gold: str
    candidate: str
    f1: float
    retrieval_type: str = ""


@dataclass
class InstanceResult:
    instance_id: str
    per_question: list[QuestionResult] = field(default_factory=list)

    @property
    def mean_f1(self) -> float:
        if not self.per_question:
            return 0.0
        return sum(r.f1 for r in self.per_question) / len(self.per_question)


def score_candidate(gold: str, candidate: str) -> float:
    return set_f1(parse_answer_items(gold), parse_answer_items(candidate))


def run_direct_mode(
    instance: BenchmarkInstance,
    *,
    system_factory,
) -> InstanceResult:
    """Observe each event, then recall per question. Returns per-question F1.

    `system_factory` is a zero-arg callable returning a fresh `MemorySystem`.
    The runner asserts fragment_count==0 at start (instance-isolation guard
    from the plan §5 / benchmarks.md §148) when the system supports inspect().
    """
    system = system_factory()
    probe = getattr(system, "inspect", None)
    if callable(probe):
        info = probe()
        if info is not None and info.get("fragment_count", 0) != 0:
            raise RuntimeError(
                f"Instance isolation violated: fragment_count="
                f"{info.get('fragment_count')} at instance start"
            )
    from psychology_base.eval.adapters import Observation

    for turn in instance.history:
        system.observe(
            Observation(
                timestamp=turn.timestamp,
                speaker=turn.speaker,
                content=turn.content,
                affect={},
                tags=(),
            )
        )

    result = InstanceResult(instance_id=instance.instance_id)
    for q in instance.questions:
        recall = system.recall(q.cue, {})
        f1 = score_candidate(q.gold, recall.content)
        result.per_question.append(
            QuestionResult(
                question_id=q.id,
                gold=q.gold,
                candidate=recall.content,
                f1=f1,
                retrieval_type=q.question_type,
            )
        )
    return result
