"""Common intermediate representation for the three benchmark adapters.

Per-benchmark loaders (LongMemEval / LoCoMo / EpBench — step 2+) each convert
their native schema into `BenchmarkInstance` so the runner, SUT invocation, and
dual-mode result reporting are benchmark-agnostic.

Design notes:
  - A `BenchmarkInstance` is one evaluation unit: the history (conversation
    turns to `observe()`) + the questions to answer afterwards. For LongMemEval
    this is one of 500 user-histories; for LoCoMo it is one of 10 ultra-long
    dialogues; for EpBench it is one of ~2000 synthetic story-event sequences.
  - Instance isolation is non-negotiable: each instance gets a fresh
    `CompanionSystem()`. The runner asserts `fragment_count == 0` on
    instantiation to catch memory-of-memory contamination (benchmarks.md §148).
  - The loader is a `Protocol` so benchmark-specific loaders live in sibling
    modules without a base-class hierarchy. Each loader exposes `name` and
    `iter_instances(root)` — a thin surface so a Phase 2 benchmark can plug in
    without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal, Optional, Protocol, runtime_checkable


BenchmarkName = Literal["longmemeval", "locomo", "epbench"]


@dataclass(frozen=True)
class BenchmarkTurn:
    """One conversation turn in an instance history — fed to `observe()`."""

    timestamp: datetime
    speaker: str
    content: str
    session_id: str = ""


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One question/answer pair.

    `gold` is the reference answer (string for LongMemEval/LoCoMo short-form;
    structured dict for EpBench quadruples — adapters coerce to str for the
    cache key but may expose structured form via `metadata`).

    `question_type` is benchmark-specific (e.g. LongMemEval "single-session-user"
    vs "multi-session-abstention"). The runner forwards it to the scorer so
    per-category breakdowns are possible without parsing the raw JSON twice.
    """

    id: str
    cue: str
    gold: str
    question_type: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkInstance:
    """One evaluation unit — history + questions.

    `name` is the benchmark the instance came from; `instance_id` is unique
    within the benchmark. The combination (name, instance_id) is what the
    judgment cache keys on, together with the gold+candidate SHA.
    """

    name: BenchmarkName
    instance_id: str
    history: tuple[BenchmarkTurn, ...]
    questions: tuple[BenchmarkQuestion, ...]
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class BenchmarkLoader(Protocol):
    """Per-benchmark loader. Implementations live in sibling modules.

    `iter_instances` streams rather than returning a list because LongMemEval
    at 500 instances × 50 sessions does not fit comfortably in memory as a
    single list of fully-materialized histories; the runner consumes one at a
    time.
    """

    name: BenchmarkName

    def iter_instances(self, root: Optional[Path] = None) -> Iterator[BenchmarkInstance]: ...
