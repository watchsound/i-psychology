"""Judge interface + file-based judgment cache.

The three benchmarks have free-form or near-free-form answers that require an
LLM-as-judge step for honest scoring (LongMemEval multi-session, LoCoMo
open-domain, EpBench partially). Judge calls are expensive (~$0.01-0.05/call)
and non-deterministic across model versions, so every judgment is cached on a
stable key.

Cache key (SHA-256 of a canonical JSON blob):
    (benchmark, instance_id, question_id, gold, candidate, judge_id, judge_prompt_sha)

- `judge_id` includes model + version (e.g. "gpt-4o-2024-11-20"). Changing it
  invalidates the cache naturally.
- `judge_prompt_sha` prevents silent prompt drift from re-using stale
  judgments — if the prompt template changes, the key does too.

The actual LLM call is pluggable via the `JudgeClient` protocol. For CI /
deterministic-smoke tests the harness uses `NullJudge` which refuses to judge
(returns None); real runs plug a provider-specific client at step 3.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "PSYCHOLOGY_BASE_JUDGMENT_CACHE",
        str(Path.home() / ".cache" / "psychology_base" / "judgments"),
    )
)


@dataclass(frozen=True)
class JudgmentKey:
    benchmark: str
    instance_id: str
    question_id: str
    gold: str
    candidate: str
    judge_id: str
    judge_prompt_sha: str

    def sha(self) -> str:
        blob = json.dumps(
            {
                "benchmark": self.benchmark,
                "instance_id": self.instance_id,
                "question_id": self.question_id,
                "gold": self.gold,
                "candidate": self.candidate,
                "judge_id": self.judge_id,
                "judge_prompt_sha": self.judge_prompt_sha,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class Judgment:
    """Outcome of a judge call. `score` is in [0, 1]; `verdict` is a short
    categorical label the judge produced (e.g. "correct" / "partial" /
    "incorrect"); `rationale` is free-form for later inspection."""

    score: float
    verdict: str
    rationale: str = ""


@runtime_checkable
class JudgeClient(Protocol):
    """LLM-judge call surface. Implementations wrap a provider SDK."""

    judge_id: str
    prompt_sha: str

    def judge(
        self,
        *,
        benchmark: str,
        question: str,
        gold: str,
        candidate: str,
        question_type: str = "",
    ) -> Judgment: ...


class NullJudge:
    """Refuses to judge. Used in CI smoke tests + deterministic-only runs.

    Deterministic scorers (exact-match, F1, EpBench quadruple-F1) do not call
    the judge, so NullJudge is sufficient for the step-1 gate (which does not
    exercise free-form scoring) and for the step-4 deterministic smoke subset.
    """

    judge_id = "null-judge"
    prompt_sha = "null"

    def judge(
        self,
        *,
        benchmark: str,
        question: str,
        gold: str,
        candidate: str,
        question_type: str = "",
    ) -> Judgment:
        raise RuntimeError(
            "NullJudge cannot judge free-form answers. "
            "Supply a real JudgeClient (step 3+) or use a deterministic scorer only."
        )


# ---------- LLM-backed judge (step 3) ----------

# Short-form Q&A scoring prompt. Locked at step 3 — changing it invalidates
# every cached judgment, so edits should be deliberate. The scoring rubric
# intentionally allows partial credit for near-correct answers (common in
# LongMemEval multi-session: candidate names 2 of 3 items, or paraphrases
# the gold); but reserves 1.0 for answers that would satisfy a human grader
# with no reservations.
_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for a long-term memory benchmark. Compare a "
    "candidate answer to a reference gold answer and return a score and a "
    "short verdict.\n\n"
    "Scoring:\n"
    "- 1.0 'correct': the candidate conveys the same information as the gold, "
    "ignoring phrasing. Paraphrases, extra context, and minor formatting "
    "differences are fine as long as the substance matches.\n"
    "- 0.5 'partial': the candidate is partially right (e.g. names some but "
    "not all required items; captures the fact but misses a key qualifier).\n"
    "- 0.0 'incorrect': the candidate is wrong, contradicts the gold, or says "
    "'not stated' when the gold is substantive. Also 0.0 if the gold is 'not "
    "stated' / abstention and the candidate fabricates a specific answer.\n\n"
    "Respond with exactly one JSON object: "
    '{"score": <number>, "verdict": "<correct|partial|incorrect>", '
    '"rationale": "<one sentence>"}'
)

_JUDGE_PROMPT_VERSION = "v1-2026-10-05"


def _judge_prompt_sha() -> str:
    return hashlib.sha256(
        (_JUDGE_SYSTEM_PROMPT + "|" + _JUDGE_PROMPT_VERSION).encode("utf-8")
    ).hexdigest()[:16]


@dataclass
class LLMJudge:
    """LLM-as-judge wrapping the project's LLM router.

    Routes to the ADVANCED tier (qwen-max under the default Tongyi config)
    because judge calls are the primary trust surface for free-form scoring
    — a wrong judgment invalidates downstream analysis in a way a wrong
    reader completion does not. Judge-model drift is managed by pinning
    ``judge_id`` into the cache key: changing router config or model
    naturally invalidates stale judgments.
    """

    router: "object"  # psychology_base.llm.router.LLMRouter — avoid circular import
    judge_id: str = ""
    prompt_sha: str = ""
    max_tokens: int = 256
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.judge_id:
            # Resolve the ADVANCED-tier model id from the router's active
            # provider. Changing providers naturally bumps the id.
            try:
                from psychology_base.llm.types import TaskTier

                _, decision = self.router.route(TaskTier.ADVANCED)
                self.judge_id = f"{decision.provider}:{decision.model_id}"
            except Exception:
                self.judge_id = "llm-judge:unknown"
        if not self.prompt_sha:
            self.prompt_sha = _judge_prompt_sha()

    def judge(
        self,
        *,
        benchmark: str,
        question: str,
        gold: str,
        candidate: str,
        question_type: str = "",
    ) -> Judgment:
        from psychology_base.llm.types import ChatMessage, TaskTier

        user = (
            f"Benchmark: {benchmark}\n"
            f"Question type: {question_type or 'unspecified'}\n"
            f"Question: {question}\n"
            f"Gold answer: {gold}\n"
            f"Candidate answer: {candidate}\n\n"
            "Grade the candidate against the gold using the rubric. "
            "Respond with the JSON object only."
        )
        completion = self.router.chat(
            [
                ChatMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user),
            ],
            tier=TaskTier.ADVANCED,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return _parse_judge_response(completion.content)


def _parse_judge_response(content: str) -> Judgment:
    """Parse the judge's JSON response. Robust to code-fence wrapping and
    stray prose because models occasionally reach for both; falls back to a
    0.0 ``incorrect`` judgment with the raw content as rationale when
    nothing parses — so a flaky judge call never raises mid-run, it just
    scores zero and leaves a diagnostic breadcrumb.
    """
    stripped = (content or "").strip()
    # Peel ```json ... ``` or ``` ... ``` fencing.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    # First-brace extraction fallback.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start : end + 1]
    try:
        obj = json.loads(stripped)
        score = float(obj.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        verdict = str(obj.get("verdict", "incorrect"))
        rationale = str(obj.get("rationale", ""))
        return Judgment(score=score, verdict=verdict, rationale=rationale)
    except (ValueError, TypeError, json.JSONDecodeError):
        return Judgment(
            score=0.0,
            verdict="unparsed",
            rationale=f"judge_response_unparsed: {content[:200]!r}",
        )


@dataclass
class JudgmentCache:
    """File-backed cache. One JSON file per key sha, under `cache_dir`.

    Chose per-file over a single JSON index because concurrent-run races are
    annoying with a single file and we expect this cache to be re-used across
    many iterations of the same benchmark; per-file writes are atomic enough
    (rename-after-write) and git-friendly for debugging ("show me the judgment
    for question_42").
    """

    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key_sha: str) -> Path:
        # Shard by first two hex chars so a directory never grows past 65k entries.
        return self.cache_dir / key_sha[:2] / f"{key_sha}.json"

    def get(self, key: JudgmentKey) -> Optional[Judgment]:
        path = self._path(key.sha())
        if not path.exists():
            self.misses += 1
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.hits += 1
        return Judgment(
            score=float(raw["score"]),
            verdict=str(raw["verdict"]),
            rationale=str(raw.get("rationale", "")),
        )

    def put(self, key: JudgmentKey, judgment: Judgment) -> None:
        path = self._path(key.sha())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": {
                "benchmark": key.benchmark,
                "instance_id": key.instance_id,
                "question_id": key.question_id,
                "judge_id": key.judge_id,
                "judge_prompt_sha": key.judge_prompt_sha,
            },
            "score": judgment.score,
            "verdict": judgment.verdict,
            "rationale": judgment.rationale,
        }
        # Rename-after-write for atomicity on crash mid-write.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)

    def judge_cached(
        self,
        *,
        client: JudgeClient,
        benchmark: str,
        instance_id: str,
        question_id: str,
        question: str,
        gold: str,
        candidate: str,
        question_type: str = "",
    ) -> Judgment:
        key = JudgmentKey(
            benchmark=benchmark,
            instance_id=instance_id,
            question_id=question_id,
            gold=gold,
            candidate=candidate,
            judge_id=client.judge_id,
            judge_prompt_sha=client.prompt_sha,
        )
        hit = self.get(key)
        if hit is not None:
            return hit
        verdict = client.judge(
            benchmark=benchmark,
            question=question,
            gold=gold,
            candidate=candidate,
            question_type=question_type,
        )
        self.put(key, verdict)
        return verdict
