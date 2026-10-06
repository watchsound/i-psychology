"""RAG-mode reader for benchmark adapters (plan §5 step 3, Option γ-RAG).

The reader is the post-recall composition step: given
``(question, recall.content, fragments_used)`` it returns a short-form
answer suitable for judge comparison against the gold. This is the Mem0
pattern — retrieve-then-compose — layered on top of our reconstruction
substrate.

Design notes:
  - The reader is intentionally minimal. The plan's §8 "reader-prompt as
    hidden parameter" risk calls out that moving reader-prompt knobs can
    silently absorb substrate wins or losses. We lock a conservative prompt
    at step 3 and treat later edits as substrate iteration.
  - The reader does NOT re-retrieve, re-rank, or augment. It composes an
    answer strictly from the material the substrate chose to surface. If
    the substrate missed the relevant fragments, the reader has nothing to
    cover for — which is the honest signal we want.
  - The reader uses the LLM router's ADVANCED tier (qwen-max under the
    default Tongyi config). Reading / composition is a planning-grade task
    where a wrong answer silently invalidates the whole benchmark number;
    this is exactly the slot the ADVANCED tier was reserved for.
  - A ``MockReader`` is provided for deterministic smoke tests that don't
    want to spend tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from psychology_base.eval.adapters import Fragment
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


# ---------- protocol ----------


@runtime_checkable
class Reader(Protocol):
    """Compose a short-form answer from a question + the substrate's recall."""

    def answer(
        self,
        *,
        question: str,
        recall_content: str,
        fragments: Sequence[Fragment],
        question_type: str = "",
    ) -> str: ...


# ---------- locked prompt (step-3) ----------


_SYSTEM_PROMPT = (
    "You answer questions using ONLY the context provided by a memory system. "
    "The context is the system's own reconstruction plus a short list of "
    "memory fragments. If the context does not contain the information needed, "
    "say exactly 'not stated' — do not guess, do not use outside knowledge.\n\n"
    "Answer rules:\n"
    "1. Prefer one short line. Match the specificity of the question — a date "
    "question gets a date, a list question gets a comma-separated list.\n"
    "2. Do not restate the question. Do not include rationale or caveats.\n"
    "3. If the question asks about something not covered by the context, "
    "answer exactly 'not stated'. Do not say 'I don't know' or 'the context "
    "doesn't say' or any variant."
)


def _format_fragments(fragments: Sequence[Fragment], *, limit: int = 50) -> str:
    """Render fragments in a compact prompt-friendly block. Capped at
    ``limit`` to keep the reader-prompt bounded regardless of recall depth."""
    if not fragments:
        return "(no fragments surfaced)"
    lines = []
    for i, frag in enumerate(fragments[:limit]):
        ts = frag.timestamp.isoformat() if frag.timestamp else ""
        tag = f" [{','.join(frag.tags)}]" if frag.tags else ""
        lines.append(f"- ({i}){tag} {ts}: {frag.content}")
    if len(fragments) > limit:
        lines.append(f"- ... ({len(fragments) - limit} more fragments omitted)")
    return "\n".join(lines)


# ---------- LLM-backed reader ----------


@dataclass
class LLMReader:
    """ADVANCED-tier reader; locked prompt (see module docstring)."""

    router: LLMRouter
    tier: TaskTier = TaskTier.ADVANCED
    max_tokens: int = 256
    temperature: float = 0.0

    def answer(
        self,
        *,
        question: str,
        recall_content: str,
        fragments: Sequence[Fragment],
        question_type: str = "",
    ) -> str:
        frag_block = _format_fragments(fragments)
        user = (
            f"Question: {question}\n\n"
            f"Memory system reconstruction:\n{recall_content or '(empty reconstruction)'}\n\n"
            f"Memory fragments surfaced by the system:\n{frag_block}\n\n"
            f"Short-form answer:"
        )
        completion = self.router.chat(
            [
                ChatMessage(role="system", content=_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user),
            ],
            tier=self.tier,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return completion.content.strip()


# ---------- mock reader ----------


@dataclass
class MockReader:
    """Deterministic reader for tests. Returns the first fragment's content,
    or the reconstruction if no fragments, or 'not stated' if both are empty.
    Lets smoke tests exercise the RAG pipeline without any LLM calls."""

    def answer(
        self,
        *,
        question: str,
        recall_content: str,
        fragments: Sequence[Fragment],
        question_type: str = "",
    ) -> str:
        if fragments:
            return fragments[0].content
        if recall_content:
            return recall_content
        return "not stated"
