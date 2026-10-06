"""Phase 1.5 fact extraction — SIMPLE-tier LLM over a fragment window.

Per [phase-1.5-build-plan.md](../../../docs/phases/phase-1.5-build-plan.md)
§5 step 2:
  - Locked prompt (changing it mid-phase invalidates the extraction cache
    and makes gate numbers non-comparable).
  - Pipe-separated-text output format; parse-stability was the explicit
    reason over JSON mode (SIMPLE-tier models produce unreliable JSON).
  - Pure function: returns ``list[FactOperation]``; mutations happen in
    ``consolidation.py``.
  - Best-effort parsing: lines that fail to parse are dropped, logged by
    caller — crashing on a malformed line would discard the whole window.

Prompt-output grammar::

    ADD    | <entity> | <relation> | <value> | <confidence>
    UPDATE | <target_fact_id> | <entity> | <relation> | <value> | <confidence>
    NOOP   | <target_fact_id>

Confidence is a float 0.0..1.0.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from psychology_base.companion.facts import Fact, FactOp, FactOperation
from psychology_base.eval.adapters import Fragment
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


logger = logging.getLogger(__name__)


MAX_FACTS_PER_WINDOW = 80
MAX_PRIOR_FACTS_INCLUDED = 20


EXTRACTION_SYSTEM_PROMPT = """You read a short conversation window and emit factual claims.

Output format — one line per fact, no commentary, no preamble:
  ADD    | entity | relation | value | confidence
  UPDATE | target_fact_id | entity | relation | value | confidence
  NOOP   | target_fact_id

Rules:
- entity: short noun phrase, the subject of the fact (e.g. "Jennifer", "user", "user's mother")
- relation: short predicate, verb or short phrase (e.g. "owns", "lives_in", "studies", "hobby", "pet_name")
- value: short object. PREFER concrete named entities over abstract summaries:
    * places (cities, countries, buildings): "Sweden", "Boston", "the aquarium"
    * named items (books, movies, songs, products): "Charlotte's Web", "Nothing is Impossible"
    * specific hobbies/activities: "pottery", "running", "piano"
    * specific people, pets, foods, plants with their names
    * concrete numbers, dates, durations: "4 years ago", "two kids"
  AVOID abstract values like "support_group", "self_care", "feels_good" when a concrete name is available in the text.
- When the speaker lists MULTIPLE specific items (hobbies, places, foods, book titles),
  emit ONE fact PER ITEM rather than one fact with a joined list. Example: if the user
  says "I like running, pottery and painting", emit three ADD facts (one per hobby).
- confidence: 0.0..1.0 float, your subjective reliability
- Only facts that are EXPLICIT in the conversation window
- Omit speculation, meta-commentary, pleasantries, questions
- If a PRIOR FACT contradicts a new fact, emit UPDATE <prior_fact_id> with the new values
- If a PRIOR FACT is restated unchanged, emit NOOP <prior_fact_id>
- Return at most {max_facts} lines
"""


EXTRACTION_USER_TEMPLATE = """PRIOR FACTS (id | entity | relation | value):
{prior_block}

CONVERSATION WINDOW:
{window_block}

Emit facts now, one per line, in the output format."""


def _format_window(fragments: Sequence[Fragment]) -> str:
    lines: list[str] = []
    for f in fragments:
        speaker = ""
        for tag in f.tags:
            if tag.startswith("speaker:"):
                speaker = tag.split(":", 1)[1]
                break
        prefix = f"{speaker}: " if speaker else ""
        lines.append(f"{prefix}{f.content}")
    return "\n".join(lines) if lines else "(empty)"


def _format_prior_facts(facts: Sequence[Fact], limit: int) -> str:
    if not facts:
        return "(none)"
    rows = [
        f"{f.id} | {f.entity} | {f.relation} | {f.value}"
        for f in facts[-limit:]
    ]
    return "\n".join(rows)


_FLOAT_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _parse_confidence(raw: str) -> Optional[float]:
    raw = raw.strip()
    if not _FLOAT_RE.match(raw):
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    if v < 0.0 or v > 1.0:
        return None
    return v


def _parse_line(
    line: str,
    *,
    tick_count: int,
    source_fragment_ids: tuple[str, ...],
) -> Optional[FactOperation]:
    """Parse one output line into a FactOperation. Returns None if the
    line is blank, a comment, or malformed."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = [p.strip() for p in stripped.split("|")]
    if not parts:
        return None

    op_token = parts[0].upper()

    if op_token == "ADD" and len(parts) == 5:
        _, entity, relation, value, confidence_raw = parts
        conf = _parse_confidence(confidence_raw)
        if conf is None or not entity or not relation or not value:
            return None
        return FactOperation(
            op=FactOp.ADD,
            fact=Fact(
                id="",
                entity=entity,
                relation=relation,
                value=value,
                confidence=conf,
                source_fragment_ids=source_fragment_ids,
                created_tick=tick_count,
                updated_tick=tick_count,
            ),
        )

    if op_token == "UPDATE" and len(parts) == 6:
        _, target_id, entity, relation, value, confidence_raw = parts
        conf = _parse_confidence(confidence_raw)
        if conf is None or not target_id or not entity or not relation or not value:
            return None
        return FactOperation(
            op=FactOp.UPDATE,
            fact=Fact(
                id=target_id,
                entity=entity,
                relation=relation,
                value=value,
                confidence=conf,
                source_fragment_ids=source_fragment_ids,
                created_tick=tick_count,
                updated_tick=tick_count,
            ),
            target_fact_id=target_id,
        )

    if op_token == "NOOP" and len(parts) >= 2:
        target_id = parts[1]
        if not target_id:
            return None
        return FactOperation(op=FactOp.NOOP, target_fact_id=target_id)

    return None


def parse_extraction_output(
    raw: str,
    *,
    tick_count: int,
    source_fragment_ids: tuple[str, ...],
    max_facts: int = MAX_FACTS_PER_WINDOW,
) -> list[FactOperation]:
    """Parse the LLM's raw text output into a list of operations.

    Lines that fail to parse are dropped (logged at debug level). The cap
    at ``max_facts`` is enforced on successfully parsed operations only —
    a bad line doesn't count toward the budget."""
    ops: list[FactOperation] = []
    for line in raw.splitlines():
        if len(ops) >= max_facts:
            break
        op = _parse_line(
            line,
            tick_count=tick_count,
            source_fragment_ids=source_fragment_ids,
        )
        if op is None:
            if line.strip():
                logger.debug("extraction: dropped unparseable line: %r", line)
            continue
        ops.append(op)
    return ops


def _relevant_prior_facts(
    prior_facts: Sequence[Fact], window: Sequence[Fragment]
) -> list[Fact]:
    """Pre-filter prior facts by entity-string presence in the window.

    Cheap heuristic — avoids shipping the full fact store to the LLM when
    only a handful of entities are mentioned in the window. Falls back to
    the most-recently-updated facts if the entity filter is empty."""
    if not prior_facts:
        return []
    text = " ".join(f.content for f in window).lower()
    entity_match = [f for f in prior_facts if f.entity.lower() in text]
    if entity_match:
        return sorted(entity_match, key=lambda f: f.updated_tick)[-MAX_PRIOR_FACTS_INCLUDED:]
    return sorted(prior_facts, key=lambda f: f.updated_tick)[-MAX_PRIOR_FACTS_INCLUDED:]


def extract_facts(
    window: Sequence[Fragment],
    prior_facts: Sequence[Fact],
    router: LLMRouter,
    *,
    tick_count: int = 0,
    temperature: float = 0.0,
) -> list[FactOperation]:
    """Extract facts from a fragment window. Pure w.r.t. the FactStore —
    returns operations the caller applies.

    Empty window → empty op list; no LLM call made.
    """
    if not window:
        return []

    relevant_priors = _relevant_prior_facts(prior_facts, window)
    system = EXTRACTION_SYSTEM_PROMPT.format(max_facts=MAX_FACTS_PER_WINDOW)
    user = EXTRACTION_USER_TEMPLATE.format(
        prior_block=_format_prior_facts(relevant_priors, MAX_PRIOR_FACTS_INCLUDED),
        window_block=_format_window(window),
    )
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]

    completion = router.chat(
        messages,
        tier=TaskTier.SIMPLE,
        temperature=temperature,
        # Each fact line is ≈ 30 tokens; MAX_FACTS_PER_WINDOW=80 ⇒ ~2400
        # tokens of output. A 3600-token cap prevents a chatty model from
        # spiraling into prose while leaving headroom for occasional long
        # value strings.
        max_tokens=3600,
    )

    source_ids = tuple(f.id for f in window)
    return parse_extraction_output(
        completion.content,
        tick_count=tick_count,
        source_fragment_ids=source_ids,
    )


__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_USER_TEMPLATE",
    "MAX_FACTS_PER_WINDOW",
    "MAX_PRIOR_FACTS_INCLUDED",
    "extract_facts",
    "parse_extraction_output",
]
