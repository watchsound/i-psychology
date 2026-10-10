"""Higher-order ToM scorer — Pillar E internal scoring loop.

**Design.** HiToM carries 5 orders (0–4) and BigToM carries order 1.
Items are projected onto :class:`FANToMItem` via the two adapters
(:func:`hitom_item_to_fantom_item` / :func:`bigtom_item_to_fantom_item`).

**Mentalistic routing — witness filter (Option A).** The memory
substrate's ``_filter_fragments_by_scope`` AND-contracts on
``subject:X`` AND ``entity:X`` tags; FANToM-style ingest only sets
``subject``, so scoped recall returns empty. Falling back to unscoped
top-K cue similarity picks the WRONG fragments (entry/exit over
object-move events) because cue similarity isn't a ToM primitive.
The correct mentalistic routing for symbolic container-world ToM is
**witness-at-time intersection** — a stateful computation over parsed
events (see :mod:`hitom_witness`). For HiToM, mentalistic context is
produced by parsing the story grammar and filtering events every
agent in the chain was in the room for. For BigToM (and any corpus
without a tracked grammar), the mentalistic path falls back to the
full rendered context so the scorer stays dataset-agnostic — the
Pillar E architectural win is HiToM-specific where the witness
intersection does real work.

The scope-filter contract bug is flagged for triage; see
``src/psychology_base/companion/system.py:411`` (flagged 2026-10-09).
A general-purpose witness primitive on the memory substrate (modeling
``location, participants, event_time`` on ``Observation``) is deferred
to Phase 9+; the HiToM witness interpreter here is the shape that
primitive should take.

Phase 5 Track 1's +0.791 FANToM delta was a happy accident of empty
recall producing correct "I don't know" on INACCESSIBLE items — the
scorer never actually surfaced scoped observations.

The Pillar E hard gate — ``accuracy_mentalistic −
accuracy_bare_llm ≥ +0.30`` at HiToM order 2 with
``accuracy_mentalistic ≥ 0.40`` and ``min_items=50`` on both paths —
is encoded in :meth:`HigherOrderToMReport.meets_pillar_e_gate`.

**Scope-selector contract.** The ADAPTER step uses Option S-flat —
``target_person`` is set to the FIRST agent in the question's chain,
not the deepest. ``target_person`` is still carried on the verdict for
reporting; the witness filter uses the FULL ``agent_chain`` metadata
field (not just ``target_person``) since mentalistic routing needs the
whole chain to intersect.

**Mutex kwarg contract.** Mirrors
``sycophancy_eval_scorer.score_sycophancy_eval`` and
``broad_factual_scorer.score_truthful_qa``: callers pass EXACTLY ONE
of ``responder`` (shared single callable, both paths) or
``item_responder`` (per-item responder factory, lets the CLI build a
fresh :class:`CompanionAgent` per conversation on the mentalistic
path). The two forms are structurally incompatible so the scorer
rejects both-at-once with a ValueError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from psychology_base.companion.perspective.grammars.hitom import (
    build_hitom_bare_prompt,
    build_hitom_perspective_prompt,
    hitom_chain_scaffold,
    hitom_choices_block,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.public_datasets.fantom import FANToMItem
from psychology_base.eval.benchmarks.public_datasets.fantom_tom_scorer import (
    AnswerJudge,
    FANToMToMReport,
    FANToMVerdict,
    Responder,
    _build_bare_llm_prompt,
    _build_mentalistic_prompt,
    lexical_answer_judge,
)
from psychology_base.eval.benchmarks.public_datasets.hitom_witness import (
    events_witnessed_by_chain,
    parse_hitom_story,
    render_witnessed_events,
)


# Private-name shims preserved for the test suite (see
# ``tests/test_higher_order_tom_scorer.py``). The authoritative builders
# live in ``psychology_base.companion.perspective.grammars.hitom``; these
# aliases exist so the HiToM protocol-conformance pins carried over from
# Pillar E v2 keep importing the names they already know.
_hitom_choices_block = hitom_choices_block
_hitom_chain_scaffold = hitom_chain_scaffold
_build_hitom_mentalistic_prompt = build_hitom_perspective_prompt
_build_hitom_bare_prompt = build_hitom_bare_prompt


_PATH_MENTALISTIC = "mentalistic"
_PATH_BARE_LLM = "bare_llm"


@dataclass(frozen=True)
class HigherOrderToMReport:
    """Per-order view over a shared :class:`FANToMToMReport`.

    ``cid_to_order`` maps conversation IDs to the HiToM/BigToM order
    integer so verdicts can be sliced by ToM depth. ``dataset_tag``
    records which sub-corpus (hitom / bigtom / combined) the report
    covers for cross-dataset comparisons in the close-out doc.
    """

    base_report: FANToMToMReport
    cid_to_order: dict[str, int]
    dataset_tag: str

    def _subset(
        self,
        *,
        path: str,
        order: Optional[int] = None,
    ) -> tuple[FANToMVerdict, ...]:
        out: list[FANToMVerdict] = []
        for v in self.base_report.verdicts:
            if v.source_path != path:
                continue
            if order is not None and self.cid_to_order.get(v.conversation_id) != order:
                continue
            out.append(v)
        return tuple(out)

    def item_count(self, *, path: str, order: Optional[int] = None) -> int:
        return len(self._subset(path=path, order=order))

    def accuracy(
        self,
        *,
        path: str,
        order: Optional[int] = None,
        drop_undecidable: bool = True,
    ) -> float:
        subset = self._subset(path=path, order=order)
        if drop_undecidable:
            subset = tuple(v for v in subset if v.label != "undecidable")
        if not subset:
            return 0.0
        correct = sum(1 for v in subset if v.label == "correct")
        return correct / len(subset)

    def undecidable_rate(
        self,
        *,
        path: str,
        order: Optional[int] = None,
    ) -> float:
        subset = self._subset(path=path, order=order)
        if not subset:
            return 0.0
        n = sum(1 for v in subset if v.label == "undecidable")
        return n / len(subset)

    def architectural_delta(self, *, order: Optional[int] = None) -> float:
        return (
            self.accuracy(path=_PATH_MENTALISTIC, order=order)
            - self.accuracy(path=_PATH_BARE_LLM, order=order)
        )

    def meets_pillar_e_gate(
        self,
        *,
        order: int = 2,
        delta_floor: float = 0.30,
        accuracy_floor: float = 0.40,
        min_items: int = 50,
        max_undecidable_rate: float = 1.0,
    ) -> bool:
        """Phase 8 Pillar E hard gate.

        At HiToM ``order=2`` with ``min_items=50`` on BOTH paths,
        require ``accuracy_mentalistic ≥ 0.40`` AND
        ``accuracy_mentalistic − accuracy_bare_llm ≥ +0.30``.

        Passing arguments: ``order=3`` is the soft criterion; ``order=1``
        is the cross-check from BigToM. ``order=0`` and ``order=4`` are
        exploratory and should not be reported as gated.
        """

        n_mental = self.item_count(path=_PATH_MENTALISTIC, order=order)
        n_bare = self.item_count(path=_PATH_BARE_LLM, order=order)
        if n_mental < min_items or n_bare < min_items:
            return False
        if self.undecidable_rate(path=_PATH_MENTALISTIC, order=order) > max_undecidable_rate:
            return False
        if self.undecidable_rate(path=_PATH_BARE_LLM, order=order) > max_undecidable_rate:
            return False
        acc_mental = self.accuracy(path=_PATH_MENTALISTIC, order=order)
        if acc_mental < accuracy_floor:
            return False
        return self.architectural_delta(order=order) >= delta_floor


# ---------------------------------------------------------------------------
# Pillar-E mentalistic context builder — witness filter for HiToM.
# ---------------------------------------------------------------------------


def _mentalistic_context_for(item: FANToMItem) -> str:
    """Build mentalistic context for one item.

    For HiToM items (metadata ``source="hitom"``), parse the story
    grammar and return events witnessed by every agent in the chain —
    the correct ToM scope-reduction for symbolic container-world ToM.
    For other sources (BigToM / future corpora without a parsed grammar)
    fall back to the full rendered context so the scorer stays
    dataset-agnostic; mentalistic ≈ bare on those surfaces (no
    architectural win, no regression).
    """

    metadata = item.conversation.metadata or {}
    source = metadata.get("source")
    if source == "hitom":
        chain_raw = metadata.get("agent_chain", []) or []
        chain = tuple(chain_raw)
        story = item.conversation.context[0].utterance if item.conversation.context else ""
        events = parse_hitom_story(story)
        witnessed = events_witnessed_by_chain(events, chain)
        return render_witnessed_events(witnessed)
    return item.conversation.render_context()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def score_higher_order_tom(
    items: Sequence[FANToMItem],
    *,
    responder: Optional[Responder] = None,
    item_responder: Optional[Callable[[FANToMItem], Responder]] = None,
    judge: AnswerJudge = lexical_answer_judge,
    system_factory: Optional[Callable[[], CompanionSystem]] = None,
    paths: tuple[str, ...] = (_PATH_MENTALISTIC, _PATH_BARE_LLM),
    judge_name: str = "lexical",
    responder_name: str = "unknown",
    dataset_tag: str = "hitom",
) -> HigherOrderToMReport:
    """Score a flat batch of ``FANToMItem``s (BigToM + HiToM mixed OK).

    Mutex: pass EXACTLY ONE of ``responder`` (shared callable) or
    ``item_responder`` (per-item factory, used when the CLI needs a
    fresh :class:`CompanionAgent` per conversation).
    """

    if responder is None and item_responder is None:
        raise ValueError(
            "score_higher_order_tom requires one of `responder` or `item_responder`"
        )
    if responder is not None and item_responder is not None:
        raise ValueError(
            "score_higher_order_tom requires EXACTLY ONE of `responder` or "
            "`item_responder`; got both"
        )

    cid_to_order: dict[str, int] = {}
    for it in items:
        order = it.question.tom_order
        cid_to_order[it.conversation.conversation_id] = (
            int(order) if order is not None else -1
        )

    base_report = _run_pillar_e_loop(
        items=items,
        responder=responder,
        item_responder=item_responder,
        judge=judge,
        system_factory=system_factory,
        paths=paths,
        judge_name=judge_name,
        responder_name=responder_name,
    )

    return HigherOrderToMReport(
        base_report=base_report,
        cid_to_order=cid_to_order,
        dataset_tag=dataset_tag,
    )


def _run_pillar_e_loop(
    *,
    items: Sequence[FANToMItem],
    responder: Optional[Responder],
    item_responder: Optional[Callable[[FANToMItem], Responder]],
    judge: AnswerJudge,
    system_factory: Optional[Callable[[], CompanionSystem]],
    paths: tuple[str, ...],
    judge_name: str,
    responder_name: str,
) -> FANToMToMReport:
    """Pillar-E-local two-path scoring loop.

    One fresh :class:`CompanionSystem` per conversation on the
    mentalistic path (same discipline as :func:`score_fantom_tom`).
    Ingest splits by newline (:func:`_load_conversation_split_by_lines`);
    recall is unscoped (:func:`_recall_unscoped`); prompts reuse the
    FANToM scorer's :func:`_build_mentalistic_prompt` /
    :func:`_build_bare_llm_prompt` shells so judge semantics are
    identical.
    """

    verdicts: list[FANToMVerdict] = []
    grouped: dict[str, list[FANToMItem]] = {}
    order: list[str] = []
    for item in items:
        cid = item.conversation.conversation_id
        if cid not in grouped:
            grouped[cid] = []
            order.append(cid)
        grouped[cid].append(item)

    # ``system_factory`` is kept on the signature for API-compat with
    # ``score_fantom_tom`` (callers may pass one through the CLI); it is
    # NOT invoked here because mentalistic routing moved to a pure
    # witness filter — the memory substrate is not on the Pillar E
    # critical path. Phase 9+ substrate work will revisit this.
    _ = system_factory

    # Cache per-item responders: ``item_responder`` is called at most
    # once per item, and the returned callable is reused across both
    # paths — matches the FANToM scorer's per-item contract.
    responder_cache: dict[int, Responder] = {}

    def _responder_for(item: FANToMItem) -> Responder:
        if item_responder is not None:
            key = id(item)
            cached = responder_cache.get(key)
            if cached is None:
                cached = item_responder(item)
                responder_cache[key] = cached
            return cached
        assert responder is not None  # mutex-guaranteed
        return responder

    for cid in order:
        bucket = grouped[cid]
        conversation = bucket[0].conversation

        if _PATH_MENTALISTIC in paths:
            for item in bucket:
                mentalistic_ctx = _mentalistic_context_for(item)
                source = (item.conversation.metadata or {}).get("source")
                if source == "hitom":
                    prompt = build_hitom_perspective_prompt(item, mentalistic_ctx)
                else:
                    prompt = _build_mentalistic_prompt(item.question, mentalistic_ctx)
                answer = _responder_for(item)(prompt)
                label = judge(item.question, answer)
                verdicts.append(
                    FANToMVerdict(
                        conversation_id=cid,
                        question_text=item.question.question_text,
                        question_type=item.question.question_type,
                        target_person=item.question.target_person,
                        source_path=_PATH_MENTALISTIC,
                        model_answer=answer,
                        label=label,
                    )
                )

        if _PATH_BARE_LLM in paths:
            for item in bucket:
                source = (item.conversation.metadata or {}).get("source")
                if source == "hitom":
                    prompt = build_hitom_bare_prompt(item)
                else:
                    prompt = _build_bare_llm_prompt(item.question, conversation)
                answer = _responder_for(item)(prompt)
                label = judge(item.question, answer)
                verdicts.append(
                    FANToMVerdict(
                        conversation_id=cid,
                        question_text=item.question.question_text,
                        question_type=item.question.question_type,
                        target_person=item.question.target_person,
                        source_path=_PATH_BARE_LLM,
                        model_answer=answer,
                        label=label,
                    )
                )

    return FANToMToMReport(
        verdicts=tuple(verdicts),
        judge_name=judge_name,
        responder_name=responder_name,
    )


__all__ = [
    "HigherOrderToMReport",
    "score_higher_order_tom",
]
