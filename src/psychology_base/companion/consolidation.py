"""Phase 1.5 Consolidator — sleep-tick orchestration.

Per [phase-1.5-build-plan.md](../../../docs/phases/phase-1.5-build-plan.md)
§5 step 2 + §5 step 3:

- Detects a "window ready" signal (explicit session boundary from
  benchmark adapters, OR N-observation fallback when no boundary is
  marked).
- Calls ``extract_facts`` on the window of fragments accumulated since
  the previous consolidation.
- Applies each ``FactOperation`` to the ``FactStore``:
    - ADD → ``store.add(...)``
    - UPDATE → ``store.upsert(...)`` (overwrite; version chain is Phase 2)
    - NOOP → no mutation; the prior fact's updated_tick is bumped so it
      stays "fresh" for the entity-recency prior-filter
- On a per-op basis: embeds the fact text via ``router.embed`` and
  attaches the embedding to the stored record. NOOP does not re-embed —
  the fact text is unchanged.

Pure logging:
- Each ADD/UPDATE/NOOP writes one line to
  ``~/.cache/psychology_base/phase-1.5/facts/<instance_id>.jsonl`` when
  ``log_dir`` is set.
- Logs are gate-failure instrumentation (build plan §2); they are not
  consumed by any other code path, so a logging failure must not break
  the consolidation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

from psychology_base.companion.extraction import extract_facts
from psychology_base.companion.facts import (
    Fact,
    FactOp,
    FactOperation,
    FactStore,
    embed_text_for,
)
from psychology_base.eval.adapters import Fragment
from psychology_base.llm.router import LLMRouter


logger = logging.getLogger(__name__)


DEFAULT_FALLBACK_WINDOW = 20  # observations between implicit boundaries


@dataclass
class ConsolidationStats:
    adds: int = 0
    updates: int = 0
    noops: int = 0
    dropped: int = 0  # ops referencing unknown target_fact_id

    def total(self) -> int:
        return self.adds + self.updates + self.noops


@dataclass
class Consolidator:
    """Session-boundary extractor + FactStore applicator.

    Caller signals boundaries explicitly via ``mark_session_boundary``;
    ``maybe_consolidate`` (called from ``CompanionSystem.tick``) runs the
    pass when either (a) a boundary was marked since the last run, or
    (b) ``fallback_window`` fragments have accrued since the last run.
    """

    router: LLMRouter
    store: FactStore
    fallback_window: int = DEFAULT_FALLBACK_WINDOW
    log_dir: Optional[Path] = None
    instance_id: str = "default"
    _pending: list[Fragment] = field(default_factory=list)
    _boundary_pending: bool = False
    stats: ConsolidationStats = field(default_factory=ConsolidationStats)

    def observe(self, fragment: Fragment) -> None:
        """Called from ``CompanionSystem.observe`` to track the window."""
        self._pending.append(fragment)

    def mark_session_boundary(self) -> None:
        """Caller signals that the accumulated window closes a session.
        The actual extraction runs on the next ``maybe_consolidate``
        (sleep tick), not on this call — keeping sleep work on sleep
        ticks, per the heartbeat discipline."""
        self._boundary_pending = True

    def maybe_consolidate(self, tick_count: int) -> ConsolidationStats:
        """Run extraction+apply if a boundary is pending OR the fallback
        window is full. Returns per-run stats. If nothing to do, returns
        a zeroed stats record."""
        if not self._pending:
            self._boundary_pending = False
            return ConsolidationStats()

        if not (self._boundary_pending or len(self._pending) >= self.fallback_window):
            return ConsolidationStats()

        window = list(self._pending)
        self._pending.clear()
        self._boundary_pending = False

        prior = self.store.all()
        ops = extract_facts(
            window,
            prior_facts=prior,
            router=self.router,
            tick_count=tick_count,
        )
        return self._apply(ops, tick_count=tick_count)

    def _apply(self, ops: Sequence[FactOperation], *, tick_count: int) -> ConsolidationStats:
        stats = ConsolidationStats()
        for op in ops:
            if op.op == FactOp.ADD and op.fact is not None:
                embedded = self._embed(op.fact)
                added = self.store.add(embedded)
                stats.adds += 1
                self._log("ADD", added, tick_count)
            elif op.op == FactOp.UPDATE and op.fact is not None and op.target_fact_id:
                prior = self.store.by_id(op.target_fact_id)
                if prior is None:
                    stats.dropped += 1
                    self._log("UPDATE_DROPPED", op.fact, tick_count, target=op.target_fact_id)
                    continue
                embedded = self._embed(op.fact)
                self.store.upsert(embedded)
                stats.updates += 1
                self._log("UPDATE", embedded, tick_count, target=op.target_fact_id)
            elif op.op == FactOp.NOOP and op.target_fact_id:
                prior = self.store.by_id(op.target_fact_id)
                if prior is None:
                    stats.dropped += 1
                    self._log("NOOP_DROPPED", None, tick_count, target=op.target_fact_id)
                    continue
                refreshed = replace(prior, updated_tick=tick_count)
                self.store.upsert(refreshed)
                stats.noops += 1
                self._log("NOOP", refreshed, tick_count, target=op.target_fact_id)
        self.stats.adds += stats.adds
        self.stats.updates += stats.updates
        self.stats.noops += stats.noops
        self.stats.dropped += stats.dropped
        return stats

    def _embed(self, fact: Fact) -> Fact:
        text = embed_text_for(fact)
        vector = tuple(self.router.embed(text))
        return replace(fact, embedding=vector)

    def _log(
        self,
        op_label: str,
        fact: Optional[Fact],
        tick_count: int,
        *,
        target: Optional[str] = None,
    ) -> None:
        if self.log_dir is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{self.instance_id}.jsonl"
            entry = {
                "tick": tick_count,
                "op": op_label,
                "target": target,
                "fact": {
                    "id": fact.id,
                    "entity": fact.entity,
                    "relation": fact.relation,
                    "value": fact.value,
                    "confidence": fact.confidence,
                    "source_fragment_ids": list(fact.source_fragment_ids),
                } if fact is not None else None,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("consolidation log write failed: %s", exc)


__all__ = ["Consolidator", "ConsolidationStats", "DEFAULT_FALLBACK_WINDOW"]
