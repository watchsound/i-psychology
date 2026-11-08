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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence
from contextlib import nullcontext
from threading import Lock, RLock
import uuid
import hashlib

from psychology_base.companion.pam_export import pam_export, write_pam
from psychology_base.llm.execution import ExecutionDeferred

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
    _pending_lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)
    _run_lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)
    _boundary_version: int = field(default=0, init=False, repr=False)
    _operation_id: Optional[str] = field(default=None, init=False, repr=False)
    checkpoint_path: Optional[Path] = None
    fragment_source: Optional[Callable[[], Sequence[Fragment]]] = field(default=None, repr=False)
    _source_hashes: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _active_window_ids: tuple[str, ...] = field(default=(), init=False, repr=False)
    _active_boundary_version: int = field(default=0, init=False, repr=False)
    _last_tick: int = field(default=0, init=False, repr=False)
    _restoring: bool = field(default=False, init=False, repr=False)
    _recovery_required: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.checkpoint_path is not None and self.fragment_source is None:
            raise ValueError("Durable consolidation requires the authoritative fragment source")
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
            self._recovery_required = self.checkpoint_path.exists()

    def _require_restored(self):
        if self._recovery_required and not self._restoring:
            raise ExecutionDeferred("Restore the existing PAM checkpoint before accepting new consolidation work")

    @staticmethod
    def _fragment_hash(fragment):
        payload = json.dumps(pam_export([fragment]), ensure_ascii=False, sort_keys=True, allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _embedding_config(self):
        # Bind saved vectors to the same configured backend/model. This is not
        # immutable model-byte attestation; pinned providers also carry a digest.
        if not hasattr(self.router, "_embedding_target"):
            return None
        provider, model = self.router._embedding_target("auto", probe=False)
        return dict(provider=provider.name, model=model, base_url=getattr(provider, "base_url", None),
                    digest=getattr(provider, "expected_digest", None), context_length=getattr(provider, "context_length", None))

    def checkpoint_state(self, *, _for_validation=False, **overrides):
        if not _for_validation:
            self._require_restored()
        state = dict(version=1, instance_id=self.instance_id, source_hashes=dict(self._source_hashes),
            pending_ids=[fragment.id for fragment in self._pending], boundary_pending=self._boundary_pending,
            boundary_version=self._boundary_version, operation_id=self._operation_id,
            active_window_ids=list(self._active_window_ids), active_boundary_version=self._active_boundary_version,
            facts=self.store.checkpoint_state(), stats=asdict(self.stats), embedding_config=self._embedding_config(),
            tick_count=self._last_tick)
        state.update(overrides)
        return state

    def _persist(self, **overrides):
        if self.checkpoint_path is None or self._restoring:
            return
        self._require_restored()
        try:
            fragments = list(self.fragment_source())
            state = self.checkpoint_state(**overrides)
            self.validate_checkpoint_sources(state, fragments)
            write_pam(self.checkpoint_path, fragments, consolidation_state=state)
        except ValueError as exc:
            raise ExecutionDeferred("Consolidation sources changed; recovery checkpoint held for rebuild") from exc
        except OSError as exc:
            raise ExecutionDeferred("Consolidation checkpoint unavailable; work remains pending") from exc

    def validate_checkpoint_sources(self, state, fragments):
        sources = {fragment.id: fragment for fragment in fragments}
        if len(sources) != len(fragments) or any(
                key not in sources or value != self._fragment_hash(sources[key])
                for key, value in state["source_hashes"].items()):
            raise ValueError("Consolidation sources differ from original fragments")

    def prepare_restore(self, state, fragments):
        """Validate against original PAM content before ingestion or publication."""
        if not isinstance(state, dict) or set(state) != set(self.checkpoint_state(_for_validation=True)):
            raise ValueError("Invalid consolidation checkpoint shape")
        if type(state["version"]) is not int or state["version"] != 1 or state["instance_id"] != self.instance_id:
            raise ValueError("Consolidation checkpoint version/instance differs")
        if state["embedding_config"] != self._embedding_config():
            raise ValueError("Consolidation embedding configuration changed; rebuild from original fragments")
        sources = {fragment.id: fragment for fragment in fragments}
        hashes = state["source_hashes"]
        if not isinstance(hashes, dict):
            raise ValueError("Consolidation sources differ from original fragments")
        self.validate_checkpoint_sources(state, fragments)
        pending, active = state["pending_ids"], state["active_window_ids"]
        if (not isinstance(pending, list) or not isinstance(active, list)
                or any(not isinstance(key, str) or key not in hashes for key in pending)
                or len(set(pending)) != len(pending) or pending[:len(active)] != active
                or any(type(state[key]) is not int or state[key] < 0 for key in ("boundary_version", "active_boundary_version", "tick_count"))
                or state["active_boundary_version"] > state["boundary_version"]
                or type(state["boundary_pending"]) is not bool
                or (bool(active) != (isinstance(state["operation_id"], str) and bool(state["operation_id"])))):
            raise ValueError("Invalid consolidation pending window/counters")
        if ((active and (not isinstance(state["operation_id"], str) or not state["operation_id"].strip()))
                or (not active and state["operation_id"] is not None)):
            raise ValueError("Invalid consolidation operation identity")
        store = FactStore.from_checkpoint_state(state["facts"])
        if any(fact.updated_tick > state["tick_count"] for fact in store.all()):
            raise ValueError("Derived facts exceed the saved consolidation clock")
        if any(source not in hashes for fact in store.all() for source in fact.source_fragment_ids):
            raise ValueError("Derived fact references an unknown authoritative source")
        stats = state["stats"]
        if (not isinstance(stats, dict) or set(stats) != set(asdict(ConsolidationStats()))
                or any(type(value) is not int or value < 0 for value in stats.values())):
            raise ValueError("Invalid consolidation statistics")
        # Merging a PAM into an existing source set must register unseen input.
        new_ids = [fragment.id for fragment in fragments if fragment.id not in hashes]
        prepared = dict(state)
        prepared.update(store=store, pending=[sources[key] for key in pending + new_ids],
                        stats=ConsolidationStats(**stats), source_hashes={key: self._fragment_hash(value) for key, value in sources.items()})
        return prepared

    def restore_prepared(self, state):
        with self._pending_lock:
            self.store.commit_from(state["store"])
            self._pending = state["pending"]
            self._source_hashes = state["source_hashes"]
            self.stats = state["stats"]
            self._boundary_pending = state["boundary_pending"]
            self._boundary_version = state["boundary_version"]
            self._active_boundary_version = state["active_boundary_version"]
            self._active_window_ids = tuple(state["active_window_ids"])
            self._operation_id = state["operation_id"]
            self._last_tick = state["tick_count"]
            self._recovery_required = False

    def observe(self, fragment: Fragment) -> None:
        """Called from ``CompanionSystem.observe`` to track the window."""
        with self._pending_lock:
            self._require_restored()
            fingerprint = self._fragment_hash(fragment)
            if fragment.id in self._source_hashes:
                if self._source_hashes[fragment.id] != fingerprint:
                    raise ValueError("Observed fragment id has conflicting content")
                return
            hashes = {**self._source_hashes, fragment.id: fingerprint}
            self._source_hashes = hashes
            self._pending.append(fragment)
            # A storage refusal must retain the in-process registration too;
            # the authoritative source was already ingested by CompanionSystem.
            self._persist()

    def mark_session_boundary(self) -> None:
        """Caller signals that the accumulated window closes a session.
        The actual extraction runs on the next ``maybe_consolidate``
        (sleep tick), not on this call — keeping sleep work on sleep
        ticks, per the heartbeat discipline."""
        with self._pending_lock:
            self._require_restored()
            self._boundary_pending = True
            self._boundary_version += 1
            self._persist()

    def maybe_consolidate(self, tick_count: int) -> ConsolidationStats:
        """Run extraction+apply if a boundary is pending OR the fallback
        window is full. Returns per-run stats. If nothing to do, returns
        a zeroed stats record."""
        if not self._run_lock.acquire(blocking=False):
            return ConsolidationStats()
        try:
            self._require_restored()
            tick_count = max(tick_count, self._last_tick)
            with self._pending_lock:
                if not self._pending:
                    self._boundary_pending = False
                    return ConsolidationStats()
                if not (self._active_window_ids or self._boundary_pending or len(self._pending) >= self.fallback_window):
                    return ConsolidationStats()
                window = list(self._pending[:len(self._active_window_ids)]) if self._active_window_ids else list(self._pending)
                boundary_version = self._active_boundary_version if self._active_window_ids else self._boundary_version
                operation_id = self._operation_id or uuid.uuid4().hex
                self._persist(operation_id=operation_id, active_window_ids=[f.id for f in window],
                              active_boundary_version=boundary_version, tick_count=tick_count)
                self._operation_id = operation_id
                self._active_window_ids = tuple(f.id for f in window)
                self._active_boundary_version = boundary_version
                self._last_tick = tick_count
            operation = getattr(self.router, "operation", None)
            with operation("memory_consolidation", operation_id=self._operation_id) if operation else nullcontext():
                ops = extract_facts(window, prior_facts=self.store.all(), router=self.router, tick_count=tick_count)
                staged, stats, logs = self._prepare(ops, tick_count=tick_count)
            with self._pending_lock:
                totals = ConsolidationStats(**{key: getattr(self.stats, key) + getattr(stats, key) for key in asdict(stats)})
                boundary_pending = self._boundary_version != boundary_version
                self._persist(facts=staged.checkpoint_state(), stats=asdict(totals),
                    pending_ids=[f.id for f in self._pending[len(window):]], boundary_pending=boundary_pending,
                    operation_id=None, active_window_ids=[], active_boundary_version=0)
                self.store.commit_from(staged)
                self.stats = totals
                del self._pending[:len(window)]
                self._boundary_pending = boundary_pending
                self._operation_id = None
                self._active_window_ids = ()
                self._active_boundary_version = 0
            for label, fact, target in logs:
                self._log(label, fact, tick_count, target=target)
            return stats
        finally:
            self._run_lock.release()

    def _prepare(self, ops: Sequence[FactOperation], *, tick_count: int):
        stats = ConsolidationStats()
        staged = self.store.staged_copy()
        logs = []
        for op in ops:
            if op.op == FactOp.ADD and op.fact is not None:
                embedded = self._embed(op.fact)
                added = staged.add(embedded)
                stats.adds += 1
                logs.append(("ADD", added, None))
            elif op.op == FactOp.UPDATE and op.fact is not None and op.target_fact_id:
                prior = staged.by_id(op.target_fact_id)
                if prior is None:
                    stats.dropped += 1
                    logs.append(("UPDATE_DROPPED", op.fact, op.target_fact_id))
                    continue
                embedded = self._embed(op.fact)
                staged.upsert(embedded)
                stats.updates += 1
                logs.append(("UPDATE", embedded, op.target_fact_id))
            elif op.op == FactOp.NOOP and op.target_fact_id:
                prior = staged.by_id(op.target_fact_id)
                if prior is None:
                    stats.dropped += 1
                    logs.append(("NOOP_DROPPED", None, op.target_fact_id))
                    continue
                refreshed = replace(prior, updated_tick=tick_count)
                staged.upsert(refreshed)
                stats.noops += 1
                logs.append(("NOOP", refreshed, op.target_fact_id))
        return staged, stats, logs

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
