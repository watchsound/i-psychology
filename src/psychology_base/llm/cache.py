"""Disk-backed response cache for LLM calls.

Semantic analysis is a high-frequency operation across this project
(per-fragment summarization, per-recall relevance scoring, reconsolidation
delta detection, Witness divergence). Deterministic re-runs over the same
inputs — especially during benchmark iteration — should hit disk, not the
model. The cache is sharded (first 2 hex chars as subdirectory) to keep
any one directory under ~65k files.

Key composition (SHA-256 of canonical JSON):
    {provider, model_id, tier, temperature, max_tokens, messages,
     purpose='chat' | 'embed'}

A different model, tier, or temperature yields a different key — so a
model swap invalidates the cache without a manual flush.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from psychology_base.llm.types import ChatMessage, Completion, TaskTier


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def chat_cache_key(
    *,
    provider: str,
    model_id: str,
    tier: TaskTier,
    messages: list[ChatMessage],
    temperature: Optional[float],
    max_tokens: Optional[int],
    response_schema: Optional[dict] = None,
    think: Optional[bool] = None,
) -> str:
    payload = {
        "purpose": "chat",
        "provider": provider,
        "model_id": model_id,
        "tier": tier.value,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    if response_schema is not None:
        json.dumps(response_schema, allow_nan=False)
        payload['response_schema'] = response_schema
    if think is not None:
        if type(think) is not bool:
            raise ValueError('Thinking cache identity requires a boolean or None')
        payload['think'] = think
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def embedding_cache_key(
    *,
    provider: str,
    model_id: str,
    text: str,
    identity: Optional[dict] = None,
) -> str:
    payload = {
        "purpose": "embed",
        "provider": provider,
        "model_id": model_id,
        "text": text,
    }
    if identity is not None:
        if not isinstance(identity, dict) or not identity:
            raise ValueError("Embedding cache identity must be a nonempty dictionary")
        # Validate the opt-in namespace without changing historical keys.
        json.dumps(identity, allow_nan=False)
        payload["embedding_identity"] = identity
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class LLMCache:
    """SHA-sharded on-disk cache for chat and embedding responses.

    Writes are atomic (``.tmp`` then rename). A failed read deletes the
    corrupt entry silently so the next call re-populates — cache misses
    on cold load are preferable to crashing the research run.
    """

    def __init__(self, cache_dir: Optional[Path] = None, *, derived_max_entries=0, derived_max_bytes=0) -> None:
        if (any(type(n) is not int or n < 0 for n in (derived_max_entries, derived_max_bytes))
                or (derived_max_entries == 0) != (derived_max_bytes == 0)):
            raise ValueError('Derived reuse requires explicit nonnegative entry and byte limits')
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "psychology_base" / "llm"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        # Opt-in, process-lived derived completions. No second disk store or
        # authoritative memory: owners must validate on publication and reuse.
        self._derived_limits = (derived_max_entries, derived_max_bytes)
        self._derived = OrderedDict()
        self._derived_lock = threading.Lock()
        self._derived_bytes = 0
        self._derived_counts = dict(hits=0, misses=0, evictions=0, rejected=0)

    @property
    def derived_limits(self):
        return self._derived_limits

    def __deepcopy__(self, memo):
        """Preserve existing cache/system copying with an independent lock."""
        from copy import deepcopy
        result = object.__new__(type(self))
        memo[id(self)] = result
        with self._derived_lock:
            state = dict(self.__dict__)
            state.pop('_derived_lock')
            state['_derived'] = self._derived.copy()
            state['_derived_counts'] = dict(self._derived_counts)
        result.__dict__.update(deepcopy(state, memo))
        result._derived_lock = threading.Lock()
        return result

    def get_derived(self, key):
        """Return a detached completion; the semantic owner revalidates it."""
        with self._derived_lock:
            row = self._derived.get(key)
            if row is None:
                self._derived_counts['misses'] += 1
                return None
            self._derived.move_to_end(key)
            self._derived_counts['hits'] += 1
            value = json.loads(row[0])
        value['tier'] = TaskTier(value['tier'])
        return Completion(**value)

    def put_derived(self, key, completion):
        if (type(key) is not str or len(key) != 64 or any(c not in '0123456789abcdef' for c in key)
                or type(completion) is not Completion):
            raise ValueError('Derived reuse requires a hashed key and a completion')
        value = asdict(completion)
        value['tier'] = completion.tier.value
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        size = len(key.encode('utf-8')) + len(encoded.encode('utf-8'))
        entries, capacity = self.derived_limits
        with self._derived_lock:
            if entries == 0 or size > capacity:
                self._derived_counts['rejected'] += 1
                return False
            self._discard_derived(key)
            while len(self._derived) >= entries or self._derived_bytes + size > capacity:
                _, (_, removed) = self._derived.popitem(last=False)
                self._derived_bytes -= removed
                self._derived_counts['evictions'] += 1
            self._derived[key] = (encoded, size)
            self._derived_bytes += size
        return True

    def _discard_derived(self, key):
        row = self._derived.pop(key, None)
        if row is not None:
            self._derived_bytes -= row[1]

    def discard_derived(self, key):
        with self._derived_lock:
            self._discard_derived(key)

    def clear_derived(self):
        with self._derived_lock:
            self._derived.clear()
            self._derived_bytes = 0

    def derived_snapshot(self):
        with self._derived_lock:
            return dict(self._derived_counts, entries=len(self._derived), bytes=self._derived_bytes,
                        max_entries=self.derived_limits[0], max_bytes=self.derived_limits[1])

    # ---------- chat ----------

    def get_chat(self, key: str) -> Optional[Completion]:
        path = self._path_for(key, kind="chat")
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            completion = Completion(
                content=data["content"],
                provider=data["provider"],
                model_id=data["model_id"],
                tier=TaskTier(data["tier"]),
                prompt_tokens=data.get("prompt_tokens", 0),
                completion_tokens=data.get("completion_tokens", 0),
                latency_ms=data.get("latency_ms", 0.0),
                finish_reason=data.get("finish_reason", ""),
                metadata=data.get("metadata", {}),
            )
            self.hits += 1
            return completion
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            try:
                path.unlink()
            except OSError:
                pass
            self.misses += 1
            return None

    def put_chat(self, key: str, completion: Completion) -> None:
        path = self._path_for(key, kind="chat")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(completion) if dataclasses.is_dataclass(completion) else {
            "content": completion.content,
            "provider": completion.provider,
            "model_id": completion.model_id,
            "tier": completion.tier.value,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "latency_ms": completion.latency_ms,
            "finish_reason": completion.finish_reason,
            "metadata": completion.metadata,
        }
        payload["tier"] = completion.tier.value  # enum → str for JSON
        self._atomic_write(path, _canonical_json(payload))

    # ---------- embeddings ----------

    def get_embedding(self, key: str) -> Optional[list[float]]:
        path = self._path_for(key, kind="embed")
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.hits += 1
            return list(data["vector"])
        except (OSError, json.JSONDecodeError, KeyError):
            try:
                path.unlink()
            except OSError:
                pass
            self.misses += 1
            return None

    def put_embedding(self, key: str, vector: list[float], *, model_id: str) -> None:
        path = self._path_for(key, kind="embed")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"vector": list(vector), "model_id": model_id}
        self._atomic_write(path, _canonical_json(payload))

    # ---------- internals ----------

    def _path_for(self, key: str, *, kind: str) -> Path:
        shard = key[:2]
        return self.cache_dir / kind / shard / f"{key}.json"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        # Per-writer tmp name so concurrent writers on the same key
        # (chat_batch over duplicate prompts) don't fight for a shared
        # ``.tmp``. Windows raises PermissionError on the second
        # ``os.replace`` otherwise.
        suffix = f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        tmp = path.with_name(path.name + suffix)
        tmp.write_text(text, encoding="utf-8")
        try:
            os.replace(tmp, path)
        except PermissionError:
            # Another writer already finalized this key — our content
            # is semantically identical (SHA-keyed), so dropping is fine.
            try:
                tmp.unlink()
            except OSError:
                pass
