"""Pure, source-bound cue hints. Providers and authoritative memories live elsewhere."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math

from psychology_base.companion.facts import cosine
from psychology_base.companion.turn_provenance import activation_tie_key, evidence_key
from psychology_base.eval.adapters import content_hash
from psychology_base.i18n import tokenizer_identity
from psychology_base.llm.types import validate_embedding


MAX_GROUPS = 32
MAX_OCCURRENCES = 128
MAX_BYTES = 8 * 1024 * 1024
MAX_FEATURES = 16
MAX_QUERY_FEATURES = 8
MAX_MEMBERSHIPS = 65536


@dataclass(frozen=True)
class NavigationHints:
    snapshot: str
    policy: str
    weights: tuple[tuple[str, float], ...]
    reason: str
    accounted_bytes: int = 0


def snapshot_identity(fragments, cue, focus_cue=None, cjk_tokenizer="unicode"):
    rows = []
    ids = set()
    for f in sorted(fragments, key=lambda f: f.id):
        if f.id in ids:
            raise ValueError("Duplicate fragment IDs in navigation input")
        ids.add(f.id)
        rows.append((f.id, content_hash(f.content, f.timestamp, f.affect, f.tags, f.parent_ids)))
    payload = [rows, cue, focus_cue, tokenizer_identity(cjk_tokenizer)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _policy(name, **parameters):
    return json.dumps(dict(name=name, **parameters), sort_keys=True, allow_nan=False)


def _group_hints(fragments, scores, snapshot, policy, *, accounted_bytes=0):
    if set(scores) != {f.id for f in fragments}:
        raise ValueError("Navigation scores must cover exactly the eligible IDs")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in scores.values()):
        raise ValueError("Navigation scores must be finite and nonnegative")
    if accounted_bytes > MAX_BYTES:
        return NavigationHints(snapshot, policy, (), "resource-bytes", accounted_bytes)
    groups = {}
    for f in fragments:
        groups.setdefault(evidence_key(f), []).append(f)
    ranked = []
    for key, members in groups.items():
        weight = max(scores[f.id] for f in members)
        if weight > 0:
            digest = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            ranked.append((weight, min(activation_tie_key(f) for f in members), digest, members))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
    ranked = ranked[:MAX_GROUPS]
    if sum(len(row[3]) for row in ranked) > MAX_OCCURRENCES:
        return NavigationHints(snapshot, policy, (), "resource-occurrences", accounted_bytes)
    total = math.fsum(row[0] for row in ranked)
    weights = tuple(sorted((f.id, weight / total / len(members))
                           for weight, _, _, members in ranked for f in members))
    return NavigationHints(snapshot, policy, weights, "matched" if weights else "no-hit", accounted_bytes)


def semantic_policy(floor, model_identity):
    if type(floor) not in (int, float) or not math.isfinite(floor) or not -1 <= floor <= 1:
        raise ValueError("Semantic cosine floor must be finite and within [-1, 1]")
    return _policy("semantic-cosine-v1", floor=floor, model=model_identity)


def semantic_hints(fragments, cue, vectors, query_vector, *, floor, model_identity,
                   focus_cue=None, cjk_tokenizer="unicode"):
    snapshot = snapshot_identity(fragments, cue, focus_cue, cjk_tokenizer)
    policy = semantic_policy(floor, model_identity)
    if (focus_cue if focus_cue is not None else cue) == "" or not fragments:
        return NavigationHints(snapshot, policy, (), "empty-input")
    if set(vectors) != {f.id for f in fragments}:
        raise ValueError("Embedding vectors must cover exactly the eligible IDs")
    query = validate_embedding(query_vector)
    query_norm = math.hypot(*query)
    query = tuple(value / query_norm for value in query)
    scores = {}
    accounted = 8 * (len(query) + sum(len(v) for v in vectors.values()))
    accounted += sum(len(f.id.encode()) + 32 for f in fragments)
    if accounted > MAX_BYTES:
        return NavigationHints(snapshot, policy, (), "resource-bytes", accounted)
    for f in fragments:
        vector = validate_embedding(vectors[f.id], dimension=len(query))
        norm = math.hypot(*vector)
        similarity = max(-1., min(1., cosine(query, tuple(value / norm for value in vector))))
        scores[f.id] = max(0., similarity - floor)
    return _group_hints(fragments, scores, snapshot, policy, accounted_bytes=accounted)


def _windows(words):
    for size in (1, 2, 3):
        for i in range(len(words) - size + 1):
            key = tuple(words[i:i + size])
            if all(len(word) >= 2 for word in key):
                yield key


def lexical_hints(fragments, cue, ordered_tokens, query_tokens, *, focus_cue=None,
                  cjk_tokenizer="unicode"):
    snapshot = snapshot_identity(fragments, cue, focus_cue, cjk_tokenizer)
    policy = _policy("lexical-specificity-v1")
    if set(ordered_tokens) != {f.id for f in fragments}:
        raise ValueError("Lexical tokens must cover exactly the eligible IDs")
    groups = {}
    for f in fragments:
        groups.setdefault(evidence_key(f), f)
    frequencies = Counter()
    memberships = accounted = 0
    group_keys = {}
    for group, fragment in groups.items():
        keys = set()
        for key in _windows(ordered_tokens[fragment.id]):
            if key in keys:
                continue
            memberships += 1
            accounted += sum(len(word.encode()) for word in key) + 32
            if memberships > MAX_MEMBERSHIPS or accounted > MAX_BYTES:
                return NavigationHints(snapshot, policy, (), "resource-features", accounted)
            keys.add(key)
        frequencies.update(keys)
        group_keys[group] = keys
    specificity = {key: math.log1p(len(groups) / (1 + df)) * len(key)
                   for key, df in frequencies.items()}
    observed_query = set()
    for key in _windows(query_tokens):
        if key in specificity:
            observed_query.add(key)
        if len(observed_query) > MAX_MEMBERSHIPS:
            return NavigationHints(snapshot, policy, (), "resource-query", accounted)
    query_keys = set(sorted(observed_query, key=lambda k: (-specificity[k], k))[:MAX_QUERY_FEATURES])
    group_scores = {}
    for group, keys in group_keys.items():
        retained = set(sorted(keys, key=lambda k: (-specificity[k], k))[:MAX_FEATURES])
        group_scores[group] = max((specificity[k] for k in retained & query_keys), default=0.)
    return _group_hints(fragments, {f.id: group_scores[evidence_key(f)] for f in fragments},
                        snapshot, policy, accounted_bytes=accounted)


def validate_hints(hints, fragments, snapshot, mixture):
    if not isinstance(hints, NavigationHints) or hints.snapshot != snapshot:
        raise ValueError("Navigation hints have a stale or mismatched snapshot")
    if type(mixture) not in (int, float) or not math.isfinite(mixture) or not 0 <= mixture <= 1:
        raise ValueError("Navigation mixture must be finite and within [0, 1]")
    policy = json.loads(hints.policy)
    json.dumps(policy, allow_nan=False)
    if not isinstance(policy, dict) or policy.get("name") not in ("semantic-cosine-v1", "lexical-specificity-v1"):
        raise ValueError("Unknown navigation policy")
    if len(hints.weights) > MAX_OCCURRENCES:
        raise ValueError("Navigation hint occurrence limit exceeded")
    eligible = {f.id for f in fragments}
    seen = set()
    for fid, value in hints.weights:
        if (fid not in eligible or fid in seen or type(value) not in (int, float)
                or not math.isfinite(value) or value <= 0):
            raise ValueError("Navigation hints contain invalid IDs or weights")
        seen.add(fid)
    if hints.weights and not math.isclose(math.fsum(v for _, v in hints.weights), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("Navigation hints must conserve one cue budget")


def compose_seed(lexical_seed, hints, mixture, seed_boost):
    if not hints.weights or mixture == 0:
        return lexical_seed
    navigation = {fid: seed_boost * value for fid, value in hints.weights}
    if not lexical_seed or lexical_seed == navigation:
        return navigation if not lexical_seed else lexical_seed
    return {fid: (1 - mixture) * lexical_seed.get(fid, 0.) + mixture * navigation.get(fid, 0.)
            for fid in sorted(lexical_seed.keys() | navigation.keys())}
