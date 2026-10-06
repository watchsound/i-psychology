from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from psychology_base.eval.adapters import MemorySystem, Observation


def _tokens(text: str) -> set[str]:
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine_bow(a: str, b: str) -> float:
    ca, cb = Counter(a.lower().split()), Counter(b.lower().split())
    shared = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in shared)
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class VMetrics:
    v1_state_continuity: float
    v2_adjacent_jaccard: float
    v2_longhorizon_jaccard: float
    v3_mentalistic_liveness: float
    v4_mirror_ablation: Optional[float] = None
    v5_rule_internalization: Optional[float] = None
    notes: dict = field(default_factory=dict)


MemoryFactory = Callable[[], MemorySystem]


def measure_v_metrics(
    memory_factory: MemoryFactory,
    probe_cue: str,
    scenario_observations: list[Observation],
    mentalistic_contexts: tuple[dict, dict] = ({"mode": "self"}, {"mode": "other"}),
    long_horizon_ticks: int = 20,
    has_witness_coupling: bool = False,
    has_rule_mechanism: bool = False,
) -> VMetrics:
    """Measure the four always-applicable V-metrics against any MemorySystem.

    V4 (Mirror ablation) and V5 (rule internalization) require architectural features
    EchoMemory does not have; they return None with a note. Phase 1+ systems that have
    Witness coupling / rule-prompt mechanisms set the corresponding flags and get real
    numbers.

    Each measurement uses a fresh memory instance built from `memory_factory` to avoid
    cross-measurement state contamination.
    """
    notes: dict = {}

    # V1: cosine of state-signature before vs. after a tick with no new observation.
    m1 = memory_factory()
    for obs in scenario_observations:
        m1.observe(obs)
    sig_before = _state_signature(m1, probe_cue)
    m1.tick()
    sig_after = _state_signature(m1, probe_cue)
    v1 = _cosine_bow(sig_before, sig_after)

    # V2 adjacent: full scenario observed, recall twice back-to-back.
    # (Using the full scenario rather than only the first observation gives a
    # token pool large enough for Jaccard to be stable under short recalls —
    # a one-fragment pool is pathologically noisy. Semantics unchanged: same
    # cue, consecutive ticks, no new observation between.)
    m2 = memory_factory()
    for obs in scenario_observations:
        m2.observe(obs)
    r_a = m2.recall(probe_cue).content
    r_b = m2.recall(probe_cue).content
    v2_adj = _jaccard(_tokens(r_a), _tokens(r_b))

    # V2 long-horizon: full scenario observed, recall now vs. recall after N ticks.
    m3 = memory_factory()
    for obs in scenario_observations:
        m3.observe(obs)
    r_ep1 = m3.recall(probe_cue).content
    for _ in range(long_horizon_ticks):
        m3.tick()
    r_epN = m3.recall(probe_cue).content
    v2_lh = _jaccard(_tokens(r_ep1), _tokens(r_epN))

    # V3 mentalistic liveness: same cue under two contexts, within one session.
    m4 = memory_factory()
    for obs in scenario_observations:
        m4.observe(obs)
    r_c1 = m4.recall(probe_cue, mentalistic_contexts[0]).content
    r_c2 = m4.recall(probe_cue, mentalistic_contexts[1]).content
    v3_distance = 1.0 - _jaccard(_tokens(r_c1), _tokens(r_c2))

    if not has_witness_coupling:
        notes["v4"] = "N/A — architecture has no Witness coupling to ablate (baseline floor)"
    if not has_rule_mechanism:
        notes["v5"] = "N/A — architecture has no rule-prompt mechanism (baseline floor)"

    return VMetrics(
        v1_state_continuity=v1,
        v2_adjacent_jaccard=v2_adj,
        v2_longhorizon_jaccard=v2_lh,
        v3_mentalistic_liveness=v3_distance,
        notes=notes,
    )


def _state_signature(memory: MemorySystem, probe_cue: str) -> str:
    recall_content = memory.recall(probe_cue).content
    inspection = memory.inspect() if hasattr(memory, "inspect") else None
    return recall_content + " || " + str(inspection)


# ---------------------------------------------------------------------------
# V4 — Mirror ablation (Jensen-Shannon divergence over token distributions)
# ---------------------------------------------------------------------------


@dataclass
class V4Result:
    jsd_ablation: float
    jsd_anchor: float
    jsd_null: float
    anchor_ratio: float  # Target ≥ 1.3× (Ouroboros reference: 1.36×)
    null_ratio: float    # Target ≥ 10×  (Ouroboros reference: 83×)


def _normalize_dist(counts: Counter, vocab: list[str]) -> list[float]:
    total = sum(counts.values())
    if total == 0:
        return [1.0 / len(vocab)] * len(vocab) if vocab else []
    return [counts.get(t, 0) / total for t in vocab]


def _jensen_shannon(a: Counter, b: Counter) -> float:
    """JSD base-2 ∈ [0, 1]. Smoothed vocabulary over the union; empty-vs-empty = 0."""
    vocab = sorted(set(a) | set(b))
    if not vocab:
        return 0.0
    p = _normalize_dist(a, vocab)
    q = _normalize_dist(b, vocab)
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

    def _kl(x: list[float], y: list[float]) -> float:
        s = 0.0
        for xi, yi in zip(x, y):
            if xi > 0 and yi > 0:
                s += xi * math.log2(xi / yi)
        return s

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


MemoryFactoryWithSeed = Callable[[int, bool], MemorySystem]


def _gather_token_distribution(
    factory: MemoryFactoryWithSeed,
    witness_active: bool,
    scenario: list[Observation],
    probe_cue: str,
    seeds: list[int],
    recalls_per_run: int,
) -> Counter:
    counter: Counter = Counter()
    for seed in seeds:
        mem = factory(seed, witness_active)
        for obs in scenario:
            mem.observe(obs)
        for _ in range(recalls_per_run):
            r = mem.recall(probe_cue)
            for t in r.content.lower().split():
                counter[t] += 1
    return counter


def measure_v4_mirror_ablation(
    factory: MemoryFactoryWithSeed,
    scenario_observations: list[Observation],
    probe_cue: str,
    base_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    anchor_seeds_a: tuple[int, ...] = (10, 11, 12, 13, 14),
    anchor_seeds_b: tuple[int, ...] = (20, 21, 22, 23, 24),
    recalls_per_run: int = 5,
    null_shuffle_seed: int = 42,
) -> V4Result:
    """V4 — JSD(coupled responses, ablated responses) ratioed against two baselines.

    `anchor_ratio` ≥ 1.3× means: Witness ablation changes the response distribution
    more than natural seed-noise variation inside the coupled condition. This is
    Ouroboros's `1.36× anchor JSD` reference.

    `null_ratio` ≥ 10× means: ablation shifts responses in a specific direction,
    not just random drift from fragment-order shuffling. Ouroboros `83× null-shuffle`.

    `factory(seed, witness_active)` builds a memory system; the ablation condition
    passes witness_active=False, every other call passes True.
    """
    coupled = _gather_token_distribution(
        factory, True, scenario_observations, probe_cue, list(base_seeds), recalls_per_run,
    )
    ablated = _gather_token_distribution(
        factory, False, scenario_observations, probe_cue, list(base_seeds), recalls_per_run,
    )

    # Anchor: two disjoint subsets of coupled runs (natural seed variation).
    anchor_a = _gather_token_distribution(
        factory, True, scenario_observations, probe_cue, list(anchor_seeds_a), recalls_per_run,
    )
    anchor_b = _gather_token_distribution(
        factory, True, scenario_observations, probe_cue, list(anchor_seeds_b), recalls_per_run,
    )

    # Null: measure the Witness ablation effect on a scenario where the Witness
    # has NO structural purchase — all tags collapsed to a single neutral tag so
    # the flipped-context mechanism (W7) has nothing to bias toward. If V4 is a
    # real identity signal, ablation on the structured scenario must shift
    # responses dramatically more than ablation on this stripped version.
    # Contents are also shuffled across observations to further break the
    # content-position coupling.
    rng = random.Random(null_shuffle_seed)
    contents = [o.content for o in scenario_observations]
    rng.shuffle(contents)
    null_scenario = [
        Observation(
            timestamp=o.timestamp,
            speaker=o.speaker,
            content=c,
            tags=("neutral",),
            affect=o.affect,
        )
        for o, c in zip(scenario_observations, contents)
    ]
    null_coupled = _gather_token_distribution(
        factory, True, null_scenario, probe_cue, list(base_seeds), recalls_per_run,
    )
    null_ablated = _gather_token_distribution(
        factory, False, null_scenario, probe_cue, list(base_seeds), recalls_per_run,
    )

    jsd_ablation = _jensen_shannon(coupled, ablated)
    jsd_anchor = _jensen_shannon(anchor_a, anchor_b)
    jsd_null = _jensen_shannon(null_coupled, null_ablated)

    anchor_ratio = jsd_ablation / jsd_anchor if jsd_anchor > 0 else float("inf")
    null_ratio = jsd_ablation / jsd_null if jsd_null > 0 else float("inf")

    return V4Result(
        jsd_ablation=jsd_ablation,
        jsd_anchor=jsd_anchor,
        jsd_null=jsd_null,
        anchor_ratio=anchor_ratio,
        null_ratio=null_ratio,
    )
