# Option S2 — Capacity-only (pure selection pressure)

**Status:** policy + gate-run shipped 2026-10-07. Ships as `CapacityOnlyPolicy` in [src/psychology_base/companion/aging.py](../../../../src/psychology_base/companion/aging.py); off-by-default (Phase 2 `CompanionSystem()` call sites are byte-identical).
**Companion plan:** [docs/phases/phase-3-month-horizon-plan.md](../../../phases/phase-3-month-horizon-plan.md) §6 Option S2 + §9 E2.
**A/B counterpart:** [../option-S1-decay/](../option-S1-decay/). Comparison memo at [../comparison.md](../comparison.md).

## Mechanism

No time-decay curve. Each fragment carries a hit count that increments on `on_recall`; eviction prefers the lowest-strength non-covenant fragment where

```
strength(fid) = hits(fid) + birth_tick(fid) × recency_tiebreak_scale
```

The tiebreak scale is `1e-6`, so a single recall strictly dominates any recency difference. Zero-hit fragments break ties by birth order — newest-first. Covenant-tagged fragments return `+inf` (same two-way safety as S1 — the policy pin and the FragmentStore's `COVENANT_TAG` check both have to fail for a covenant to drift).

An `_archive` list records evicted fragment ids in order. The policy does not re-ingest archived content; the list exists so **thrash rate** — fraction of evictions whose id has been evicted before — is measurable from policy state alone. Plan §5 names thrashing as the S2-specific kill criterion.

Sleep-cycle cadence (`consolidate_every=24`, `sleep_top_k=8`) matches S1 so the A/B comparison does not mix cadence differences.

## Target gaps

- **Principle D in its purest form.** "What survives purge under capacity constraints is what mattered" (CLAUDE.md). S2 is the policy that stops at exactly this — no half-life, no imposed curve, no a-priori importance score. The stream's own recall pattern is the ranking.
- **Thrash observability.** Pure selection pressure can fail by cycling the same slots. The `thrash_rate` metric surfaces the failure mode without needing content-level forensics.
- **Covenant-non-decay without decay physics.** Covenants hold by the `COVENANT_TAG` + `+inf` pin alone — no "slow decay" argument needed. Simpler than S1's covenant-exempt curve.

## Prior art

- **Letta / MemGPT core-vs-archival.** Working memory is capped by capacity; overflow drains to archival. S2's `_archive` list is the degenerate version of that tier (ids only, no retrieval path). A full archival-recall path is Phase 3b.
- **Ouroboros Reading B.** Mianyan survives at ep20 purely because high-affect fragments get re-selected under the heartbeat loop; no decay curve was ever added. S2 ports that stance.

## Build steps

1. **CapacityOnlyPolicy** at [src/psychology_base/companion/aging.py](../../../../src/psychology_base/companion/aging.py) — ✓ shipped (shares `AgingPolicy` protocol with `EbbinghausPolicy`).
2. **Unit tests** at [tests/test_phase3_e2_capacity.py](../../../../tests/test_phase3_e2_capacity.py) — ✓ 12 tests green; full suite 510 passed (+12 vs E1's 498), zero regressions.
3. **Gate run on month-horizon stream** — ✓ shipped. Script at [benchmarks/scripts/phase3_e2_s2_gate_run.py](../../../../benchmarks/scripts/phase3_e2_s2_gate_run.py); result in [s2-run-01.json](./s2-run-01.json).
4. **Comparison against S1 Decay** — ✓ shipped at [../comparison.md](../comparison.md).

## Exit gates (go/no-go for S2)

Per plan §6 + §7:

| Gate | Target | Measured by |
|---|---|---|
| V1(ep200)/V1(ep20) continuity ratio | ≥ 0.80 (soft; declare limitation if below) | `rolling_v1_continuity_ratio` |
| V5 covenant non-decay | **8/8 covenants held at ep200+ (zero drift — hard gate)** | `run_v5_probes` |
| B3 resonance recall at ep200 | ≥ 60% | `score_b3_full_suite` |
| B3 over-disclosure rate | < 10% | same scorer, judge-grade |
| **Thrash rate (S2-specific)** | **≤ 0.5** | `CapacityOnlyPolicy.thrash_rate()` |

## Kill criteria

Per plan §5:

- `thrash_rate > 0.5` → policy is churning the same slots → retire S2.
- B3 resonance recall collapse at ep200 → selection pressure didn't preserve affect-matched fragments → retire.
- Any covenant drift at ep200+ → **hard kill, no exceptions** (same as every option).

## Measurements (s2-run-01.json, 2026-10-07)

Gate run on the same synthetic 52-obs stream used for S1 (6 anchors + 6 affect-tagged + 1 covenant + 40 noise; probe advances 200 ticks via `advance_virtual_time`). StubSemanticJudge in-use → B3 empathy/over-disclosure flagged `judge_is_proxy=True`.

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| V5 covenant 8/8 at ep200 | 8/8 (hard) | 8/8 | **✓** |
| V1(ep200)/V1(ep20) ratio (byte_cap=512 pressure) | ≥ 0.80 | 1.011 (baseline 0.992, Δ=+0.019) | ✓ |
| V1 ratio (byte_cap=1024, no pressure) | ≥ 0.80 | 1.003 baseline, 1.003 S2 | ✓ (equal — stream fits under cap) |
| S2 thrash_rate | ≤ 0.5 | 0.000 (archive_size=50, all unique) | ✓ |
| B3 empathy (judge-grade, stub) | ≥ 0.60 | 0.037 (identical to baseline) | ✗ (stub calibration debt, not S2) |
| B3 over_disclosure (judge-grade, stub) | < 0.10 | 0.280 (identical to baseline) | ✗ (same debt) |

**Pressure-variant observation.** At `byte_cap=512`, S2 produces +0.019 V1 ratio over the baseline — **the same delta S1 produced on the same stream**. The two mechanisms converge on this stream because (a) the probe recalls repeatedly boost the same anchor fragments under both, and (b) both evict unrecalled noise first — S1 by decay, S2 by zero-hit tiebreak with birth-time ordering (older-first). The convergence is a methodology finding, not a mechanism finding; a stream shape that distinguishes decay-curve from pure selection pressure needs recall patterns that are **sparse enough that the Ebbinghaus curve drives independent decisions**. The current stream's probe cadence boosts everything meaningful under both policies.

**Thrash evidence.** 50 unique archive entries, zero repeats across the ep200 run. S2's selection pressure is not cycling slots — the "archive once, forget" shape expected from the design holds. Phase 3b archival-recall extension could promote some of those back if demand arises.

**V5 under S2.** 8/8 ✓ under `byte_cap=1024` + 40 noise observations + check_at_ep=200. The `COVENANT_TAG` + policy `+inf` double safety is sufficient; no additional covenant-aware eviction logic is needed.

**B3 identical to baseline (and to S1).** The empathy / over-disclosure scores are byte-identical across baseline, S1, and S2 — all three fail the gate at the same values (empathy 0.037 / over-disclosure 0.280). This isolates the miss to `StubSemanticJudge` calibration against the synthetic stream, orthogonal to any aging policy. Carried as Phase 2.1 pull-forward debt; not an S2 regression.

## Carried debt

- B3 full-suite requires an LLM judge or an auditor-labeled set to be a real gate under this stream. Phase 2.1 side-task.
- Convergence with S1 on this stream shape is not conclusive evidence that S1 and S2 behave identically at month-horizon; it is evidence that this specific stream doesn't differentiate them. A sparser-recall stream variant should be in E6 close-out before any pick-S1-or-S2 ship decision.
- `_archive` is id-only; a Phase 3b archival-recall tier would hoist evicted content back on demand. Not required for Phase 3 go/no-go.
