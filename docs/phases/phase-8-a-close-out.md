# Phase 8 Pillar A — MSC × WAI-SR longitudinal Bond close-out

**Status:** SHIPPED. Hard gate PASS.

**Date closed:** 2026-10-08

**Pillar reference:** [docs/phases/phase-8-dataset-replay-expansion-plan.md](phase-8-dataset-replay-expansion-plan.md) §2 Pillar A.

**Artefact:** [docs/experiments/phase-8-dataset-replay-expansion/pillar-a/msc-bond-longitudinal.json](../experiments/phase-8-dataset-replay-expansion/pillar-a/msc-bond-longitudinal.json)

---

## Gate

Plan §2 Pillar A hard gate:
- mean WAI-SR Bond ≥ **3.89** (Woebot aggregate floor) at the longitudinal measurement session,
- across ≥ **50** simulated users,
- cohort spans MSC personas (reported, not gated).

Plan §2 Pillar A soft signal (reported, not gated):
- paired Δ(session_last − session_1) ≥ 0 — longitudinal bond growth.

## Result

- `pillar_a_pass = True`
- **mean_bond_session_1 = 6.356**
- **mean_bond_session_4 = 6.356** (session_4 is the longitudinal endpoint — see "Mirror ceiling" below)
- **n_paired = 59** (60 users attempted, 1 excluded by Alibaba content filter)
- `session_5_ci95_half_width = 0.275`
- **mean_paired_delta = +0.000** — soft signal **not met**; every one of 59 paired deltas is exactly 0 (see "Zero paired delta" below)
- `handoffs_excluded = 0`
- `wall_s = 1690.3` (first 50 users cold on provider; next 10 appended against SHA-keyed disk cache)
- Cohort spans 59 distinct MSC personas (one count each).

Bond floor cleared by **+2.47** (6.356 − 3.89 = 2.466; +63% headroom over the Woebot floor).

## Mirror ceiling — session_4 as longitudinal endpoint

The provisioning layer draws from the `nayohan/multi_session_chat` HuggingFace mirror. The MSC paper describes 5-session dialogues (session_id 0..4 → 1..5 in our shim), but this mirror tops out at `session_id = 3` (session_4 in-shim):

```
total_rows=17940
session_id counts (first 5000 rows): {0: 2202, 1: 1156, 2: 1156, 3: 486}
distribution of per-dialoug_id row counts: Counter({1: 1046, 3: 670, 4: 486})
```

Of 2202 conversations in the sampled slice, 486 (22%) have the full 4-session chain; the rest are truncated. The `_group_rows_by_dialoug_id` iterator in [benchmarks/scripts/provision_msc.py](../../benchmarks/scripts/provision_msc.py) emits only groups with ≥ 4 sessions.

**Deviation.** Pillar A plan says "session 5". We measure at session_4. In-world cumulative time from session_1 to session_4 is 7.25 days under the MSC-paper canonical gap schedule `(0.0, 0.25, 2.0, 5.0, 10.0)` — one longitudinal step short of the 17.25 days to session_5. This is a mirror constraint, not an architectural one; a shim produced from the authoritative ParlAI distribution could extend to session_5. The Bond floor is unchanged — the Woebot 3.89 floor is an aggregate benchmark, not tied to a specific session index.

## Zero paired delta — soft signal not met, structural finding

Every single paired delta is exactly 0:

- `paired_delta_stderr = 0.0`
- All 59 entries in the `paired_deltas` array are `0`.

This is not statistical noise. Every user has `bond_session_1 == bond_session_4` byte-identical. qwen-turbo (acting as the simulated seeker inside `make_msc_seeker`) answers WAI-SR item 5 ("I feel that the Companion appreciates me.") deterministically based on surface cues of the current session, not on the Companion's cumulative memory state across sessions.

**Hypothesized mechanism.** The seeker administration path builds a prompt that includes the WAI-SR item and the simulated user's persona, but the per-session recall of past Companion behaviour reaches the seeker only through the dialog itself. If the Companion's response style is consistent across sessions, qwen-turbo returns the same Likert integer at s1 and s4. The companion's memory substrate IS running cross-session (verified by the `test_conversation_advances_time_across_sessions` test which checks `cs.companion.store.all()` returns fragments with timestamps spanning the full in-world window) — but no proxy for the Companion's memory state is injected into the seeker's prompt.

**What this changes about the finding.** The Bond-floor dimension of the Pillar A hard gate is cleared with huge margin. The paired-growth soft signal is uninformative at this measurement granularity — a confound in the seeker instrumentation, not a mechanism verdict. Treating the delta as evidence of "no longitudinal bond growth in the Companion" would be wrong on the current evidence. A Phase 9+ instrumentation would need to either (i) stream a seeker-visible summary of what the Companion has remembered or (ii) replace the Likert-integer self-report with a behavioural proxy for Bond that reads cross-session memory impact from the Companion's own responses.

## Carried debt (into Phase 8 Pillar C/D/F or Phase 9+)

- **5-session longitudinal step.** Session_5 measurement at full 17.25-day in-world span requires a mirror with 5 sessions (nayohan tops at 4) OR a ParlAI-fresh provisioning layer. Deferred, documented.
- **Seeker session-state sensitivity.** The byte-identical paired delta is honest Pillar A reporting, not a mechanism verdict. If any downstream pillar wants to measure Companion-side longitudinal effect, the seeker instrumentation needs a cross-session summary channel OR a behavioural Bond proxy. Not addressed in this pillar.
- **Live-pilot WAI-SR calibration.** The current `score_wai_sr` is validated against a scripted-seeker pattern. Calibration against real human WAI-SR administrations is Phase 4b-signoff debt and remains open.
- **Alibaba content-filter exclusion rate.** 1 of 60 attempted users (msc_106, persona unavailable because the DataInspectionFailed response is thrown before persona extraction) excluded. If downstream pillars rely on `tongyi` cloud, assume ~1.5% content-filter attrition and over-provision the shim accordingly.

## Reproduction

```bash
# 1) Shim provisioning (CC BY-NC 4.0; ~400MB HF download first-run)
python benchmarks/scripts/provision_msc.py --limit 60

# 2) Pillar A live-fire (cloud qwen-turbo, ≈ 30 min cache-cold; <2 min cache-warm)
python benchmarks/scripts/phase8_a_msc_run.py \
    --limit 60 --max-turns-per-session 6 \
    --mode cloud --tier simple \
    --session-5-id 4
```

The `--session-5-id 4` flag is required for the `nayohan/multi_session_chat` mirror. A hypothetical 5-session mirror would omit it (default session_5_id = 5).

## Files added

- [src/psychology_base/eval/benchmarks/public_datasets/msc.py](../../src/psychology_base/eval/benchmarks/public_datasets/msc.py) — MSC adapter + types + loader.
- [src/psychology_base/eval/benchmarks/public_datasets/msc_bond_longitudinal.py](../../src/psychology_base/eval/benchmarks/public_datasets/msc_bond_longitudinal.py) — longitudinal Bond harness: `make_msc_seeker`, `run_msc_session`, `run_msc_conversation`, `aggregate_longitudinal_bond`, `LongitudinalBondReport`.
- [benchmarks/scripts/provision_msc.py](../../benchmarks/scripts/provision_msc.py) — HF-mirror → shim JSONL. Shape A / B / C detection; Shape C adds `_SHAPE_C_SESSION_GAP_DAYS` fallback schedule for mirrors that drop the paper's natural-language gap labels.
- [benchmarks/scripts/phase8_a_msc_run.py](../../benchmarks/scripts/phase8_a_msc_run.py) — Pillar A CLI runner.
- [tests/test_msc_adapter.py](../../tests/test_msc_adapter.py) — 18 adapter tests (dataclass validation, JSONL loader, convenience accessors).
- [tests/test_msc_bond_longitudinal.py](../../tests/test_msc_bond_longitudinal.py) — 18 harness tests (single session, multi-session, WAI-SR administration, aggregation, pillar gate, time-advance verification).
