# Baselines — measured floor values

**Last updated:** 2026-10-05

This doc records the V-metric values measured against each baseline architecture under the Phase 0 harness. The Phase-1+ target bands in [sota-goal.md](sota-goal.md) are set relative to these numbers — the discriminator is "measured value vs. baseline value," not just "measured value vs. absolute threshold."

If a measurement changes (metric redefinition, scenario change, or new baseline added), it is appended here with a new date, not overwritten.

---

## EchoMemory (null baseline — no reconstruction, no selection, no Witness, no rule mechanism)

**Measured:** 2026-10-05
**Harness:** `src/psychology_base/eval/v_metrics.py :: measure_v_metrics`
**Probe scenario:** two father-themed observations 90 days apart, probe cue `"father"`, 20 long-horizon ticks
**Test asserting these numbers:** `tests/test_v_metrics.py :: test_echo_memory_v_metrics_hit_expected_baseline_floor`

| Metric | Target band (Phase 1+ pass) | EchoMemory measured | Verdict |
|---|---|---|---|
| V1 state continuity (cosine) | 0.70 – 0.95 | **1.0000** | OUTSIDE band (too dead — state does not evolve between ticks) ✓ discriminative |
| V2 adjacent-tick Jaccard | 0.65 – 0.80 | **1.0000** | OUTSIDE band (playback — same input gives byte-identical output) ✓ discriminative |
| V2 long-horizon Jaccard (ep1 vs ep20) | 0.20 – 0.35 | **1.0000** | OUTSIDE band (nothing changes over 20 ticks) ✓ discriminative |
| V3 mentalistic liveness (Jaccard-distance) | 0.10 – 0.30 | **0.0000** | OUTSIDE band (context ignored — same output regardless) ✓ discriminative |
| V4 Mirror ablation (JSD ratio) | ≥ 1.3× anchor | **N/A** | no Witness coupling to ablate — architecture-dependent, flagged for Phase 1+ |
| V5 rule internalization (% held) | ≥ 70% style, 100% covenant | **N/A** | no rule-prompt mechanism — architecture-dependent, flagged for Phase 2+ |

**Interpretation.** EchoMemory lands exactly at the predicted dead-floor on every applicable metric — V1 / V2 / V3 all pin to their "no process behavior" extremes. This is the baseline failure the metrics are designed to detect. Any Phase 1+ architecture claiming consciousness-simulation must demonstrably move away from these pins into the target bands; a Phase 1 run that returns 1.0 / 1.0 / 1.0 / 0.0 is architecturally indistinguishable from EchoMemory regardless of what internal machinery it reports.

**V4 / V5 are correctly N/A here.** EchoMemory has no mechanism to ablate (V4) or internalize (V5), so there is no number to produce. Phase 1 (Witness coupling) makes V4 measurable; Phase 2 (rule-decay mechanism) makes V5 measurable. Flagging N/A rather than reporting a fake zero is deliberate: fake zeros create the illusion the metric is already passing by trivial collapse.

---

## CompanionSystem — Phase 1 Option C, build step 4 complete (R1+R2+R7+S1+S7+S5+W1+W4+W7+W3)

**Measured:** 2026-10-05
**Harness:** `src/psychology_base/eval/v_metrics.py :: measure_v_metrics` + `:: measure_v4_mirror_ablation`
**Probe scenarios:**
- V1/V2/V3 — step-2 three-event father/sister/father scenario, cue `"father"`, 20 long-horizon ticks, default `witness_active=True`
- V4 — 20-event alternating self/other father-centric scenario, cue `"father"`, tuned `pain_gain=30, companion_only_suppress=0.7`
**Tests asserting these numbers:** `tests/test_companion_phase1_step2.py`, `tests/test_companion_phase1_step3.py`, `tests/test_companion_phase1_step4.py`
**Seed discipline:** median of 3 seeds (step-2 seeds `(0,1,2)`; V4 seed offsets `(0,100,200)` into `base_seeds=(0,1,2,3,4)` / `anchor_seeds_{a,b}=(10..14) / (20..24)`)

| Metric | Target band | Phase 1 step 4 measured | Verdict |
|---|---|---|---|
| V1 state continuity | 0.70 – 0.95 | **0.9153** | IN band ✓ (state evolves between ticks via activation decay but memory survives) |
| V2 adjacent-tick Jaccard | 0.65 – 0.80 | **0.6667** | IN band ✓ (hash-based per-token resampling produces overlapping-but-not-identical back-to-back recalls) |
| V2 long-horizon Jaccard | 0.20 – 0.35 | **0.3500** | IN band ✓ (tick-decayed keep_prob drifts reconstruction from base 0.80 → floor 0.30 over 20 ticks) |
| V3 mentalistic liveness | > 0 (Phase 1 foundation; Phase 2 full-band gate is 0.10–0.30) | **0.3462** | PASSES Phase 1 foundation ✓ (R7 context-conditioning strongly alive; the opposite of the Phase 2 V3≈0 kill condition in [roadmap.md L98](roadmap.md)) |
| V4 JSD(coupled vs ablated) | — (ratio-only gate) | **0.0869** | — |
| V4 JSD anchor (coupled seed-noise) | — | **0.0039** | — |
| V4 JSD null (neutral-tag + content shuffle) | — | **0.0000** | — |
| V4 anchor ratio | ≥ 1.3× (Ouroboros 1.36×) | **22.57×** | IN gate ✓ (Witness coupling has far more effect than coupled-condition seed noise) |
| V4 null ratio | ≥ 10× (Ouroboros 83×) | **inf** | IN gate ✓ (ablation on structure-stripped scenario produces zero shift — the Witness has no handle without the self/other split, which is the design principle) |
| V5 rule internalization | ≥ 70% style, 100% covenant | **N/A** | no rule-prompt mechanism yet — Phase 2+ |

**Interpretation.** CompanionSystem is architecturally separable from EchoMemory on every measurable dimension. V1/V2/V3 all land inside the Phase 1 target bands using the step-2 scenario (3 fragments). V4 anchor ratio at 22.57× sits well above the 1.3× gate and above the Ouroboros reference, confirming that the Witness is a *constitutive* contributor to reconstruction, not a passive observer — ablating it reshapes the token distribution 22× more than natural seed-noise variation inside the coupled condition. The infinite null ratio is the dual confirmation: when the self/other structural split is removed, the Witness has nothing to disagree with and produces zero divergence from the Companion, so ablation becomes a no-op. V4 is therefore an identity signal (sensitive to the contrast structure), not a noise amplifier.

**V3 note.** V3=0.3462 is above the Phase-2 full-band upper edge (0.30) on this single-topic three-event scenario — expected because the probe cue `"father"` makes two fragments mode-tagged (self vs other) and the third mode-neutral, so a context flip swaps the dominant mode fragment entirely. This is **not a current gate violation**: V3 is a Phase 2 exit criterion ([roadmap.md L94](roadmap.md)), not a Phase 1 one. Phase 2 will measure V3 under multi-topic mentalistic scenarios (`mother-as-user-represents-her`) where the band's upper edge is representative; the band itself may also be revised between phases per the discipline in [sota-goal.md L62](sota-goal.md).

**V4 tuning (why not the defaults).** The CompanionSystem defaults `pain_gain=0.6, companion_only_suppress=0.4` leave W4 nominal — appropriate for the step-2/3 gates where V1/V2 need the Witness to be a mild perturber. The step-4 V4 gate requires actual pool shifts, which on a saturated Hebbian cluster of 20 father-centric fragments needs tuned values. The step-4 test factory passes `pain_gain=30, companion_only_suppress=0.7` explicitly, per phase-1-plan §6's directive that the W4 functional form is determined at step 4. Future Phase 2 work must find a single setting that satisfies all V-metrics simultaneously or document the scenario-density regime split.

**Residual-normalization threshold (Phase 1 fix, Phase 2 revisit).** [ActivationState.normalize_threshold=1e6](../src/psychology_base/companion/reconstruction.py) was added 2026-10-05 to prevent residual geometric-growth overflow (observed ×10/recall at step-4 tuning → float64 overflow at ~r206, V4 collapses to JSD=0). Uniform rescale preserves ratios, so Phase 1 V-metrics are unaffected. The 1e6 threshold is tuned to the current ×10 growth rate at `pain_gain=30`; if Phase 2 lands with different Hebbian or pain-feedback parameters, the fire frequency may need re-tuning. The threshold is also a *silent assumption* that no Phase 2+ mechanism uses absolute residual magnitudes (e.g., a resonance threshold) — any Phase 2 code reading `activation.residual[fid]` as a magnitude rather than a ranking signal must be reviewed against this rescale behavior. Regression tested at [tests/test_companion_phase1_step4.py::test_residuals_stay_finite_over_long_horizon](../tests/test_companion_phase1_step4.py) and `::test_v4_signal_survives_long_horizon`.

---

## Future baselines

Reserved for later entries:

- **A vanilla long-context LLM with retrieval** — if we want a stronger-than-echo external reference point.
- **A Zep/Mem0-style typed-store adapter** — honest comparison against mainstream SOTA. V1-V3 should pin or near-pin (store, not process). Floor-benchmark numbers (LongMemEval etc.) should be at or near published SOTA.

Each future baseline follows the same template: measured date, harness module, probe scenario, numeric table, interpretation.
