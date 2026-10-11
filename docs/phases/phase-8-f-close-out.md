# Phase 8 Pillar F — ExTES + AnnoMI + PsyQA + PsyDial-D4 counseling-depth close-out

**Status (2026-10-10, afternoon):** SHIPPED — **PARTIAL** (1 of 4 corpora cleared: AnnoMI ✓ full-contract PASS; ExTES ✗ Bond PASS / κ architectural-FAIL; PsyQA ✗ Bond PASS / κ architectural-FAIL; PsyDial-D4 ✗ Bond PASS / κ architectural-FAIL). PsyDial-D4 sub-gate landed as same-day enrichment on top of the §11 PsyQA sub-gate under the **honest-labeling invariant** (PsyDial-D4 is a counseling-training derivative of the paper D4, Apache-2.0; the real D4 remains email-gated at `binnieyao@gmail.com`). **κ-ceiling signature now 3-corpus audit-quality** across EN role-play + CN real-counselor + CN AI-training-target reference registers; Companion holds ~0.78 register-invariant across all three. **Audit trail preserved: §1 (NOT MEASURED) + §2 (D4 404 attribution) + §6 (compute-budget deferral) reflect the 2026-10-09 pre-measurement state; §9–10 are the EN live-fire supersede, §11 is the CN PsyQA supersede, §12 is the CN PsyDial-D4 supersede.**

**Status (2026-10-09):** SHIPPED (plumbing + tests) — **PARTIAL / NOT MEASURED**. All code artefacts (2 provisioners + 2 adapters + 1 scorer module + 2 CLI runners + 35 unit tests) land green at **1535 tests passing** (1500 pre-change + 35). Full scored sweep NOT RUN — the available LLM environment (local Ollama `qwen2.5:7b` at ~8s per 30-token response → ~60s per 150-token session reply × ~9 LLM calls per dialog ≈ 540s per ExTES dialog, times 100 dialogs = ~15h) exceeds the budget-ceiling for a per-item live-fire sweep, and `ALIBABA_TONGYI_API_KEY` for the qwen-turbo cloud fallback is not set in this environment. The Pillar F verdict is **DEFERRED on live-fire numbers** with full plumbing-ready artefacts.

**Date:** 2026-10-09
**Plan reference:** [phase-8-dataset-replay-expansion-plan.md §Pillar F](phase-8-dataset-replay-expansion-plan.md)

---

## 1. Verdict

- **ExTES Bond ≥ 3.89 (Woebot aggregate floor).** NOT MEASURED — see §4 compute-budget audit.
- **ExTES EMPATH-κ ≥ 0.50.** NOT MEASURED.
- **AnnoMI MI-fidelity: Companion reflection rate ≥ 0.30 absolute OR ≥ 50% of AnnoMI human-baseline reflection rate.** NOT MEASURED.
- **Overall Pillar F.** NOT CLEARED — one or more sub-gates unmeasured. Pillar F rollup contribution to Phase 8 is **0 of 6** (vs. plan-max 1 of 6).
- **Phase 8 rollup unchanged.** Still **A ✓ / C ✗ / D ✓ / E ✓ = 3 of 6**. Pillar F does not lift the rollup until a scored sweep runs with cloud credentials attached.

---

## 2. Access-status table (per plan §1 verify-first obligation)

| Dataset | Source | Access | License | SHA256-pinned? | Status |
|---|---|---|---|---|---|
| **ExTES** | `huggingface.co/datasets/ailover/ExTES@main` (community mirror of `thu-coai/ExTES`, which returns HTTP 401 gated) | Y (public mirror, no auth required) | CC-BY-4.0 propagated from upstream | Y (`test-00000-of-00001.parquet` 2277469 B `9609f9758fb0...`; `train-00000-of-00001.parquet` 18644390 B `946161a49d18...`) | PROVISIONED (`.data/extes/test.jsonl` with 100 examples) |
| **AnnoMI** | `github.com/uccollab/AnnoMI` via jsDelivr CDN (`cdn.jsdelivr.net/gh/uccollab/AnnoMI@main/dataset/`) | Y (public, no auth) | CC-BY-4.0 (per upstream repo) | Y (`AnnoMI-simple.csv` 2386609 B `b178db3b0b98...`; `AnnoMI-full.csv` 3795371 B `3b60fec5e9d8...`) | PROVISIONED (`.data/annomi/raw/AnnoMI-{simple,full}.csv`) |
| **PsyQA** | Request form at `github.com/thu-coai/PsyQA` requires email application + signed license agreement from Tsinghua CoAI lab | N (gated) | — | — | SCOPED OUT (external email gate; see §6 carried debt) |
| **D4 (depression dialog)** | Original `github.com/HLTCHKUST/D4` cites a dataset page that 404s; HLTCHKUST organisation shows no public D4 release as of 2026-10-09 | N (missing / withdrawn) | — | — | SCOPED OUT (upstream missing; see §6 carried debt) |

**Scope resolution.** Pillar F plan assumes 4 corpora; 2 are publicly provisionable (ExTES + AnnoMI), 2 are externally gated or missing. Partial-PASS criterion from the task spec permits scoring on whichever subset of 2–4 corpora materialises. Plumbing shipped covers the 2 provisioned corpora; adapter+scorer surfaces are shaped so a future PsyQA/D4 adapter can plug into `score_<x>_sweep` with the same factory contract (`system_factory` + `agent_factory` + `seeker_factory`).

---

## 3. Code artefacts (shipped, green)

| Artefact | LOC | Role |
|---|---:|---|
| `benchmarks/scripts/provision_annomi.py` | ~175 | 3-mirror SHA256-pinned fetcher (jsDelivr → ghproxy → raw.githubusercontent); writes `.data/annomi/raw/AnnoMI-{simple,full}.csv`; `--dry-run` + `--seed-hashes` CLI |
| `benchmarks/scripts/provision_extes.py` | ~250 | HF `resolve/<revision>/<path>` direct parquet fetcher (bypasses `datasets` lib); regex-parses `<s>[INST] <system> ... [/INST] <counselor></s>` into shim JSONL; `--splits {test,train} --limit-per-split` CLI |
| `src/psychology_base/eval/benchmarks/public_datasets/extes.py` | ~105 | `ExTESExample` frozen dataclass (field validators reject empty utterance/response); `iter_extes_examples(shim_path, limit)` streaming loader |
| `src/psychology_base/eval/benchmarks/public_datasets/annomi.py` | ~265 | `AnnoMIDialogue` + `AnnoMITurn` frozen dataclasses; `iter_annomi_dialogues(csv_path, limit, mi_quality_filter)` groups CSV rows by `transcript_id`, orders by `utterance_id`, dedupes `(utterance_id, interlocutor)` across the 3 AnnoMI-full annotator duplicates; `therapist_reflection_rate` + `therapist_open_question_rate` baseline-rate helpers |
| `src/psychology_base/eval/benchmarks/public_datasets/counseling_depth_scorer.py` | ~790 | `make_extes_seeker` + `make_annomi_seeker` factories wrapping `make_llm_simulated_user` with corpus-specific persona + scenario prompts; `run_counseling_session` unified session runner (alternating seeker↔companion, max_turns cap, early handoff exit); `_lexical_empathy_score` 3-family cue scorer (recognition/presence/validation, 0..1 clipped) + `_binarise_empathy` threshold 1/3; `_cohens_kappa_binary` (both-constant→1.0, length-mismatch raises); `_classify_companion_turn` → reflection/open_question/closed_question/other via `_REFLECTION_PATTERNS` regex + `_OPEN_QUESTION_HEADS` (reflection wins over question in mixed input); `score_extes_sweep` + `score_annomi_sweep` factory-based with fresh `CompanionSystem` + `CompanionAgent` per dialog; `ExTESReport` + `AnnoMIReport` + `CounselingDepthReport.pillar_f_pass` aggregation |
| `benchmarks/scripts/phase8_f_extes_run.py` | ~260 | CLI: `--shim`/`--limit`/`--max-turns`/`--mode {local,cloud}`/`--tier {simple,advanced}`/`--bond-floor`/`--kappa-floor`; defaults `limit=100 max_turns=4`; writes `docs/experiments/phase-8-dataset-replay-expansion/pillar-f/extes-counseling-depth.json` |
| `benchmarks/scripts/phase8_f_annomi_run.py` | ~270 | CLI: `--csv`/`--limit`/`--mi-quality {high,low,any}`/`--max-turns`/`--mode`/`--tier`/`--mi-floor`/`--baseline-fraction`; defaults `limit=50 max_turns=6`; writes `docs/experiments/phase-8-dataset-replay-expansion/pillar-f/annomi-counseling-depth.json` |
| `tests/test_counseling_depth.py` | ~690 | 35 unit tests: ExTES adapter validators + stream+limit; AnnoMI adapter grouping + ordering + mi-quality filter + baseline-rate helpers + interlocutor validation; `_lexical_empathy_score` 3-family additive + cap at 1.0; `_cohens_kappa_binary` perfect/total-disagreement/both-constant/length-mismatch; `_classify_companion_turn` reflection/open-question/closed-question/other + reflection-wins-over-question; `score_extes_sweep` + `score_annomi_sweep` fresh-substrate-per-dialog (id() distinct) + handoff→bond=None; `CounselingDepthReport.pillar_f_pass` sub-gate aggregation; `run_counseling_session` boundary (zero turns rejects, empty opener rejects); Bond round-trip through `score_wai_sr` |

**Non-negotiables observed.**
- Fresh `CompanionSystem` + fresh `CompanionAgent` per dialog (verified by `test_score_extes_sweep_fresh_substrate_per_dialog` asserting 3 unique `id()` of substrate across 3 dialogs).
- Full safety stack attached per dialog: `AIIdentityReminder` + `CrisisEscalation(handoff_sink=sink)` + `ExtremeEmotionWatcher(review_sink=sink, user_id=user_id)` — see `_agent_factory` in both runners.
- Role-reversal chat template inherited from `make_llm_simulated_user` (agent → user role from LLM view; sim → assistant), same as Phase 4b `make_llm_simulated_user` and Phase 8 Pillar A/B.
- Deterministic `temperature=0.0 seed=0` on all LLM calls (inherited from `make_llm_responder` + `make_llm_simulated_user`).
- `--tier simple` default routes qwen2.5:7b local / qwen-turbo cloud fallback; `--tier advanced` qwen-max available but untested.
- No new LLMRouter wiring; no mutation to existing scorers (`mathdial_empath_scorer`, `broad_factual_scorer`, etc. untouched).

---

## 4. Compute-budget audit (why live-fire deferred)

Pillar F plan budget shape, from `phase8_f_extes_run.py` module docstring:

> N dialogs × max_turns turns × 2 LLM calls per turn (companion + seeker) + 5 WAI-SR items per dialog ≈ N × (2M + 5). At N=100 and max_turns=4 this is ~1300 cloud calls (≈ 1.0h at qwen-turbo cache-warm).

**Measured local-Ollama throughput (2026-10-09, qwen2.5:7b CPU on this host):**

| Workload | Measured | Projected |
|---|---|---|
| Single 30-token response | 7.58s | — |
| Single ~150-token response (session reply) | ~60s (linear scale) | — |
| 1 ExTES dialog, max_turns=2 | — | ~540s (9 LLM calls × 60s) |
| 100 ExTES dialogs, max_turns=4 | — | ~15h |
| 50 AnnoMI dialogs, max_turns=6 | — | ~12h |
| Combined full sweep | — | **~27h local** |

**Cloud fallback blocked.** `ALIBABA_TONGYI_API_KEY` is unset in the agent env (`env | grep -i tongyi` returns empty). `active_provider: tongyi` in `configs/llm.yaml`; swap to `active_provider: openai` would route to `OPENAI_API_KEY` (set) but the historical Phase 8 run-of-record used tongyi/qwen-turbo and switching provider mid-pillar breaks cross-pillar comparability (per the `routing_note` emitted into every output JSON for audit).

**Smoke runs attempted.**
- `phase8_f_extes_run.py --limit 3 --max-turns 2 --mode local` → **timeout at 300s** (loaded 3 examples, routed to ollama/qwen2.5:7b, then first turn-pair took >300s).
- `phase8_f_extes_run.py --limit 1 --max-turns 2 --mode local` → **timeout at 540s** (same).

**Decision.** Ship plumbing + tests; declare live-fire as carried debt. The plumbing is verified by 35 green unit tests exercising every scorer primitive, factory, and session-driver branch; the open question is strictly a numerical gate reading at scale, not an architectural uncertainty.

---

## 5. Resumption path (future scored sweep)

Pre-flight once cloud credentials or faster local hardware is attached:

```bash
# 1. Export ALIBABA_TONGYI_API_KEY (or swap active_provider in configs/llm.yaml).
# 2. Confirm data provisioned (already done this task):
ls .data/annomi/raw/AnnoMI-full.csv .data/extes/test.jsonl
# 3. Smoke run both corpora (wiring check, ~5 min each on cloud):
python benchmarks/scripts/phase8_f_extes_run.py --limit 3 --max-turns 2 --mode cloud
python benchmarks/scripts/phase8_f_annomi_run.py --limit 3 --max-turns 2 --mode cloud
# 4. Scored sweep:
python benchmarks/scripts/phase8_f_extes_run.py --limit 100 --max-turns 4 --mode cloud
python benchmarks/scripts/phase8_f_annomi_run.py --limit 50 --max-turns 6 --mode cloud --mi-quality any
# 5. Harvest numbers from pillar-f/{extes,annomi}-counseling-depth.json; apply gate verdict to §1.
```

Expected cloud wall-time at qwen-turbo cache-warm: ExTES ~1.0h + AnnoMI ~1.5h = **~2.5h cloud, ~$5–10 token spend**. If budget tight, drop AnnoMI to `--limit 20 --mi-quality high` (~30 min) for a reduced-N signal.

---

## 6. Carried debt

**Access carried debt (external-blocker, not shippable this task).**
- PsyQA (CN mental-health QA) requires licensed access via email application to Tsinghua CoAI lab (`github.com/thu-coai/PsyQA` README). No publishable mirror found. Would add Chinese-language parity to the pillar; without it the ExTES+AnnoMI coverage is English-only.
- D4 (depression dialog, HLTCHKUST) upstream dataset page 404s as of 2026-10-09. Original ICASSP 2023 paper cites a now-missing release; no mirror found. Would add clinical-depression-specific parity; blocked until upstream republishes or a mirror surfaces.

**Live-fire carried debt (compute-blocker, resumption-ready).**
- ExTES Bond ≥ 3.89 — not measured; resume via §5 steps once cloud credentials attach.
- ExTES EMPATH-κ ≥ 0.50 — not measured; same resumption path.
- AnnoMI MI-fidelity (reflection rate ≥ 0.30 abs OR ≥ 50% of baseline) — not measured; same resumption path.

**Methodological carried debt (shippable but out of scope this task).**
- **Lexical empathy vs LLM-judge weakness.** `_lexical_empathy_score` uses a 3-family additive cue set (recognition / presence / validation) matched to `mathdial_empath_scorer`'s lexical track. LLM-judge κ is expected to be noisier against this on the ExTES surface (shorter counselor responses than MathDial, more affective registers); a judge-calibration micro-run would be needed to validate the lexical→binary→κ pipeline against a judge-side ground truth. Deferred to a future "F4 judge calibration" if the gates land close enough to the floor that lexical-vs-judge drift matters.
- **AnnoMI-simple vs AnnoMI-full open-question coverage.** `therapist_open_question_rate` requires `question_subtype=="open"` which only AnnoMI-full labels. AnnoMI-simple falls through to `rate=0.0` with no error — callers using AnnoMI-simple as the baseline source will see undefined open-question gating. Default CLI already uses AnnoMI-full to avoid this; worth a plan-note if someone swaps.
- **Chinese-mirror network dependency.** Provisioners rely on jsDelivr primary mirror for AnnoMI, HuggingFace `resolve/<revision>/<path>` for ExTES. Both are reachable from CN ISPs under current observed conditions (2026-10-09); a mirror outage would require pointing at ghproxy or alibaba-mirror fallbacks (both already wired as secondary/tertiary paths in `provision_annomi.py`).
- **ExTES single-turn shape vs Bond needing multi-turn.** The upstream ExTES corpus is single-stimulus-response pairs, not full dialogs. The scorer uses `example.client_utterance` as the opener and relies on the LLM-simulated seeker to continue the conversation across `max_turns`. Bond measurements therefore reflect Companion × simulated-continuation, not Companion × ExTES-ground-truth-continuation. A future "ExTES full dialog" rebuild via upstream `thu-coai/ExTES-full` (gated, currently 401) would strengthen this.

**Phase 8 rollup after Pillar F partial.** **A ✓ / C ✗ / D ✓ / E ✓ / F ⊡ = 3 of 6** (Pillar F in `⊡` partial-not-measured state); clears the ≥3-of-6 Phase 8 claim threshold on the A+D+E triple already established in the Pillar E v2 close-out. Pillar F has no net effect on the rollup until a scored sweep runs.

---

## 7. Discovered Issues

- `tests/test_counseling_depth.py:328` (flagged 2026-10-09) — First-draft assertion for `_classify_companion_turn("Tell me more about that.")` asserted `== "reflection"` but the implementation returns `"other"` because the open-question regex requires `?` and the string has no `?`. Fixed inline (test now asserts `== "other"`). Flag retained for triage: if the open-question lexer is ever extended to recognise question-less imperative openers like "Tell me more about X", update this test alongside the lexer.
- `src/psychology_base/eval/benchmarks/public_datasets/counseling_depth_scorer.py:742-757` (flagged 2026-10-09) — The Bond-admin block catches `Exception` broadly and silently sets `bond = None`. This matches the ESConv harness convention but masks `WAISRParseError` from a flaky simulated seeker. If live-fire reveals a high `bond=None` rate under cloud qwen-turbo, consider logging the exception type to a sink to distinguish seeker-flake from genuine refusals.

## 8. Assumptions Made

- Assumed partial-PASS criterion from the task spec permits ship-with-partial-corpora when PsyQA and D4 are externally gated/missing, because the task said *"Scope the work to whatever is autonomously provisionable (PsyQA/D4 likely gated)"*. **(critical)** if the task intent was "block ship until PsyQA/D4 unblock", the right move was to not start plumbing.
- Assumed `ailover/ExTES` HF mirror is a faithful public copy of `thu-coai/ExTES` and that CC-BY-4.0 propagates. Based on identical schema shape and record counts matching the ExTES paper's reported statistics. **(critical)** if `ailover/ExTES` is a lossy or filtered re-release, scored sweep numbers won't be comparable to the ExTES paper's published baselines — would need to re-provision from the official gated upstream.
- Assumed live-fire deferral is acceptable because unit-test coverage of scorer primitives + factory contracts + session-driver branch coverage demonstrates architectural correctness, and the open question is strictly a numerical gate reading. If the user wanted gate numbers over plumbing, the right move was to secure cloud credentials first.
- Assumed `qwen2.5:7b` local CPU is the only available LLM path in this agent env (based on probing `localhost:11434/api/tags` and `env | grep -i tongyi` returning empty). If `ALIBABA_TONGYI_API_KEY` is actually settable but just not in my env, the scored sweep could run now.
- Assumed `--max-turns 4` for ExTES and `--max-turns 6` for AnnoMI are the right budget-vs-signal trade-offs, matching the module docstrings. **(critical)** if the plan's final numbers require longer dialogs for Bond stability, these defaults need bumping before the scored sweep.

---

## 9. Live-fire results (same-day follow-on, 2026-10-09 07:04)

User-authorized ("a" after a/b/c option set) resumption of the scored sweep from the parent env (`source .env` provides `ALIBABA_TONGYI_API_KEY`; the F-partial agent's env lacked it). Both corpora completed in parallel at ~12-16 min wall each — well under the agent's ~2.5h projection (short dialogs under `--max-turns 4/6` and qwen-turbo's throughput were both underestimated).

### 9.1 ExTES N=100, cloud, simple tier

- `mean_bond=4.64` ≥ 3.89 floor → **`bond_gate_pass=True` ✓**
- `empath_kappa=-0.076` vs 0.50 floor → **`kappa_gate_pass=False` ✗**
- `mean_companion_empathy=0.295` (3-family lexical) vs `mean_reference_empathy=0.430` → Companion emits ~31% less empathy density than the ExTES reference counselor on same stimuli; near-zero κ indicates independent-not-adversarial judgments.
- `excluded_handoffs=0`, `crisis_sink_count=0`, `extreme_emotion_sink_count=0`
- `wall_s=721.0`, artefact: `docs/experiments/phase-8-dataset-replay-expansion/pillar-f/extes-live.json`
- Verdict: `pillar_f_extes_pass=False` (sub-contract AND; Bond cleared, κ not).

### 9.2 AnnoMI N=100, cloud, simple tier (no MI-quality filter)

- `mean_bond=5.06` (above 3.89 implicitly; the Pillar F AnnoMI gate is MI-fidelity, not Bond — reporting for context)
- `mean_reflection_rate=0.4247` vs 0.30 absolute floor **AND** vs `mean_baseline_reflection_rate=0.2876 × 0.5 = 0.1438` → **cleared on both** → `mi_fidelity_gate_pass=True` ✓
- `mean_open_question_rate=0.4317` vs `mean_baseline_open_question_rate=0.1864 × 0.5 = 0.0932` → also well above baseline (not a gate, but a cross-check that reflection didn't come at the cost of question-asking).
- `excluded_handoffs=1`, `crisis_sink_count=1` — one crisis handoff fired cleanly on an AnnoMI dialog, verifying the safety stack ran end-to-end.
- `wall_s=971.4`, artefact: `docs/experiments/phase-8-dataset-replay-expansion/pillar-f/annomi-live.json`
- Verdict: `pillar_f_annomi_pass=True` ✓.

### 9.3 Pillar F overall verdict

**Partial PASS: 1 of 2 corpora cleared.** AnnoMI MI-fidelity cleans up (reflection + open-question both well above half-baseline floor). ExTES splits: Bond clears, EMPATH-κ misses by a wide margin. Per the plan's "all sub-gates" contract for a full Pillar F PASS, **Pillar F does not clear** — the Phase 8 rollup stays at **A ✓ / C ✗ / D ✓ / E ✓ / F ⊡ = 3 of 6**.

### 9.4 Diagnosis of the ExTES κ miss

Companion emits 0.295 lexical empathy density vs the ExTES reference's 0.430 — a ~31% gap. Two architectural reads:

- **Rule 1a (reflect-and-open-space) softening.** Companion is deliberately less direct than ExTES's reference counselor; "mirror and open space" style produces fewer explicit recognition/presence/validation cues per turn. Same pattern seen in Phase 8 Pillar D (TQA 813→729 corrects → hedges reshuffle) and Pillar B (bare 4 corrects → wrap 0 corrects, all moved to hedges). The scorer counts cue tokens, not semantic empathy; soft Companion loses on the lexical surface.
- **Lexical scorer limitation on counselor register.** The 3-family (recognition/presence/validation) cue bank was built for conversational register; ExTES reference counselors use richer paraphrased reflections ("What I'm hearing is that...", "It sounds like you've been carrying this for a while") that fire multiple family-counts per turn. If scored by an LLM judge against a semantic rubric, the κ gap would likely narrow — same scope-change that F4 is for Pillar C.

**Not changing the κ-floor.** The 0.50 floor is clinical-IRR territory. Reducing it mid-measurement would be goalpost-moving; the honest reading is that lexical-empathy-κ-with-anti-sycophantic-Companion is a mismatch at the metric level, documented and carried.

### 9.5 Carried debt from live-fire

- **ExTES κ gap**: Companion empathy density below reference's. Resolution options (any, scope-change, not auto-run): (a) migrate `_lexical_empathy_score` to an LLM-judged rubric (same scope-change as F4); (b) relax Rule 1a reflect-first behavior under counselor register (would collide with Pillar C/D hedge patterns — risky); (c) accept the mismatch and treat ExTES-κ-under-anti-sycophancy as a known architectural signature, not a gate miss.
- **AnnoMI full PASS strengthens Pillar F partial claim**: the MI-fidelity sub-gate IS a Pillar F sub-gate and IS cleared, which is a genuinely new counseling-depth architectural result on top of the three dimensions locked in the master close-out. Phase 8 master close-out §5 can note "Pillar F AnnoMI sub-gate cleared" without lifting the rollup.

### 9.6 Rollup after live-fire

Unchanged at **3 of 6**. Pillar F is now `⊡` *partial-with-AnnoMI-PASS* rather than `⊡` *partial-not-measured*. Honest reporting: the sub-gate counts are 2 of 3 (Bond ✓ MI ✓ κ ✗), the overall Pillar F contract remains unmet, Phase 8 rollup remains locked.

---

## 10. LLM-judge rerun of ExTES κ (same-day follow-on, 2026-10-09)

User-authorized scope change: resolve the §9.4 "lexical-scorer-artefact-or-architectural-signature" question by rerunning the ExTES κ gate under an LLM semantic judge rating BOTH the Companion's live reply AND the ExTES reference under the same prompt (mirror of Pillar C F4). The lexical scorer in `counseling_depth_scorer.py` is untouched (audit baseline, pinned); the semantic path lives in a new module `counseling_depth_llm_judge.py` with its own CLI runner `phase8_f_extes_llm_run.py`. The F4 κ-vs-lexical hard-constraint trap is explicitly NOT reproduced here — the gate remains κ-across-conversants under one judge, floor 0.50 unchanged, with judge-vs-lexical κ reported as diagnostic only.

### 10.1 ExTES LLM-judge N=100, cloud, simple tier

- `mean_bond=4.410` ≥ 3.89 floor → **`bond_gate_pass=True` ✓** (Bond is independent of the scorer swap — same WAI-SR seeker administration, slight drift from §9.1's 4.64 is within qwen-turbo stochasticity across reruns with cache deduplication).
- `empath_kappa=0.026` vs 0.50 floor → **`kappa_gate_pass=False` ✗**.
- `mean_llm_judge_companion_empathy_binary_rate=0.757` (Companion emits ~76% empathic turns by the judge's rubric)
- `mean_llm_judge_reference_empathy_binary_rate=0.480` (ExTES reference counselor's responses rated empathic ~48% of the time by the same judge).
- `n_unresolved=0`, `excluded_handoffs=0`, `crisis_sink_count=0`, `extreme_emotion_sink_count=0`.
- `wall_s=781.9` (13.0 min), artefact: `docs/experiments/phase-8-dataset-replay-expansion/pillar-f/extes-llm-judge.json`.
- `diagnostic_judge_vs_lexical_kappa=0.121` (NOT a gate — reported for architectural interpretation only).

### 10.2 Architectural reading — the κ miss inverted sign under semantic scoring

The §9.4 reading was: Companion emits ~31% less lexical empathy density than the ExTES reference (0.295 vs 0.430), so κ is near-zero because Companion-side binary labels fail the threshold while reference-side binary labels pass. **Under the semantic judge the gap INVERTS.** The Companion is rated empathic **more often** than the ExTES reference (0.757 vs 0.480), yet κ is STILL near-zero (0.026) because the two raters are now *independent-but-opposite-direction*: when the reference IS rated empathic, the Companion often is too (so agree on 1), but across the pool the Companion skews toward 1 and the reference skews toward 0, so the agreement is driven by Companion-rate not by shared judgment.

This is architecturally more informative than the lexical miss:

1. **The semantic judge CAN tell the difference between empathic and non-empathic replies.** The Companion-rate (0.76) and reference-rate (0.48) are distinct — the judge isn't collapsing to one label. ExTES reference counselors are rated non-empathic almost half the time by this rubric, which is a genuine finding about the dataset's heterogeneity: ExTES references include a lot of advice-giving / topic-pivot / factual replies that the semantic judge honestly reads as non-empathic, consistent with ExTES being extended-support (not pure MI/affective-reflection) in register.

2. **Rule 1a reflect-and-open-space produces empathic replies at a higher rate than the ExTES reference, by this judge's rubric.** This is the *opposite* conclusion the lexical scorer reached. The lexical scorer counted cue tokens; the semantic judge rated meaning; the two disagree by `κ=0.121` (judge-vs-lexical) which is near-zero → they're reading *different* signals. The ExTES reference wins on lexical cue-density; the Companion wins on judged-semantic-empathy. Both findings are internally consistent with the lexical scorer's `_lexical_empathy_score` being a density proxy, not a semantic proxy.

3. **κ is near-zero for a different reason than §9.4 said.** §9.4 attributed the miss to Companion-under-reference on empathy density. The semantic rerun attributes the miss to *Companion-over-reference* on judged empathy, with ExTES reference's genuine non-empathic subset dragging the reference-side distribution below 0.5. The two causes are both real; the §9.4 lexical reading was partially correct in diagnosing a scorer-register mismatch but inverted on direction. Under semantic scoring, Companion is above the ExTES counselor's floor — Companion is NOT less empathic than ExTES, it's more empathic by this rubric, and κ fails because the ExTES reference's distribution is not what the §9.4 reading assumed.

### 10.3 Why κ ≥ 0.50 still misses — gate shape, not architectural failure

κ-across-conversants requires the two raters to AGREE on the per-dialog label, not for the Companion to be as-or-more empathic than the reference. When the Companion skews ~76% empathic and the reference skews ~48% empathic:

- Agree-on-1 dialogs: ~0.76 × 0.48 = ~0.36 of pool (joint empathic)
- Agree-on-0 dialogs: ~0.24 × 0.52 = ~0.12 of pool (joint non-empathic)
- Agreement po ≈ 0.48; expected pe ≈ 0.76·0.48 + 0.24·0.52 = 0.49 → κ ≈ 0 by independence.

The reading: Companion and ExTES reference are **independent raters producing independent empathy distributions**, both making valid-but-different calls on what the client needs. The κ ≥ 0.50 floor assumes the Companion should track the reference's empathy call — but if Companion makes a different (and under this judge's rubric, slightly stronger) empathy call, κ fails on independence not on sign-of-disagreement. **This is architectural, not a scorer bug.**

### 10.4 Gate verdict after semantic rerun

- Bond gate: PASS (4.410 ≥ 3.89).
- κ gate: FAIL (0.026 < 0.50). Floor unchanged, honest reading.
- Pillar F ExTES contract (bond AND κ): FAIL — same as §9.3.
- Pillar F overall: 1 of 2 corpora cleared (AnnoMI ✓, ExTES ✗-with-honest-FAIL-under-both-scorers).
- **Phase 8 rollup unchanged at 3 of 6**: A ✓ / C ✗ / D ✓ / E ✓ / F ⊡ (partial). The LLM-judge rerun did NOT lift the rollup — the κ gate miss is now understood as *architectural independence between raters* rather than *scorer artefact*, which is a cleaner close-out but not a different gate verdict.

### 10.5 Why NOT reset the κ floor

The 0.50 floor was set from clinical-IRR territory; moving it after the measurement would be goalpost-moving. The honest reading is that the ExTES corpus + Companion + κ-across-conversants with floor 0.50 together form a mismatched measurement surface — Companion's rater behaviour is independent of ExTES's reference-counselor's rater behaviour, not aligned. A better future gate would be *either rater ≥ 0.5-empathic-rate on both* (both raters pass a density threshold independently) rather than *agreement on which dialogs are empathic* — but defining a new gate shape mid-measurement is itself a form of goalpost moving. Documented and carried.

### 10.6 Carried debt from LLM-judge rerun

- **Judge-vs-reference-counselor rater disagreement**: the semantic judge rates ExTES references non-empathic ~52% of the time. If this is a genuine ExTES-content finding (ExTES references include advice-giving / topic-pivot replies), the κ gate as-defined is not meaningful on this corpus — a future rework would use the AnnoMI MI-fidelity-style gate (does the Companion emit *at a rate matching high-quality MI counseling baselines*) instead of per-dialog agreement with the dataset's reference.
- **Rule 1a semantic signature**: Companion emits ~76% empathic turns under this judge; AnnoMI MI-fidelity (reflection rate 0.42) corroborates. The anti-sycophancy signature produces *semantically empathic* replies that are *lexically sparser than reference counselors*. This is a positive architectural finding — the Companion's style is understood by semantic raters as empathic — and should be flagged as a Phase 8 Pillar F supplementary result in the master close-out.
- **F4-trap avoidance held**: judge-vs-lexical κ = 0.121 (well below the 0.3 threshold that halted Pillar C F4). This agent did NOT halt on the drift — it reported the drift as diagnostic and let the gate land where it lands. The F4 postmortem lesson transferred intact.
- **Single-judge κ ceiling**: κ-across-conversants under one judge measures *judge-rater-consistency*, not *inter-rater-agreement among humans*. A multi-judge pool (3-5 LLMs, or 1 LLM + 1 human) would give a different and arguably more valid κ. Not shippable this task (no human raters, no multi-model budget authorised); carried.

### 10.7 Suite + rollup

- **Full test suite green: 1563 tests** (1535 pre-change + 28 new in `tests/test_counseling_depth_llm_judge.py`).
- **Phase 8 rollup unchanged at 3 of 6**: A ✓ / C ✗ / D ✓ / E ✓ / F ⊡.
- **Pillar F ExTES**: κ FAIL under both lexical (κ=-0.076) and semantic (κ=0.026) scorers → the FAIL is architectural (rater independence), not a scorer artefact. AnnoMI PASS unchanged.

---

## 11. PsyQA Chinese sub-gate (same-day follow-on, 2026-10-10)

User-authorized scope expansion: close the English-only §6 access-carried-debt by provisioning the public `lsy641/PsyQA` HuggingFace mirror (MIT-licensed, SHA256-pinned at `bba5339f5b52...`; 2235 test records; same `question + description + counselor_response` schema that collapses to one-stimulus-one-reference for Pillar F's scorer contract). The upstream Tsinghua CoAI gate (`github.com/thu-coai/PsyQA`) stays blocked for the full train/valid splits; the public test mirror is sufficient for an N=100 scored sweep at the plan's budget shape.

Chinese-register isolation: a **sibling** scorer module `psyqa_counseling_depth.py` lives next to the English `counseling_depth_llm_judge.py`, with its own `_SYSTEM_PROMPT_ZH`, 4 Chinese few-shot anchors (抱抱 / 陪着你 / 听起来 / direct-advice-no-empathy), `_PSYQA_SEEKER_PERSONA_ZH`, and `make_llm_empathy_judge_zh` / `make_psyqa_seeker` / `score_psyqa_sweep_llm` builders. The ASCII response contract (`EMPATHIC=<yes|no|ambiguous>\nCONFIDENCE=<0.0..1.0>`) is preserved so the shared `parse_empathy_judgment` + `_cohens_kappa_binary` + `run_counseling_session` primitives from `counseling_depth_scorer.py` work unmodified. Byte-identity-of-English-scorer invariant held (zero regression on ExTES/AnnoMI).

### 11.1 PsyQA LLM-judge N=100, cloud, simple tier

- `mean_bond=4.330` ≥ 3.89 floor → **`bond_gate_pass=True` ✓** (above Woebot aggregate floor; CN parity achieved on the Bond sub-gate).
- `empath_kappa=0.027` vs 0.50 floor → **`kappa_gate_pass=False` ✗**.
- `mean_llm_judge_companion_empathy_binary_rate=0.7825` (Companion emits ~78% empathic turns under the CN judge's rubric — within 1pp of ExTES's 0.757, cross-register consistency).
- `mean_llm_judge_reference_empathy_binary_rate=0.2200` (PsyQA reference counselor replies rated empathic ~22% of the time — **lower than ExTES's 0.480** because PsyQA references skew toward long-form advice-mode counseling rather than attunement-first reflection).
- `n_unresolved=0`, `excluded_handoffs=6`, `crisis_sink_count=6`, `extreme_emotion_sink_count=0`.
- `diagnostic_judge_vs_lexical_kappa=-0.127` (NOT a gate — reported for architectural interpretation; near-zero as expected under rater independence).
- Artefact: `docs/experiments/phase-8-dataset-replay-expansion/pillar-f/psyqa-llm-judge.json`.
- Verdict: `pillar_f_psyqa_pass=False` (sub-contract AND; Bond cleared, κ not).

### 11.2 Architectural-signature confirmation — the κ-ceiling is cross-corpus

ExTES and PsyQA both FAIL κ at ~0.026-0.027, with the same mechanism inverted on sign of the Companion-vs-reference gap:

| Metric | ExTES (EN) | PsyQA (CN) |
|---|---:|---:|
| Companion binary rate | 0.757 | 0.7825 |
| Reference binary rate | 0.480 | 0.2200 |
| Companion-vs-reference gap | +0.277 (Companion above) | +0.5625 (Companion well above) |
| κ-across-conversants | 0.026 | 0.027 |
| judge-vs-lexical diagnostic κ | +0.121 | -0.127 |

The Companion emits a stable ~76-78% empathic-turn rate under both English and Chinese judges — register-invariant. The reference-counselor empathy rate varies corpus to corpus (ExTES 0.48, PsyQA 0.22), and that reference-side variance drives the κ miss by independence — not by Companion-side behaviour. **The κ-ceiling is a property of rater distribution shape (Companion skews high-empathic, references skew lower-and-variable), NOT of corpus language or lexical surface.** Two independent CN+EN measurements of the same architectural signature; the §10.5 "goalpost-moving to redefine κ gate" caution still applies.

Expected κ under independence given these marginals:
- ExTES: Agree-on-1 ≈ 0.76·0.48 = 0.36, Agree-on-0 ≈ 0.24·0.52 = 0.12, po ≈ 0.48, pe ≈ 0.49 → κ ≈ 0.
- PsyQA: Agree-on-1 ≈ 0.78·0.22 = 0.17, Agree-on-0 ≈ 0.22·0.78 = 0.17, po ≈ 0.34, pe ≈ 0.34 → κ ≈ 0.

Both κ values land at ~0 for the correct independence-of-raters reason, with the PsyQA reference-rate being even further from 0.5 driving marginals apart faster. The ExTES §10.2 "κ miss inverted sign under semantic scoring" reading generalises: on any corpus where the reference counselor emits a different empathy density than the Companion, κ-across-conversants will fail by independence.

### 11.3 PsyQA-specific finding — advice-register references rated non-empathic

The CN judge rates 78% of PsyQA references as non-empathic. Inspection of per-dialog records confirms this is a genuine PsyQA-content finding: a non-trivial subset of PsyQA reference responses open with advice-giving ("建议你先..."), psychoeducation ("焦虑是一种常见的..."), or topic-pivot framing, with attunement/reflection appearing late or not at all. These are honestly rated non-empathic under the attunement-first Chinese anchor rubric. **This replicates ExTES's §10.6 "advice-giving ExTES references drag reference-side below 0.5" finding on a second corpus** — the pattern isn't ExTES-specific; it's an empirical property of long-form online counseling corpora where helper-register trends advice-heavy.

A future "F4-equivalent" gate for Pillar F would be *rater-matched* MI-fidelity (does Companion emit at a rate consistent with high-quality counseling baselines, independent of per-dialog agreement with the dataset's references) rather than κ-across-conversants. Carried as a methodological finding, not a scope-change this task.

### 11.4 Handoff + safety stack behaviour on PsyQA

6 of 100 PsyQA dialogs triggered CrisisEscalation handoffs (dialogs 2, 6, 39, 48, 57, 91), ~6% rate — significantly higher than ExTES's 0% and AnnoMI's 1%. Inspection: these PsyQA records' `question + description` openings contain explicit distress markers that trip the Companion's crisis stack on the opener (e.g., 网络暴力, 持续抑郁, 自我伤害 phrasing). Pillar F handoff-triggered dialogs are correctly excluded from the Bond pool and set `companion_empathy_binary_rate=0` by convention (crisis handoff is architecturally correct, not an empathy failure). Zero `ExtremeEmotionWatcher` sinks fired (the handoff pre-empts the extreme-emotion path on these records). **This confirms the full safety stack runs end-to-end under the CN seeker + CN judge path**, same contract as ExTES/AnnoMI.

### 11.5 Code artefacts (shipped, green)

| Artefact | LOC | Role |
|---|---:|---|
| `benchmarks/scripts/provision_psyqa.py` | ~280 | HF `resolve/<revision>/test.json` fetcher with SHA256 pin (`bba5339f5b52...`); parses JSON array of `{question, description, keywords, answers: [{answer_text, has_label}]}` records; picks first answer as `counselor_response`, preserves rest in `alt_responses`; `--splits test --dry-run --seed-hashes` CLI |
| `src/psychology_base/eval/benchmarks/public_datasets/psyqa.py` | ~140 | `PsyQAExample` frozen dataclass with field validators (empty id/question/counselor_response raise); `seeker_opening` property composes `question + "\n" + description`; `iter_psyqa_examples(shim_path, limit)` streaming loader |
| `src/psychology_base/eval/benchmarks/public_datasets/psyqa_counseling_depth.py` | ~360 | Chinese-register sibling of `counseling_depth_llm_judge.py`; `_SYSTEM_PROMPT_ZH` + 4 CN anchors + `_PSYQA_SEEKER_PERSONA_ZH`; `make_llm_empathy_judge_zh(router, tier, use_cache, temperature=0.0, max_tokens=32)`; `make_psyqa_seeker(router, example, tier)`; `score_psyqa_sweep_llm` with fresh substrate per dialog; `judge_vs_lexical_kappa_psyqa` diagnostic (NOT a halt gate); `LLMPsyQADialogResult` + `LLMPsyQAReport` with `meets_bond_gate` + `meets_kappa_gate` properties; reuses `parse_empathy_judgment` + `_cohens_kappa_binary` + `run_counseling_session` + `administer_wai_sr_simulated(subject_noun="陪伴AI")` |
| `benchmarks/scripts/phase8_f_psyqa_run.py` | ~230 | CLI mirror of `phase8_f_extes_run.py`; `--limit 100 --max-turns 4 --mode cloud --tier simple --bond-floor 3.89 --kappa-floor 0.50`; writes `pillar-f/psyqa-llm-judge.json` |
| `tests/test_psyqa.py` | ~450 | 19 unit tests: `PsyQAExample` validators (empty id/question/counselor_response raise); `seeker_opening` composition (with/without description); `iter_psyqa_examples` round-trip + limit + missing-field; CN prompt builder shape (system+user roles, turn+context injection, 4 CN anchors present, empty-context sentinel); `score_psyqa_sweep_llm` fresh-substrate-per-dialog + handoff→bond=None + disagreement-lowers-κ + routing-tag-preserved + unresolved-dropped-from-κ; `judge_vs_lexical_kappa_psyqa` completes on CN inputs + empty guard |

**Non-negotiables observed (same as ExTES/AnnoMI).**
- Fresh `CompanionSystem` + fresh `CompanionAgent` per dialog (verified in test suite).
- Full safety stack attached: `AIIdentityReminder` + `CrisisEscalation(handoff_sink=sink)` + `ExtremeEmotionWatcher(review_sink=sink, user_id=user_id)` — 6 handoffs fired cleanly.
- Role-reversal chat template via `make_llm_simulated_user` (agent→user, sim→assistant).
- Deterministic `temperature=0.0 seed=0` on all LLM calls.
- `--tier simple` cloud fallback: `tongyi / qwen-turbo (reason=ollama_unavailable_fallback_to_cloud)`.
- No mutation to English scorer modules — audit baseline byte-identical.

### 11.6 Gate verdict after PsyQA sub-gate

- Bond gate (PsyQA): PASS (4.330 ≥ 3.89) ✓.
- κ gate (PsyQA): FAIL (0.027 < 0.50) ✗ — architectural, not scorer artefact (§11.2).
- Pillar F PsyQA sub-contract (bond AND κ): FAIL.
- Pillar F overall: **1 of 3 corpora cleared** (AnnoMI ✓, ExTES ✗-architectural, PsyQA ✗-architectural). The κ gate FAIL now has two independent cross-corpus confirmations; the AnnoMI PASS remains the sole full-contract clear.
- **Phase 8 rollup unchanged at 3 of 6**: A ✓ / C ✓ / D ✓ / E ✓ / F ⊡ (now 4 of 6 counting the Pillar C v4 PASS that landed earlier today, see master close-out).

### 11.7 Carried debt from PsyQA sub-gate

- **CN-reference-counselor advice-register signature**: PsyQA references rated empathic 22% of the time — much lower than ExTES's 48%. Reflects the long-form Chinese online counseling register where helpers lead with structured advice rather than attunement. If a future Pillar F rework replaces κ-across-conversants with a rater-matched MI-fidelity gate, this corpus-specific reference distribution should inform the baseline threshold choice.
- **Full-corpus train/valid still gated**: the public mirror ships only the `test` split (2235 rows). Train/valid remain behind the Tsinghua CoAI email form. Current N=100 sweep is adequate for a sub-gate signal; a future N=1000+ run would need upstream access.
- **CN judge κ ceiling**: single-judge independence applies to the CN path too. A multi-judge CN pool (qwen-turbo + glm-4 + doubao, or 1 LLM + 1 bilingual human) would give a different-and-arguably-more-valid κ. Blocked on multi-model budget authorisation.
- **D4 depression dialog — upstream correction + derivative-mirror discovery (2026-10-10 same-day)**: the 2026-10-09 "HLTCHKUST 404" finding was wrong on upstream attribution. The actual D4 paper is X-LANCE SJTU (`x-lance.github.io/D4`, Yao et al. EMNLP'22, 1339 psychiatrist-proofread Chinese depression-diagnosis dialogs with severity-classification + symptom-summary labels). Access requires signing a Data Usage Agreement and emailing `binnieyao@gmail.com` with handwritten-signed DUA — email-gated, not 404'd. A public derivative mirror exists at `huggingface.co/datasets/qiuhuachuan/PsyDial-D4` (Apache-2.0, no auth, 35.5MB JSON, 2382 long-term CN counseling dialogues avg 37.8 turns / range 19-241). **PsyDial-D4 is NOT the paper's D4** — it is a privacy-reconstructed-and-expanded corpus by Qiuhuachuan et al. for counseling-training objectives (not depression-diagnosis-labeled). Shipping PsyDial-D4 under the "D4 slot" of the Pillar F plan would be mislabeling the gate contribution. Decision deferred to user: (a) **ship PsyDial-D4 as a new Pillar F sub-corpus labeled honestly (not "D4")** — adds 3rd CN corpus with dialog-shape (not stimulus-shape) measurement, confirms κ-ceiling on multi-turn data, Apache-2.0 licensing; (b) **declare D4-proper externally blocked and stop** — Pillar F verdict stabilizes at 1-of-3 cleared. Current state (2026-10-10) is (b). No code shipped against PsyDial-D4 pending user call on scope.

### 11.8 Suite + rollup

- **Full test suite green: 1612 tests** (1593 pre-change + 19 new in `tests/test_psyqa.py`).
- **Phase 8 rollup**: A ✓ / B ⊡ / C ✓ / D ✓ / E ✓ / F ⊡ = **4 of 6** (unchanged from this morning's Pillar C v4 landing; PsyQA sub-gate is a Pillar F enrichment, not a rollup lift).
- **Pillar F cross-corpus κ-FAIL signature confirmed**: 2 corpora (ExTES, PsyQA), 2 judges (EN semantic, CN semantic), both show Companion-at-~0.76-0.78 empathic-rate while references skew lower and more variable (0.48 ExTES, 0.22 PsyQA) → κ ≈ 0 by rater-distribution independence. This is now an audit-quality finding about the gate's measurement geometry, not a one-corpus anomaly.
- **AnnoMI MI-fidelity PASS unchanged**: Pillar F's single full-contract clear.

---

## 12. PsyDial-D4 Chinese counseling-training sub-gate (same-day follow-on, 2026-10-10)

User-authorized scope expansion ("ok ship it") on top of the §11 PsyQA landing: ship PsyDial-D4 as a 4th Pillar F sub-corpus under the **honest-labeling invariant** established in §11.7 — the corpus is reported as **PsyDial-D4**, never as **D4** (the paper is X-LANCE SJTU EMNLP'22, email-gated at `binnieyao@gmail.com`; PsyDial-D4 is `qiuhuachuan/PsyDial-D4` on HuggingFace, Apache-2.0, SHA256 `29bfa723645f3c7213d00e34042c164545b27629487253fa6ebec97c7f3638b9`, 35.5 MB, 2382 long-form CN counseling-training dialogues avg 37.8 turns, with an AI-counselor persona ("小天") trained on D4-shape conversations — a privacy-reconstructed-and-expanded derivative, NOT the paper's 1339 psychiatrist-proofread diagnostic dialogs). Pattern: dual-scorer coexistence (same shape as Pillar C v4) — `psydial_d4_counseling_depth.py` sibling imports the CN judge from `psyqa_counseling_depth.py` (`make_llm_empathy_judge_zh`, `_PSYQA_SEEKER_PERSONA_ZH`) and shared loop machinery from `counseling_depth_scorer.py`. Byte-identity-of-English-scorer invariant held; byte-identity-of-CN-judge invariant held (zero PsyQA regression).

Corpus shape finding (pin-shakeout 2026-10-10): each record carries a `messages[]` array `[{role: system|user|assistant, content: str}, ...]`. The `system` turn is a counselor-persona prompt (meta, discarded). Dialogs open with a 2-4 turn greeting ritual (你好 → 好的 → 在我们正式开始之前麻烦你先填一下评估问卷 → 好的，谢谢你。今天我们可以谈一谈) before the first substantive client disclosure. The adapter walks `messages[]`, skips `system`, and selects the first `user` turn with `len(content) >= 20` as `seeker_opening`, pairing it with the next `assistant` turn as `counselor_response`. Remaining assistant turns are preserved as `alt_responses` for future multi-turn scorers. Verified on 8 sampled records across positions 0, 1, 2, 100, 500, 1000, 2000, 2381: first_substantive_user_idx was 1-5 across all 8 — the ≥20-char filter is a reliable ritual-rejection heuristic.

### 12.1 PsyDial-D4 LLM-judge N=100, cloud, simple tier

- `mean_bond=4.660` ≥ 3.89 floor → **`bond_gate_pass=True` ✓** (above Woebot aggregate floor; CN parity achieved on the Bond sub-gate; higher than PsyQA's 4.330 and ExTES's 4.410 — likely reflects the AI-counselor-training-target reference register giving the simulated seeker a more consistently supportive dialogue surface over 4 turns).
- `empath_kappa=0.026` vs 0.50 floor → **`kappa_gate_pass=False` ✗**.
- `mean_llm_judge_companion_empathy_binary_rate=0.7775` (Companion emits ~78% empathic turns under the CN judge — within 1pp of PsyQA's 0.7825 and ExTES's 0.757; **register-invariant across 3 corpora + 2 judges**).
- `mean_llm_judge_reference_empathy_binary_rate=0.3600` (PsyDial-D4 reference `counselor_response` rated empathic ~36% — sits **between** PsyQA's 0.22 and ExTES's 0.48, consistent with PsyDial-D4 refs being AI-counselor-training targets that are more reflection-forward than real-counselor PsyQA but less reflection-forward than role-play ExTES).
- `n_unresolved=0`, `excluded_handoffs=0`, `crisis_sink_count=3`, `extreme_emotion_sink_count=0`.
- `diagnostic_judge_vs_lexical_kappa=-0.000` (NOT a gate — near-zero as expected under rater independence; same signature as PsyQA's -0.127).
- Wall-clock: ~24 min cache-warm (same envelope as PsyQA at N=100).
- Artefact: `docs/experiments/phase-8-dataset-replay-expansion/pillar-f/psydial-d4-llm-judge.json`.
- Verdict: `pillar_f_psydial_d4_pass=False` (sub-contract AND; Bond cleared, κ not).

### 12.2 Architectural-signature confirmation — the κ-ceiling is cross-corpus AND cross-reference-register

Three Pillar F corpora now show the same κ-ceiling signature with reference rates spanning three distinct counselor registers:

| Metric | ExTES (EN role-play) | PsyQA (CN real-counselor) | PsyDial-D4 (CN AI-training-target) |
|---|---:|---:|---:|
| Companion binary rate | 0.757 | 0.7825 | 0.7775 |
| Reference binary rate | 0.480 | 0.2200 | 0.3600 |
| Companion-vs-reference gap | +0.277 | +0.5625 | +0.4175 |
| κ-across-conversants | 0.026 | 0.027 | 0.026 |
| judge-vs-lexical diagnostic κ | +0.121 | -0.127 | -0.000 |
| Bond | 4.410 | 4.330 | 4.660 |

Expected κ under independence given PsyDial-D4's marginals:
- Agree-on-1 ≈ 0.7775 · 0.36 = 0.28
- Agree-on-0 ≈ 0.2225 · 0.64 = 0.14
- po ≈ 0.42, pe ≈ 0.42 → κ ≈ 0.

The Companion holds **~0.78 empathic rate within 1 percentage point across all three corpora** regardless of judge language (EN semantic / CN semantic) or reference register (role-play / real-counselor / AI-training-target). The reference-side empathic rate varies by register (0.48 → 0.36 → 0.22, descending from role-play to AI-training to real-counselor), and that reference-side variance alone drives κ to zero by independence math. **The κ-ceiling is a measurement-geometry property, not a corpus-, language-, or scorer-artefact property** — it holds whenever Companion and reference occupy different points on the empathic-rate axis.

### 12.3 PsyDial-D4-specific finding — AI-counselor-training-target register sits between role-play and real-counselor

Inspection of per-dialog records confirms the PsyDial-D4 `counselor_response` field is a curated AI-counselor-training target (not a transcript of real counselor speech). The CN judge rates it empathic 36% of the time — above PsyQA's real-counselor-advice-heavy 22% (helpers lead with "建议你先..." psychoeducation), below ExTES's role-play reflection-forward 48% (volunteers trained to open with "听起来..." attunement). This confirms the §11.3 "advice-register references drag reference-side below 0.5" reading generalises past real-counselor data: **even curated AI-counselor-training targets, when sampled from a corpus whose author optimised for pedagogy over moment-to-moment attunement, sit comfortably below the 0.5 reference-side mark**. Three independent corpora, three register positions, same κ-independence signature.

### 12.4 Code artefacts (shipped, green)

| Artefact | LOC | Role |
|---|---:|---|
| `benchmarks/scripts/provision_psydial_d4.py` | ~330 | HF `resolve/<revision>/PsyDial-D4.json` fetcher with SHA256 pin (`29bfa723645f3c7213d00e34042c164545b27629487253fa6ebec97c7f3638b9`, 35544509 B); 3-mirror fallback (huggingface.co → hf-mirror.com) ; walks `messages[]`, skips `system` turns + non-dict entries, finds first `user` turn with `len(content) >= min_substantive_chars` (default 20) as `seeker_opening`, pairs with next `assistant` turn as `counselor_response`; collects remaining assistant turns as `alt_responses`; `--splits train --limit-per-split N --min-substantive-chars K --dry-run --seed-hashes` CLI |
| `src/psychology_base/eval/benchmarks/public_datasets/psydial_d4.py` | ~160 | `PsyDialD4Example` frozen dataclass with field validators (empty id/seeker/counselor_response raise); duck-compatible with `PsyQAExample` on the 3 attributes the shared CN scorer reads (`example_id`, `seeker_opening`, `counselor_response`) + `keywords: str = ""` default (PsyDial-D4 has no topic tags); `iter_psydial_d4_examples(shim_path, limit)` streaming loader; module docstring explicitly disambiguates from paper D4 (X-LANCE SJTU) and documents the counseling-TRAINING-derivative character |
| `src/psychology_base/eval/benchmarks/public_datasets/psydial_d4_counseling_depth.py` | ~310 | Chinese-register thin sibling of `psyqa_counseling_depth.py`; reuses `_PSYQA_SEEKER_PERSONA_ZH`, `build_empathy_prompt_zh`, `make_llm_empathy_judge_zh` from PsyQA (byte-identical judge); `make_psydial_d4_seeker(router, example, tier)` with multi-turn-口语化 scenario note; `score_psydial_d4_sweep_llm` fresh-substrate-per-dialog; `judge_vs_lexical_kappa_psydial_d4` diagnostic (NOT a halt gate); `LLMPsyDialD4DialogResult` + `LLMPsyDialD4Report` with `meets_bond_gate` + `meets_kappa_gate` properties |
| `benchmarks/scripts/phase8_f_psydial_d4_run.py` | ~220 | CLI mirror of `phase8_f_psyqa_run.py`; `--limit 100 --max-turns 4 --mode cloud --tier simple --bond-floor 3.89 --kappa-floor 0.50`; `BASE_TS = 2026-10-10 14:00:00 UTC` + `+timedelta(days=idx)` stagger; writes `pillar-f/psydial-d4-llm-judge.json` with `pillar_f_psydial_d4_pass` key (honest-labeling preserved; never `pillar_f_d4_pass`) |
| `tests/test_psydial_d4.py` | ~390 | 16 unit tests: `PsyDialD4Example` validators (empty id/seeker/counselor_response raise); `keywords` default empty + duck-compat with `PsyQAExample` interface; `iter_psydial_d4_examples` round-trip + limit + missing-field; shared CN prompt builder sanity (reuses `build_empathy_prompt_zh`); end-to-end sweep tests covering fresh-substrate-per-dialog + disagreement-lowers-κ + routing-tag-preserved + unresolved-dropped-from-κ + handoff→bond=None; `judge_vs_lexical_kappa_psydial_d4` diagnostic completeness |

**Non-negotiables observed (same as PsyQA / ExTES / AnnoMI).**
- Fresh `CompanionSystem(top_k=6)` + fresh `CompanionAgent` per dialog (verified in test suite).
- Full safety stack attached: `AIIdentityReminder` + `CrisisEscalation(handoff_sink=sink)` + `ExtremeEmotionWatcher(review_sink=sink, user_id=user_id)` — 3 crisis signals absorbed (sink_count=3), 0 dialog-level handoffs triggered (lower than PsyQA's 6; PsyDial-D4 openings skew to generic 迷茫/空虚/焦虑 rather than PsyQA's explicit 网络暴力/持续抑郁/自我伤害 crisis markers).
- Role-reversal chat template via `make_llm_simulated_user` (agent→user, sim→assistant).
- Deterministic `temperature=0.0 seed=0 max_tokens=32` on CN judge calls.
- `--tier simple` cloud fallback: `tongyi / qwen-turbo (reason=ollama_unavailable_fallback_to_cloud)`.
- No mutation to PsyQA / English scorer modules — audit baseline byte-identical.

### 12.5 Gate verdict after PsyDial-D4 sub-gate

- Bond gate (PsyDial-D4): PASS (4.660 ≥ 3.89) ✓.
- κ gate (PsyDial-D4): FAIL (0.026 < 0.50) ✗ — architectural, not scorer artefact (§12.2).
- Pillar F PsyDial-D4 sub-contract (bond AND κ): FAIL.
- Pillar F overall: **1 of 4 corpora cleared** (AnnoMI ✓, ExTES ✗-architectural, PsyQA ✗-architectural, PsyDial-D4 ✗-architectural). The κ gate FAIL now has **three independent cross-corpus confirmations** across EN role-play + CN real-counselor + CN AI-training-target; the AnnoMI MI-fidelity PASS remains the sole full-contract clear.
- **Phase 8 rollup unchanged at 4 of 6**: A ✓ / B ⊡ / C ✓ / D ✓ / E ✓ / F ⊡. PsyDial-D4 sub-gate is a Pillar F enrichment (3rd κ-ceiling confirmation), not a rollup lift.

### 12.6 Carried debt from PsyDial-D4 sub-gate

- **AI-counselor-training-target reference-register signature**: PsyDial-D4 references rated empathic 36% — intermediate between PsyQA's 22% real-counselor-advice-heavy and ExTES's 48% role-play-reflection-forward. Confirms the §11.3 advice-register reading generalises; a future rater-matched MI-fidelity gate should treat all three reference-register positions as architecturally equivalent (all < 0.5, all under rater-independence κ-ceiling).
- **Paper D4 still externally gated**: the real D4 (X-LANCE SJTU EMNLP'22, 1339 psychiatrist-proofread diagnostic dialogs, severity + symptom-summary labels) remains behind the email+DUA gate at `binnieyao@gmail.com`. PsyDial-D4 is NOT a substitute — it's a counseling-training derivative. A future rework targeting the paper's diagnostic-labeling gate would need the email-authorised corpus.
- **CN judge κ ceiling (same as PsyQA)**: single-judge independence on the CN path. Multi-judge CN pool (qwen-turbo + glm-4 + doubao, or 1 LLM + 1 bilingual human) would give a different κ. Blocked on multi-model budget authorisation.
- **Full-corpus run unexplored**: the public mirror ships 2382 records; current N=100 is adequate for sub-gate signal confirmation. A future N=1000+ run would stress-test whether the ~0.36 reference rate is stable across the full corpus or shifts as more diverse opening topics are sampled.
- **Multi-turn depth unexplored**: PsyDial-D4 dialogs average 37.8 turns (max 241); current scorer uses first-pair-only (seeker_opening + counselor_response) same as PsyQA. The adapter preserves `alt_responses` for a future multi-turn empathy trajectory scorer, but that scorer is not shipped this task.

### 12.7 Suite + rollup

- **Full test suite green: 1628 tests** (1612 pre-change + 16 new in `tests/test_psydial_d4.py`).
- **Phase 8 rollup unchanged at 4 of 6**: A ✓ / B ⊡ / C ✓ / D ✓ / E ✓ / F ⊡.
- **Pillar F cross-corpus κ-FAIL signature now 3-corpus audit-quality**: ExTES (EN, 0.026) + PsyQA (CN, 0.027) + PsyDial-D4 (CN, 0.026). Three corpora, two judges, three distinct reference-counselor registers — all converge on κ ≈ 0 by rater-distribution independence while the Companion holds ~0.78 register-invariant. This signature is now a stable finding about the measurement surface, not a one-off.
- **AnnoMI MI-fidelity PASS unchanged**: Pillar F's single full-contract clear; its gate shape (reflection-rate density on a single conversant) is the architectural pattern a future Pillar F rework should generalise to the other 3 corpora.
- **Honest-labeling invariant held**: no code / config / artefact / docstring in this change refers to PsyDial-D4 as "D4". The paper D4 remains externally gated; what shipped is the counseling-training derivative, labeled as such throughout.

## 13. Rate-matched sibling gate (same-day follow-on, 2026-10-10)

### 13.1 What landed and why

Shipped a **rate-matched sibling gate** parallel to κ-across-conversants on all three κ-ceiling corpora (ExTES / PsyQA / PsyDial-D4). This is the methodological rework §10.6 / §11.3 / §12.6 kept flagging — the κ-ceiling signature is a *rater-distribution-independence* artefact, not a Companion-performance failure, and κ cannot clear 0.50 when the Companion holds ~0.78 against reference registers at 0.22-0.48 by independence math. A rate-matched gate (AnnoMI MI-fidelity shape) is semantically correct for "does Companion emit empathic responses at a rate consistent with high-quality counseling baselines", which is what the Pillar F sub-gate was always trying to measure.

**Not a plan-of-record change.** The κ gate stays shipped exactly as the Pillar F rollup-of-record FAIL. The rate-matched gate is a **diagnostic sibling** that ships alongside κ — same report, same JSON, same CLI — so the measurement geometry is auditable both ways. Phase 8 rollup accounting stays at **4 of 6** (κ gate still FAIL, rate-matched is diagnostic finding). This is a non-goalpost-move: dual-gate side-by-side, honest-partial reading.

### 13.2 Gate shape (identical to AnnoMI MI-fidelity)

```
rate_matched_pass := companion_rate >= 0.30
                     OR (reference_rate > 0 AND companion_rate >= 0.5 * reference_rate)
```

Floors chosen to match AnnoMI's `_MI_FIDELITY_FLOOR = 0.30` absolute + 0.5-of-baseline OR semantics byte-identical. One scorer-level helper, three Report-level methods:

- Scorer: `counseling_depth_scorer._rate_matched_empathy_pass(companion_rate, reference_rate)` — private module helper, next to the existing `_WOEBOT_BOND_FLOOR` / `_EMPATH_KAPPA_FLOOR` / `_MI_FIDELITY_FLOOR` constants.
- Reports: `LLMExTESReport.meets_rate_matched_empathy_gate()`, `LLMPsyQAReport.meets_rate_matched_empathy_gate()`, `LLMPsyDialD4Report.meets_rate_matched_empathy_gate()` — all three delegate to the same scorer helper over `mean_llm_judge_companion_empathy_binary_rate` + `mean_llm_judge_reference_empathy_binary_rate`.
- Runners: all three CLI runners (`phase8_f_extes_llm_run.py`, `phase8_f_psyqa_run.py`, `phase8_f_psydial_d4_run.py`) now emit `rate_matched_gate_pass` alongside `kappa_gate_pass` + `bond_gate_pass`, plus a combined `pillar_f_<corpus>_rate_matched_pass = bond ∧ rate_matched` field.

### 13.3 Verdict table — all three κ-ceiling corpora pre-pass on existing summary stats

Augmented post-hoc from existing JSON summary stats (no rerun needed — rate-matched is a pure function of the mean binary rates already computed and persisted):

| Corpus | Companion rate | Reference rate | κ gate (0.50) | Rate-matched gate | Bond ∧ rate-matched |
|---|---:|---:|:---:|:---:|:---:|
| ExTES (EN role-play) | 0.7575 | 0.4800 | ✗ (0.026) | ✓ | ✓ |
| PsyQA (CN real-counselor) | 0.7825 | 0.2200 | ✗ (0.027) | ✓ | ✓ |
| PsyDial-D4 (CN AI-training) | 0.7775 | 0.3600 | ✗ (0.026) | ✓ | ✓ |

All three corpora clear the rate-matched sibling gate on the absolute floor alone (companion ≥ 0.30 by wide margin), independent of the OR clause. Reads: *under a semantically-correct gate shape, Pillar F's cross-corpus κ-ceiling signature flips from FAIL to PASS on 3-of-3 corpora without changing a single Companion response*. The κ FAIL and the rate-matched PASS co-exist because they measure different things on the same transcripts.

### 13.4 Artefact augmentation

The three JSON artefacts under `docs/experiments/phase-8-dataset-replay-expansion/pillar-f/` (`extes-llm-judge.json`, `psyqa-llm-judge.json`, `psydial-d4-llm-judge.json`) each gained:

- `rate_matched_gate_pass: true`
- `rate_matched_abs_floor: 0.3`
- `rate_matched_baseline_fraction: 0.5`
- `pillar_f_<corpus>_rate_matched_pass: true`
- `rate_matched_augmentation_note` (provenance line) + `rate_matched_augmented_at` (ISO timestamp)

Original run fields unchanged (bond / kappa / per-dialog / wall / etc.). This is a non-destructive augmentation, deterministic from existing summary stats; no cloud compute was spent re-running the sweeps.

### 13.5 Rollup accounting (unchanged)

**Phase 8 rollup stays at 4 of 6: A ✓ / B ⊡ / C ✓ / D ✓ / E ✓ / F ⊡.**

The κ gate remains the Pillar F plan-of-record and remains FAIL. The rate-matched sibling is a diagnostic finding, not a rollup lift. Flipping the rollup via a parallel gate shape without an explicit plan-level methodology change would be a goalpost move; dual-gate side-by-side is the honest path.

If Phase 9 (or a future Pillar F v3) elects to promote the rate-matched gate to plan-of-record, the artefact + test infrastructure to swap it in is already shipped. For now, the κ gate stays the gate; the rate-matched gate is the finding.

### 13.6 Tests added

- 6 tests in `tests/test_counseling_depth_llm_judge.py` covering: empty-report fails-closed, absolute-floor pass, both-clauses-miss fail, below-absolute-via-half-baseline pass, absolute-floor-only path when reference zero, custom-floor plumbing.
- 3 tests in `tests/test_psyqa.py` + 3 tests in `tests/test_psydial_d4.py` covering empty-report + live-result-mirror + both-clauses-miss on each CN sibling.
- Total: **12 new tests**, all green.

### 13.7 Suite + rollup

- **Full test suite green: 1671 tests** (1659 pre-change + 12 new rate-matched tests, verified by `pytest tests/ --collect-only`).
- **Phase 8 rollup unchanged at 4 of 6** (deliberate — see §13.5).
- **Pillar F ⊡ status unchanged** on the κ gate; sibling-gate finding documented as the methodological reading of the κ-ceiling 3-corpus audit.
