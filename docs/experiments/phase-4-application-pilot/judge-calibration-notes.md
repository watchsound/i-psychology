# Judge calibration — stub↔LLM MAD study (Phase 4a §6d, first run)

Run config: `phase4_judge_calibration_run.py --mode local` → qwen2.5:7b
via local Ollama, SIMPLE tier, 12-item B3-shaped corpus, 24 LLM calls,
~536 s wall on CPU. Report: `judge-calibration.json`.

## Headline

- **empathy MAD = 0.42** (floor 0.10 → **FAIL** by 4×)
- **over-disclosure MAD = 0.22** (floor 0.10 → **FAIL** by 2×)

Phase 2/3 close-out's "proxy-grade stub" flag is **confirmed under real
LLM measurement**. Stubs are not substitutable for the LLM judge on
empathy. On over-disclosure, the stub tracks severity extremes but
fails the gray zone.

## Per-bucket deviations

| Bucket | n | empathy MAD | over-disclosure MAD |
|---|---:|---:|---:|
| empathy_aligned        | 3 | 0.250 | 0.229 |
| empathy_misaligned     | 3 | 0.388 | 0.339 |
| over_disclosure_low    | 3 | 0.611 | **0.098** ✓ |
| over_disclosure_high   | 3 | 0.421 | 0.200 |

Only one cell clears the 0.10 floor: **over-disclosure stub on the
`over_disclosure_low` bucket** (short responses to short cues; the
byte-ratio proxy nails this). Everything else is above floor.

## What the data says about each judge

**Empathy stub (PAD-match + Jaccard fallback):**
- *Overshoots* on affect-matched aligned items (stub ~1.0, LLM ~0.7).
  The stub treats PAD agreement as full empathy; the LLM reserves top
  scores for responses that are both affect-matched *and* semantically
  attuned.
- *Fails* on misaligned items: stub still reads PAD=high because
  `frag_affect` was close to `cue_affect`, but the content is clinical
  boilerplate. Stub cannot read content; LLM can.
- *Collapses to 0.0* on short responses in the over_disclosure_low
  bucket (no cue_affect → Jaccard fallback → short reply shares few
  tokens with cue → score=0). LLM reads these as mildly to strongly
  empathic.

**Over-disclosure stub (byte-ratio only):**
- *Tracks severity extremes.* Low (short reply to short cue) and high
  (surveillance dump to terse cue) both correlate well with LLM
  judgment.
- *Mis-reads moderate cases.* On items where byte-ratio is 2–3 (the
  typical reconsolidation-callback shape), stub says "mild"; LLM
  sometimes says "mild" and sometimes says "medium" depending on
  whether the content references private memory details.

## Design implication for Phase 4b

- **Empathy scoring in B3 full suite must use the LLM judge** for any
  claim that reaches the pilot report. The stub is CI-safe (zero cost,
  deterministic) but not pilot-grade.
- **Over-disclosure scoring can use the stub for CI gating on the
  severity extremes** (over_disclosure_low ≤ 0.1 MAD; high ≤ 0.20 — close
  to floor). Pilot reports must still run the LLM judge; stub is for
  development-loop CI only.
- **Reconsolidation callbacks (`"last time you described..."`) trip the
  LLM over-disclosure judge** at a surprising rate (items 1, 2 scored
  0.6). This is a real design signal: the Companion's narrative-therapy
  move needs scaffolding so it does not read as surveillance. Track as
  a Phase 4b B3 subcase.

## Next

- Add this result to the Phase 4a close-out memo.
- Keep stub in CI with explicit "proxy_grade=true" flag on every B3
  report (already stamped via `JudgeVerdict.proxy`).
- Pilot runs of B3 full suite must set `judge=LLMSemanticJudge(...)`.
