# v1 harness scope

**Date:** 2026-10-05
**Phase:** Phase 0 per [roadmap.md](roadmap.md)
**Purpose:** make goals measurable before building anything that will be measured.

---

## What the harness is for

The three gap-closing benchmarks (B1 / B2 / B3) and the five vitality metrics (V1–V5) defined in [benchmarks.md](benchmarks.md) all require two things that do not yet exist:

1. **A uniform interface** against which any candidate memory architecture — the EchoMemory baseline, a future Phase-1 consciousness-primitive system, a Phase-2 full mentalistic-process system — can be run and scored.
2. **A format for scripted longitudinal sessions** that compresses weeks or months of simulated interaction into runnable test data, and a player that replays those sessions through a memory system under test.

The harness is nothing more than those two things in v1. It is not a complete benchmark implementation; it is the structural minimum that makes benchmark implementation possible without re-plumbing.

## What ships in v1

- `MemorySystem` Protocol — the interface. Minimal surface: `observe(event)`, `recall(cue, context)`, `tick()`, optional `inspect()` for fragment-level introspection.
- `Observation`, `Recall`, `Fragment` dataclasses — the typed values passed through the interface.
- Pydantic schema for scripted-session YAML — one file per scripted session, versioned.
- `SessionPlayer` — reads a YAML, iterates turns, drives a `MemorySystem`, collects outputs.
- `EchoMemory` baseline — a nothing-conscious-simulated key/value store with timestamps. **Point:** sets the floor against which V-track thresholds (currently TBD in [sota-goal.md](sota-goal.md)) are calibrated.
- One sample B1 YAML — a short childhood-event reinterpretation sequence, enough to prove the player works end-to-end.
- One pytest that plays the sample YAML through EchoMemory and asserts the pipeline produces metric-shaped output.

## What does NOT ship in v1

- Any consciousness-simulation primitive (heartbeat, reconstruction, selection pressure, Witness).
- Full B1/B2/B3 metric implementations. v1 ships the pipeline that will later produce them; the metric functions themselves land in Phase 1 alongside the first non-echo architecture.
- Standard-benchmark adapters (LongMemEval / LoCoMo / EpBench). Those land in Phase 1 as the regression-guard surface.
- LLM calls of any kind. v1 runs offline, deterministic, no network, no model keys required.

## Exit criteria

Phase 0 is done when:

- `pip install -e .[dev]` succeeds.
- `pytest` passes end-to-end: the sample B1 YAML parses, plays through EchoMemory, and the player's transcript + `inspect()` output expose the information B1 metrics will later read.
- EchoMemory `inspect()` returns a fragment list; the baseline establishes that fragment-count-shaped metrics (B1 fragment preservation) have a measurable floor.

Metric *implementations* (B1 preservation / rendering divergence / within-session continuity scorers) and a CLI runner for picking between baselines land in Phase 1 alongside the first non-echo architecture — v1 ships the pipeline, not the metric suite.

## Design discipline for the harness itself

The harness is tooling, not architecture. It must not accidentally bake in the store-shaped assumption the architecture rejects.

- The `MemorySystem` interface takes `observe` and `recall` as separate verbs, not `set`/`get`. This leaves room for a system where "recall" is a reconstruction, not a lookup.
- `tick()` is first-class even though EchoMemory ignores it. Consciousness-sim systems need wake/sleep separation; the harness must not force them to pretend they are stateless-between-queries.
- `inspect()` is optional. Fragment-level metrics (B1 fragment preservation) call it when available; black-box systems can return `None` and skip those metrics rather than lie.
- The scripted-session format is time-aware: each turn has a timestamp. A system that uses time (reconstruction, decay, drift) sees it; a system that ignores time still runs.
