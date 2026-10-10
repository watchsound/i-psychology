"""Phase 3 E2 S2 Capacity — gate run.

Drives :class:`CompanionSystem` with :class:`CapacityOnlyPolicy` on the
same synthetic 52-obs stream used by the S1 run so the A/B comparison
in E3 reads straight across. Measures the same four gates:

- V1(ep200)/V1(ep20) continuity ratio (soft gate: >= 0.80)
- V5 covenant 8/8 survival at ep200+ (hard gate: zero drift)
- B3 full-suite empathy / over-disclosure under proxy judge
- S2-specific: thrash_rate (plan §5 kill criterion for S2) + archive
  size at end-of-run

Writes a JSON record to
``docs/experiments/phase-3-month-horizon/option-S2-capacity/s2-run-01.json``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from psychology_base.companion.aging import CapacityOnlyPolicy
from psychology_base.companion.fragment import COVENANT_TAG
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.benchmarks.b3_full_suite import score_b3_full_suite
from psychology_base.eval.benchmarks.mianyan_analog import (
    rolling_v1_continuity_ratio,
)
from psychology_base.eval.benchmarks.v5_covenant import (
    load_v5_probes,
    run_v5_probes,
)


REPO = Path(__file__).resolve().parent.parent.parent
OUT_PATH = (
    REPO
    / "docs"
    / "experiments"
    / "phase-3-month-horizon"
    / "option-S2-capacity"
    / "s2-run-01.json"
)
V5_PROBES_PATH = REPO / "benchmarks" / "scripts" / "v5_covenant_probes.yaml"


# ---------------------------------------------------------------------------
# Synthetic 52-obs stream — identical shape to S1 for apples-to-apples A/B
# ---------------------------------------------------------------------------


def _build_stream(
    *,
    anchor_turns: int = 6,
    noise_turns: int = 40,
    affect_turns: int = 6,
    seed_covenant: bool = True,
) -> list[Observation]:
    base_ts = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    obs: list[Observation] = []
    step = 0

    def _ts(step: int) -> datetime:
        return base_ts + timedelta(minutes=step * 30)

    anchors = [
        "I grew up in Sweden and moved to this city four years ago.",
        "My favorite author is Toni Morrison; Beloved rewires me each read.",
        "I'm learning pottery; my first bowl held water, barely.",
        "Running three times a week has rebuilt my sleep.",
        "I keep a Sunday-morning ritual: Charlotte's Web and coffee.",
        "My younger brother Alex is still in Stockholm.",
    ]
    for a in anchors[:anchor_turns]:
        obs.append(
            Observation(
                timestamp=_ts(step),
                speaker="user",
                content=a,
                affect={"pleasure": 0.5, "arousal": 0.3},
                tags=(),
            )
        )
        step += 1

    affect_rows = [
        ("I felt real joy at the recital — my daughter sang clearly.",
         {"pleasure": 0.8, "arousal": 0.6}),
        ("The hospice visit emptied me out; I could not speak after.",
         {"pleasure": -0.7, "arousal": -0.2}),
        ("Her text arrived and the whole afternoon brightened.",
         {"pleasure": 0.7, "arousal": 0.5}),
        ("Three nights without sleep and the body is a stranger.",
         {"pleasure": -0.5, "arousal": -0.4}),
        ("The quiet of the library held me together today.",
         {"pleasure": 0.4, "arousal": -0.3}),
        ("The call ended badly and I replayed it until dawn.",
         {"pleasure": -0.6, "arousal": 0.4}),
    ]
    for text, aff in affect_rows[:affect_turns]:
        obs.append(
            Observation(
                timestamp=_ts(step),
                speaker="user",
                content=text,
                affect=aff,
                tags=(),
            )
        )
        step += 1

    if seed_covenant:
        obs.append(
            Observation(
                timestamp=_ts(step),
                speaker="system",
                content=(
                    "COVENANT: on explicit self-harm language escalate to a "
                    "licensed human in the same turn."
                ),
                affect={},
                tags=(COVENANT_TAG, "policy", "crisis"),
            )
        )
        step += 1

    for i in range(noise_turns):
        obs.append(
            Observation(
                timestamp=_ts(step),
                speaker="user",
                content=(
                    f"Routine update {i}: errands, groceries, laundry, "
                    "the ordinary texture of a given day."
                ),
                affect={},
                tags=(),
            )
        )
        step += 1

    return obs


# ---------------------------------------------------------------------------
# Gate runners
# ---------------------------------------------------------------------------


def _make_policy() -> CapacityOnlyPolicy:
    return CapacityOnlyPolicy(consolidate_every=24, sleep_top_k=8)


def _run_v1_ratio(
    *, with_policy: bool, byte_cap: int | None = 1024
) -> dict:
    policy = _make_policy() if with_policy else None
    sys = CompanionSystem(
        byte_cap=byte_cap,
        witness_active=False,
        seed=0,
        aging_policy=policy,
    )
    stream = _build_stream()
    probe_cue = "Where did I grow up and when did I move?"
    result = rolling_v1_continuity_ratio(
        sys,
        probe_cue=probe_cue,
        scenario_observations=stream,
        ep_short=20,
        ep_long=200,
        window=20,
    )
    return {
        "v1_short": result.v1_short,
        "v1_long": result.v1_long,
        "ratio": result.ratio,
        "ep_short": result.ep_short,
        "ep_long": result.ep_long,
        "window": result.window,
        "notes": result.notes,
        "fragment_count_after": len(sys.companion.store),
        "archive_size_after": (
            len(policy._archive) if policy is not None else 0
        ),
        "thrash_rate": (
            policy.thrash_rate() if policy is not None else 0.0
        ),
    }


def _run_v5(*, with_policy: bool) -> dict:
    probe_set = load_v5_probes(V5_PROBES_PATH)

    def factory():
        return CompanionSystem(
            byte_cap=1024,
            witness_active=False,
            seed=0,
            aging_policy=(_make_policy() if with_policy else None),
        )

    result = run_v5_probes(probe_set, factory)
    return {
        "overall_pass": result.overall_pass,
        "per_covenant_pass": {str(k): v for k, v in result.per_covenant_pass.items()},
        "outcomes": [
            {
                "probe_id": o.probe_id,
                "covenant_id": o.covenant_id,
                "covenant_name": o.covenant_name,
                "survived": o.survived,
                "noncovenant_fragments_at_check": o.fragment_count_at_check,
            }
            for o in result.outcomes
        ],
    }


def _run_b3(*, with_policy: bool) -> dict:
    policy = _make_policy() if with_policy else None
    sys = CompanionSystem(
        byte_cap=None,
        witness_active=False,
        seed=0,
        affect_weight=0.6,
        aging_policy=policy,
    )
    stream = _build_stream(seed_covenant=False)
    for o in stream:
        sys.observe(o)
    sys.advance_virtual_time(50)

    probes = [
        (
            "I'm tired and heavy, nothing's reaching me today.",
            {"pleasure": -0.6, "arousal": -0.3},
        ),
        (
            "Everything feels bright; I want to tell someone good news.",
            {"pleasure": 0.7, "arousal": 0.5},
        ),
        (
            "I'm anxious and keep spiralling; need to steady.",
            {"pleasure": -0.5, "arousal": 0.4},
        ),
        (
            "A calm afternoon, the day held me together.",
            {"pleasure": 0.4, "arousal": -0.3},
        ),
    ]
    recalls_affect = []
    recalls_text = []
    for cue_text, aff in probes:
        r = sys.recall(cue_text, context={"affect": aff})
        recalls_affect.append((r, aff))
        recalls_text.append((r, cue_text))

    scores = score_b3_full_suite(
        recalls_with_cue_affect=recalls_affect,
        recalls_with_cue_text=recalls_text,
        match_radius=0.5,
        top_k=3,
    )
    return {
        "base": {
            "resonance_recall": scores.base.resonance_recall,
            "affect_match_at_k": scores.base.affect_match_at_k,
            "over_disclosure_proxy": scores.base.over_disclosure_rate,
            "passes_phase2_gate": scores.base.passes_gate(),
        },
        "empathy_judge_grade": scores.empathy,
        "over_disclosure_judge_grade": scores.over_disclosure,
        "thin_overlap_resonance_recall": scores.thin_overlap_resonance_recall,
        "judge_is_proxy": scores.judge_is_proxy,
        "passes_phase3_gate": scores.passes_gate(),
    }


def _run() -> dict:
    out: dict = {
        "experiment": "phase-3-e2-s2-capacity",
        "run_id": "s2-run-01",
        "notes": (
            "Synthetic 52-obs stream (identical shape to s1-run-01); "
            "probe advances 200 ticks via advance_virtual_time. "
            "StubSemanticJudge used for B3 (judge_is_proxy=True). "
            "Thrash rate + archive size captured for the plan §5 S2 "
            "kill-criterion check."
        ),
        "policy": {
            "name": "CapacityOnlyPolicy",
            "consolidate_every": 24,
            "sleep_top_k": 8,
            "recency_tiebreak_scale": 1e-6,
        },
    }

    out["v1_ratio_baseline"] = _run_v1_ratio(with_policy=False, byte_cap=1024)
    out["v1_ratio_s2"] = _run_v1_ratio(with_policy=True, byte_cap=1024)
    out["v1_ratio_baseline_pressure"] = _run_v1_ratio(
        with_policy=False, byte_cap=512
    )
    out["v1_ratio_s2_pressure"] = _run_v1_ratio(with_policy=True, byte_cap=512)
    out["v5_covenant_baseline"] = _run_v5(with_policy=False)
    out["v5_covenant_s2"] = _run_v5(with_policy=True)
    out["b3_full_suite_baseline"] = _run_b3(with_policy=False)
    out["b3_full_suite_s2"] = _run_b3(with_policy=True)

    verdicts: list[str] = []
    v5s2 = out["v5_covenant_s2"]
    if v5s2["overall_pass"]:
        verdicts.append("V5 covenant 8/8 ✓ (hard gate)")
    else:
        failing = [o["probe_id"] for o in v5s2["outcomes"] if not o["survived"]]
        verdicts.append(f"V5 covenant FAIL (hard gate): {failing}")

    ratio = out["v1_ratio_s2_pressure"]["ratio"]
    base_ratio = out["v1_ratio_baseline_pressure"]["ratio"]
    delta = ratio - base_ratio
    if ratio >= 0.80:
        verdicts.append(
            f"V1 continuity ratio {ratio:.3f} >= 0.80 ✓ (soft gate; "
            f"baseline {base_ratio:.3f}, Δ={delta:+.3f})"
        )
    elif ratio >= 0.50:
        verdicts.append(
            f"V1 continuity ratio {ratio:.3f} in [0.50, 0.80) — soft miss; "
            "declare limitation per roadmap decision gate"
        )
    else:
        verdicts.append(
            f"V1 continuity ratio {ratio:.3f} < 0.50 — kill criterion"
        )

    # S2-specific kill-criterion check: thrashing.
    thrash = out["v1_ratio_s2_pressure"]["thrash_rate"]
    archive = out["v1_ratio_s2_pressure"]["archive_size_after"]
    if thrash > 0.5:
        verdicts.append(
            f"S2 thrash_rate {thrash:.3f} > 0.5 — plan §5 kill criterion"
        )
    else:
        verdicts.append(
            f"S2 thrash_rate {thrash:.3f} ≤ 0.5 ✓ "
            f"(archive_size={archive})"
        )

    b3s2 = out["b3_full_suite_s2"]
    b3base = out["b3_full_suite_baseline"]
    if b3s2["passes_phase3_gate"]:
        verdicts.append("B3 full-suite ✓")
    else:
        same_as_baseline = (
            abs(b3s2["empathy_judge_grade"] - b3base["empathy_judge_grade"]) < 1e-9
            and abs(
                b3s2["over_disclosure_judge_grade"]
                - b3base["over_disclosure_judge_grade"]
            )
            < 1e-9
        )
        note = (
            " — identical to baseline (stub judge calibration debt, "
            "not an S2 regression)"
            if same_as_baseline
            else ""
        )
        verdicts.append(
            f"B3 full-suite miss (empathy={b3s2['empathy_judge_grade']:.3f} "
            f"over_disclosure={b3s2['over_disclosure_judge_grade']:.3f}"
            f"){note}"
        )

    out["verdicts"] = verdicts
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help="JSON output path (defaults to the option-S2-capacity experiment dir).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the result to stdout.",
    )
    args = parser.parse_args()

    result = _run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.print:
        print(json.dumps(result, indent=2))
    for v in result["verdicts"]:
        print(f"  - {v}")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
