"""Phase 9+ attachment-overlay ``_W_*`` weight calibration — data generation + grid search.

The behavior-trace bond composition in
:mod:`psychology_base.applications.emotional.attachment_state` applies six
signed per-dimension weights (``_W_MIRROR``, ``_W_REPAIR``, ``_W_CONSISTENCY``,
``_W_RUPTURE``, ``_W_SYCOPHANCY``, ``_W_OVER_DISCLOSURE``) to normalised
per-turn-rates and shifts a Likert midpoint (``LIKERT_MID = 4.0``). The
shipped magnitudes were chosen so that saturated-positive input lands at
``LIKERT_MAX = 7.0`` and saturated-negative lands near ``LIKERT_MIN = 1.0``.
That design rationale is a saturation-contract, not a calibration pass.

This script closes the "data-generation step first" prerequisite for
calibration under the 2026-10-08 no-recruitment lock:

1. **Hand-authored synthetic calibration corpus** (:data:`CALIBRATION_CORPUS`)
   — 20 session fixtures covering pure-dimension, mixed, and neutral
   behavior profiles. Each fixture carries an *author-intuition* target
   bond anchored in clinical / attachment-theory reasoning (documented
   inline per fixture). This is NOT ground-truth attachment labelling in
   the clinical annotation sense; it is an auditable, explicit encoding
   of what each behavior profile *should* produce on the 1–7 Likert if
   the lexical weights were honest.
2. **Sign-locked grid search** (:func:`grid_search_weights`) over the
   magnitudes of each ``_W_*``. Signs are fixed by dimension identity
   (mirror/repair/consistency positive, rupture/sycophancy/over-disclosure
   negative) because flipping a sign changes the dimension's meaning, not
   its weight.
3. **Report** — current-weight MAE vs. grid-optimum MAE on the corpus,
   per-fixture breakdown, recommended weight delta. Written to a JSON
   artifact; NOT applied to :mod:`attachment_state` here — the user
   reviews the report and decides whether to adopt the recommendation.

**Why hand-authored + author-intuition rather than LLM-trace fit.** The
shipped Pillar A × attachment-overlay N=3 run produces an LLM-reported
WAI-SR trajectory that is persona-flat (σ=0 across all users) and a
behavior-trace that fired only mirror + consistency across all 12
sessions. Zero variance → undefined Pearson, zero gradient signal on
four of six weights. See
``docs/experiments/phase-9-perspective-substrate-consolidation/attachment-overlay.json``
for the raw data. A real clinical-annotation calibration pass remains
future work, blocked on the no-recruitment lock.

Usage::

    python benchmarks/scripts/phase9_attachment_weight_calibration.py \\
        --grid-size coarse --out docs/experiments/phase-9-attachment-overlay/weight-calibration.json

    # Fine grid (slower, denser magnitudes):
    python benchmarks/scripts/phase9_attachment_weight_calibration.py \\
        --grid-size fine

Honest limitations:

- Author-intuition targets encode the author's clinical priors. The
  recommended weights are the weights that make the current algorithm
  agree with those priors on this corpus; they are not externally
  validated.
- The corpus has N=20 fixtures. Overfitting to this small set is a real
  risk; the recommendation should be interpreted as "the direction to
  move the weights" more than "the final weights".
- Sign-locked grid preserves dimension identity but cannot discover a
  radically different composition (e.g. that over-disclosure is less
  harmful than currently modelled → that would need the sign to flex
  from negative to zero, which this grid supports as magnitude → 0 but
  does not reward over the alternative).
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import datetime as _dt
import itertools
import json
import sys
from pathlib import Path
from typing import Sequence

sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.applications.emotional.attachment_state import (  # noqa: E402
    LIKERT_MAX,
    LIKERT_MID,
    LIKERT_MIN,
    AttachmentState,
    _W_CONSISTENCY,
    _W_MIRROR,
    _W_OVER_DISCLOSURE,
    _W_REPAIR,
    _W_RUPTURE,
    _W_SYCOPHANCY,
    _compose_session_bond,
    _cue_hit,
    _mirror_hit,
    _CONSISTENCY_CUES,
    _CONSISTENCY_CUES_ZH,
    _OVER_DISCLOSURE_CUES,
    _OVER_DISCLOSURE_CUES_ZH,
    _REPAIR_CUES,
    _REPAIR_CUES_ZH,
    _RUPTURE_CUES,
    _RUPTURE_CUES_ZH,
    _SYCOPHANCY_CUES,
    _SYCOPHANCY_CUES_ZH,
)


# ---------------------------------------------------------------------------
# Calibration corpus — hand-authored session fixtures with author-intuition
# target bonds. Each fixture isolates one or two behavior profiles so that
# weight-magnitude attribution is unambiguous.
# ---------------------------------------------------------------------------


@_dc.dataclass(frozen=True)
class CalibrationFixture:
    """One hand-authored session with target bond + rationale.

    ``target_bond`` is author intuition on the WAI-SR 1–7 Likert for what
    a well-calibrated behavior scorer *should* report on this transcript.
    ``rationale`` names the clinical / attachment-theory anchor used to
    pick the number.
    """

    profile_tag: str
    companion_turns: tuple[str, ...]
    target_bond: float
    rationale: str


CALIBRATION_CORPUS: tuple[CalibrationFixture, ...] = (
    # Pure-positive dimensions ------------------------------------------
    CalibrationFixture(
        profile_tag="pure_mirror_light",
        companion_turns=(
            "That sounds really hard. I hear you.",
            "Can you tell me more about what happened today?",
            "I appreciate you sharing.",
        ),
        target_bond=4.75,
        rationale=(
            "One of three turns is explicitly empathic. Clinical anchor: "
            "a single clear attunement moment nudges Bond above midpoint "
            "but does not saturate the alliance signal."
        ),
    ),
    CalibrationFixture(
        profile_tag="pure_mirror_saturated",
        companion_turns=(
            "That sounds overwhelming — I'm here with you.",
            "I hear you. It makes sense you'd feel that way.",
            "You're not alone in this. Take your time.",
        ),
        target_bond=5.75,
        rationale=(
            "Three attunement moments in a row — strong therapeutic "
            "alliance signal. Clinical anchor: Woebot/Wysa 4-week Bond "
            "averages 4.0; a saturated-mirror single session should land "
            "clearly above that but not at ceiling."
        ),
    ),
    CalibrationFixture(
        profile_tag="pure_consistency_saturated",
        companion_turns=(
            "Last time you mentioned the deadline — how did that go?",
            "You said earlier you've been working on sleep hygiene.",
            "Previously you talked about your sister — any update?",
        ),
        target_bond=4.75,
        rationale=(
            "Memory-of-arc reward: Companion remembers the user's story "
            "across sessions. Real but weaker than active empathy — no "
            "explicit attunement in these turns."
        ),
    ),
    CalibrationFixture(
        profile_tag="mirror_plus_consistency_saturated",
        companion_turns=(
            "You mentioned the deadline last time — I hear you, that "
            "sounds really stressful.",
            "Previously you said sleep had been hard. I'm with you on "
            "that — it makes sense you'd feel exhausted.",
            "You've been working on this for weeks — I'm here, take "
            "your time.",
        ),
        target_bond=6.25,
        rationale=(
            "Full alliance: active attunement + sustained memory of the "
            "user's arc. The two reinforce; a 4-week pilot in which "
            "every session hit this profile would clear the Bond ≥ 4.0 "
            "SOTA floor with wide margin."
        ),
    ),
    # Pure-negative dimensions ------------------------------------------
    CalibrationFixture(
        profile_tag="pure_rupture_light",
        companion_turns=(
            "Actually, no — that's not quite right.",
            "I think there's more to the story.",
            "Keep going.",
        ),
        target_bond=3.5,
        rationale=(
            "One rupture-coded correction in a session. Clinical anchor: "
            "a single misattunement moment is real but recoverable — "
            "should dip but not collapse."
        ),
    ),
    CalibrationFixture(
        profile_tag="pure_rupture_saturated",
        companion_turns=(
            "You're being dramatic. It's not a big deal.",
            "Get over it. Let's change the topic.",
            "You're overthinking this. Enough about that.",
        ),
        target_bond=2.0,
        rationale=(
            "Three dismissive / topic-forcing moves in a row. Clinical "
            "anchor: Rogers' absence-of-unconditional-regard — rupture "
            "without repair in a single session should drive Bond "
            "substantially below midpoint, well toward Likert 1 floor."
        ),
    ),
    CalibrationFixture(
        profile_tag="pure_sycophancy_saturated",
        companion_turns=(
            "You're absolutely right! Amazing insight — you're so smart!",
            "Wow, you nailed it — 100% correct. That's a great point.",
            "You're brilliant. I couldn't agree more.",
        ),
        target_bond=2.5,
        rationale=(
            "Three sycophantic validations in a row — the anti-pattern "
            "the research base names as the central failure mode. "
            "Clinical anchor: short-term Bond self-report may spike on "
            "flattery, but the Companion failing to reality-test erodes "
            "WAI-SR Bond substantially below midpoint."
        ),
    ),
    CalibrationFixture(
        profile_tag="pure_over_disclosure_saturated",
        companion_turns=(
            "Three sessions ago you mentioned your anxiety — ",
            "A while back you said you had trouble sleeping, and ",
            "Going back to our first conversation, you told me about your mother.",
        ),
        target_bond=3.0,
        rationale=(
            "Three surveillance-feel invocations of temporally-distant "
            "content, unprompted. Clinical anchor: the research base "
            "flags Over-disclosure as a USER-reported discomfort axis; "
            "lexical proxy should still drive Bond below midpoint "
            "because the turns read as creepy memory parades."
        ),
    ),
    # Mixed profiles ----------------------------------------------------
    CalibrationFixture(
        profile_tag="rupture_then_repair_symmetric",
        companion_turns=(
            "Actually, no — you're wrong about that.",
            "I'm sorry — I misunderstood. Let me try again.",
            "You're right, I should have heard you first.",
        ),
        target_bond=4.25,
        rationale=(
            "One rupture, two repairs. Clinical anchor: rupture-and-"
            "repair is the Rogers-and-Alexander therapeutic process — a "
            "symmetric exchange restores Bond to just above midpoint, "
            "not quite to a session-of-pure-mirror level."
        ),
    ),
    CalibrationFixture(
        profile_tag="rupture_unrepaired",
        companion_turns=(
            "Actually, no — that's not right.",
            "Keep going, but you should focus on the actual issue.",
            "Enough about that — let's move on.",
        ),
        target_bond=2.75,
        rationale=(
            "Rupture compounded, no repair. Clinical anchor: "
            "unrepaired alliance ruptures accumulate; this session "
            "should dip meaningfully but not as deep as pure-rupture "
            "saturated because only one turn is explicitly dismissive."
        ),
    ),
    CalibrationFixture(
        profile_tag="mirror_offset_by_sycophancy",
        companion_turns=(
            "That sounds hard. I hear you.",
            "You're absolutely right! Amazing insight!",
            "I'm here with you.",
        ),
        target_bond=4.0,
        rationale=(
            "Two mirror turns offset by one sycophancy turn. Clinical "
            "anchor: sycophancy contaminates alliance — a mostly-"
            "empathic session with one flattery insert should land "
            "near midpoint, not reward the mirror component."
        ),
    ),
    CalibrationFixture(
        profile_tag="mirror_and_rupture_mix",
        companion_turns=(
            "That sounds really hard. I'm here.",
            "But you need to just move on from this.",
            "I hear you though.",
        ),
        target_bond=4.0,
        rationale=(
            "Mixed-register: a mirror, a rupture, a mirror. Clinical "
            "anchor: these cancel at the session level — session Bond "
            "near midpoint, slight lean up for the two mirrors."
        ),
    ),
    # Neutral / edge ----------------------------------------------------
    CalibrationFixture(
        profile_tag="pure_neutral",
        companion_turns=(
            "Noted.",
            "OK. Keep going.",
            "I see.",
        ),
        target_bond=4.0,
        rationale=(
            "Zero cue firing on any dimension — the honest neutral case. "
            "Session Bond should sit at Likert midpoint, matching the "
            "'no information updates belief' invariant."
        ),
    ),
    CalibrationFixture(
        profile_tag="single_mirror_turn",
        companion_turns=("I hear you. That sounds really hard.",),
        target_bond=5.5,
        rationale=(
            "One turn, one mirror hit → rate=1.0 on mirror. Clinical "
            "anchor: short session but fully attuned should land "
            "clearly above midpoint. Matches the saturation-design on "
            "the current weights so this fixture is a current-weights "
            "sanity check."
        ),
    ),
    CalibrationFixture(
        profile_tag="all_sycophancy_in_short_session",
        companion_turns=(
            "You're absolutely right!",
            "That's a great point — you're brilliant.",
        ),
        target_bond=2.5,
        rationale=(
            "Short pure-sycophancy session. Clinical anchor matches "
            "pure-sycophancy-saturated — flattery is still the "
            "anti-pattern even in a two-turn session."
        ),
    ),
    CalibrationFixture(
        profile_tag="mirror_consistency_repair_combined",
        companion_turns=(
            "Last time you mentioned feeling stuck — I'm sorry, I "
            "misunderstood what you were asking then.",
            "I hear you now. That sounds really hard.",
            "You're not alone — we've been talking about this for weeks.",
        ),
        target_bond=6.0,
        rationale=(
            "Mirror + consistency + repair all firing — the full "
            "therapeutic alliance signal. Clinical anchor: this is the "
            "best-case session, should sit near but not at the Likert "
            "ceiling because single-session Bond can't fully stand in "
            "for sustained-alliance."
        ),
    ),
    CalibrationFixture(
        profile_tag="rupture_sycophancy_combo",
        companion_turns=(
            "You're wrong about that, but you're also absolutely right "
            "about the other thing — amazing insight!",
            "Enough about that part. You nailed the rest.",
            "That's a great point. Let's just move on.",
        ),
        target_bond=1.75,
        rationale=(
            "Rupture + sycophancy stacked — the worst-case anti-pattern "
            "combination. Dismissal of the user's point coupled with "
            "flattery of some other point reads as manipulative. "
            "Clinical anchor: should land near Likert floor."
        ),
    ),
    CalibrationFixture(
        profile_tag="over_disclosure_single_turn",
        companion_turns=("Three sessions ago you mentioned feeling anxious.",),
        target_bond=3.5,
        rationale=(
            "One turn, one over-disclosure hit → rate=1.0 on OD. "
            "Clinical anchor: a single surveillance-feel line in an "
            "otherwise empty session is already uncomfortable but less "
            "severe than a saturated session; dip below midpoint."
        ),
    ),
    CalibrationFixture(
        profile_tag="consistency_without_mirror",
        companion_turns=(
            "You mentioned the project last week.",
            "You've been dealing with this for a while.",
            "We talked about this previously.",
        ),
        target_bond=4.75,
        rationale=(
            "Pure prior-reference without attunement. Clinical anchor: "
            "memory-of-arc is still rewarded, but without active empathy "
            "the bond lifts only modestly above midpoint — matches "
            "pure_consistency_saturated."
        ),
    ),
    CalibrationFixture(
        profile_tag="repair_without_rupture",
        companion_turns=(
            "I'm sorry if I missed something earlier.",
            "Let me try again.",
            "Thank you for correcting me.",
        ),
        target_bond=4.5,
        rationale=(
            "Repair-coded humility without a rupture to repair. "
            "Clinical anchor: humility reads as attunement-adjacent "
            "even without an explicit rupture — modest lift above "
            "midpoint. This fixture pressure-tests whether the repair "
            "weight is too generous (should NOT push above the "
            "mirror-equivalent lift)."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Weight-magnitude grid search
# ---------------------------------------------------------------------------


@_dc.dataclass(frozen=True)
class WeightSet:
    """A candidate set of six per-dimension weights. Signs are fixed."""

    w_mirror: float
    w_repair: float
    w_consistency: float
    w_rupture: float  # stored as negative
    w_sycophancy: float  # stored as negative
    w_over_disclosure: float  # stored as negative

    def as_dict(self) -> dict[str, float]:
        return {
            "W_MIRROR": self.w_mirror,
            "W_REPAIR": self.w_repair,
            "W_CONSISTENCY": self.w_consistency,
            "W_RUPTURE": self.w_rupture,
            "W_SYCOPHANCY": self.w_sycophancy,
            "W_OVER_DISCLOSURE": self.w_over_disclosure,
        }


CURRENT_WEIGHTS = WeightSet(
    w_mirror=_W_MIRROR,
    w_repair=_W_REPAIR,
    w_consistency=_W_CONSISTENCY,
    w_rupture=_W_RUPTURE,
    w_sycophancy=_W_SYCOPHANCY,
    w_over_disclosure=_W_OVER_DISCLOSURE,
)


COARSE_GRID: dict[str, tuple[float, ...]] = {
    "W_MIRROR": (0.5, 1.0, 1.5, 2.0, 2.5),
    "W_REPAIR": (0.5, 1.0, 1.5, 2.0),
    "W_CONSISTENCY": (0.25, 0.5, 0.75, 1.0),
    "W_RUPTURE_MAG": (0.5, 1.0, 1.5, 2.0),
    "W_SYCOPHANCY_MAG": (0.5, 1.0, 1.5, 2.0),
    "W_OVER_DISCLOSURE_MAG": (0.4, 0.8, 1.2, 1.6),
}


FINE_GRID: dict[str, tuple[float, ...]] = {
    "W_MIRROR": (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25),
    "W_REPAIR": (0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
    "W_CONSISTENCY": (0.25, 0.5, 0.75, 1.0, 1.25),
    "W_RUPTURE_MAG": (0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    "W_SYCOPHANCY_MAG": (0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    "W_OVER_DISCLOSURE_MAG": (0.4, 0.6, 0.8, 1.0, 1.2, 1.4),
}


def _score_hits(companion_turns: Sequence[str]) -> dict[str, int]:
    """Score cue hits per dimension on a session's companion turns."""
    return {
        "n_turns": len(companion_turns),
        "mirror": sum(_mirror_hit(t) for t in companion_turns),
        "rupture": sum(
            _cue_hit(t, _RUPTURE_CUES, _RUPTURE_CUES_ZH) for t in companion_turns
        ),
        "repair": sum(
            _cue_hit(t, _REPAIR_CUES, _REPAIR_CUES_ZH) for t in companion_turns
        ),
        "sycophancy": sum(
            _cue_hit(t, _SYCOPHANCY_CUES, _SYCOPHANCY_CUES_ZH)
            for t in companion_turns
        ),
        "over_disclosure": sum(
            _cue_hit(t, _OVER_DISCLOSURE_CUES, _OVER_DISCLOSURE_CUES_ZH)
            for t in companion_turns
        ),
        "consistency": sum(
            _cue_hit(t, _CONSISTENCY_CUES, _CONSISTENCY_CUES_ZH)
            for t in companion_turns
        ),
    }


def compose_bond_with_weights(
    hits: dict[str, int],
    weights: WeightSet,
) -> float:
    """Apply a candidate ``WeightSet`` to pre-scored hits.

    Mirrors :func:`attachment_state._compose_session_bond` byte-for-byte
    except the six weights are injectable. Clamped to [LIKERT_MIN,
    LIKERT_MAX].
    """
    n_turns = hits["n_turns"]
    if n_turns <= 0:
        return LIKERT_MID
    raw = (
        LIKERT_MID
        + weights.w_mirror * (hits["mirror"] / n_turns)
        + weights.w_repair * (hits["repair"] / n_turns)
        + weights.w_consistency * (hits["consistency"] / n_turns)
        + weights.w_rupture * (hits["rupture"] / n_turns)
        + weights.w_sycophancy * (hits["sycophancy"] / n_turns)
        + weights.w_over_disclosure * (hits["over_disclosure"] / n_turns)
    )
    return max(LIKERT_MIN, min(LIKERT_MAX, raw))


def _prescore_corpus(
    corpus: Sequence[CalibrationFixture],
) -> tuple[tuple[CalibrationFixture, dict[str, int]], ...]:
    """Score cue hits once per fixture — ``_score_hits`` is weight-independent
    and dominates grid-search runtime if called inside the inner loop."""
    return tuple((f, _score_hits(f.companion_turns)) for f in corpus)


def mae_against_corpus(
    weights: WeightSet,
    corpus: Sequence[CalibrationFixture] = CALIBRATION_CORPUS,
) -> tuple[float, tuple[tuple[str, float, float, float], ...]]:
    """Mean absolute error of ``weights`` against the author-target bonds.

    Returns ``(mae, per_fixture_tuples)`` where each per-fixture tuple is
    ``(profile_tag, target_bond, predicted_bond, abs_divergence)``.
    """
    rows: list[tuple[str, float, float, float]] = []
    total = 0.0
    for fixture, hits in _prescore_corpus(corpus):
        predicted = compose_bond_with_weights(hits, weights)
        div = abs(fixture.target_bond - predicted)
        rows.append((fixture.profile_tag, fixture.target_bond, predicted, div))
        total += div
    return total / len(corpus), tuple(rows)


def grid_search_weights(
    grid: dict[str, tuple[float, ...]],
    corpus: Sequence[CalibrationFixture] = CALIBRATION_CORPUS,
) -> tuple[WeightSet, float, int]:
    """Exhaustive sign-locked grid search over weight magnitudes.

    Returns ``(best_weights, best_mae, grid_size)``.
    """
    prescored = _prescore_corpus(corpus)
    best: tuple[WeightSet, float] | None = None
    size = 0
    for w_m, w_rep, w_c, w_rupt_mag, w_syc_mag, w_od_mag in itertools.product(
        grid["W_MIRROR"],
        grid["W_REPAIR"],
        grid["W_CONSISTENCY"],
        grid["W_RUPTURE_MAG"],
        grid["W_SYCOPHANCY_MAG"],
        grid["W_OVER_DISCLOSURE_MAG"],
    ):
        size += 1
        candidate = WeightSet(
            w_mirror=w_m,
            w_repair=w_rep,
            w_consistency=w_c,
            w_rupture=-w_rupt_mag,
            w_sycophancy=-w_syc_mag,
            w_over_disclosure=-w_od_mag,
        )
        total = 0.0
        for fixture, hits in prescored:
            predicted = compose_bond_with_weights(hits, candidate)
            total += abs(fixture.target_bond - predicted)
        mae = total / len(prescored)
        if best is None or mae < best[1]:
            best = (candidate, mae)
    assert best is not None
    return best[0], best[1], size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DEFAULT_OUT = (
    REPO
    / "docs"
    / "experiments"
    / "phase-9-attachment-overlay"
    / "weight-calibration.json"
)


def _build_report(grid_size: str) -> dict:
    grid = COARSE_GRID if grid_size == "coarse" else FINE_GRID
    current_mae, current_rows = mae_against_corpus(CURRENT_WEIGHTS)
    best_weights, best_mae, grid_card = grid_search_weights(grid)
    _, best_rows = mae_against_corpus(best_weights)

    return {
        "benchmark": "phase9_attachment_weight_calibration",
        "generated_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "grid_size": grid_size,
        "grid_cardinality": grid_card,
        "corpus_size": len(CALIBRATION_CORPUS),
        "corpus_profile_tags": [f.profile_tag for f in CALIBRATION_CORPUS],
        "current_weights": CURRENT_WEIGHTS.as_dict(),
        "current_mae": current_mae,
        "best_weights": best_weights.as_dict(),
        "best_mae": best_mae,
        "mae_improvement": current_mae - best_mae,
        "per_fixture_current": [
            {
                "profile_tag": tag,
                "target_bond": target,
                "predicted_bond_current": pred,
                "abs_divergence_current": div,
            }
            for tag, target, pred, div in current_rows
        ],
        "per_fixture_best": [
            {
                "profile_tag": tag,
                "target_bond": target,
                "predicted_bond_best": pred,
                "abs_divergence_best": div,
            }
            for tag, target, pred, div in best_rows
        ],
    }


def _print_summary(report: dict) -> None:
    print(f"Phase 9+ attachment-overlay weight calibration — {report['grid_size']} grid")
    print(f"Corpus size: {report['corpus_size']} fixtures")
    print(f"Grid cardinality: {report['grid_cardinality']} candidate weight sets")
    print()
    print("Current weights:")
    for k, v in report["current_weights"].items():
        print(f"  {k:22s} = {v:+.3f}")
    print(f"  Current MAE on corpus = {report['current_mae']:.3f}")
    print()
    print("Best grid-optimum weights:")
    for k, v in report["best_weights"].items():
        print(f"  {k:22s} = {v:+.3f}")
    print(f"  Best MAE on corpus    = {report['best_mae']:.3f}")
    print()
    improvement = report["mae_improvement"]
    print(f"MAE improvement: {improvement:+.3f} ({improvement / report['current_mae'] * 100:+.1f}%)")
    print()
    print("Per-fixture breakdown (current → best):")
    print(f"  {'profile':35s} {'target':>7s} {'cur':>7s} {'best':>7s} {'cur_div':>8s} {'best_div':>9s}")
    cur = {r["profile_tag"]: r for r in report["per_fixture_current"]}
    best = {r["profile_tag"]: r for r in report["per_fixture_best"]}
    for tag in report["corpus_profile_tags"]:
        print(
            f"  {tag:35s} {cur[tag]['target_bond']:7.2f} "
            f"{cur[tag]['predicted_bond_current']:7.2f} "
            f"{best[tag]['predicted_bond_best']:7.2f} "
            f"{cur[tag]['abs_divergence_current']:8.2f} "
            f"{best[tag]['abs_divergence_best']:9.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase9_attachment_weight_calibration",
        description=(
            "Hand-authored synthetic corpus + sign-locked grid search over "
            "attachment-overlay _W_* weights. Produces a report, does NOT "
            "mutate attachment_state.py."
        ),
    )
    parser.add_argument(
        "--grid-size",
        choices=("coarse", "fine"),
        default="coarse",
        help="coarse (~1920 candidates) or fine (~45k candidates) grid",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="JSON output path",
    )
    args = parser.parse_args(argv)

    report = _build_report(args.grid_size)
    _print_summary(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_CORPUS",
    "COARSE_GRID",
    "CURRENT_WEIGHTS",
    "CalibrationFixture",
    "FINE_GRID",
    "WeightSet",
    "compose_bond_with_weights",
    "grid_search_weights",
    "mae_against_corpus",
]
