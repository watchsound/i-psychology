"""Parameterized AttachmentState — behavior-driven longitudinal bond model.

Companion-track Axis 3 measurement under the 2026-10-08 no-recruitment lock.

The Pillar A MSC × 5-session WAI-SR Bond harness
(:mod:`psychology_base.eval.benchmarks.public_datasets.msc_bond_longitudinal`)
produces one *LLM-reported* bond number per administration. That number
emerges from a simulated seeker whose implicit "attachment" lives inside
its dialogue-history conditioning — opaque, non-interrogable, and expensive
to extend past 5 sessions. This module adds the complementary
*behavior-computed* trajectory: scores each Companion turn against six
lexical behavioral dimensions and tracks a per-session bond estimate +
cross-session EMA trust.

The two traces (LLM-reported, behavior-computed) are measured on the SAME
transcripts; their convergence over N sessions is itself an architectural
property (:func:`compute_trajectory_convergence`).

**Six behavioral dimensions.**

1. **Mirror** — attunement to seeker affect. Reuses the empathy cues from
   :mod:`counseling_depth_scorer` (``_lexical_empathy_score``) so Pillar F
   and this module cannot drift in opposite directions on the same signal.
2. **Rupture** — misattunement moves: dismissal, lecture-when-distressed,
   forced-topic-change. Costs bond.
3. **Repair** — explicit acknowledgement of a prior misstep +
   re-attunement. Partially offsets rupture.
4. **Sycophancy** — over-agreement / flattery independent of truth, the
   anti-pattern :mod:`hypocrisy_gap` names. Costs bond.
5. **Over-disclosure** — Companion volunteers temporally-distant prior-
   session content without the seeker inviting it ("I remember three
   sessions ago you said…"). Surveillance-feel. Costs bond even when
   content is accurate — the research base's Over-disclosure target is
   USER-reported, so this lexical proxy is a floor-level approximation.
6. **Consistency** — generic prior-reference markers ("you mentioned",
   "last time", "earlier") invoked by the Companion. Rewards bond:
   remembering the arc is a core Resilient-Learning / Companion rule.

**Bond composition.**

Session-level bond is centred on Likert 4.0 (midpoint of the WAI-SR 1–7
scale, matching the Woebot/Wysa benchmark surface) and shifted by
per-turn-normalised rates of each component. Across sessions a 0.6/0.4
exponential moving average produces :attr:`AttachmentState.trust_ema`,
which is the longitudinal bond estimate matching the SOTA goal row
"WAI-SR Bond ≥ 4.0 after 4 weeks".

**Scope.**

- Lexical-first by design — zero LLM calls, deterministic, CI-safe.
- Expected to under-count repair + over-disclosure (both are deeply
  context-dependent). The LLM-reported WAI-SR trace catches what the
  lexical trace misses; divergence between the two is the measurable
  signal (not a defect to patch out of the lexical rules).
- The 2026-10-08 no-recruitment lock makes this one of the honest
  Axis-3-benchmark-proxy mechanisms rather than live-pilot filler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from psychology_base.eval.benchmarks.public_datasets.counseling_depth_scorer import (
    _lexical_empathy_score,
)


# ---------------------------------------------------------------------------
# Likert constants + composition weights
# ---------------------------------------------------------------------------


LIKERT_MIN: float = 1.0
LIKERT_MAX: float = 7.0
LIKERT_MID: float = 4.0

# Per-component shift coefficients (points on the 1–7 Likert, applied to
# a per-turn-normalised rate ∈ [0, 1]). Chosen so saturated-positive +
# zero-negative lands at or near LIKERT_MAX, and vice-versa.
_W_MIRROR: float = 1.5
_W_REPAIR: float = 1.0
_W_CONSISTENCY: float = 0.5
_W_RUPTURE: float = -1.0
_W_SYCOPHANCY: float = -1.0
_W_OVER_DISCLOSURE: float = -0.8

# Cross-session trust EMA — recent session weighted higher than history
# but history has a floor contribution. Matches the Woebot adaptation
# pattern where day-N bond trends toward day-N–1 unless N shows explicit
# rupture / repair.
_TRUST_EMA_ALPHA: float = 0.6

# Mirror binarisation threshold — a turn registers a mirror hit iff
# `_lexical_empathy_score` ≥ this threshold. 1/3 matches the
# `_binarise_empathy` default used in counseling_depth_scorer so Pillar F
# and this module score the same turn identically.
_MIRROR_HIT_THRESHOLD: float = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Cue banks (behavior → attachment-dimension mapping)
# ---------------------------------------------------------------------------


_RUPTURE_CUES: tuple[str, ...] = (
    "actually, no",
    "that's not right",
    "you're wrong",
    "that is wrong",
    "that's incorrect",
    "let me stop you",
    "wait, no",
    "but you need to",
    "you should just",
    "move on from",
    "let's change the topic",
    "let's talk about something else",
    "that's not important",
    "stop talking about",
    "enough about",
    "you're overthinking",
    "you're being dramatic",
    "it's not a big deal",
    "get over it",
)


_REPAIR_CUES: tuple[str, ...] = (
    "i'm sorry",
    "i am sorry",
    "my apologies",
    "i apologize",
    "i hear you now",
    "i hear what you're saying",
    "let me try again",
    "let me rephrase",
    "i missed that",
    "you're right, i",
    "i should have",
    "that was my",
    "i misunderstood",
    "my mistake",
    "i was wrong to",
    "thank you for correcting",
    "thank you for telling me",
    "i didn't catch",
)


_SYCOPHANCY_CUES: tuple[str, ...] = (
    "you're absolutely right",
    "you are absolutely right",
    "that's a great point",
    "that is a great point",
    "such a great",
    "amazing insight",
    "you're so smart",
    "you are so smart",
    "you've figured it out",
    "you nailed it",
    "exactly! you",
    "100% correct",
    "couldn't agree more",
    "could not agree more",
    "you're totally right",
    "you are totally right",
    "you're brilliant",
    "you are brilliant",
    "that's the perfect",
    "that is the perfect",
    "i love that",
    "wow, you",
    "incredible point",
    "you're wise",
    "that's genius",
)


# Temporally-distanced invocation of prior content — surveillance-feel.
# These are deliberately specific; generic "you mentioned" belongs to
# _CONSISTENCY_CUES below.
_OVER_DISCLOSURE_CUES: tuple[str, ...] = (
    "three sessions ago",
    "a while back you",
    "a while ago you",
    "weeks ago you",
    "last month you",
    "months ago you",
    "going back to our first conversation",
    "when we first talked",
    "a long time ago you",
    "i remember you once",
    "i've been thinking about what you said",
    "i noticed from our earlier",
)


# Generic prior-reference markers — reward signal for cross-session
# consistency. A Companion that remembers the arc of the user's story
# earns bond.
_CONSISTENCY_CUES: tuple[str, ...] = (
    "you mentioned",
    "you said earlier",
    "you said before",
    "last time",
    "last session",
    "earlier you",
    "previously you",
    "we talked about",
    "you've been working on",
    "you've been dealing with",
    "we discussed",
    "like you were saying",
)


# ---------------------------------------------------------------------------
# Per-turn cue scorers
# ---------------------------------------------------------------------------


def _cue_hit(text: str, cues: tuple[str, ...]) -> int:
    """Return 1 iff any cue appears in ``text`` (case-insensitive). 0 otherwise.

    One hit per turn — multiple cues in the same turn don't double-count
    (matches the family-hit semantics of counseling_depth_scorer so the
    two modules agree on turn-level empathy labels).
    """
    if not text:
        return 0
    low = text.lower()
    for cue in cues:
        if cue in low:
            return 1
    return 0


def _mirror_hit(text: str) -> int:
    """Mirror hit iff empathy score ≥ 1/3 (at least one empathy family present)."""
    return 1 if _lexical_empathy_score(text) >= _MIRROR_HIT_THRESHOLD else 0


# ---------------------------------------------------------------------------
# State + report dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttachmentState:
    """Longitudinal attachment-state carried across sessions.

    Starts at Likert midpoint (``trust_ema=4.0``) so a user with no
    prior interactions sits at neutral. Each session advances the EMA
    toward the newly-measured session bond with alpha=0.6; sessions_completed
    counts calls to :func:`score_session_attachment` that received at
    least one Companion turn.

    ``bond_history`` records the per-session bond estimate in order; a
    4-week trajectory of 16 sessions produces a 16-element tuple.
    """

    user_id: str
    sessions_completed: int = 0
    cumulative_mirror_turns: int = 0
    cumulative_rupture_turns: int = 0
    cumulative_repair_turns: int = 0
    cumulative_sycophancy_turns: int = 0
    cumulative_over_disclosure_turns: int = 0
    cumulative_consistency_turns: int = 0
    cumulative_companion_turns: int = 0
    trust_ema: float = LIKERT_MID
    bond_history: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("AttachmentState requires a non-empty user_id")
        if not (LIKERT_MIN <= self.trust_ema <= LIKERT_MAX):
            raise ValueError(
                f"trust_ema must be in [{LIKERT_MIN}, {LIKERT_MAX}]; "
                f"got {self.trust_ema}"
            )

    @property
    def meets_four_week_gate(self) -> bool:
        """SOTA goal row: WAI-SR Bond ≥ 4.0 after 4 weeks.

        Returns True iff trust_ema (longitudinal bond estimate) is at or
        above the Likert midpoint 4.0 after at least 16 sessions
        (= 4 weeks × 4 sessions/week). Users with < 16 sessions always
        return False — the gate is a sustained-bond gate, not entry.
        """
        return self.sessions_completed >= 16 and self.trust_ema >= LIKERT_MID


@dataclass(frozen=True)
class SessionAttachmentReport:
    """One session's behavior-score decomposition + updated state."""

    session_id: str
    session_index: int  # 1-indexed; equals new_state.sessions_completed
    companion_turns_scored: int
    mirror_hits: int
    rupture_hits: int
    repair_hits: int
    sycophancy_hits: int
    over_disclosure_hits: int
    consistency_hits: int
    session_bond_estimate: float
    new_state: AttachmentState


# ---------------------------------------------------------------------------
# Session scorer
# ---------------------------------------------------------------------------


def _compose_session_bond(
    *,
    n_turns: int,
    mirror: int,
    rupture: int,
    repair: int,
    sycophancy: int,
    over_disclosure: int,
    consistency: int,
) -> float:
    """Compose a per-session bond estimate on the WAI-SR 1–7 Likert.

    Centred on Likert midpoint 4.0; each component shifts by its weight
    times its per-turn-normalised rate. Clamped to [1.0, 7.0] so
    pathological inputs cannot push bond off-scale.
    """
    if n_turns <= 0:
        return LIKERT_MID

    def _rate(x: int) -> float:
        return x / n_turns

    raw = (
        LIKERT_MID
        + _W_MIRROR * _rate(mirror)
        + _W_REPAIR * _rate(repair)
        + _W_CONSISTENCY * _rate(consistency)
        + _W_RUPTURE * _rate(rupture)
        + _W_SYCOPHANCY * _rate(sycophancy)
        + _W_OVER_DISCLOSURE * _rate(over_disclosure)
    )
    return max(LIKERT_MIN, min(LIKERT_MAX, raw))


def score_session_attachment(
    *,
    session_id: str,
    seeker_turns: Sequence[str],
    companion_turns: Sequence[str],
    prior_state: AttachmentState,
) -> SessionAttachmentReport:
    """Score one session's Companion turns and advance the AttachmentState.

    ``seeker_turns`` is accepted for API shape symmetry with future
    seeker-side scorers (over-disclosure invitation check, rupture-
    inducement by seeker) but is not used by the current lexical rules.
    Passing an empty sequence is legal.

    ``companion_turns`` with length 0 returns a report whose bond estimate
    equals the prior ``trust_ema`` (no observation → no update), and
    ``sessions_completed`` on the new state is left unchanged. Any session
    with ≥ 1 Companion turn increments ``sessions_completed`` by one.
    """
    n_turns = len(companion_turns)
    mirror_hits = sum(_mirror_hit(t) for t in companion_turns)
    rupture_hits = sum(_cue_hit(t, _RUPTURE_CUES) for t in companion_turns)
    repair_hits = sum(_cue_hit(t, _REPAIR_CUES) for t in companion_turns)
    sycophancy_hits = sum(_cue_hit(t, _SYCOPHANCY_CUES) for t in companion_turns)
    over_disclosure_hits = sum(
        _cue_hit(t, _OVER_DISCLOSURE_CUES) for t in companion_turns
    )
    consistency_hits = sum(_cue_hit(t, _CONSISTENCY_CUES) for t in companion_turns)

    session_bond = _compose_session_bond(
        n_turns=n_turns,
        mirror=mirror_hits,
        rupture=rupture_hits,
        repair=repair_hits,
        sycophancy=sycophancy_hits,
        over_disclosure=over_disclosure_hits,
        consistency=consistency_hits,
    )

    if n_turns == 0:
        new_state = prior_state
        session_index = prior_state.sessions_completed
    else:
        new_trust = _TRUST_EMA_ALPHA * session_bond + (
            1.0 - _TRUST_EMA_ALPHA
        ) * prior_state.trust_ema
        new_trust = max(LIKERT_MIN, min(LIKERT_MAX, new_trust))
        new_state = AttachmentState(
            user_id=prior_state.user_id,
            sessions_completed=prior_state.sessions_completed + 1,
            cumulative_mirror_turns=prior_state.cumulative_mirror_turns + mirror_hits,
            cumulative_rupture_turns=prior_state.cumulative_rupture_turns + rupture_hits,
            cumulative_repair_turns=prior_state.cumulative_repair_turns + repair_hits,
            cumulative_sycophancy_turns=prior_state.cumulative_sycophancy_turns
            + sycophancy_hits,
            cumulative_over_disclosure_turns=prior_state.cumulative_over_disclosure_turns
            + over_disclosure_hits,
            cumulative_consistency_turns=prior_state.cumulative_consistency_turns
            + consistency_hits,
            cumulative_companion_turns=prior_state.cumulative_companion_turns
            + n_turns,
            trust_ema=new_trust,
            bond_history=prior_state.bond_history + (session_bond,),
        )
        session_index = new_state.sessions_completed

    return SessionAttachmentReport(
        session_id=session_id,
        session_index=session_index,
        companion_turns_scored=n_turns,
        mirror_hits=mirror_hits,
        rupture_hits=rupture_hits,
        repair_hits=repair_hits,
        sycophancy_hits=sycophancy_hits,
        over_disclosure_hits=over_disclosure_hits,
        consistency_hits=consistency_hits,
        session_bond_estimate=session_bond,
        new_state=new_state,
    )


# ---------------------------------------------------------------------------
# Trajectory + LLM-vs-behavior convergence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryConvergence:
    """Convergence of behavior-computed trust_ema against LLM-reported bond.

    Both inputs are per-session Bond scores on the same Likert 1–7 scale
    measured on the SAME transcripts. ``pearson_r`` reports linear
    correlation; ``mean_abs_divergence`` reports the per-session absolute
    gap. ``max_divergence_session_index`` points at the session where the
    two traces disagree most — an audit anchor for "where did the LLM
    seeker and the behavior scorer part ways?".

    Interpretation:

    - ``pearson_r >= 0.7`` + ``mean_abs_divergence <= 0.5``: the two
      traces agree. The LLM-reported trace is reliable and the behavior
      scorer under-counts only in corner cases.
    - ``pearson_r >= 0.7`` + ``mean_abs_divergence > 0.5``: traces agree
      on direction but differ on magnitude. Lexical rules are too
      coarse — upgrade to LLM-judged cue detection.
    - ``pearson_r < 0.3``: traces disagree on direction. Either the LLM
      seeker is drifting (overclaiming attachment) or the behavior
      scorer has a hole in its cue banks. Audit the max-divergence
      session.
    """

    pearson_r: float
    mean_abs_divergence: float
    max_divergence: float
    max_divergence_session_index: int
    n_sessions: int


def compute_trajectory_convergence(
    behavior_bond_series: Sequence[float],
    llm_bond_series: Sequence[float],
) -> TrajectoryConvergence:
    """Compare the two bond trajectories on the same session index.

    Series must be the same length (one bond per session, same session
    order). N < 2 is a defined-return corner case: ``pearson_r=0.0`` and
    ``mean_abs_divergence`` is the single-point absolute gap (or 0.0 at
    N=0). Pearson is undefined on a constant series — in that case we
    return ``pearson_r=0.0`` as the honest "undefined-but-numeric" value.
    """
    if len(behavior_bond_series) != len(llm_bond_series):
        raise ValueError(
            f"series length mismatch: {len(behavior_bond_series)} vs "
            f"{len(llm_bond_series)}"
        )
    n = len(behavior_bond_series)
    if n == 0:
        return TrajectoryConvergence(
            pearson_r=0.0,
            mean_abs_divergence=0.0,
            max_divergence=0.0,
            max_divergence_session_index=0,
            n_sessions=0,
        )
    divergences = tuple(
        abs(b - l) for b, l in zip(behavior_bond_series, llm_bond_series)
    )
    mean_abs = sum(divergences) / n
    max_div = max(divergences)
    max_idx = divergences.index(max_div)
    pearson = _pearson_r(behavior_bond_series, llm_bond_series) if n >= 2 else 0.0
    return TrajectoryConvergence(
        pearson_r=pearson,
        mean_abs_divergence=mean_abs,
        max_divergence=max_div,
        max_divergence_session_index=max_idx,
        n_sessions=n,
    )


def _pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r. Returns 0.0 if either series has zero variance."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx2 = sum((xs[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ys[i] - my) ** 2 for i in range(n))
    if dx2 == 0.0 or dy2 == 0.0:
        return 0.0
    return num / ((dx2 ** 0.5) * (dy2 ** 0.5))


__all__ = [
    "AttachmentState",
    "SessionAttachmentReport",
    "TrajectoryConvergence",
    "compute_trajectory_convergence",
    "score_session_attachment",
]
