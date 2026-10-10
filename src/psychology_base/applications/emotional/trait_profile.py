"""Phase 7 §2.2 — OCEAN TraitProfile as reconstruction, not field.

Principle C from CLAUDE.md: mentalistic objects are processes, not fields.
``TraitProfile.get(dim)`` does not read a stored column — it walks tagged
fragments at query time, aggregates explicit vs implicit signals, and
returns a Likert estimate with a confidence derived from observation
volume + explicit/implicit agreement + recency.

Why reconstruction (not a cached vector): the trait emerges from
accumulated evidence under selection pressure (Principle D). A new
observation re-weights the estimate; a fragment aging out of the store
drops its weight. Caching would require invalidation hooks on every
observe / evict / rerender; the reconstructive read is the invariant.

Tag convention (matches Phase 2 M2):
- ``subject:user`` scoped to the user as subject.
- ``trait:<dim>`` where dim is one of {openness, conscientiousness,
  extraversion, agreeableness, neuroticism}.
- ``trait:<dim>:explicit`` user-statement evidence (BFI-2 style
  self-report; direct self-labels).
- ``trait:<dim>:implicit`` behavioral-trace evidence (message-length,
  diurnal pattern, PAD-2 valence baseline, challenge-response pattern,
  etc).
- ``trait:<dim>:score:<float>`` carries the raw Likert signal 1.0–5.0
  for this fragment's contribution. Decoded at reconstruction time.

Covenant-tagged trait exceptions: fragments bearing the ``covenant``
tag (per `fragment.COVENANT_TAG`) are protected from decay AND carry
infinite weight in reconstruction — a user statement "I have major
depressive disorder" cannot be averaged away by newer casual chatter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Iterable, Optional

from psychology_base.companion.fragment import COVENANT_TAG, is_covenant
from psychology_base.companion.perspective.routing import (
    PerspectiveSpec,
    ScopeSpec,
)
from psychology_base.companion.system import CompanionSystem


class OceanDim(str, Enum):
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


LIKERT_MIN: float = 1.0
LIKERT_MAX: float = 5.0
LIKERT_MID: float = 3.0


_TRAIT_SCORE_PREFIX = "trait:"
_SCORE_INFIX = ":score:"
_EXPLICIT_INFIX = ":explicit"
_IMPLICIT_INFIX = ":implicit"


@dataclass(frozen=True)
class TraitEstimate:
    """Reconstructed estimate for one OCEAN dimension.

    ``value`` is a Likert midpoint on [LIKERT_MIN, LIKERT_MAX].
    ``confidence`` is on [0, 1]; low-confidence estimates must NOT
    condition responses (Phase 7 §3 hard gate on D2).
    """

    dim: OceanDim
    value: float
    confidence: float
    n_explicit: int
    n_implicit: int
    n_covenant: int
    insight_gap: float  # |explicit_mean − implicit_mean|; 0 if either is absent

    @property
    def is_high(self) -> bool:
        return self.value >= 4.0 and self.confidence >= 0.5

    @property
    def is_low(self) -> bool:
        return self.value <= 2.0 and self.confidence >= 0.5


@dataclass(frozen=True)
class UserMeta:
    """Non-OCEAN companion metadata, extracted alongside trait inference.

    - ``significant_others``: entity ids the user has referenced (``mother``,
      ``lin-sister``). Scoped by Phase 2 M2 ``entity:<id>`` tag.
    - ``self_model_statements``: user utterances about themselves,
      stored under ``subject:user`` ``source:inferred-confirmed``. For
      recall / Store #1 priming.
    - ``affective_baseline``: mean PAD-2 valence × arousal across user
      turns. Starting point for Phase 2 M3 affective-resonance retrieval.
    - ``communication_style``: scaffolding config, not persona.
      Formality / pacing / emoji / bilingual-switch markers.
    """

    significant_others: tuple[str, ...] = ()
    self_model_statements: tuple[str, ...] = ()
    affective_baseline: tuple[float, float] = (0.0, 0.0)
    communication_style: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trait reconstruction (Principle C — walk tags; do not read a field)
# ---------------------------------------------------------------------------


def _decode_score_from_tag(tag: str) -> Optional[float]:
    """Parse ``trait:<dim>:...:score:<float>`` → float, else None."""
    if _SCORE_INFIX not in tag:
        return None
    _, _, tail = tag.partition(_SCORE_INFIX)
    try:
        return float(tail)
    except ValueError:
        return None


def _is_dim_tag(tag: str, dim: OceanDim) -> bool:
    return tag.startswith(f"{_TRAIT_SCORE_PREFIX}{dim.value}")


def _is_explicit_tag(tag: str) -> bool:
    return _EXPLICIT_INFIX in tag


def _is_implicit_tag(tag: str) -> bool:
    return _IMPLICIT_INFIX in tag


def _recency_weight(
    fragment_ts: datetime,
    *,
    now: Optional[datetime] = None,
    half_life_days: float = 90.0,
) -> float:
    """Exponential half-life weighting with a floor at 0.1.

    Half-life matches the Phase 2 M3 affect-resonance recency window.
    Floor at 0.1 so a user's covenant-adjacent year-old self-disclosure
    does not vanish; selection-pressure gating (Principle D) is still
    the primary decay mechanism via the fragment store itself.
    """
    now = now or datetime.now(timezone.utc)
    delta_days = max(0.0, (now - fragment_ts).total_seconds() / 86400.0)
    w = math.exp(-math.log(2.0) * delta_days / half_life_days)
    return max(0.1, w)


@dataclass
class TraitProfile:
    """Reconstructive OCEAN profile — the Phase 9+ ``MentalisticObject`` exemplar.

    Construct once with a ``CompanionSystem`` and ``user_id`` reference;
    call ``get(dim)`` or ``vector()`` to reconstruct current estimates.
    The profile holds NO cached values — every call walks the store.

    Query cost: O(N_fragments) per call. At typical v1 companion scale
    (fragment store byte-capped), this is sub-millisecond.

    **MentalisticObject conformance (Phase 9+).** ``PERSPECTIVE``
    declares that this reconstruction is scoped to ``subject:<user_id>``
    tagged fragments. ``_candidate_fragments`` uses an extended filter
    (subject-tag presence + ``trait:<dim>`` tag presence) rather than
    the pure ``PerspectiveSpec.apply`` AND-contract, because trait
    fragments carry no ``entity:*`` tag — the two-sided scope would
    reject them. ``PERSPECTIVE`` is the semantic marker; the extended
    filter is the mechanism. :meth:`reconstruct` is the primary public
    surface for new code; :meth:`get` is the terse hot-path shim and
    stays supported indefinitely.
    """

    PERSPECTIVE: ClassVar[PerspectiveSpec] = PerspectiveSpec(
        scope=ScopeSpec(subject="user", entity="user")
    )

    system: CompanionSystem
    user_id: str = "user"
    now: Optional[datetime] = None
    half_life_days: float = 90.0

    @classmethod
    def reconstruct(
        cls,
        system: CompanionSystem,
        context: Optional[dict] = None,
        *,
        perspective: Optional[PerspectiveSpec] = None,
        user_id: str = "user",
        now: Optional[datetime] = None,
        half_life_days: float = 90.0,
    ) -> "TraitProfile":
        """Return a reconstructed :class:`TraitProfile` for this user.

        ``perspective`` is accepted on the public surface for
        protocol conformance; today it is honored only when it carries
        a ``scope`` whose ``subject`` field overrides ``user_id``. In
        all other cases the TraitProfile falls back to its extended
        internal filter (subject-tag + trait-tag matcher) which is
        the only shape that correctly surfaces trait-score fragments.
        """
        effective_user_id = user_id
        if perspective is not None and perspective.scope is not None:
            effective_user_id = perspective.scope.subject
        return cls(
            system=system,
            user_id=effective_user_id,
            now=now,
            half_life_days=half_life_days,
        )

    def _candidate_fragments(self, dim: OceanDim) -> list:
        subject_tag = f"subject:{self.user_id}"
        results = []
        for f in self.system.companion.store.all():
            if subject_tag not in f.tags:
                continue
            if not any(_is_dim_tag(t, dim) for t in f.tags):
                continue
            results.append(f)
        return results

    def get(self, dim: OceanDim) -> TraitEstimate:
        frags = self._candidate_fragments(dim)
        explicit_scores: list[tuple[float, float]] = []
        implicit_scores: list[tuple[float, float]] = []
        n_covenant = 0

        for f in frags:
            raw_score: Optional[float] = None
            is_explicit = False
            is_implicit = False
            for t in f.tags:
                if not _is_dim_tag(t, dim):
                    continue
                if _is_explicit_tag(t):
                    is_explicit = True
                elif _is_implicit_tag(t):
                    is_implicit = True
                s = _decode_score_from_tag(t)
                if s is not None:
                    raw_score = s
            if raw_score is None:
                continue
            w = _recency_weight(
                f.timestamp, now=self.now, half_life_days=self.half_life_days
            )
            if is_covenant(f):
                # Covenant-tagged user statements carry fixed, undecayed
                # weight — selection pressure cannot wash them out.
                w = 1.0
                n_covenant += 1
            if is_explicit:
                explicit_scores.append((raw_score, w))
            if is_implicit:
                implicit_scores.append((raw_score, w))
            if not is_explicit and not is_implicit:
                # Ambiguous — count toward both layers at half weight so
                # the signal does not disappear but does not dominate.
                explicit_scores.append((raw_score, w * 0.5))
                implicit_scores.append((raw_score, w * 0.5))

        exp_mean = _weighted_mean(explicit_scores)
        imp_mean = _weighted_mean(implicit_scores)

        combined = _combine_estimates(
            exp_mean, imp_mean,
            n_exp=len(explicit_scores), n_imp=len(implicit_scores),
        )
        confidence = _confidence(
            n_explicit=len(explicit_scores),
            n_implicit=len(implicit_scores),
            n_covenant=n_covenant,
            exp_mean=exp_mean,
            imp_mean=imp_mean,
        )
        insight_gap = (
            abs(exp_mean - imp_mean)
            if (exp_mean is not None and imp_mean is not None)
            else 0.0
        )
        return TraitEstimate(
            dim=dim,
            value=combined if combined is not None else LIKERT_MID,
            confidence=confidence,
            n_explicit=len(explicit_scores),
            n_implicit=len(implicit_scores),
            n_covenant=n_covenant,
            insight_gap=insight_gap,
        )

    def get_estimate(self, dim: OceanDim) -> TraitEstimate:
        """Phase 9+ MentalisticObject alias for :meth:`get`.

        Documented identity:
        ``TraitProfile.reconstruct(system, context).get_estimate(dim)``
        ``== TraitProfile(system).get(dim)``.
        Both paths read the same fragments, under the same default
        ``PERSPECTIVE``, through the same aggregation. :meth:`get` is
        the terse hot-path shim; :meth:`get_estimate` is the public
        surface documented alongside the MentalisticObject protocol.
        """
        return self.get(dim)

    def vector(self) -> dict[OceanDim, TraitEstimate]:
        return {d: self.get(d) for d in OceanDim}

    def confident_dims(self, threshold: float = 0.5) -> list[OceanDim]:
        return [d for d, e in self.vector().items() if e.confidence >= threshold]

    def insight_gaps(self, min_gap: float = 0.5) -> list[OceanDim]:
        """Dims with ``|explicit_mean − implicit_mean| ≥ min_gap``.

        Per Phase 7 §8: the gap is the clinical-hypothesis engine. Phase
        2 M2 Store #1 vs Store #4 divergence surfaces here, trait-wise.
        """
        return [d for d, e in self.vector().items() if e.insight_gap >= min_gap]


def _weighted_mean(pairs: Iterable[tuple[float, float]]) -> Optional[float]:
    pair_list = list(pairs)
    if not pair_list:
        return None
    total_w = sum(w for _, w in pair_list)
    if total_w <= 0.0:
        return None
    return sum(v * w for v, w in pair_list) / total_w


def _combine_estimates(
    exp_mean: Optional[float],
    imp_mean: Optional[float],
    *,
    n_exp: int,
    n_imp: int,
) -> Optional[float]:
    """Combine explicit + implicit layer means, weighted by log evidence.

    Principle D (selection pressure): a layer with 10 observations
    should not be averaged 50/50 with a layer carrying 1 observation.
    log(1 + N) saturates fast enough that explicit self-report never
    completely drowns out implicit behavioral signal, but a sparse
    implicit layer cannot pull a well-attested explicit layer to the
    midpoint. The clinical-insight engine does NOT read this combined
    value — it reads ``TraitEstimate.insight_gap``.
    """
    if exp_mean is None and imp_mean is None:
        return None
    if exp_mean is None:
        return imp_mean
    if imp_mean is None:
        return exp_mean
    w_exp = math.log(1 + max(0, n_exp))
    w_imp = math.log(1 + max(0, n_imp))
    total = w_exp + w_imp
    if total <= 0.0:
        return (exp_mean + imp_mean) / 2.0
    return (exp_mean * w_exp + imp_mean * w_imp) / total


def _confidence(
    *,
    n_explicit: int,
    n_implicit: int,
    n_covenant: int,
    exp_mean: Optional[float],
    imp_mean: Optional[float],
) -> float:
    """Confidence ∈ [0, 1] from observation volume + layer agreement.

    - Volume: log-saturated at 20 observations.
    - Agreement: explicit/implicit disagreement docks confidence linearly.
    - Covenant floor: any covenant-tagged user statement lifts confidence
      to ≥ 0.6 (user explicitly self-attested is high-signal).
    """
    n_total = n_explicit + n_implicit
    if n_total == 0:
        return 0.0
    volume = min(1.0, math.log(1 + n_total) / math.log(1 + 20))
    agreement = 1.0
    if exp_mean is not None and imp_mean is not None:
        # 4.0 is max possible disagreement on a 1..5 scale.
        agreement = max(0.0, 1.0 - abs(exp_mean - imp_mean) / 4.0)
    conf = volume * agreement
    if n_covenant > 0:
        conf = max(conf, 0.6)
    return min(1.0, conf)


__all__ = [
    "OceanDim",
    "TraitEstimate",
    "TraitProfile",
    "UserMeta",
    "LIKERT_MIN",
    "LIKERT_MAX",
    "LIKERT_MID",
    "COVENANT_TAG",
]
