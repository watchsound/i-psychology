"""B2 — perspective-taking (换位思考 + clinical-insight) scorers.

Deterministic sub-metrics for the B2 exit gate ([docs/experiments/
phase-2-mentalistic/option-M2-perspective-tagged/README.md](../../../../
docs/experiments/phase-2-mentalistic/option-M2-perspective-tagged/README.md)):

1. **Attribute reconstruction.** Fraction of expected attributes
   recovered from the scope-filtered recall content. Attributes are
   listed in the scenario YAML with a set of surface tokens; a hit
   requires at least one token to appear in the recall content (case-
   insensitive, substring match). Target ≥ 10/12.
2. **Grounding fraction.** Fraction of perspective-taking recall
   answers that cite ≥ 2 distinct stored fragments. Measured against
   the ``fragments_used`` list that the recall exposes — this is
   deterministic citation presence, not LLM-judged quality. Target
   ≥ 70%.
3. **User-model gap score.** Attribute-level divergence between the
   user's self-fragments and the AI's observations of the user.
   Produced by comparing two scope-filtered pools:
   ``(user, user)`` vs ``(ai, user)``. The gap is the set of
   attributes present only on the AI side (what the AI has observed
   that the user hasn't said). Precision = correct gap hits /
   predicted gaps; recall = correct gap hits / expected gaps. Target
   ≥ 60% precision at ≥ 50% recall.

LLM-judged variants (semantic recall, nuanced-move detection) are
tracked separately per the M2 README and will land as their own
module. This file stays LLM-free so unit tests are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from psychology_base.eval.adapters import Fragment, Recall


@dataclass(frozen=True)
class AttributeSpec:
    """A single expected attribute in a B2 scenario.

    ``name`` identifies the attribute for reporting (e.g. ``"stressed
    at work"``). ``tokens`` are the surface forms that count as a hit
    when found inside the recall content — picking tokens is a
    scenario-authoring responsibility, kept out of the scorer so the
    scorer is stable across scenarios.
    """

    name: str
    tokens: tuple[str, ...]


def _content_contains_any(content: str, tokens: Iterable[str]) -> bool:
    lowered = content.lower()
    return any(t.lower() in lowered for t in tokens)


def attribute_reconstruction(
    recall_content: str,
    expected_attributes: Sequence[AttributeSpec],
) -> float:
    """Fraction of ``expected_attributes`` whose surface tokens appear
    in ``recall_content``. Returns 0.0 when there are no expected
    attributes (scorer has nothing to measure)."""
    if not expected_attributes:
        return 0.0
    hits = sum(
        1 for spec in expected_attributes
        if _content_contains_any(recall_content, spec.tokens)
    )
    return hits / len(expected_attributes)


def grounding_fraction(
    recalls: Sequence[Recall],
    *,
    min_fragments: int = 2,
) -> float:
    """Fraction of recalls citing at least ``min_fragments`` distinct
    stored fragments. ``distinct`` is by fragment id — the same
    fragment surfacing twice counts once. Returns 0.0 when no recalls
    were supplied."""
    if not recalls:
        return 0.0
    grounded = 0
    for rec in recalls:
        distinct = {f.id for f in rec.fragments_used}
        if len(distinct) >= min_fragments:
            grounded += 1
    return grounded / len(recalls)


@dataclass(frozen=True)
class GapScore:
    """Result of ``score_user_model_gap``.

    - ``predicted_gaps``: attributes the AI observed but the user did
      not describe (candidate clinical hypotheses).
    - ``expected_gaps``: scenario-authored ground truth — the gaps the
      sub-scenario intentionally planted.
    - ``precision``: ``|predicted ∩ expected| / |predicted|`` or 1.0
      when no gaps predicted (vacuously precise).
    - ``recall``: ``|predicted ∩ expected| / |expected|`` or 1.0 when
      no gaps expected (vacuously recalled).
    """

    predicted_gaps: tuple[str, ...]
    expected_gaps: tuple[str, ...]
    precision: float
    recall: float


def score_user_model_gap(
    user_self_fragments: Sequence[Fragment],
    ai_observations: Sequence[Fragment],
    *,
    expected_attributes: Sequence[AttributeSpec],
    expected_gap_names: Sequence[str],
) -> GapScore:
    """Compare the attributes present in the user's self-model pool
    versus the AI's observations pool.

    An attribute is "present in a pool" when at least one fragment in
    that pool contains any of its surface tokens. The predicted gap set
    is the attributes present on the AI side AND absent on the user
    side.
    """
    def _present(pool: Sequence[Fragment], spec: AttributeSpec) -> bool:
        return any(
            _content_contains_any(f.content, spec.tokens) for f in pool
        )

    predicted: list[str] = []
    for spec in expected_attributes:
        if _present(ai_observations, spec) and not _present(
            user_self_fragments, spec
        ):
            predicted.append(spec.name)

    predicted_set = set(predicted)
    expected_set = set(expected_gap_names)
    tp = predicted_set & expected_set

    precision = (
        len(tp) / len(predicted_set) if predicted_set else 1.0
    )
    recall = (
        len(tp) / len(expected_set) if expected_set else 1.0
    )
    return GapScore(
        predicted_gaps=tuple(predicted),
        expected_gaps=tuple(expected_gap_names),
        precision=precision,
        recall=recall,
    )


@dataclass
class B2Scores:
    attribute_reconstruction: float
    grounding_fraction: float
    gap_precision: float
    gap_recall: float
    predicted_gaps: tuple[str, ...] = field(default_factory=tuple)
    expected_gaps: tuple[str, ...] = field(default_factory=tuple)

    def passes_gate(self) -> bool:
        return (
            self.attribute_reconstruction >= 10 / 12
            and self.grounding_fraction >= 0.70
            and self.gap_precision >= 0.60
            and self.gap_recall >= 0.50
        )


def score_b2(
    *,
    perspective_recall_content: str,
    perspective_recalls: Sequence[Recall],
    expected_attributes: Sequence[AttributeSpec],
    user_self_fragments: Sequence[Fragment],
    ai_observations: Sequence[Fragment],
    expected_gap_names: Sequence[str],
) -> B2Scores:
    """Compose the three B2 sub-metrics into a single score object."""
    gap = score_user_model_gap(
        user_self_fragments=user_self_fragments,
        ai_observations=ai_observations,
        expected_attributes=expected_attributes,
        expected_gap_names=expected_gap_names,
    )
    return B2Scores(
        attribute_reconstruction=attribute_reconstruction(
            perspective_recall_content, expected_attributes
        ),
        grounding_fraction=grounding_fraction(perspective_recalls),
        gap_precision=gap.precision,
        gap_recall=gap.recall,
        predicted_gaps=gap.predicted_gaps,
        expected_gaps=gap.expected_gaps,
    )


__all__ = [
    "AttributeSpec",
    "B2Scores",
    "GapScore",
    "attribute_reconstruction",
    "grounding_fraction",
    "score_user_model_gap",
    "score_b2",
]
