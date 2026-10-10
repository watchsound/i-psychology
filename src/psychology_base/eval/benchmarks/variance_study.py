"""Phase 4a §6d — Multi-trial variance study for benchmark monitors.

The Phase 3 close-out ships a LoCoMo monitor floor of 0.514 as a
*single-trial* point estimate. Pilot B3 reporting needs a
confidence-bounded floor so that regressions can be distinguished from
LLM-induced noise. This module converts any benchmark-run callable
into a multi-trial study: N trials over the same instance(s) with the
same ``system_factory``, then aggregate statistics with 95% CI.

Scope. Not tied to LoCoMo specifically — any `run_fn(instance,
system_factory=..., **kwargs) -> InstanceResult` can be studied. This
is intentional so later phases can reuse the harness for B1/B2/B3
monitors, EpBench variance, or custom benchmarks the pilot adds.

Variance sources captured:

- LLM reader / judge stochasticity (temperature > 0 or non-cached calls).
- Any random seed in the system under test (e.g. S2 capacity tie-break).

Variance sources NOT captured:

- Instance-level variability — N-trial over a single instance yields
  within-instance variance, not between-instance. For between-instance,
  run the study over multiple instances and compare means.

Statistics. Each per-question score is averaged across trials; the
instance-level mean + stderr + 95% CI are reported. Stderr uses the
sample standard deviation (unbiased, n-1 denominator) divided by
sqrt(n_trials). The 95% CI uses the normal approximation (`±1.96·se`)
which is adequate for n ≥ 10; smaller N should be reported with the
caveat flagged in `VarianceReport.ci_warning`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from psychology_base.eval.benchmarks.locomo import InstanceResult, QuestionResult


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerQuestionVariance:
    question_id: str
    category: str
    scores: tuple[float, ...]
    mean: float
    stderr: float


@dataclass(frozen=True)
class VarianceReport:
    """Aggregate variance over N trials of a benchmark run."""

    instance_id: str
    n_trials: int
    per_question: tuple[PerQuestionVariance, ...]
    instance_mean: float
    instance_stderr: float
    """Standard error of the mean over trials (sample stddev / sqrt N)."""

    ci95_lower: float
    ci95_upper: float
    """95% confidence interval using the normal approximation."""

    per_category_mean: Mapping[str, float] = field(default_factory=dict)
    per_category_stderr: Mapping[str, float] = field(default_factory=dict)
    ci_warning: Optional[str] = None
    """Non-None when the N-trial count is too small for the normal
    approximation to be defensible. Pilot reports should surface the
    warning rather than hide it."""

    def meets_confidence_floor(self, floor: float = 0.514) -> bool:
        """True when the 95% CI lower bound ≥ `floor`.

        Default `floor=0.514` matches the Phase 1.5 single-trial point
        estimate — i.e. the regression gate the pilot team can defend
        once CI is bounded. Callers set their own floor for other
        monitors.
        """
        return self.ci95_lower >= floor


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------


RunFn = Callable[..., InstanceResult]


def _sample_stddev(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def _ci95(mean: float, stderr: float) -> tuple[float, float]:
    half = 1.96 * stderr
    return max(0.0, mean - half), min(1.0, mean + half)


def run_multi_trial(
    *,
    instance: Any,
    system_factory: Callable[[], Any],
    run_fn: RunFn,
    n_trials: int,
    run_kwargs: Optional[Mapping[str, Any]] = None,
) -> VarianceReport:
    """Run ``run_fn`` ``n_trials`` times against the same instance.

    Each trial rebuilds the system via ``system_factory`` to isolate
    trial-level state. ``run_fn`` must accept ``instance=`` and
    ``system_factory=`` kwargs plus whatever is in ``run_kwargs``.

    Variance is only meaningful if the run contains a stochastic path
    (LLM temperature > 0, or disabled cache with a non-deterministic
    judge). The harness itself is deterministic — if the run is
    deterministic too, every trial will return identical scores and
    stderr will be zero (the report is honest about that).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be ≥ 1.")
    run_kwargs = dict(run_kwargs or {})

    trials: list[InstanceResult] = []
    for _ in range(n_trials):
        res = run_fn(instance=instance, system_factory=system_factory, **run_kwargs)
        trials.append(res)

    # Align per-question across trials by question_id. All trials must
    # cover the same question set; mismatch is a hard error — pilot
    # should not silently drop questions.
    first_qids = tuple(r.question_id for r in trials[0].per_question)
    for t_idx, trial in enumerate(trials[1:], start=1):
        trial_qids = tuple(r.question_id for r in trial.per_question)
        if trial_qids != first_qids:
            raise ValueError(
                f"Trial {t_idx} question_ids differ from trial 0; "
                "variance study requires identical question sets across trials."
            )

    per_question: list[PerQuestionVariance] = []
    by_cat_scores: dict[str, list[list[float]]] = {}
    instance_trial_means: list[float] = []

    for trial in trials:
        instance_trial_means.append(trial.mean_score)

    for q_idx, qid in enumerate(first_qids):
        scores = tuple(trial.per_question[q_idx].score for trial in trials)
        mean = sum(scores) / n_trials
        se = _sample_stddev(scores) / math.sqrt(n_trials) if n_trials > 1 else 0.0
        category = trials[0].per_question[q_idx].category
        per_question.append(
            PerQuestionVariance(
                question_id=qid,
                category=category,
                scores=scores,
                mean=mean,
                stderr=se,
            )
        )
        by_cat_scores.setdefault(category or "unknown", []).append(list(scores))

    instance_mean = sum(instance_trial_means) / n_trials
    instance_se = (
        _sample_stddev(instance_trial_means) / math.sqrt(n_trials)
        if n_trials > 1 else 0.0
    )
    lo, hi = _ci95(instance_mean, instance_se)

    per_cat_mean: dict[str, float] = {}
    per_cat_se: dict[str, float] = {}
    for cat, matrix in by_cat_scores.items():
        # matrix: list of trials of per-question score lists in that category.
        # Trial-level per-category mean, then variance across trials.
        trial_means = [
            sum(row[t] for row in matrix) / len(matrix)
            for t in range(n_trials)
        ]
        per_cat_mean[cat] = sum(trial_means) / n_trials
        per_cat_se[cat] = (
            _sample_stddev(trial_means) / math.sqrt(n_trials)
            if n_trials > 1 else 0.0
        )

    ci_warning = None
    if n_trials < 10:
        ci_warning = (
            f"N={n_trials} trials — normal-approximation 95% CI may be unreliable; "
            "pilot reports should quote the raw per-trial scores alongside the CI."
        )

    return VarianceReport(
        instance_id=trials[0].instance_id,
        n_trials=n_trials,
        per_question=tuple(per_question),
        instance_mean=instance_mean,
        instance_stderr=instance_se,
        ci95_lower=lo,
        ci95_upper=hi,
        per_category_mean=per_cat_mean,
        per_category_stderr=per_cat_se,
        ci_warning=ci_warning,
    )


__all__ = [
    "PerQuestionVariance",
    "VarianceReport",
    "RunFn",
    "run_multi_trial",
]
