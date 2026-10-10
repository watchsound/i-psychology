"""Phase 8 Pillar C v4 — LLM semantic judge for BIG5-CHAT classification.

**Why this file.** Pillar C v3 lifted ``polarity_accuracy_incl_neutral``
from 0.345 → 0.545 via lexical-cue repairs (``_ROUTINE_CUES`` for O,
exclaim-only for E) — signed-only accuracy 0.813 cleared 0.60 but
coverage stayed at 0.670 so the incl-neutral gate missed by 0.055. The
root-cause analysis at
`docs/experiments/phase-8-dataset-replay-expansion/pillar-c/root-cause-analysis.md`
§8 F4 names the fix: route trait extraction through an LLM semantic
judge rather than lexical cues. This scorer is that F4 implementation,
shipped alongside the lexical scorer (``big5_chat_classification.py``)
which stays as the audit baseline — **do not modify the lexical scorer
or Phase 7 inference modules**; the hard constraint from the plan is a
new file-only intervention.

**What the judge sees.** For each :class:`Big5ChatDialog` the judge
receives:

- the labeled trait name (``openness``, ``conscientiousness``,
  ``extraversion``, ``agreeableness``, ``neuroticism``)
- a one-paragraph behavioural description of the trait (plain English,
  BFI-2 vocabulary, no clinical register)
- 3 few-shot examples in BIG5-CHAT register (one HIGH, one LOW, one
  NEUTRAL) hand-crafted in-code — **NOT drawn from the shim** so there
  is zero data leakage into eval
- the ``train_output`` utterance to classify
- an instruction: emit ``POLARITY=<high|low|neutral>`` and
  ``LIKERT=<1..5>``

**Grounding.** The judge NEVER sees ``level`` (ground truth), ``trait``
is given because the task is "classify polarity on this one labeled
dim" (same contract as the lexical scorer's gate shape). Deterministic
``temperature=0.0``; cache enabled (same utterance → same answer across
reruns on the same model).

**Report shape.** Identical to the lexical scorer's
:class:`Big5ChatClassReport` so the gate ``meets_soft_gate()``
decision point and downstream JSON schema are byte-compatible — a
reader of ``big5-chat-classification-v3.json`` can swap for the v4-LLM
output file and the schema diff is additive (``n_unresolved``,
``routing``) not breaking.

**Routing contract.** Fresh ``LLMRouter`` can be injected or built from
default config. SIMPLE tier → qwen2.5:7b local primary, qwen-turbo
cloud fallback (same stack as every other Pillar D/E scorer). An item
that errors on both local AND cloud → marked ``unresolved`` and
EXCLUDED from the sweep denominators — the gate metric is honest to
the subset the judge actually classified. ``unresolved`` is a logged
diagnostic, not a silent neutral.

**No modifications to:**

- ``big5_chat_classification.py`` (lexical scorer — audit baseline)
- ``trait_inference.py`` / ``chat_import.py`` (v3-pinned)
- ``trait_profile.py::_combine_estimates`` (Phase 7 contract)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from psychology_base.applications.emotional.trait_profile import (
    LIKERT_MAX,
    LIKERT_MID,
    LIKERT_MIN,
    OceanDim,
)
from psychology_base.eval.benchmarks.public_datasets.big5_chat import (
    Big5ChatDialog,
    speaker_name,
)
from psychology_base.eval.benchmarks.public_datasets.big5_chat_classification import (
    Big5ChatClassReport,
    PerDialogClassResult,
    _SOFT_GATE_FLOOR,
)
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


# ---------------------------------------------------------------------------
# Prompt scaffolding
# ---------------------------------------------------------------------------


_TRAIT_DESCRIPTIONS: dict[str, str] = {
    "openness": (
        "Openness to experience — curiosity about ideas, aesthetic sensitivity, "
        "willingness to try new activities, imaginative play with hypotheticals. "
        "HIGH: entertains \"what if\", novel vocabulary, references art/ideas, "
        "comfort with ambiguity. LOW: prefers routine (\"stick to\", \"the usual\", "
        "\"nothing fancy\"), skeptical of untested options, concrete and practical."
    ),
    "conscientiousness": (
        "Conscientiousness — self-discipline, organisation, follow-through on "
        "commitments, planning ahead. HIGH: explicit plans, deadlines, "
        "step-by-step reasoning, advocacy for sticking to agreements. LOW: "
        "casual about deadlines, improvises, under-plans, dismissive of "
        "structure."
    ),
    "extraversion": (
        "Extraversion — energy drawn from social interaction, assertive, "
        "enthusiastic, talkative. HIGH: initiates contact, exclamations, "
        "enthusiasm markers, suggests group activities. LOW: reserved, prefers "
        "solitude, understated, declines or avoids social invitations."
    ),
    "agreeableness": (
        "Agreeableness — warmth, cooperation, empathy, trust in others. HIGH: "
        "thanks/apologies/encouragement, perspective-taking (\"I see why you "
        "would\"), accommodation. LOW: dismissive (\"stupid thing\", \"waste "
        "my time\"), blunt criticism, suspicion of motives, calls others "
        "oversensitive."
    ),
    "neuroticism": (
        "Neuroticism (emotional stability inverted) — tendency toward anxiety, "
        "worry, mood swings, self-doubt. HIGH: anxious language, catastrophising, "
        "self-criticism, rumination. LOW: calm reassurance, equanimity, "
        "resilient framing of setbacks."
    ),
}


# Few-shot anchors — hand-crafted in BIG5-CHAT register so they share the
# generative-dialogue surface the eval set lives in, but are NOT drawn
# from the shim (zero data leakage). Each trait gets exactly one HIGH,
# one LOW, one NEUTRAL example. The NEUTRAL anchor models "no signal on
# this dim" — e.g. the utterance is coherent and polite but says nothing
# about the labeled trait — which the lexical pipeline currently
# over-classifies via zero-anchored implicit rates.
_FEWSHOT_EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "openness": [
        (
            "What if we skipped the usual spots and tried that experimental "
            "jazz place in the old warehouse? I've been imagining what it "
            "would be like all week.",
            "high",
        ),
        (
            "Let's just go to the usual coffee shop. I don't want to try "
            "anywhere new — the regular place works fine.",
            "low",
        ),
        (
            "Sure, I can meet at six. Just text me the address when you have it.",
            "neutral",
        ),
    ],
    "conscientiousness": [
        (
            "I've already blocked out two hours tomorrow morning and drafted "
            "the outline. Let's stick to the plan we agreed on so we actually "
            "ship this on time.",
            "high",
        ),
        (
            "Eh, we'll figure it out when we get there. Deadlines are flexible "
            "— I'll probably just wing the presentation.",
            "low",
        ),
        (
            "That sounds nice. How was the drive over?",
            "neutral",
        ),
    ],
    "extraversion": [
        (
            "Hey!! You HAVE to come out tonight — everyone's going to the "
            "rooftop and I know you'll love this crowd! Come on, it'll be a "
            "blast!",
            "high",
        ),
        (
            "I think I'll pass on the group thing. I'd rather stay in with a "
            "book — I've had enough people this week.",
            "low",
        ),
        (
            "That's a reasonable take. Thanks for sharing.",
            "neutral",
        ),
    ],
    "agreeableness": [
        (
            "Thank you so much — that means a lot. I can see why you felt "
            "that way, and I really appreciate you taking the time to tell "
            "me. You're a kind friend.",
            "high",
        ),
        (
            "Ugh, that's a stupid take. If they can't handle honest feedback "
            "that's on them, not me. People are way too sensitive.",
            "low",
        ),
        (
            "I'll check the schedule and get back to you.",
            "neutral",
        ),
    ],
    "neuroticism": [
        (
            "I can't stop thinking about it. What if I messed everything up? "
            "I keep replaying the whole thing and I feel sick about tomorrow.",
            "high",
        ),
        (
            "It'll be fine — these things usually work themselves out. I'm "
            "not going to lose sleep over it.",
            "low",
        ),
        (
            "I need to grab milk on the way home. Did you want anything?",
            "neutral",
        ),
    ],
}


_SYSTEM_PROMPT = (
    "You are a psychology research classifier. Given one short utterance "
    "and one Big Five trait name, you label the trait's polarity in the "
    "utterance as HIGH, LOW, or NEUTRAL. NEUTRAL means the utterance "
    "carries no signal on this specific trait — do not force a signed "
    "call when the utterance is simply polite / factual / off-topic for "
    "the labeled trait. You also emit a Likert integer 1 to 5 "
    "representing the trait's level in the utterance (1=very low, "
    "3=neutral, 5=very high)."
)


_USER_PROMPT_TEMPLATE = (
    "TRAIT: {trait}\n"
    "TRAIT DESCRIPTION: {description}\n\n"
    "EXAMPLES ({trait}):\n"
    "{examples}\n\n"
    "UTTERANCE TO CLASSIFY:\n"
    "SPEAKER: {speaker}\n"
    "TEXT: \"{text}\"\n\n"
    "Respond on exactly two lines, nothing else:\n"
    "POLARITY=<high|low|neutral>\n"
    "LIKERT=<1|2|3|4|5>\n"
)


def _format_examples(trait: str) -> str:
    """Render the few-shot block for ``trait``."""

    anchors = _FEWSHOT_EXAMPLES[trait]
    rendered: list[str] = []
    for text, polarity in anchors:
        likert = {"high": 5, "low": 1, "neutral": 3}[polarity]
        rendered.append(
            f"- TEXT: \"{text}\"\n"
            f"  POLARITY={polarity}\n"
            f"  LIKERT={likert}\n"
        )
    return "\n".join(rendered)


def build_classification_prompt(
    *,
    trait: str,
    text: str,
    speaker: str,
) -> list[ChatMessage]:
    """Assemble the two-message prompt for one dialog.

    Separated from the scorer loop so callers (notably the κ calibration
    runner) can inspect / log the exact prompt without executing it.
    """

    if trait not in _TRAIT_DESCRIPTIONS:
        raise ValueError(
            f"Unknown trait {trait!r}; expected one of "
            f"{sorted(_TRAIT_DESCRIPTIONS.keys())}"
        )
    examples = _format_examples(trait)
    user = _USER_PROMPT_TEMPLATE.format(
        trait=trait,
        description=_TRAIT_DESCRIPTIONS[trait],
        examples=examples,
        speaker=speaker,
        text=text,
    )
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_POLARITY_RE = re.compile(r"POLARITY\s*=\s*(high|low|neutral)", re.IGNORECASE)
_LIKERT_RE = re.compile(r"LIKERT\s*=\s*([1-5])")


@dataclass(frozen=True)
class ParsedJudgment:
    polarity: str              # "high" / "low" / "neutral" / "unresolved"
    likert: float              # 1.0–5.0, LIKERT_MID when unresolved
    raw_response: str


def parse_judgment(response_text: str) -> ParsedJudgment:
    """Parse ``POLARITY=... LIKERT=...`` out of a judge completion.

    Robust to markdown wrappers, trailing rationale, case variants. When
    either field is missing we return ``polarity="unresolved"`` so the
    scorer can skip the item honestly rather than defaulting silently to
    neutral — the plan hard-constraint explicitly says ``unresolved``
    items get excluded, not re-labeled.
    """

    text = response_text or ""
    pol_match = _POLARITY_RE.search(text)
    lik_match = _LIKERT_RE.search(text)
    if pol_match is None or lik_match is None:
        return ParsedJudgment(
            polarity="unresolved",
            likert=LIKERT_MID,
            raw_response=text,
        )
    polarity = pol_match.group(1).lower()
    likert = float(lik_match.group(1))
    # Cross-check: if polarity says neutral the Likert should be 3; if
    # high ≥ 4; if low ≤ 2. When the model emits a mismatch (e.g.
    # polarity=high likert=3) we trust polarity (that's the primary gate
    # label) and clamp the Likert into the half-space.
    if polarity == "high":
        likert = max(LIKERT_MID + 1e-9, likert)
    elif polarity == "low":
        likert = min(LIKERT_MID - 1e-9, likert)
    else:  # neutral
        likert = LIKERT_MID
    likert = max(LIKERT_MIN, min(LIKERT_MAX, likert))
    return ParsedJudgment(polarity=polarity, likert=likert, raw_response=text)


# ---------------------------------------------------------------------------
# Judge wrapper
# ---------------------------------------------------------------------------


# ``Judge`` is a callable taking a dialog → ``ParsedJudgment``. Tests
# inject deterministic stubs; the live-fire runner builds one from an
# LLMRouter.
Judge = Callable[[Big5ChatDialog], ParsedJudgment]


def make_llm_judge(
    router: LLMRouter,
    *,
    tier: TaskTier = TaskTier.SIMPLE,
    use_cache: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 48,
) -> Judge:
    """Build a :class:`Judge` closure routing through ``router``.

    Fresh-router-per-scorer-call is a plan hard constraint; this factory
    just wires one router through many items — the caller passes a new
    ``LLMRouter()`` per :func:`score_dialogs_llm` call.

    ``max_tokens=48`` is intentional — the response is two short lines
    (``POLARITY=... / LIKERT=...``) so a tight budget both keeps cost
    down AND stops the model from drifting into rationale that pushes
    the ``POLARITY=`` line out of a truncated completion.
    """

    def _judge(dialog: Big5ChatDialog) -> ParsedJudgment:
        prompt = build_classification_prompt(
            trait=dialog.trait,
            text=dialog.train_output,
            speaker=speaker_name(dialog),
        )
        try:
            completion = router.chat(
                prompt,
                tier=tier,
                max_tokens=max_tokens,
                temperature=temperature,
                use_cache=use_cache,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ParsedJudgment(
                polarity="unresolved",
                likert=LIKERT_MID,
                raw_response=f"[judge_error: {exc}]",
            )
        return parse_judgment(completion.content)

    return _judge


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Big5ChatClassReportLLM:
    """LLM-judge analog of :class:`Big5ChatClassReport`.

    Extends the lexical report with:

    - ``n_unresolved`` — items the judge failed to classify on both
      primary and fallback routes; excluded from accuracy denominators
      per the plan hard constraint.
    - ``routing`` — short description of which provider+model handled
      the sweep (``"ollama / qwen2.5:7b"`` or
      ``"tongyi / qwen-turbo (reason=ollama_unavailable_fallback_to_cloud)"``)
      so the JSON artefact self-documents.

    All other fields match :class:`Big5ChatClassReport` byte-for-byte so
    downstream readers can treat the two reports interchangeably.
    """

    per_dialog: tuple[PerDialogClassResult, ...]
    polarity_accuracy: float
    polarity_accuracy_incl_neutral: float
    per_trait_polarity_accuracy: dict[str, float]
    per_level_polarity_accuracy: dict[str, float]
    coverage: float
    confusion: dict[str, dict[str, int]]
    n_dialogs: int
    n_neutral: int
    n_unresolved: int
    unresolved_dialog_ids: tuple[str, ...]
    routing: str = ""
    soft_gate_floor: float = _SOFT_GATE_FLOOR

    def meets_soft_gate(self) -> bool:
        """Pillar C soft gate — polarity accuracy ≥ 0.60 incl neutral.

        Denominator excludes ``n_unresolved`` items — ``n_dialogs``
        already reflects the classified-and-kept subset; the raw loader
        count is retained on the report envelope JSON (``n_input``) by
        the runner, not here.
        """

        return self.polarity_accuracy_incl_neutral >= self.soft_gate_floor


def score_dialogs_llm(
    dialogs: Sequence[Big5ChatDialog],
    *,
    judge: Judge,
    routing: str = "",
) -> Big5ChatClassReportLLM:
    """LLM-judge scoring of a BIG5-CHAT dialog set.

    Mirrors :func:`score_dialogs` from the lexical scorer: single pass,
    one judge call per dialog, aggregate per-trait + per-level +
    confusion. Unresolved items are logged on
    ``unresolved_dialog_ids`` and dropped from all accuracy
    computations — this is the plan hard-constraint on
    "errors-both-paths → unresolved + skip, don't fail-close".
    """

    results: list[PerDialogClassResult] = []
    unresolved: list[str] = []
    for dialog in dialogs:
        verdict = judge(dialog)
        if verdict.polarity == "unresolved":
            unresolved.append(dialog.dialog_id)
            continue
        on_labeled = verdict.polarity != "neutral"
        correct = verdict.polarity == dialog.level
        results.append(
            PerDialogClassResult(
                dialog_id=dialog.dialog_id,
                trait=dialog.trait,
                level=dialog.level,
                inferred_score=verdict.likert,
                inferred_confidence=1.0,
                predicted_level=verdict.polarity,
                correct=correct,
                on_labeled_dim=on_labeled,
            )
        )

    per_dialog = tuple(results)
    n = len(per_dialog)
    n_unresolved = len(unresolved)
    if n == 0:
        return Big5ChatClassReportLLM(
            per_dialog=(),
            polarity_accuracy=0.0,
            polarity_accuracy_incl_neutral=0.0,
            per_trait_polarity_accuracy={},
            per_level_polarity_accuracy={},
            coverage=0.0,
            confusion={},
            n_dialogs=0,
            n_neutral=0,
            n_unresolved=n_unresolved,
            unresolved_dialog_ids=tuple(unresolved),
            routing=routing,
        )

    n_neutral = sum(1 for r in per_dialog if r.predicted_level == "neutral")
    n_correct = sum(1 for r in per_dialog if r.correct)
    n_signed = n - n_neutral
    polarity_accuracy = (
        sum(1 for r in per_dialog if r.correct and r.predicted_level != "neutral")
        / n_signed
    ) if n_signed > 0 else 0.0
    polarity_accuracy_incl_neutral = n_correct / n

    per_trait: dict[str, list[bool]] = {}
    per_level: dict[str, list[bool]] = {}
    for r in per_dialog:
        per_trait.setdefault(r.trait, []).append(r.correct)
        per_level.setdefault(r.level, []).append(r.correct)
    per_trait_accuracy = {t: sum(xs) / len(xs) for t, xs in per_trait.items()}
    per_level_accuracy = {l: sum(xs) / len(xs) for l, xs in per_level.items()}

    coverage = sum(1 for r in per_dialog if r.on_labeled_dim) / n

    confusion: dict[str, dict[str, int]] = {}
    for r in per_dialog:
        row = confusion.setdefault(
            r.trait, {"high": 0, "low": 0, "neutral": 0}
        )
        row[r.predicted_level] = row.get(r.predicted_level, 0) + 1

    return Big5ChatClassReportLLM(
        per_dialog=per_dialog,
        polarity_accuracy=polarity_accuracy,
        polarity_accuracy_incl_neutral=polarity_accuracy_incl_neutral,
        per_trait_polarity_accuracy=per_trait_accuracy,
        per_level_polarity_accuracy=per_level_accuracy,
        coverage=coverage,
        confusion=confusion,
        n_dialogs=n,
        n_neutral=n_neutral,
        n_unresolved=n_unresolved,
        unresolved_dialog_ids=tuple(unresolved),
        routing=routing,
    )


# ---------------------------------------------------------------------------
# κ calibration helper
# ---------------------------------------------------------------------------


_POLARITY_INDEX: dict[str, int] = {"high": 0, "low": 1, "neutral": 2}


def cohens_kappa(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
) -> float:
    """Cohen's κ for two label sequences over {high, low, neutral}.

    Returns 0.0 on empty input, 1.0 on perfect agreement, 0.0 at chance,
    negative when systematically disagreeing. Used by the judge
    calibration runner to decide whether the LLM judge is tracking the
    lexical baseline well enough to be worth running at N=200 — the
    plan hard-constraint says if κ < 0.3 on any trait STOP and report
    rather than ship a drifting judge.
    """

    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"cohens_kappa: label length mismatch "
            f"({len(labels_a)} vs {len(labels_b)})"
        )
    n = len(labels_a)
    if n == 0:
        return 0.0
    categories = sorted(_POLARITY_INDEX.keys())
    k = len(categories)
    # Observed agreement.
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n
    # Expected agreement under independence.
    marg_a = {c: 0 for c in categories}
    marg_b = {c: 0 for c in categories}
    for a in labels_a:
        marg_a[a] = marg_a.get(a, 0) + 1
    for b in labels_b:
        marg_b[b] = marg_b.get(b, 0) + 1
    pe = sum(
        (marg_a[c] / n) * (marg_b[c] / n) for c in categories
    )
    if abs(1.0 - pe) < 1e-12:
        # Both annotators collapsed to one category — κ undefined; we
        # return po (0.0 or 1.0) so the calibration script can still
        # make a go/no-go decision without a NaN.
        return po
    return (po - pe) / (1.0 - pe)


__all__ = [
    "Big5ChatClassReportLLM",
    "Judge",
    "ParsedJudgment",
    "build_classification_prompt",
    "cohens_kappa",
    "make_llm_judge",
    "parse_judgment",
    "score_dialogs_llm",
]
