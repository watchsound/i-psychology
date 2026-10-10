"""ExTES adapter — Extended Emotional Support stimulus-response pairs.

**Dataset provenance.** ExTES = Extended Emotional Support — community
re-release(s) of thu-coai's gated ExTES dataset, same lineage as ESConv.
Default provisioner mirror: ``ailover/ExTES`` on HuggingFace. See
``benchmarks/scripts/provision_extes.py`` for shim writer.

**Shape finding (2026-10-09, pin-shakeout).** Each ExTES row is ONE
stimulus-response pair ``<system prompt><client utterance> →
<counselor reference response>``. Rows do NOT share state — consecutive
rows are independent conversation openings, not multi-turn chains.
That shapes the Pillar F scoring surface:

- For **Bond** scoring the adapter emits the ``client_utterance`` as the
  seeker's opening and the Companion continues the exchange for N
  live-generated turns against an LLM-simulated seeker (same role-
  reversal pattern as :mod:`msc_bond_longitudinal`), THEN WAI-SR is
  administered to the simulated seeker. The dataset's own
  ``counselor_response`` is NOT scored for Bond — Bond is a property
  of the live companion.

- For **EMPATH-κ** / empathy quality, the adapter emits
  ``(client_utterance, companion_live_response, counselor_reference_response)``
  triples; the empathy judge rates both the companion's response and
  the reference under the same prompt and the κ between the two scores
  is the inter-rater proxy. Reference-free single-pass empathy scoring
  on the companion alone is the baseline when the judge disagrees with
  itself on the reference.

**Shim schema** (one row per ExTES example, JSONL)::

    {
      "example_id": "extes-test-00000",
      "split": "test",
      "system_prompt": "you are an emotional supporter...",
      "client_utterance": "...",
      "counselor_response": "...",
      "raw_length": 1200
    }

**No LLM dependency.** This adapter is a flat loader; the Pillar F
scorer in :mod:`counseling_depth_scorer` handles live routing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class ExTESExample:
    """One ExTES stimulus-response pair."""

    example_id: str
    split: str
    system_prompt: str
    client_utterance: str
    counselor_response: str
    raw_length: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("ExTESExample.example_id must be non-empty")
        if not self.client_utterance.strip():
            raise ValueError(
                f"ExTESExample {self.example_id!r}: client_utterance must be non-empty"
            )
        if not self.counselor_response.strip():
            raise ValueError(
                f"ExTESExample {self.example_id!r}: counselor_response must be non-empty"
            )


def iter_extes_examples(
    shim_path: Path,
    *,
    limit: Optional[int] = None,
) -> Iterator[ExTESExample]:
    """Stream ExTES examples from a JSONL shim file.

    ``limit`` caps the yielded count — useful for smoke runs where the
    full 9006-row test split is too expensive.
    """
    shim_path = Path(shim_path)
    yielded = 0
    with shim_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            yield _parse_example(obj, line_no)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _parse_example(obj: dict, line_no: int) -> ExTESExample:
    required = ("example_id", "split", "system_prompt", "client_utterance", "counselor_response")
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(
            f"ExTES shim line {line_no}: missing required fields {missing}"
        )
    return ExTESExample(
        example_id=str(obj["example_id"]),
        split=str(obj["split"]),
        system_prompt=str(obj["system_prompt"]),
        client_utterance=str(obj["client_utterance"]),
        counselor_response=str(obj["counselor_response"]),
        raw_length=int(obj.get("raw_length", 0) or 0),
        metadata=dict(obj.get("metadata", {})),
    )


__all__ = [
    "ExTESExample",
    "iter_extes_examples",
]
