"""M3 — affect-resonance primitives (PAD-2 valence × arousal).

Deliberately small and self-contained. The two primitives are:

- ``pair_affect_score`` — turn a (fragment-affect, cue-affect) pair
  into a resonance score ∈ [0, 1]. 1.0 means identical PAD state,
  0.0 means PAD-diagonal opposites (max euclidean distance in the
  valence × arousal [-1, 1]² square = ``sqrt(8)``).
- ``cue_affect_from_context`` — extract a PAD dict from a recall
  context passed in by a scenario / application. The project does
  not infer cue affect from the cue text in v1; the application
  layer is expected to attach it as context={"affect": {...}}.

Design decisions:

- Fragments without ``valence`` + ``arousal`` keys yield score 0.0
  (not a penalty — the retrieval math at the call site treats a
  zero score as "no boost", not "demote this fragment"). This keeps
  M3 strictly additive over M1/M2 for pre-M3 fragments.
- Scalars are coerced to float and clamped to [-1, 1] on the way in,
  so a scenario that writes ``valence: -0.7`` and another that
  writes ``valence: -10`` behave identically after clamp. Values are
  stored canonically on the fragment, so clamp only applies on read.
- Only valence + arousal in v1. Dominance (PAD third axis) is
  deferred — unstable in short texts, rarely consequential for
  resonance-recall. Readers that want it can swap the helper for
  a 3-vector variant without changing the call-site signature.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional


_PAD_MAX_DIST = math.sqrt(8.0)  # valence × arousal both in [-1, 1]


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _extract_valence_arousal(
    affect: Optional[Mapping[str, object]],
) -> Optional[tuple[float, float]]:
    """Return ``(valence, arousal)`` as clamped floats when both keys
    are present and numeric; otherwise ``None``. Non-numeric values
    (strings, ``None``) are treated as missing — the fragment / cue
    simply does not participate in affect resonance for this recall.
    """
    if not affect:
        return None
    v = affect.get("valence")
    a = affect.get("arousal")
    if not isinstance(v, (int, float)) or not isinstance(a, (int, float)):
        return None
    return _clamp(float(v)), _clamp(float(a))


def pair_affect_score(
    frag_affect: Optional[Mapping[str, object]],
    cue_affect: Optional[Mapping[str, object]],
) -> float:
    """Resonance score for a (fragment, cue) pair.

    Returns ``1.0`` on identical PAD states, scaling linearly down to
    ``0.0`` at the max PAD-2 euclidean distance (``sqrt(8)``, i.e.
    one corner to the diagonal opposite). Returns ``0.0`` when either
    side lacks valence+arousal — "no boost" rather than "demote".
    """
    f = _extract_valence_arousal(frag_affect)
    c = _extract_valence_arousal(cue_affect)
    if f is None or c is None:
        return 0.0
    dist = math.hypot(f[0] - c[0], f[1] - c[1])
    return max(0.0, 1.0 - dist / _PAD_MAX_DIST)


def cue_affect_from_context(
    context: Optional[Mapping[str, object]],
) -> Optional[dict]:
    """Pull the cue-side PAD dict off a recall context.

    Convention: context={"affect": {"valence": v, "arousal": a}} is
    where application / scenario YAML attaches it. Returns ``None``
    when the context is empty, lacks an ``"affect"`` key, or the
    attached value is not a mapping with numeric valence+arousal.
    """
    if not context:
        return None
    raw = context.get("affect")
    if not isinstance(raw, Mapping):
        return None
    va = _extract_valence_arousal(raw)
    if va is None:
        return None
    return {"valence": va[0], "arousal": va[1]}


__all__ = [
    "pair_affect_score",
    "cue_affect_from_context",
]
