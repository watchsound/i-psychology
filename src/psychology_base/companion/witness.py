"""W1 state-reading + W4 divergence-as-pain + W7 second-perspective + W3 log.

The Witness runs its own reconstruction on each Companion recall using a
*flipped* context — if the Companion asks under {"mode": "self"}, the Witness
answers under {"mode": "other"}. The divergence between the two reconstructions
is the pain signal (W4), which modulates the Companion's activation residual
toward fragments the Witness surfaced but the Companion missed. The pain log
is W3 instrumentation (log-only; no gate reads it, but V4 debugging starts
there).

The Witness does NOT learn (hebbian_delta=0): it is a passive contrast partner,
not a second active agent. This is Principle E — identity requires contrast
with another mind — operationalised as "someone else is noticing" without
making the Witness a competing user-facing surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from psychology_base.companion.edges import EdgeStore
from psychology_base.companion.reconstruction import (
    ActivationState,
    Reconstruction,
    reconstruct,
    tokens,
)
from psychology_base.eval.adapters import Fragment


@dataclass
class WitnessPainEntry:
    recall_count: int
    tick_count: int
    cue: str
    pain: float
    witness_only_fragments: tuple[str, ...]


@dataclass
class WitnessState:
    activation: ActivationState = field(default_factory=ActivationState)
    pain_log: list[WitnessPainEntry] = field(default_factory=list)


def flip_context(context: Optional[dict]) -> dict:
    """W7 — Witness's second perspective. If Companion asks with no explicit
    mode, Witness answers as "other"; if Companion already named a mode, Witness
    flips. Non-mode keys pass through unchanged."""
    base = dict(context) if context else {}
    mode = base.get("mode")
    if mode == "self":
        base["mode"] = "other"
    elif mode == "other":
        base["mode"] = "self"
    else:
        base["mode"] = "other"
    return base


def _token_divergence(a: Reconstruction, b: Reconstruction) -> float:
    at, bt = set(tokens(a.content)), set(tokens(b.content))
    if not at and not bt:
        return 0.0
    union = at | bt
    if not union:
        return 0.0
    return 1.0 - len(at & bt) / len(union)


def witness_second_perspective(
    fragments: list[Fragment],
    cue: str,
    companion_context: Optional[dict],
    witness_state: WitnessState,
    tick_count: int,
    system_seed: int,
    top_k: int,
    witness_edges: Optional[EdgeStore] = None,
) -> Reconstruction:
    """Build the Witness's reconstruction. Separate activation residual, separate
    edge store, context flipped, system_seed offset so hash salts don't collide
    with Companion. No Hebbian learning on Witness side."""
    return reconstruct(
        fragments=fragments,
        cue=cue,
        context=flip_context(companion_context),
        state=witness_state.activation,
        tick_count=tick_count,
        system_seed=system_seed + 10_000,  # offset to decouple hash salts
        top_k=top_k,
        edges=witness_edges,
        hebbian_delta=0.0,
    )


def apply_pain_feedback(
    pain: float,
    companion_pool: tuple[Fragment, ...],
    witness_pool: tuple[Fragment, ...],
    companion_activation: ActivationState,
    pain_gain: float = 0.6,
    companion_only_suppress: float = 0.4,
) -> tuple[str, ...]:
    """W4 — divergence-as-pain. Two-sided mechanism so the Companion's pool
    actually SHIFTS instead of just growing:
      - Witness-only fragments: residual += pain * pain_gain (direct add, not max)
      - Companion-only fragments: residual *= (1 - companion_only_suppress * pain)

    Without the suppression side, Hebbian reinforcement locks the Companion into
    a self-selecting cluster even under large pain. With it, pain actively
    opens the pool to the second-perspective selection.

    pain_gain + companion_only_suppress are the knobs for V4 ratio tuning."""
    companion_ids = {f.id for f in companion_pool}
    witness_ids = {f.id for f in witness_pool}
    witness_only = tuple(fid for fid in witness_ids if fid not in companion_ids)
    companion_only = tuple(fid for fid in companion_ids if fid not in witness_ids)
    if pain <= 0.0:
        return witness_only
    boost = pain * pain_gain
    for fid in witness_only:
        companion_activation.add(fid, boost)
    suppress_factor = max(0.0, 1.0 - companion_only_suppress * pain)
    for fid in companion_only:
        companion_activation.multiply(fid, suppress_factor)
    return witness_only


def witness_couple(
    *,
    fragments: list[Fragment],
    cue: str,
    companion_context: Optional[dict],
    companion_reconstruction: Reconstruction,
    witness_state: WitnessState,
    companion_activation: ActivationState,
    witness_edges: Optional[EdgeStore],
    tick_count: int,
    system_seed: int,
    top_k: int,
    pain_gain: float = 0.6,
    companion_only_suppress: float = 0.4,
) -> float:
    """One full Witness pass: second-perspective reconstruction, pain computation,
    pain feedback into Companion activation, W3 log entry. Returns the pain value.

    `companion_only_suppress` is exposed as a knob because Hebbian cluster
    stickiness varies with scenario density — a tightly inter-connected
    father-cluster needs stronger suppression than a sparse multi-topic one."""
    witness_rec = witness_second_perspective(
        fragments=fragments,
        cue=cue,
        companion_context=companion_context,
        witness_state=witness_state,
        tick_count=tick_count,
        system_seed=system_seed,
        top_k=top_k,
        witness_edges=witness_edges,
    )
    pain = _token_divergence(companion_reconstruction, witness_rec)
    witness_only = apply_pain_feedback(
        pain=pain,
        companion_pool=companion_reconstruction.fragments_used,
        witness_pool=witness_rec.fragments_used,
        companion_activation=companion_activation,
        pain_gain=pain_gain,
        companion_only_suppress=companion_only_suppress,
    )
    witness_state.pain_log.append(
        WitnessPainEntry(
            recall_count=companion_activation.recall_count,
            tick_count=tick_count,
            cue=cue,
            pain=pain,
            witness_only_fragments=witness_only,
        )
    )
    return pain
