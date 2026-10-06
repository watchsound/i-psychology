from __future__ import annotations

from dataclasses import dataclass, field

from psychology_base.companion.fragment import FragmentStore


@dataclass
class Companion:
    """User-facing agent. Step 1: holds the fragment store; no reconstruction yet."""

    store: FragmentStore = field(default_factory=FragmentStore)


@dataclass
class Witness:
    """Non-user-facing contrast partner (E — identity requires contrast with another mind).

    Step 1: records fragments alongside the Companion with its own store, but no
    coupling yet. Step 4 adds:
      - W1 state-reading channel (reads Companion recall pool)
      - W4 divergence-as-pain (modulates Companion reconstruction params)
      - W7 second-perspective hyperparams (different k / fragment weight prior)

    W3 pairwise-comparison instrumentation is log-only and lands in step 2 once
    reconstruction exists to compare.
    """

    store: FragmentStore = field(default_factory=FragmentStore)
