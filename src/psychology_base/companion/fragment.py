"""Fragment store with S1 hard byte cap + S5 covenant pins.

Step 1 was append-only. Step 3 adds:
    S1 — approximate byte accounting (len(content) per fragment, constant per
         edge); when total bytes exceed cap, evict the weakest non-covenant
         fragment. Weakness = sum of incident edge weights (lowest first);
         ties broken by oldest timestamp.
    S5 — fragments tagged "covenant" are immune from eviction (Principle D's
         explicit carve-out — covenant rules do not decay).

The eviction decision reads edge weights from an EdgeStore passed in by the
system; the store itself does not own edges. On eviction, the EdgeStore's
`remove_fragment` is called so dangling edges don't accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from psychology_base.companion.edges import EdgeStore
from psychology_base.eval.adapters import Fragment


COVENANT_TAG = "covenant"
BYTES_PER_EDGE = 32  # constant estimate; edge payload is id-pair + weight float


def is_covenant(fragment: Fragment) -> bool:
    return COVENANT_TAG in fragment.tags


def fragment_bytes(fragment: Fragment) -> int:
    return len(fragment.content.encode("utf-8"))


@dataclass
class FragmentStore:
    """Byte-capped fragment container with covenant-pin eviction policy.

    `byte_cap=None` disables eviction (step-1 behaviour; the step-2 scenario
    stays well under any realistic cap so step-2 metrics are unaffected).
    """

    _fragments: list[Fragment] = field(default_factory=list)
    byte_cap: Optional[int] = None
    # Phase 3 E1 — optional strength override. When set, eviction's
    # weakness ranking reads this callable instead of the Hebbian
    # edge-sum. Signature matches `AgingPolicy.strength(fid)`. Phase 2
    # default (None) keeps Hebbian strength intact; `_evicted_callback`
    # fires per eviction so the aging policy can purge its own state.
    strength_fn: Optional[Callable[[str], float]] = None
    evicted_callback: Optional[Callable[[str], None]] = None

    def add(self, fragment: Fragment, edges: Optional[EdgeStore] = None) -> None:
        """Append fragment, then enforce byte cap by evicting weakest non-covenant.
        `edges` is required when `byte_cap` is set (eviction reads edge weights
        for the weakness ranking)."""
        self._fragments.append(fragment)
        if self.byte_cap is not None:
            self._enforce_cap(edges)

    def all(self) -> list[Fragment]:
        return list(self._fragments)

    def by_id(self, fid: str) -> Optional[Fragment]:
        for f in self._fragments:
            if f.id == fid:
                return f
        return None

    def by_substring(self, needle: str) -> list[Fragment]:
        n = needle.lower()
        return [f for f in self._fragments if n in f.content.lower()]

    def total_bytes(self, edges: Optional[EdgeStore] = None) -> int:
        frag_bytes = sum(fragment_bytes(f) for f in self._fragments)
        edge_bytes = (len(edges) if edges is not None else 0) * BYTES_PER_EDGE
        return frag_bytes + edge_bytes

    def __len__(self) -> int:
        return len(self._fragments)

    def _enforce_cap(self, edges: Optional[EdgeStore]) -> None:
        """Evict weakest non-covenant fragments until total bytes <= cap. If
        only covenant fragments remain and we're still over cap, stop — covenant
        pins override the cap (deliberate: Principle D)."""
        if self.byte_cap is None:
            return
        while self.total_bytes(edges) > self.byte_cap:
            victim = self._weakest_evictable(edges)
            if victim is None:
                return  # all remaining are covenant pins
            self._fragments.remove(victim)
            if edges is not None:
                edges.remove_fragment(victim.id)
            if self.evicted_callback is not None:
                self.evicted_callback(victim.id)

    def _weakest_evictable(self, edges: Optional[EdgeStore]) -> Optional[Fragment]:
        candidates = [f for f in self._fragments if not is_covenant(f)]
        if not candidates:
            return None

        if self.strength_fn is not None:
            policy_strength = self.strength_fn

            def strength(f: Fragment) -> float:
                return policy_strength(f.id)

        else:

            def strength(f: Fragment) -> float:
                if edges is None:
                    return 0.0
                return sum(
                    edges.get(f.id, g.id) for g in self._fragments if g.id != f.id
                )

        return min(candidates, key=lambda f: (strength(f), f.timestamp))
