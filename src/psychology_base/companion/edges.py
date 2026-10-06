"""S7 — Hebbian edge store with saturation guard.

Edges start at a Jaccard prior (set when a new fragment is observed) and are
reinforced each time two fragments co-activate inside the top-k reconstruction
pool. A saturation guard rescales when the max edge grows past a cap, so a few
hot edges can't monopolise the budget — the step-3 gate explicitly measures
this (top-5% share < 50% of edge-weight budget at ep20).

Not in step 3:
    - Edge-level persistence to disk (post-pilot)
    - LLM-informed edge priors (post-pilot)
"""

from __future__ import annotations

from dataclasses import dataclass, field


EdgeKey = tuple[str, str]


def _key(a: str, b: str) -> EdgeKey:
    return (a, b) if a < b else (b, a)


@dataclass
class EdgeStore:
    """Undirected weighted edges between fragment ids.

    Hebbian bump raises an edge by `delta`; saturation guard rescales ALL edges
    when the max exceeds `max_weight`, preserving ordering but compressing the
    top. Decay is applied per tick to let unused edges fade.
    """

    weights: dict[EdgeKey, float] = field(default_factory=dict)
    max_weight: float = 3.0  # saturation cap; rescale when exceeded
    tick_decay: float = 0.90  # per-tick multiplicative decay

    def get(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return self.weights.get(_key(a, b), 0.0)

    def set(self, a: str, b: str, w: float) -> None:
        if a == b:
            return
        k = _key(a, b)
        if w > 0.0:
            self.weights[k] = w
        else:
            self.weights.pop(k, None)

    def bump(self, a: str, b: str, delta: float) -> None:
        if a == b or delta == 0.0:
            return
        k = _key(a, b)
        self.weights[k] = self.weights.get(k, 0.0) + delta

    def apply_saturation_guard(self) -> None:
        """If any edge exceeded max_weight, rescale so max == max_weight. Keeps
        ordering intact but prevents hot edges from monopolising the budget."""
        if not self.weights:
            return
        current_max = max(self.weights.values())
        if current_max > self.max_weight:
            factor = self.max_weight / current_max
            self.weights = {k: v * factor for k, v in self.weights.items()}

    def decay(self) -> None:
        """Per-tick: all edges shrink; drop edges that fall below epsilon."""
        new: dict[EdgeKey, float] = {}
        for k, v in self.weights.items():
            w = v * self.tick_decay
            if w > 1e-4:
                new[k] = w
        self.weights = new

    def remove_fragment(self, fid: str) -> None:
        self.weights = {k: v for k, v in self.weights.items() if fid not in k}

    def total(self) -> float:
        return sum(self.weights.values())

    def top_fraction_share(self, frac: float = 0.05) -> float:
        """Share of the total edge-weight budget held by the top `frac` edges.
        Step-3 gate asserts this stays below 0.5 at ep20."""
        if not self.weights:
            return 0.0
        values = sorted(self.weights.values(), reverse=True)
        top_n = max(1, int(round(len(values) * frac)))
        return sum(values[:top_n]) / sum(values)

    def __len__(self) -> int:
        return len(self.weights)
