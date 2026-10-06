from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Heartbeat:
    """H3 dual-rate: wake tick on observe (fast), sleep tick on tick() (slow).

    Step 1 only tracks counts. Later steps:
      - step 3: sleep tick runs eviction + Hebbian edge decay
      - step 4: sleep tick runs Witness divergence computation + pain propagation

    Phase 1.5 adds ``session_boundary_tick`` — a lightweight signal benchmark
    adapters can emit between sessions (LoCoMo has `session_N` boundaries;
    LongMemEval has them too) so the Consolidator extracts on coherent
    windows instead of counting raw observations.
    """

    wake_ticks: int = 0
    sleep_ticks: int = 0
    session_boundaries: int = 0

    def wake(self) -> None:
        self.wake_ticks += 1

    def sleep(self) -> None:
        self.sleep_ticks += 1

    def session_boundary_tick(self) -> None:
        self.session_boundaries += 1

    @property
    def total(self) -> int:
        return self.wake_ticks + self.sleep_ticks
