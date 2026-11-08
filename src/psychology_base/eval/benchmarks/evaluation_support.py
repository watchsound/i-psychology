"""Passive recall capture and the small checks shared by Phase 9 studies."""
import copy
import time
import tracemalloc

from psychology_base.companion.system import CompanionSystem


class AuditedSystem(CompanionSystem):
    """Capture the returned recall without another reconstruction or commit."""

    def recall(self, *args, **kwargs):
        recalled = super().recall(*args, **kwargs)
        self.last_context_fragments = recalled.fragments_used[:6]
        self.last_recall = copy.deepcopy(recalled)
        self.audit_recall_count = getattr(self, "audit_recall_count", 0) + 1
        return recalled


def complete_keys(actual, expected):
    """Validate one explicitly constructed matrix; no schema inference."""
    actual = list(actual)
    if not expected or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Missing, duplicated or unexpected scored trials")
    return {"expected": len(expected), "actual": len(actual), "unique": len(set(actual))}


def measure_recall(system, cue, **kwargs):
    """Own allocation tracing around exactly one actual memory call."""
    if tracemalloc.is_tracing():
        raise ValueError("Stop existing allocation tracing before measuring recall")
    tracemalloc.start()
    try:
        start = time.perf_counter()
        recalled = system.recall(cue, **kwargs)
        recall_ms = (time.perf_counter() - start) * 1000
        _, recall_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return recalled, {"recall_ms": recall_ms, "recall_peak": recall_peak}
