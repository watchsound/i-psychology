"""Standard-SOTA regression-floor benchmark adapters.

Three benchmarks (LongMemEval / LoCoMo / EpBench) wrapped as pluggable loaders +
scorers behind the shared `MemorySystem` protocol. See
[docs/phases/phase-1-benchmark-adapters-plan.md] for the design and
[docs/experiments/phase-1-benchmark-adapters/] for measured option-wise results.
"""

from psychology_base.eval.benchmarks.instance import (
    BenchmarkInstance,
    BenchmarkLoader,
    BenchmarkQuestion,
    BenchmarkTurn,
    BenchmarkName,
)

__all__ = [
    "BenchmarkInstance",
    "BenchmarkLoader",
    "BenchmarkQuestion",
    "BenchmarkTurn",
    "BenchmarkName",
]
