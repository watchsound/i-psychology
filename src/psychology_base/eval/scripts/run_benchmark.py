"""Unified CLI entry point for standard-SOTA benchmark adapter runs.

Step-1 scaffolding — the three loaders are not implemented yet (step 2+), so
`--benchmark {name}` currently resolves to an empty iterator for each name and
the runner prints an empty-run report. The point of step 1 is to prove the
CLI, loader-registry, and judgment-cache plumbing work end-to-end without
errors; step 2 lands the first real loader (EpBench).

Usage (post-scaffold):
    python -m psychology_base.eval.scripts.run_benchmark \\
        --benchmark epbench --mode direct --split smoke --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, Optional

from psychology_base.eval.benchmarks.instance import (
    BenchmarkInstance,
    BenchmarkLoader,
    BenchmarkName,
)
from psychology_base.eval.benchmarks.epbench import EpBenchLoader
from psychology_base.eval.benchmarks.judge import JudgmentCache
from psychology_base.eval.benchmarks.locomo import LoCoMoLoader
from psychology_base.eval.benchmarks.longmemeval import LongMemEvalLoader


class _EmptyLoader:
    """Stub loader used until step-2+ real loaders land.

    Returns zero instances. The empty-run assertion in the step-1 gate (plan
    §5) exercises this: the harness must handle "benchmark name recognized,
    data unavailable" without crashing, so the CLI can be wired up and tested
    before the real loaders exist.
    """

    def __init__(self, name: BenchmarkName) -> None:
        self.name = name

    def iter_instances(self, root: Optional[Path] = None) -> Iterator[BenchmarkInstance]:
        return iter(())


_LOADER_REGISTRY: dict[BenchmarkName, BenchmarkLoader] = {
    "longmemeval": LongMemEvalLoader(),
    "locomo": LoCoMoLoader(),
    "epbench": EpBenchLoader(),
}


def get_loader(name: BenchmarkName) -> BenchmarkLoader:
    """Look up a loader by benchmark name. Real loaders register themselves
    here as step 2+ land."""
    if name not in _LOADER_REGISTRY:
        raise KeyError(f"Unknown benchmark: {name!r}")
    return _LOADER_REGISTRY[name]


def register_loader(name: BenchmarkName, loader: BenchmarkLoader) -> None:
    """Register a loader implementation. Called by per-benchmark modules at
    import time once they exist."""
    _LOADER_REGISTRY[name] = loader


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_benchmark",
        description="Run a standard-SOTA regression benchmark against CompanionSystem.",
    )
    p.add_argument(
        "--benchmark",
        choices=("longmemeval", "locomo", "epbench"),
        required=True,
    )
    p.add_argument(
        "--mode",
        choices=("direct", "rag", "both"),
        default="direct",
        help=(
            "direct: recall.content as answer (Option α). "
            "rag: external reader composes from recall.content + fragments (Option γ-RAG). "
            "both: run both and report side-by-side (Option γ full)."
        ),
    )
    p.add_argument(
        "--split",
        choices=("smoke", "full", "sealed"),
        default="smoke",
        help="smoke: 50-instance development subset. full: entire benchmark. sealed: 100-instance hold-out (gate-only).",
    )
    p.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated SUT seeds for 3-seed median (direct-mode only; RAG runs single-seed).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override benchmark data root. Defaults to the lazy-download cache.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override judgment cache dir. Defaults to ~/.cache/psychology_base/judgments.",
    )
    return p


def run(args: argparse.Namespace) -> int:
    loader = get_loader(args.benchmark)
    cache = JudgmentCache(cache_dir=args.cache_dir) if args.cache_dir else JudgmentCache()

    instance_count = 0
    question_count = 0
    for instance in loader.iter_instances(args.data_root):
        instance_count += 1
        question_count += len(instance.questions)

    print(
        f"benchmark={args.benchmark} mode={args.mode} split={args.split} "
        f"seeds={args.seeds} instances={instance_count} questions={question_count} "
        f"cache_dir={cache.cache_dir}"
    )
    if instance_count == 0:
        if isinstance(loader, _EmptyLoader):
            print(
                f"NOTE: loader for {args.benchmark!r} is a stub (step 1). "
                f"Real loader lands in step 2+."
            )
        else:
            print(
                f"NOTE: loader for {args.benchmark!r} returned zero instances. "
                f"Pass --data-root pointing at the dataset's shim file "
                f"(or configure the loader's shim_path directly)."
            )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
