#!/usr/bin/env python3
"""
Spec-driven CLI for running experiments.
"""

import argparse
import sys
from typing import Any

from benchmarks.core.registry import REGISTRY
from benchmarks.core.runner import Runner
from benchmarks.core.spec import load_spec
from benchmarks.core.sweep import SweepRunner
from benchmarks.core.compare import compare_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark/profile experiments")
    parser.add_argument("--spec", type=str, help="Path to YAML/JSON spec file")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="Compare two run result JSON files and print Markdown")
    return parser.parse_args()


def cmd_list() -> int:
    # Discover experiments
    REGISTRY.discover_package("benchmarks.implementations")
    infos = REGISTRY.list()
    print("Available experiments:")
    for info in infos:
        print(f"- {info.slug}: {info.summary}")
    return 0


def main() -> int:
    args = parse_args()
    if args.list:
        return cmd_list()

    if args.compare:
        a_path, b_path = args.compare
        md = compare_results(a_path, b_path)
        print(md)
        return 0

    if not args.spec:
        print("Error: --spec is required (or use --list)")
        return 2

    # Discover experiments before loading spec
    REGISTRY.discover_package("benchmarks.implementations")

    # Load spec and create experiment
    spec = load_spec(args.spec)
    if spec.sweep:
        sweeper = SweepRunner(
            experiment_slug=spec.experiment,
            base_params=spec.params,
            output_dir=spec.output_dir,
            iterations_warmup=spec.iterations["warmup"],
            iterations_measured=spec.iterations["measured"],
            seed=spec.seed,
            tags=spec.tags,
            notes=spec.notes,
            instruments=spec.instrumentation,
        )
        if spec.strategy == "cartesian":
            results = sweeper.run_cartesian(spec.sweep)
        elif spec.strategy == "random":
            if not spec.samples:
                print("Error: 'samples' required for strategy=random")
                return 2
            results = sweeper.run_random(spec.sweep, samples=spec.samples)
        else:
            print(f"Error: unknown sweep strategy '{spec.strategy}'")
            return 2
        print(f"\nSweep complete: {len(results)} runs")
        for r in results:
            line = f"  {r.run_id}"
            if "duration_s" in r.metrics:
                m = r.metrics["duration_s"]
                line += f"  mean={m.get('mean', 0):.4f}s p95={m.get('p95', 0):.4f}s"
            print(line)
        print(f"  Output dir: {spec.output_dir}")
        return 0
    else:
        experiment = REGISTRY.create(spec.experiment, spec.params)
        runner = Runner(
            name=spec.experiment,
            experiment=experiment,
            output_dir=spec.output_dir,
            iterations_warmup=spec.iterations["warmup"],
            iterations_measured=spec.iterations["measured"],
            seed=spec.seed,
            tags=spec.tags,
            notes=spec.notes,
            instruments=spec.instrumentation,
        )
        result = runner.run()
        print("\nRun complete:")
        print(f"  Experiment: {result.name}")
        print(f"  Run ID: {result.run_id}")
        if "duration_s" in result.metrics:
            m = result.metrics["duration_s"]
            print(f"  Duration (s): mean={m.get('mean', 0):.4f}, p50={m.get('p50', 0):.4f}, p95={m.get('p95', 0):.4f}")
        if result.iteration_metrics:
            last = result.iteration_metrics[-1].metrics
            if "observes_per_sec" in last:
                print(f"  Observes/sec (last): {float(last['observes_per_sec']):.1f}")
        print(f"  Output dir: {runner.run_dir}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
