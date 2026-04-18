#!/usr/bin/env python3
"""Run crossover-mode comparison experiments and report fitness/diversity impact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import BoundaryMode, CrossoverMode
from farm.runners import EvolutionExperiment, EvolutionExperimentConfig


def _parse_csv_ints(raw: str) -> List[int]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("At least one seed must be provided.")
    return [int(token) for token in tokens]


def _parse_modes(raw: str) -> List[CrossoverMode]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("At least one crossover mode must be provided.")
    return [CrossoverMode(token) for token in tokens]


def _build_stats(values: Iterable[float]) -> Dict[str, float] | None:
    resolved = list(values)
    if not resolved:
        return None
    return {
        "mean": float(mean(resolved)),
        "stdev": float(stdev(resolved)) if len(resolved) > 1 else 0.0,
        "min": float(min(resolved)),
        "max": float(max(resolved)),
    }


def _summarize_mode_runs(mode: CrossoverMode, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    final_best_fitness_values = [float(run["final_best_fitness"]) for run in runs]
    final_mean_fitness_values = [float(run["final_mean_fitness"]) for run in runs]
    final_diversity_values = [
        float(run["final_diversity"])
        for run in runs
        if run["final_diversity"] is not None
    ]
    return {
        "mode": mode.value,
        "num_runs": len(runs),
        "num_runs_with_diversity": len(final_diversity_values),
        "final_best_fitness": _build_stats(final_best_fitness_values),
        "final_mean_fitness": _build_stats(final_mean_fitness_values),
        "final_diversity": _build_stats(final_diversity_values),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare crossover strategy impact on fitness and diversity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="development",
        choices=["development", "production", "testing"],
        help="Centralized config environment.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["benchmark", "simulation", "research"],
        help="Optional centralized config profile.",
    )
    parser.add_argument("--generations", type=int, default=3, help="Number of generations per run.")
    parser.add_argument("--population-size", type=int, default=6, help="Candidates per generation.")
    parser.add_argument("--steps-per-candidate", type=int, default=50, help="Simulation steps per candidate.")
    parser.add_argument("--mutation-rate", type=float, default=0.25, help="Mutation probability per gene.")
    parser.add_argument("--mutation-scale", type=float, default=0.2, help="Mutation magnitude for mutated genes.")
    parser.add_argument(
        "--boundary-mode",
        type=str,
        default=BoundaryMode.CLAMP.value,
        choices=[mode.value for mode in BoundaryMode],
        help="Boundary strategy after mutation overshoots gene bounds.",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.5,
        help="BLX-alpha extent used when mode=blend.",
    )
    parser.add_argument(
        "--num-crossover-points",
        type=int,
        default=2,
        help="Pivot count used when mode=multi_point.",
    )
    parser.add_argument(
        "--crossover-modes",
        type=str,
        default="uniform,blend,multi_point,single_point",
        help="Comma-separated crossover modes to compare.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds (one full run per seed per mode).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/evolution/crossover_strategy_comparison.json"),
        help="Destination JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    modes = _parse_modes(args.crossover_modes)
    seeds = _parse_csv_ints(args.seeds)
    base_config = SimulationConfig.from_centralized_config(
        environment=args.environment,
        profile=args.profile,
    )

    run_records: List[Dict[str, Any]] = []
    mode_summaries: List[Dict[str, Any]] = []

    for mode in modes:
        mode_runs: List[Dict[str, Any]] = []
        for seed in seeds:
            config = EvolutionExperimentConfig(
                num_generations=args.generations,
                population_size=args.population_size,
                num_steps_per_candidate=args.steps_per_candidate,
                mutation_rate=args.mutation_rate,
                mutation_scale=args.mutation_scale,
                boundary_mode=BoundaryMode(args.boundary_mode),
                crossover_mode=mode,
                blend_alpha=args.blend_alpha,
                num_crossover_points=args.num_crossover_points,
                seed=seed,
            )
            result = EvolutionExperiment(base_config, config).run()
            final_summary = result.generation_summaries[-1]
            record = {
                "mode": mode.value,
                "seed": seed,
                "best_candidate_fitness": float(result.best_candidate.fitness),
                "final_best_fitness": float(final_summary.best_fitness),
                "final_mean_fitness": float(final_summary.mean_fitness),
                "final_diversity": (
                    float(final_summary.diversity)
                    if final_summary.diversity is not None
                    else None
                ),
            }
            run_records.append(record)
            mode_runs.append(record)
        mode_summaries.append(_summarize_mode_runs(mode, mode_runs))

    report = {
        "config": {
            "environment": args.environment,
            "profile": args.profile,
            "generations": args.generations,
            "population_size": args.population_size,
            "steps_per_candidate": args.steps_per_candidate,
            "mutation_rate": args.mutation_rate,
            "mutation_scale": args.mutation_scale,
            "boundary_mode": args.boundary_mode,
            "blend_alpha": args.blend_alpha,
            "num_crossover_points": args.num_crossover_points,
            "crossover_modes": [mode.value for mode in modes],
            "seeds": seeds,
        },
        "mode_summaries": mode_summaries,
        "runs": run_records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"Wrote crossover comparison report to: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
