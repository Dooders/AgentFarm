#!/usr/bin/env python3
"""CLI entrypoint for the intrinsic-goals experiment.

Runs two simulations with identical seeds and configuration that differ only in
the agents' reinforcement-learning goals:

- ``uniform`` — every agent shares the default reward function (control).
- ``unique``  — every initial agent is given an independently sampled reward
  function (Chromosome C genes), so each one optimizes for a different goal;
  offspring inherit and mutate their parent's goal.

It writes ``intrinsic_goals_summary.json`` and (when matplotlib is available)
``intrinsic_goals_comparison.png`` to the output directory, plus the usual
per-arm simulation artifacts under ``<output-dir>/<arm>/``.

Example::

    python scripts/run_intrinsic_goals_experiment.py \
        --num-steps 600 --seed 42 --output-dir experiments/intrinsic_goals
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Allow running directly from repo root without installing the package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.config import SimulationConfig  # noqa: E402
from farm.core.hyperparameter_chromosome import (  # noqa: E402
    BoundaryMode,
    MutationMode,
)
from farm.runners.intrinsic_goals_experiment import (  # noqa: E402
    IntrinsicGoalsExperiment,
    IntrinsicGoalsExperimentConfig,
)
from farm.utils.logging import configure_logging, get_logger  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an intrinsic-goals experiment: compare a population where every "
            "agent shares the default reward function (uniform) against one where "
            "every agent has an independently sampled reward function (unique)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--environment", type=str, default="development")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-replicates",
        type=int,
        default=1,
        help=(
            "Number of paired replicates. Replicate r runs both arms with "
            "seed+r. Use >1 to get per-arm mean/std and paired t-tests on the "
            "unique-vs-uniform deltas (required to statistically assess the effect)."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/intrinsic_goals"
    )
    parser.add_argument("--record-interval", type=int, default=1)

    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--mutation-scale", type=float, default=0.1)
    parser.add_argument(
        "--mutation-mode",
        type=str,
        default=MutationMode.GAUSSIAN.value,
        choices=[m.value for m in MutationMode],
    )
    parser.add_argument(
        "--boundary-mode",
        type=str,
        default=BoundaryMode.REFLECT.value,
        choices=[m.value for m in BoundaryMode],
    )
    parser.add_argument(
        "--selection-pressure",
        type=str,
        default="low",
        help="Preset ('none','low','medium','high') or float in [0,1].",
    )
    parser.add_argument(
        "--initial-agent-resource-level",
        type=float,
        default=12.0,
        help="Starting resource level for each agent (stability knob).",
    )
    parser.add_argument(
        "--initial-resource-count",
        type=int,
        default=60,
        help="Number of resource nodes at simulation start (stability knob).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def _parse_selection_pressure(raw: str):
    if raw is None:
        return None
    if raw.lower() in {"none", "low", "medium", "high"}:
        return raw.lower()
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(
            f"--selection-pressure must be a preset or float in [0,1]; got {raw!r}"
        ) from exc


def main() -> int:
    args = _build_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(
        environment=args.environment,
        log_dir="logs",
        log_level=args.log_level,
        disable_console=False,
    )
    logger = get_logger(__name__)

    base_config = SimulationConfig.from_centralized_config(
        environment=args.environment,
        profile=args.profile,
    )

    config = IntrinsicGoalsExperimentConfig(
        num_steps=args.num_steps,
        seed=args.seed,
        num_replicates=args.num_replicates,
        output_dir=args.output_dir,
        record_interval=args.record_interval,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        mutation_mode=MutationMode(args.mutation_mode),
        boundary_mode=BoundaryMode(args.boundary_mode),
        selection_pressure=_parse_selection_pressure(args.selection_pressure),
        initial_agent_resource_level=args.initial_agent_resource_level,
        initial_resource_count=args.initial_resource_count,
    )

    manifest = {
        "script": "scripts/run_intrinsic_goals_experiment.py",
        "args": vars(args),
    }
    with open(
        os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, default=str)

    logger.info(
        "intrinsic_goals_cli_start",
        environment=args.environment,
        num_steps=args.num_steps,
        seed=args.seed,
        num_replicates=args.num_replicates,
        output_dir=args.output_dir,
    )

    start = time.time()
    result = IntrinsicGoalsExperiment(base_config, config).run()
    elapsed = time.time() - start

    print(
        json.dumps(
            {
                "elapsed_seconds": round(elapsed, 3),
                "num_replicates": args.num_replicates,
                "summary_path": result.summary_path,
                "figure_path": result.figure_path,
                "comparison": result.comparison,
                "aggregate": result.aggregate,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
