#!/usr/bin/env python3
"""CLI for the Layer C union-emergence AgentFarm port.

Example::

    python scripts/run_union_emergence.py --num-steps 150 --seed 42
    python scripts/run_union_emergence.py --mode first_glance
    python scripts/run_union_emergence.py --courtship-steps 0 --exit-tax 0.2
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import BoundaryMode, MutationMode
from farm.runners.union_emergence_experiment import (
    ARM_NAMES,
    UnionCell,
    UnionEmergenceExperiment,
    UnionEmergenceExperimentConfig,
    default_cells,
)
from farm.utils.logging import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AgentFarm union-emergence port (Layer C).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--environment", type=str, default="development")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=("first_glance", "full", "custom"),
        default="first_glance",
        help="first_glance = 100 steps × 2 seeds; full = 250 steps × 2 seeds",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-replicates", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/union_emergence/layer_c")
    parser.add_argument("--selection-pressure", type=str, default="low")
    parser.add_argument("--exit-tax", type=float, default=None)
    parser.add_argument("--courtship-steps", type=int, default=None)
    parser.add_argument("--social-range", type=float, default=None)
    parser.add_argument("--bonding-cost", type=float, default=None)
    parser.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help=f"Subset of {list(ARM_NAMES)}",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the baseline cell (skip ablations)",
    )
    parser.add_argument("--log-level", type=str, default="WARNING")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(environment=args.environment)
    if args.mode == "full":
        num_steps = args.num_steps or 250
        replicates = args.num_replicates or 2
    else:
        num_steps = args.num_steps or 100
        replicates = args.num_replicates or 2

    cells = default_cells()
    if args.baseline_only:
        cells = (UnionCell(name="baseline"),)
    knob_override = any(
        value is not None
        for value in (args.courtship_steps, args.exit_tax, args.social_range, args.bonding_cost)
    )
    if knob_override:
        cells = (
            UnionCell(
                name="custom",
                courtship_steps=args.courtship_steps if args.courtship_steps is not None else 12,
                exit_tax=args.exit_tax if args.exit_tax is not None else 1.1,
                social_range=args.social_range if args.social_range is not None else 3.2,
                bonding_cost=args.bonding_cost if args.bonding_cost is not None else 0.8,
            ),
        )

    arms = tuple(args.arms) if args.arms else ARM_NAMES
    base_config = SimulationConfig.from_centralized_config(
        environment=args.environment,
        profile=args.profile,
    )
    config = UnionEmergenceExperimentConfig(
        num_steps=num_steps,
        seed=args.seed,
        num_replicates=replicates,
        output_dir=args.output_dir,
        selection_pressure=args.selection_pressure,
        courtship_steps=cells[0].courtship_steps,
        exit_tax=cells[0].exit_tax,
        social_range=cells[0].social_range,
        bonding_cost=cells[0].bonding_cost,
        arms=arms,
        cells=cells,
        mutation_mode=MutationMode.GAUSSIAN,
        boundary_mode=BoundaryMode.REFLECT,
    )
    result = UnionEmergenceExperiment(base_config, config).run()
    wins = result.get("win_conditions", {})
    print(f"Wrote {result.get('summary_path')}")
    print(f"Win-condition checks: {wins.get('passed', 0)}/{wins.get('total', 0)}")
    for name, ok in wins.get("checks", {}).items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
