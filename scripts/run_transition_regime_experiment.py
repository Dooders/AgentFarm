#!/usr/bin/env python3
"""Run intrinsic-evolution sweeps for transition-regime analysis.

The matrix produced by this runner is designed for the evidence-gated
transition-regime analyzer.  It varies a continuous resource-buffer parameter
around the balanced stable profile and can optionally run mechanism
interventions such as crossover-enabled gene flow or a long-horizon balanced
cohort.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from farm.config import SimulationConfig  # noqa: E402
from farm.core.hyperparameter_chromosome import BoundaryMode, MutationMode  # noqa: E402
from farm.core.initial_diversity import InitialDiversityConfig, SeedingMode  # noqa: E402
from farm.runners.intrinsic_evolution_experiment import (  # noqa: E402
    InitialConditionsConfig,
    IntrinsicEvolutionExperiment,
    IntrinsicEvolutionExperimentConfig,
    IntrinsicEvolutionPolicy,
    SpeciationConfig,
)
from farm.utils.logging import configure_logging, get_logger  # noqa: E402


DEFAULT_SEEDS: List[int] = [42, 7, 19, 101, 137, 256]
DEFAULT_RESOURCE_LEVELS: List[float] = [9.0, 9.5, 10.0, 10.5, 11.0]
DEFAULT_LONG_HORIZON_LEVELS: List[float] = [10.0]
INTERVENTIONS: Tuple[str, ...] = ("baseline", "crossover_on", "long_horizon")


def resolve_resource_buffer_line(level: float) -> Dict[str, Any]:
    """Resolve the coupled stable-profile resource-buffer line.

    The line is anchored so the existing stable sub-profiles are recovered:
    8→(32, 0.14), 10→(34, 0.15), and 12→(36, 0.16).
    """

    return {
        "initial_agent_resource_level": float(level),
        "initial_resource_count": int(round(float(level) + 24.0)),
        "resource_regen_rate": 0.10 + 0.005 * float(level),
        "resource_regen_amount": 3,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a transition-regime matrix around the balanced stable profile. "
            "Use analyze_transition_regime.py afterwards to classify modes and "
            "estimate transition probabilities."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--environment", type=str, default="development")
    parser.add_argument("--profile", type=str, default="stable")
    parser.add_argument("--parameter-name", type=str, default="initial_agent_resource_level")
    parser.add_argument("--resource-levels", nargs="+", type=float, default=DEFAULT_RESOURCE_LEVELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=str, default="experiments/transition_regime")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--mutation-scale", type=float, default=0.10)
    parser.add_argument("--selection-pressure", type=str, default="low")
    parser.add_argument("--initial-diversity-mutation-rate", type=float, default=1.0)
    parser.add_argument("--initial-diversity-mutation-scale", type=float, default=0.25)
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=list(INTERVENTIONS),
        default=["baseline"],
        help="Experimental conditions to include.",
    )
    parser.add_argument(
        "--long-horizon-resource-levels",
        nargs="+",
        type=float,
        default=DEFAULT_LONG_HORIZON_LEVELS,
        help="Resource levels used only for the long_horizon intervention.",
    )
    parser.add_argument("--long-horizon-num-steps", type=int, default=5000)
    parser.add_argument(
        "--couple-resource-buffer-line",
        dest="couple_resource_buffer_line",
        action="store_true",
        default=True,
        help="Couple resource count and regen rate to initial agent resources.",
    )
    parser.add_argument(
        "--no-couple-resource-buffer-line",
        dest="couple_resource_buffer_line",
        action="store_false",
        help="Sweep only initial_agent_resource_level; keep resource count/regen fixed.",
    )
    parser.add_argument("--fixed-initial-resource-count", type=int, default=34)
    parser.add_argument("--fixed-resource-regen-rate", type=float, default=0.15)
    parser.add_argument("--fixed-resource-regen-amount", type=int, default=3)
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_initial_conditions(level: float, args: argparse.Namespace) -> Dict[str, Any]:
    if args.couple_resource_buffer_line:
        return resolve_resource_buffer_line(level)
    return {
        "initial_agent_resource_level": float(level),
        "initial_resource_count": args.fixed_initial_resource_count,
        "resource_regen_rate": args.fixed_resource_regen_rate,
        "resource_regen_amount": args.fixed_resource_regen_amount,
    }


def _levels_for_intervention(intervention: str, args: argparse.Namespace) -> List[float]:
    if intervention == "long_horizon":
        return [float(level) for level in args.long_horizon_resource_levels]
    return [float(level) for level in args.resource_levels]


def _num_steps_for_intervention(intervention: str, args: argparse.Namespace) -> int:
    return int(args.long_horizon_num_steps if intervention == "long_horizon" else args.num_steps)


def _parse_selection_pressure(raw: str) -> Any:
    if raw is None:
        return None
    if raw.lower() in {"none", "low", "medium", "high"}:
        return raw.lower()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"selection pressure must be one of none/low/medium/high or a float in [0, 1]; got {raw!r}"
        ) from exc


def _run_dir(output_dir: Path, intervention: str, level: float, seed: int) -> Path:
    return output_dir / intervention / f"resource_level_{level:g}" / f"seed_{seed}"


def _factor_metadata(
    intervention: str,
    level: float,
    seed: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> Dict[str, Any]:
    initial_conditions = _resolve_initial_conditions(level, args)
    return {
        "profile": args.profile,
        "intervention": intervention,
        "seed": seed,
        "parameter_name": args.parameter_name,
        "parameter_value": float(level),
        "num_steps": _num_steps_for_intervention(intervention, args),
        "warmup_steps": args.warmup_steps,
        "snapshot_interval": args.snapshot_interval,
        "crossover_enabled": intervention == "crossover_on",
        "run_dir": str(run_dir),
        "resolved_initial_conditions": initial_conditions,
    }


def _planned_runs(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    planned: List[Dict[str, Any]] = []
    for intervention in args.interventions:
        for level in _levels_for_intervention(intervention, args):
            for seed in args.seeds:
                run_dir = _run_dir(output_dir, intervention, level, seed)
                planned.append(_factor_metadata(intervention, level, seed, args, run_dir))
    return planned


def build_transition_run(
    factor: Dict[str, Any],
    args: argparse.Namespace,
) -> IntrinsicEvolutionExperiment:
    """Construct one intrinsic-evolution run from transition-regime factors."""

    seed = int(factor["seed"])
    base_config = SimulationConfig.from_centralized_config(environment=args.environment)
    base_config.initial_diversity = InitialDiversityConfig(
        mode=SeedingMode.INDEPENDENT_MUTATION,
        mutation_rate=args.initial_diversity_mutation_rate,
        mutation_scale=args.initial_diversity_mutation_scale,
        mutation_mode=MutationMode.GAUSSIAN,
        boundary_mode=BoundaryMode.REFLECT,
        seed=seed,
    )

    policy = IntrinsicEvolutionPolicy(
        enabled=True,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        mutation_mode=MutationMode.GAUSSIAN,
        boundary_mode=BoundaryMode.REFLECT,
        crossover_enabled=bool(factor["crossover_enabled"]),
        selection_pressure=_parse_selection_pressure(args.selection_pressure),
        seed=seed,
    )

    resolved = factor["resolved_initial_conditions"]
    initial_conditions = InitialConditionsConfig(
        profile=args.profile,
        warmup_steps=args.warmup_steps,
        initial_agent_resource_level=resolved["initial_agent_resource_level"],
        initial_resource_count=resolved["initial_resource_count"],
        resource_regen_rate=resolved["resource_regen_rate"],
        resource_regen_amount=resolved["resource_regen_amount"],
    )
    speciation = SpeciationConfig(
        enabled=True,
        algorithm="gmm",
        max_k=4,
        seed=seed,
        scaler="none",
    )
    config = IntrinsicEvolutionExperimentConfig(
        num_steps=int(factor["num_steps"]),
        snapshot_interval=args.snapshot_interval,
        install_default_initial_diversity=True,
        initial_conditions=initial_conditions,
        policy=policy,
        speciation=speciation,
        output_dir=factor["run_dir"],
        seed=seed,
    )
    return IntrinsicEvolutionExperiment(base_config, config)


def _write_factor_metadata(run_dir: Path, factor: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "transition_factor_metadata.json"
    path.write_text(json.dumps(factor, indent=2, default=str), encoding="utf-8")


def _run_one(factor: Dict[str, Any], args: argparse.Namespace, logger: Any) -> Tuple[bool, Dict[str, Any]]:
    run_dir = Path(factor["run_dir"])
    _write_factor_metadata(run_dir, factor)
    record = dict(factor)
    record.update(
        status="pending",
        elapsed_seconds=None,
        num_steps_completed=None,
        final_population=None,
        error=None,
    )
    try:
        logger.info("transition_regime_run_start", **factor)
        start = time.time()
        result = build_transition_run(factor, args).run()
        elapsed = time.time() - start
        record.update(
            status="ok",
            elapsed_seconds=round(elapsed, 3),
            num_steps_completed=result.num_steps_completed,
            final_population=result.final_population,
        )
        logger.info(
            "transition_regime_run_ok",
            intervention=factor["intervention"],
            parameter_value=factor["parameter_value"],
            seed=factor["seed"],
            elapsed=round(elapsed, 1),
            final_population=result.final_population,
        )
        return True, record
    except Exception as exc:
        record.update(status="error", error=str(exc))
        logger.error("transition_regime_run_failed", error=str(exc), **factor)
        traceback.print_exc(file=sys.stderr)
        return False, record


def execute_matrix(args: argparse.Namespace, output_dir: Path, logger: Any) -> Tuple[List[Dict[str, Any]], int, int]:
    records: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    for factor in _planned_runs(args, output_dir):
        ok, record = _run_one(factor, args, logger)
        records.append(record)
        if ok:
            n_ok += 1
            continue
        n_fail += 1
        if args.fail_fast:
            print("Aborting transition-regime matrix (--fail-fast).", file=sys.stderr)
            break
    return records, n_ok, n_fail


def _print_dry_run(planned: List[Dict[str, Any]], output_dir: Path) -> None:
    print("Transition-regime experiment — DRY RUN")
    print(f"  output_dir : {output_dir}")
    print(f"  total runs : {len(planned)}")
    print()
    for factor in planned:
        ic = factor["resolved_initial_conditions"]
        print(
            f"  {factor['intervention']:13s} level={factor['parameter_value']:g} "
            f"seed={factor['seed']} steps={factor['num_steps']} "
            f"resources(agent={ic['initial_agent_resource_level']}, "
            f"nodes={ic['initial_resource_count']}, regen={ic['resource_regen_rate']:.4f})"
        )


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    planned = _planned_runs(args, output_dir)
    if args.dry_run:
        _print_dry_run(planned, output_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        environment=args.environment,
        log_dir="logs",
        log_level=args.log_level,
        disable_console=False,
    )
    logger = get_logger(__name__)
    print(f"Transition-regime matrix: {len(planned)} run(s)")
    print(f"  interventions : {args.interventions}")
    print(f"  seeds         : {args.seeds}")
    print(f"  output        : {output_dir}")

    records, n_ok, n_fail = execute_matrix(args, output_dir, logger)
    manifest = {
        "sweep_type": "transition_regime",
        "parameter_name": args.parameter_name,
        "resource_levels": args.resource_levels,
        "seeds": args.seeds,
        "interventions": args.interventions,
        "couple_resource_buffer_line": args.couple_resource_buffer_line,
        "num_steps": args.num_steps,
        "long_horizon_num_steps": args.long_horizon_num_steps,
        "warmup_steps": args.warmup_steps,
        "snapshot_interval": args.snapshot_interval,
        "runs": records,
        "n_ok": n_ok,
        "n_fail": n_fail,
    }
    manifest_path = output_dir / "transition_regime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"\nMatrix complete: {n_ok}/{len(planned)} runs succeeded.")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext step: python scripts/analyze_transition_regime.py --sweep-dir {output_dir}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
