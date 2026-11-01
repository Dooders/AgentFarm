#!/usr/bin/env python3
"""CLI entrypoint for the intrinsic cooperation experiment.

Runs four simulations per seed that differ only in whether agents can form
cooperative units:

- ``baseline``    -- the unmodified intrinsic-evolution world.
- ``placebo``     -- bond/leave actions and cooperation loci exist, bonds never form.
- ``cooperation`` -- units form, pool gathered resources, pay out and enforce
  their heritable terms.
- ``cooperation_no_motive`` -- the same without the motive-driven membership
  reward (the institution must pay for itself).
- ``cooperation_zero_synergy`` (optional, ``--arms ... cooperation_zero_synergy``)
  -- the working institution with no pooling surplus.

Arms whose ``arm_result.json`` already exists under the output directory are
loaded instead of re-run (``--no-resume`` to override), so an interrupted
run can be restarted with the same command line.

Writes ``intrinsic_cooperation_summary.json`` and (when matplotlib is
available) ``intrinsic_cooperation_comparison.png`` to the output directory,
plus each arm's simulation artifacts and cooperation telemetry under
``<output-dir>/[rep_XX/]<arm>/``.

Registered run (seeds 101-120)::

    python scripts/run_intrinsic_cooperation_experiment.py \\
        --num-steps 3000 --seed 101 --num-replicates 20 --jobs 6 \\
        --output-dir experiments/intrinsic_cooperation_confirmatory

At the registered settings the population regulates itself at 30-75 agents
(the cap of 300 never binds), so each job peaks at about 5 GB and one seed's
four arms take about 1.6 h on one core (full-length pilot, seeds 1-2). Size
``--jobs`` to RAM and cores: 6 on a 64 GB host. The script re-executes itself
with ``PYTHONHASHSEED=0`` when it is not already set.

Quick smoke test::

    python scripts/run_intrinsic_cooperation_experiment.py --num-steps 100 --num-replicates 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Many tiny per-agent DQNs: pin BLAS/OpenMP before numpy/torch import their pools.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

# Hash randomisation is fixed at interpreter start, so setting the variable
# here would only reach child processes; restart instead.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.config import SimulationConfig  # noqa: E402
from farm.core.cooperation import CooperationConfig  # noqa: E402
from farm.core.cooperation.config import EXIT_COST_BASES  # noqa: E402
from farm.core.hyperparameter_chromosome import BoundaryMode, MutationMode  # noqa: E402
from farm.runners.intrinsic_cooperation_experiment import (  # noqa: E402
    ARM_NAMES,
    DEFAULT_ARMS,
    IntrinsicCooperationExperiment,
    IntrinsicCooperationExperimentConfig,
    run_provenance,
)
from farm.utils.logging import configure_logging, get_logger  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--environment", type=str, default="development")
    p.add_argument("--profile", type=str, default=None)
    p.add_argument("--num-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--num-replicates", type=int, default=1, help="Paired replicates; replicate r uses seed+r.")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Concurrent seeds. Each worker still runs that seed's four arms in series and peaks at about 5 GB "
            "at the registered settings."
        ),
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device preference (cpu is fastest for the tiny per-agent DQNs).",
    )
    p.add_argument(
        "--arms",
        nargs="+",
        default=list(DEFAULT_ARMS),
        choices=list(ARM_NAMES),
        help="Arms to run; add cooperation_zero_synergy for the optional fifth arm.",
    )
    p.add_argument("--output-dir", type=str, default="experiments/intrinsic_cooperation")
    p.add_argument("--record-interval", type=int, default=1)
    p.add_argument("--no-resume", action="store_true", help="Re-run arms even if their arm_result.json exists.")
    p.add_argument("--mutation-rate", type=float, default=0.1)
    p.add_argument("--mutation-scale", type=float, default=0.1)
    p.add_argument("--mutation-mode", default=MutationMode.GAUSSIAN.value, choices=[m.value for m in MutationMode])
    p.add_argument("--boundary-mode", default=BoundaryMode.REFLECT.value, choices=[m.value for m in BoundaryMode])
    p.add_argument("--no-crossover", action="store_true", help="Disable crossover (and so partner co-parenting).")
    p.add_argument("--selection-pressure", type=str, default="medium")
    p.add_argument("--initial-agent-resource-level", type=float, default=12.0)
    p.add_argument("--initial-resource-count", type=int, default=60)
    p.add_argument("--max-population", type=int, default=300)
    p.add_argument(
        "--offspring-initial-resources",
        type=float,
        default=None,
        help="Resources a newborn starts with (default: the environment's value).",
    )
    p.add_argument(
        "--min-reproduction-resources",
        type=float,
        default=None,
        help="Reserve required on top of the offspring cost to reproduce (default: the environment's value).",
    )
    # Learning / evolution regime
    p.add_argument(
        "--policy-temperature",
        type=float,
        default=0.1,
        help="Softmax temperature of the policies' action distribution before innate priors.",
    )
    p.add_argument(
        "--initial-diversity-scale",
        type=float,
        default=None,
        help="Turn the platform's initial-diversity layer on with this mutation scale (default: off).",
    )
    p.add_argument(
        "--no-freeze-platform-genes",
        action="store_true",
        help="Let the learning hyperparameters and reward weights (Chromosomes A and C) evolve too.",
    )
    # Cooperation knobs
    p.add_argument("--exit-cost-base", default="received", choices=list(EXIT_COST_BASES))
    p.add_argument("--no-mask-pointless-actions", action="store_true", help="Offer leave/bond even when they cannot act.")
    p.add_argument("--no-protect-partners", action="store_true", help="Let members attack their own partners.")
    p.add_argument("--bond-range", type=float, default=6.0)
    p.add_argument("--offer-window", type=int, default=10)
    p.add_argument("--terms", default="mean", choices=["mean", "initiator"])
    p.add_argument("--no-evolve-genes", action="store_true", help="Freeze the cooperation loci at their start values.")
    p.add_argument("--no-randomize-founder-genes", action="store_true")
    p.add_argument("--fixed-settings", type=str, default=None, help='JSON, e.g. \'{"scope":0.7,"exit_cost":0.5}\'.')
    p.add_argument("--membership-reward-scale", type=float, default=1.0)
    p.add_argument("--no-partners-as-allies", action="store_true")
    p.add_argument("--family-on-birth", action="store_true")
    p.add_argument("--synergy", type=float, default=0.2)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p


def _parse_selection_pressure(raw: str):
    if raw.lower() in {"none", "low", "medium", "high"}:
        return raw.lower()
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"--selection-pressure must be a preset or float in [0,1]; got {raw!r}") from exc


def main() -> int:
    args = _build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(environment=args.environment, log_dir="logs", log_level=args.log_level, disable_console=False)
    logger = get_logger(__name__)

    base_config = SimulationConfig.from_centralized_config(environment=args.environment, profile=args.profile)
    base_config.device.device_preference = args.device
    cooperation = CooperationConfig(
        enabled=True,
        bond_range=args.bond_range,
        offer_window=args.offer_window,
        terms=args.terms,
        evolve_genes=not args.no_evolve_genes,
        randomize_founder_genes=not args.no_randomize_founder_genes,
        fixed_settings=json.loads(args.fixed_settings) if args.fixed_settings else None,
        membership_reward_scale=args.membership_reward_scale,
        partners_as_allies=not args.no_partners_as_allies,
        family_on_birth=args.family_on_birth,
        synergy=args.synergy,
        exit_cost_base=args.exit_cost_base,
        mask_pointless_actions=not args.no_mask_pointless_actions,
        protect_partners=not args.no_protect_partners,
        record_interval=args.record_interval,
    )
    config = IntrinsicCooperationExperimentConfig(
        num_steps=args.num_steps,
        seed=args.seed,
        num_replicates=args.num_replicates,
        output_dir=args.output_dir,
        record_interval=args.record_interval,
        arms=tuple(args.arms),
        cooperation=cooperation,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        mutation_mode=MutationMode(args.mutation_mode),
        boundary_mode=BoundaryMode(args.boundary_mode),
        crossover_enabled=not args.no_crossover,
        selection_pressure=_parse_selection_pressure(args.selection_pressure),
        initial_agent_resource_level=args.initial_agent_resource_level,
        initial_resource_count=args.initial_resource_count,
        max_population=args.max_population,
        offspring_initial_resources=args.offspring_initial_resources,
        min_reproduction_resources=args.min_reproduction_resources,
        initial_diversity_scale=args.initial_diversity_scale,
        freeze_platform_genes=not args.no_freeze_platform_genes,
        policy_temperature=args.policy_temperature,
        resume=not args.no_resume,
        jobs=args.jobs,
    )
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "script": "scripts/run_intrinsic_cooperation_experiment.py",
                **run_provenance(),
                "args": vars(args),
            },
            handle,
            indent=2,
        )

    logger.info("intrinsic_cooperation_cli_start", num_steps=args.num_steps, seed=args.seed, arms=args.arms)
    start = time.time()
    result = IntrinsicCooperationExperiment(base_config, config).run()
    print(
        json.dumps(
            {
                "elapsed_seconds": round(time.time() - start, 1),
                "summary_path": result.summary_path,
                "figure_path": result.figure_path,
                "arms": {arm: r.summary() for arm, r in result.arms.items()},
                "hypotheses": result.hypotheses,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
