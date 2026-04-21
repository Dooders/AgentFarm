"""Shared argparse helpers for evolution experiment CLIs."""

from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------
# Each preset is a dict whose keys match argparse ``dest`` names.  When a
# preset is selected via ``--preset``, its values become the argparse
# *defaults* for those arguments, so any explicit CLI flag still wins.
#
# ``stable_hyper_evo`` rationale
# --------------------------------
# Follow-up analysis in ``notebooks/hyperparameter_evolution_results.ipynb``
# and ``docs/experiments/hyperparameter_evolution_convergence.md`` revealed
# two recurring failure modes with the bare defaults:
#
#   1. **Lower-bound collapse** – tournament selection with aggressive mutation
#      pushes the winning learning rate to its minimum boundary and keeps it
#      there.  Switching to ``boundary_mode=reflect`` lets genes bounce back
#      off the wall instead of sticking.
#
#   2. **Diversity collapse** – without adaptive mutation the population
#      converges prematurely.  Enabling adaptive mutation with both the
#      fitness-stall and diversity-collapse rules keeps the search alive.
#
# The mutation magnitudes (rate 0.20, scale 0.15) come from the
# ``run_tournament_mut020_g6`` closure run, which showed the best trade-off
# between exploration and exploitation across the evaluated configs.


def get_presets(
    *,
    evolution_selection_method: type,
    boundary_mode: type,
) -> dict[str, dict[str, object]]:
    """Return the evolution CLI preset table (requires runner/chromosome enums)."""
    return {
        "stable_hyper_evo": {
            "selection_method": evolution_selection_method.TOURNAMENT.value,
            "boundary_mode": boundary_mode.REFLECT.value,
            "mutation_rate": 0.20,
            "mutation_scale": 0.15,
            "adaptive_mutation": True,
            "tournament_size": 3,
            "elitism_count": 1,
        },
    }


def parse_per_gene_multipliers(raw: str | None, *, label: str) -> dict[str, float]:
    """Parse a comma-separated ``gene=value`` string into a multiplier dict.

    Returns an empty dict for ``None`` or empty input.  Raises ``ValueError``
    on malformed entries; ``argparse`` will surface this as a CLI error.
    """
    if not raw:
        return {}
    multipliers: dict[str, float] = {}
    for entry in raw.split(","):
        token = entry.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"{label} entry '{token}' must be of the form 'gene_name=value'."
            )
        name, _, value_str = token.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"{label} entry '{token}' has an empty gene name.")
        try:
            multipliers[name] = float(value_str)
        except ValueError as exc:
            raise ValueError(
                f"{label} entry '{token}' has a non-numeric value."
            ) from exc
    return multipliers


class EvolutionExperimentHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Formatter that preserves raw description layout and also shows defaults."""


def add_evolution_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    presets: dict[str, dict[str, object]],
    preset_help: str,
    evolution_fitness_metric: type,
    evolution_selection_method: type,
    boundary_mode: type,
    crossover_mode: type,
    generations_help: str = "Number of generations to run.",
    fitness_metric_help: str = "Built-in fitness metric used for parent selection.",
) -> None:
    """Add shared evolution hyperparameter / adaptive-mutation flags to *parser*."""
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(presets),
        help=preset_help,
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
    parser.add_argument("--generations", type=int, default=2, help=generations_help)
    parser.add_argument("--population-size", type=int, default=4, help="Candidates per generation.")
    parser.add_argument(
        "--steps-per-candidate",
        type=int,
        default=20,
        help="Simulation steps used to evaluate each candidate.",
    )
    parser.add_argument(
        "--fitness-metric",
        type=str,
        default=evolution_fitness_metric.FINAL_POPULATION.value,
        choices=[metric.value for metric in evolution_fitness_metric],
        help=fitness_metric_help,
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default=evolution_selection_method.TOURNAMENT.value,
        choices=[method.value for method in evolution_selection_method],
        help="Parent selection strategy.",
    )
    parser.add_argument("--mutation-rate", type=float, default=0.25, help="Mutation probability per gene.")
    parser.add_argument(
        "--mutation-scale",
        type=float,
        default=0.2,
        help="Mutation magnitude for mutated genes.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Tournament bracket size when tournament selection is used.",
    )
    parser.add_argument(
        "--boundary-mode",
        type=str,
        default=boundary_mode.CLAMP.value,
        choices=[mode.value for mode in boundary_mode],
        help="Boundary strategy after mutation overshoots gene bounds.",
    )
    parser.add_argument(
        "--interior-bias-fraction",
        type=float,
        default=1e-3,
        help=(
            "When --boundary-mode=interior_biased, fraction of the gene span used as the "
            "upper bound of the inward nudge applied to values landing exactly on a boundary. "
            "Must be non-negative.  Default: 1e-3."
        ),
    )
    parser.add_argument(
        "--boundary-penalty-enabled",
        action="store_true",
        help="Enable soft near-boundary fitness penalty.",
    )
    parser.add_argument(
        "--boundary-penalty-strength",
        type=float,
        default=0.01,
        help="Max per-gene penalty at exact boundary when penalty is enabled.",
    )
    parser.add_argument(
        "--boundary-penalty-threshold",
        type=float,
        default=0.05,
        help="Near-boundary zone width as fraction of gene range (0, 0.5].",
    )
    parser.add_argument(
        "--crossover-mode",
        type=str,
        default=crossover_mode.UNIFORM.value,
        choices=[mode.value for mode in crossover_mode],
        help="Crossover operator used to create offspring chromosomes.",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.5,
        help="BLX-alpha extent used when --crossover-mode=blend.",
    )
    parser.add_argument(
        "--num-crossover-points",
        type=int,
        default=2,
        help="Pivot count used when --crossover-mode=multi_point.",
    )
    parser.add_argument("--elitism-count", type=int, default=1, help="Top candidates copied to next generation.")
    parser.add_argument(
        "--adaptive-mutation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable adaptive mutation rate/scale based on fitness progress and diversity.",
    )
    parser.add_argument(
        "--adaptive-stall-window",
        type=int,
        default=3,
        help="Generations to look back when detecting a fitness stall.",
    )
    parser.add_argument(
        "--adaptive-improve-threshold",
        type=float,
        default=1e-6,
        help="Minimum best-fitness improvement over the window that counts as progress.",
    )
    parser.add_argument(
        "--adaptive-stall-multiplier",
        type=float,
        default=1.5,
        help="Multiplier applied to mutation rate/scale when the search stalls.",
    )
    parser.add_argument(
        "--adaptive-improve-multiplier",
        type=float,
        default=0.8,
        help="Multiplier applied to mutation rate/scale when fitness is improving.",
    )
    parser.add_argument(
        "--adaptive-diversity-threshold",
        type=float,
        default=0.05,
        help="Normalized diversity at or below which mutation is boosted.",
    )
    parser.add_argument(
        "--adaptive-diversity-multiplier",
        type=float,
        default=1.5,
        help="Multiplier applied to mutation rate/scale when diversity collapses.",
    )
    parser.add_argument(
        "--adaptive-disable-fitness",
        action="store_true",
        help="When --adaptive-mutation is set, skip the fitness-progress adaptation rule.",
    )
    parser.add_argument(
        "--adaptive-disable-diversity",
        action="store_true",
        help="When --adaptive-mutation is set, skip the diversity-collapse adaptation rule.",
    )
    parser.add_argument(
        "--adaptive-max-step-multiplier",
        type=float,
        default=2.0,
        help=(
            "Maximum factor by which the accumulated mutation multiplier may change in a "
            "single generation.  Bounds per-step change to [1/max_step, max_step] so large "
            "stall/improve factors are dampened.  Must be >= 1.0.  Default: 2.0."
        ),
    )
    parser.add_argument(
        "--adaptive-default-per-gene",
        action="store_true",
        help=(
            "Apply built-in per-gene scale defaults tuned for sensitivity "
            "(learning_rate=0.5×, gamma=0.75×, epsilon_decay=0.75× scale).  "
            "User --adaptive-per-gene-scale values override these defaults."
        ),
    )
    parser.add_argument(
        "--adaptive-per-gene-rate",
        type=str,
        default=None,
        help=(
            "Comma-separated per-gene mutation-rate multipliers, e.g. "
            "'learning_rate=0.5,gamma=2.0'.  Each value must be non-negative."
        ),
    )
    parser.add_argument(
        "--adaptive-per-gene-scale",
        type=str,
        default=None,
        help=(
            "Comma-separated per-gene mutation-scale multipliers, e.g. "
            "'learning_rate=0.5,gamma=2.0'.  Each value must be non-negative."
        ),
    )


def add_evolution_convergence_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared convergence-criteria flags used by evolution cohort CLIs."""
    parser.add_argument(
        "--convergence-enabled",
        action="store_true",
        help=(
            "Enable convergence checking.  When set, the run will detect "
            "fitness plateau and diversity collapse and optionally stop early."
        ),
    )
    parser.add_argument(
        "--convergence-fitness-window",
        type=int,
        default=5,
        help=(
            "Number of trailing generations over which best-fitness improvement "
            "is measured for the plateau criterion."
        ),
    )
    parser.add_argument(
        "--convergence-fitness-threshold",
        type=float,
        default=1e-4,
        help=(
            "Minimum absolute improvement in best fitness over the window "
            "required to avoid a plateau declaration."
        ),
    )
    parser.add_argument(
        "--convergence-diversity-window",
        type=int,
        default=3,
        help=(
            "Number of consecutive generations with diversity below the threshold "
            "required to trigger a diversity-collapse declaration."
        ),
    )
    parser.add_argument(
        "--convergence-diversity-threshold",
        type=float,
        default=0.01,
        help="Normalized diversity at or below which diversity-collapse is considered.",
    )
    parser.add_argument(
        "--convergence-min-generations",
        type=int,
        default=1,
        help="Minimum number of completed generations before convergence checks begin.",
    )
    parser.add_argument(
        "--convergence-no-early-stop",
        action="store_true",
        help=(
            "When --convergence-enabled is set, annotate the result with convergence "
            "metadata but do not halt the run early."
        ),
    )
