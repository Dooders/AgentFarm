"""Intrinsic evolution experiment runner.

Runs a single long simulation in which each agent carries its own
:class:`HyperparameterChromosome` and offspring inherit it (with optional
crossover and configurable mutation).  Selection emerges from the resource
environment itself: agents that survive and reproduce pass their
hyperparameters on; lineages that fail die out.

Compare with :mod:`farm.runners.evolution_experiment`, which runs an outer-loop
GA *between* simulations.  Both runners are complementary, not substitutes.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from farm.config import SimulationConfig
from farm.core.agent.config.component_configs import ReproductionPressureConfig
from farm.core.agent.core import compute_effective_reproduction_cost
from farm.core.hyperparameter_chromosome import (
    BoundaryMode,
    CrossoverMode,
    HyperparameterChromosome,
    MutationMode,
    compute_gene_statistics,
)
from farm.core.initial_diversity import (
    InitialDiversityConfig,
    SeedingMode,
)
from farm.core.simulation import run_simulation
from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger
from farm.utils.logging import get_logger

logger = get_logger(__name__)


CoparentStrategy = Literal["nearest_alive_same_type", "random_alive_same_type"]
SelectionPressurePreset = Literal["none", "low", "medium", "high"]
InitialConditionsProfileName = Literal["stable", "stress", "exploratory", "legacy"]

# ── Selection-pressure preset definitions ─────────────────────────────────────
# Each preset maps to a ReproductionPressureConfig with sensible defaults that
# give progressively stronger selection without requiring the user to tune raw
# coefficients.  "none" is the identity (no density cost).

_SELECTION_PRESSURE_PRESETS: Dict[str, ReproductionPressureConfig] = {
    "none": ReproductionPressureConfig(
        local_density_radius=5.0,
        local_density_coefficient=0.0,
        global_carrying_capacity=0,
        global_carrying_capacity_coefficient=0.0,
    ),
    "low": ReproductionPressureConfig(
        local_density_radius=5.0,
        local_density_coefficient=0.5,
        global_carrying_capacity=0,
        global_carrying_capacity_coefficient=0.0,
    ),
    "medium": ReproductionPressureConfig(
        local_density_radius=5.0,
        local_density_coefficient=1.0,
        global_carrying_capacity=100,
        global_carrying_capacity_coefficient=0.5,
    ),
    "high": ReproductionPressureConfig(
        local_density_radius=5.0,
        local_density_coefficient=2.0,
        global_carrying_capacity=100,
        global_carrying_capacity_coefficient=1.0,
    ),
}


def _preset_or_scale_to_pressure_config(
    selection_pressure: Any,
) -> ReproductionPressureConfig:
    """Derive a :class:`ReproductionPressureConfig` from a preset name or scale.

    - A ``str`` must be one of ``"none"``, ``"low"``, ``"medium"``, ``"high"``
      and maps to the corresponding preset.
    - A ``float`` in *[0, 1]* is treated as a linear scale factor applied to
      the ``"high"`` preset's coefficients.
    """
    if isinstance(selection_pressure, str):
        if selection_pressure not in _SELECTION_PRESSURE_PRESETS:
            raise ValueError(
                f"selection_pressure preset must be one of "
                f"{list(_SELECTION_PRESSURE_PRESETS)}; got {selection_pressure!r}."
            )
        return _SELECTION_PRESSURE_PRESETS[selection_pressure]
    if isinstance(selection_pressure, bool):
        raise TypeError(
            f"selection_pressure must be a str preset or float in [0, 1]; "
            f"got {type(selection_pressure).__name__!r}."
        )
    if isinstance(selection_pressure, (int, float)):
        scale = float(selection_pressure)
        if not 0.0 <= scale <= 1.0:
            raise ValueError(
                "selection_pressure as a float must be in [0, 1]; "
                f"got {scale}."
            )
        high = _SELECTION_PRESSURE_PRESETS["high"]
        return ReproductionPressureConfig(
            local_density_radius=high.local_density_radius,
            local_density_coefficient=high.local_density_coefficient * scale,
            global_carrying_capacity=high.global_carrying_capacity,
            global_carrying_capacity_coefficient=high.global_carrying_capacity_coefficient * scale,
        )
    raise TypeError(
        f"selection_pressure must be a str preset or float in [0, 1]; "
        f"got {type(selection_pressure).__name__!r}."
    )


# ── Initial-conditions preset definitions ─────────────────────────────────────
# Each preset bundles overrides for agent starting resources, environmental
# resource count, and regeneration rates.  "stable" is the default and is
# tuned to reduce the early boom-bust transient common in fresh simulations.
# "legacy" applies no overrides and reproduces the pre-feature behavior.

@dataclass(frozen=True)
class _InitialConditionsPreset:
    """Internal container for the values that an initial-conditions preset sets."""

    initial_agent_resource_level: Optional[float]
    """Starting resource level for each newly-spawned agent (``None`` = no override)."""
    initial_resource_count: Optional[int]
    """Number of resource nodes placed at simulation start (``None`` = no override)."""
    resource_regen_rate: Optional[float]
    """Per-step probability of a resource node regenerating (``None`` = no override)."""
    resource_regen_amount: Optional[int]
    """Amount regenerated per node per step when regeneration fires (``None`` = no override)."""


_INITIAL_CONDITIONS_PRESETS: Dict[str, _InitialConditionsPreset] = {
    # stable: higher agent and environment resources → agents survive early steps,
    # reproduce sooner, and the population stabilizes without a sharp spike/crash.
    #
    # Rationale for specific values:
    #   initial_agent_resource_level=20.0: agents start with ~2–4× the default
    #     fallback level (5.0) so they can gather and meet the default
    #     min_reproduction_resources threshold (8) within the first few steps
    #     rather than starving before they find food.
    #   initial_resource_count=30: 50% more resource nodes than the default (20)
    #     ensures food density is high enough for the whole starting population.
    #   resource_regen_rate=0.15: ~50% higher than the default (0.1) so that
    #     the environment recovers quickly after the initial feeding burst.
    #   resource_regen_amount=3: slightly above the default (2) to keep per-node
    #     regeneration meaningful while not making resources unlimited.
    "stable": _InitialConditionsPreset(
        initial_agent_resource_level=20.0,
        initial_resource_count=30,
        resource_regen_rate=0.15,
        resource_regen_amount=3,
    ),
    # stress: scarce resources → intense early selection; use when studying
    # how quickly adaptive traits emerge under pressure.
    #
    # Rationale:
    #   initial_agent_resource_level=3.0: well below the default, forcing
    #     immediate resource-seeking and creating strong early mortality.
    #   initial_resource_count=10: half the default node count; food scarcity
    #     is the dominant selection pressure from step 1.
    #   resource_regen_rate=0.05: slow recovery prevents any surplus from
    #     building up; only the most efficient foragers survive.
    #   resource_regen_amount=1: minimal per-node yield amplifies scarcity.
    "stress": _InitialConditionsPreset(
        initial_agent_resource_level=3.0,
        initial_resource_count=10,
        resource_regen_rate=0.05,
        resource_regen_amount=1,
    ),
    # exploratory: moderate resources similar to defaults; intended for runs
    # focused on genetic diversity rather than survival pressure.
    #
    # Rationale: values match the SimulationConfig defaults (resource_regen_rate=0.1,
    # resource_regen_amount=2, initial_resources=20) with a modest agent resource
    # bump (12.0 vs the fallback 5.0) so agents are not immediately at risk
    # while still experiencing meaningful selection.
    "exploratory": _InitialConditionsPreset(
        initial_agent_resource_level=12.0,
        initial_resource_count=20,
        resource_regen_rate=0.1,
        resource_regen_amount=2,
    ),
    # legacy: no overrides; reproduces pre-InitialConditionsConfig behavior
    # exactly so existing benchmarks and comparisons remain valid.
    "legacy": _InitialConditionsPreset(
        initial_agent_resource_level=None,
        initial_resource_count=None,
        resource_regen_rate=None,
        resource_regen_amount=None,
    ),
}


@dataclass(frozen=True)
class InitialConditionsConfig:
    """Initial-conditions configuration for an intrinsic evolution run.

    Controls the starting state of the simulation (agent resources, environmental
    resources, and regeneration rates) to reduce or shape early boom-bust dynamics.

    Profiles
    --------
    ``profile`` selects a named preset:

    - ``"stable"`` *(default)*: Higher agent and environment resources reduce the
      early growth/death wave while preserving normal adaptive dynamics thereafter.
    - ``"stress"``: Scarce resources amplify early selection pressure; useful for
      studying rapid adaptation.
    - ``"exploratory"``: Moderate resources with default regeneration; a neutral
      baseline for genetic-diversity research.
    - ``"legacy"``: No overrides; exactly reproduces behavior from before this
      feature was introduced.  Use to run apples-to-apples comparisons with older
      runs.
    - ``None``: Skip preset application entirely; only explicit manual overrides
      below are applied (all default to ``None``, i.e., no override).

    Manual overrides
    ----------------
    All override fields default to ``None`` (meaning "use the preset value or
    leave the base config unchanged").  Non-``None`` values always win over the
    preset.

    ``warmup_steps``
    ----------------
    When > 0 the simulation runs for ``warmup_steps`` extra steps *before* the
    main telemetry window.  Gene-trajectory snapshots are suppressed during the
    warmup phase so the persistent artifacts reflect only the stabilized
    population.  The ``startup_transient_metrics`` in
    :class:`IntrinsicEvolutionResult` are still computed from the very first
    ``transient_window`` steps regardless of warmup.

    ``transient_window``
    --------------------
    Number of *post-environment-ready* steps used to derive startup-transient
    metrics (peak birth rate, peak death rate, oscillation amplitude).  Defaults
    to 50.  Does not affect the total number of steps run.
    """

    profile: Optional[InitialConditionsProfileName] = "stable"
    # Per-field manual overrides (win over preset when set)
    initial_agent_resource_level: Optional[float] = None
    initial_resource_count: Optional[int] = None
    resource_regen_rate: Optional[float] = None
    resource_regen_amount: Optional[int] = None
    warmup_steps: int = 0
    transient_window: int = 50

    def __post_init__(self) -> None:
        if self.profile is not None and self.profile not in _INITIAL_CONDITIONS_PRESETS:
            raise ValueError(
                f"profile must be one of {list(_INITIAL_CONDITIONS_PRESETS)} or None; "
                f"got {self.profile!r}."
            )
        if self.initial_agent_resource_level is not None:
            if (
                not math.isfinite(self.initial_agent_resource_level)
                or self.initial_agent_resource_level < 0.0
            ):
                raise ValueError(
                    "initial_agent_resource_level must be a finite, non-negative number when set."
                )
        if self.initial_resource_count is not None and self.initial_resource_count < 0:
            raise ValueError("initial_resource_count must be non-negative when set.")
        if self.resource_regen_rate is not None:
            if (
                not math.isfinite(self.resource_regen_rate)
                or self.resource_regen_rate < 0.0
                or self.resource_regen_rate > 1.0
            ):
                raise ValueError(
                    "resource_regen_rate must be a finite value between 0.0 and 1.0 when set."
                )
        if self.resource_regen_amount is not None and self.resource_regen_amount < 0:
            raise ValueError("resource_regen_amount must be non-negative when set.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.transient_window < 1:
            raise ValueError("transient_window must be at least 1.")

    def resolve(self) -> Dict[str, Any]:
        """Return the effective settings after merging preset defaults with manual overrides.

        The returned dict always contains the following keys:
        ``initial_agent_resource_level``, ``initial_resource_count``,
        ``resource_regen_rate``, ``resource_regen_amount``,
        ``warmup_steps``, ``transient_window``.
        Values may be ``None`` when neither a preset nor a manual override supplies them.
        """
        if self.profile is not None:
            preset = _INITIAL_CONDITIONS_PRESETS[self.profile]
            resolved: Dict[str, Any] = {
                "initial_agent_resource_level": preset.initial_agent_resource_level,
                "initial_resource_count": preset.initial_resource_count,
                "resource_regen_rate": preset.resource_regen_rate,
                "resource_regen_amount": preset.resource_regen_amount,
            }
        else:
            resolved = {
                "initial_agent_resource_level": None,
                "initial_resource_count": None,
                "resource_regen_rate": None,
                "resource_regen_amount": None,
            }

        # Manual overrides win over preset defaults.
        if self.initial_agent_resource_level is not None:
            resolved["initial_agent_resource_level"] = self.initial_agent_resource_level
        if self.initial_resource_count is not None:
            resolved["initial_resource_count"] = self.initial_resource_count
        if self.resource_regen_rate is not None:
            resolved["resource_regen_rate"] = self.resource_regen_rate
        if self.resource_regen_amount is not None:
            resolved["resource_regen_amount"] = self.resource_regen_amount

        resolved["warmup_steps"] = self.warmup_steps
        resolved["transient_window"] = self.transient_window
        return resolved

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict including the resolved effective values."""
        resolved = self.resolve()
        return {
            "profile": self.profile,
            "resolved": {
                k: v for k, v in resolved.items()
                if k not in ("warmup_steps", "transient_window")
            },
            "initial_agent_resource_level": self.initial_agent_resource_level,
            "initial_resource_count": self.initial_resource_count,
            "resource_regen_rate": self.resource_regen_rate,
            "resource_regen_amount": self.resource_regen_amount,
            "warmup_steps": self.warmup_steps,
            "transient_window": self.transient_window,
        }


def _compute_startup_transient_metrics(
    birth_rates: List[float],
    death_rates: List[float],
    populations: List[int],
    transient_window: int,
) -> Dict[str, Any]:
    """Compute startup-transient summary statistics from per-step series.

    Parameters
    ----------
    birth_rates:
        Realized birth rates for the first N steps.
    death_rates:
        Realized death rates for the first N steps.
    populations:
        Alive-agent counts for the first N steps.
    transient_window:
        Configured window size (recorded even when fewer steps were observed).

    Returns
    -------
    Dict with keys ``peak_birth_rate``, ``peak_death_rate``,
    ``oscillation_amplitude``, ``n_steps_observed``, and ``transient_window``.
    """
    n = len(birth_rates)
    if n == 0:
        return {
            "peak_birth_rate": 0.0,
            "peak_death_rate": 0.0,
            "oscillation_amplitude": 0,
            "n_steps_observed": 0,
            "transient_window": transient_window,
        }
    return {
        "peak_birth_rate": max(birth_rates),
        "peak_death_rate": max(death_rates),
        "oscillation_amplitude": max(populations) - min(populations) if populations else 0,
        "n_steps_observed": n,
        "transient_window": transient_window,
    }


@dataclass(frozen=True)
class IntrinsicEvolutionPolicy:
    """Policy controlling per-agent chromosome inheritance during a sim.

    When attached to an :class:`Environment` (as ``intrinsic_evolution_policy``)
    and ``enabled`` is ``True``, :meth:`AgentCore.reproduce` will derive each
    child's chromosome by optionally crossing the parent with a co-parent and
    then mutating the result according to these settings.  When the policy is
    absent or disabled, children inherit the parent's chromosome unchanged.

    Attributes:
        enabled: Master switch for in-situ chromosome inheritance.
        mutation_rate / mutation_scale / mutation_mode: Per-reproduction
            mutation parameters threaded through to
            :func:`mutate_chromosome`.
        boundary_mode / interior_bias_fraction: Boundary handling for mutated
            values; matches semantics in :class:`BoundaryMode`.
        crossover_enabled: When ``True``, reproduction picks a co-parent and
            runs :func:`crossover_chromosomes` before mutation.
        crossover_mode / blend_alpha / num_crossover_points: Crossover
            operator settings (see :class:`CrossoverMode`).
        coparent_strategy: How to pick the co-parent. Accepted values are
            ``"nearest_alive_same_type"`` and
            ``"random_alive_same_type"``. By default, both strategies filter
            to alive agents of the same ``agent_type``; the ``nearest`` variant
            is deterministic-modulo-tiebreak and reflects spatial sociality,
            while the ``random`` variant is uniform over the candidate pool.
            When ``allow_cross_type_pollination`` is ``True``, the same
            selection behavior applies but the candidate pool expands to alive
            agents of any ``agent_type``.
        coparent_max_radius: Optional spatial cap on the co-parent search;
            ``None`` means unbounded.
        allow_cross_type_pollination: When ``True``, the co-parent candidate
            pool includes agents of any ``agent_type``, enabling cross-type
            gene flow.  When ``False`` (default), only agents with the same
            ``agent_type`` as the reproducing agent are eligible.
        seed: Optional seed for the policy's RNG; when ``None`` the runner
            seeds from its own configured seed.
        selection_pressure: Convenience knob that sets density-dependent
            reproduction costs.  Accepts a named preset (``"none"``,
            ``"low"``, ``"medium"``, ``"high"``) or a float in *[0, 1]*
            that scales the ``"high"`` preset's coefficients.  When set,
            ``reproduction_pressure`` is derived automatically and any
            explicit ``reproduction_pressure`` value is ignored.  When
            ``None`` (default), ``reproduction_pressure`` is used as-is.
        reproduction_pressure: Fine-grained density-dependent cost config.
            Ignored when ``selection_pressure`` is not ``None``.
    """

    enabled: bool = True
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1
    mutation_mode: MutationMode = MutationMode.GAUSSIAN
    boundary_mode: BoundaryMode = BoundaryMode.CLAMP
    interior_bias_fraction: float = 1e-3
    crossover_enabled: bool = False
    crossover_mode: CrossoverMode = CrossoverMode.UNIFORM
    blend_alpha: float = 0.5
    num_crossover_points: int = 2
    coparent_strategy: CoparentStrategy = "nearest_alive_same_type"
    coparent_max_radius: Optional[float] = None
    allow_cross_type_pollination: bool = False
    seed: Optional[int] = None
    selection_pressure: Union[SelectionPressurePreset, float, None] = None
    reproduction_pressure: ReproductionPressureConfig = field(
        default_factory=ReproductionPressureConfig
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if self.mutation_scale < 0.0:
            raise ValueError("mutation_scale must be non-negative.")
        if self.interior_bias_fraction < 0.0:
            raise ValueError("interior_bias_fraction must be non-negative.")
        if self.blend_alpha < 0.0:
            raise ValueError("blend_alpha must be non-negative.")
        if self.num_crossover_points < 1:
            raise ValueError("num_crossover_points must be at least 1.")
        if self.coparent_max_radius is not None and self.coparent_max_radius < 0.0:
            raise ValueError("coparent_max_radius must be non-negative when set.")
        # Coerce string-friendly enum values to enum instances, raising if invalid.
        object.__setattr__(self, "mutation_mode", MutationMode(self.mutation_mode))
        object.__setattr__(self, "boundary_mode", BoundaryMode(self.boundary_mode))
        object.__setattr__(self, "crossover_mode", CrossoverMode(self.crossover_mode))
        if self.coparent_strategy not in ("nearest_alive_same_type", "random_alive_same_type"):
            raise ValueError(
                "coparent_strategy must be 'nearest_alive_same_type' or 'random_alive_same_type'."
            )
        # Derive reproduction_pressure from the selection_pressure preset/scale when set.
        if self.selection_pressure is not None:
            derived = _preset_or_scale_to_pressure_config(self.selection_pressure)
            object.__setattr__(self, "reproduction_pressure", derived)


@dataclass(frozen=True)
class IntrinsicEvolutionExperimentConfig:
    """Top-level config for the intrinsic evolution runner.

    ``install_default_initial_diversity`` keeps historical behavior by
    installing the runner's independent-mutation defaults when the provided
    base config still has ``initial_diversity.mode=none``. Set it to ``False``
    to preserve an explicit ``none`` mode (monoculture start) unchanged.

    ``initial_conditions`` controls the starting state of the simulation to
    reduce early boom-bust dynamics.  Defaults to the ``"stable"`` profile;
    use ``InitialConditionsConfig(profile="legacy")`` to reproduce the
    pre-feature behavior exactly.
    """

    num_steps: int = 2000
    snapshot_interval: int = 100
    install_default_initial_diversity: bool = True
    initial_conditions: InitialConditionsConfig = field(
        default_factory=InitialConditionsConfig
    )
    policy: IntrinsicEvolutionPolicy = field(default_factory=IntrinsicEvolutionPolicy)
    output_dir: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be at least 1.")


@dataclass
class IntrinsicEvolutionResult:
    """Summary of an intrinsic evolution run.

    ``num_steps_completed`` may be smaller than ``config.num_steps`` if the
    population went extinct early (mirroring ``run_simulation`` semantics).
    ``final_gene_statistics`` is empty when the final population is empty.
    ``startup_transient_metrics`` summarises the early boom-bust characteristics
    of the run; keys: ``peak_birth_rate``, ``peak_death_rate``,
    ``oscillation_amplitude``, ``n_steps_observed``, ``transient_window``.
    """

    num_steps_completed: int
    final_population: int
    final_gene_statistics: Dict[str, Dict[str, float]]
    startup_transient_metrics: Dict[str, Any] = field(default_factory=dict)


class IntrinsicEvolutionExperiment:
    """Run a single simulation in which agents carry inheritable chromosomes."""

    def __init__(
        self,
        base_config: SimulationConfig,
        config: IntrinsicEvolutionExperimentConfig,
    ) -> None:
        self.base_config = base_config
        self.config = config
        self._last_speciation_config: Dict[str, Any] = {
            "enabled": False,
            "algorithm": None,
            "max_k": None,
            "seed": None,
            "scaler": None,
        }

    def run(self) -> IntrinsicEvolutionResult:
        """Execute the configured number of steps and persist artifacts."""
        seed = self.config.seed if self.config.policy.seed is None else self.config.policy.seed
        rng = random.Random(seed)
        run_config = self.base_config.copy()

        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)

        gene_logger = GeneTrajectoryLogger(
            output_dir=self.config.output_dir,
            snapshot_interval=self.config.snapshot_interval,
        )

        policy = self.config.policy

        # ── Apply initial-conditions overrides ───────────────────────────────
        # Resolve the effective initial-condition settings by merging the
        # selected profile defaults with any explicit per-field overrides.
        if self.config.initial_conditions.profile == "stable":
            logger.info(
                "intrinsic_evolution_initial_conditions_profile_applied",
                profile="stable",
                note=(
                    "stable profile modifies startup resources; set profile='legacy' "
                    "for parity with pre-profile behavior."
                ),
            )
        resolved_ic = self.config.initial_conditions.resolve()
        if resolved_ic.get("initial_agent_resource_level") is not None:
            run_config.agent_behavior.initial_resource_level = round(
                resolved_ic["initial_agent_resource_level"]
            )
        if resolved_ic.get("initial_resource_count") is not None:
            run_config.resources.initial_resources = int(
                resolved_ic["initial_resource_count"]
            )
        if resolved_ic.get("resource_regen_rate") is not None:
            run_config.resources.resource_regen_rate = float(
                resolved_ic["resource_regen_rate"]
            )
        if resolved_ic.get("resource_regen_amount") is not None:
            run_config.resources.resource_regen_amount = int(
                resolved_ic["resource_regen_amount"]
            )

        effective_warmup: int = int(resolved_ic.get("warmup_steps", 0))
        transient_window: int = int(resolved_ic.get("transient_window", 50))
        total_sim_steps = self.config.num_steps + effective_warmup

        # The intrinsic-evolution runner relies on a non-monoculture starting
        # population.  When the caller has not customized the platform-wide
        # initial-diversity config, install the runner's defaults so seeding
        # happens inside run_simulation.  Boundary handling mirrors the
        # per-reproduction policy so seeded values respect the same invariants
        # the policy enforces during the loop.
        if (
            self.config.install_default_initial_diversity
            and run_config.initial_diversity.mode is SeedingMode.NONE
        ):
            run_config.initial_diversity = InitialDiversityConfig(
                mode=SeedingMode.INDEPENDENT_MUTATION,
                mutation_rate=1.0,
                mutation_scale=0.2,
                mutation_mode=policy.mutation_mode,
                boundary_mode=policy.boundary_mode,
                interior_bias_fraction=policy.interior_bias_fraction,
                seed=seed,
            )

        latest_step = 0
        latest_population = 0
        latest_gene_statistics: Dict[str, Dict[str, float]] = {}
        latest_diversity_metrics: Optional[Any] = None

        # Startup-transient metric accumulators (filled for the first
        # transient_window post-environment-ready steps).
        _transient_birth_rates: List[float] = []
        _transient_death_rates: List[float] = []
        _transient_populations: List[int] = []

        # Track agent IDs from the previous step to compute birth/death rates.
        prev_agent_ids: set = set()

        def _agent_telemetry_key(agent: Any) -> tuple:
            """Stable per-agent key: string id when set, else object identity."""
            return (getattr(agent, "agent_id", None), id(agent))

        def _capture_current_state(environment: Any, step: int) -> None:
            """Capture the same state we emit to the trajectory logger.

            The simulation loop performs a final post-loop environment update.
            To keep metadata/result summaries aligned with persisted telemetry,
            derive final result fields from the last callback-observed state.
            """
            nonlocal latest_step, latest_population, latest_gene_statistics
            alive_agents = list(environment.alive_agent_objects)
            chromosomes: List[HyperparameterChromosome] = [
                agent.hyperparameter_chromosome
                for agent in alive_agents
                if getattr(agent, "hyperparameter_chromosome", None) is not None
            ]
            latest_step = step
            latest_population = len(alive_agents)
            latest_gene_statistics = compute_gene_statistics(
                chromosomes,
                evolvable_only=True,
            )

        def _compute_step_telemetry(
            environment: Any, prev_ids: set
        ) -> Dict[str, Any]:
            """Compute selection-pressure telemetry for the current step.

            Returns a dict of extra fields for the trajectory record:
            - ``mean_reproduction_cost``: average effective reproduction cost
              over alive agents (using the current density).
            - ``realized_birth_rate``: births / previous population (0 when
              the previous population was 0).
            - ``realized_death_rate``: deaths / previous population (0 when
              the previous population was 0).
            - ``effective_selection_strength``: coefficient of variation (std /
              mean) of per-agent effective reproduction costs; a proxy for the
              opportunity-for-selection imposed by density-dependent costs.
            """

            alive_agents = list(environment.alive_agent_objects)
            current_ids = {_agent_telemetry_key(a) for a in alive_agents}

            prev_pop = len(prev_ids)
            births = len(current_ids - prev_ids)
            deaths = len(prev_ids - current_ids)

            realized_birth_rate = births / prev_pop if prev_pop > 0 else 0.0
            realized_death_rate = deaths / prev_pop if prev_pop > 0 else 0.0

            # Per-agent effective reproduction costs — only computed when at
            # least one pressure coefficient is non-zero, to avoid O(N) spatial
            # neighbour queries on every step when pressure is disabled.
            pressure = getattr(policy, "reproduction_pressure", None)
            pressure_active = pressure is not None and (
                getattr(pressure, "local_density_coefficient", 0.0) > 0.0
                or getattr(pressure, "global_carrying_capacity_coefficient", 0.0) > 0.0
            )
            effective_costs: List[float] = []
            if pressure_active:
                for agent in alive_agents:
                    repro_comp = (
                        agent.get_component("reproduction")
                        if callable(getattr(agent, "get_component", None))
                        else None
                    )
                    if repro_comp is None:
                        continue
                    base = getattr(getattr(repro_comp, "config", None), "offspring_cost", 0.0)
                    effective_costs.append(compute_effective_reproduction_cost(agent, base))

            if effective_costs:
                import statistics

                mean_cost = statistics.mean(effective_costs)
                if len(effective_costs) > 1 and mean_cost > 0:
                    std_cost = statistics.stdev(effective_costs)
                    sel_strength = std_cost / mean_cost
                else:
                    sel_strength = 0.0
            else:
                mean_cost = 0.0
                sel_strength = 0.0

            return {
                "mean_reproduction_cost": mean_cost,
                "realized_birth_rate": realized_birth_rate,
                "realized_death_rate": realized_death_rate,
                "effective_selection_strength": sel_strength,
            }

        def _on_environment_ready(environment: Any) -> None:
            nonlocal prev_agent_ids, latest_diversity_metrics
            environment.intrinsic_evolution_policy = policy
            environment.intrinsic_evolution_rng = rng
            # Initial diversity seeding now happens inside run_simulation
            # before this hook fires; capture the metrics so they can be
            # included in the persisted experiment metadata.
            latest_diversity_metrics = getattr(
                environment, "initial_diversity_metrics", None
            )
            # At step 0 there is no previous step, so birth/death rates are 0.
            alive_agents = list(environment.alive_agent_objects)
            prev_agent_ids = {_agent_telemetry_key(a) for a in alive_agents}
            # Only record the initial snapshot when there is no warmup phase;
            # warmup runs silence the logger until the population has stabilized.
            if effective_warmup == 0:
                gene_logger.snapshot(environment, step=0)
            _capture_current_state(environment, step=0)

        def _on_step_end(environment: Any, step: int) -> None:
            nonlocal prev_agent_ids
            logical_step = step + 1
            extra = _compute_step_telemetry(environment, prev_agent_ids)
            alive_agents = list(environment.alive_agent_objects)
            prev_agent_ids = {_agent_telemetry_key(a) for a in alive_agents}

            # Accumulate startup-transient metrics for the first
            # transient_window post-environment-ready steps (regardless of
            # whether those steps fall inside a warmup phase).
            if logical_step <= transient_window:
                _transient_birth_rates.append(extra["realized_birth_rate"])
                _transient_death_rates.append(extra["realized_death_rate"])
                _transient_populations.append(len(alive_agents))

            # Suppress trajectory snapshots during the warmup phase so
            # persisted artifacts only cover the stabilized population.
            if logical_step > effective_warmup:
                post_warmup_step = logical_step - effective_warmup
                gene_logger.snapshot(environment, step=post_warmup_step, extra_fields=extra)

            _capture_current_state(environment, step=logical_step)

        try:
            run_simulation(
                num_steps=total_sim_steps,
                config=run_config,
                path=self.config.output_dir,
                save_config=False,
                seed=self.config.seed,
                on_environment_ready=_on_environment_ready,
                on_step_end=_on_step_end,
            )
        finally:
            self._last_speciation_config = {
                "enabled": bool(getattr(gene_logger, "_enable_speciation", False)),
                "algorithm": getattr(gene_logger, "_speciation_algorithm", None),
                "max_k": getattr(gene_logger, "_speciation_max_k", None),
                "seed": getattr(gene_logger, "_speciation_seed", None),
                "scaler": getattr(gene_logger, "_speciation_scaler", None),
            }
            gene_logger.close()

        startup_transient_metrics = _compute_startup_transient_metrics(
            _transient_birth_rates,
            _transient_death_rates,
            _transient_populations,
            transient_window,
        )

        # num_steps_completed is relative to the post-warmup window.
        post_warmup_latest = max(0, latest_step - effective_warmup)
        result = IntrinsicEvolutionResult(
            num_steps_completed=min(self.config.num_steps, post_warmup_latest),
            final_population=latest_population,
            final_gene_statistics=latest_gene_statistics,
            startup_transient_metrics=startup_transient_metrics,
        )
        self._persist(
            result,
            initial_diversity_config=run_config.initial_diversity,
            initial_diversity_metrics=latest_diversity_metrics,
            resolved_initial_conditions=resolved_ic,
        )
        logger.info(
            "intrinsic_evolution_completed",
            output_dir=self.config.output_dir,
            num_steps_completed=result.num_steps_completed,
            final_population=result.final_population,
        )
        return result

    def _persist(
        self,
        result: IntrinsicEvolutionResult,
        *,
        initial_diversity_config: InitialDiversityConfig,
        initial_diversity_metrics: Optional[Any] = None,
        resolved_initial_conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.config.output_dir:
            return
        metadata_path = os.path.join(
            self.config.output_dir, "intrinsic_evolution_metadata.json"
        )
        payload: Dict[str, Any] = {
            "num_steps_completed": result.num_steps_completed,
            "num_steps_configured": self.config.num_steps,
            "snapshot_interval": self.config.snapshot_interval,
            "final_population": result.final_population,
            "final_gene_statistics": result.final_gene_statistics,
            "startup_transient_metrics": result.startup_transient_metrics,
            "policy": _serialize_policy(self.config.policy),
            "initial_conditions": self.config.initial_conditions.to_dict(),
            "resolved_initial_conditions": resolved_initial_conditions,
            "initial_diversity": initial_diversity_config.to_dict(),
            "initial_diversity_metrics": (
                initial_diversity_metrics.to_dict()
                if initial_diversity_metrics is not None
                else None
            ),
            "speciation": dict(self._last_speciation_config),
            "seed": self.config.seed,
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _serialize_policy(policy: IntrinsicEvolutionPolicy) -> Dict[str, Any]:
    """Render policy as a JSON-friendly dict, coercing enums to their values."""
    raw = asdict(policy)
    for key in ("mutation_mode", "boundary_mode", "crossover_mode"):
        value = raw.get(key)
        if hasattr(value, "value"):
            raw[key] = value.value
    return raw
