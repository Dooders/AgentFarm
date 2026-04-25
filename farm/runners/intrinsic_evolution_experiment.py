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
    apply_chromosome_to_learning_config,
    compute_gene_statistics,
    mutate_chromosome,
)
from farm.core.simulation import run_simulation
from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger
from farm.utils.logging import get_logger

logger = get_logger(__name__)


CoparentStrategy = Literal["nearest_alive_same_type", "random_alive_same_type"]
SelectionPressurePreset = Literal["none", "low", "medium", "high"]

# ── Preset definitions ────────────────────────────────────────────────────────
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
        seed_initial_diversity: When ``True``, the runner mutates every
            initial agent's chromosome once before the loop starts so the
            starting population is not a monoculture.
        seed_mutation_rate / seed_mutation_scale: Mutation knobs used **only**
            for the initial diversity seed.  Defaults are intentionally larger
            than per-reproduction values to spread the starting population.
        mutation_rate / mutation_scale / mutation_mode: Per-reproduction
            mutation parameters threaded through to
            :func:`mutate_chromosome`.
        boundary_mode / interior_bias_fraction: Boundary handling for mutated
            values; matches semantics in :class:`BoundaryMode`.
        crossover_enabled: When ``True``, reproduction picks a co-parent and
            runs :func:`crossover_chromosomes` before mutation.
        crossover_mode / blend_alpha / num_crossover_points: Crossover
            operator settings (see :class:`CrossoverMode`).
        coparent_strategy: How to pick the co-parent.  Both strategies filter
            to alive agents of the same ``agent_type``; ``nearest`` is
            deterministic-modulo-tiebreak and reflects spatial sociality,
            ``random`` is uniform over the candidate pool.
        coparent_max_radius: Optional spatial cap on the co-parent search;
            ``None`` means unbounded.
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
    seed_initial_diversity: bool = True
    seed_mutation_rate: float = 1.0
    seed_mutation_scale: float = 0.2
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
        if not 0.0 <= self.seed_mutation_rate <= 1.0:
            raise ValueError("seed_mutation_rate must be between 0 and 1.")
        if self.seed_mutation_scale < 0.0:
            raise ValueError("seed_mutation_scale must be non-negative.")
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
    """Top-level config for the intrinsic evolution runner."""

    num_steps: int = 2000
    snapshot_interval: int = 100
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
    """

    num_steps_completed: int
    final_population: int
    final_gene_statistics: Dict[str, Dict[str, float]]


def seed_population_diversity(
    environment: Any,
    policy: IntrinsicEvolutionPolicy,
    rng: random.Random,
) -> None:
    """Mutate each initial agent's chromosome to spread the starting population.

    When ``policy.seed_initial_diversity`` is ``False`` this is a no-op.
    Both the agent's ``hyperparameter_chromosome`` attribute and its
    ``config.decision`` are updated so subsequent decision-module construction
    sees the seeded values.  If the agent already has a ``LearningAgentBehavior``
    with a ``DecisionModule`` (i.e., the module was constructed before seeding),
    the module's config is updated and its algorithm is reinitialized so that
    optimizer hyper-parameters (LR, gamma, …) reflect the seeded chromosome.
    """
    if not policy.seed_initial_diversity:
        return

    for agent in list(environment.agent_objects):
        chromosome = getattr(agent, "hyperparameter_chromosome", None)
        if chromosome is None:
            continue
        mutated = mutate_chromosome(
            chromosome,
            mutation_rate=policy.seed_mutation_rate,
            mutation_scale=policy.seed_mutation_scale,
            mutation_mode=policy.mutation_mode,
            boundary_mode=policy.boundary_mode,
            interior_bias_fraction=policy.interior_bias_fraction,
            rng=rng,
        )
        agent.hyperparameter_chromosome = mutated
        new_decision_config = apply_chromosome_to_learning_config(
            agent.config.decision, mutated
        )
        agent.config.decision = new_decision_config

        # If a decision module was already constructed (common when
        # on_environment_ready fires after create_initial_agents), update its
        # config and reinitialize the algorithm so the optimizer reflects the
        # seeded hyperparameters rather than the pre-seed defaults.
        behavior = getattr(agent, "behavior", None)
        decision_module = getattr(behavior, "decision_module", None)
        reinitialize = getattr(decision_module, "reinitialize_algorithm", None)
        if callable(reinitialize):
            reinitialize(new_decision_config)


class IntrinsicEvolutionExperiment:
    """Run a single simulation in which agents carry inheritable chromosomes."""

    def __init__(
        self,
        base_config: SimulationConfig,
        config: IntrinsicEvolutionExperimentConfig,
    ) -> None:
        self.base_config = base_config
        self.config = config

    def run(self) -> IntrinsicEvolutionResult:
        """Execute the configured number of steps and persist artifacts."""
        seed = self.config.seed if self.config.policy.seed is None else self.config.policy.seed
        rng = random.Random(seed)

        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)

        gene_logger = GeneTrajectoryLogger(
            output_dir=self.config.output_dir,
            snapshot_interval=self.config.snapshot_interval,
        )

        policy = self.config.policy
        latest_step = 0
        latest_population = 0
        latest_gene_statistics: Dict[str, Dict[str, float]] = {}

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
            nonlocal prev_agent_ids
            environment.intrinsic_evolution_policy = policy
            environment.intrinsic_evolution_rng = rng
            seed_population_diversity(environment, policy, rng)
            # At step 0 there is no previous step, so birth/death rates are 0.
            alive_agents = list(environment.alive_agent_objects)
            prev_agent_ids = {_agent_telemetry_key(a) for a in alive_agents}
            gene_logger.snapshot(environment, step=0)
            _capture_current_state(environment, step=0)

        def _on_step_end(environment: Any, step: int) -> None:
            nonlocal prev_agent_ids
            logical_step = step + 1
            extra = _compute_step_telemetry(environment, prev_agent_ids)
            alive_agents = list(environment.alive_agent_objects)
            prev_agent_ids = {_agent_telemetry_key(a) for a in alive_agents}
            gene_logger.snapshot(environment, step=logical_step, extra_fields=extra)
            _capture_current_state(environment, step=logical_step)

        try:
            run_simulation(
                num_steps=self.config.num_steps,
                config=self.base_config,
                path=self.config.output_dir,
                save_config=False,
                seed=self.config.seed,
                on_environment_ready=_on_environment_ready,
                on_step_end=_on_step_end,
            )
        finally:
            gene_logger.close()

        result = IntrinsicEvolutionResult(
            num_steps_completed=min(self.config.num_steps, max(0, latest_step)),
            final_population=latest_population,
            final_gene_statistics=latest_gene_statistics,
        )
        self._persist(result)
        logger.info(
            "intrinsic_evolution_completed",
            output_dir=self.config.output_dir,
            num_steps_completed=result.num_steps_completed,
            final_population=result.final_population,
        )
        return result

    def _persist(self, result: IntrinsicEvolutionResult) -> None:
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
            "policy": _serialize_policy(self.config.policy),
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
