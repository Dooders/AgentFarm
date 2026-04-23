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
from typing import Any, Dict, List, Literal, Optional

from farm.config import SimulationConfig
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
        # Coerce string-friendly enum values, raising if invalid.
        MutationMode(self.mutation_mode)
        BoundaryMode(self.boundary_mode)
        CrossoverMode(self.crossover_mode)
        if self.coparent_strategy not in ("nearest_alive_same_type", "random_alive_same_type"):
            raise ValueError(
                "coparent_strategy must be 'nearest_alive_same_type' or 'random_alive_same_type'."
            )


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
    sees the seeded values.
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
        agent.config.decision = apply_chromosome_to_learning_config(
            agent.config.decision, mutated
        )


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

        def _on_environment_ready(environment: Any) -> None:
            environment.intrinsic_evolution_policy = policy
            environment.intrinsic_evolution_rng = rng
            seed_population_diversity(environment, policy, rng)
            gene_logger.snapshot(environment, step=0)

        def _on_step_end(environment: Any, step: int) -> None:
            gene_logger.snapshot(environment, step=step + 1)

        try:
            environment = run_simulation(
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

        final_alive = environment.alive_agent_objects
        final_chromosomes: List[HyperparameterChromosome] = [
            agent.hyperparameter_chromosome
            for agent in final_alive
            if getattr(agent, "hyperparameter_chromosome", None) is not None
        ]
        result = IntrinsicEvolutionResult(
            num_steps_completed=min(
                self.config.num_steps,
                max(0, int(environment.time) - 1),
            ),
            final_population=len(final_alive),
            final_gene_statistics=compute_gene_statistics(
                final_chromosomes, evolvable_only=True
            ),
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
