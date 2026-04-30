"""Platform-wide initial genotype diversity for simulation startup.

This module implements the seeding feature originally introduced for the
intrinsic-evolution runner as a first-class capability available to **every**
simulation.  Callers opt in via :class:`InitialDiversityConfig` on
:class:`farm.config.SimulationConfig`; :func:`apply_initial_diversity` is
invoked from :func:`farm.core.simulation.run_simulation` immediately after the
initial agent population is created and before any ``on_environment_ready``
hook fires.

Design notes (see ``docs/initial_diversity.md`` for user-facing docs):

- :class:`InitialDiversitySource` is a narrow ``Protocol`` so additional
  genotype scopes (per-agent type parameters, decision-module weights,
  spatial layouts, etc.) can ship later without changing the orchestration
  contract or :class:`InitialDiversityConfig`.
- :class:`ChromosomeDiversitySource` is the default implementation and reuses
  the existing :func:`farm.core.hyperparameter_chromosome.mutate_chromosome`
  primitive.  It supports four modes:

  - ``NONE``: no-op; callers receive zeroed metrics.
  - ``INDEPENDENT_MUTATION``: matches the legacy intrinsic-evolution seeding.
  - ``UNIQUE``: bounded retries until each accepted chromosome is novel
    against all previously accepted ones at the gene encoding precision.
  - ``MIN_DISTANCE``: bounded retries until each accepted chromosome is at
    least ``cfg.min_distance`` away (normalized Euclidean over evolvable
    genes) from every previously accepted chromosome.

  Strict modes always accept the latest sample once retries are exhausted
  and increment ``fallbacks`` so callers can audit how often the constraint
  could not be satisfied.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, List, Mapping, Optional, Protocol, Tuple

from farm.core.hyperparameter_chromosome import (
    BoundaryMode,
    GeneEncodingSpec,
    HyperparameterChromosome,
    MutationMode,
    apply_chromosome_to_learning_config,
    encode_chromosome_vector,
    mutate_chromosome,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class SeedingMode(str, Enum):
    """Strategies for seeding initial genotype diversity.

    - ``NONE``: leave the starting population untouched.
    - ``INDEPENDENT_MUTATION``: mutate each agent independently.  Matches the
      legacy intrinsic-evolution seeding behaviour.
    - ``UNIQUE``: independent mutation with bounded retries until each
      chromosome is unique at the gene encoding precision.
    - ``MIN_DISTANCE``: independent mutation with bounded retries until each
      chromosome is at least ``min_distance`` away (normalized Euclidean
      across evolvable genes) from all previously accepted chromosomes.
    """

    NONE = "none"
    INDEPENDENT_MUTATION = "independent_mutation"
    UNIQUE = "unique"
    MIN_DISTANCE = "min_distance"


@dataclass(frozen=True)
class InitialDiversityConfig:
    """Configuration for platform-wide initial genotype diversity seeding.

    Attributes:
        mode: Which seeding strategy to apply.  Default ``NONE`` keeps
            existing simulations unchanged.
        mutation_rate: Probability of mutating each evolvable gene per draw.
            Must be in ``[0, 1]``.
        mutation_scale: Perturbation scale passed to ``mutate_chromosome``.
            Must be non-negative.
        mutation_mode: Perturbation operator (``gaussian`` or
            ``multiplicative``).
        boundary_mode: Out-of-range handling for mutated values.
        interior_bias_fraction: Inward nudge fraction used by
            :attr:`BoundaryMode.INTERIOR_BIASED`.  Must be non-negative.
        max_retries_per_agent: Upper bound on retries per agent in strict
            modes (``UNIQUE`` / ``MIN_DISTANCE``).  Must be at least 1.
        min_distance: Minimum normalized Euclidean distance between any two
            accepted chromosomes in ``MIN_DISTANCE`` mode.  Must be
            non-negative; values larger than ``sqrt(num_evolvable_genes)``
            are unsatisfiable in the unit hypercube.
        seed: Optional override for the seeding RNG; when ``None`` the
            simulation seed is used so the same simulation seed yields the
            same initial population.
    """

    mode: SeedingMode = SeedingMode.NONE
    mutation_rate: float = 1.0
    mutation_scale: float = 0.2
    mutation_mode: MutationMode = MutationMode.GAUSSIAN
    boundary_mode: BoundaryMode = BoundaryMode.CLAMP
    interior_bias_fraction: float = 1e-3
    max_retries_per_agent: int = 32
    min_distance: float = 0.05
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if self.mutation_scale < 0.0:
            raise ValueError("mutation_scale must be non-negative.")
        if self.interior_bias_fraction < 0.0:
            raise ValueError("interior_bias_fraction must be non-negative.")
        if self.max_retries_per_agent < 1:
            raise ValueError("max_retries_per_agent must be at least 1.")
        if self.min_distance < 0.0:
            raise ValueError("min_distance must be non-negative.")
        # Coerce string-friendly enum values to enum instances; raises for invalid values.
        object.__setattr__(self, "mode", SeedingMode(self.mode))
        object.__setattr__(self, "mutation_mode", MutationMode(self.mutation_mode))
        object.__setattr__(self, "boundary_mode", BoundaryMode(self.boundary_mode))

    def to_dict(self) -> dict:
        """Render config as a JSON-friendly dict, coercing enums to their values."""
        raw = dataclasses.asdict(self)
        for key in ("mode", "mutation_mode", "boundary_mode"):
            value = raw.get(key)
            if hasattr(value, "value"):
                raw[key] = value.value
        return raw


@dataclass
class InitialDiversityMetrics:
    """Outcome telemetry from an initial-diversity seeding pass.

    Attributes:
        mode: The seeding mode that produced this report.
        agents_processed: Number of agents whose chromosomes were considered.
        unique_count: Number of distinct chromosome signatures observed
            after seeding (at gene encoding precision).
        collision_count: Number of accepted chromosomes whose signature
            matched a previously accepted one.  Always 0 in ``UNIQUE`` mode
            unless retries were exhausted (``fallbacks > 0``).
        retries_used: Total number of failed candidate draws across all
            agents.  Zero for ``NONE`` and ``INDEPENDENT_MUTATION``.
        fallbacks: Number of agents whose final chromosome had to be
            accepted under fallback because retries were exhausted.
        min_pairwise_distance: Smallest normalized Euclidean distance
            between any two accepted chromosomes.  ``None`` when fewer
            than 2 chromosomes were processed.
        mean_pairwise_distance: Mean normalized Euclidean distance across
            all accepted chromosome pairs.  ``None`` when fewer than 2
            chromosomes were processed.
        notes: Optional human-readable diagnostics surfaced during seeding.
    """

    mode: SeedingMode
    agents_processed: int = 0
    unique_count: int = 0
    collision_count: int = 0
    retries_used: int = 0
    fallbacks: int = 0
    min_pairwise_distance: Optional[float] = None
    mean_pairwise_distance: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Render metrics as a JSON-friendly dict, coercing the mode enum."""
        raw = dataclasses.asdict(self)
        mode_value = raw.get("mode")
        if hasattr(mode_value, "value"):
            raw["mode"] = mode_value.value
        return raw


class InitialDiversitySource(Protocol):
    """Pluggable strategy that seeds diversity into the starting population.

    Implementations receive the live ``environment``, the user-supplied
    :class:`InitialDiversityConfig`, and a seeded :class:`random.Random`.
    They are expected to return :class:`InitialDiversityMetrics` describing
    the outcome.  Implementations should treat the environment as live state:
    they may mutate agent attributes in place, but should not rely on a
    specific ``Environment`` class beyond the iteration contract documented
    by ``ChromosomeDiversitySource``.
    """

    def seed(  # pragma: no cover - Protocol method
        self,
        environment: Any,
        cfg: InitialDiversityConfig,
        rng: random.Random,
    ) -> InitialDiversityMetrics: ...


def _chromosome_signature(
    chromosome: HyperparameterChromosome,
    *,
    encoding_specs: Optional[Mapping[str, GeneEncodingSpec]] = None,
) -> Tuple[float, ...]:
    """Quantize a chromosome to its gene encoding precision.

    Two chromosomes that differ only below the smallest encoding bucket map
    to the same signature, so uniqueness is meaningful at the precision the
    learning agents actually express.
    """
    return encode_chromosome_vector(
        chromosome,
        include_fixed=False,
        encoding_specs=encoding_specs,
    )


def _normalized_gene_vector(chromosome: HyperparameterChromosome) -> Tuple[float, ...]:
    """Project a chromosome's evolvable genes into the unit hypercube.

    For ``min_value == max_value`` genes the projection is ``0.0`` so the
    distance contribution is zero (the locus carries no diversity).
    """
    coords: List[float] = []
    for gene in chromosome.genes:
        if not gene.evolvable:
            continue
        span = gene.max_value - gene.min_value
        if span == 0.0:
            coords.append(0.0)
            continue
        coords.append((gene.value - gene.min_value) / span)
    return tuple(coords)


def _pairwise_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("Chromosome vectors must have matching length.")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _compute_pairwise_summary(
    vectors: List[Tuple[float, ...]],
) -> Tuple[Optional[float], Optional[float]]:
    """Return (min_distance, mean_distance) over all unordered pairs.

    Returns ``(None, None)`` for fewer than 2 vectors.  The cost is
    O(n^2), which is acceptable because seeding runs once per simulation
    over modest populations; if very large populations become routine this
    can be replaced with a spatial structure without changing the public
    metrics contract.
    """
    if len(vectors) < 2:
        return None, None
    distances: List[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distances.append(_pairwise_distance(vectors[i], vectors[j]))
    if not distances:
        return None, None
    return min(distances), statistics.fmean(distances)


def _apply_chromosome_to_agent(agent: Any, chromosome: HyperparameterChromosome) -> None:
    """Push a freshly seeded chromosome into the agent and any live module.

    This preserves the two-stage update used by the legacy intrinsic-evolution
    seeding: rewrite ``hyperparameter_chromosome``, recompute the learning
    config, and reinitialize the decision module's algorithm when one already
    exists so optimizer hyperparameters track the seeded values.
    """
    agent.hyperparameter_chromosome = chromosome
    new_decision_config = apply_chromosome_to_learning_config(
        agent.config.decision, chromosome
    )
    agent.config.decision = new_decision_config

    behavior = getattr(agent, "behavior", None)
    decision_module = getattr(behavior, "decision_module", None)
    reinitialize = getattr(decision_module, "reinitialize_algorithm", None)
    if callable(reinitialize):
        reinitialize(new_decision_config)


class ChromosomeDiversitySource:
    """Default seeding source that mutates each agent's hyperparameter chromosome.

    The source iterates ``environment.agent_objects`` in order, skipping any
    agent without a ``hyperparameter_chromosome``.  For each agent it draws
    candidate mutations via :func:`mutate_chromosome` and, depending on
    ``cfg.mode``, either accepts the first draw or retries until uniqueness
    or a minimum distance constraint is met.  Strict modes accept the last
    drawn candidate when ``cfg.max_retries_per_agent`` is exhausted and
    record a fallback.
    """

    def __init__(
        self,
        encoding_specs: Optional[Mapping[str, GeneEncodingSpec]] = None,
    ) -> None:
        self._encoding_specs = encoding_specs

    def seed(
        self,
        environment: Any,
        cfg: InitialDiversityConfig,
        rng: random.Random,
    ) -> InitialDiversityMetrics:
        if cfg.mode is SeedingMode.NONE:
            return InitialDiversityMetrics(mode=SeedingMode.NONE)

        agents: List[Any] = [
            agent
            for agent in environment.agent_objects
            if getattr(agent, "hyperparameter_chromosome", None) is not None
        ]

        metrics = InitialDiversityMetrics(mode=cfg.mode)
        accepted_signatures: set = set()
        accepted_vectors: List[Tuple[float, ...]] = []

        for agent in agents:
            chromosome = agent.hyperparameter_chromosome
            chosen, attempts, fell_back, collided = self._draw_candidate(
                chromosome=chromosome,
                cfg=cfg,
                rng=rng,
                accepted_signatures=accepted_signatures,
                accepted_vectors=accepted_vectors,
            )

            metrics.agents_processed += 1
            metrics.retries_used += max(attempts - 1, 0)
            if fell_back:
                metrics.fallbacks += 1

            signature = _chromosome_signature(chosen, encoding_specs=self._encoding_specs)
            if signature in accepted_signatures:
                metrics.collision_count += 1
            accepted_signatures.add(signature)
            accepted_vectors.append(_normalized_gene_vector(chosen))
            _apply_chromosome_to_agent(agent, chosen)

            # Late-binding sanity note: if the user asked for a minimum distance
            # but we accepted under fallback, surface the violation.
            if collided and cfg.mode is SeedingMode.UNIQUE and not fell_back:
                # Unreachable in practice because we only mark `collided=True`
                # when we exhausted retries; left as defensive bookkeeping.
                metrics.notes.append(
                    f"agent={getattr(agent, 'agent_id', '?')} accepted duplicate signature."
                )

        metrics.unique_count = len(accepted_signatures)
        min_d, mean_d = _compute_pairwise_summary(accepted_vectors)
        metrics.min_pairwise_distance = min_d
        metrics.mean_pairwise_distance = mean_d
        return metrics

    def _draw_candidate(
        self,
        *,
        chromosome: HyperparameterChromosome,
        cfg: InitialDiversityConfig,
        rng: random.Random,
        accepted_signatures: set,
        accepted_vectors: List[Tuple[float, ...]],
    ) -> Tuple[HyperparameterChromosome, int, bool, bool]:
        """Return ``(chromosome, attempts, fell_back, collided)``.

        ``attempts`` is the number of mutate_chromosome calls made (>=1).
        ``fell_back`` is True if a strict mode exhausted retries.
        ``collided`` flags an accepted-with-collision in UNIQUE mode after fallback.
        """
        last_candidate: Optional[HyperparameterChromosome] = None
        attempts = 0
        retry_budget = cfg.max_retries_per_agent if cfg.mode in (
            SeedingMode.UNIQUE,
            SeedingMode.MIN_DISTANCE,
        ) else 1

        for _ in range(retry_budget):
            attempts += 1
            candidate = mutate_chromosome(
                chromosome,
                mutation_rate=cfg.mutation_rate,
                mutation_scale=cfg.mutation_scale,
                mutation_mode=cfg.mutation_mode,
                boundary_mode=cfg.boundary_mode,
                interior_bias_fraction=cfg.interior_bias_fraction,
                rng=rng,
            )
            last_candidate = candidate

            if cfg.mode is SeedingMode.INDEPENDENT_MUTATION:
                return candidate, attempts, False, False

            if cfg.mode is SeedingMode.UNIQUE:
                signature = _chromosome_signature(candidate, encoding_specs=self._encoding_specs)
                if signature not in accepted_signatures:
                    return candidate, attempts, False, False
                continue

            # SeedingMode.MIN_DISTANCE
            vector = _normalized_gene_vector(candidate)
            ok = all(
                _pairwise_distance(vector, prev) >= cfg.min_distance
                for prev in accepted_vectors
            )
            if ok:
                return candidate, attempts, False, False

        # Fallback: strict mode exhausted retries; accept the last candidate
        # so seeding is bounded and deterministic.
        assert last_candidate is not None  # retry_budget >= 1 guarantees this
        collided = False
        if cfg.mode is SeedingMode.UNIQUE:
            signature = _chromosome_signature(last_candidate, encoding_specs=self._encoding_specs)
            collided = signature in accepted_signatures
        return last_candidate, attempts, True, collided


def apply_initial_diversity(
    environment: Any,
    cfg: InitialDiversityConfig,
    rng: random.Random,
    *,
    source: Optional[InitialDiversitySource] = None,
) -> InitialDiversityMetrics:
    """Seed the population's initial genotype diversity.

    Returns an :class:`InitialDiversityMetrics` describing the outcome.
    When ``cfg.mode`` is :attr:`SeedingMode.NONE`, returns a zeroed report
    and does not touch any agent.  Otherwise the work is delegated to
    ``source`` (defaults to :class:`ChromosomeDiversitySource`).
    """
    if cfg.mode is SeedingMode.NONE:
        return InitialDiversityMetrics(mode=SeedingMode.NONE)
    seeder = source if source is not None else ChromosomeDiversitySource()
    return seeder.seed(environment, cfg, rng)


def persist_initial_diversity_metrics(
    output_dir: str,
    metrics: InitialDiversityMetrics,
    *,
    filename: str = "initial_diversity_metadata.json",
) -> str:
    """Write metrics to ``<output_dir>/<filename>`` and return the path.

    Creates ``output_dir`` if it does not yet exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, indent=2)
    return path


def emit_initial_diversity_event(
    metrics: InitialDiversityMetrics,
    *,
    event_logger=logger,
) -> None:
    """Emit a structlog event summarizing the seeding outcome.

    Centralized here so callers (currently :func:`run_simulation`) get
    consistent field names and downstream analysis tooling can rely on
    them.
    """
    event_logger.info(
        "initial_diversity_seeded",
        **metrics.to_dict(),
    )


__all__ = [
    "ChromosomeDiversitySource",
    "InitialDiversityConfig",
    "InitialDiversityMetrics",
    "InitialDiversitySource",
    "SeedingMode",
    "apply_initial_diversity",
    "emit_initial_diversity_event",
    "persist_initial_diversity_metrics",
]
