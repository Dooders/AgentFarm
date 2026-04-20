"""Adaptive mutation rate/scale controller for hyperparameter evolution.

This module implements adaptive mutation schedules that respond to search
state during generation-based hyperparameter evolution:

- **Generation-level adaptation** based on recent best-fitness improvement.
  Mutation pressure is reduced when fitness is improving and increased when
  search has stalled.
- **Diversity-aware adaptation** based on population spread.  When the
  normalized gene diversity collapses below a configured threshold, the
  mutation rate and scale are boosted to encourage exploration.
- **Per-gene adaptation** via user-supplied multipliers that scale the
  resolved mutation probability and scale for individual loci.

The controller is intentionally stateless apart from the bounded multipliers
it updates each generation.  It is composed into
:class:`farm.runners.evolution_experiment.EvolutionExperiment` and used to
derive effective mutation parameters before each child population is produced.

Design notes:
    - All multipliers are clamped to configurable ``[min, max]`` envelopes
      to prevent runaway adaptation.  When a clamp actually changes the
      value, the controller emits a ``rate_clamped``/``scale_clamped`` tag
      in :attr:`AdaptiveMutationController.last_event` so analysts can see
      when the bounds bite.
    - Diversity is computed as the mean, across evolvable genes, of the
      per-gene population standard deviation normalized by the gene's bounded
      range.  A value of ``0.0`` means the population has collapsed on every
      evolvable gene; ``~0.29`` is the normalized std of a uniform
      distribution on ``[0, 1]`` and represents a highly diverse population.
    - Fitness improvement is measured over a trailing window as the
      difference between the most recent ``best_fitness`` and the maximum
      ``best_fitness`` observed in the ``stall_window`` generations
      immediately preceding it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import List, Mapping, Optional, Sequence, Tuple

from farm.core.hyperparameter_chromosome import validate_non_negative_mapping


def _freeze_mapping(mapping: Mapping[str, float]) -> Mapping[str, float]:
    """Return an immutable view of ``mapping`` so frozen dataclasses stay frozen."""
    return MappingProxyType(dict(mapping))


# Built-in per-gene defaults tuned for the sensitivity of each evolvable locus.
#
# ``learning_rate`` uses log-space encoding but even so its effective value is
# very sensitive to mutation scale near the lower bound.  Halving the scale
# multiplier reduces chaotic jumps when exploration pressure is otherwise high.
#
# ``gamma`` and ``epsilon_decay`` are linear-encoded but both live near the
# boundary at 1.0, making overshoots likely when the global scale is large.
# A 0.75× scale softens those boundary hits without suppressing evolution.
#
# Users who set ``use_default_per_gene_multipliers=True`` receive these mappings
# merged with (but overridable by) their own per-gene configuration.
DEFAULT_PER_GENE_SCALE_MULTIPLIERS: Mapping[str, float] = _freeze_mapping(
    {
        "learning_rate": 0.5,
        "gamma": 0.75,
        "epsilon_decay": 0.75,
    }
)

# Rate defaults keep per-gene mutation probability unchanged; callers who want
# to suppress certain loci entirely can override via per_gene_rate_multipliers.
DEFAULT_PER_GENE_RATE_MULTIPLIERS: Mapping[str, float] = _freeze_mapping(
    {
        "learning_rate": 1.0,
        "gamma": 1.0,
        "epsilon_decay": 1.0,
    }
)


@dataclass(frozen=True)
class AdaptiveMutationConfig:
    """Configuration for adaptive mutation rate/scale schedules.

    When ``enabled`` is ``False`` (default), the experiment behaves exactly as
    before: ``mutation_rate`` and ``mutation_scale`` are used verbatim and no
    per-gene multipliers are applied.  This keeps the feature opt-in and
    existing runs reproducible.

    Attributes:
        enabled: Master switch for adaptive mutation behavior.
        use_fitness_adaptation: If ``True`` and ``enabled`` is ``True``,
            multiply mutation rate/scale by ``stall_multiplier`` when no
            meaningful improvement is observed over the trailing
            ``stall_window`` generations, and by ``improve_multiplier``
            when best fitness improves by more than
            ``improvement_threshold``.
        use_diversity_adaptation: If ``True`` and ``enabled`` is ``True``,
            multiply mutation rate/scale by ``diversity_multiplier`` whenever
            the mean normalized gene diversity falls below
            ``diversity_threshold``.
        stall_window: Number of generations to look back for improvement.
            Must be >= 1. During startup, when fewer than
            ``stall_window + 1`` observations exist, the controller uses the
            available history so adaptation can begin without waiting for a
            full window.
        improvement_threshold: Minimum absolute increase in best fitness over
            the window that counts as "improving".  Increases smaller than
            (or equal to) this value are treated as a stall.  Must be >= 0.
        stall_multiplier: Multiplier applied to the current rate/scale
            multipliers when the search stalls.  Values > 1 encourage
            exploration.  Must be > 0.
        improve_multiplier: Multiplier applied to the current rate/scale
            multipliers when the search is clearly improving.  Values < 1
            encourage exploitation.  Must be > 0.
        diversity_threshold: Normalized diversity value at or below which the
            population is considered to be collapsing.  Must be in
            ``[0.0, 1.0]``.
        diversity_multiplier: Multiplier applied to the current rate/scale
            multipliers when diversity collapses.  Values > 1 broaden the
            search.  Must be > 0.
        min_rate_multiplier: Lower bound on the accumulated rate multiplier.
            Must be > 0.
        max_rate_multiplier: Upper bound on the accumulated rate multiplier.
            Must be >= ``min_rate_multiplier``.
        min_scale_multiplier: Lower bound on the accumulated scale
            multiplier.  Must be > 0.
        max_scale_multiplier: Upper bound on the accumulated scale multiplier.
            Must be >= ``min_scale_multiplier``.
        per_gene_rate_multipliers: Optional mapping of gene name to a
            non-negative multiplier applied to that gene's mutation
            probability at mutation time.  Missing genes keep their resolved
            probability unchanged.  Always applied when ``enabled`` is
            ``True``, regardless of fitness/diversity adaptation flags.
        per_gene_scale_multipliers: Optional mapping of gene name to a
            non-negative multiplier applied to that gene's mutation scale at
            mutation time.  Missing genes keep their resolved scale
            unchanged.
        max_step_multiplier: Maximum factor by which the accumulated
            rate/scale multiplier may change in a single generation.  The
            raw factor derived from ``stall_multiplier`` or
            ``improve_multiplier`` is clamped to
            ``[1/max_step_multiplier, max_step_multiplier]`` before being
            applied, so large step sizes in either direction are dampened.
            Must be >= 1.0.  Set to a large value (e.g. ``1e9``) to
            effectively disable damping.
        use_default_per_gene_multipliers: When ``True`` and ``enabled`` is
            ``True``, the built-in :data:`DEFAULT_PER_GENE_SCALE_MULTIPLIERS`
            and :data:`DEFAULT_PER_GENE_RATE_MULTIPLIERS` constants are
            applied as a baseline for ``learning_rate``, ``gamma``, and
            ``epsilon_decay``.  Any keys present in ``per_gene_rate_multipliers``
            or ``per_gene_scale_multipliers`` take precedence over the
            defaults.
    """

    enabled: bool = False
    use_fitness_adaptation: bool = True
    use_diversity_adaptation: bool = True
    stall_window: int = 3
    improvement_threshold: float = 1e-6
    stall_multiplier: float = 1.5
    improve_multiplier: float = 0.8
    diversity_threshold: float = 0.05
    diversity_multiplier: float = 1.5
    min_rate_multiplier: float = 0.1
    max_rate_multiplier: float = 5.0
    min_scale_multiplier: float = 0.1
    max_scale_multiplier: float = 5.0
    per_gene_rate_multipliers: Mapping[str, float] = field(default_factory=dict)
    per_gene_scale_multipliers: Mapping[str, float] = field(default_factory=dict)
    max_step_multiplier: float = 2.0
    use_default_per_gene_multipliers: bool = False

    def __post_init__(self) -> None:
        if self.stall_window < 1:
            raise ValueError("stall_window must be at least 1.")
        if self.improvement_threshold < 0.0:
            raise ValueError("improvement_threshold must be non-negative.")
        if self.stall_multiplier <= 0.0:
            raise ValueError("stall_multiplier must be positive.")
        if self.improve_multiplier <= 0.0:
            raise ValueError("improve_multiplier must be positive.")
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError("diversity_threshold must be in [0.0, 1.0].")
        if self.diversity_multiplier <= 0.0:
            raise ValueError("diversity_multiplier must be positive.")
        if self.min_rate_multiplier <= 0.0:
            raise ValueError("min_rate_multiplier must be positive.")
        if self.max_rate_multiplier < self.min_rate_multiplier:
            raise ValueError("max_rate_multiplier must be >= min_rate_multiplier.")
        if self.min_scale_multiplier <= 0.0:
            raise ValueError("min_scale_multiplier must be positive.")
        if self.max_scale_multiplier < self.min_scale_multiplier:
            raise ValueError("max_scale_multiplier must be >= min_scale_multiplier.")
        if self.max_step_multiplier < 1.0:
            raise ValueError("max_step_multiplier must be >= 1.0.")
        validate_non_negative_mapping("per_gene_rate_multipliers", self.per_gene_rate_multipliers)
        validate_non_negative_mapping("per_gene_scale_multipliers", self.per_gene_scale_multipliers)
        # Re-bind to immutable views so the frozen dataclass is actually immutable.
        object.__setattr__(self, "per_gene_rate_multipliers", _freeze_mapping(self.per_gene_rate_multipliers))
        object.__setattr__(self, "per_gene_scale_multipliers", _freeze_mapping(self.per_gene_scale_multipliers))


def compute_normalized_diversity(
    gene_statistics: Mapping[str, Mapping[str, float]],
    evolvable_gene_names: Sequence[str],
    gene_bounds: Mapping[str, Tuple[float, float]],
) -> float:
    """Compute mean normalized standard deviation across evolvable genes.

    For each evolvable gene, divides the gene's population standard deviation
    by its ``(max_value - min_value)`` range.  Genes with a zero span are
    skipped.  Returns ``0.0`` when no evolvable gene contributes (e.g., empty
    population or all spans zero).

    Args:
        gene_statistics: Mapping from gene name to per-gene stats dict as
            produced by ``EvolutionExperiment._build_gene_statistics``.  Must
            contain a ``"std"`` key for each included gene.
        evolvable_gene_names: Names of genes that should contribute to the
            diversity measure.
        gene_bounds: Mapping from gene name to a ``(min_value, max_value)``
            tuple describing the allowable range.

    Returns:
        A non-negative float, typically in ``[0, ~0.29]``.
    """
    normalized_values: List[float] = []
    for gene_name in evolvable_gene_names:
        stats = gene_statistics.get(gene_name)
        if stats is None:
            continue
        bounds = gene_bounds.get(gene_name)
        if bounds is None:
            continue
        min_value, max_value = bounds
        span = max_value - min_value
        if span <= 0.0:
            continue
        std = stats.get("std", 0.0)
        normalized_values.append(max(0.0, std) / span)
    if not normalized_values:
        return 0.0
    return statistics.mean(normalized_values)


class AdaptiveMutationController:
    """Stateful controller that updates mutation multipliers per generation.

    The controller maintains accumulated ``rate_multiplier`` and
    ``scale_multiplier`` values.  Call :meth:`observe` once per evaluated
    generation (after fitness has been computed) to update those multipliers
    based on best-fitness history and population diversity.  Use
    :meth:`effective_rate` and :meth:`effective_scale` to derive the
    mutation parameters that should be passed to
    :func:`farm.core.hyperparameter_chromosome.mutate_chromosome` when
    producing the next generation.

    The controller always applies its bounds to keep adaptation stable across
    many generations, even when the ``enabled`` flag is ``False`` (in which
    case multipliers stay pinned at ``1.0``).
    """

    def __init__(self, config: AdaptiveMutationConfig):
        self._config = config
        self._best_fitness_history: List[float] = []
        self._rate_multiplier = 1.0
        self._scale_multiplier = 1.0
        self._last_diversity: Optional[float] = None
        self._last_event: str = "baseline"
        self._last_fitness_delta: Optional[float] = None

    @property
    def config(self) -> AdaptiveMutationConfig:
        return self._config

    @property
    def rate_multiplier(self) -> float:
        return self._rate_multiplier

    @property
    def scale_multiplier(self) -> float:
        return self._scale_multiplier

    @property
    def last_diversity(self) -> Optional[float]:
        return self._last_diversity

    @property
    def last_fitness_delta(self) -> Optional[float]:
        """Fitness improvement observed in the most recent :meth:`observe` call.

        Positive means fitness improved; zero or negative means it stalled.
        ``None`` when :meth:`observe` has not been called yet or when
        ``use_fitness_adaptation`` is ``False``.
        """
        return self._last_fitness_delta

    @property
    def last_event(self) -> str:
        """Human-readable tag describing the most recent adaptation action."""
        return self._last_event

    def observe(self, *, best_fitness: float, diversity: Optional[float]) -> None:
        """Update state based on a newly completed generation.

        Args:
            best_fitness: Best (adjusted) fitness observed in the generation.
            diversity: Normalized diversity measure for the generation (see
                :func:`compute_normalized_diversity`).  May be ``None`` when
                diversity cannot be computed (e.g., all genes fixed).
        """
        self._best_fitness_history.append(float(best_fitness))
        self._last_diversity = diversity
        self._last_fitness_delta = None
        events: List[str] = []

        if not self._config.enabled:
            self._last_event = "disabled"
            return

        if self._config.use_fitness_adaptation and len(self._best_fitness_history) >= 2:
            window = self._best_fitness_history[-(self._config.stall_window + 1):]
            prior_best = max(window[:-1])
            latest_best = window[-1]
            improvement = latest_best - prior_best
            self._last_fitness_delta = improvement
            if improvement > self._config.improvement_threshold:
                raw_factor = self._config.improve_multiplier
                events.append("improving")
            else:
                raw_factor = self._config.stall_multiplier
                events.append("stalled")
            # Dampen: clamp per-step factor to [1/max_step, max_step] so
            # a single generation cannot swing the multiplier too far in
            # either direction, reducing oscillation.
            max_step = self._config.max_step_multiplier
            bounded_factor = min(max_step, max(1.0 / max_step, raw_factor))
            self._rate_multiplier *= bounded_factor
            self._scale_multiplier *= bounded_factor

        if (
            self._config.use_diversity_adaptation
            and diversity is not None
            and diversity <= self._config.diversity_threshold
        ):
            self._rate_multiplier *= self._config.diversity_multiplier
            self._scale_multiplier *= self._config.diversity_multiplier
            events.append("diversity_collapse")

        # Clamp multipliers and tag whenever the clamp actually moved the
        # value, so analysts can spot saturation in `evolution_generation_summaries.json`.
        clamped_rate = min(
            self._config.max_rate_multiplier,
            max(self._config.min_rate_multiplier, self._rate_multiplier),
        )
        if clamped_rate != self._rate_multiplier:
            events.append("rate_clamped")
        self._rate_multiplier = clamped_rate

        clamped_scale = min(
            self._config.max_scale_multiplier,
            max(self._config.min_scale_multiplier, self._scale_multiplier),
        )
        if clamped_scale != self._scale_multiplier:
            events.append("scale_clamped")
        self._scale_multiplier = clamped_scale

        self._last_event = "+".join(events) if events else "baseline"

    def effective_rate(self, base_rate: float) -> float:
        """Return the current effective mutation rate, clamped to ``[0, 1]``."""
        return max(0.0, min(1.0, base_rate * self._rate_multiplier))

    def effective_scale(self, base_scale: float) -> float:
        """Return the current effective mutation scale, clamped to ``>= 0``."""
        return max(0.0, base_scale * self._scale_multiplier)

    def per_gene_rate_multipliers(self) -> Mapping[str, float]:
        """Return the effective per-gene rate multipliers (empty if disabled).

        When ``use_default_per_gene_multipliers`` is ``True``, the built-in
        :data:`DEFAULT_PER_GENE_RATE_MULTIPLIERS` are used as a baseline.
        Any keys present in the config's ``per_gene_rate_multipliers`` take
        precedence over the defaults.
        """
        if not self._config.enabled:
            return {}
        if not self._config.use_default_per_gene_multipliers:
            return self._config.per_gene_rate_multipliers
        merged = dict(DEFAULT_PER_GENE_RATE_MULTIPLIERS)
        merged.update(self._config.per_gene_rate_multipliers)
        return merged

    def per_gene_scale_multipliers(self) -> Mapping[str, float]:
        """Return the effective per-gene scale multipliers (empty if disabled).

        When ``use_default_per_gene_multipliers`` is ``True``, the built-in
        :data:`DEFAULT_PER_GENE_SCALE_MULTIPLIERS` are used as a baseline.
        Any keys present in the config's ``per_gene_scale_multipliers`` take
        precedence over the defaults.
        """
        if not self._config.enabled:
            return {}
        if not self._config.use_default_per_gene_multipliers:
            return self._config.per_gene_scale_multipliers
        merged = dict(DEFAULT_PER_GENE_SCALE_MULTIPLIERS)
        merged.update(self._config.per_gene_scale_multipliers)
        return merged
