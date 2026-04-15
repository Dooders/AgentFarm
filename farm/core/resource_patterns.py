"""
Resource regeneration patterns and ecosystem modeling for AgentFarm.

This module provides a pluggable regeneration strategy system for resources. Each
regenerator strategy implements the ``ResourceRegenerator`` interface and can be
composed or extended to model increasingly complex ecological dynamics.

Regenerator classes
-------------------
BasicRegenerator
    Simple stochastic regeneration controlled by a fixed rate.
TimeBasedRegenerator
    Regeneration at fixed step intervals.
SeasonalRegenerator
    Regeneration that follows a sinusoidal seasonal cycle.
ProximityRegenerator
    Regeneration that is boosted by nearby resources and suppressed by competitors.
ResourceDependentRegenerator
    Regeneration driven by weighted ecosystem dependencies (e.g. water + nutrients).
AdaptiveRegenerator
    Regeneration rate adapts automatically to observed consumption patterns.
EnvironmentalRegenerator
    Regeneration governed by environmental factors: temperature, moisture, light,
    and soil/substrate quality.
EcosystemRegenerator
    Full ecosystem model combining environmental conditions, mutualistic relationships,
    and competitive resource interactions.
EvolutionaryRegenerator
    Regeneration parameters that drift over time in response to environmental stress,
    simulating resource adaptation and evolution.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from farm.core.resources import Resource


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ResourceGenerationConfig:
    """Configuration parameters shared by all regenerators.

    Attributes
    ----------
    regen_rate : float
        Base probability (0–1) that a resource regenerates in any given step.
    regen_amount : int
        Base amount added when regeneration occurs.
    max_amount : int
        Hard ceiling on resource amount; regeneration stops when reached.
    min_amount : int
        Floor value; a resource is considered depleted when at or below this.
    """

    regen_rate: float = 0.1
    regen_amount: int = 1
    max_amount: int = 10
    min_amount: int = 0


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ResourceRegenerator(ABC):
    """Abstract base class for resource regeneration patterns.

    All regenerators accept a :class:`ResourceGenerationConfig` on construction
    and must implement :meth:`should_regenerate` and :meth:`get_regen_amount`.
    """

    def __init__(self, config: ResourceGenerationConfig) -> None:
        """Initialise the regenerator with the given configuration.

        Parameters
        ----------
        config : ResourceGenerationConfig
            Shared regeneration parameters.
        """
        self.config = config

    @abstractmethod
    def should_regenerate(self, resource: "Resource") -> bool:
        """Determine whether *resource* should regenerate in the current step.

        Parameters
        ----------
        resource : Resource
            The resource to evaluate.

        Returns
        -------
        bool
            ``True`` if regeneration should occur this step.
        """

    @abstractmethod
    def get_regen_amount(self, resource: "Resource") -> int:
        """Return the amount to add to *resource* when regeneration occurs.

        Parameters
        ----------
        resource : Resource
            The resource being regenerated.

        Returns
        -------
        int
            Amount to regenerate (always ≥ 1).
        """


# ---------------------------------------------------------------------------
# Concrete regenerators
# ---------------------------------------------------------------------------


class BasicRegenerator(ResourceRegenerator):
    """Simple stochastic regeneration at a fixed probability per step."""

    def should_regenerate(self, resource: "Resource") -> bool:
        if self.config.max_amount and resource.amount >= self.config.max_amount:
            return False
        return random.random() < self.config.regen_rate

    def get_regen_amount(self, resource: "Resource") -> int:
        return self.config.regen_amount


class TimeBasedRegenerator(ResourceRegenerator):
    """Regenerates at fixed step intervals.

    Parameters
    ----------
    interval : int
        Number of steps between regeneration events for each resource.
    """

    def __init__(self, config: ResourceGenerationConfig, interval: int = 10) -> None:
        super().__init__(config)
        self.interval = interval
        self._step_counter: Dict[str, int] = {}

    def should_regenerate(self, resource: "Resource") -> bool:
        if self.config.max_amount and resource.amount >= self.config.max_amount:
            return False
        rid = resource.resource_id
        self._step_counter[rid] = self._step_counter.get(rid, 0) + 1
        return self._step_counter[rid] % self.interval == 0

    def get_regen_amount(self, resource: "Resource") -> int:
        return self.config.regen_amount


class SeasonalRegenerator(ResourceRegenerator):
    """Regeneration follows a sinusoidal seasonal cycle.

    Parameters
    ----------
    season_length : int
        Number of steps in one full season cycle.
    peak_rate_multiplier : float
        Maximum multiplier applied to the base regeneration rate at seasonal peak.
    """

    def __init__(
        self,
        config: ResourceGenerationConfig,
        season_length: int = 100,
        peak_rate_multiplier: float = 2.0,
    ) -> None:
        super().__init__(config)
        self.season_length = season_length
        self.peak_rate_multiplier = peak_rate_multiplier
        self._global_step: int = 0

    def _seasonal_factor(self) -> float:
        """Sinusoidal factor in ``[0, peak_rate_multiplier]``."""
        angle = 2 * math.pi * (self._global_step % self.season_length) / self.season_length
        # Oscillates between 1 (trough, sin=-1) and peak_rate_multiplier (peak, sin=+1)
        return (1 + math.sin(angle)) / 2 * (self.peak_rate_multiplier - 1) + 1

    def should_regenerate(self, resource: "Resource") -> bool:
        if self.config.max_amount and resource.amount >= self.config.max_amount:
            return False
        self._global_step += 1
        effective_rate = self.config.regen_rate * self._seasonal_factor()
        return random.random() < effective_rate

    def get_regen_amount(self, resource: "Resource") -> int:
        factor = self._seasonal_factor()
        return max(1, int(self.config.regen_amount * factor))


class ProximityRegenerator(ResourceRegenerator):
    """Regeneration boosted by nearby friendly resources and suppressed by competitors.

    Parameters
    ----------
    boost_range : float
        Distance within which neighbouring resources provide a boost.
    competition_range : float
        Distance within which competing resources suppress regeneration.
    boost_factor : float
        Multiplier applied per neighbouring resource (capped at ``max_boost``).
    competition_factor : float
        Reduction applied per competing resource.
    max_boost : float
        Maximum cumulative boost multiplier.
    """

    def __init__(
        self,
        config: ResourceGenerationConfig,
        boost_range: float = 50.0,
        competition_range: float = 20.0,
        boost_factor: float = 0.1,
        competition_factor: float = 0.15,
        max_boost: float = 2.0,
    ) -> None:
        super().__init__(config)
        self.boost_range = boost_range
        self.competition_range = competition_range
        self.boost_factor = boost_factor
        self.competition_factor = competition_factor
        self.max_boost = max_boost

    def _proximity_factor(self, resource: "Resource") -> float:
        """Return a combined proximity factor for *resource*."""
        if not hasattr(resource, "environment") or resource.environment is None:
            return 1.0

        px, py = resource.position
        boost = 1.0
        for r in resource.environment.resources:
            if r is resource:
                continue
            dist = math.sqrt((px - r.position[0]) ** 2 + (py - r.position[1]) ** 2)
            if dist <= self.competition_range:
                boost -= self.competition_factor
            elif dist <= self.boost_range:
                boost += self.boost_factor

        return max(0.0, min(self.max_boost, boost))

    def should_regenerate(self, resource: "Resource") -> bool:
        if self.config.max_amount and resource.amount >= self.config.max_amount:
            return False
        factor = self._proximity_factor(resource)
        return random.random() < self.config.regen_rate * factor

    def get_regen_amount(self, resource: "Resource") -> int:
        factor = self._proximity_factor(resource)
        return max(1, int(self.config.regen_amount * factor))


class ResourceDependentRegenerator(ResourceRegenerator):
    """Regenerates resources based on the presence and amounts of other resources.

    Parameters
    ----------
    dependencies : Dict[str, Dict[str, float]]
        Mapping of dependency resource type to parameters::

            {
                "water":     {"weight": 1.5, "range": 50.0},
                "nutrients": {"weight": 1.0, "range": 30.0},
            }

    resource_type : str
        The type identifier of the resource being managed.
    """

    def __init__(
        self,
        config: ResourceGenerationConfig,
        dependencies: Dict[str, Dict[str, float]],
        resource_type: str = "default",
    ) -> None:
        super().__init__(config)
        self.dependencies = dependencies
        self.resource_type = resource_type

    def _evaluate_dependencies(self, resource: "Resource") -> float:
        """Calculate dependency satisfaction level (0–1)."""
        if not hasattr(resource, "environment") or resource.environment is None:
            return 1.0
        if not self.dependencies:
            return 1.0

        total_weight = sum(dep["weight"] for dep in self.dependencies.values())
        satisfaction = 0.0

        for dep_type, params in self.dependencies.items():
            weight = params["weight"]
            range_limit = params["range"]

            nearby = [
                r
                for r in resource.environment.resources
                if (
                    hasattr(r, "resource_type")
                    and r.resource_type == dep_type
                    and math.sqrt(
                        (resource.position[0] - r.position[0]) ** 2
                        + (resource.position[1] - r.position[1]) ** 2
                    )
                    <= range_limit
                )
            ]

            if nearby:
                avg_amount = sum(r.amount for r in nearby) / len(nearby)
                satisfaction += weight * min(1.0, avg_amount / self.config.max_amount)

        return satisfaction / total_weight if total_weight > 0 else 1.0

    def should_regenerate(self, resource: "Resource") -> bool:
        if self.config.max_amount and resource.amount >= self.config.max_amount:
            return False
        dependency_factor = self._evaluate_dependencies(resource)
        modified_rate = self.config.regen_rate * dependency_factor
        return random.random() < modified_rate

    def get_regen_amount(self, resource: "Resource") -> int:
        dependency_factor = self._evaluate_dependencies(resource)
        modified_amount = self.config.regen_amount * dependency_factor
        return max(1, int(modified_amount))


class AdaptiveRegenerator(ResourceRegenerator):
    """Adapts regeneration based on observed resource consumption patterns.

    The regenerator maintains a rolling history of consumption events for each
    resource. When consumption is high the regeneration rate is increased; when
    consumption is low it is decreased, keeping the rate within ``[0.01, 0.99]``.

    Parameters
    ----------
    memory_length : int
        Number of past steps to retain in the consumption history.
    adaptation_rate : float
        Fraction by which the regeneration rate is adjusted each step.
    """

    def __init__(
        self,
        config: ResourceGenerationConfig,
        memory_length: int = 100,
        adaptation_rate: float = 0.1,
    ) -> None:
        super().__init__(config)
        self.memory_length = memory_length
        self.adaptation_rate = adaptation_rate
        self.consumption_history: Dict[str, List[float]] = {}
        self.base_rates: Dict[str, float] = {}

    def _update_history(self, resource: "Resource") -> None:
        """Record consumption since last update."""
        rid = resource.resource_id
        if rid not in self.consumption_history:
            self.consumption_history[rid] = []
            self.base_rates[rid] = self.config.regen_rate

        history = self.consumption_history[rid]
        if hasattr(resource, "last_amount"):
            consumption = max(0.0, resource.last_amount - resource.amount)
            history.append(consumption)

        if len(history) > self.memory_length:
            history.pop(0)

        resource.last_amount = resource.amount

    def _calculate_adaptation(self, resource: "Resource") -> float:
        """Return an adaptation multiplier based on consumption history."""
        rid = resource.resource_id
        if rid not in self.consumption_history:
            return 1.0
        history = self.consumption_history[rid]
        if not history:
            return 1.0

        avg_consumption = sum(history) / len(history)
        max_expected = self.config.max_amount * 0.5

        if avg_consumption > max_expected:
            return 1 + self.adaptation_rate
        elif avg_consumption < max_expected * 0.2:
            return 1 - self.adaptation_rate
        return 1.0

    def should_regenerate(self, resource: "Resource") -> bool:
        if self.config.max_amount and resource.amount >= self.config.max_amount:
            return False
        self._update_history(resource)
        adaptation = self._calculate_adaptation(resource)
        rid = resource.resource_id
        base_rate = self.base_rates.get(rid, self.config.regen_rate)
        new_rate = max(0.01, min(0.99, base_rate * adaptation))
        self.base_rates[rid] = new_rate
        return random.random() < new_rate

    def get_regen_amount(self, resource: "Resource") -> int:
        adaptation = self._calculate_adaptation(resource)
        return max(1, int(self.config.regen_amount * adaptation))


# ---------------------------------------------------------------------------
# New ecosystem-level regenerators
# ---------------------------------------------------------------------------


class EnvironmentalRegenerator(ResourceRegenerator):
    """Regenerates resources based on abiotic environmental conditions.

    Four environmental factors are tracked: temperature, moisture, light, and
    soil/substrate quality. Each factor has a current level (0–1) and an
    optimal value. A Gaussian tolerance curve is used so that conditions far
    from the optimum yield low regeneration, while near-optimal conditions
    yield high regeneration.

    Parameters
    ----------
    temperature : float
        Current temperature level (0 = freezing, 1 = scorching).
    moisture : float
        Current moisture level (0 = arid, 1 = waterlogged).
    light : float
        Current light level (0 = darkness, 1 = full sunlight).
    soil_quality : float
        Soil/substrate quality (0 = barren, 1 = rich).
    temperature_weight : float
        Relative weight of temperature in the combined factor.
    moisture_weight : float
        Relative weight of moisture in the combined factor.
    light_weight : float
        Relative weight of light in the combined factor.
    soil_weight : float
        Relative weight of soil quality in the combined factor.
    optimal_temperature : float
        Optimal temperature value for maximum growth (0–1).
    optimal_moisture : float
        Optimal moisture value for maximum growth (0–1).
    optimal_light : float
        Optimal light value for maximum growth (0–1).
    optimal_soil : float
        Optimal soil quality value for maximum growth (0–1).
    tolerance_width : float
        Width (σ) of the Gaussian tolerance curve; smaller values mean
        narrower tolerance ranges.
    """

    def __init__(
        self,
        config: ResourceGenerationConfig,
        temperature: float = 0.5,
        moisture: float = 0.5,
        light: float = 0.5,
        soil_quality: float = 0.5,
        temperature_weight: float = 0.25,
        moisture_weight: float = 0.35,
        light_weight: float = 0.25,
        soil_weight: float = 0.15,
        optimal_temperature: float = 0.6,
        optimal_moisture: float = 0.7,
        optimal_light: float = 0.6,
        optimal_soil: float = 0.7,
        tolerance_width: float = 0.3,
    ) -> None:
        super().__init__(config)
        self.temperature = temperature
        self.moisture = moisture
        self.light = light
        self.soil_quality = soil_quality
        self.temperature_weight = temperature_weight
        self.moisture_weight = moisture_weight
        self.light_weight = light_weight
        self.soil_weight = soil_weight
        self.optimal_temperature = optimal_temperature
        self.optimal_moisture = optimal_moisture
        self.optimal_light = optimal_light
        self.optimal_soil = optimal_soil
        self.tolerance_width = tolerance_width

    # ------------------------------------------------------------------
    # Public interface for updating conditions
    # ------------------------------------------------------------------

    def update_conditions(
        self,
        temperature: Optional[float] = None,
        moisture: Optional[float] = None,
        light: Optional[float] = None,
        soil_quality: Optional[float] = None,
    ) -> None:
        """Update one or more environmental conditions.

        All values are clamped to ``[0, 1]``.

        Parameters
        ----------
        temperature : float, optional
            New temperature level.
        moisture : float, optional
            New moisture level.
        light : float, optional
            New light level.
        soil_quality : float, optional
            New soil quality level.
        """
        if temperature is not None:
            self.temperature = max(0.0, min(1.0, temperature))
        if moisture is not None:
            self.moisture = max(0.0, min(1.0, moisture))
        if light is not None:
            self.light = max(0.0, min(1.0, light))
        if soil_quality is not None:
            self.soil_quality = max(0.0, min(1.0, soil_quality))

    # ------------------------------------------------------------------
    # Internal calculations
    # ------------------------------------------------------------------

    def _gaussian_tolerance(self, value: float, optimal: float) -> float:
        """Gaussian bell-shaped tolerance score centred at *optimal*."""
        return math.exp(-((value - optimal) ** 2) / (2 * self.tolerance_width ** 2))

    def environmental_factor(self) -> float:
        """Return the weighted combined environmental suitability (0–1).

        Each factor is scored via a Gaussian tolerance curve and then
        combined using the configured weights.
        """
        total_weight = (
            self.temperature_weight
            + self.moisture_weight
            + self.light_weight
            + self.soil_weight
        )
        combined = (
            self.temperature_weight * self._gaussian_tolerance(self.temperature, self.optimal_temperature)
            + self.moisture_weight * self._gaussian_tolerance(self.moisture, self.optimal_moisture)
            + self.light_weight * self._gaussian_tolerance(self.light, self.optimal_light)
            + self.soil_weight * self._gaussian_tolerance(self.soil_quality, self.optimal_soil)
        )
        return combined / total_weight if total_weight > 0 else 1.0

    # ------------------------------------------------------------------
    # ResourceRegenerator interface
    # ------------------------------------------------------------------

    def should_regenerate(self, resource: "Resource") -> bool:
        if self.config.max_amount and resource.amount >= self.config.max_amount:
            return False
        effective_rate = self.config.regen_rate * self.environmental_factor()
        return random.random() < effective_rate

    def get_regen_amount(self, resource: "Resource") -> int:
        factor = self.environmental_factor()
        return max(1, int(self.config.regen_amount * factor))


class EcosystemRegenerator(ResourceRegenerator):
    """Full ecosystem model combining environmental conditions and resource relationships.

    This regenerator layers three types of ecological interactions:

    1. **Environmental factors** — temperature, moisture, light, and soil quality
       modulate a base rate just like :class:`EnvironmentalRegenerator`.
    2. **Mutualistic relationships** — nearby resources of specified types
       *boost* regeneration (e.g. nitrogen-fixing plants increasing soil nutrients).
    3. **Competitive relationships** — nearby resources of specified types
       *suppress* regeneration (e.g. invasive species crowding out natives).

    Parameters
    ----------
    temperature : float
        Current temperature level (0–1).
    moisture : float
        Current moisture level (0–1).
    light : float
        Current light level (0–1).
    soil_quality : float
        Current soil quality (0–1).
    optimal_temperature : float
        Optimal temperature for peak growth.
    optimal_moisture : float
        Optimal moisture for peak growth.
    optimal_light : float
        Optimal light for peak growth.
    optimal_soil : float
        Optimal soil quality for peak growth.
    tolerance_width : float
        Width (σ) of Gaussian tolerance curves.
    mutualistic_types : Dict[str, Dict[str, float]]
        Mapping of mutualistic resource types to parameters::

            {
                "nitrogen_fixer": {"weight": 0.5, "range": 40.0},
            }

    competitive_types : Dict[str, Dict[str, float]]
        Mapping of competitive resource types to parameters::

            {
                "invasive_plant": {"weight": 0.3, "range": 25.0},
            }

    carrying_capacity_factor : float
        Global multiplier that scales the effective carrying capacity (0–1);
        values below 1 reduce both the rate and amount for congested ecosystems.
    """

    def __init__(
        self,
        config: ResourceGenerationConfig,
        temperature: float = 0.5,
        moisture: float = 0.5,
        light: float = 0.5,
        soil_quality: float = 0.5,
        optimal_temperature: float = 0.6,
        optimal_moisture: float = 0.7,
        optimal_light: float = 0.6,
        optimal_soil: float = 0.7,
        tolerance_width: float = 0.3,
        mutualistic_types: Optional[Dict[str, Dict[str, float]]] = None,
        competitive_types: Optional[Dict[str, Dict[str, float]]] = None,
        carrying_capacity_factor: float = 1.0,
    ) -> None:
        super().__init__(config)
        # Environmental conditions
        self.temperature = temperature
        self.moisture = moisture
        self.light = light
        self.soil_quality = soil_quality
        self.optimal_temperature = optimal_temperature
        self.optimal_moisture = optimal_moisture
        self.optimal_light = optimal_light
        self.optimal_soil = optimal_soil
        self.tolerance_width = tolerance_width
        # Relationship networks
        self.mutualistic_types: Dict[str, Dict[str, float]] = mutualistic_types or {}
        self.competitive_types: Dict[str, Dict[str, float]] = competitive_types or {}
        # Carrying capacity
        self.carrying_capacity_factor = max(0.0, min(1.0, carrying_capacity_factor))

    # ------------------------------------------------------------------
    # Condition update
    # ------------------------------------------------------------------

    def update_conditions(
        self,
        temperature: Optional[float] = None,
        moisture: Optional[float] = None,
        light: Optional[float] = None,
        soil_quality: Optional[float] = None,
        carrying_capacity_factor: Optional[float] = None,
    ) -> None:
        """Update environmental conditions and/or carrying capacity factor."""
        if temperature is not None:
            self.temperature = max(0.0, min(1.0, temperature))
        if moisture is not None:
            self.moisture = max(0.0, min(1.0, moisture))
        if light is not None:
            self.light = max(0.0, min(1.0, light))
        if soil_quality is not None:
            self.soil_quality = max(0.0, min(1.0, soil_quality))
        if carrying_capacity_factor is not None:
            self.carrying_capacity_factor = max(0.0, min(1.0, carrying_capacity_factor))

    # ------------------------------------------------------------------
    # Internal calculations
    # ------------------------------------------------------------------

    def _gaussian_tolerance(self, value: float, optimal: float) -> float:
        return math.exp(-((value - optimal) ** 2) / (2 * self.tolerance_width ** 2))

    def _environmental_factor(self) -> float:
        """Average Gaussian tolerance across four environmental factors."""
        scores = [
            self._gaussian_tolerance(self.temperature, self.optimal_temperature),
            self._gaussian_tolerance(self.moisture, self.optimal_moisture),
            self._gaussian_tolerance(self.light, self.optimal_light),
            self._gaussian_tolerance(self.soil_quality, self.optimal_soil),
        ]
        return sum(scores) / len(scores)

    def _relationship_factor(self, resource: "Resource") -> float:
        """Compute the net relationship modifier from mutualism and competition.

        Returns a multiplier; values > 1 indicate net benefit, < 1 net suppression.
        """
        if not hasattr(resource, "environment") or resource.environment is None:
            return 1.0

        modifier = 0.0

        # Mutualistic relationships increase modifier
        for rtype, params in self.mutualistic_types.items():
            weight = params.get("weight", 1.0)
            range_limit = params.get("range", 50.0)
            nearby = [
                r
                for r in resource.environment.resources
                if (
                    r is not resource
                    and hasattr(r, "resource_type")
                    and r.resource_type == rtype
                    and math.sqrt(
                        (resource.position[0] - r.position[0]) ** 2
                        + (resource.position[1] - r.position[1]) ** 2
                    )
                    <= range_limit
                )
            ]
            if nearby:
                avg_amount = sum(r.amount for r in nearby) / len(nearby)
                modifier += weight * min(1.0, avg_amount / self.config.max_amount)

        # Competitive relationships decrease modifier
        for rtype, params in self.competitive_types.items():
            weight = params.get("weight", 1.0)
            range_limit = params.get("range", 25.0)
            nearby = [
                r
                for r in resource.environment.resources
                if (
                    r is not resource
                    and hasattr(r, "resource_type")
                    and r.resource_type == rtype
                    and math.sqrt(
                        (resource.position[0] - r.position[0]) ** 2
                        + (resource.position[1] - r.position[1]) ** 2
                    )
                    <= range_limit
                )
            ]
            if nearby:
                avg_amount = sum(r.amount for r in nearby) / len(nearby)
                modifier -= weight * min(1.0, avg_amount / self.config.max_amount)

        # Normalise and clamp to [0, 2]
        return max(0.0, min(2.0, 1.0 + modifier))

    def combined_factor(self, resource: "Resource") -> float:
        """Return the product of environmental, relationship, and carrying-capacity factors."""
        env_f = self._environmental_factor()
        rel_f = self._relationship_factor(resource)
        return env_f * rel_f * self.carrying_capacity_factor

    # ------------------------------------------------------------------
    # ResourceRegenerator interface
    # ------------------------------------------------------------------

    def should_regenerate(self, resource: "Resource") -> bool:
        if self.config.max_amount and resource.amount >= self.config.max_amount:
            return False
        factor = self.combined_factor(resource)
        return random.random() < self.config.regen_rate * factor

    def get_regen_amount(self, resource: "Resource") -> int:
        factor = self.combined_factor(resource)
        return max(1, int(self.config.regen_amount * factor))


class EvolutionaryRegenerator(ResourceRegenerator):
    """Regeneration parameters evolve over time in response to environmental pressure.

    This regenerator simulates resource adaptation through a simple
    per-resource evolutionary process:

    * **Stress tracking** — when a resource's amount falls below a stress
      threshold, stress is accumulated.  High stress triggers upward adaptation
      of the regeneration rate; low stress allows gradual relaxation back to
      the base rate.
    * **Mutation** — every ``mutation_interval`` steps the regeneration rate
      undergoes a small Gaussian perturbation, modelling random genetic drift.
    * **Carrying capacity** — the rate is adjusted downward when the resource
      approaches the global carrying capacity (``max_amount``), preventing
      runaway growth.
    * **Bounds** — the evolved rate is always clamped to ``[min_rate, max_rate]``.

    Parameters
    ----------
    stress_threshold : float
        Resource amount fraction (relative to ``max_amount``) below which
        the resource is considered under stress (default 0.3 = 30 % of max).
    stress_adaptation_rate : float
        How strongly the regeneration rate increases under stress per step.
    relaxation_rate : float
        How quickly the rate decays back towards the base rate when not stressed.
    mutation_rate : float
        Standard deviation of the Gaussian mutation applied at each mutation event.
    mutation_interval : int
        Steps between mutation events for each resource.
    min_rate : float
        Minimum allowable evolved regeneration rate.
    max_rate : float
        Maximum allowable evolved regeneration rate.
    """

    def __init__(
        self,
        config: ResourceGenerationConfig,
        stress_threshold: float = 0.3,
        stress_adaptation_rate: float = 0.05,
        relaxation_rate: float = 0.02,
        mutation_rate: float = 0.01,
        mutation_interval: int = 50,
        min_rate: float = 0.01,
        max_rate: float = 0.99,
    ) -> None:
        super().__init__(config)
        self.stress_threshold = stress_threshold
        self.stress_adaptation_rate = stress_adaptation_rate
        self.relaxation_rate = relaxation_rate
        self.mutation_rate = mutation_rate
        self.mutation_interval = mutation_interval
        self.min_rate = min_rate
        self.max_rate = max_rate

        # Per-resource state
        self._evolved_rates: Dict[str, float] = {}
        self._step_counters: Dict[str, int] = {}
        self._stress_accumulators: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_resource(self, resource: "Resource") -> None:
        """Initialise per-resource evolutionary state on first encounter."""
        rid = resource.resource_id
        if rid not in self._evolved_rates:
            self._evolved_rates[rid] = self.config.regen_rate
            self._step_counters[rid] = 0
            self._stress_accumulators[rid] = 0.0

    def _evolve(self, resource: "Resource") -> float:
        """Update and return the evolved regeneration rate for *resource*."""
        self._init_resource(resource)
        rid = resource.resource_id
        rate = self._evolved_rates[rid]
        self._step_counters[rid] += 1
        step = self._step_counters[rid]

        # Stress response: adapt upward when resource is depleted
        stress_level = self.stress_threshold * self.config.max_amount
        if resource.amount < stress_level:
            self._stress_accumulators[rid] = min(
                1.0, self._stress_accumulators[rid] + 0.1
            )
            rate += self.stress_adaptation_rate * self._stress_accumulators[rid]
        else:
            # Gradual relaxation toward base rate
            self._stress_accumulators[rid] = max(0.0, self._stress_accumulators[rid] - 0.05)
            rate += self.relaxation_rate * (self.config.regen_rate - rate)

        # Carrying capacity suppression
        if self.config.max_amount > 0:
            capacity_fraction = resource.amount / self.config.max_amount
            if capacity_fraction > 0.8:
                rate *= 1.0 - (capacity_fraction - 0.8) * 2.5

        # Periodic mutation (Gaussian perturbation)
        if step % self.mutation_interval == 0:
            rate += random.gauss(0.0, self.mutation_rate)

        rate = max(self.min_rate, min(self.max_rate, rate))
        self._evolved_rates[rid] = rate
        return rate

    def evolved_rate(self, resource: "Resource") -> float:
        """Return the current evolved rate for *resource* without advancing state."""
        self._init_resource(resource)
        return self._evolved_rates[resource.resource_id]

    # ------------------------------------------------------------------
    # ResourceRegenerator interface
    # ------------------------------------------------------------------

    def should_regenerate(self, resource: "Resource") -> bool:
        if self.config.max_amount and resource.amount >= self.config.max_amount:
            return False
        rate = self._evolve(resource)
        return random.random() < rate

    def get_regen_amount(self, resource: "Resource") -> int:
        rate = self.evolved_rate(resource)
        scale = rate / max(self.config.regen_rate, 1e-9)
        return max(1, int(self.config.regen_amount * scale))


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

_REGENERATOR_REGISTRY: Dict[str, type] = {
    "basic": BasicRegenerator,
    "time_based": TimeBasedRegenerator,
    "seasonal": SeasonalRegenerator,
    "proximity": ProximityRegenerator,
    "dependent": ResourceDependentRegenerator,
    "adaptive": AdaptiveRegenerator,
    "environmental": EnvironmentalRegenerator,
    "ecosystem": EcosystemRegenerator,
    "evolutionary": EvolutionaryRegenerator,
}


def create_regenerator(
    regenerator_type: str,
    config: Optional[ResourceGenerationConfig] = None,
    **kwargs,
) -> ResourceRegenerator:
    """Instantiate a regenerator by name.

    Parameters
    ----------
    regenerator_type : str
        One of: ``"basic"``, ``"time_based"``, ``"seasonal"``, ``"proximity"``,
        ``"dependent"``, ``"adaptive"``, ``"environmental"``, ``"ecosystem"``,
        ``"evolutionary"``.
    config : ResourceGenerationConfig, optional
        Shared regeneration configuration.  A default instance is used when not
        provided.
    **kwargs
        Additional keyword arguments forwarded to the regenerator constructor.

    Returns
    -------
    ResourceRegenerator
        Instantiated regenerator.

    Raises
    ------
    ValueError
        If *regenerator_type* is not recognised.
    """
    if config is None:
        config = ResourceGenerationConfig()
    cls = _REGENERATOR_REGISTRY.get(regenerator_type)
    if cls is None:
        valid = ", ".join(sorted(_REGENERATOR_REGISTRY))
        raise ValueError(
            f"Unknown regenerator type '{regenerator_type}'. Valid types: {valid}"
        )
    return cls(config, **kwargs)
