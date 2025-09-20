"""
MetricsTracker: Centralized tracking system for simulation metrics.

This module provides a dedicated class for tracking various simulation metrics
including births, deaths, combat encounters, resource sharing, and other
key performance indicators. It separates tracking concerns from the main
Environment class and provides a clean interface for metric collection.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from farm.core.agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single simulation step."""

    births: int = 0
    deaths: int = 0
    combat_encounters: int = 0
    successful_attacks: int = 0
    combat_encounters_this_step: int = 0
    successful_attacks_this_step: int = 0
    resources_shared: float = 0.0
    resources_shared_this_step: float = 0.0
    reproduction_attempts: int = 0
    reproduction_successes: int = 0
    resource_consumption: float = 0.0

    def reset(self) -> None:
        """Reset all step-specific metrics to zero."""
        self.births = 0
        self.deaths = 0
        self.combat_encounters = 0
        self.successful_attacks = 0
        self.combat_encounters_this_step = 0
        self.successful_attacks_this_step = 0
        self.resources_shared = 0.0
        self.resources_shared_this_step = 0.0
        self.reproduction_attempts = 0
        self.reproduction_successes = 0
        self.resource_consumption = 0.0


@dataclass
class CumulativeMetrics:
    """Cumulative metrics across the entire simulation."""

    total_births: int = 0
    total_deaths: int = 0
    total_combat_encounters: int = 0
    total_successful_attacks: int = 0
    total_resources_shared: float = 0.0
    total_reproduction_attempts: int = 0
    total_reproduction_successes: int = 0
    total_resource_consumption: float = 0.0

    def update_from_step(self, step_metrics: StepMetrics) -> None:
        """Update cumulative metrics from step metrics."""
        self.total_births += step_metrics.births
        self.total_deaths += step_metrics.deaths
        self.total_combat_encounters += step_metrics.combat_encounters
        self.total_successful_attacks += step_metrics.successful_attacks
        self.total_resources_shared += step_metrics.resources_shared
        self.total_reproduction_attempts += step_metrics.reproduction_attempts
        self.total_reproduction_successes += step_metrics.reproduction_successes
        self.total_resource_consumption += step_metrics.resource_consumption


class MetricsTracker:
    """
    Centralized tracking system for simulation metrics.

    This class provides a clean interface for tracking various simulation
    metrics including births, deaths, combat encounters, resource sharing,
    and other key performance indicators. It maintains both step-specific
    and cumulative metrics.

    Attributes:
        step_metrics (StepMetrics): Current step metrics
        cumulative_metrics (CumulativeMetrics): Cumulative metrics across simulation
        custom_metrics (Dict[str, Any]): Additional custom metrics
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.step_metrics = StepMetrics()
        self.cumulative_metrics = CumulativeMetrics()
        self.custom_metrics: Dict[str, Any] = {}
        self.previous_total_resources = 0

    def record_birth(self) -> None:
        """Record a birth event."""
        self.step_metrics.births += 1

    def record_death(self) -> None:
        """Record a death event."""
        self.step_metrics.deaths += 1

    def record_combat_encounter(self) -> None:
        """Record a combat encounter."""
        self.step_metrics.combat_encounters += 1
        self.step_metrics.combat_encounters_this_step += 1

    def record_successful_attack(self) -> None:
        """Record a successful attack."""
        self.step_metrics.successful_attacks += 1
        self.step_metrics.successful_attacks_this_step += 1

    def record_resources_shared(self, amount: float) -> None:
        """Record resources shared between agents."""
        self.step_metrics.resources_shared += amount
        self.step_metrics.resources_shared_this_step += amount

    def record_reproduction_attempt(self) -> None:
        """Record a reproduction attempt."""
        self.step_metrics.reproduction_attempts += 1

    def record_reproduction_success(self) -> None:
        """Record a successful reproduction."""
        self.step_metrics.reproduction_successes += 1

    def record_resource_consumption(self, amount: float) -> None:
        """Record resource consumption."""
        self.step_metrics.resource_consumption += amount

    def add_custom_metric(self, name: str, value: Any) -> None:
        """Add a custom metric."""
        self.custom_metrics[name] = value

    def get_step_metrics(self) -> Dict[str, Any]:
        """Get current step metrics as a dictionary."""
        return {
            "births": self.step_metrics.births,
            "deaths": self.step_metrics.deaths,
            "combat_encounters": self.step_metrics.combat_encounters,
            "successful_attacks": self.step_metrics.successful_attacks,
            "combat_encounters_this_step": self.step_metrics.combat_encounters_this_step,
            "successful_attacks_this_step": self.step_metrics.successful_attacks_this_step,
            "resources_shared": self.step_metrics.resources_shared,
            "resources_shared_this_step": self.step_metrics.resources_shared_this_step,
            "reproduction_attempts": self.step_metrics.reproduction_attempts,
            "reproduction_successes": self.step_metrics.reproduction_successes,
            "resource_consumption": self.step_metrics.resource_consumption,
            **self.custom_metrics,
        }

    def get_cumulative_metrics(self) -> Dict[str, Any]:
        """Get cumulative metrics as a dictionary."""
        return {
            "total_births": self.cumulative_metrics.total_births,
            "total_deaths": self.cumulative_metrics.total_deaths,
            "total_combat_encounters": self.cumulative_metrics.total_combat_encounters,
            "total_successful_attacks": self.cumulative_metrics.total_successful_attacks,
            "total_resources_shared": self.cumulative_metrics.total_resources_shared,
            "total_reproduction_attempts": self.cumulative_metrics.total_reproduction_attempts,
            "total_reproduction_successes": self.cumulative_metrics.total_reproduction_successes,
            "total_resource_consumption": self.cumulative_metrics.total_resource_consumption,
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get both step and cumulative metrics combined."""
        return {**self.get_step_metrics(), **self.get_cumulative_metrics()}

    def end_step(self) -> Dict[str, Any]:
        """
        End the current step and return metrics.

        Returns:
            Dict containing both step and cumulative metrics
        """
        # Update cumulative metrics
        self.cumulative_metrics.update_from_step(self.step_metrics)

        # Get combined metrics
        metrics = self.get_all_metrics()

        # Reset step metrics for next step
        self.step_metrics.reset()
        self.custom_metrics.clear()

        return metrics

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.step_metrics.reset()
        self.cumulative_metrics = CumulativeMetrics()
        self.custom_metrics.clear()
        self.previous_total_resources = 0

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics."""
        return {
            "population_growth": self.cumulative_metrics.total_births
            - self.cumulative_metrics.total_deaths,
            "combat_success_rate": (
                self.cumulative_metrics.total_successful_attacks
                / max(self.cumulative_metrics.total_combat_encounters, 1)
            ),
            "reproduction_success_rate": (
                self.cumulative_metrics.total_reproduction_successes
                / max(self.cumulative_metrics.total_reproduction_attempts, 1)
            ),
            "total_resources_shared": self.cumulative_metrics.total_resources_shared,
            "total_resource_consumption": self.cumulative_metrics.total_resource_consumption,
        }

    def update_metrics(
        self, metrics: Dict, db=None, time=None, agent_objects=None, resources=None
    ):
        """Update environment metrics and log to database.

        Parameters
        ----------
        metrics : Dict
            Dictionary containing metrics to update and log
        db : Optional database object for logging
        time : Optional current simulation time/step
        agent_objects : Optional dict of agent objects for state preparation
        resources : Optional list of resources for state preparation
        """
        try:
            # Log metrics to database
            if db and time is not None:
                # Prepare agent states if agent_objects provided
                agent_states = []
                if agent_objects:
                    agent_states = [
                        self._prepare_agent_state(agent, time)
                        for agent in agent_objects.values()
                        if agent.alive
                    ]

                # Prepare resource states if resources provided
                resource_states = []
                if resources:
                    resource_states = [
                        self._prepare_resource_state(resource) for resource in resources
                    ]

                db.logger.log_step(
                    step_number=time,
                    agent_states=agent_states,
                    resource_states=resource_states,
                    metrics=metrics,
                )
        except (ValueError, TypeError, AttributeError, IndexError) as e:
            logging.error("Error updating metrics: %s", e)

    def _prepare_agent_state(self, agent, time):
        """Prepare agent state for database logging."""
        return (
            agent.agent_id,
            agent.position[0],  # x coordinate
            agent.position[1],  # y coordinate
            agent.resource_level,
            agent.current_health,
            agent.starting_health,
            agent.starvation_counter,
            int(agent.is_defending),
            agent.total_reward,
            time - agent.birth_time,  # age
        )

    def _prepare_resource_state(self, resource):
        """Prepare resource state for database logging."""
        return (
            resource.resource_id,
            resource.amount,
            resource.position[0],  # x coordinate
            resource.position[1],  # y coordinate
        )

    def calculate_metrics(self, agent_objects, resources, time, config=None):
        """
        Calculate metrics for the current simulation state.

        Aggregates population, resources, diversity, and event counters from the
        tracker into a single dict. Also computes per-step resource consumption
        based on the change in total resource amount since the last call.

        Returns a dict with keys like:
        - total_agents, system_agents
        - total_resources, resources_consumed
        - average_agent_resources, average_agent_health, average_agent_age, average_reward
        - resource_efficiency, resource_distribution_entropy
        - births, deaths
        - genetic_diversity, dominant_genome_ratio
        - combat_encounters, successful_attacks
        - resources_shared (step aggregate), resources_shared_this_step
        - combat_encounters_this_step, successful_attacks_this_step
        """
        # Step-level aggregates are sourced from the internal tracker
        try:
            # Get alive agents
            alive_agents = [agent for agent in agent_objects.values() if agent.alive]
            total_agents = len(alive_agents)

            # Calculate agent type counts
            system_agents = len([a for a in alive_agents if isinstance(a, BaseAgent)])

            # Get metrics from tracker
            tracker_metrics = self.get_step_metrics()
            births = tracker_metrics["births"]
            deaths = tracker_metrics["deaths"]

            # Calculate generation metrics
            current_max_generation = (
                max([a.generation for a in alive_agents]) if alive_agents else 0
            )

            # Calculate health and age metrics
            average_health = (
                sum(a.current_health for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            )
            average_age = (
                sum(time - a.birth_time for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            )
            average_reward = (
                sum(a.total_reward for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            )

            # Calculate resource metrics
            total_resources = sum(r.amount for r in resources)
            average_agent_resources = (
                sum(a.resource_level for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            )
            resource_efficiency = (
                total_resources
                / (len(resources) * (config.max_resource_amount if config else 30))
                if resources
                else 0
            )

            # Calculate genetic diversity
            genome_counts = {}
            for agent in alive_agents:
                genome_counts[agent.genome_id] = (
                    genome_counts.get(agent.genome_id, 0) + 1
                )
            genetic_diversity = (
                len(genome_counts) / total_agents if total_agents > 0 else 0
            )
            dominant_genome_ratio = (
                max(genome_counts.values()) / total_agents if genome_counts else 0
            )

            # Get combat and sharing metrics from tracker
            combat_encounters = tracker_metrics["combat_encounters"]
            successful_attacks = tracker_metrics["successful_attacks"]
            resources_shared = tracker_metrics["resources_shared"]
            resources_shared_this_step = tracker_metrics["resources_shared_this_step"]
            combat_encounters_this_step = tracker_metrics["combat_encounters_this_step"]
            successful_attacks_this_step = tracker_metrics["successful_attacks_this_step"]

            # Calculate resource distribution entropy
            resource_amounts = [r.amount for r in resources]
            if resource_amounts:
                total = sum(resource_amounts)
                if total > 0:
                    probabilities = [amt / total for amt in resource_amounts]
                    resource_distribution_entropy = -sum(
                        p * np.log(p) if p > 0 else 0 for p in probabilities
                    )
                else:
                    resource_distribution_entropy = 0.0
            else:
                resource_distribution_entropy = 0.0

            # Calculate resource consumption for this step
            previous_resources = getattr(self, "previous_total_resources", 0)
            current_resources = sum(r.amount for r in resources)
            resources_consumed = max(0, previous_resources - current_resources)
            self.previous_total_resources = current_resources

            # Add births and deaths to metrics
            metrics = {
                "total_agents": total_agents,
                "system_agents": system_agents,
                "total_resources": total_resources,
                "average_agent_resources": average_agent_resources,
                "resources_consumed": resources_consumed,
                "births": births,
                "deaths": deaths,
                "current_max_generation": current_max_generation,
                "resource_efficiency": resource_efficiency,
                "resource_distribution_entropy": resource_distribution_entropy,
                "average_agent_health": average_health,
                "average_agent_age": average_age,
                "average_reward": average_reward,
                "combat_encounters": combat_encounters,
                "successful_attacks": successful_attacks,
                "resources_shared": resources_shared,
                "resources_shared_this_step": resources_shared_this_step,
                "genetic_diversity": genetic_diversity,
                "dominant_genome_ratio": dominant_genome_ratio,
                "combat_encounters_this_step": combat_encounters_this_step,
                "successful_attacks_this_step": successful_attacks_this_step,
            }

            # End step in tracker (resets step metrics and updates cumulative)
            self.end_step()

            return metrics
        except (ValueError, TypeError, AttributeError, IndexError) as e:
            logging.error("Error calculating metrics: %s", e)
            return {}  # Return empty metrics on error
