"""
MetricsTracker: Centralized tracking system for simulation metrics.

This module provides a dedicated class for tracking various simulation metrics
including births, deaths, combat encounters, resource sharing, and other
key performance indicators. It separates tracking concerns from the main
Environment class and provides a clean interface for metric collection.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single simulation step."""

    # Population metrics
    births: int = 0
    deaths: int = 0

    # Combat metrics
    combat_encounters: int = 0
    successful_attacks: int = 0

    # Resource sharing metrics
    resources_shared: float = 0.0

    # Additional metrics can be added here
    reproduction_attempts: int = 0
    reproduction_successes: int = 0
    resource_consumption: float = 0.0

    def reset(self) -> None:
        """Reset all step-specific metrics to zero."""
        self.births = 0
        self.deaths = 0
        self.combat_encounters = 0
        self.successful_attacks = 0
        self.resources_shared = 0.0
        self.reproduction_attempts = 0
        self.reproduction_successes = 0
        self.resource_consumption = 0.0


@dataclass
class CumulativeMetrics:
    """Cumulative metrics across the entire simulation."""

    # Population metrics
    total_births: int = 0
    total_deaths: int = 0

    # Combat metrics
    total_combat_encounters: int = 0
    total_successful_attacks: int = 0

    # Resource sharing metrics
    total_resources_shared: float = 0.0

    # Additional cumulative metrics
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

    def record_birth(self) -> None:
        """Record a birth event."""
        self.step_metrics.births += 1

    def record_death(self) -> None:
        """Record a death event."""
        self.step_metrics.deaths += 1

    def record_combat_encounter(self) -> None:
        """Record a combat encounter."""
        self.step_metrics.combat_encounters += 1

    def record_successful_attack(self) -> None:
        """Record a successful attack."""
        self.step_metrics.successful_attacks += 1

    def record_resources_shared(self, amount: float) -> None:
        """Record resources shared between agents."""
        self.step_metrics.resources_shared += amount

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
            "resources_shared": self.step_metrics.resources_shared,
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
