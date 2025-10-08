from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore
    from farm.core.environment import Environment


class FeatureEngineer:
    """Feature engineering utilities for ML action algorithms.

    Produces a compact, normalized feature vector from the agent and environment
    that works well with traditional ML algorithms.
    """

    def extract_features(self, agent: "AgentCore", environment: "Environment") -> np.ndarray:
        features: List[float] = []

        # Agent state features (normalized where possible)
        # Get max resources from reproduction config if available, otherwise use default
        max_resources = 24.0  # Default fallback
        if hasattr(agent, "config") and agent.config:
            # Try to get from reproduction config
            if hasattr(agent.config, "reproduction") and hasattr(agent.config.reproduction, "reproduction_threshold"):
                max_resources = max(1.0, float(agent.config.reproduction.reproduction_threshold) * 3.0)
            elif hasattr(agent.config, "min_reproduction_resources"):
                max_resources = max(1.0, float(agent.config.min_reproduction_resources) * 3.0)

        features.extend(
            [
                float(agent.current_health) / max(1.0, float(agent.starting_health)),
                float(agent.resource_level) / max_resources,
            ]
        )

        # Position features normalized by environment size
        width = max(1.0, float(getattr(environment, "width", 100)))
        height = max(1.0, float(getattr(environment, "height", 100)))
        features.extend(
            [
                float(agent.position[0]) / width,
                float(agent.position[1]) / height,
            ]
        )

        # Environmental features
        features.extend(self._extract_environmental_features(agent, environment))

        # Social features
        features.extend(self._extract_social_features(agent, environment))

        # Time feature (bounded, to avoid unbounded growth)
        time_norm = float(getattr(environment, "time", 0) % 1000) / 1000.0
        features.append(time_norm)

        return np.asarray(features, dtype=float)

    def _extract_environmental_features(self, agent: "AgentCore", environment: "Environment") -> List[float]:
        # Nearby resource density normalized
        gathering_range = 30  # Default fallback
        if hasattr(agent, "config") and agent.config:
            # Try to get from movement config or legacy config
            if hasattr(agent.config, "movement") and hasattr(agent.config.movement, "gathering_range"):
                gathering_range = agent.config.movement.gathering_range
            elif hasattr(agent.config, "gathering_range"):
                gathering_range = agent.config.gathering_range

        # Defensive check for get_nearby_resources method
        nearby_resources = []
        if environment and hasattr(environment, "get_nearby_resources"):
            try:
                nearby_resources = environment.get_nearby_resources(agent.position, gathering_range)
            except (AttributeError, TypeError, ValueError):
                # Fallback to empty list if method fails
                nearby_resources = []

        total_resources = max(1, len(getattr(environment, "resources", [])))
        resource_density = float(len(nearby_resources)) / float(total_resources)

        # Starvation/consumption context
        starvation_ratio = float(agent.starvation_counter) / max(1.0, float(agent.starvation_threshold))

        return [resource_density, starvation_ratio]

    def _extract_social_features(self, agent: "AgentCore", environment: "Environment") -> List[float]:
        social_range = 30  # Default fallback
        if hasattr(agent, "config") and agent.config:
            # Try to get from movement config or legacy config
            if hasattr(agent.config, "movement") and hasattr(agent.config.movement, "social_range"):
                social_range = agent.config.movement.social_range
            elif hasattr(agent.config, "social_range"):
                social_range = agent.config.social_range

        # Defensive check for get_nearby_agents method
        nearby_agents = []
        if environment and hasattr(environment, "get_nearby_agents"):
            try:
                nearby_agents = environment.get_nearby_agents(agent.position, social_range)
            except (AttributeError, TypeError, ValueError):
                # Fallback to empty list if method fails
                nearby_agents = []

        total_agents = max(1, len(getattr(environment, "agents", [])))
        agent_density = float(len(nearby_agents)) / float(total_agents)

        # Defensive status flag
        is_defending = 1.0 if getattr(agent, "is_defending", False) else 0.0

        return [agent_density, is_defending]
