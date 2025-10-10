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
        # Get max resources from reproduction component if available, otherwise use default
        max_resources = 24.0  # Default fallback # TODO: Get from config
        reproduction_component = agent.get_component("reproduction")
        if reproduction_component and hasattr(reproduction_component, "_config"):
            reproduction_config = reproduction_component._config
            if hasattr(reproduction_config, "reproduction_threshold"):
                max_resources = max(1.0, float(reproduction_config.reproduction_threshold) * 3.0)

        # Get components safely, handling missing components
        combat_component = agent.get_component("combat")
        resource_component = agent.get_component("resource")
        
        features.extend(
            [
                float(combat_component.health) / max(1.0, float(combat_component.max_health)) if combat_component else 0.0,
                float(resource_component.level) / max_resources if resource_component else 0.0,
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
        movement_component = agent.get_component("movement")
        if movement_component and hasattr(movement_component, "_config"):
            movement_config = movement_component._config
            # Use max_movement as gathering range (common pattern)
            if hasattr(movement_config, "max_movement"):
                gathering_range = movement_config.max_movement

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
        resource_component = agent.get_component("resource")
        if resource_component:
            starvation_counter = getattr(resource_component, "starvation_steps", 0)
            starvation_threshold = 10
            rc_config = getattr(resource_component, "_config", None)
            if rc_config is not None and hasattr(rc_config, "starvation_threshold"):
                starvation_threshold = getattr(rc_config, "starvation_threshold")
            starvation_ratio = float(starvation_counter) / max(1.0, float(starvation_threshold))
        else:
            starvation_ratio = 0.0

        return [resource_density, starvation_ratio]

    def _extract_social_features(self, agent: "AgentCore", environment: "Environment") -> List[float]:
        social_range = 30  # Default fallback
        movement_component = agent.get_component("movement")
        if movement_component and hasattr(movement_component, "_config"):
            movement_config = movement_component._config
            # Use max_movement as social range (common pattern)
            if hasattr(movement_config, "max_movement"):
                social_range = movement_config.max_movement

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
        combat_component = agent.get_component("combat")
        is_defending = 1.0 if (combat_component and combat_component.is_defending) else 0.0

        return [agent_density, is_defending]
