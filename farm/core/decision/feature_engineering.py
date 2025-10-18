from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from farm.core.agent import AgentCore
    from farm.core.environment import Environment


class FeatureEngineer:
    """Feature engineering utilities for ML action algorithms.

    Produces a compact, normalized feature vector from the agent and environment
    that works well with traditional ML algorithms.
    """

    def extract_features(
        self, agent: "AgentCore", environment: "Environment"
    ) -> np.ndarray:
        features: List[float] = []

        # Agent state features (normalized where possible)
        # Get resource component for max resources calculation
        resource_comp = agent.get_component("resource")
        reproduction_comp = agent.get_component("reproduction")
        
        # Calculate max resources based on reproduction cost
        max_resources = 1.0
        if reproduction_comp and reproduction_comp.config:
            max_resources = max(1.0, float(reproduction_comp.config.offspring_cost) * 3.0)
        elif resource_comp and resource_comp.config:
            max_resources = max(1.0, float(resource_comp.config.offspring_cost) * 3.0)
        else:
            max_resources = 24.0  # Default fallback
        
        # Get combat component for health
        combat_comp = agent.get_component("combat")
        current_health = combat_comp.health if combat_comp else 0.0
        starting_health = combat_comp.config.starting_health if combat_comp and combat_comp.config else 100.0
        
        features.extend(
            [
                float(current_health) / max(1.0, float(starting_health)),
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

    def _extract_environmental_features(
        self, agent: "AgentCore", environment: "Environment"
    ) -> List[float]:
        # Nearby resource density normalized
        # Get gathering range from perception component
        perception_comp = agent.get_component("perception")
        gathering_range = 30  # Default fallback
        if perception_comp and perception_comp.config:
            gathering_range = perception_comp.config.perception_radius

        # Defensive check for get_nearby_resources method
        nearby_resources = []
        if environment and hasattr(environment, "get_nearby_resources"):
            try:
                nearby_resources = environment.get_nearby_resources(
                    agent.position, gathering_range
                )
            except (AttributeError, TypeError, ValueError):
                # Fallback to empty list if method fails
                nearby_resources = []

        total_resources = max(1, len(getattr(environment, "resources", [])))
        resource_density = float(len(nearby_resources)) / float(total_resources)

        # Starvation/consumption context - access through resource component
        resource_comp = agent.get_component("resource")
        starvation_counter = resource_comp.starvation_counter if resource_comp else 0
        starvation_threshold = (
            resource_comp.config.starvation_threshold 
            if resource_comp and resource_comp.config 
            else 100  # Default fallback
        )
        starvation_ratio = float(starvation_counter) / max(1.0, float(starvation_threshold))

        return [resource_density, starvation_ratio]

    def _extract_social_features(
        self, agent: "AgentCore", environment: "Environment"
    ) -> List[float]:
        # Get social range from perception component
        perception_comp = agent.get_component("perception")
        social_range = 30  # Default fallback
        if perception_comp and perception_comp.config:
            social_range = perception_comp.config.perception_radius

        # Defensive check for get_nearby_agents method
        nearby_agents = []
        if environment and hasattr(environment, "get_nearby_agents"):
            try:
                nearby_agents = environment.get_nearby_agents(
                    agent.position, social_range
                )
            except (AttributeError, TypeError, ValueError):
                # Fallback to empty list if method fails
                nearby_agents = []

        total_agents = max(1, len(getattr(environment, "agents", [])))
        agent_density = float(len(nearby_agents)) / float(total_agents)

        # Defensive status flag - access through combat component
        combat_comp = agent.get_component("combat")
        is_defending = 1.0 if (combat_comp and combat_comp.is_defending) else 0.0

        return [agent_density, is_defending]
