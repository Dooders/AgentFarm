from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


def make_agent(environment, position: Tuple[float, float] = (0.0, 0.0), resource_level: float = 1.0):
    """Create and register an AgentCore in the provided environment."""
    from farm.core.agent import AgentFactory, AgentServices
    from farm.core.agent.config.component_configs import AgentComponentConfig

    # Create services from environment
    services = AgentServices(
        spatial_service=environment.spatial_service,
        time_service=getattr(environment, "time_service", None),
        metrics_service=getattr(environment, "metrics_service", None),
        logging_service=getattr(environment, "logging_service", None),
        validation_service=getattr(environment, "validation_service", None),
        lifecycle_service=getattr(environment, "lifecycle_service", None),
    )

    # Create factory and agent
    factory = AgentFactory(services)
    agent_config = AgentComponentConfig.from_simulation_config(environment.config) if environment.config else None

    agent = factory.create_default_agent(
        agent_id=environment.get_next_agent_id(),
        position=position,
        initial_resources=resource_level,
        config=agent_config,
        environment=environment,
    )
    environment.add_agent(agent)
    return agent


@dataclass
class SimulationDefaults:
    width: int = 20
    height: int = 20
    initial_resources: int = 3
    seed: int = 1234
