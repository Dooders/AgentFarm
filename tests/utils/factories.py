from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


def make_agent(environment, position: Tuple[float, float] = (0.0, 0.0), resource_level: float = 1.0):
    """Create and register an agent in the provided environment using AgentFactory."""
    from farm.core.agent import AgentFactory

    # Services will be injected by environment when agent is added
    factory = AgentFactory(
        spatial_service=environment.spatial_service,
    )
    
    agent = factory.create_default_agent(
        agent_id=environment.get_next_agent_id(),
        position=position,
        initial_resources=int(resource_level),
    )
    environment.add_agent(agent)
    return agent


@dataclass
class SimulationDefaults:
    width: int = 20
    height: int = 20
    initial_resources: int = 3
    seed: int = 1234

