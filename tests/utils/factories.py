from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


def make_agent(environment, position: Tuple[float, float] = (0.0, 0.0), resource_level: float = 1.0):
    """Create and register a BaseAgent in the provided environment."""
    from farm.core.agent import AgentCore

    agent = AgentCore(
        agent_id=environment.get_next_agent_id(),
        position=position,
        resource_level=resource_level,
        environment=environment,
        spatial_service=environment.spatial_service,
    )
    environment.add_agent(agent)
    return agent


@dataclass
class SimulationDefaults:
    width: int = 20
    height: int = 20
    initial_resources: int = 3
    seed: int = 1234

