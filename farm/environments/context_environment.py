import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from farm.core.config import SimulationConfig
from farm.core.environment import Environment as CoreEnvironment
from farm.core.state import EnvironmentState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Context:
    """Read-only context passed to agents each step.

    Provides safe accessors over the environment state without letting agents
    mutate global state directly.
    """

    time: int
    width: int
    height: int
    agent_positions: List[Tuple[float, float]]
    resource_positions: List[Tuple[float, float]]
    resource_amounts: List[int]

    def get_time(self) -> int:
        return self.time

    def get_dimensions(self) -> Tuple[int, int]:
        return self.width, self.height

    def get_agent_positions(self) -> List[Tuple[float, float]]:
        return self.agent_positions

    def get_resource_positions(self) -> List[Tuple[float, float]]:
        return self.resource_positions

    def get_resource_amounts(self) -> List[int]:
        return self.resource_amounts


class EnvironmentV2(CoreEnvironment):
    """Context-based environment that wraps the core Environment.

    This class preserves the existing Environment API for compatibility while
    adding a `build_context()` method and a `step_with_context()` loop where
    agents are expected to consume a Context object during decision making.
    """

    def __init__(
        self,
        width: int,
        height: int,
        resource_distribution: Dict,
        db_path: Optional[str] = "simulation.db",
        max_resource: Optional[int] = None,
        config: Optional[SimulationConfig] = None,
        simulation_id: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            width=width,
            height=height,
            resource_distribution=resource_distribution,
            db_path=db_path,
            max_resource=max_resource,
            config=config,
            simulation_id=simulation_id,
            seed=seed,
        )

    def build_context(self) -> Context:
        """Build a read-only snapshot of the environment state for agents."""
        agent_positions: List[Tuple[float, float]] = [
            tuple(a.position) for a in self.agents if a.alive
        ]
        resource_positions: List[Tuple[float, float]] = [
            tuple(r.position) for r in self.resources
        ]
        resource_amounts: List[int] = [int(r.amount) for r in self.resources]

        return Context(
            time=self.time,
            width=self.width,
            height=self.height,
            agent_positions=agent_positions,
            resource_positions=resource_positions,
            resource_amounts=resource_amounts,
        )

    def step_with_context(self) -> None:
        """Advance one tick having each alive agent act with a Context.

        Agents may implement an optional `act_with_context(context: Context)`;
        if not present, falls back to existing `act()` behavior.
        """
        context = self.build_context()

        for agent in list(self.agents):
            if not getattr(agent, "alive", True):
                continue
            act_with_ctx = getattr(agent, "act_with_context", None)
            if callable(act_with_ctx):
                act_with_ctx(context)
            else:
                agent.act()

        # Preserve original environment updates (resources, metrics, kd-trees, time)
        self.update()

    # Convenience methods for future PettingZoo-style integrations
    def get_observations(self) -> Dict[str, np.ndarray]:
        observations: Dict[str, np.ndarray] = {}
        for agent in self.agents:
            if not agent.alive:
                continue
            if hasattr(agent, "get_state"):
                state = agent.get_state()
                observations[agent.agent_id] = state.to_tensor(device=agent.device).cpu().numpy()
        return observations

    def get_environment_state(self) -> EnvironmentState:
        return self.get_state()

