from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from farm.core.environment import Environment

from farm.actions.attack import attack_action
from farm.actions.gather import gather_action
from farm.actions.move import move_action
from farm.actions.share import share_action
from farm.agents.base_agent import BaseAgent
from farm.core.action import Action


class ControlAgent(BaseAgent):
    """A balanced agent implementation that maintains equilibrium between
    cooperative and individualistic behaviors."""

    def __init__(
        self,
        agent_id: str,
        position: tuple[int, int],
        resource_level: int,
        environment: "Environment",
        generation: int = 0,
        action_set: list[Action] = None,
    ):
        """Initialize a ControlAgent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent
        position : tuple[int, int]
            Initial (x,y) coordinates
        resource_level : int
            Starting resource amount
        environment : Environment
            Reference to simulation environment
        generation : int
            Generation number in evolutionary lineage
        action_set : list[Action], optional
            Custom action set for this agent
        """
        # Get agent-specific parameters from config
        agent_params = environment.config.agent_parameters.get("ControlAgent", {})

        # Create default action set if none provided
        if action_set is None:
            action_set = [
                Action("move", 0.30, move_action),  # Balanced movement
                Action("gather", 0.40, gather_action),  # Moderate focus on gathering
                Action("share", 0.15, share_action),  # Moderate sharing
                Action("attack", 0.15, attack_action),  # Moderate aggression
            ]

        # Initialize base agent with custom action set and genealogy info
        super().__init__(
            agent_id=agent_id,
            position=position,
            resource_level=resource_level,
            environment=environment,
            action_set=action_set,
            generation=generation,
        )

        # Configure gather module with balanced parameters
        self.gather_module.config.gather_efficiency_multiplier = (
            0.55  # Balanced efficiency
        )
        self.gather_module.config.gather_cost_multiplier = (
            0.3  # Balanced movement penalty
        )
        self.gather_module.config.min_resource_threshold = 0.125  # Balanced threshold
