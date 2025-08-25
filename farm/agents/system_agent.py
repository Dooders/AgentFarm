from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.environment import Environment

from farm.actions.attack import attack_action
from farm.actions.gather import gather_action
from farm.actions.move import move_action
from farm.actions.reproduce import reproduce_action
from farm.actions.share import share_action
from farm.agents.base_agent import BaseAgent
from farm.core.action import Action


class SystemAgent(BaseAgent):
    """System-oriented agent implementation focused on cooperation."""

    def __init__(
        self,
        agent_id: str,
        position: tuple[int, int],
        resource_level: int,
        environment: "Environment",
        generation: int = 0,
        action_set: Optional[list[Action]] = None,
    ):
        """Initialize a SystemAgent.

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
        agent_params = {}
        if environment.config is not None:
            agent_params = environment.config.agent_parameters.get("SystemAgent", {})

        # Create default action set if none provided
        if action_set is None:
            action_set = [
                Action("move", 0.25, move_action),
                Action("gather", 0.25, gather_action),
                Action("share", 0.20, share_action),  # Higher weight for sharing
                Action("attack", 0.05, attack_action),  # Lower weight for attacking
                Action(
                    "reproduce", 0.25, reproduce_action
                ),  # Lower weight for reproducing
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

        # Configure gather module for more sustainable resource collection
        self.gather_module.config.efficiency_multiplier = 0.4  # Lower efficiency reward
        self.gather_module.config.cost_multiplier = 0.4  # Higher movement penalty
        self.gather_module.config.min_resource_threshold = (
            0.2  # Higher threshold for gathering
        )
