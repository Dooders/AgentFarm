from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from core.environment import Environment

from core.action import Action
from actions.attack import attack_action
from actions.gather import gather_action
from actions.move import move_action
from actions.share import share_action

from .base_agent import BaseAgent


class IndependentAgent(BaseAgent):
    """An agent that makes independent decisions based on its own learning."""

    def __init__(
        self,
        agent_id: int,
        position: tuple[int, int],
        resource_level: int,
        environment: "Environment",
        parent_id: Optional[int] = None,
        generation: int = 0,
        skip_logging: bool = False,
        action_set: list[Action] = None  # Make action_set optional
    ):
        """Initialize an IndependentAgent.

        Parameters
        ----------
        agent_id : int
            Unique identifier for this agent
        position : tuple[int, int]
            Initial (x,y) coordinates
        resource_level : int
            Starting resource amount
        environment : Environment
            Reference to simulation environment
        parent_id : Optional[int]
            ID of parent agent if this agent was created through reproduction
        generation : int
            Generation number in evolutionary lineage
        skip_logging : bool
            If True, skip database logging during initialization
        action_set : list[Action], optional
            Custom action set for this agent
        """
        # Get agent-specific parameters from config
        agent_params = environment.config.agent_parameters.get("IndependentAgent", {})

        # Create default action set if none provided
        if action_set is None:
            action_set = [
                Action("move", 0.25, move_action),
                Action("gather", 0.45, gather_action),  # Higher weight for gathering
                Action("share", 0.05, share_action),  # Lower weight for sharing
                Action("attack", 0.25, attack_action),  # Moderate weight for attacking
            ]

        # Initialize base agent with custom action set and genealogy info
        super().__init__(
            agent_id=agent_id,
            position=position,
            resource_level=resource_level,
            environment=environment,
            action_set=action_set,
            parent_id=parent_id,
            generation=generation,
            skip_logging=skip_logging
        )

        # Configure gather module for more aggressive resource collection
        self.gather_module.config.gather_efficiency_multiplier = 0.7  # Higher efficiency reward
        self.gather_module.config.gather_cost_multiplier = 0.2  # Lower movement penalty
        self.gather_module.config.min_resource_threshold = 0.05  # Lower threshold for gathering
