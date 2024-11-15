import numpy as np

from action import Action, gather_action, share_action
from actions.move import move_action
from actions.attack import attack_action

from .base_agent import BaseAgent


class IndependentAgent(BaseAgent):
    def __init__(self, agent_id, position, resource_level, environment):
        super().__init__(agent_id, position, resource_level, environment)

        # Override default actions with IndependentAgent-specific weights
        self.actions = [
            Action("move", 0.25, move_action),
            Action("gather", 0.35, gather_action),
            Action("share", 0.05, share_action),  # Lower weight for sharing
            Action("attack", 0.35, attack_action),  # Higher weight for attacking
        ]

        # Normalize weights
        total_weight = sum(action.weight for action in self.actions)
        for action in self.actions:
            action.weight /= total_weight
