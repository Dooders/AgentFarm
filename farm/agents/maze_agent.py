from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.environment import Environment

from farm.actions.move import move_action
from farm.agents.base_agent import BaseAgent
from farm.core.action import Action


class MazeAgent(BaseAgent):
    """An agent specialized for maze navigation using DQN learning."""

    def __init__(
        self,
        agent_id: int,
        position: tuple[int, int],
        resource_level: int,
        environment: "Environment",
        generation: int = 0,
        action_set: list[Action] = None,
    ):
        # Create maze-specific action set if none provided
        if action_set is None:
            action_set = [
                Action("move_up", 0.25, move_action),
                Action("move_down", 0.25, move_action),
                Action("move_left", 0.25, move_action),
                Action("move_right", 0.25, move_action),
            ]

        # Initialize base agent
        super().__init__(
            agent_id=agent_id,
            position=position,
            resource_level=resource_level,
            environment=environment,
            action_set=action_set,
            generation=generation,
        )

        # Configure maze-specific parameters
        self.move_module.config.max_movement = 1  # Limit to single-cell moves
        self.move_module.config.movement_cost = 0.01  # Small movement penalty

    def calculate_move_reward(self, old_pos, new_pos):
        """Override move reward calculation for maze navigation."""
        # Base movement cost
        reward = -0.01

        # Check if reached goal
        if new_pos == self.environment.goal:
            return 100

        # Calculate distances to goal
        old_distance = np.linalg.norm(
            np.array(self.environment.goal) - np.array(old_pos)
        )
        new_distance = np.linalg.norm(
            np.array(self.environment.goal) - np.array(new_pos)
        )

        # Reward for moving closer to goal
        reward += 10 * (old_distance - new_distance)

        # Penalty for invalid moves (hitting walls)
        if old_pos == new_pos:
            reward -= 0.1

        return reward

    def act(self, state: np.ndarray) -> Action:
        """Override act method for maze navigation."""
        return super().act(state)
