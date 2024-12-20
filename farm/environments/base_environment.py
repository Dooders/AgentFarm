from abc import ABC, abstractmethod

import gymnasium as gym

from database.database import SimulationDatabase


class BaseEnvironment(gym.Env, ABC):
    def __init__(self, width, height, **kwargs):
        super().__init__()
        self.width = width
        self.height = height
        self.agents = []
        self.resources = []
        self.time = 0
        self.config = kwargs.get("config", None)
        self.db = SimulationDatabase("simulation.db")

    @abstractmethod
    def reset(self):
        """Reset the environment to its initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """Apply action and return next_state, reward, done, info."""
        pass

    @abstractmethod
    def render(self, mode="human"):
        """Render the environment."""
        pass

    def add_agent(self, agent):
        """Add an agent to the environment."""
        self.agents.append(agent)

    def add_resource(self, resource):
        """Add a resource to the environment."""
        self.resources.append(resource)

    def is_valid_position(self, position):
        """Check if a position is valid within the environment bounds."""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height
