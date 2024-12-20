import logging
import os
import random

from base_environment import BaseEnvironment

from agents import IndependentAgent
from database.database import SimulationDatabase
from core.resources import Resource

logger = logging.getLogger(__name__)


class Environment(BaseEnvironment):
    def __init__(
        self,
        width,
        height,
        resource_distribution,
        db_path="simulation_results.db",
        **kwargs,
    ):
        super().__init__(width, height, **kwargs)
        self.resource_distribution = resource_distribution
        self.db_path = db_path
        self.db = self._initialize_database()
        self.agents = []
        self.resources = []
        self.time = 0

        # Initialize resources and agents
        self.initialize_resources(resource_distribution)
        self._initialize_agents()

    def _initialize_database(self):
        """Initialize or reset the database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        return SimulationDatabase(self.db_path)

    def reset(self):
        """Reset the environment."""
        self.time = 0
        self.agents = []
        self.resources = []
        self.initialize_resources(self.resource_distribution)
        self._initialize_agents()

    def step(self, action):
        """Update the environment state."""
        self.time += 1
        # Example action handling logic
        for agent in self.agents:
            if agent.alive:
                agent.perform_action(action)
        self._update_resources()
        return self.get_state(), {}, False, {}

    def render(self, mode="human"):
        """Render the environment."""
        print(f"Environment at time {self.time}")
        for agent in self.agents:
            print(agent)
        for resource in self.resources:
            print(resource)

    def initialize_resources(self, distribution):
        """Initialize resources in the environment."""
        for _ in range(distribution.get("amount", 0)):
            position = (random.uniform(0, self.width), random.uniform(0, self.height))
            resource = Resource(
                resource_id=random.randint(0, 1000),
                position=position,
                amount=random.randint(1, 10),
            )
            self.resources.append(resource)

    def _initialize_agents(self):
        """Initialize agents in the environment."""
        # Example agent initialization logic
        for _ in range(5):  # Adjust the number of agents as needed
            position = (random.uniform(0, self.width), random.uniform(0, self.height))
            agent = IndependentAgent(
                agent_id=random.randint(0, 1000), position=position, environment=self
            )
            self.add_agent(agent)

    def _update_resources(self):
        """Update resource regeneration logic."""
        for resource in self.resources:
            resource.amount = min(resource.amount + 1, 10)  # Example regeneration logic

    def get_state(self):
        """Return the current environment state."""
        return {
            "time": self.time,
            "agents": [str(agent) for agent in self.agents],
            "resources": [str(resource) for resource in self.resources],
        }
