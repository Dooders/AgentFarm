import logging
import random
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from action import *
from actions.move import MoveModule

if TYPE_CHECKING:
    from environment import Environment

logger = logging.getLogger(__name__)


class AgentModel(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(AgentModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = nn.Sequential(
            nn.Linear(input_dim, config.dqn_hidden_size),
            nn.ReLU(),
            nn.Linear(config.dqn_hidden_size, config.dqn_hidden_size),
            nn.ReLU(),
            nn.Linear(config.dqn_hidden_size, output_dim),
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=config.memory_size)
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.config = config

    def forward(self, x):
        return self.network(x)

    def learn(self, batch):
        if len(batch) < self.config.batch_size:
            return None

        states = torch.stack([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch], device=self.device)
        rewards = torch.tensor(
            [x[2] for x in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([x[3] for x in batch])

        with torch.no_grad():
            next_q_values = self(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values)

        current_q_values = self(states).gather(1, actions.unsqueeze(1))
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()


class AgentState:
    def __init__(self, distance, angle, resource_level, target_resource_amount):
        self.normalized_distance = distance  # Distance to nearest resource (normalized by diagonal of environment)
        self.normalized_angle = angle  # Angle to nearest resource (normalized by π)
        self.normalized_resource_level = (
            resource_level  # Agent's current resources (normalized by 20)
        )
        self.normalized_target_amount = (
            target_resource_amount  # Target resource amount (normalized by 20)
        )

    def to_tensor(self, device):
        return torch.FloatTensor(
            [
                self.normalized_distance,
                self.normalized_angle,
                self.normalized_resource_level,
                self.normalized_target_amount,
            ]
        ).to(device)


BASE_ACTION_SET = [
    Action("move", 0.4, move_action),
    Action("gather", 0.3, gather_action),
    Action("share", 0.2, share_action),
    Action("attack", 0.1, attack_action),
]


class BaseAgent:
    def __init__(
        self,
        agent_id: int,
        position: tuple[int, int],
        resource_level: int,
        environment: "Environment",
        action_set: list[Action] = BASE_ACTION_SET,
    ):
        # Add default actions
        self.actions = action_set

        # Normalize weights
        total_weight = sum(action.weight for action in self.actions)
        for action in self.actions:
            action.weight /= total_weight

        self.agent_id = agent_id
        self.position = position
        self.resource_level = resource_level
        self.alive = True
        self.environment = environment
        self.config = environment.config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AgentModel(
            input_dim=len(self.get_state()),
            output_dim=len(self.actions),
            config=self.config,
        )
        self.last_state = None
        self.last_action = None
        self.max_movement = self.config.max_movement
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []
        self.starvation_threshold = self.config.starvation_threshold
        self.max_starvation = self.config.max_starvation_time
        self.birth_time = environment.time
        logger.info(
            f"Agent {self.agent_id} created at {self.position} during step {environment.time}"
        )

        # Log agent creation to database
        environment.db.log_agent(
            agent_id=self.agent_id,
            birth_time=environment.time,
            agent_type=self.__class__.__name__,
            position=self.position,
            initial_resources=self.resource_level,
        )

        # Add move module
        self.move_module = MoveModule(self.config)

    def get_state(self):
        # Get closest resource position
        closest_resource = None
        min_distance = float("inf")
        for resource in self.environment.resources:
            if resource.amount > 0:
                dist = np.sqrt(
                    (self.position[0] - resource.position[0]) ** 2
                    + (self.position[1] - resource.position[1]) ** 2
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_resource = resource

        if closest_resource is None:
            return torch.zeros(4, device=self.device)

        # Calculate normalized values
        dx = closest_resource.position[0] - self.position[0]
        dy = closest_resource.position[1] - self.position[1]
        angle = np.arctan2(dy, dx)

        normalized_distance = min_distance / np.sqrt(
            self.environment.width**2 + self.environment.height**2
        )
        normalized_angle = angle / np.pi
        normalized_resource_level = self.resource_level / 20
        normalized_target_amount = closest_resource.amount / 20

        state = AgentState(
            distance=normalized_distance,
            angle=normalized_angle,
            resource_level=normalized_resource_level,
            target_resource_amount=normalized_target_amount,
        )

        return state.to_tensor(self.device)

    def learn(self, reward):
        if self.last_state is None:
            return

        self.total_reward += reward
        self.episode_rewards.append(reward)

        # Store experience
        self.model.memory.append(
            (self.last_state, self.last_action, reward, self.get_state())
        )

        # Only train on larger batches less frequently
        if (
            len(self.model.memory) >= self.config.batch_size * 4
            and len(self.model.memory) % (self.config.training_frequency * 4) == 0
        ):

            batch = random.sample(self.model.memory, self.config.batch_size * 4)
            loss = self.model.learn(batch)
            if loss is not None:
                self.losses.append(loss)

    def select_action(self):
        """Select an action using a combination of weighted probabilities and state awareness.
        
        Uses both predefined weights and current state to make intelligent decisions:
        1. Gets base probabilities from action weights
        2. Adjusts probabilities based on current state
        3. Applies epsilon-greedy exploration
        
        Returns:
            Action: Selected action object to execute
        """
        # Get base probabilities from weights
        actions = [action for action in self.actions]
        action_weights = [action.weight for action in actions]
        
        # Normalize base weights
        total_weight = sum(action_weights)
        base_probs = [weight / total_weight for weight in action_weights]
        
        # State-based adjustments
        adjusted_probs = self._adjust_probabilities(base_probs)
        
        # Epsilon-greedy exploration
        if random.random() < self.model.epsilon:
            # Random exploration
            return random.choice(actions)
        else:
            # Weighted selection using adjusted probabilities
            return random.choices(actions, weights=adjusted_probs, k=1)[0]

    def _adjust_probabilities(self, base_probs):
        """Adjust action probabilities based on agent's current state.
        
        Uses configurable multipliers to adjust probabilities based on:
        - Resource levels
        - Nearby resources
        - Nearby agents
        - Current health/starvation
        
        Args:
            base_probs (list[float]): Original action probabilities
            
        Returns:
            list[float]: Adjusted probability distribution
        """
        adjusted_probs = base_probs.copy()
        
        # Get relevant state information
        state = self.get_state()
        resource_level = self.resource_level
        starvation_risk = self.starvation_threshold / self.max_starvation
        
        # Find nearby entities
        nearby_resources = [r for r in self.environment.resources 
                           if not r.is_depleted() and 
                           np.sqrt(((np.array(r.position) - np.array(self.position)) ** 2).sum()) 
                           < self.config.gathering_range]
        
        nearby_agents = [a for a in self.environment.agents 
                        if a != self and a.alive and
                        np.sqrt(((np.array(a.position) - np.array(self.position)) ** 2).sum()) 
                        < self.config.social_range]
        
        # Adjust move probability
        move_idx = next(i for i, a in enumerate(self.actions) if a.name == "move")
        if not nearby_resources:
            # Increase move probability if no resources nearby
            adjusted_probs[move_idx] *= self.config.move_mult_no_resources
        
        # Adjust gather probability
        gather_idx = next(i for i, a in enumerate(self.actions) if a.name == "gather")
        if nearby_resources and resource_level < self.config.min_reproduction_resources:
            # Increase gather probability if resources needed
            adjusted_probs[gather_idx] *= self.config.gather_mult_low_resources
        
        # Adjust share probability
        share_idx = next(i for i, a in enumerate(self.actions) if a.name == "share")
        if resource_level > self.config.min_reproduction_resources and nearby_agents:
            # Increase share probability if wealthy and agents nearby
            adjusted_probs[share_idx] *= self.config.share_mult_wealthy
        else:
            # Decrease share probability if resources needed
            adjusted_probs[share_idx] *= self.config.share_mult_poor
        
        # Adjust attack probability
        attack_idx = next(i for i, a in enumerate(self.actions) if a.name == "attack")
        if starvation_risk > self.config.attack_starvation_threshold and nearby_agents and resource_level > 2:
            # Increase attack probability if desperate
            adjusted_probs[attack_idx] *= self.config.attack_mult_desperate
        else:
            # Decrease attack probability if stable
            adjusted_probs[attack_idx] *= self.config.attack_mult_stable
        
        # Renormalize probabilities
        total = sum(adjusted_probs)
        adjusted_probs = [p/total for p in adjusted_probs]
        
        return adjusted_probs

    def act(self):
        # First check if agent should die
        if not self.alive:
            return
        initial_resources = self.resource_level
        # Base resource consumption
        self.resource_level -= self.config.base_consumption_rate

        if self.resource_level <= 0:
            self.starvation_threshold += 1
            if self.starvation_threshold >= self.max_starvation:
                self.die()
                return
        else:
            self.starvation_threshold = 0

        # Select and execute an action
        action = self.select_action()
        action.execute(self)

        # Calculate reward based on resource change
        reward = self.resource_level - initial_resources
        self.learn(reward)

    def reproduce(self):
        if len(self.environment.agents) >= self.config.max_population:
            return

        if self.resource_level >= self.config.min_reproduction_resources:
            if self.resource_level >= self.config.offspring_cost + 2:
                new_agent = self.create_offspring()
                self.environment.add_agent(new_agent)
                self.resource_level -= self.config.offspring_cost

                logger.info(
                    f"Agent {self.agent_id} reproduced at {self.position} during step {self.environment.time} creating agent {new_agent.agent_id}"
                )

    def create_offspring(self):
        return type(self)(
            agent_id=self.environment.get_next_agent_id(),
            position=self.position,
            resource_level=self.config.offspring_initial_resources,
            environment=self.environment,
        )

    def die(self):
        """
        Handle agent death by:
        1. Setting alive status to False
        2. Logging death to database
        3. Notifying environment for visualization updates
        """
        self.alive = False

        # Log death to database
        self.environment.db.update_agent_death(
            agent_id=self.agent_id, death_time=self.environment.time
        )

        # Remove agent from environment's active agents list
        if hasattr(self.environment, "agents"):
            try:
                self.environment.agents.remove(self)
            except ValueError:
                pass  # Agent was already removed

        logger.info(
            f"Agent {self.agent_id} died at {self.position} during step {self.environment.time}"
        )

    def get_environment(self) -> "Environment":
        return self._environment

    def set_environment(self, environment: "Environment") -> None:
        self._environment = environment