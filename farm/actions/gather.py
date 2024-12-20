"""Resource gathering optimization module using Deep Q-Learning (DQN).

This module implements an intelligent gathering system that learns optimal gathering
strategies based on resource locations, amounts, and agent needs. It uses Deep Q-Learning
to make decisions about when and where to gather resources.

Key Components:
    - GatherQNetwork: Neural network for Q-value approximation of gathering actions
    - GatherModule: Main class handling gathering decisions and learning
    - Experience Replay: Stores gathering experiences for stable learning
    - Reward System: Complex reward structure based on gathering efficiency
"""

import logging
import random
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork
from farm.core.resources import Resource

if TYPE_CHECKING:

    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GatherConfig(BaseDQNConfig):
    """Configuration specific to gathering behavior."""

    # Reward parameters
    gather_success_reward: float = 1.0
    gather_fail_penalty: float = -0.1
    gather_efficiency_multiplier: float = 0.5  # Rewards gathering larger amounts
    gather_cost_multiplier: float = 0.3  # Penalizes movement costs

    # Gathering parameters
    min_resource_threshold: float = 0.1  # Minimum resource amount worth gathering
    max_wait_steps: int = 5  # Maximum steps to wait for resource regeneration


class GatherActionSpace:
    """Possible gathering actions."""

    GATHER: int = 0  # Attempt gathering
    WAIT: int = 1  # Wait for better opportunity
    SKIP: int = 2  # Skip gathering this step


class GatherQNetwork(BaseQNetwork):
    """Neural network for gathering decisions."""

    def __init__(self, input_dim: int = 6, hidden_size: int = 64) -> None:
        """
        Initialize the gathering Q-network.

        Args:
            input_dim: Size of input state (default: 6)
                - Distance to nearest resource
                - Resource amount
                - Agent's current resources
                - Resource density in area
                - Steps since last gather
                - Resource regeneration rate
            hidden_size: Number of neurons in hidden layers
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=3,  # GATHER, WAIT, or SKIP
            hidden_size=hidden_size,
        )


class GatherModule(BaseDQNModule):
    """Module for learning and executing optimal gathering strategies."""

    def __init__(
        self, config: GatherConfig = GatherConfig(), device: torch.device = DEVICE
    ) -> None:
        """Initialize the gathering module."""
        # Store dimensions as instance variables
        self.input_dim = 6  # State space dimension
        self.output_dim = 3  # Action space dimension (GATHER, WAIT, SKIP)

        super().__init__(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=config,
            device=device,
        )

        # Initialize Q-network after super().__init__
        self.q_network = GatherQNetwork(
            input_dim=self.input_dim, hidden_size=config.dqn_hidden_size
        ).to(device)

        self.target_network = GatherQNetwork(
            input_dim=self.input_dim, hidden_size=config.dqn_hidden_size
        ).to(device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.last_gather_step = 0
        self.steps_since_gather = 0
        self.consecutive_failed_attempts = 0

    def select_action(
        self, state_tensor: torch.Tensor, epsilon: Optional[float] = None
    ) -> int:
        """Select gathering action using epsilon-greedy strategy.

        Args:
            state_tensor: Current state observation
            epsilon: Override default epsilon value

        Returns:
            int: Selected action index from GatherActionSpace
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            # Random exploration
            return random.randint(0, self.output_dim - 1)

        with torch.no_grad():
            # Get Q-values and select best action
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def get_gather_decision(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[bool, Optional["Resource"]]:
        """
        Determine whether to gather resources and from which resource node.

        Args:
            agent: Agent making the gathering decision
            state: Current state tensor

        Returns:
            Tuple of (should_gather, target_resource)
        """
        # Ensure state is a tensor
        if not isinstance(state, torch.Tensor):
            state = self._process_gather_state(agent)

        # Get action from Q-network
        action = self.select_action(state)

        if action == GatherActionSpace.SKIP:
            return False, None

        if action == GatherActionSpace.WAIT:
            self.steps_since_gather += 1
            if self.steps_since_gather >= self.config.max_wait_steps:
                # Force gathering if waited too long
                action = GatherActionSpace.GATHER
            else:
                return False, None

        # Find best resource to gather from
        target_resource = self._find_best_resource(agent)
        should_gather = target_resource is not None

        # Store state for learning
        self.previous_state = state
        self.previous_action = action

        return should_gather, target_resource

    def _process_gather_state(self, agent: "BaseAgent") -> torch.Tensor:
        """Create state representation for gathering decisions."""
        closest_resource = self._find_best_resource(agent)

        if closest_resource is None:
            return torch.zeros(6, device=self.device)

        # Calculate resource density using KD-tree
        resources_in_range = agent.environment.get_nearby_resources(
            agent.position, agent.config.gathering_range
        )
        resource_density = len(resources_in_range) / (
            np.pi * agent.config.gathering_range**2
        )

        state = torch.tensor(
            [
                np.sqrt(
                    (
                        (np.array(closest_resource.position) - np.array(agent.position))
                        ** 2
                    ).sum()
                ),
                closest_resource.amount,
                agent.resource_level,
                resource_density,
                self.steps_since_gather,
                closest_resource.regeneration_rate,
            ],
            device=self.device,
            dtype=torch.float32,
        )

        return state

    def _find_best_resource(self, agent: "BaseAgent") -> Optional["Resource"]:
        """Find the most promising resource to gather from."""
        # Get resources within gathering range using KD-tree
        resources_in_range = agent.environment.get_nearby_resources(
            agent.position, agent.config.gathering_range
        )

        # Filter depleted resources
        resources_in_range = [
            r
            for r in resources_in_range
            if r.amount >= self.config.min_resource_threshold
        ]

        if not resources_in_range:
            return None

        # Score each resource based on amount and distance
        def score_resource(resource):
            distance = np.sqrt(
                ((np.array(resource.position) - np.array(agent.position)) ** 2).sum()
            )
            return (
                resource.amount * self.config.gather_efficiency_multiplier
                - distance * self.config.gather_cost_multiplier
            )

        return max(resources_in_range, key=score_resource)

    def calculate_gather_reward(
        self,
        agent: "BaseAgent",
        initial_resources: float,
        target_resource: Optional["Resource"],
    ) -> float:
        """Calculate reward for gathering attempt."""
        if target_resource is None:
            return self.config.gather_fail_penalty

        resources_gained = agent.resource_level - initial_resources

        if resources_gained <= 0:
            self.consecutive_failed_attempts += 1
            return self.config.gather_fail_penalty * self.consecutive_failed_attempts

        self.consecutive_failed_attempts = 0
        self.steps_since_gather = 0

        # Calculate efficiency bonus
        efficiency = resources_gained / target_resource.max_amount
        efficiency_bonus = efficiency * self.config.gather_efficiency_multiplier

        # Calculate base reward
        base_reward = self.config.gather_success_reward * resources_gained

        return base_reward + efficiency_bonus


def gather_action(agent: "BaseAgent") -> None:
    """Execute gathering action using the gather module."""
    # Get current state
    state = agent.gather_module._process_gather_state(agent)
    initial_resources = agent.resource_level

    # Get gathering decision
    should_gather, target_resource = agent.gather_module.get_gather_decision(
        agent, state
    )

    if not should_gather or not target_resource:
        # Log skipped gather action
        if agent.environment.db is not None:
            agent.environment.db.logger.log_agent_action(
                step_number=agent.environment.time,
                agent_id=agent.agent_id,
                action_type="gather",
                position_before=agent.position,
                position_after=agent.position,
                resources_before=initial_resources,
                resources_after=initial_resources,
                reward=0,
                details={
                    "success": False,
                    "reason": (
                        "decided_not_to_gather"
                        if not should_gather
                        else "no_target_resource"
                    ),
                },
            )
        return

    # Record initial resource amount
    resource_amount_before = target_resource.amount

    # Attempt gathering
    if not target_resource.is_depleted():
        gather_amount = min(agent.config.max_gather_amount, target_resource.amount)
        target_resource.consume(gather_amount)
        agent.resource_level += gather_amount

        # Calculate reward
        reward = agent.gather_module.calculate_gather_reward(
            agent, initial_resources, target_resource
        )
        agent.total_reward += reward

        # Log successful gather action
        if agent.environment.db is not None:
            agent.environment.db.logger.log_agent_action(
                step_number=agent.environment.time,
                agent_id=agent.agent_id,
                action_type="gather",
                position_before=agent.position,
                position_after=agent.position,
                resources_before=initial_resources,
                resources_after=agent.resource_level,
                reward=reward,
                details={
                    "success": True,
                    "amount_gathered": gather_amount,
                    "resource_before": resource_amount_before,
                    "resource_after": target_resource.amount,
                    "resource_depleted": target_resource.is_depleted(),
                    "distance_to_resource": np.linalg.norm(
                        np.array(target_resource.position) - np.array(agent.position)
                    ),
                },
            )
