"""Resource gathering optimization module using Deep Q-Learning (DQN).

This module implements an intelligent gathering system that learns optimal gathering
strategies based on resource locations, amounts, and agent needs. It uses Deep Q-Learning
to make decisions about when and where to gather resources.

The gathering system operates through several key components:

Key Components:
    - GatherQNetwork: Neural network for Q-value approximation of gathering actions
    - GatherModule: Main class handling gathering decisions and learning
    - Experience Replay: Stores gathering experiences for stable learning
    - Reward System: Complex reward structure based on gathering efficiency

The system considers multiple factors when making gathering decisions:
    - Distance to available resources
    - Resource amounts and regeneration rates
    - Agent's current resource levels
    - Resource density in the surrounding area
    - Time since last successful gathering attempt
    - Movement costs and efficiency trade-offs

Actions Available:
    - GATHER: Attempt to gather from the best available resource
    - WAIT: Delay gathering to wait for better opportunities
    - SKIP: Skip gathering this step entirely

The reward system encourages efficient gathering by:
    - Rewarding successful resource collection
    - Penalizing failed attempts with increasing penalties
    - Providing efficiency bonuses for gathering larger amounts
    - Considering movement costs in the decision process
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
    """Configuration for gathering behavior and reward parameters.

    This class defines all the parameters that control how the gathering module
    behaves, including reward values, thresholds, and timing parameters.

    Attributes:
        gather_success_reward: Base reward for successful resource gathering
        gather_fail_penalty: Base penalty for failed gathering attempts
        gather_efficiency_multiplier: Multiplier for efficiency bonuses
        gather_cost_multiplier: Multiplier for movement cost penalties
        min_resource_threshold: Minimum resource amount worth gathering
        max_wait_steps: Maximum steps to wait before forcing gathering
    """

    # Reward parameters
    gather_success_reward: float = 1.0
    gather_fail_penalty: float = -0.1
    gather_efficiency_multiplier: float = 0.5  # Rewards gathering larger amounts
    gather_cost_multiplier: float = 0.3  # Penalizes movement costs

    # Gathering parameters
    min_resource_threshold: float = 0.1  # Minimum resource amount worth gathering
    max_wait_steps: int = 5  # Maximum steps to wait for resource regeneration


class GatherActionSpace:
    """Defines the possible actions for the gathering decision process.

    The gathering module can choose from three distinct actions:

    Attributes:
        GATHER: Attempt to gather resources from the best available target
        WAIT: Delay gathering to wait for better opportunities
        SKIP: Skip gathering entirely for this time step
    """

    GATHER: int = 0  # Attempt gathering
    WAIT: int = 1  # Wait for better opportunity
    SKIP: int = 2  # Skip gathering this step


class GatherQNetwork(BaseQNetwork):
    """Neural network for Q-value approximation of gathering decisions.

    This network takes the current gathering state as input and outputs Q-values
    for each possible gathering action (GATHER, WAIT, SKIP). The network learns
    to approximate the expected future rewards for each action given the current
    state, enabling the agent to make optimal gathering decisions.

    The input state includes:
        - Distance to nearest resource
        - Resource amount available
        - Agent's current resource level
        - Resource density in the area
        - Steps since last successful gather
        - Resource regeneration rate
    """

    def __init__(self, input_dim: int = 6, hidden_size: int = 64) -> None:
        """
        Initialize the gathering Q-network.

        Args:
            input_dim: Size of input state vector (default: 6)
                The state vector contains:
                - Distance to nearest resource
                - Resource amount
                - Agent's current resources
                - Resource density in area
                - Steps since last gather
                - Resource regeneration rate
            hidden_size: Number of neurons in hidden layers (default: 64)
                Controls the network's capacity to learn complex patterns
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=3,  # GATHER, WAIT, or SKIP
            hidden_size=hidden_size,
        )


class GatherModule(BaseDQNModule):
    """Module for learning and executing optimal gathering strategies.

    This module implements the core gathering decision-making logic using Deep Q-Learning.
    It maintains the Q-networks, handles state processing, and manages the learning
    process for gathering decisions.

    The module tracks various metrics to improve decision making:
        - Steps since last successful gathering
        - Consecutive failed attempts
        - Resource availability and quality

    Key Features:
        - Epsilon-greedy action selection for exploration vs exploitation
        - Experience replay for stable learning
        - Target network for stable Q-value updates
        - Sophisticated reward calculation based on efficiency
    """

    def __init__(
        self, config: GatherConfig = GatherConfig(), device: torch.device = DEVICE
    ) -> None:
        """Initialize the gathering module with configuration and device settings.

        Args:
            config: Configuration object containing gathering parameters
            device: PyTorch device for network computations (CPU/GPU)

        The module initializes:
            - Q-network and target network for learning
            - State and action space dimensions
            - Tracking variables for gathering metrics
        """
        # Store dimensions as instance variables
        self.input_dim = 6  # State space dimension
        self.output_dim = 3  # Action space dimension (GATHER, WAIT, SKIP)

        super().__init__(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            config=config,
            device=device,
        )

        # Ensure we're using the correct config type
        self.config: GatherConfig = config

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

        This method implements the epsilon-greedy exploration strategy. With probability
        epsilon, a random action is selected for exploration. Otherwise, the action
        with the highest Q-value is selected for exploitation.

        Args:
            state_tensor: Current state observation as a PyTorch tensor
            epsilon: Exploration rate override (uses module's epsilon if None)

        Returns:
            int: Selected action index from GatherActionSpace (0=GATHER, 1=WAIT, 2=SKIP)
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Type assertion to help linter understand epsilon is not None
        assert epsilon is not None
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

        This method processes the current state through the Q-network to determine
        the optimal gathering action. It handles the WAIT action by tracking time
        and forcing gathering if too much time has passed.

        Args:
            agent: Agent making the gathering decision
            state: Current state tensor representing the gathering environment

        Returns:
            Tuple of (should_gather, target_resource):
                - should_gather: Boolean indicating if gathering should be attempted
                - target_resource: The resource to gather from, or None if no gathering
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
            if self.steps_since_gather >= self.config.max_wait_steps:  # type: ignore
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
        """Create state representation for gathering decisions.

        This method constructs a 6-dimensional state vector that captures all
        relevant information for making gathering decisions. The state includes
        environmental factors, agent status, and resource characteristics.

        Args:
            agent: The agent for which to process the state

        Returns:
            torch.Tensor: 6-dimensional state vector containing:
                - Distance to nearest resource
                - Resource amount available
                - Agent's current resource level
                - Resource density in the area
                - Steps since last successful gather
                - Resource regeneration rate
        """
        closest_resource = self._find_best_resource(agent)

        if closest_resource is None:
            return torch.zeros(6, device=self.device)

        # Calculate resource density using KD-tree
        if agent.environment is None or agent.environment.config is None:
            return torch.zeros(6, device=self.device)

        # Type assertion to help linter understand config structure
        config = agent.config
        assert config is not None, "Agent config cannot be None"
        
        resources_in_range = agent.environment.get_nearby_resources(
            agent.position, config.gathering_range
        )
        resource_density = len(resources_in_range) / (
            np.pi * config.gathering_range**2
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
        """Find the most promising resource to gather from.

        This method evaluates all available resources within the agent's gathering
        range and selects the optimal target based on a scoring system that
        considers resource amount, distance, and efficiency factors.

        The scoring algorithm:
            - Filters out depleted resources below the minimum threshold
            - Calculates distance-based costs
            - Applies efficiency multipliers for resource amounts
            - Returns the resource with the highest score

        Args:
            agent: The agent seeking resources

        Returns:
            Optional[Resource]: The best resource to gather from, or None if none available
        """
        # Get resources within gathering range using KD-tree
        if agent.environment is None or agent.environment.config is None:
            return None

        # Type assertion to help linter understand config structure
        config = agent.config
        assert config is not None, "Agent config cannot be None"
        
        resources_in_range = agent.environment.get_nearby_resources(
            agent.position, config.gathering_range
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
            """Calculate a score for a resource based on amount and distance.

            This function implements the scoring algorithm used to rank available
            resources. The score considers both the resource amount and the distance
            to the agent, with configurable multipliers for efficiency and cost.

            Args:
                resource: The resource to score

            Returns:
                float: Score value (higher is better)
                    Positive values favor resources with high amounts and low distances
            """
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
        """Calculate reward for gathering attempt.

        This method implements a sophisticated reward system that considers multiple
        factors including success/failure, efficiency, and consecutive failures.

        Reward Components:
            - Base success reward for gathering resources
            - Efficiency bonus for gathering larger amounts
            - Penalties for failed attempts (increasing with consecutive failures)
            - Resource utilization bonuses

        Args:
            agent: The agent that attempted gathering
            initial_resources: Agent's resource level before gathering
            target_resource: The resource that was targeted (None if no target)

        Returns:
            float: Calculated reward value (positive for success, negative for failure)
        """
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
    """Execute gathering action using the gather module.

    This function orchestrates the complete gathering process:
    1. Processes the current state for decision making
    2. Determines whether and where to gather
    3. Executes the gathering action if appropriate
    4. Calculates and applies rewards
    5. Logs the action results for analysis

    The function handles both successful and failed gathering attempts,
    updating the agent's resource levels and logging detailed information
    about the gathering process for analysis and debugging.

    Args:
        agent: The agent performing the gathering action
    """
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
        if agent.environment is None or agent.environment.config is None:
            return

        # Type assertion to help linter understand config structure
        config = agent.config
        assert config is not None, "Agent config cannot be None"
        
        gather_amount = min(
            config.max_gather_amount, target_resource.amount
        )
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
