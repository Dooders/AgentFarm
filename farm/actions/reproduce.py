"""Reproduction learning module using Deep Q-Learning (DQN).

This module implements intelligent reproduction strategies using Deep Q-Learning,
allowing agents to learn optimal timing and conditions for reproduction based on
environmental factors, resource availability, and population dynamics.

The module provides a complete reproduction decision-making system that considers:
- Agent health and resource levels
- Local population density and resource availability
- Global population balance
- Environmental conditions and spatial constraints

Key Components:
    - ReproduceConfig: Configuration parameters for reproduction behavior and rewards
    - ReproduceActionSpace: Defines possible reproduction actions (WAIT/REPRODUCE)
    - ReproduceQNetwork: Neural network for learning reproduction decisions
    - ReproduceModule: Main class handling reproduction logic and DQN learning
    - Experience Replay: Stores reproduction outcomes for continuous learning
    - Reward System: Encourages successful population growth and sustainability

The system uses an 8-dimensional state space including resource ratios, health status,
local density, resource availability, population balance, starvation risk, defensive
status, and generation information.
"""

import logging
import random
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder
from farm.actions.config import ReproduceConfig

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReproduceActionSpace:
    """Defines the action space for reproduction decisions.

    This class contains the constants that define the possible actions
    an agent can take regarding reproduction. The actions are designed
    to be mutually exclusive and cover all possible reproduction scenarios.

    Attributes:
        WAIT: Action to wait for better conditions before reproducing
        REPRODUCE: Action to attempt reproduction immediately
    """

    WAIT: int = 0  # Wait for better conditions
    REPRODUCE: int = 1  # Attempt reproduction


class ReproduceQNetwork(BaseQNetwork):
    """Neural network specialized for reproduction decision-making.

    This Q-network is specifically designed for reproduction decisions,
    taking an 8-dimensional state representation and outputting Q-values
    for the two possible reproduction actions (WAIT or REPRODUCE).

    The network architecture is inherited from BaseQNetwork but configured
    specifically for reproduction scenarios with appropriate input and output
    dimensions.

    Args:
        input_dim: Dimension of the input state vector (default: 8)
        hidden_size: Number of neurons in hidden layers (default: 64)
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_size: int = 64,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=2,  # WAIT or REPRODUCE
            hidden_size=hidden_size,
            shared_encoder=shared_encoder,
        )


class ReproduceModule(BaseDQNModule):
    """Main module for learning and executing reproduction strategies.

    This module implements the complete reproduction decision-making system
    using Deep Q-Learning. It handles state processing, action selection,
    reward calculation, and learning from reproduction outcomes.

    The module maintains both a main Q-network and a target network for
    stable learning, and includes experience replay for efficient learning
    from past reproduction attempts.

    Args:
        config: Configuration object containing reproduction parameters
        device: PyTorch device for tensor operations (CPU/GPU)
    """

    def __init__(
        self,
        config: ReproduceConfig = ReproduceConfig(),
        device: torch.device = DEVICE,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        super().__init__(
            input_dim=8,  # State dimensions for reproduction (matches actual state size)
            output_dim=2,  # Number of reproduction actions
            config=config,
            device=device,
        )

        # Initialize reproduction-specific Q-network with shared encoder if provided
        self.q_network = ReproduceQNetwork(
            input_dim=8,
            hidden_size=config.dqn_hidden_size,
            shared_encoder=shared_encoder,
        ).to(device)

        self.target_network = ReproduceQNetwork(
            input_dim=8,
            hidden_size=config.dqn_hidden_size,
            shared_encoder=shared_encoder,
        ).to(device)

        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(
        self, state: torch.Tensor, epsilon: Optional[float] = None
    ) -> int:
        """Select reproduction action using epsilon-greedy strategy.

        This method implements the epsilon-greedy exploration strategy for
        reproduction decisions. With probability epsilon, a random action is
        chosen for exploration. Otherwise, the action with the highest Q-value
        is selected for exploitation.

        Args:
            state: Current state tensor containing agent and environment information
            epsilon: Optional override for exploration rate. If None, uses the
                   module's default epsilon value.

        Returns:
            int: Selected action (ReproduceActionSpace.WAIT or ReproduceActionSpace.REPRODUCE)
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Type assertion to help linter understand epsilon is not None
        assert epsilon is not None
        if random.random() < epsilon:
            return random.randint(0, 1)  # Random choice between WAIT and REPRODUCE

        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def get_reproduction_decision(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[bool, float]:
        """Determine whether to reproduce based on current state and confidence.

        This method processes the current state and makes a reproduction decision
        using the trained Q-network. It returns both the decision (whether to
        reproduce) and a confidence score indicating the model's certainty.

        Args:
            agent: The agent considering reproduction
            state: Current state tensor containing environmental and agent information

        Returns:
            Tuple[bool, float]: A tuple containing:
                - should_reproduce: Boolean indicating whether reproduction should occur
                - confidence_score: Float between 0 and 1 indicating decision confidence
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        # Get Q-values and select action
        action = self.select_action(state)

        if action == ReproduceActionSpace.WAIT:
            return False, 0.0

        # Check reproduction conditions
        if not _check_reproduction_conditions(agent):
            return False, 0.0

        # Calculate confidence score
        with torch.no_grad():
            q_values = self.q_network(state)
            confidence = torch.softmax(q_values, dim=0)[
                ReproduceActionSpace.REPRODUCE
            ].item()

        return True, confidence


def reproduce_action(agent: "BaseAgent") -> None:
    """Execute reproduction action using the agent's reproduce module.

    This function is the main entry point for reproduction actions. It handles
    the complete reproduction process including:
    - State assessment and decision making
    - Condition validation
    - Offspring creation
    - Reward calculation and assignment
    - Comprehensive logging of reproduction events

    The function integrates with the agent's reproduce_module to make intelligent
    decisions and handles both successful and failed reproduction attempts with
    appropriate error handling and logging.

    Args:
        agent: The agent attempting to reproduce
    """
    # Get current state
    state = _get_reproduce_state(agent)
    initial_resources = agent.resource_level

    # Get reproduction decision
    should_reproduce, confidence = agent.reproduce_module.get_reproduction_decision(
        agent, state
    )

    if not should_reproduce or not _check_reproduction_conditions(agent):
        # Log failed reproduction event
        if agent.environment.db is not None:
            agent.environment.db.logger.log_agent_action(
                agent_id=agent.agent_id,
                action_type="reproduce",
                step_number=agent.environment.time,
                details={
                    "success": False,
                    "reason": "conditions_not_met",
                    "confidence": confidence if should_reproduce else 0.0,
                },
            )
        # Reproduction is a self-contained action; log as self-edge for lineage attempt
        if getattr(agent, "logging_service", None) is not None:
            agent.logging_service.log_interaction_edge(
            source_type="agent",
            source_id=agent.agent_id,
            target_type="agent",
            target_id=agent.agent_id,
            interaction_type="reproduce_attempt",
            action_type="reproduce",
            details={
                "success": False,
                "reason": "conditions_not_met",
                "confidence": confidence if should_reproduce else 0.0,
            },
        )
        return

    # Attempt reproduction
    try:
        # Create offspring
        offspring = agent.create_offspring()

        # Calculate reward based on success and conditions
        reward = _calculate_reproduction_reward(agent, offspring)
        agent.total_reward = agent.total_reward + reward

        # Log successful reproduction event
        if agent.environment.db is not None:
            agent.environment.db.logger.log_agent_action(
                agent_id=agent.agent_id,
                action_type="reproduce",
                step_number=agent.environment.time,
                details={
                    "success": True,
                    "offspring_id": offspring.agent_id,
                    "confidence": confidence,
                    "reward": reward,
                },
            )
        # Log lineage interaction edge parent -> offspring
        if getattr(agent, "logging_service", None) is not None:
            agent.logging_service.log_interaction_edge(
            source_type="agent",
            source_id=agent.agent_id,
            target_type="agent",
            target_id=offspring.agent_id,
            interaction_type="reproduce",
            action_type="reproduce",
            details={
                "success": True,
                "reward": reward,
                "confidence": confidence,
            },
        )

    except Exception as e:
        logger.error(f"Reproduction failed for agent {agent.agent_id}: {str(e)}")
        # Log failed reproduction event
        if agent.environment.db is not None:
            agent.environment.db.logger.log_agent_action(
                agent_id=agent.agent_id,
                action_type="reproduce",
                step_number=agent.environment.time,
                details={
                    "success": False,
                    "reason": "reproduction_error",
                    "error": str(e),
                },
            )
        if getattr(agent, "logging_service", None) is not None:
            agent.logging_service.log_interaction_edge(
            source_type="agent",
            source_id=agent.agent_id,
            target_type="agent",
            target_id=agent.agent_id,
            interaction_type="reproduce_failed",
            action_type="reproduce",
            details={
                "success": False,
                "reason": "reproduction_error",
                "error": str(e),
            },
        )


def _get_reproduce_state(agent: "BaseAgent") -> torch.Tensor:
    """Create state representation for reproduction decisions.

    This function constructs an 8-dimensional state vector that captures all
    relevant information for making reproduction decisions. The state includes
    normalized values for resource levels, health status, local population
    density, resource availability, global population balance, starvation risk,
    defensive status, and generation information.

    The state vector is designed to provide the Q-network with comprehensive
    information about both the agent's condition and the environment's suitability
    for reproduction.

    Args:
        agent: The agent for which to create the state representation

    Returns:
        torch.Tensor: 8-dimensional state vector with dtype float32 on the agent's device
    """
    # Type assertion to help linter understand config structure
    config = agent.config
    assert config is not None, "Agent config cannot be None"
    # Calculate local population density
    nearby_agents = [
        a
        for a in agent.environment._agent_objects.values()
        if a != agent
        and a.alive
        and np.sqrt(((np.array(a.position) - np.array(agent.position)) ** 2).sum())
        < config.ideal_density_radius
    ]

    local_density = len(nearby_agents) / max(
        1, len(agent.environment._agent_objects.values())
    )

    # Calculate resource availability in area
    nearby_resources = [
        r
        for r in agent.environment.resources
        if not r.is_depleted()
        and np.sqrt(((np.array(r.position) - np.array(agent.position)) ** 2).sum())
        < config.gathering_range
    ]

    resource_availability = len(nearby_resources) / max(
        1, len(agent.environment.resources)
    )

    state = torch.tensor(
        [
            agent.resource_level / config.min_reproduction_resources,  # Resource ratio
            agent.current_health / agent.starting_health,  # Health ratio
            local_density,  # Local population density
            resource_availability,  # Local resource availability
            len(agent.environment.agents)
            / config.max_population,  # Global population ratio
            agent.starvation_threshold / agent.max_starvation,  # Starvation risk
            float(agent.is_defending),  # Defensive status
            agent.generation / 10.0,  # Normalized generation number
        ],
        dtype=torch.float32,
        device=agent.device,
    )

    return state


def _check_reproduction_conditions(agent: "BaseAgent") -> bool:
    """Check if conditions are suitable for reproduction.

    This function validates whether the current conditions allow for successful
    reproduction. It checks multiple criteria including resource requirements,
    health status, and spatial constraints to ensure reproduction is viable
    and sustainable.

    The function implements a comprehensive validation system that considers:
    - Minimum resource requirements for reproduction and offspring survival
    - Agent health status relative to starting health
    - Local population density to prevent overcrowding
    - Spatial requirements to ensure adequate space for offspring

    Args:
        agent: The agent attempting to reproduce

    Returns:
        bool: True if all reproduction conditions are met, False otherwise
    """
    # Type assertion to help linter understand config structure
    config = agent.config
    assert config is not None, "Agent config cannot be None"

    # Check basic requirements
    if agent.resource_level < config.min_reproduction_resources:
        return False

    # Check offspring cost requirement
    offspring_cost = getattr(config, "offspring_cost", 3)  # Default to 3 if not set
    if agent.resource_level < offspring_cost + 2:
        return False

    # Check health status
    if agent.current_health < agent.starting_health * config.min_health_ratio:
        return False

    # Check local population density
    nearby_agents = [
        a
        for a in agent.environment.agents
        if a != agent
        and a.alive
        and np.sqrt(((np.array(a.position) - np.array(agent.position)) ** 2).sum())
        < config.min_space_required
    ]

    if (
        len(nearby_agents) / max(1, len(agent.environment.agents))
        > config.max_local_density
    ):
        return False

    return True


def _calculate_reproduction_reward(agent: "BaseAgent", offspring: "BaseAgent") -> float:
    """Calculate reward for successful reproduction attempt.

    This function computes the reward value for a successful reproduction event.
    The reward system encourages sustainable reproduction by providing bonuses
    for maintaining good health and resource levels after reproduction, as well
    as for maintaining optimal population balance.

    The reward calculation considers:
    - Base success reward for successful reproduction
    - Bonus for maintaining adequate resources after reproduction
    - Bonus for maintaining good population balance

    Args:
        agent: The parent agent that successfully reproduced
        offspring: The newly created offspring agent

    Returns:
        float: Calculated reward value for the reproduction event
    """
    # Type assertion to help linter understand config structure
    config = agent.config
    assert config is not None, "Agent config cannot be None"

    reward = config.success_reward

    # Add bonus for maintaining good health/resources after reproduction
    if agent.resource_level > config.min_reproduction_resources:
        reward += config.offspring_survival_bonus

    # Add bonus for maintaining good population balance
    population_ratio = len(agent.environment.agents) / config.max_population
    if 0.4 <= population_ratio <= 0.8:
        reward += config.population_balance_bonus

    return reward


# Default configuration instance
DEFAULT_REPRODUCE_CONFIG = ReproduceConfig()

# Register the action at the end of the file after the function is defined
from farm.core.action import action_registry

action_registry.register("reproduce", 0.15, reproduce_action)
