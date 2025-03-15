"""Reproduction learning module using Deep Q-Learning (DQN).

This module implements intelligent reproduction strategies using Deep Q-Learning,
allowing agents to learn optimal timing and conditions for reproduction.

Key Components:
    - ReproduceQNetwork: Neural network for reproduction decisions
    - ReproduceModule: Main class handling reproduction logic and learning
    - Experience Replay: Stores reproduction outcomes for learning
    - Reward System: Encourages successful population growth strategies
"""

import logging
import random
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReproduceConfig(BaseDQNConfig):
    """Configuration specific to reproduction behavior."""

    # Reward parameters
    reproduce_success_reward: float = 1.0
    reproduce_fail_penalty: float = -0.2
    offspring_survival_bonus: float = 0.5
    population_balance_bonus: float = 0.3

    # Reproduction thresholds
    min_health_ratio: float = 0.5
    min_resource_ratio: float = 0.6
    ideal_density_radius: float = 50.0

    # Population control
    max_local_density: float = 0.7
    min_space_required: float = 20.0


class ReproduceActionSpace:
    """Possible reproduction actions."""

    WAIT: int = 0  # Wait for better conditions
    REPRODUCE: int = 1  # Attempt reproduction


class ReproduceQNetwork(BaseQNetwork):
    """Reproduction-specific Q-network."""

    def __init__(self, input_dim: int = 8, hidden_size: int = 64) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=2,  # WAIT or REPRODUCE
            hidden_size=hidden_size,
        )


class ReproduceModule(BaseDQNModule):
    """Module for learning and executing reproduction strategies."""

    def __init__(
        self, config: ReproduceConfig = ReproduceConfig(), device: torch.device = DEVICE
    ) -> None:
        super().__init__(
            input_dim=8,  # State dimensions for reproduction
            output_dim=2,  # Number of reproduction actions
            config=config,
            device=device,
        )

        # Initialize reproduction-specific Q-network
        self.q_network = ReproduceQNetwork(
            input_dim=8, hidden_size=config.dqn_hidden_size
        ).to(device)

        self.target_network = ReproduceQNetwork(
            input_dim=8, hidden_size=config.dqn_hidden_size
        ).to(device)

        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(
        self, state: torch.Tensor, epsilon: Optional[float] = None
    ) -> int:
        """Select reproduction action using epsilon-greedy strategy.

        Args:
            state: Current state tensor
            epsilon: Optional override for exploration rate

        Returns:
            int: Selected action (WAIT or REPRODUCE)
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randint(0, 1)  # Random choice between WAIT and REPRODUCE

        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def get_reproduction_decision(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[bool, float]:
        """Determine whether to reproduce based on current state.

        Args:
            agent: Agent considering reproduction
            state: Current state tensor

        Returns:
            Tuple of (should_reproduce, confidence_score)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        # Get Q-values and select action
        action = self.select_action(state)

        if action == ReproduceActionSpace.WAIT:
            return False, 0.0

        # Calculate confidence score
        with torch.no_grad():
            q_values = self.q_network(state)
            confidence = torch.softmax(q_values, dim=0)[
                ReproduceActionSpace.REPRODUCE
            ].item()

        return True, confidence


def reproduce_action(agent: "BaseAgent") -> None:
    """Execute reproduction action using the reproduce module."""
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
            agent.environment.db.logger.log_reproduction_event(
                step_number=agent.environment.time,
                parent_id=agent.agent_id,
                offspring_id=None,
                success=False,
                parent_resources_before=initial_resources,
                parent_resources_after=initial_resources,
                offspring_initial_resources=None,
                failure_reason="conditions_not_met",
                parent_generation=agent.generation,
                offspring_generation=None,
                parent_position=agent.position,
            )
        return

    # Attempt reproduction
    try:
        # Create offspring
        offspring = agent.create_offspring()

        # Calculate reward based on success and conditions
        reward = _calculate_reproduction_reward(agent, offspring)
        agent.total_reward += reward

        # Log successful reproduction event
        if agent.environment.db is not None:
            agent.environment.db.logger.log_reproduction_event(
                step_number=agent.environment.time,
                parent_id=agent.agent_id,
                offspring_id=offspring.agent_id,
                success=True,
                parent_resources_before=initial_resources,
                parent_resources_after=agent.resource_level,
                offspring_initial_resources=offspring.resource_level,
                failure_reason=None,
                parent_generation=agent.generation,
                offspring_generation=offspring.generation,
                parent_position=agent.position,
            )

        # Log action (keep existing logging)
        if agent.environment.db is not None:
            agent.environment.db.logger.log_agent_action(
                step_number=agent.environment.time,
                agent_id=agent.agent_id,
                action_type="reproduce",
                position_before=agent.position,
                position_after=agent.position,
                resources_before=initial_resources,
                resources_after=agent.resource_level,
                reward=reward,
                details={
                    "success": True,
                    "offspring_id": offspring.agent_id,
                    "confidence": confidence,
                    "parent_resources_remaining": agent.resource_level,
                },
            )

    except Exception as e:
        logger.error(f"Reproduction failed for agent {agent.agent_id}: {str(e)}")
        # Log failed reproduction event
        if agent.environment.db is not None:
            agent.environment.db.logger.log_reproduction_event(
                step_number=agent.environment.time,
                parent_id=agent.agent_id,
                offspring_id=None,
                success=False,
                parent_resources_before=initial_resources,
                parent_resources_after=agent.resource_level,
                offspring_initial_resources=None,
                failure_reason=str(e),
                parent_generation=agent.generation,
                offspring_generation=None,
                parent_position=agent.position,
            )
            # Keep existing error logging
            agent.environment.db.logger.log_agent_action(
                step_number=agent.environment.time,
                agent_id=agent.agent_id,
                action_type="reproduce",
                position_before=agent.position,
                position_after=agent.position,
                resources_before=initial_resources,
                resources_after=agent.resource_level,
                reward=ReproduceConfig.reproduce_fail_penalty,
                details={
                    "success": False,
                    "reason": "reproduction_error",
                    "error": str(e),
                },
            )


def _get_reproduce_state(agent: "BaseAgent") -> torch.Tensor:
    """Create state representation for reproduction decisions."""
    # Calculate local population density
    nearby_agents = [
        a
        for a in agent.environment.agents.values()
        if a != agent
        and a.alive
        and np.sqrt(((np.array(a.position) - np.array(agent.position)) ** 2).sum())
        < ReproduceConfig.ideal_density_radius
    ]

    local_density = len(nearby_agents) / max(1, len(agent.environment.agents))

    # Calculate resource availability in area
    nearby_resources = [
        r
        for r in agent.environment.resources.values()
        if not r.is_depleted()
        and np.sqrt(((np.array(r.position) - np.array(agent.position)) ** 2).sum())
        < agent.config.gathering_range
    ]

    resource_availability = len(nearby_resources) / max(
        1, len(agent.environment.resources)
    )

    state = torch.tensor(
        [
            agent.resource_level
            / agent.config.min_reproduction_resources,  # Resource ratio
            agent.current_health / agent.starting_health,  # Health ratio
            local_density,  # Local population density
            resource_availability,  # Local resource availability
            len(agent.environment.agents)
            / agent.config.max_population,  # Global population ratio
            agent.starvation_threshold / agent.max_starvation,  # Starvation risk
            float(agent.is_defending),  # Defensive status
            agent.generation / 10.0,  # Normalized generation number
        ],
        dtype=torch.float32,
        device=agent.device,
    )

    return state


def _check_reproduction_conditions(agent: "BaseAgent") -> bool:
    """Check if conditions are suitable for reproduction."""
    # Check basic requirements
    if agent.resource_level < agent.config.min_reproduction_resources:
        return False

    if agent.resource_level < agent.config.offspring_cost + 2:
        return False

    # Check health status
    if agent.current_health < agent.starting_health * ReproduceConfig.min_health_ratio:
        return False

    # Check local population density
    nearby_agents = [
        a
        for a in agent.environment.agents.values()
        if a != agent
        and a.alive
        and np.sqrt(((np.array(a.position) - np.array(agent.position)) ** 2).sum())
        < ReproduceConfig.min_space_required
    ]

    if (
        len(nearby_agents) / max(1, len(agent.environment.agents))
        > ReproduceConfig.max_local_density
    ):
        return False

    return True


def _calculate_reproduction_reward(agent: "BaseAgent", offspring: "BaseAgent") -> float:
    """Calculate reward for reproduction attempt."""
    reward = ReproduceConfig.reproduce_success_reward

    # Add bonus for maintaining good health/resources after reproduction
    if agent.resource_level > agent.config.min_reproduction_resources:
        reward += ReproduceConfig.offspring_survival_bonus

    # Add bonus for maintaining good population balance
    population_ratio = len(agent.environment.agents) / agent.config.max_population
    if 0.4 <= population_ratio <= 0.8:
        reward += ReproduceConfig.population_balance_bonus

    return reward
