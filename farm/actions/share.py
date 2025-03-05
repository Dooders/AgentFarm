"""Resource sharing module using Deep Q-Learning for intelligent cooperation.

This module implements a learning-based approach for agents to develop optimal 
sharing strategies in a multi-agent environment. It considers factors like
resource levels, agent relationships, and environmental conditions.

Key Components:
    - ShareQNetwork: Neural network for share decision making
    - ShareModule: Main class handling sharing logic and learning
    - Experience Replay: Stores sharing interactions for learning
    - Reward System: Encourages beneficial sharing behavior
"""

import logging
import random
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ShareConfig(BaseDQNConfig):
    """Configuration specific to sharing behavior."""

    share_range: float = 30.0  # Maximum distance for sharing
    min_share_amount: int = 1  # Minimum resources to share
    share_success_reward: float = 0.3  # Reward for successful sharing
    share_failure_penalty: float = -0.1  # Penalty for failed sharing attempt
    altruism_bonus: float = 0.2  # Extra reward for sharing when recipient is low
    cooperation_memory: int = 100  # Number of interactions to remember
    max_resources: int = 30  # Maximum resources an agent can have (for normalization)


class ShareActionSpace:
    """Possible sharing actions."""

    NO_SHARE: int = 0
    SHARE_LOW: int = 1  # Share minimum amount
    SHARE_MEDIUM: int = 2  # Share moderate amount
    SHARE_HIGH: int = 3  # Share larger amount


DEFAULT_SHARE_CONFIG = ShareConfig()


class ShareQNetwork(BaseQNetwork):
    """Share-specific Q-network architecture."""

    def __init__(self, input_dim: int = 6, hidden_size: int = 64) -> None:
        # Input features: [agent_resources, nearby_agents, avg_neighbor_resources,
        #                 min_neighbor_resources, max_neighbor_resources, cooperation_score]
        super().__init__(
            input_dim=input_dim,
            output_dim=4,  # NO_SHARE, SHARE_LOW, SHARE_MEDIUM, SHARE_HIGH
            hidden_size=hidden_size,
        )


class ShareModule(BaseDQNModule):
    """Module for learning and executing sharing behavior."""

    def __init__(
        self,
        config: ShareConfig = DEFAULT_SHARE_CONFIG,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        # Initialize parent class with share-specific network
        super().__init__(
            input_dim=6,  # State dimensions for sharing
            output_dim=4,  # Number of sharing actions
            config=config,
            device=device,
        )
        self.cooperation_history = {}  # Track sharing interactions
        self._setup_action_space()

        # Initialize Q-network specific to sharing
        self.q_network = ShareQNetwork(
            input_dim=6, hidden_size=config.dqn_hidden_size
        ).to(device)
        self.target_network = ShareQNetwork(
            input_dim=6, hidden_size=config.dqn_hidden_size
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(
        self, state: torch.Tensor, epsilon: Optional[float] = None
    ) -> int:
        """Select sharing action using epsilon-greedy strategy.

        Args:
            state: Current state tensor
            epsilon: Optional override for exploration rate

        Returns:
            int: Selected action index
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randint(0, len(self.action_space) - 1)

        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def _setup_action_space(self) -> None:
        """Define sharing amounts for each action."""
        self.action_space = {
            ShareActionSpace.NO_SHARE: 0,
            ShareActionSpace.SHARE_LOW: 1,
            ShareActionSpace.SHARE_MEDIUM: 2,
            ShareActionSpace.SHARE_HIGH: 3,
        }

    def get_share_decision(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[int, Optional["BaseAgent"], int]:
        """Determine sharing action and target agent.

        Returns:
            Tuple containing:
            - Selected action
            - Target agent (or None if NO_SHARE)
            - Amount to share
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        action = self.select_action(state)

        if action == ShareActionSpace.NO_SHARE:
            return action, None, 0

        # Find potential sharing targets
        nearby_agents = self._get_nearby_agents(agent)
        if not nearby_agents:
            return ShareActionSpace.NO_SHARE, None, 0

        # Select target based on need and cooperation history
        target = self._select_target(agent, nearby_agents)
        share_amount = self._calculate_share_amount(agent, action)

        return action, target, share_amount

    def _get_nearby_agents(self, agent: "BaseAgent") -> List["BaseAgent"]:
        """Find agents within sharing range."""
        return agent.environment.get_nearby_agents(
            agent.position, 
            self.config.share_range
        )

    def _select_target(
        self, agent: "BaseAgent", nearby_agents: List["BaseAgent"]
    ) -> "BaseAgent":
        """Select target agent based on need and cooperation history."""
        # Calculate selection weights based on need and past cooperation
        weights = []
        for target in nearby_agents:
            weight = 1.0
            # Increase weight for agents with low resources
            if target.resource_level < target.config.starvation_threshold:
                weight *= 2.0
            # Consider past cooperation
            coop_score = self._get_cooperation_score(target.agent_id)
            weight *= 1.0 + coop_score
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights) / sum(weights)
        return np.random.choice(nearby_agents, p=weights)

    def _calculate_share_amount(self, agent: "BaseAgent", action: int) -> int:
        """Calculate amount to share based on action and available resources."""
        if action == ShareActionSpace.NO_SHARE:
            return 0

        available = max(0, agent.resource_level - self.config.min_share_amount)
        share_amounts = {
            ShareActionSpace.SHARE_LOW: min(1, available),
            ShareActionSpace.SHARE_MEDIUM: min(2, available),
            ShareActionSpace.SHARE_HIGH: min(3, available),
        }
        return share_amounts.get(action, 0)

    def _get_cooperation_score(self, agent_id: int) -> float:
        """Get cooperation score for an agent based on history."""
        if agent_id not in self.cooperation_history:
            return 0.0
        return sum(
            self.cooperation_history[agent_id][-self.config.cooperation_memory :]
        ) / len(self.cooperation_history[agent_id][-self.config.cooperation_memory :])

    def update_cooperation(self, agent_id: int, cooperative: bool) -> None:
        """Update cooperation history for an agent."""
        if agent_id not in self.cooperation_history:
            self.cooperation_history[agent_id] = []
        self.cooperation_history[agent_id].append(1.0 if cooperative else -1.0)


def share_action(agent: "BaseAgent") -> None:
    """Execute sharing action using learned policy."""
    # Get state information
    state = _get_share_state(agent)
    initial_resources = agent.resource_level

    # Get sharing decision
    action, target, share_amount = agent.share_module.get_share_decision(agent, state)

    if not target or share_amount <= 0 or agent.resource_level < share_amount:
        # Log failed share action
        if agent.environment.db is not None:
            agent.environment.db.logger.log_agent_action(
                step_number=agent.environment.time,
                agent_id=agent.agent_id,
                action_type="share",
                position_before=agent.position,
                position_after=agent.position,
                resources_before=initial_resources,
                resources_after=initial_resources,
                reward=DEFAULT_SHARE_CONFIG.share_failure_penalty,
                details={
                    "success": False,
                    "reason": "invalid_share_conditions",
                    "attempted_amount": share_amount,
                }
            )
        return

    # Execute sharing
    target_initial_resources = target.resource_level
    agent.resource_level -= share_amount
    target.resource_level += share_amount
    
    # Update environment's resources_shared counter
    agent.environment.resources_shared += share_amount
    agent.environment.resources_shared_this_step += share_amount

    # Calculate reward
    reward = _calculate_share_reward(agent, target, share_amount)
    agent.total_reward += reward

    # Update cooperation history
    agent.share_module.update_cooperation(target.agent_id, True)

    # Log successful share action
    if agent.environment.db is not None:
        agent.environment.db.logger.log_agent_action(
            step_number=agent.environment.time,
            agent_id=agent.agent_id,
            action_type="share",
            action_target_id=target.agent_id,
            position_before=agent.position,
            position_after=agent.position,
            resources_before=initial_resources,
            resources_after=agent.resource_level,
            reward=reward,
            details={
                "success": True,
                "amount_shared": share_amount,
                "target_resources_before": target_initial_resources,
                "target_resources_after": target.resource_level,
                "target_was_starving": target_initial_resources
                < target.config.starvation_threshold,
            }
        )


def _get_share_state(agent: "BaseAgent") -> List[float]:
    """Create state representation for sharing decisions."""
    nearby_agents = agent.environment.get_nearby_agents(
        agent.position,
        DEFAULT_SHARE_CONFIG.share_range
    )

    neighbor_resources = (
        [a.resource_level for a in nearby_agents] if nearby_agents else [0]
    )

    # Use ShareConfig's max_resources for normalization
    max_resources = DEFAULT_SHARE_CONFIG.max_resources

    return [
        agent.resource_level / max_resources,  # Normalized agent resources
        len(nearby_agents) / len(agent.environment.agents),  # Normalized nearby agents
        np.mean(neighbor_resources) / max_resources,  # Avg neighbor resources
        min(neighbor_resources) / max_resources,  # Min neighbor resources
        max(neighbor_resources) / max_resources,  # Max neighbor resources
        agent.share_module._get_cooperation_score(agent.agent_id),  # Cooperation score
    ]


def _calculate_share_reward(
    agent: "BaseAgent", target: "BaseAgent", amount: int
) -> float:
    """Calculate reward for sharing action."""
    reward = DEFAULT_SHARE_CONFIG.share_success_reward

    # Add altruism bonus if target was in need
    if target.resource_level < target.config.starvation_threshold:
        reward += DEFAULT_SHARE_CONFIG.altruism_bonus

    # Scale reward based on amount shared
    reward *= amount / DEFAULT_SHARE_CONFIG.min_share_amount

    return reward
