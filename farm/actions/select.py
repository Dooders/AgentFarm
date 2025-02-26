"""Action selection module for intelligent action prioritization.

This module provides a flexible framework for agents to make intelligent decisions
about which action to take during their turn, considering:
- Current state and environment
- Action weights and probabilities
- State-based adjustments
- Exploration vs exploitation

The module uses a combination of predefined weights and learned preferences to
select optimal actions for different situations.
"""

import logging
import random
from typing import TYPE_CHECKING, List

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork
from farm.core.action import Action

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SelectConfig(BaseDQNConfig):
    """Configuration for action selection behavior."""

    # Base action weights
    move_weight: float = 0.3
    gather_weight: float = 0.3
    share_weight: float = 0.15
    attack_weight: float = 0.1
    reproduce_weight: float = 0.15

    # State-based multipliers
    move_mult_no_resources: float = 1.5
    gather_mult_low_resources: float = 1.5
    share_mult_wealthy: float = 1.3
    share_mult_poor: float = 0.5
    attack_mult_desperate: float = 1.4
    attack_mult_stable: float = 0.6
    reproduce_mult_wealthy: float = 1.4
    reproduce_mult_poor: float = 0.3

    # Thresholds
    attack_starvation_threshold: float = 0.5
    attack_defense_threshold: float = 0.3
    reproduce_resource_threshold: float = 0.7


class SelectQNetwork(BaseQNetwork):
    """Neural network for action selection decisions."""

    def __init__(self, input_dim: int, num_actions: int, hidden_size: int = 64) -> None:
        super().__init__(input_dim, num_actions, hidden_size)


class SelectModule(BaseDQNModule):
    """Module for learning and executing action selection."""

    def __init__(
        self,
        num_actions: int,
        config: SelectConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        super().__init__(
            input_dim=8,  # State dimensions for selection
            output_dim=num_actions,
            config=config,
            device=device,
        )
        # Pre-compute action indices for faster lookup
        self.action_indices = {}

    def select_action(
        self, agent: "BaseAgent", actions: List[Action], state: torch.Tensor
    ) -> Action:
        """Select an action using both predefined weights and learned preferences."""
        # Cache action indices for this agent if not already done
        if agent.agent_id not in self.action_indices:
            self.action_indices[agent.agent_id] = {
                action.name: i for i, action in enumerate(actions)
            }

        # Get base probabilities from weights
        base_probs = [action.weight for action in actions]

        # Adjust probabilities based on state - use faster implementation
        adjusted_probs = self._fast_adjust_probabilities(
            agent, base_probs, self.action_indices[agent.agent_id]
        )

        # Use epsilon-greedy for exploration
        if random.random() < self.epsilon:
            return random.choices(actions, weights=adjusted_probs, k=1)[0]

        # Get Q-values from network
        with torch.no_grad():
            q_values = self.q_network(state)

        # Combine Q-values with adjusted probabilities
        combined_probs = self._combine_probs_and_qvalues(
            adjusted_probs, q_values.cpu().numpy()
        )

        return random.choices(actions, weights=combined_probs, k=1)[0]

    def _fast_adjust_probabilities(
        self, agent: "BaseAgent", base_probs: List[float], action_indices: dict
    ) -> List[float]:
        """Optimized version of probability adjustment."""
        adjusted_probs = base_probs.copy()
        config = self.config

        # Get state information
        resource_level = agent.resource_level
        starvation_risk = agent.starvation_threshold / agent.max_starvation
        health_ratio = agent.current_health / agent.starting_health

        # Get nearby entities using environment's spatial indexing
        nearby_resources = agent.environment.get_nearby_resources(
            agent.position, agent.config.gathering_range
        )

        nearby_agents = agent.environment.get_nearby_agents(
            agent.position, agent.config.social_range
        )

        # Adjust move probability
        if "move" in action_indices and not nearby_resources:
            adjusted_probs[action_indices["move"]] *= config.move_mult_no_resources

        # Adjust gather probability
        if (
            "gather" in action_indices
            and nearby_resources
            and resource_level < agent.config.min_reproduction_resources
        ):
            adjusted_probs[action_indices["gather"]] *= config.gather_mult_low_resources

        # Adjust share probability
        if "share" in action_indices:
            if (
                resource_level > agent.config.min_reproduction_resources
                and nearby_agents
            ):
                adjusted_probs[action_indices["share"]] *= config.share_mult_wealthy
            else:
                adjusted_probs[action_indices["share"]] *= config.share_mult_poor

        # Adjust attack probability
        if "attack" in action_indices:
            if (
                starvation_risk > config.attack_starvation_threshold
                and nearby_agents
                and resource_level > 2
            ):
                adjusted_probs[action_indices["attack"]] *= config.attack_mult_desperate
            else:
                adjusted_probs[action_indices["attack"]] *= config.attack_mult_stable

            if health_ratio < config.attack_defense_threshold:
                adjusted_probs[action_indices["attack"]] *= 0.5
            elif (
                health_ratio > 0.8
                and resource_level > agent.config.min_reproduction_resources
            ):
                adjusted_probs[action_indices["attack"]] *= 1.5

        # Adjust reproduce probability
        if "reproduce" in action_indices:
            if (
                resource_level > agent.config.min_reproduction_resources * 1.5
                and health_ratio > 0.8
            ):
                adjusted_probs[
                    action_indices["reproduce"]
                ] *= config.reproduce_mult_wealthy
            else:
                adjusted_probs[
                    action_indices["reproduce"]
                ] *= config.reproduce_mult_poor

            population_ratio = (
                len(agent.environment.agents) / agent.config.max_population
            )
            if population_ratio > config.reproduce_resource_threshold:
                adjusted_probs[action_indices["reproduce"]] *= 0.5

        # Normalize probabilities
        total = sum(adjusted_probs)
        return [p / total for p in adjusted_probs]

    def _combine_probs_and_qvalues(
        self, probs: List[float], q_values: np.ndarray
    ) -> List[float]:
        """Combine adjusted probabilities with Q-values."""
        # Normalize Q-values to [0,1] range
        q_normalized = (q_values - q_values.min()) / (
            q_values.max() - q_values.min() + 1e-8
        )

        # Combine using weighted average
        combined = 0.7 * np.array(probs) + 0.3 * q_normalized

        # Normalize
        return combined / combined.sum()


def create_selection_state(agent: "BaseAgent") -> torch.Tensor:
    """Create state representation for action selection."""
    # Calculate normalized values
    max_resources = agent.config.min_reproduction_resources * 3
    resource_ratio = agent.resource_level / max_resources
    health_ratio = agent.current_health / agent.starting_health
    starvation_ratio = agent.starvation_threshold / agent.max_starvation

    # Use environment's spatial indexing for faster nearby entity detection
    nearby_resources = len(
        agent.environment.get_nearby_resources(
            agent.position, agent.config.gathering_range
        )
    )

    nearby_agents = len(
        agent.environment.get_nearby_agents(agent.position, agent.config.social_range)
    )

    # Normalize counts
    resource_density = nearby_resources / max(1, len(agent.environment.resources))
    agent_density = nearby_agents / max(1, len(agent.environment.agents))

    # Create state tensor
    state = torch.tensor(
        [
            resource_ratio,
            health_ratio,
            starvation_ratio,
            resource_density,
            agent_density,
            float(
                agent.environment.time > 0
            ),  # Simple binary indicator if not first step
            float(agent.is_defending),
            float(agent.alive),
        ],
        dtype=torch.float32,
        device=agent.device,
    )

    return state
