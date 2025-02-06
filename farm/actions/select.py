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

from farm.core.action import Action
from farm.actions.base_dqn import BaseDQNConfig, BaseDQNModule, BaseQNetwork

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

    def select_action(
        self, agent: "BaseAgent", actions: List[Action], state: torch.Tensor
    ) -> Action:
        #! change to decide_action
        """Select an action using both predefined weights and learned preferences.

        Args:
            agent: Agent making the decision
            actions: List of available actions
            state: Current state tensor

        Returns:
            Selected Action object
        """
        # Get base probabilities from weights
        base_probs = [action.weight for action in actions]

        # Adjust probabilities based on state
        adjusted_probs = self._adjust_probabilities(agent, base_probs)

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

    def _adjust_probabilities(self, agent: "BaseAgent", base_probs: List[float]) -> List[float]:
        """Adjust action probabilities based on agent's current state."""
        adjusted_probs = base_probs.copy()
        config = self.config

        # Get state information
        resource_level = agent.resource_level
        starvation_risk = agent.starvation_threshold / agent.max_starvation
        health_ratio = agent.current_health / agent.starting_health

        # Helper function to safely get action index
        def get_action_index(action_name: str) -> int:
            try:
                return next(i for i, a in enumerate(agent.actions) if a.name == action_name)
            except StopIteration:
                return -1

        # Find nearby entities
        nearby_resources = [
            r for r in agent.environment.resources
            if not r.is_depleted() and np.sqrt(((np.array(r.position) - np.array(agent.position)) ** 2).sum()) < agent.config.gathering_range
        ]

        nearby_agents = [
            a for a in agent.environment.agents
            if a != agent and a.alive and np.sqrt(((np.array(a.position) - np.array(agent.position)) ** 2).sum()) < agent.config.social_range
        ]

        # Adjust probabilities only for actions that exist
        move_idx = get_action_index("move")
        if move_idx >= 0 and not nearby_resources:
            adjusted_probs[move_idx] *= config.move_mult_no_resources

        gather_idx = get_action_index("gather")
        if gather_idx >= 0 and nearby_resources and resource_level < agent.config.min_reproduction_resources:
            adjusted_probs[gather_idx] *= config.gather_mult_low_resources

        share_idx = get_action_index("share")
        if share_idx >= 0:
            if resource_level > agent.config.min_reproduction_resources and nearby_agents:
                adjusted_probs[share_idx] *= config.share_mult_wealthy
            else:
                adjusted_probs[share_idx] *= config.share_mult_poor

        attack_idx = get_action_index("attack")
        if attack_idx >= 0:
            if starvation_risk > config.attack_starvation_threshold and nearby_agents and resource_level > 2:
                adjusted_probs[attack_idx] *= config.attack_mult_desperate
            else:
                adjusted_probs[attack_idx] *= config.attack_mult_stable

            if health_ratio < config.attack_defense_threshold:
                adjusted_probs[attack_idx] *= 0.5
            elif health_ratio > 0.8 and resource_level > agent.config.min_reproduction_resources:
                adjusted_probs[attack_idx] *= 1.5

        reproduce_idx = get_action_index("reproduce")
        if reproduce_idx >= 0:
            if resource_level > agent.config.min_reproduction_resources * 1.5 and health_ratio > 0.8:
                adjusted_probs[reproduce_idx] *= config.reproduce_mult_wealthy
            else:
                adjusted_probs[reproduce_idx] *= config.reproduce_mult_poor

            population_ratio = len(agent.environment.agents) / agent.config.max_population
            if population_ratio > config.reproduce_resource_threshold:
                adjusted_probs[reproduce_idx] *= 0.5

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

    # Get nearby entities
    nearby_resources = len(
        [
            r
            for r in agent.environment.resources
            if not r.is_depleted()
            and np.sqrt(((np.array(r.position) - np.array(agent.position)) ** 2).sum())
            < agent.config.gathering_range
        ]
    )

    nearby_agents = len(
        [
            a
            for a in agent.environment.agents
            if a != agent
            and a.alive
            and np.sqrt(((np.array(a.position) - np.array(agent.position)) ** 2).sum())
            < agent.config.social_range
        ]
    )

    # Create state tensor - removed time/max_steps ratio since max_steps isn't available
    state = torch.tensor(
        [
            resource_ratio,
            health_ratio,
            starvation_ratio,
            nearby_resources / max(1, len(agent.environment.resources)),
            nearby_agents / max(1, len(agent.environment.agents)),
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
