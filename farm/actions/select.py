"""Action selection module for intelligent action prioritization.

This module provides a flexible framework for agents to make intelligent decisions
about which action to take during their turn, considering:
- Current state and environment conditions
- Action weights and probabilities
- State-based adjustments for different scenarios
- Exploration vs exploitation balance
- Learned preferences through Q-learning

The module uses a combination of predefined weights and learned preferences to
select optimal actions for different situations, with dynamic adjustments based
on agent state, environment conditions, and learned behavior patterns.

Classes:
    SelectConfig: Configuration for action selection behavior and parameters
    SelectQNetwork: Neural network for learning action selection decisions
    SelectModule: Main module for learning and executing action selection
"""

import logging
import random
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from farm.utils.config_utils import get_config_value

from farm.actions.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder
from farm.actions.config import SelectConfig, DEFAULT_SELECT_CONFIG
from farm.core.action import Action
from farm.actions.feature_engineering import FeatureEngineer
from farm.actions.algorithms.base import AlgorithmRegistry, ActionAlgorithm

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)





class SelectQNetwork(BaseQNetwork):
    """Neural network for learning action selection decisions.

    This network learns to predict Q-values for different action choices
    based on the current state representation. It inherits from BaseQNetwork
    to provide standard Q-learning functionality.

    Args:
        input_dim: Dimension of the input state vector
        num_actions: Number of possible actions to choose from
        hidden_size: Size of hidden layers in the network
    """

    def __init__(self, input_dim: int, num_actions: int, hidden_size: int = 64, shared_encoder: Optional[SharedEncoder] = None) -> None:
        super().__init__(input_dim, num_actions, hidden_size, shared_encoder=shared_encoder)


class SelectModule(BaseDQNModule):
    """Module for learning and executing intelligent action selection.

    This module combines rule-based probability adjustments with learned
    Q-values to make intelligent action decisions. It considers agent state,
    environment conditions, and learned preferences to select optimal actions.

    The module uses epsilon-greedy exploration and can dynamically adjust
    action probabilities based on various factors like resource levels,
    health status, nearby entities, and population density.

    Args:
        num_actions: Number of possible actions to choose from
        config: Configuration object containing weights and thresholds
        device: PyTorch device for computation (CPU/GPU)

    Attributes:
        action_indices: Dictionary mapping action names to indices for fast lookup
    """

    def __init__(
        self,
        num_actions: int,
        config: SelectConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        super().__init__(
            input_dim=8,  # State dimensions for selection (matches actual state size)
            output_dim=num_actions,
            config=config,
            device=device,
        )
        # Pre-compute action indices for faster lookup
        self.action_indices = {}

        # Choose algorithm path: DQN (default) or traditional ML via registry
        self._algo_name = getattr(config, "algorithm_type", "dqn")
        self._ml_algo: Optional[ActionAlgorithm] = None
        self._feature_engineer: Optional[FeatureEngineer] = None

        if self._algo_name == "dqn":
            # Initialize Q-networks with shared encoder if provided
            self.q_network = SelectQNetwork(
                input_dim=8, num_actions=num_actions, hidden_size=config.dqn_hidden_size, shared_encoder=shared_encoder
            ).to(device)
            self.target_network = SelectQNetwork(
                input_dim=8, num_actions=num_actions, hidden_size=config.dqn_hidden_size, shared_encoder=shared_encoder
            ).to(device)
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Initialize traditional ML algorithm
            params = getattr(config, "algorithm_params", {}) or {}
            self._ml_algo = AlgorithmRegistry.create(self._algo_name, num_actions=num_actions, **params)
            self._feature_engineer = FeatureEngineer()

    def select_action(
        self, agent: "BaseAgent", actions: List[Action], state: torch.Tensor
    ) -> Action:
        """Select an action using both predefined weights and learned preferences.

        This method combines rule-based probability adjustments with learned
        Q-values to make intelligent action decisions. It considers:
        - Base action weights from configuration
        - State-based probability adjustments
        - Learned Q-values from the neural network
        - Exploration vs exploitation balance

        Args:
            agent: The agent making the action decision
            actions: List of available actions to choose from
            state: Current state representation as a tensor

        Returns:
            The selected action based on combined probabilities and Q-values

        Example:
            >>> module = SelectModule(5, config)
            >>> action = module.select_action(agent, available_actions, state)
        """
        # Cache action indices for this agent if not already done
        if agent.agent_id not in self.action_indices:
            self.action_indices[agent.agent_id] = {
                action.name: i for i, action in enumerate(actions)
            }

        # Get base probabilities from weights
        base_probs: List[float] = [action.weight for action in actions]

        # Adjust probabilities based on state - use faster implementation
        adjusted_probs = self._fast_adjust_probabilities(
            agent, base_probs, self.action_indices[agent.agent_id]
        )

        # If using ML algorithm, predict probabilities from engineered features
        if self._ml_algo is not None and self._feature_engineer is not None:
            ml_state = self._feature_engineer.extract_features(agent, agent.environment)
            ml_probs = self._ml_algo.predict_proba(ml_state)

            # Optional exploration bonus
            if getattr(self.config, "use_exploration_bonus", True):
                ml_probs = (ml_probs + (self.epsilon / max(1, len(ml_probs))))
                ml_probs = ml_probs / ml_probs.sum()

            # Blend ML probabilities with adjusted rule-based probabilities
            combined = 0.5 * np.array(adjusted_probs) + 0.5 * np.array(ml_probs)
            combined = combined / combined.sum()
            return random.choices(actions, weights=combined.tolist(), k=1)[0]
        else:
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
        """Optimized version of probability adjustment based on agent state.

        This method adjusts action probabilities based on various factors:
        - Resource levels and nearby resources
        - Health status and starvation risk
        - Nearby agents and social context
        - Population density and reproduction conditions

        Args:
            agent: The agent whose state is being considered
            base_probs: Base probability weights for each action
            action_indices: Dictionary mapping action names to their indices

        Returns:
            List of adjusted probabilities that sum to 1.0

        Note:
            This is an optimized version that uses environment spatial indexing
            for faster nearby entity detection and minimizes redundant calculations.
        """
        adjusted_probs = base_probs.copy()
        config = self.config

        # Get state information
        resource_level = agent.resource_level
        starvation_risk = agent.starvation_threshold / agent.max_starvation
        health_ratio = agent.current_health / agent.starting_health

        # Get nearby entities using environment's spatial indexing
        if agent.config is None:
            # Fallback to default values if config is not available
            gathering_range = 30
            social_range = 30
        else:
            gathering_range = agent.config.gathering_range
            social_range = agent.config.social_range

        nearby_resources = agent.environment.get_nearby_resources(
            agent.position, gathering_range
        )

        nearby_agents = agent.environment.get_nearby_agents(
            agent.position, social_range
        )

        # Get config values with fallbacks
        min_reproduction_resources = get_config_value(
            agent.config, "min_reproduction_resources", 8, (int, float)
        )
        max_population = get_config_value(
            agent.config, "max_population", 300, (int, float)
        )

        # Adjust move probability
        if "move" in action_indices and not nearby_resources:
            adjusted_probs[action_indices["move"]] *= getattr(
                config, "move_mult_no_resources", 1.5
            )

        # Adjust gather probability
        if (
            "gather" in action_indices
            and nearby_resources
            and resource_level < min_reproduction_resources
        ):
            adjusted_probs[action_indices["gather"]] *= getattr(
                config, "gather_mult_low_resources", 1.5
            )

        # Adjust share probability
        if "share" in action_indices:
            if resource_level > min_reproduction_resources and len(nearby_agents) > 0:
                adjusted_probs[action_indices["share"]] *= getattr(
                    config, "share_mult_wealthy", 1.3
                )
            else:
                adjusted_probs[action_indices["share"]] *= getattr(
                    config, "share_mult_poor", 0.5
                )

        # Adjust attack probability
        if "attack" in action_indices:
            if (
                starvation_risk > getattr(config, "attack_starvation_threshold", 0.5)
                and len(nearby_agents) > 0
                and resource_level > 2
            ):
                adjusted_probs[action_indices["attack"]] *= getattr(
                    config, "attack_mult_desperate", 1.4
                )
            else:
                adjusted_probs[action_indices["attack"]] *= getattr(
                    config, "attack_mult_stable", 0.6
                )

            if health_ratio < getattr(config, "attack_defense_threshold", 0.3):
                adjusted_probs[action_indices["attack"]] *= 0.5
            elif health_ratio > 0.8 and resource_level > min_reproduction_resources:
                adjusted_probs[action_indices["attack"]] *= 1.5

        # Adjust reproduce probability
        if "reproduce" in action_indices:
            if resource_level > min_reproduction_resources * 1.5 and health_ratio > 0.8:
                adjusted_probs[action_indices["reproduce"]] *= getattr(
                    config, "reproduce_mult_wealthy", 1.4
                )
            else:
                adjusted_probs[action_indices["reproduce"]] *= getattr(
                    config, "reproduce_mult_poor", 0.3
                )

            population_ratio = len(agent.environment.agents) / max_population
            if population_ratio > getattr(config, "reproduce_resource_threshold", 0.7):
                adjusted_probs[action_indices["reproduce"]] *= 0.5

        # Normalize probabilities
        total = sum(adjusted_probs)
        return [p / total for p in adjusted_probs]

    def _combine_probs_and_qvalues(
        self, probs: List[float], q_values: np.ndarray
    ) -> List[float]:
        """Combine adjusted probabilities with Q-values using weighted average.

        This method combines rule-based probability adjustments with learned
        Q-values to create a final action selection probability distribution.
        The combination uses a weighted average favoring rule-based adjustments
        (70%) over learned Q-values (30%) to maintain interpretable behavior
        while incorporating learned preferences.

        Args:
            probs: List of rule-based adjusted probabilities
            q_values: Array of Q-values from the neural network

        Returns:
            List of combined probabilities that sum to 1.0

        Note:
            Q-values are normalized to [0,1] range before combination to
            ensure they're on the same scale as probabilities.
        """
        # Normalize Q-values to [0,1] range
        q_normalized = (q_values - q_values.min()) / (
            q_values.max() - q_values.min() + 1e-8
        )

        # Combine using weighted average
        combined = 0.7 * np.array(probs) + 0.3 * q_normalized

        # Normalize
        return combined / combined.sum()


def create_selection_state(agent: "BaseAgent") -> torch.Tensor:
    """Create state representation for action selection decisions.

    This function creates a normalized state vector that captures the key
    factors influencing action selection decisions. The state includes:
    - Resource and health ratios
    - Environmental conditions (nearby entities, population density)
    - Agent status indicators (defending, alive, time progression)

    Args:
        agent: The agent for whom to create the state representation

    Returns:
        A tensor of shape (8,) containing normalized state values

    Note:
        The state vector is designed to be compatible with the SelectQNetwork
        input dimension and provides a comprehensive view of the agent's
        current situation for making intelligent action decisions.

    Example:
        >>> state = create_selection_state(agent)
        >>> print(state.shape)  # torch.Size([8])
    """
    # Calculate normalized values with fallbacks
    min_reproduction_resources = (
        getattr(agent.config, "min_reproduction_resources", 8) if agent.config else 8
    )
    gathering_range = (
        getattr(agent.config, "gathering_range", 30) if agent.config else 30
    )
    social_range = getattr(agent.config, "social_range", 30) if agent.config else 30

    max_resources = min_reproduction_resources * 3
    resource_ratio = agent.resource_level / max_resources
    health_ratio = agent.current_health / agent.starting_health
    starvation_ratio = agent.starvation_threshold / agent.max_starvation

    # Use environment's spatial indexing for faster nearby entity detection
    nearby_resources = len(
        agent.environment.get_nearby_resources(agent.position, gathering_range)
    )

    nearby_agents = len(
        agent.environment.get_nearby_agents(agent.position, social_range)
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
