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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import numpy as np
import torch

from farm.core.action import Action
from farm.core.decision.algorithms.base import ActionAlgorithm, AlgorithmRegistry
from farm.core.decision.algorithms.rl_base import RLAlgorithm
from farm.core.decision.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder
from farm.core.decision.config import DEFAULT_DECISION_CONFIG, DecisionConfig
from farm.core.decision.feature_engineering import FeatureEngineer
from farm.utils.config_utils import get_config_value

if TYPE_CHECKING:
    from farm.core.agent import BaseAgent

logger = logging.getLogger(__name__)


class DecisionQNetwork(BaseQNetwork):
    """Neural network for learning action selection decisions.

    This network learns to predict Q-values for different action choices
    based on the current state representation. It inherits from BaseQNetwork
    to provide standard Q-learning functionality.

    Args:
        input_dim: Dimension of the input state vector
        num_actions: Number of possible actions to choose from
        hidden_size: Size of hidden layers in the network
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_size: int = 64,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        super().__init__(
            input_dim, num_actions, hidden_size, shared_encoder=shared_encoder
        )


class DecisionModule(BaseDQNModule):
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
        config: DecisionConfig,
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

        # Choose algorithm path: DQN (default), RL algorithms, or traditional ML via registry
        self._algo_name = getattr(config, "algorithm_type", "dqn")
        self._ml_algo: Optional[ActionAlgorithm] = None
        self._rl_algo: Optional[RLAlgorithm] = None
        self._feature_engineer: Optional[FeatureEngineer] = None

        if self._algo_name == "dqn":
            # Initialize Q-networks with shared encoder if provided
            self.q_network = DecisionQNetwork(
                input_dim=8,
                num_actions=num_actions,
                hidden_size=config.dqn_hidden_size,
                shared_encoder=shared_encoder,
            ).to(device)
            self.target_network = DecisionQNetwork(
                input_dim=8,
                num_actions=num_actions,
                hidden_size=config.dqn_hidden_size,
                shared_encoder=shared_encoder,
            ).to(device)
            self.target_network.load_state_dict(self.q_network.state_dict())
        elif self._algo_name in ["ppo", "sac", "a2c", "td3"]:
            # Initialize RL algorithm from Stable Baselines
            rl_params = getattr(config, "algorithm_params", {}).copy()
            rl_params.update(
                {
                    "state_dim": getattr(config, "rl_state_dim", 8),
                    "buffer_size": getattr(config, "rl_buffer_size", 10000),
                    "batch_size": getattr(config, "rl_batch_size", 32),
                    "train_freq": getattr(config, "rl_train_freq", 4),
                }
            )
            self._rl_algo = cast(
                RLAlgorithm,
                AlgorithmRegistry.create(
                    self._algo_name, num_actions=num_actions, **rl_params
                ),
            )
        else:
            # Initialize traditional ML algorithm
            params = getattr(config, "algorithm_params", {}) or {}
            self._ml_algo = AlgorithmRegistry.create(
                self._algo_name, num_actions=num_actions, **params
            )
            self._feature_engineer = FeatureEngineer()

    def decide_action(
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
            >>> module = DecisionModule(5, config)
            >>> action = module.decide_action(agent, available_actions, state)
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

        # If using RL algorithm (PPO, SAC, A2C, TD3)
        if self._rl_algo is not None:
            # Convert torch tensor to numpy for RL algorithms
            state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state

            # Get action from RL algorithm
            action_idx = self._rl_algo.select_action(state_np)

            # Ensure action is within valid range
            if 0 <= action_idx < len(actions):
                return actions[action_idx]
            else:
                # Fallback to rule-based selection
                return random.choices(actions, weights=adjusted_probs, k=1)[0]

        # If using ML algorithm, predict probabilities from engineered features
        elif self._ml_algo is not None and self._feature_engineer is not None:
            ml_state = self._feature_engineer.extract_features(agent, agent.environment)
            ml_probs = self._ml_algo.predict_proba(ml_state)

            # Optional exploration bonus
            if getattr(self.config, "use_exploration_bonus", True):
                ml_probs = ml_probs + (self.epsilon / len(ml_probs))
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

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        **kwargs: Any
    ) -> None:
        """Store experience for RL algorithm training.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode ended
            **kwargs: Additional experience data
        """
        if self._rl_algo is not None:
            self._rl_algo.store_experience(
                state, action, reward, next_state, done, **kwargs
            )
            self._rl_algo.update_step_count()

            # Train if needed
            if self._rl_algo.should_train():
                batch = self._rl_algo.replay_buffer.sample(
                    min(self._rl_algo.batch_size, len(self._rl_algo.replay_buffer))
                )
                metrics = self._rl_algo.train_on_batch(batch)
                # Could log metrics here if needed

    def get_model_state(self) -> Dict[str, Any]:
        """Get the current model state for saving."""
        state_dict = {}

        if self._rl_algo is not None:
            state_dict["rl_model"] = self._rl_algo.get_model_state()
        elif hasattr(self, "q_network"):
            # For DQN, save network states
            state_dict["q_network"] = self.q_network.state_dict()
            if hasattr(self, "target_network"):
                state_dict["target_network"] = self.target_network.state_dict()

        return state_dict

    def load_model_state(self, state_dict: Dict[str, Any]) -> None:
        """Load a saved model state."""
        if "rl_model" in state_dict and self._rl_algo is not None:
            self._rl_algo.load_model_state(state_dict["rl_model"])
        elif "q_network" in state_dict and hasattr(self, "q_network"):
            self.q_network.load_state_dict(state_dict["q_network"])
            if "target_network" in state_dict and hasattr(self, "target_network"):
                self.target_network.load_state_dict(state_dict["target_network"])


def create_decision_state(agent: "BaseAgent") -> torch.Tensor:
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
        >>> state = create_decision_state(agent)
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
