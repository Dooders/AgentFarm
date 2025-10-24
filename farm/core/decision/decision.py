"""Decision module for agent action selection using configurable algorithms.

This module provides a DecisionModule class that determines the action to take given
an agent's state/observation. It supports configurable algorithms with Tianshou
as the default RL library, and ensures each agent has its own model/policy.

Key Features:
    - Configurable decision algorithms (PPO, SAC, DQN, A2C, DDPG via Tianshou)
    - Per-agent model/policy isolation
    - Integration with environment action spaces
    - Tianshou integration for reinforcement learning
    - Experience replay and training mechanisms
    - Save/load functionality for model persistence
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from farm.core.decision.config import DecisionConfig
from farm.utils.logging import get_logger

logger = get_logger(__name__)

# Check Tianshou availability
try:
    import tianshou

    TIANSHOU_AVAILABLE = True
except ImportError:
    TIANSHOU_AVAILABLE = False
    logger.warning("tianshou_unavailable", message="Using fallback algorithms")

# Initialize algorithm registry - populated conditionally based on Tianshou availability
_ALGORITHM_REGISTRY = {}

# Import Tianshou wrappers if available and populate registry
if TIANSHOU_AVAILABLE:
    try:
        from farm.core.decision.algorithms.tianshou import (
            A2CWrapper,
            DDPGWrapper,
            DQNWrapper,
            PPOWrapper,
            SACWrapper,
        )

        # Populate registry with available wrappers
        _ALGORITHM_REGISTRY.update(
            {
                "a2c": A2CWrapper,
                "ddpg": DDPGWrapper,
                "dqn": DQNWrapper,
                "ppo": PPOWrapper,
                "sac": SACWrapper,
            }
        )

    except ImportError:
        logger.warning(
            "tianshou_wrappers_unavailable", message="Using fallback algorithms"
        )
        TIANSHOU_AVAILABLE = False

if TYPE_CHECKING:
    from farm.core.agent import AgentCore


class DecisionModule:
    """Configurable decision module for agent action selection.

    This class encapsulates the decision-making logic for agents, supporting
    various algorithms with Tianshou as the default RL library.
    Each agent instance gets its own model/policy for personalized learning.

    Attributes:
        agent_id (str): Unique identifier of the associated agent
        config (DecisionConfig): Configuration object with algorithm settings
        algorithm: The underlying decision algorithm (Tianshou PPO, SAC, etc.)
        num_actions (int): Number of possible actions
        state_dim (int): Dimension of state/observation space
        _is_trained (bool): Whether the model has been trained
    """

    def __init__(
        self,
        agent: "AgentCore",
        action_space: Any,
        observation_space: Any,
        config: DecisionConfig = DecisionConfig(),
    ):
        """Initialize the DecisionModule.

        Args:
            agent: The AgentCore instance this module serves
            action_space: The action space for the agent (required)
            observation_space: The observation space for the agent (required)
            config: Configuration object with algorithm parameters
        """
        self.agent_id = agent.agent_id
        self.config = config
        self.agent = agent

        # Use provided action space
        self.action_space = action_space
        # Get number of actions directly from action space
        if hasattr(action_space, "n"):
            self.num_actions = int(action_space.n)
        else:
            # Fallback: count actions in Action enum
            from farm.core.action import ActionType

            self.num_actions = len(ActionType)

        # Use provided observation space
        self.observation_space = observation_space
        # Store the full observation shape for multi-dimensional support
        if hasattr(observation_space, "shape"):
            self.observation_shape = observation_space.shape
            # For CNN-based algorithms, pass the full shape, not flattened
            # For traditional ML algorithms, use the configured rl_state_dim
            if len(observation_space.shape) > 1:
                # Multi-dimensional observation - keep full shape for CNNs
                # but also provide flattened size for compatibility
                self.state_dim = int(np.prod(observation_space.shape))
            else:
                # 1D observation - use the dimension as-is
                if config.rl_state_dim > 0:
                    self.state_dim = config.rl_state_dim
                else:
                    self.state_dim = int(observation_space.shape[0])
        else:
            self.observation_shape = (config.rl_state_dim,)
            self.state_dim = config.rl_state_dim

        # Initialize algorithm
        self.algorithm: Any = None
        self._is_trained = False

        # Initialize the decision algorithm
        self._initialize_algorithm()

        logger.info(
            f"Initialized DecisionModule for agent {self.agent_id} with {config.algorithm_type}"
        )

    def _initialize_algorithm(self):
        """Initialize the decision algorithm based on configuration."""
        algorithm_type = self.config.algorithm_type

        if algorithm_type == "ppo" and TIANSHOU_AVAILABLE:
            self._initialize_tianshou_ppo()
        elif algorithm_type == "sac" and TIANSHOU_AVAILABLE:
            self._initialize_tianshou_sac()
        elif algorithm_type == "dqn" and TIANSHOU_AVAILABLE:
            self._initialize_tianshou_dqn()
        elif algorithm_type == "a2c" and TIANSHOU_AVAILABLE:
            self._initialize_tianshou_a2c()
        elif algorithm_type == "ddpg" and TIANSHOU_AVAILABLE:
            self._initialize_tianshou_ddpg()
        elif algorithm_type == "fallback":
            # Explicit fallback algorithm
            self._initialize_fallback()
        else:
            logger.warning(
                "algorithm_unavailable",
                algorithm_type=algorithm_type,
                agent_id=self.agent_id,
                message="Using fallback",
            )
            self._initialize_fallback()

    def _initialize_tianshou_ppo(self):
        """Initialize PPO using Tianshou."""
        if "ppo" not in _ALGORITHM_REGISTRY:
            logger.warning(
                "algorithm_not_available", algorithm="ppo", agent_id=self.agent_id
            )
            self._initialize_fallback()
            return

        try:
            # Configure PPO parameters
            algorithm_config = {
                "lr": self.config.learning_rate,
                "gamma": self.config.gamma,
                "eps_clip": 0.2,
                "max_grad_norm": 0.5,
                "vf_coef": 0.5,
                "ent_coef": 0.01,
                "gae_lambda": 0.95,
                "max_batchsize": self.config.rl_batch_size,
            }

            # Add any additional parameters from config
            algorithm_config.update(self.config.algorithm_params)

            self.algorithm = _ALGORITHM_REGISTRY["ppo"](
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                observation_shape=self.observation_shape,
                algorithm_config=algorithm_config,
                buffer_size=self.config.rl_buffer_size,
                batch_size=self.config.rl_batch_size,
                train_freq=self.config.rl_train_freq,
            )
            logger.info(
                "algorithm_initialized", algorithm="ppo", agent_id=self.agent_id
            )

        except Exception as e:
            logger.error(
                "algorithm_initialization_failed",
                algorithm="ppo",
                agent_id=self.agent_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self._initialize_fallback()

    def _initialize_tianshou_sac(self):
        """Initialize SAC using Tianshou."""
        if "sac" not in _ALGORITHM_REGISTRY:
            logger.warning(
                "algorithm_not_available", algorithm="sac", agent_id=self.agent_id
            )
            self._initialize_fallback()
            return

        try:
            # Configure SAC parameters
            algorithm_config = {
                "lr": self.config.learning_rate,
                "gamma": self.config.gamma,
                "tau": 0.005,
                "alpha": 0.2,
                "auto_alpha": True,
                "target_entropy": "auto",
            }

            # Add any additional parameters from config
            algorithm_config.update(self.config.algorithm_params)

            self.algorithm = _ALGORITHM_REGISTRY["sac"](
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                observation_shape=self.observation_shape,
                algorithm_config=algorithm_config,
                buffer_size=self.config.rl_buffer_size,
                batch_size=self.config.rl_batch_size,
                train_freq=self.config.rl_train_freq,
            )
            logger.info(
                "algorithm_initialized", algorithm="sac", agent_id=self.agent_id
            )

        except Exception as e:
            logger.error(
                "algorithm_initialization_failed",
                algorithm="sac",
                agent_id=self.agent_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self._initialize_fallback()

    def _initialize_tianshou_dqn(self):
        """Initialize DQN using Tianshou."""
        if "dqn" not in _ALGORITHM_REGISTRY:
            logger.warning(f"Tianshou DQN not available for agent {self.agent_id}")
            self._initialize_fallback()
            return

        try:
            # Configure DQN parameters
            algorithm_config = {
                "lr": self.config.learning_rate,
                "gamma": self.config.gamma,
                "n_step": 3,
                "target_update_freq": 500,
                "eps_test": self.config.epsilon_min,
                "eps_train": self.config.epsilon_start,
                "eps_train_final": self.config.epsilon_min,
            }

            # Add any additional parameters from config
            algorithm_config.update(self.config.algorithm_params)

            self.algorithm = _ALGORITHM_REGISTRY["dqn"](
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                observation_shape=self.observation_shape,
                algorithm_config=algorithm_config,
                buffer_size=self.config.rl_buffer_size,
                batch_size=self.config.rl_batch_size,
                train_freq=self.config.rl_train_freq,
            )
            logger.info(
                "algorithm_initialized", algorithm="dqn", agent_id=self.agent_id
            )

        except Exception as e:
            logger.error(
                "algorithm_initialization_failed",
                algorithm="dqn",
                agent_id=self.agent_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self._initialize_fallback()

    def _initialize_tianshou_a2c(self):
        """Initialize A2C using Tianshou."""
        if "a2c" not in _ALGORITHM_REGISTRY:
            logger.warning(f"Tianshou A2C not available for agent {self.agent_id}")
            self._initialize_fallback()
            return

        try:
            # Configure A2C parameters
            algorithm_config = {
                "lr": self.config.learning_rate,
                "gamma": self.config.gamma,
                "gae_lambda": 1.0,
                "max_grad_norm": 0.5,
                "vf_coef": 0.5,
                "ent_coef": 0.01,
                "max_batchsize": self.config.rl_batch_size,
            }

            # Add any additional parameters from config
            algorithm_config.update(self.config.algorithm_params)

            self.algorithm = _ALGORITHM_REGISTRY["a2c"](
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                observation_shape=self.observation_shape,
                algorithm_config=algorithm_config,
                buffer_size=self.config.rl_buffer_size,
                batch_size=self.config.rl_batch_size,
                train_freq=self.config.rl_train_freq,
            )
            logger.info(
                "algorithm_initialized", algorithm="a2c", agent_id=self.agent_id
            )

        except Exception as e:
            logger.error(
                "algorithm_initialization_failed",
                algorithm="a2c",
                agent_id=self.agent_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self._initialize_fallback()

    def _initialize_tianshou_ddpg(self):
        """Initialize DDPG using Tianshou."""
        if "ddpg" not in _ALGORITHM_REGISTRY:
            logger.warning(f"Tianshou DDPG not available for agent {self.agent_id}")
            self._initialize_fallback()
            return

        try:
            # Configure DDPG parameters
            algorithm_config = {
                "lr": self.config.learning_rate,
                "gamma": self.config.gamma,
                "tau": 0.005,
                "exploration_noise": 0.1,
            }

            # Add any additional parameters from config
            algorithm_config.update(self.config.algorithm_params)

            self.algorithm = _ALGORITHM_REGISTRY["ddpg"](
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                observation_shape=self.observation_shape,
                algorithm_config=algorithm_config,
                buffer_size=self.config.rl_buffer_size,
                batch_size=self.config.rl_batch_size,
                train_freq=self.config.rl_train_freq,
            )
            logger.info(
                "algorithm_initialized", algorithm="ddpg", agent_id=self.agent_id
            )

        except Exception as e:
            logger.error(
                "algorithm_initialization_failed",
                algorithm="ddpg",
                agent_id=self.agent_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self._initialize_fallback()

    def _initialize_fallback(self):
        """Initialize a fallback decision mechanism."""
        logger.info("using_fallback_algorithm", agent_id=self.agent_id)

        # Simple epsilon-greedy random action selection
        class FallbackAlgorithm:
            def __init__(self, num_actions, epsilon=0.1):
                self.num_actions = num_actions
                self.epsilon = epsilon
                # Store experiences for testing purposes
                self.experiences = []
                self._has_trained = False  # Track if we've already trained once

            def predict(self, observation, deterministic=False):
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.num_actions)
                else:
                    action = np.random.randint(self.num_actions)
                return action, None

            def select_action(self, observation):
                """Select action for compatibility with Tianshou-style algorithms."""
                return np.random.randint(self.num_actions)

            def select_action_with_mask(self, observation, action_mask):
                """Select action with mask for compatibility with Tianshou-style algorithms."""
                # Get valid actions from mask
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) == 0:
                    return 0  # Fallback to first action
                return np.random.choice(valid_actions)

            def predict_proba(self, observation):
                """Return uniform probabilities for fallback algorithm."""
                # Return uniform distribution over all actions
                return np.full(
                    (1, self.num_actions), 1.0 / self.num_actions, dtype=np.float32
                )

            def learn(self, total_timesteps=1):
                pass  # No learning in fallback

            def train(self, batch=None):
                """Train method for compatibility with Tianshou-style algorithms."""
                # No learning in fallback, but mark as trained
                self._has_trained = True

            def should_train(self):
                """Should train method for compatibility with Tianshou-style algorithms."""
                # Only train once to mark as trained, then return False for performance
                return not self._has_trained

            def store_experience(self, **kwargs):
                """Store experience method for compatibility with Tianshou-style algorithms."""
                # Store experiences for testing - this enables database logging tests
                self.experiences.append(kwargs)

        self.algorithm = FallbackAlgorithm(self.num_actions, self.config.epsilon_start)

    def decide_action(
        self,
        state: Union[torch.Tensor, np.ndarray],
        enabled_actions: Optional[List[int]] = None,
    ) -> int:
        """Decide which action to take given the current state.

        Args:
            state: Current state observation as tensor or numpy array
            enabled_actions: Optional list of enabled action indices. If provided,
                           only these actions will be considered valid. If None,
                           all actions in the full action space are considered valid.

        Returns:
            int: Index of the selected action within the `enabled_actions` list if provided (i.e., 0 to len(enabled_actions)-1), otherwise index within the full action space (0 to num_actions-1)
        """
        try:
            # Convert state to numpy for algorithm compatibility
            if isinstance(state, torch.Tensor):
                state_np = state.detach().cpu().numpy()
            else:
                state_np = np.array(state, dtype=np.float32)

            # Ensure correct shape for algorithm input
            # Handle multi-dimensional observations
            if state_np.ndim == 1:
                # 1D observation - add batch dimension for fully-connected networks
                state_np = state_np.reshape(1, -1)
            elif state_np.ndim == 3:
                # 3D observation (channels, height, width) - add batch dimension only
                # Don't reshape, keep the spatial structure for CNNs
                state_np = state_np[np.newaxis, ...]  # Add batch dimension
            elif state_np.ndim == 2:
                # 2D observation - add batch dimension
                state_np = state_np[np.newaxis, ...]  # Add batch dimension
            else:
                # For any other dimensionality, add batch dimension
                state_np = state_np[np.newaxis, ...]  # Add batch dimension

            # Create action mask for curriculum restrictions
            action_mask = self._create_action_mask(enabled_actions)

            # Get action from algorithm with masking support
            if self.algorithm is not None and hasattr(
                self.algorithm, "select_action_with_mask"
            ):
                # Use action masking support if available
                # Returned action is in FULL action space; convert to relative index if enabled_actions provided
                action_full = self.algorithm.select_action_with_mask(
                    state_np, action_mask
                )
                if enabled_actions is not None and len(enabled_actions) > 0:
                    # The algorithm should return an action that's in enabled_actions due to masking
                    # but we need to handle edge cases where this might not be true
                    if action_full in enabled_actions:
                        return enabled_actions.index(action_full)
                    else:
                        # This should rarely happen if masking is working correctly
                        # Log a warning and find the closest valid action
                        logger.warning(
                            f"Algorithm returned action {action_full} not in enabled_actions {enabled_actions}. "
                            f"This suggests an issue with action masking. Using fallback."
                        )
                        # Find the first valid action that's actually enabled
                        # This ensures we return a valid relative index even if the algorithm
                        # returned an invalid action due to masking implementation issues
                        for i, enabled_action in enumerate(enabled_actions):
                            if (
                                enabled_action < self.num_actions
                                and action_mask[enabled_action]
                            ):
                                return i
                        # Ultimate fallback: random valid action
                        # This should never be reached if enabled_actions is properly constructed
                        return int(np.random.randint(len(enabled_actions)))
                # No enabled_actions restriction: return full-space index
                action = action_full
            elif self.algorithm is not None and hasattr(
                self.algorithm, "select_action"
            ):
                # Fallback: get action and filter manually
                logger.debug(
                    f"Algorithm {type(self.algorithm).__name__} does not implement select_action_with_mask; using manual action filtering."
                )
                action = self.algorithm.select_action(state_np)
                action = self._filter_action_with_mask(action, enabled_actions)
            else:
                # Fallback algorithm - respect enabled actions
                action = self._filter_action_with_mask(
                    np.random.randint(self.num_actions), enabled_actions
                )

            # Ensure action is within valid range after masking
            action = int(action)
            if enabled_actions is not None and len(enabled_actions) > 0:
                # Return index within enabled_actions list
                if action < len(enabled_actions):
                    return action
                else:
                    # Fallback to random valid action
                    return np.random.randint(len(enabled_actions))
            else:
                # Full action space - ensure valid range
                if action < 0 or action >= self.num_actions:
                    logger.warning(
                        f"Invalid action {action} for agent {self.agent_id}, using random"
                    )
                    return np.random.randint(self.num_actions)
                return action

        except Exception as e:
            logger.error(f"Error in decide_action for agent {self.agent_id}: {e}")
            # Fallback to random action (respect enabled_actions if provided)
            if enabled_actions is not None and len(enabled_actions) > 0:
                return np.random.randint(len(enabled_actions))
            else:
                return np.random.randint(self.num_actions)

    def _create_action_mask(
        self, enabled_actions: Optional[List[int]] = None
    ) -> np.ndarray:
        """Create a boolean mask for valid actions based on curriculum restrictions.

        Args:
            enabled_actions: Optional list of enabled action indices

        Returns:
            np.ndarray: Boolean mask where True indicates valid actions
        """
        if enabled_actions is None or len(enabled_actions) == 0:
            # All actions are valid
            return np.ones(self.num_actions, dtype=bool)

        # Create mask for full action space
        mask = np.zeros(self.num_actions, dtype=bool)
        mask[enabled_actions] = True
        return mask

    def _filter_action_with_mask(
        self, action: int, enabled_actions: Optional[List[int]] = None
    ) -> int:
        """Filter an action through curriculum restrictions.

        Args:
            action: Original action index from algorithm
            enabled_actions: Optional list of enabled action indices

        Returns:
            int: Valid action index, either original or remapped to enabled set
        """
        if enabled_actions is None or len(enabled_actions) == 0:
            # No restrictions, return original action
            return action

        # Check if the selected action is enabled
        if action in enabled_actions:
            # Action is valid, return its index within the enabled_actions list
            return enabled_actions.index(action)
        else:
            # Action not enabled, select random enabled action
            logger.debug(
                f"Action {action} not in enabled actions {enabled_actions} for agent {self.agent_id}, "
                "selecting random enabled action"
            )
            selected_action = np.random.choice(enabled_actions)
            return enabled_actions.index(selected_action)

    def _convert_to_full_action_space(
        self, action_index, enabled_actions: Optional[List[int]] = None
    ) -> int:
        """Convert an action index from enabled_actions space back to full action space.

        Args:
            action_index: Action index within enabled_actions list, or Action object
            enabled_actions: Optional list of enabled action indices

        Returns:
            int: Action index in full action space
        """
        # Handle Action objects by converting to index first
        if hasattr(action_index, "name"):
            # This is an Action object, convert to full action space index and return directly
            from farm.core.action import action_name_to_index

            return action_name_to_index(action_index.name)

        if enabled_actions is None or len(enabled_actions) == 0:
            # No curriculum restrictions, action_index is already in full space
            return action_index

        # Validate action_index is within enabled_actions bounds
        if action_index < 0 or action_index >= len(enabled_actions):
            logger.warning(
                f"Invalid action index {action_index} for enabled actions {enabled_actions}, "
                f"using first enabled action for agent {self.agent_id}"
            )
            return enabled_actions[0] if enabled_actions else 0

        # Convert to full action space
        return enabled_actions[action_index]

    def _has_database_logger(self) -> bool:
        """Check if the agent has a database logger available."""
        return (
            hasattr(self.agent, "environment")
            and self.agent.environment
            and hasattr(self.agent.environment, "db")
            and self.agent.environment.db
            and hasattr(self.agent.environment.db, "logger")
        )

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        enabled_actions: Optional[List[int]] = None,
    ):
        logger.debug(f"Decision module update called for agent {self.agent_id}: action={action}, reward={reward}")
        """Update the decision module with experience, respecting curriculum restrictions.

        Args:
            state: Current state
            action: Action taken (index within enabled_actions if curriculum active)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            enabled_actions: Optional list of enabled action indices at time of action
        """
        try:
            # Convert action index back to full action space if curriculum is active
            full_action_index = self._convert_to_full_action_space(
                action, enabled_actions
            )

            # For Tianshou algorithms, store experience and train
            if (
                self.algorithm is not None
                and hasattr(self.algorithm, "store_experience")
                and callable(getattr(self.algorithm, "store_experience", None))
            ):
                # Store experience in Tianshou buffer with full action space index
                self.algorithm.store_experience(
                    state=state,
                    action=full_action_index,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                # Log learning experience to database if available
                if self._has_database_logger():
                    try:
                        step_number = None
                        if (
                            hasattr(self.agent, "services")
                            and self.agent.services
                            and hasattr(self.agent.services, "time_service")
                            and self.agent.services.time_service
                        ):
                            step_number = self.agent.services.time_service.current_time()

                        action_taken_mapped = None
                        # check if full_action_index is not None and if it is, check if it is less than the length of the actions list
                        if (
                            hasattr(self.agent, "actions")
                            and full_action_index is not None
                            and isinstance(full_action_index, int)
                            and full_action_index < len(self.agent.actions)
                        ):
                            action_taken_mapped = self.agent.actions[
                                full_action_index
                            ].name

                        # Debug logging
                        logger.debug(
                            f"Learning experience logging attempt for agent {self.agent_id}: "
                            f"step_number={step_number}, action_taken_mapped={action_taken_mapped}, "
                            f"full_action_index={full_action_index}, reward={reward}"
                        )

                        if step_number is not None and action_taken_mapped is not None:
                            self.agent.environment.db.logger.log_learning_experience(
                                step_number=step_number,
                                agent_id=self.agent_id,
                                module_type=self.config.algorithm_type,
                                module_id=id(self.algorithm),
                                action_taken=full_action_index,
                                action_taken_mapped=action_taken_mapped,
                                reward=reward,
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to log learning experience for agent {self.agent_id}: {e}"
                        )

                # Train if it's time to train
                if (
                    hasattr(self.algorithm, "should_train")
                    and self.algorithm.should_train()
                ):
                    self.algorithm.train(batch=None)
                    self._is_trained = True

            # For SB3 algorithms (fallback), simulate learning process
            elif (
                self.algorithm is not None
                and hasattr(self.algorithm, "learn")
                and callable(getattr(self.algorithm, "learn", None))
            ):
                # Call learn for a small number of timesteps to update
                self.algorithm.learn(total_timesteps=1)
                self._is_trained = True

            # For any other algorithm (including fallback), mark as trained
            elif self.algorithm is not None:
                # Fallback: just mark as trained for any algorithm that doesn't have specific training methods
                self._is_trained = True

        except Exception as e:
            logger.error(
                f"Error updating DecisionModule for agent {self.agent_id}: {e}"
            )

    def get_action_probabilities(self, state: torch.Tensor) -> np.ndarray:
        """Get action probabilities for the given state.

        Args:
            state: Current state observation

        Returns:
            np.ndarray: Probability distribution over actions as a numpy array
        """
        try:
            # For Tianshou algorithms that support it, get action probabilities
            if (
                self.algorithm is not None
                and hasattr(self.algorithm, "predict_proba")
                and callable(getattr(self.algorithm, "predict_proba", None))
            ):
                if isinstance(state, torch.Tensor):
                    state_np = state.detach().cpu().numpy()
                else:
                    state_np = np.array(state, dtype=np.float32)

                if state_np.ndim == 1:
                    state_np = state_np.reshape(1, -1)

                # Get probabilities from Tianshou algorithm
                probs = self.algorithm.predict_proba(state_np)
                # Handle 2D output (batch_size, num_actions) - take first sample
                if (
                    isinstance(probs, np.ndarray)
                    and probs.ndim == 2
                    and probs.shape[0] == 1
                ):
                    probs = probs[0]
                return np.array(probs, dtype=np.float32)

            # For SB3 algorithms that support it, get action probabilities
            elif (
                self.algorithm is not None
                and hasattr(self.algorithm, "predict_proba")
                and callable(getattr(self.algorithm, "predict_proba", None))
            ):
                if isinstance(state, torch.Tensor):
                    state_np = state.detach().cpu().numpy()
                else:
                    state_np = np.array(state, dtype=np.float32)

                if state_np.ndim == 1:
                    state_np = state_np.reshape(1, -1)

                probs = self.algorithm.predict_proba(state_np)[0]
                return np.array(probs, dtype=np.float32)

            else:
                # Fallback: uniform probabilities
                return np.full(
                    self.num_actions, 1.0 / self.num_actions, dtype=np.float32
                )

        except Exception as e:
            logger.error(
                f"Error getting action probabilities for agent {self.agent_id}: {e}"
            )
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)

    def save_model(self, path: str):
        """Save the decision model to disk.

        Args:
            path: Path to save the model
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if (
                self.algorithm is not None
                and hasattr(self.algorithm, "get_model_state")
                and callable(getattr(self.algorithm, "get_model_state", None))
            ):
                # For Tianshou algorithms, get model state and save
                model_state = self.algorithm.get_model_state()
                import pickle

                with open(f"{path}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "agent_id": self.agent_id,
                            "config": self.config.model_dump(),
                            "is_trained": self._is_trained,
                            "algorithm_type": self.config.algorithm_type,
                            "model_state": model_state,
                        },
                        f,
                    )
            elif (
                self.algorithm is not None
                and hasattr(self.algorithm, "save")
                and callable(getattr(self.algorithm, "save", None))
            ):
                # For SB3 algorithms (fallback)
                self.algorithm.save(path)
            else:
                # For fallback algorithm, save basic info
                import pickle

                with open(f"{path}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "agent_id": self.agent_id,
                            "config": self.config.model_dump(),
                            "is_trained": self._is_trained,
                        },
                        f,
                    )

            logger.info(
                f"Saved DecisionModule model for agent {self.agent_id} to {path}"
            )

        except Exception as e:
            logger.error(f"Error saving model for agent {self.agent_id}: {e}")

    def load_model(self, path: str):
        """Load the decision model from disk.

        Args:
            path: Path to load the model from
        """
        try:
            # Try to load pickle file first (for Tianshou models)
            import pickle

            pickle_path = f"{path}.pkl"

            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    self._is_trained = data.get("is_trained", False)

                    # If this is a Tianshou model, load the state
                    if "model_state" in data and self.algorithm is not None:
                        if hasattr(self.algorithm, "load_model_state"):
                            self.algorithm.load_model_state(data["model_state"])
                            logger.info(
                                f"Loaded Tianshou model state for agent {self.agent_id}"
                            )

            elif (
                self.algorithm is not None
                and hasattr(self.algorithm, "load")
                and callable(getattr(self.algorithm, "load", None))
            ):
                # For SB3 models (fallback) - try to load directly
                try:
                    # This is a fallback for any algorithm that supports load
                    if hasattr(self.algorithm, "load"):
                        self.algorithm.load(path)
                except Exception as load_error:
                    logger.error(
                        f"Failed to load SB3 model for agent {self.agent_id}: {load_error}"
                    )
                    return
            else:
                # For fallback algorithm without saved state
                self._is_trained = False

            logger.info(
                f"Loaded DecisionModule model for agent {self.agent_id} from {path}"
            )

        except Exception as e:
            logger.error(f"Error loading model for agent {self.agent_id}: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dict containing model information
        """
        return {
            "agent_id": self.agent_id,
            "algorithm_type": self.config.algorithm_type,
            "num_actions": self.num_actions,
            "state_dim": self.state_dim,
            "is_trained": self._is_trained,
            "tianshou_available": TIANSHOU_AVAILABLE,
        }

    def reset(self):
        """Reset the decision module state."""
        self._is_trained = False
        # Reset algorithm if it has a reset method
        if (
            self.algorithm is not None
            and hasattr(self.algorithm, "reset")
            and callable(getattr(self.algorithm, "reset", None))
        ):
            self.algorithm.reset()
