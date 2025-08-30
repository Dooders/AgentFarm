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

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from farm.core.decision.config import DecisionConfig

# Check Tianshou availability
try:
    import tianshou

    TIANSHOU_AVAILABLE = True
except ImportError:
    TIANSHOU_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Tianshou not available. Using fallback algorithms.")

# Import Tianshou wrappers if available
if TIANSHOU_AVAILABLE:
    try:
        from farm.core.decision.algorithms.tianshou import (
            A2CWrapper,
            DDPGWrapper,
            DQNWrapper,
            PPOWrapper,
            SACWrapper,
        )
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("Could not import Tianshou wrappers. Using fallback algorithms.")
        TIANSHOU_AVAILABLE = False

if TYPE_CHECKING:
    from farm.core.agent import BaseAgent

logger = logging.getLogger(__name__)


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
        agent: "BaseAgent",
        action_space: Any,
        observation_space: Any,
        config: DecisionConfig = DecisionConfig(),
    ):
        """Initialize the DecisionModule.

        Args:
            agent: The BaseAgent instance this module serves
            action_space: The action space for the agent (required)
            observation_space: The observation space for the agent (required)
            config: Configuration object with algorithm parameters
        """
        self.agent_id = agent.agent_id
        self.config = config
        self.agent = agent

        # Set state dimension first (needed for observation space creation)
        self.state_dim = config.rl_state_dim

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
            # For backward compatibility, set state_dim to total elements
            self.state_dim = int(np.prod(observation_space.shape))
        else:
            self.observation_shape = (self.state_dim,)
            self.state_dim = int(np.prod(self.observation_shape))

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
                f"Algorithm {algorithm_type} not available or not supported. Using fallback."
            )
            self._initialize_fallback()

    def _initialize_tianshou_ppo(self):
        """Initialize PPO using Tianshou."""
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

            self.algorithm = PPOWrapper(
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                algorithm_config=algorithm_config,
                buffer_size=self.config.rl_buffer_size,
                batch_size=self.config.rl_batch_size,
                train_freq=self.config.rl_train_freq,
            )
            logger.info(f"Initialized Tianshou PPO for agent {self.agent_id}")

        except Exception as e:
            logger.error(
                f"Failed to initialize Tianshou PPO for agent {self.agent_id}: {e}"
            )
            self._initialize_fallback()

    def _initialize_tianshou_sac(self):
        """Initialize SAC using Tianshou."""
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

            self.algorithm = SACWrapper(
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                algorithm_config=algorithm_config,
                buffer_size=self.config.rl_buffer_size,
                batch_size=self.config.rl_batch_size,
                train_freq=self.config.rl_train_freq,
            )
            logger.info(f"Initialized Tianshou SAC for agent {self.agent_id}")

        except Exception as e:
            logger.error(
                f"Failed to initialize Tianshou SAC for agent {self.agent_id}: {e}"
            )
            self._initialize_fallback()

    def _initialize_tianshou_dqn(self):
        """Initialize DQN using Tianshou."""
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

            self.algorithm = DQNWrapper(
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                algorithm_config=algorithm_config,
                buffer_size=self.config.rl_buffer_size,
                batch_size=self.config.rl_batch_size,
                train_freq=self.config.rl_train_freq,
            )
            logger.info(f"Initialized Tianshou DQN for agent {self.agent_id}")

        except Exception as e:
            logger.error(
                f"Failed to initialize Tianshou DQN for agent {self.agent_id}: {e}"
            )
            self._initialize_fallback()

    def _initialize_tianshou_a2c(self):
        """Initialize A2C using Tianshou."""
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

            self.algorithm = A2CWrapper(
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                algorithm_config=algorithm_config,
                buffer_size=self.config.rl_buffer_size,
                batch_size=self.config.rl_batch_size,
                train_freq=self.config.rl_train_freq,
            )
            logger.info(f"Initialized Tianshou A2C for agent {self.agent_id}")

        except Exception as e:
            logger.error(
                f"Failed to initialize Tianshou A2C for agent {self.agent_id}: {e}"
            )
            self._initialize_fallback()

    def _initialize_tianshou_ddpg(self):
        """Initialize DDPG using Tianshou."""
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

            self.algorithm = DDPGWrapper(
                num_actions=self.num_actions,
                state_dim=self.state_dim,
                algorithm_config=algorithm_config,
                buffer_size=self.config.rl_buffer_size,
                batch_size=self.config.rl_batch_size,
                train_freq=self.config.rl_train_freq,
            )
            logger.info(f"Initialized Tianshou DDPG for agent {self.agent_id}")

        except Exception as e:
            logger.error(
                f"Failed to initialize Tianshou DDPG for agent {self.agent_id}: {e}"
            )
            self._initialize_fallback()

    def _initialize_fallback(self):
        """Initialize a fallback decision mechanism."""
        logger.info(f"Using fallback decision mechanism for agent {self.agent_id}")

        # Simple epsilon-greedy random action selection
        class FallbackAlgorithm:
            def __init__(self, num_actions, epsilon=0.1):
                self.num_actions = num_actions
                self.epsilon = epsilon

            def predict(self, observation, deterministic=False):
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.num_actions)
                else:
                    action = np.random.randint(self.num_actions)
                return action, None

            def predict_proba(self, observation):
                """Return uniform probabilities for fallback algorithm."""
                # Return uniform distribution over all actions
                return np.full(
                    (1, self.num_actions), 1.0 / self.num_actions, dtype=np.float32
                )

            def learn(self, total_timesteps=1):
                pass  # No learning in fallback

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
            int: Selected action index (always within enabled_actions if provided)
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
                state_np = state_np.reshape(1, -1)
            elif state_np.ndim > 1:
                # For multi-dimensional inputs, add batch dimension
                if state_np.shape != self.observation_shape:
                    # If shape doesn't match expected, try to reshape
                    if np.prod(state_np.shape) == self.state_dim:
                        state_np = state_np.reshape(self.observation_shape)
                state_np = state_np[np.newaxis, ...]  # Add batch dimension

            # Get action from algorithm
            if self.algorithm is not None and hasattr(self.algorithm, "select_action"):
                # For Tianshou algorithms
                action = self.algorithm.select_action(state_np)
            else:
                action = np.random.randint(self.num_actions)

            # Ensure action is within valid range
            action = int(action)
            if action < 0 or action >= self.num_actions:
                logger.warning(
                    f"Invalid action {action} for agent {self.agent_id}, using random"
                )
                action = np.random.randint(self.num_actions)

            # Handle curriculum restrictions - if enabled_actions is provided,
            # return the index within the enabled_actions list, not the full action space index
            if enabled_actions is not None and len(enabled_actions) > 0:
                # Check if the selected action is in the enabled set
                if action not in enabled_actions:
                    logger.debug(
                        f"Action {action} not in enabled actions {enabled_actions} for agent {self.agent_id}, "
                        "selecting random enabled action"
                    )
                    # Select random action from enabled set
                    selected_action = np.random.choice(enabled_actions)
                else:
                    selected_action = action
                # Convert the action index to its position within the enabled_actions list
                action = enabled_actions.index(selected_action)

            return action

        except Exception as e:
            logger.error(f"Error in decide_action for agent {self.agent_id}: {e}")
            # Fallback to random action (respect enabled_actions if provided)
            if enabled_actions is not None and len(enabled_actions) > 0:
                # Return index within enabled_actions list
                random_action = np.random.choice(enabled_actions)
                return enabled_actions.index(random_action)
            else:
                return np.random.randint(self.num_actions)

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Update the decision module with experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        try:
            # For Tianshou algorithms, store experience and train
            if (
                self.algorithm is not None
                and hasattr(self.algorithm, "store_experience")
                and callable(getattr(self.algorithm, "store_experience", None))
            ):
                # Store experience in Tianshou buffer
                self.algorithm.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                # Train if it's time to train
                if (
                    hasattr(self.algorithm, "should_train")
                    and self.algorithm.should_train()
                ):
                    self.algorithm.train()
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


# Example usage and demonstration
"""
Example: Using DecisionModule with BaseAgent

```python
from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.core.decision.config import DecisionConfig

# Create environment and agent
env = Environment(width=100, height=100, resource_distribution={})
agent = BaseAgent(
    agent_id="test_agent",
    position=(50, 50),
    resource_level=10,
    environment=env
)

# DecisionModule is automatically created in BaseAgent.__init__()
# Access it via agent.decision_module

# Manual example of creating and using DecisionModule
from farm.core.decision.decision import DecisionModule

# Create config for Tianshou PPO
config = DecisionConfig(
    algorithm_type="ppo",  # Use PPO with Tianshou
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01
)

# Create DecisionModule
decision_module = DecisionModule(agent=agent, config=config)

# Create state tensor using agent's method
state = agent.create_decision_state()

# Get action
action_index = decision_module.decide_action(state)
print(f"Selected action index: {action_index}")

# Update with experience (after action execution)
reward = 1.0  # Some reward
next_state = agent.create_decision_state()  # State after action
done = False  # Episode not done
decision_module.update(state, action_index, reward, next_state, done)

# Save/load model
decision_module.save_model("agent_model")
decision_module.load_model("agent_model")

# Get model info
info = decision_module.get_model_info()
print(f"Model info: {info}")

# Available algorithm types:
# - "ppo": Proximal Policy Optimization (default)
# - "sac": Soft Actor-Critic
# - "dqn": Deep Q-Network
# - "a2c": Advantage Actor-Critic
# - "ddpg": Deep Deterministic Policy Gradient
# - "fallback": Simple epsilon-greedy random action selection
```
"""
