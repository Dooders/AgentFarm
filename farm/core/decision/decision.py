"""Decision module for agent action selection using configurable algorithms.

This module provides a DecisionModule class that determines the action to take given
an agent's state/observation. It supports configurable algorithms with DDQN (via
Stable Baselines3) as the default, and ensures each agent has its own model/policy.

Key Features:
    - Configurable decision algorithms (DDQN, PPO, custom ML algorithms)
    - Per-agent model/policy isolation
    - Integration with environment action spaces
    - SB3 integration for reinforcement learning
    - Experience replay and training mechanisms
    - Save/load functionality for model persistence
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch

from farm.core.decision.config import DecisionConfig

if TYPE_CHECKING:
    from farm.core.agent import BaseAgent
    from farm.core.environment import Environment

logger = logging.getLogger(__name__)

# Try to import SB3, with fallback handling
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.policies import BasePolicy

    SB3_AVAILABLE = True
except ImportError:
    logger.warning(
        "Stable Baselines3 not available. Falling back to basic DQN implementation."
    )
    SB3_AVAILABLE = False
    DQN = None


class SB3Wrapper:
    """Wrapper to adapt agent observations to SB3 format."""

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """Reset the wrapper (no-op for agent-based usage)."""
        return np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )

    def step(self, action):
        """Step function (no-op for agent-based usage)."""
        return (
            np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype),
            0.0,
            False,
            False,
            {},
        )


class DecisionModule:
    """Configurable decision module for agent action selection.

    This class encapsulates the decision-making logic for agents, supporting
    various algorithms with DDQN (via Stable Baselines3) as the default.
    Each agent instance gets its own model/policy for personalized learning.

    Attributes:
        agent_id (str): Unique identifier of the associated agent
        config (DecisionConfig): Configuration object with algorithm settings
        algorithm: The underlying decision algorithm (SB3 DQN, etc.)
        num_actions (int): Number of possible actions
        state_dim (int): Dimension of state/observation space
        _is_trained (bool): Whether the model has been trained
    """

    def __init__(
        self,
        agent: "BaseAgent",
        config: DecisionConfig = DecisionConfig(),
    ):
        """Initialize the DecisionModule.

        Args:
            agent: The BaseAgent instance this module serves
            config: Configuration object with algorithm parameters
        """
        self.agent_id = agent.agent_id
        self.config = config
        self.agent = agent

        # Get action space from environment
        self.num_actions = self._get_action_space_size()
        self.state_dim = config.rl_state_dim

        # Initialize algorithm
        self.algorithm: Any = None
        self._is_trained = False

        # Create observation and action spaces for SB3
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

        # Initialize the decision algorithm
        self._initialize_algorithm()

        logger.info(
            f"Initialized DecisionModule for agent {self.agent_id} with {config.algorithm_type}"
        )

    def _get_action_space_size(self) -> int:
        """Get the number of actions from the environment's action space.

        Returns:
            int: Number of possible actions
        """
        try:
            # Get action space from environment
            if hasattr(self.agent.environment, "action_space"):
                action_space = self.agent.environment.action_space
                # If it's a callable (like a function), call it to get the actual space
                if callable(action_space):
                    action_space = action_space()
                if hasattr(action_space, "n"):
                    return int(action_space.n)
            # Fallback: count actions in Action enum
            from farm.core.action import ActionType

            return len(ActionType)
        except Exception as e:
            logger.warning(
                f"Could not determine action space size for agent {self.agent_id}: {e}"
            )
            # Default fallback
            return 6  # DEFEND, ATTACK, GATHER, SHARE, MOVE, REPRODUCE

    def _create_observation_space(self):
        """Create observation space for SB3 compatibility."""
        from gymnasium import spaces

        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

    def _create_action_space(self):
        """Create action space for SB3 compatibility."""
        from gymnasium import spaces

        return spaces.Discrete(self.num_actions)

    def _initialize_algorithm(self):
        """Initialize the decision algorithm based on configuration."""
        algorithm_type = self.config.algorithm_type

        if algorithm_type == "ddqn" and SB3_AVAILABLE:
            self._initialize_sb3_ddqn()
        elif algorithm_type == "ppo" and SB3_AVAILABLE:
            self._initialize_sb3_ppo()
        else:
            logger.warning(
                f"Algorithm {algorithm_type} not available or not supported. Using fallback."
            )
            self._initialize_fallback()

    def _initialize_sb3_ddqn(self):
        """Initialize DDQN using Stable Baselines3."""
        try:
            # Create a dummy environment for SB3
            dummy_env = SB3Wrapper(self.observation_space, self.action_space)

            # Configure DDQN parameters
            ddqn_kwargs = {
                "policy": self.config.algorithm_params.get("policy", "MlpPolicy"),
                "env": dummy_env,
                "learning_rate": self.config.learning_rate,
                "buffer_size": self.config.rl_buffer_size,
                "learning_starts": 100,
                "batch_size": self.config.rl_batch_size,
                "tau": self.config.tau,
                "gamma": self.config.gamma,
                "train_freq": self.config.rl_train_freq,
                "gradient_steps": 1,
                "target_update_interval": 1000,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": self.config.epsilon_start,
                "exploration_final_eps": self.config.epsilon_min,
                "verbose": 0,
            }

            # Add any additional parameters from config
            ddqn_kwargs.update(self.config.algorithm_params)

            if DQN is not None:
                self.algorithm = DQN(**ddqn_kwargs)
            else:
                self._initialize_fallback()
                return
            logger.info(f"Initialized SB3 DDQN for agent {self.agent_id}")

        except Exception as e:
            logger.error(
                f"Failed to initialize SB3 DDQN for agent {self.agent_id}: {e}"
            )
            self._initialize_fallback()

    def _initialize_sb3_ppo(self):
        """Initialize PPO using Stable Baselines3."""
        try:
            from stable_baselines3 import PPO

            # Create a dummy environment for SB3
            dummy_env = SB3Wrapper(self.observation_space, self.action_space)

            # Configure PPO parameters
            ppo_kwargs = {
                "policy": self.config.algorithm_params.get("policy", "MlpPolicy"),
                "env": dummy_env,
                "learning_rate": self.config.learning_rate,
                "n_steps": 2048,
                "batch_size": self.config.rl_batch_size,
                "n_epochs": 10,
                "gamma": self.config.gamma,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "verbose": 0,
            }

            # Add any additional parameters from config
            ppo_kwargs.update(self.config.algorithm_params)

            self.algorithm = PPO(**ppo_kwargs)
            logger.info(f"Initialized SB3 PPO for agent {self.agent_id}")

        except Exception as e:
            logger.error(f"Failed to initialize SB3 PPO for agent {self.agent_id}: {e}")
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

            def learn(self, total_timesteps=1):
                pass  # No learning in fallback

        self.algorithm = FallbackAlgorithm(self.num_actions, self.config.epsilon_start)

    def decide_action(self, state: torch.Tensor) -> int:
        """Decide which action to take given the current state.

        Args:
            state: Current state observation as tensor

        Returns:
            int: Selected action index
        """
        try:
            # Convert state to numpy for SB3 compatibility
            if isinstance(state, torch.Tensor):
                state_np = state.detach().cpu().numpy()
            else:
                state_np = np.array(state, dtype=np.float32)

            # Ensure correct shape
            if state_np.ndim == 1:
                state_np = state_np.reshape(1, -1)

            # Get action from algorithm
            if self.algorithm is not None and hasattr(self.algorithm, "predict"):
                action, _ = self.algorithm.predict(state_np, deterministic=False)
            else:
                action = np.random.randint(self.num_actions)

            # Ensure action is within valid range
            action = int(action)
            if action < 0 or action >= self.num_actions:
                logger.warning(
                    f"Invalid action {action} for agent {self.agent_id}, using random"
                )
                action = np.random.randint(self.num_actions)

            return action

        except Exception as e:
            logger.error(f"Error in decide_action for agent {self.agent_id}: {e}")
            # Fallback to random action
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
            # For SB3 algorithms, we need to handle experience differently
            # Since we're not using a traditional Gym environment, we'll simulate
            # the learning process by calling learn() periodically
            if (
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
            np.ndarray: Probability distribution over actions
        """
        try:
            # For SB3 algorithms that support it, get action probabilities
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
                and hasattr(self.algorithm, "save")
                and callable(getattr(self.algorithm, "save", None))
            ):
                self.algorithm.save(path)
            else:
                # For fallback algorithm, save basic info
                import pickle

                with open(f"{path}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "agent_id": self.agent_id,
                            "config": self.config.dict(),
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
            if (
                self.algorithm is not None
                and hasattr(self.algorithm, "load")
                and callable(getattr(self.algorithm, "load", None))
            ):
                # For SB3 models
                if self.config.algorithm_type == "ddqn" and DQN is not None:
                    self.algorithm = DQN.load(path)
                elif self.config.algorithm_type == "ppo":
                    try:
                        from stable_baselines3 import PPO

                        if PPO is not None:
                            self.algorithm = PPO.load(path)
                        else:
                            raise ImportError("PPO not available")
                    except ImportError:
                        logger.error(
                            f"PPO not available for loading model for agent {self.agent_id}"
                        )
                        return
            else:
                # For fallback algorithm
                import pickle

                with open(f"{path}.pkl", "rb") as f:
                    data = pickle.load(f)
                    self._is_trained = data.get("is_trained", False)

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
            "sb3_available": SB3_AVAILABLE,
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


def create_decision_state(agent: "BaseAgent") -> torch.Tensor:
    """Create a decision state tensor from an agent.

    This function extracts relevant state information from an agent
    and formats it as a tensor for the decision module.

    Args:
        agent: The agent to extract state from

    Returns:
        torch.Tensor: State tensor for decision making
    """
    try:
        # Extract basic state features
        state_features = [
            agent.current_health / agent.starting_health,  # Health ratio
            agent.resource_level / 50.0,  # Normalized resource level
            agent.position[0] / 100.0,  # Normalized x position
            agent.position[1] / 100.0,  # Normalized y position
            float(agent.is_defending),  # Defense status
            agent.starvation_threshold / agent.max_starvation,  # Starvation ratio
        ]

        # Add environment features if available
        if hasattr(agent, "environment") and agent.environment:
            env = agent.environment
            nearby_resources = len(env.get_nearby_resources(agent.position, 20.0))
            nearby_agents = len(env.get_nearby_agents(agent.position, 20.0))
            total_resources = (
                sum(r.amount for r in env.resources) if env.resources else 0
            )

            state_features.extend(
                [
                    nearby_resources / 10.0,  # Normalized nearby resources
                    nearby_agents / 10.0,  # Normalized nearby agents
                    total_resources / 1000.0,  # Normalized total resources
                    env.time / 1000.0,  # Normalized time
                ]
            )

        # Convert to tensor
        state_tensor = torch.tensor(state_features, dtype=torch.float32)
        return state_tensor

    except Exception as e:
        logger.error(f"Error creating decision state for agent {agent.agent_id}: {e}")
        # Return a zero tensor as fallback
        return torch.zeros(10, dtype=torch.float32)


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
from farm.core.decision.decision import DecisionModule, create_decision_state

# Create config
config = DecisionConfig(
    algorithm_type="ddqn",  # Use DDQN with SB3
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01
)

# Create DecisionModule
decision_module = DecisionModule(agent=agent, config=config)

# Create state tensor
state = create_decision_state(agent)

# Get action
action_index = decision_module.decide_action(state)
print(f"Selected action index: {action_index}")

# Update with experience (after action execution)
reward = 1.0  # Some reward
next_state = create_decision_state(agent)  # State after action
done = False  # Episode not done
decision_module.update(state, action_index, reward, next_state, done)

# Save/load model
decision_module.save_model("agent_model.zip")
decision_module.load_model("agent_model.zip")

# Get model info
info = decision_module.get_model_info()
print(f"Model info: {info}")
```
"""
