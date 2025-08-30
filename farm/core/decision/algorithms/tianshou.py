"""Tianshou RL algorithm wrappers for AgentFarm.

This module provides wrapper classes that integrate Tianshou algorithms
with the AgentFarm RL algorithm interface, offering a lightweight and
Windows-compatible alternative to other RL libraries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

from .rl_base import RLAlgorithm, SimpleReplayBuffer

logger = logging.getLogger(__name__)


class TianshouWrapper(RLAlgorithm):
    """Base wrapper class for Tianshou algorithms.

    This class provides a common interface for all Tianshou algorithms,
    adapting them to work with the AgentFarm RL algorithm system.
    """

    def __init__(
        self,
        num_actions: int,
        algorithm_name: str,
        state_dim: int,
        algorithm_config: Optional[Dict[str, Any]] = None,
        buffer_size: int = 10000,
        batch_size: int = 32,
        train_freq: int = 4,
        **kwargs: Any,
    ):
        """Initialize the Tianshou wrapper.

        Args:
            num_actions: Number of possible actions
            algorithm_name: Name of the Tianshou algorithm (e.g., "PPO", "SAC")
            state_dim: Dimension of the state space
            algorithm_config: Configuration for the Tianshou algorithm
            buffer_size: Size of the experience replay buffer
            batch_size: Batch size for training
            train_freq: How often to train (every N steps)
            **kwargs: Additional arguments
        """
        super().__init__(num_actions=num_actions, **kwargs)

        self.algorithm_name = algorithm_name
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.train_freq = train_freq

        # Import Tianshou components
        try:
            import torch
            from tianshou.data import ReplayBuffer
            from tianshou.env import DummyVectorEnv
            from tianshou.policy import BasePolicy
        except ImportError as e:
            raise ImportError(
                "Tianshou is required for this wrapper. Install with: pip install tianshou"
            ) from e

        # Set up algorithm configuration
        self.algorithm_config = self._get_default_config(algorithm_config or {})

        # Initialize replay buffer
        self.replay_buffer = SimpleReplayBuffer(max_size=buffer_size)

        # Initialize algorithm
        self.policy = None
        self._initialize_policy()

    def _get_default_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get default configuration for the algorithm."""
        base_config = {
            # Common configuration
            "device": "cpu",
            "lr": 3e-4,
            "gamma": 0.99,
            "n_step": 1,
            "target_update_freq": 500,
        }

        # Algorithm-specific defaults
        if self.algorithm_name == "PPO":
            ppo_defaults = {
                "lr": 3e-4,
                "gamma": 0.99,
                "eps_clip": 0.2,
                "max_grad_norm": 0.5,
                "vf_coef": 0.5,
                "ent_coef": 0.01,
                "gae_lambda": 0.95,
                "max_batchsize": 256,
                "repeat_per_collect": 10,
            }
            base_config.update(ppo_defaults)
        elif self.algorithm_name == "SAC":
            sac_defaults = {
                "lr": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "alpha": 0.2,
                "auto_alpha": True,
                "target_entropy": "auto",
            }
            base_config.update(sac_defaults)
        elif self.algorithm_name == "DQN":
            dqn_defaults = {
                "lr": 1e-3,
                "gamma": 0.99,
                "n_step": 3,
                "target_update_freq": 500,
                "eps_test": 0.05,
                "eps_train": 0.1,
                "eps_train_final": 0.05,
            }
            base_config.update(dqn_defaults)
        elif self.algorithm_name == "A2C":
            a2c_defaults = {
                "lr": 7e-4,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "max_grad_norm": 0.5,
                "vf_coef": 0.5,
                "ent_coef": 0.01,
                "max_batchsize": 256,
            }
            base_config.update(a2c_defaults)
        elif self.algorithm_name == "DDPG":
            ddpg_defaults = {
                "lr": 1e-3,
                "gamma": 0.99,
                "tau": 0.005,
                "exploration_noise": 0.1,
            }
            base_config.update(ddpg_defaults)

        # Update with user configuration
        base_config.update(user_config)
        return base_config

    def _initialize_policy(self) -> None:
        """Initialize the Tianshou policy."""
        try:
            import torch
            import torch.nn as nn
            from tianshou.data import ReplayBuffer
            from tianshou.policy import BasePolicy

            # Validate state_dim
            if self.state_dim is None or self.state_dim <= 0:
                raise ValueError(f"Invalid state_dim: {self.state_dim}")

            if self.algorithm_name == "PPO":
                from tianshou.policy import PPOPolicy

                # Create actor and critic networks using Tianshou's Net class
                from tianshou.utils.net.common import Net as TianshouNet
                from tianshou.utils.net.discrete import Actor as DiscreteActor
                from tianshou.utils.net.discrete import Critic as DiscreteCritic

                actor_net = TianshouNet(
                    self.state_dim,
                    self.num_actions,
                    [64, 64],
                    device=self.algorithm_config["device"],
                )
                actor = DiscreteActor(
                    preprocess_net=actor_net,
                    action_shape=(self.num_actions,),
                )

                critic_net = TianshouNet(
                    self.state_dim,
                    1,  # Critic outputs a single value
                    [64, 64],
                    device=self.algorithm_config["device"],
                )
                critic = DiscreteCritic(
                    preprocess_net=critic_net,
                )

                # Create optimizers
                actor_optim = torch.optim.Adam(
                    actor.parameters(), lr=self.algorithm_config["lr"]
                )
                critic_optim = torch.optim.Adam(
                    critic.parameters(), lr=self.algorithm_config["lr"]
                )

                # Create policy
                import gymnasium as gym

                # Filter out parameters that shouldn't be passed to PPOPolicy
                ppo_params = {
                    k: v
                    for k, v in self.algorithm_config.items()
                    if k
                    not in [
                        "lr",
                        "device",
                        "gamma",
                        "tau",
                        "alpha",
                        "auto_alpha",
                        "target_entropy",
                        "n_step",
                        "target_update_freq",
                        "eps_test",
                        "eps_train",
                        "eps_train_final",
                        "repeat_per_collect",
                        "max_batchsize",
                    ]
                }

                self.policy = PPOPolicy(
                    actor=actor,
                    critic=critic,
                    optim=actor_optim,
                    dist_fn=torch.distributions.Categorical,
                    action_space=gym.spaces.Discrete(self.num_actions),
                    **ppo_params,
                )

            elif self.algorithm_name == "SAC":
                from tianshou.policy import SACPolicy
                from tianshou.utils.net.continuous import ActorProb as ContinuousActor
                from tianshou.utils.net.continuous import Critic as ContinuousCritic

                # Create actor and critic networks
                actor = ContinuousActor(
                    preprocess_net=nn.Sequential(
                        nn.Linear(self.state_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                    ),
                    action_shape=(1,),  # Single continuous action, will be discretized
                    max_action=1.0,
                    device=self.algorithm_config["device"],
                )

                critic1 = ContinuousCritic(
                    preprocess_net=nn.Sequential(
                        nn.Linear(self.state_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                    ),
                    device=self.algorithm_config["device"],
                )

                critic2 = ContinuousCritic(
                    preprocess_net=nn.Sequential(
                        nn.Linear(self.state_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                    ),
                    device=self.algorithm_config["device"],
                )

                # Create optimizers
                actor_optim = torch.optim.Adam(
                    actor.parameters(), lr=self.algorithm_config["lr"]
                )
                critic1_optim = torch.optim.Adam(
                    critic1.parameters(), lr=self.algorithm_config["lr"]
                )
                critic2_optim = torch.optim.Adam(
                    critic2.parameters(), lr=self.algorithm_config["lr"]
                )

                # Create policy
                import gymnasium as gym

                self.policy = SACPolicy(
                    actor=actor,
                    actor_optim=actor_optim,
                    action_space=gym.spaces.Box(
                        low=-1, high=1, shape=(1,)
                    ),  # Continuous action space
                    **{
                        k: v
                        for k, v in self.algorithm_config.items()
                        if k not in ["lr", "device"]
                    },
                )

            elif self.algorithm_name == "DQN":
                from tianshou.policy import DQNPolicy
                from tianshou.utils.net.common import Net as DiscreteNet

                # Create Q-network
                net = DiscreteNet(
                    self.state_dim,
                    self.num_actions,
                    [64, 64],
                    device=self.algorithm_config["device"],
                )

                # Create optimizer
                optim = torch.optim.Adam(
                    net.parameters(), lr=self.algorithm_config["lr"]
                )

                # Create policy
                import gymnasium as gym

                self.policy = DQNPolicy(
                    model=net,
                    optim=optim,
                    action_space=gym.spaces.Discrete(self.num_actions),
                    **{
                        k: v
                        for k, v in self.algorithm_config.items()
                        if k not in ["lr", "device", "gamma"]
                    },
                )

            elif self.algorithm_name == "A2C":
                from tianshou.policy import A2CPolicy

                # Create actor and critic networks using Tianshou's Net class
                from tianshou.utils.net.common import Net as TianshouNet
                from tianshou.utils.net.discrete import Actor as DiscreteActor
                from tianshou.utils.net.discrete import Critic as DiscreteCritic

                actor_net = TianshouNet(
                    self.state_dim,
                    self.num_actions,
                    [64, 64],
                    device=self.algorithm_config["device"],
                )
                actor = DiscreteActor(
                    preprocess_net=actor_net,
                    action_shape=(self.num_actions,),
                )

                critic_net = TianshouNet(
                    self.state_dim,
                    1,  # Critic outputs a single value
                    [64, 64],
                    device=self.algorithm_config["device"],
                )
                critic = DiscreteCritic(
                    preprocess_net=critic_net,
                )

                # Create optimizers
                actor_optim = torch.optim.Adam(
                    actor.parameters(), lr=self.algorithm_config["lr"]
                )
                critic_optim = torch.optim.Adam(
                    critic.parameters(), lr=self.algorithm_config["lr"]
                )

                # Create policy
                import gymnasium as gym

                # Filter out parameters that shouldn't be passed to A2CPolicy
                a2c_params = {
                    k: v
                    for k, v in self.algorithm_config.items()
                    if k
                    not in [
                        "lr",
                        "device",
                        "gamma",
                        "tau",
                        "alpha",
                        "auto_alpha",
                        "target_entropy",
                        "n_step",
                        "target_update_freq",
                        "eps_test",
                        "eps_train",
                        "eps_train_final",
                    ]
                }

                self.policy = A2CPolicy(
                    actor=actor,
                    critic=critic,
                    optim=actor_optim,
                    dist_fn=torch.distributions.Categorical,
                    action_space=gym.spaces.Discrete(self.num_actions),
                    **a2c_params,
                )

            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")

            logger.info(f"Initialized {self.algorithm_name} policy with Tianshou")

        except Exception as e:
            logger.error(f"Failed to initialize {self.algorithm_name} policy: {e}")
            raise

    def select_action(self, state: np.ndarray) -> int:
        """Select an action using the Tianshou policy.

        Args:
            state: Current state observation

        Returns:
            Selected action index
        """
        if self.policy is None:
            raise RuntimeError("Policy not initialized")

        # Convert state to the expected format
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        elif hasattr(state, "cpu") and hasattr(state, "numpy"):  # Handle torch tensors
            try:
                import torch

                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy().astype(np.float32)
            except ImportError:
                pass

        # Add batch dimension if needed
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        # Convert to torch tensor
        try:
            import torch

            state_tensor = torch.from_numpy(state).float()
        except ImportError:
            raise ImportError("PyTorch is required for Tianshou")

        # Get action from the policy
        with torch.no_grad():
            action = self.policy(state_tensor, state=None)[0]

        # Handle different action types
        if isinstance(action, torch.Tensor):
            if action.ndim == 1:
                action = action[0]
            action = action.item()

        # For continuous actions, discretize
        if isinstance(action, float):
            action = int(
                np.clip(
                    np.round(action * (self.num_actions - 1)), 0, self.num_actions - 1
                )
            )

        return int(action)

    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        """Predict action probabilities for exploration.

        Args:
            state: Current state observation

        Returns:
            Action probability distribution
        """
        if self.policy is None:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=float)

        # For Tianshou, we can get action probabilities from the policy
        action = self.select_action(state)

        # Create a probability distribution concentrated on the selected action
        probs = np.zeros(self.num_actions)
        probs[action] = 0.8  # 80% probability on selected action

        # Distribute remaining probability uniformly
        remaining_prob = 0.2
        uniform_prob = remaining_prob / (self.num_actions - 1)
        probs[probs == 0] = uniform_prob

        return probs

    def store_experience(
        self,
        state: Union[np.ndarray, Any],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, Any],
        done: bool,
        **kwargs: Any,
    ) -> None:
        """Store an experience in the replay buffer.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode ended
            **kwargs: Additional experience data
        """
        # Convert states to numpy arrays
        try:
            import torch

            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.cpu().numpy()
        except ImportError:
            # torch not available, assume states are already numpy arrays
            pass

        self.replay_buffer.append(state, action, reward, next_state, done, **kwargs)

    def train_on_batch(self, batch: Dict[str, Any], **kwargs: Any) -> Dict[str, float]:
        """Train the algorithm on a batch of experiences.

        Args:
            batch: Batch of experiences
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics
        """
        if self.policy is None:
            return {"loss": 0.0}

        try:
            # Convert batch to Tianshou format
            import torch

            # Extract data from our simple replay buffer format
            if len(self.replay_buffer.buffer) >= self.batch_size:
                # Sample from our buffer
                indices = np.random.choice(
                    len(self.replay_buffer.buffer), self.batch_size, replace=False
                )
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

                for idx in indices:
                    exp = self.replay_buffer.buffer[idx]
                    states.append(exp["state"])
                    actions.append(exp["action"])
                    rewards.append(exp["reward"])
                    next_states.append(exp["next_state"])
                    dones.append(exp["done"])

                # Convert to tensors
                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(np.array(actions), dtype=torch.long)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(np.array(dones), dtype=torch.float32)

                # Create batch dictionary compatible with RolloutBatchProtocol
                tianshou_batch = {
                    "obs": states,
                    "act": actions,
                    "rew": rewards,
                    "obs_next": next_states,
                    "done": dones,
                    "terminated": dones,  # Required by RolloutBatchProtocol
                    "truncated": torch.zeros_like(
                        dones
                    ),  # Required by RolloutBatchProtocol
                    "info": [{}] * len(dones),  # Required by RolloutBatchProtocol
                }

                # Train the policy
                result = self.policy.learn(
                    tianshou_batch, batch_size=self.batch_size, repeat=1  # type: ignore
                )

                # Extract metrics
                metrics = {}
                if isinstance(result, dict):
                    for key in ["loss", "actor_loss", "critic_loss"]:
                        if key in result:
                            value = result[key]
                            try:
                                if hasattr(value, "item"):
                                    metrics[key] = float(value.item())  # type: ignore
                                else:
                                    metrics[key] = float(value)  # type: ignore
                            except (TypeError, AttributeError):
                                pass  # Skip if conversion fails

                return metrics

            return {"loss": 0.0}

        except Exception as e:
            logger.warning(f"Training failed: {e}")
            return {"loss": 0.0}

    def should_train(self) -> bool:
        """Check if the algorithm should train on the current step."""
        return (
            len(self.replay_buffer) >= self.batch_size
            and self.step_count > 0
            and self.step_count % self.train_freq == 0
        )

    def get_model_state(self) -> Dict[str, Any]:
        """Get the current model state for saving."""
        if self.policy is None:
            return {}

        try:
            import torch

            state = {
                "policy_state_dict": self.policy.state_dict(),
                "step_count": self.step_count,
                "buffer_size": len(self.replay_buffer),
                "algorithm_name": self.algorithm_name,
                "algorithm_config": self.algorithm_config,
            }
            return state
        except Exception as e:
            logger.warning(f"Failed to get model state: {e}")
            return {
                "step_count": self.step_count,
                "buffer_size": len(self.replay_buffer),
                "algorithm_name": self.algorithm_name,
            }

    def load_model_state(self, state: Dict[str, Any]) -> None:
        """Load a saved model state."""
        if "policy_state_dict" in state and self.policy is not None:
            try:
                self.policy.load_state_dict(state["policy_state_dict"])
            except Exception as e:
                logger.warning(f"Failed to load policy state: {e}")

        if "step_count" in state:
            self._step_count = state["step_count"]

    def train(self, batch: Any, **kwargs: Any) -> None:
        """Train method required by ActionAlgorithm interface."""
        if self.should_train() and self.policy is not None:
            try:
                self.train_on_batch({})
            except Exception as e:
                logger.warning(f"Training failed: {e}")


class PPOWrapper(TianshouWrapper):
    """Wrapper for PPO (Proximal Policy Optimization) algorithm using Tianshou."""

    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        algorithm_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize PPO wrapper.

        Args:
            num_actions: Number of possible actions
            state_dim: Dimension of the state space
            algorithm_config: PPO-specific configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            num_actions=num_actions,
            algorithm_name="PPO",
            state_dim=state_dim,
            algorithm_config=algorithm_config,
            **kwargs,
        )


class SACWrapper(TianshouWrapper):
    """Wrapper for SAC (Soft Actor-Critic) algorithm using Tianshou."""

    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        algorithm_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize SAC wrapper.

        Args:
            num_actions: Number of possible actions
            state_dim: Dimension of the state space
            algorithm_config: SAC-specific configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            num_actions=num_actions,
            algorithm_name="SAC",
            state_dim=state_dim,
            algorithm_config=algorithm_config,
            **kwargs,
        )


class A2CWrapper(TianshouWrapper):
    """Wrapper for A2C (Advantage Actor-Critic) algorithm using Tianshou."""

    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        algorithm_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize A2C wrapper.

        Args:
            num_actions: Number of possible actions
            state_dim: Dimension of the state space
            algorithm_config: A2C-specific configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            num_actions=num_actions,
            algorithm_name="A2C",
            state_dim=state_dim,
            algorithm_config=algorithm_config,
            **kwargs,
        )


class DQNWrapper(TianshouWrapper):
    """Wrapper for DQN (Deep Q-Network) algorithm using Tianshou."""

    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        algorithm_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize DQN wrapper.

        Args:
            num_actions: Number of possible actions
            state_dim: Dimension of the state space
            algorithm_config: DQN-specific configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            num_actions=num_actions,
            algorithm_name="DQN",
            state_dim=state_dim,
            algorithm_config=algorithm_config,
            **kwargs,
        )


class DDPGWrapper(TianshouWrapper):
    """Wrapper for DDPG (Deep Deterministic Policy Gradient) algorithm using Tianshou."""

    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        algorithm_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize DDPG wrapper.

        Args:
            num_actions: Number of possible actions
            state_dim: Dimension of the state space
            algorithm_config: DDPG-specific configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            num_actions=num_actions,
            algorithm_name="DDPG",
            state_dim=state_dim,
            algorithm_config=algorithm_config,
            **kwargs,
        )
