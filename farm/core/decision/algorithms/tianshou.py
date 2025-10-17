"""Tianshou RL algorithm wrappers for AgentFarm.

This module provides wrapper classes that integrate Tianshou algorithms
with the AgentFarm RL algorithm interface, offering a lightweight and
Windows-compatible alternative to other RL libraries.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import gymnasium

from .rl_base import RLAlgorithm, SimpleReplayBuffer

from farm.utils.logging import get_logger

logger = get_logger(__name__)


class TianshouWrapper(RLAlgorithm):
    """Base wrapper class for Tianshou algorithms.

    This class provides a common interface for all Tianshou algorithms,
    adapting them to work with the AgentFarm RL algorithm system.
    """

    # Parameters that are handled separately and should be filtered out
    # when passing config to Tianshou policy constructors
    _EXCLUDED_PARAMS = frozenset([
        "lr", "device", "gamma", "tau", "alpha", "auto_alpha", "target_entropy",
        "n_step", "target_update_freq", "eps_test", "eps_train", "eps_train_final",
        "repeat_per_collect", "max_batchsize"
    ])

    def __init__(
        self,
        num_actions: int,
        algorithm_name: str,
        state_dim: int,
        observation_shape: Optional[Tuple[int, ...]] = None,
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
            state_dim: Dimension of the state space (flattened)
            observation_shape: Shape of the observation space (e.g., (13, 13, 13))
            algorithm_config: Configuration for the Tianshou algorithm
            buffer_size: Size of the experience replay buffer
            batch_size: Batch size for training
            train_freq: How often to train (every N steps)
            **kwargs: Additional arguments
        """
        super().__init__(num_actions=num_actions, **kwargs)

        self.algorithm_name = algorithm_name
        self.state_dim = state_dim
        self.observation_shape = observation_shape or (state_dim,)
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
                import torch.nn as nn

                # Create CNN-based actor and critic networks for spatial observations
                class SpatialActorNet(nn.Module):
                    def __init__(self, observation_shape, num_actions, device):
                        super().__init__()
                        self.observation_shape = observation_shape

                        # CNN layers for spatial processing
                        self.conv_layers = nn.Sequential(
                            nn.Conv2d(observation_shape[0], 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                        )

                        # Calculate flattened size after CNN
                        with torch.no_grad():
                            dummy_input = torch.zeros(1, *observation_shape)
                            conv_output = self.conv_layers(dummy_input)
                            self.flattened_size = conv_output.numel()

                        # Fully connected layers
                        self.fc_layers = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(self.flattened_size, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                        )

                        # Action output layer
                        self.action_head = nn.Linear(256, num_actions)

                    def forward(self, obs):
                        # Handle both batched and single observations
                        if obs.dim() == 3:  # Single observation (C, H, W)
                            obs = obs.unsqueeze(0)  # Add batch dimension

                        x = self.conv_layers(obs)
                        x = self.fc_layers(x)
                        action_logits = self.action_head(x)
                        return action_logits

                class SpatialCriticNet(nn.Module):
                    def __init__(self, observation_shape, device):
                        super().__init__()
                        self.observation_shape = observation_shape

                        # CNN layers for spatial processing
                        self.conv_layers = nn.Sequential(
                            nn.Conv2d(observation_shape[0], 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                        )

                        # Calculate flattened size after CNN
                        with torch.no_grad():
                            dummy_input = torch.zeros(1, *observation_shape)
                            conv_output = self.conv_layers(dummy_input)
                            self.flattened_size = conv_output.numel()

                        # Fully connected layers
                        self.fc_layers = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(self.flattened_size, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 1),  # Single value output for critic
                        )

                    def forward(self, obs):
                        # Handle both batched and single observations
                        if obs.dim() == 3:  # Single observation (C, H, W)
                            obs = obs.unsqueeze(0)  # Add batch dimension

                        x = self.conv_layers(obs)
                        value = self.fc_layers(x)
                        return value

                # Create spatial networks
                actor_net = SpatialActorNet(self.observation_shape, self.num_actions, self.algorithm_config["device"])
                critic_net = SpatialCriticNet(self.observation_shape, self.algorithm_config["device"])

                # Move to device
                actor_net.to(self.algorithm_config["device"])
                critic_net.to(self.algorithm_config["device"])

                # Create optimizers
                actor_optim = torch.optim.Adam(
                    actor_net.parameters(), lr=self.algorithm_config["lr"]
                )

                # Create policy

                # Filter out parameters that shouldn't be passed to PPOPolicy
                ppo_params = {
                    k: v
                    for k, v in self.algorithm_config.items()
                    if k not in self._EXCLUDED_PARAMS
                }

                self.policy = PPOPolicy(
                    actor=actor_net,
                    critic=critic_net,
                    optim=actor_optim,
                    dist_fn=torch.distributions.Categorical,
                    action_space=gymnasium.spaces.Discrete(self.num_actions),
                    **ppo_params,
                )

            elif self.algorithm_name == "SAC":
                from tianshou.policy import SACPolicy
                import torch.nn as nn

                # Create CNN-based actor and critic networks for spatial observations
                class SpatialSACActorNet(nn.Module):
                    def __init__(self, observation_shape, device):
                        super().__init__()
                        self.observation_shape = observation_shape

                        # CNN layers for spatial processing
                        self.conv_layers = nn.Sequential(
                            nn.Conv2d(observation_shape[0], 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                        )

                        # Calculate flattened size after CNN
                        with torch.no_grad():
                            dummy_input = torch.zeros(1, *observation_shape)
                            conv_output = self.conv_layers(dummy_input)
                            self.flattened_size = conv_output.numel()

                        # Actor head for continuous actions
                        self.actor_layers = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(self.flattened_size, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                        )

                        # Output layers for SAC (mean and log_std)
                        self.mean_head = nn.Linear(256, 1)  # Single continuous action
                        self.log_std_head = nn.Linear(256, 1)

                    def forward(self, obs):
                        # Handle both batched and single observations
                        if obs.dim() == 3:  # Single observation (C, H, W)
                            obs = obs.unsqueeze(0)  # Add batch dimension

                        x = self.conv_layers(obs)
                        x = self.actor_layers(x)
                        mean = self.mean_head(x)
                        log_std = self.log_std_head(x)
                        return mean, log_std

                class SpatialCriticNet(nn.Module):
                    def __init__(self, observation_shape, action_dim=1, device=None):
                        super().__init__()
                        self.observation_shape = observation_shape

                        # CNN layers for spatial processing
                        self.conv_layers = nn.Sequential(
                            nn.Conv2d(observation_shape[0], 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                        )

                        # Calculate flattened size after CNN
                        with torch.no_grad():
                            dummy_input = torch.zeros(1, *observation_shape)
                            conv_output = self.conv_layers(dummy_input)
                            self.flattened_size = conv_output.numel()

                        # Critic layers (state + action)
                        self.critic_layers = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(self.flattened_size + action_dim, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 1),
                        )

                    def forward(self, obs, act):
                        # Handle both batched and single observations
                        if obs.dim() == 3:  # Single observation (C, H, W)
                            obs = obs.unsqueeze(0)  # Add batch dimension

                        x = self.conv_layers(obs)
                        x = torch.flatten(x, start_dim=1)  # Flatten spatial dims

                        # Concatenate with action
                        if act.dim() == 1:
                            act = act.unsqueeze(0)
                        x = torch.cat([x, act], dim=1)

                        value = self.critic_layers(x)
                        return value

                # Create spatial networks
                actor_net = SpatialSACActorNet(self.observation_shape, self.algorithm_config["device"])
                critic1_net = SpatialCriticNet(self.observation_shape, device=self.algorithm_config["device"])
                critic2_net = SpatialCriticNet(self.observation_shape, device=self.algorithm_config["device"])

                # Move to device
                actor_net.to(self.algorithm_config["device"])
                critic1_net.to(self.algorithm_config["device"])
                critic2_net.to(self.algorithm_config["device"])

                # Create optimizers
                actor_optim = torch.optim.Adam(
                    actor_net.parameters(), lr=self.algorithm_config["lr"]
                )
                critic1_optim = torch.optim.Adam(
                    critic1_net.parameters(), lr=self.algorithm_config["lr"]
                )
                critic2_optim = torch.optim.Adam(
                    critic2_net.parameters(), lr=self.algorithm_config["lr"]
                )

                # Create policy

                # Filter parameters specifically for SACPolicy
                sac_params = {
                    k: v
                    for k, v in self.algorithm_config.items()
                    if k in ["tau", "gamma", "alpha"]
                }

                self.policy = SACPolicy(
                    actor=actor_net,
                    actor_optim=actor_optim,
                    critic1=critic1_net,
                    critic1_optim=critic1_optim,
                    critic2=critic2_net,
                    critic2_optim=critic2_optim,
                    action_space=gymnasium.spaces.Box(
                        low=-1, high=1, shape=(1,)
                    ),  # Continuous action space (will be discretized)
                    **sac_params,
                )

            elif self.algorithm_name == "DQN":
                from tianshou.policy import DQNPolicy
                import torch.nn as nn

                # Create Q-network that handles both 1D and 3D observations
                class AdaptiveQNet(nn.Module):
                    def __init__(self, observation_shape, num_actions, device):
                        super().__init__()
                        self.observation_shape = observation_shape
                        self.is_spatial = len(observation_shape) > 1

                        if self.is_spatial:
                            # CNN layers for spatial processing (3D observations)
                            self.conv_layers = nn.Sequential(
                                nn.Conv2d(observation_shape[0], 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                            )

                            # Calculate flattened size after CNN
                            with torch.no_grad():
                                dummy_input = torch.zeros(1, *observation_shape)
                                conv_output = self.conv_layers(dummy_input)
                                self.flattened_size = conv_output.numel()

                            # Q-value output layers for spatial observations
                            self.q_layers = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(self.flattened_size, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, num_actions),  # Q-values for each action
                            )
                        else:
                            # Fully connected layers for 1D observations
                            input_size = observation_shape[0]
                            self.q_layers = nn.Sequential(
                                nn.Linear(input_size, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, num_actions),  # Q-values for each action
                            )

                    def forward(self, obs, state=None, info=None):
                        # Handle both batched and single observations
                        if obs.dim() == 1:  # Single 1D observation
                            obs = obs.unsqueeze(0)  # Add batch dimension
                        elif obs.dim() == 3:  # Single 3D observation (C, H, W)
                            obs = obs.unsqueeze(0)  # Add batch dimension

                        if self.is_spatial:
                            x = self.conv_layers(obs)
                            q_values = self.q_layers(x)
                        else:
                            q_values = self.q_layers(obs)
                        
                        return q_values

                # Create adaptive Q-network
                q_net = AdaptiveQNet(self.observation_shape, self.num_actions, self.algorithm_config["device"])

                # Move to device
                q_net.to(self.algorithm_config["device"])

                # Create optimizer
                optim = torch.optim.Adam(
                    q_net.parameters(), lr=self.algorithm_config["lr"]
                )

                # Create policy

                self.policy = DQNPolicy(
                    model=q_net,
                    optim=optim,
                    action_space=gymnasium.spaces.Discrete(self.num_actions),
                    **{
                        k: v
                        for k, v in self.algorithm_config.items()
                        if k not in self._EXCLUDED_PARAMS
                    },
                )

            elif self.algorithm_name == "A2C":
                from tianshou.policy import A2CPolicy
                import torch.nn as nn

                # Create CNN-based actor and critic networks for spatial observations
                class SpatialA2CActorNet(nn.Module):
                    def __init__(self, observation_shape, num_actions, device):
                        super().__init__()
                        self.observation_shape = observation_shape

                        # CNN layers for spatial processing
                        self.conv_layers = nn.Sequential(
                            nn.Conv2d(observation_shape[0], 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                        )

                        # Calculate flattened size after CNN
                        with torch.no_grad():
                            dummy_input = torch.zeros(1, *observation_shape)
                            conv_output = self.conv_layers(dummy_input)
                            self.flattened_size = conv_output.numel()

                        # Actor head
                        self.actor_layers = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(self.flattened_size, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, num_actions),  # Action logits
                        )

                    def forward(self, obs):
                        # Handle both batched and single observations
                        if obs.dim() == 3:  # Single observation (C, H, W)
                            obs = obs.unsqueeze(0)  # Add batch dimension

                        x = self.conv_layers(obs)
                        action_logits = self.actor_layers(x)
                        return action_logits

                class SpatialA2CCriticNet(nn.Module):
                    def __init__(self, observation_shape, device):
                        super().__init__()
                        self.observation_shape = observation_shape

                        # CNN layers for spatial processing
                        self.conv_layers = nn.Sequential(
                            nn.Conv2d(observation_shape[0], 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                        )

                        # Calculate flattened size after CNN
                        with torch.no_grad():
                            dummy_input = torch.zeros(1, *observation_shape)
                            conv_output = self.conv_layers(dummy_input)
                            self.flattened_size = conv_output.numel()

                        # Critic head
                        self.critic_layers = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(self.flattened_size, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 1),  # Single value output
                        )

                    def forward(self, obs):
                        # Handle both batched and single observations
                        if obs.dim() == 3:  # Single observation (C, H, W)
                            obs = obs.unsqueeze(0)  # Add batch dimension

                        x = self.conv_layers(obs)
                        value = self.critic_layers(x)
                        return value

                # Create spatial networks
                actor_net = SpatialA2CActorNet(self.observation_shape, self.num_actions, self.algorithm_config["device"])
                critic_net = SpatialA2CCriticNet(self.observation_shape, self.algorithm_config["device"])

                # Move to device
                actor_net.to(self.algorithm_config["device"])
                critic_net.to(self.algorithm_config["device"])

                # Create optimizers
                actor_optim = torch.optim.Adam(
                    actor_net.parameters(), lr=self.algorithm_config["lr"]
                )
                critic_optim = torch.optim.Adam(
                    critic_net.parameters(), lr=self.algorithm_config["lr"]
                )

                # Create policy

                # Filter out parameters that shouldn't be passed to A2CPolicy
                a2c_params = {
                    k: v
                    for k, v in self.algorithm_config.items()
                    if k not in self._EXCLUDED_PARAMS
                }

                self.policy = A2CPolicy(
                    actor=actor_net,
                    critic=critic_net,
                    optim=actor_optim,
                    dist_fn=torch.distributions.Categorical,
                    action_space=gymnasium.spaces.Discrete(self.num_actions),
                    **a2c_params,
                )

            elif self.algorithm_name == "DDPG":
                from tianshou.policy import DDPGPolicy
                import torch.nn as nn

                # Create CNN-based actor and critic networks for spatial observations
                class SpatialDDPGActorNet(nn.Module):
                    def __init__(self, observation_shape, device):
                        super().__init__()
                        self.observation_shape = observation_shape

                        # CNN layers for spatial processing
                        self.conv_layers = nn.Sequential(
                            nn.Conv2d(observation_shape[0], 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                        )

                        # Calculate flattened size after CNN
                        with torch.no_grad():
                            dummy_input = torch.zeros(1, *observation_shape)
                            conv_output = self.conv_layers(dummy_input)
                            self.flattened_size = conv_output.numel()

                        # Actor head for continuous actions
                        self.actor_layers = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(self.flattened_size, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 1),  # Single continuous action
                            nn.Tanh(),  # Bound output to [-1, 1]
                        )

                    def forward(self, obs):
                        # Handle both batched and single observations
                        if obs.dim() == 3:  # Single observation (C, H, W)
                            obs = obs.unsqueeze(0)  # Add batch dimension

                        x = self.conv_layers(obs)
                        action = self.actor_layers(x)
                        return action

                class SpatialDDPGCriticNet(nn.Module):
                    def __init__(self, observation_shape, action_dim=1, device=None):
                        super().__init__()
                        self.observation_shape = observation_shape

                        # CNN layers for spatial processing
                        self.conv_layers = nn.Sequential(
                            nn.Conv2d(observation_shape[0], 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                        )

                        # Calculate flattened size after CNN
                        with torch.no_grad():
                            dummy_input = torch.zeros(1, *observation_shape)
                            conv_output = self.conv_layers(dummy_input)
                            self.flattened_size = conv_output.numel()

                        # Critic layers (state + action)
                        self.critic_layers = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(self.flattened_size + action_dim, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 1),
                        )

                    def forward(self, obs, act):
                        # Handle both batched and single observations
                        if obs.dim() == 3:  # Single observation (C, H, W)
                            obs = obs.unsqueeze(0)  # Add batch dimension

                        x = self.conv_layers(obs)
                        x = torch.flatten(x, start_dim=1)  # Flatten spatial dims

                        # Concatenate with action
                        if act.dim() == 1:
                            act = act.unsqueeze(0)
                        x = torch.cat([x, act], dim=1)

                        value = self.critic_layers(x)
                        return value

                # Create spatial networks
                actor_net = SpatialDDPGActorNet(self.observation_shape, self.algorithm_config["device"])
                critic_net = SpatialDDPGCriticNet(self.observation_shape, device=self.algorithm_config["device"])

                # Move to device
                actor_net.to(self.algorithm_config["device"])
                critic_net.to(self.algorithm_config["device"])

                # Create optimizers
                actor_optim = torch.optim.Adam(
                    actor_net.parameters(), lr=self.algorithm_config["lr"]
                )
                critic_optim = torch.optim.Adam(
                    critic_net.parameters(), lr=self.algorithm_config["lr"]
                )

                # Create policy

                self.policy = DDPGPolicy(
                    actor=actor_net,
                    actor_optim=actor_optim,
                    critic=critic_net,
                    critic_optim=critic_optim,
                    action_space=gymnasium.spaces.Box(
                        low=-1, high=1, shape=(1,)
                    ),  # Continuous action space (will be discretized)
                    **{
                        k: v
                        for k, v in self.algorithm_config.items()
                        if k not in ["lr", "device"]
                    },
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")

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
        return self.select_action_with_mask(state, action_mask=None)

    def select_action_with_mask(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """Select an action using the Tianshou policy with action masking support.

        Args:
            state: Current state observation
            action_mask: Boolean mask where True indicates valid actions. If None, all actions are valid.

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

        # Add batch dimension if needed (DecisionModule may have already added it)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        elif state.ndim == 3 and state.shape[0] != 1:
            # 3D observation without batch dimension - add it
            state = np.expand_dims(state, axis=0)

        # Convert to torch tensor
        try:
            import torch

            state_tensor = torch.from_numpy(state).float()
        except ImportError as exc:
            raise ImportError("PyTorch is required for Tianshou") from exc

        # Handle action masking for curriculum learning
        if action_mask is not None:
            # For PPO specifically, use a simpler approach
            if self.algorithm_name == "PPO" and hasattr(self.policy, 'actor'):
                # Use PPO's built-in action sampling with manual masking
                max_attempts = 10
                for _ in range(max_attempts):
                    with torch.no_grad():
                        # Get action logits from PPO actor
                        try:
                            # PPO structure: policy.actor -> forward method
                            logits = self.policy.actor(state_tensor)
                            # Apply softmax to get probabilities
                            probs = torch.softmax(logits, dim=-1)
                            # Apply mask by zeroing out invalid actions
                            action_mask_tensor = torch.from_numpy(action_mask.astype(np.float32)).to(state_tensor.device)
                            if action_mask_tensor.ndim == 1:
                                action_mask_tensor = action_mask_tensor.unsqueeze(0)
                            masked_probs = probs * action_mask_tensor
                            # Renormalize
                            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)

                            # Sample from masked distribution
                            dist = torch.distributions.Categorical(probs=masked_probs)
                            action = dist.sample()

                            if isinstance(action, torch.Tensor):
                                if action.ndim == 1:
                                    action = action[0]
                                action = action.item()

                            # Check if action is valid according to mask
                            if int(action) < len(action_mask) and action_mask[int(action)]:
                                return int(action)
                        except Exception as e:
                            logger.debug(f"PPO masking attempt failed: {e}")
                            continue

                # Fallback: pick first valid action
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    return int(valid_actions[0])
                return 0

            # For other algorithms or fallback
            else:
                # Try multiple times to get a valid action
                max_attempts = 10
                for _ in range(max_attempts):
                    with torch.no_grad():
                        try:
                            action = self.policy(state_tensor, state=None)[0]
                        except Exception:
                            # Fallback to random action
                            action = np.random.randint(self.num_actions)

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

                    # Check if action is valid according to mask
                    if action_mask is None or (int(action) < len(action_mask) and action_mask[int(action)]):
                        return int(action)

                # If we couldn't get a valid action after max attempts, pick first valid action
                if action_mask is not None:
                    valid_actions = np.where(action_mask)[0]
                    if len(valid_actions) > 0:
                        return int(valid_actions[0])

                return int(action)
        else:
            # No masking - use standard action selection
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
