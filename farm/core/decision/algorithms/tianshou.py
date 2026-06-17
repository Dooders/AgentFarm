"""Tianshou RL algorithm wrappers for AgentFarm.

This module provides wrapper classes that integrate Tianshou algorithms
with the AgentFarm RL algorithm interface, offering a lightweight and
Windows-compatible alternative to other RL libraries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import gymnasium
import numpy as np
import torch

from .rl_base import PrioritizedReplayBuffer, RLAlgorithm

from farm.core.decision.shape_utils import batch_observation
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class TianshouWrapper(RLAlgorithm):
    """Base wrapper class for Tianshou algorithms.

    This class provides a common interface for all Tianshou algorithms,
    adapting them to work with the AgentFarm RL algorithm system.
    """

    # Parameters that are handled separately and should be filtered out
    # when passing config to Tianshou policy constructors.
    #
    # ``target_update_freq`` stays in this set because most Tianshou policies
    # (PPO / SAC / A2C / DDPG) do not accept it.  The DQN branch forwards the
    # value via an explicit kwarg below so the Chromosome A
    # ``target_update_freq`` gene actually controls the hard target sync.
    _EXCLUDED_PARAMS = frozenset([
        "lr", "device", "gamma", "tau", "alpha", "auto_alpha", "target_entropy",
        "n_step", "estimation_step", "target_update_freq", "eps_test", "eps_train", "eps_train_final",
        "eps_decay", "dqn_hidden_size",
        "repeat_per_collect", "max_batchsize",
    ])
    _OPTIMIZER_ATTRS = (
        "optim",
        "actor_optim",
        "critic_optim",
        "critic1_optim",
        "critic2_optim",
        "alpha_optim",
    )

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
        replay_strategy: Literal["uniform", "prioritized"] = "uniform",
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 100_000,
        per_epsilon: float = 1e-6,
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
            replay_strategy: Replay sampling strategy ("uniform" or "prioritized")
            per_alpha: PER priority exponent
            per_beta_start: Initial IS-correction exponent
            per_beta_end: Final IS-correction exponent
            per_beta_steps: Number of beta annealing steps
            per_epsilon: Stability floor added to priorities
            **kwargs: Additional arguments
        """
        super().__init__(num_actions=num_actions, **kwargs)

        self.algorithm_name = algorithm_name
        self.state_dim = state_dim
        self.observation_shape = observation_shape or (state_dim,)
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.replay_strategy = replay_strategy

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
        if self.algorithm_name == "DQN":
            configured_n_step = int(self.algorithm_config.get("n_step", 1))
            if configured_n_step != 1:
                logger.debug(
                    "DQN wrapper uses one-step replay targets with the current custom replay path; "
                    "overriding n_step=%d -> 1 to avoid inconsistent training targets.",
                    configured_n_step,
                )
                self.algorithm_config["n_step"] = 1

        # Snapshot the epsilon-greedy schedule so :meth:`select_action_with_mask`
        # can drive ``policy.eps`` directly. Tianshou's ``DQNPolicy.__init__``
        # initialises ``policy.eps = 0.0`` and there is no built-in schedule on
        # the custom-replay code path; we therefore apply the configured
        # multiplicative decay each time we ask the policy for an action.
        self._eps_train = float(self.algorithm_config.get("eps_train", 1.0))
        self._eps_train_final = float(self.algorithm_config.get("eps_train_final", 0.05))
        self._eps_test = float(self.algorithm_config.get("eps_test", 0.05))
        self._eps_decay = float(self.algorithm_config.get("eps_decay", 0.995))
        self._eps_current = self._eps_train
        self._train_mode = True

        # Initialize replay buffer (PER implementation supports uniform fallback).
        self.replay_buffer = PrioritizedReplayBuffer(
            max_size=buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_end=per_beta_end,
            beta_steps=per_beta_steps,
            epsilon=per_epsilon,
            replay_strategy=replay_strategy,
        )

        # Initialize algorithm
        self.policy = None
        self._initialize_policy()
        self._apply_eps_to_policy(initial=True)

    def _apply_eps_to_policy(self, initial: bool = False) -> None:
        """Push the wrapper-tracked epsilon onto the underlying Tianshou policy.

        Tianshou's ``DQNPolicy`` exposes a ``set_eps()`` method but does not
        otherwise drive an exploration schedule on this wrapper's custom replay
        path. We mirror :attr:`_eps_current` (or :attr:`_eps_test` when not in
        training mode) onto ``policy.eps`` so epsilon-greedy actually happens.
        """
        if self.policy is None:
            return
        if self._train_mode:
            target = self._eps_current
        else:
            target = self._eps_test
        setter = getattr(self.policy, "set_eps", None)
        try:
            if callable(setter):
                setter(float(target))
            else:
                self.policy.eps = float(target)
        except Exception as exc:
            if initial:
                logger.debug("could_not_set_initial_eps", error=str(exc))

    def _decay_eps(self) -> None:
        """Apply one tick of multiplicative epsilon decay (training mode only)."""
        if not self._train_mode:
            return
        if self._eps_decay >= 1.0:
            return
        self._eps_current = max(
            self._eps_train_final, self._eps_current * self._eps_decay
        )
        self._apply_eps_to_policy()

    def advance_epsilon(self) -> None:
        """Advance epsilon by one decision tick on demand.

        Exposed as a small public hook so callers that route action selection
        through ``predict_proba`` + weighted sampling can still advance the
        exploration schedule exactly once per real decision.
        """
        self._decay_eps()

    def set_train_mode(self, training: bool) -> None:
        """Toggle between training and evaluation epsilon."""
        self._train_mode = bool(training)
        self._apply_eps_to_policy()

    @property
    def epsilon(self) -> float:
        """Current epsilon-greedy exploration rate (mirrors ``policy.eps``)."""
        return self._eps_current if self._train_mode else self._eps_test

    def _get_policy_optimizers(self) -> Dict[str, Any]:
        """Return the Tianshou policy optimizers exposed by the current wrapper."""
        if self.policy is None:
            return {}

        optimizers: Dict[str, Any] = {}
        for attr_name in self._OPTIMIZER_ATTRS:
            optimizer = getattr(self.policy, attr_name, None)
            if optimizer is None:
                continue
            if callable(getattr(optimizer, "state_dict", None)) and callable(
                getattr(optimizer, "load_state_dict", None)
            ):
                optimizers[attr_name] = optimizer
        return optimizers

    def _get_learning_rates(self) -> Dict[str, float]:
        """Snapshot the current learning rates for all exposed optimizers."""
        learning_rates: Dict[str, float] = {}
        for attr_name, optimizer in self._get_policy_optimizers().items():
            param_groups = getattr(optimizer, "param_groups", None)
            if not param_groups:
                continue
            lr = param_groups[0].get("lr")
            if lr is None:
                continue
            learning_rates[attr_name] = float(lr)
        return learning_rates

    def _set_learning_rates(self, learning_rates: Dict[str, Any]) -> None:
        """Apply learning rates to the matching optimizers when present."""
        optimizers = self._get_policy_optimizers()
        applied_rates: List[float] = []
        for attr_name, raw_lr in learning_rates.items():
            optimizer = optimizers.get(attr_name)
            if optimizer is None:
                continue
            try:
                lr = float(raw_lr)
            except (TypeError, ValueError):
                logger.warning("Skipping invalid learning rate for %s: %r", attr_name, raw_lr)
                continue
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            applied_rates.append(lr)

        if applied_rates:
            self.algorithm_config["lr"] = applied_rates[0]

    def _serialize_replay_value(self, value: Any) -> Any:
        """Return a detached/copy-safe representation for replay payloads."""
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().copy()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _get_replay_buffer_state(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Serialize a deterministic slice of the replay buffer."""
        if self.replay_buffer is None:
            return {"entries": [], "priorities": []}

        entries = list(self.replay_buffer.buffer)
        priorities = np.array(self.replay_buffer.priorities[: len(entries)], copy=True)

        if len(entries) == self.replay_buffer.max_size and len(entries) > 0:
            start_index = int(self.replay_buffer.position)
            entries = entries[start_index:] + entries[:start_index]
            priorities = np.concatenate((priorities[start_index:], priorities[:start_index]))

        if limit is not None:
            limit = max(0, int(limit))
            entries = entries[-limit:]
            priorities = priorities[-limit:]

        return {
            "entries": [
                {
                    key: self._serialize_replay_value(value)
                    for key, value in experience.items()
                }
                for experience in entries
            ],
            "priorities": priorities.astype(np.float64).tolist(),
            "beta": float(getattr(self.replay_buffer, "beta", 0.0)),
            "beta_step_count": int(getattr(self.replay_buffer, "_beta_step_count", 0)),
            "replay_strategy": getattr(self.replay_buffer, "replay_strategy", None),
        }

    def _load_replay_buffer_state(self, replay_state: Dict[str, Any]) -> None:
        """Restore a previously serialized replay-buffer slice when compatible."""
        entries = replay_state.get("entries")
        if not isinstance(entries, list):
            return

        required_keys = {"state", "action", "reward", "next_state", "done"}
        max_size = int(getattr(self.replay_buffer, "max_size", len(entries)))
        capped_entries = entries[-max_size:]

        normalized_entries: List[Dict[str, Any]] = []
        for experience in capped_entries:
            if not isinstance(experience, dict) or not required_keys.issubset(experience):
                raise ValueError("Replay entry is missing required transition fields")
            normalized_entries.append(dict(experience))

        self.replay_buffer.clear()
        for experience in normalized_entries:
            extra_fields = {
                key: value
                for key, value in experience.items()
                if key not in required_keys
            }
            self.replay_buffer.append(
                experience["state"],
                int(experience["action"]),
                float(experience["reward"]),
                experience["next_state"],
                bool(experience["done"]),
                **extra_fields,
            )

        raw_priorities = replay_state.get("priorities")
        if isinstance(raw_priorities, list) and raw_priorities:
            priorities = np.asarray(raw_priorities[-len(normalized_entries):], dtype=np.float64)
            if priorities.shape[0] == len(normalized_entries):
                self.replay_buffer.priorities[: len(normalized_entries)] = priorities
                self.replay_buffer.priorities[len(normalized_entries):] = 0.0

        beta = replay_state.get("beta")
        if beta is not None:
            self.replay_buffer.beta = float(beta)

        beta_step_count = replay_state.get("beta_step_count")
        if beta_step_count is not None:
            self.replay_buffer._beta_step_count = int(beta_step_count)

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
                # The custom replay integration in this wrapper trains on one-step transitions.
                "n_step": 1,
                "target_update_freq": 500,
                # Default exploration schedule: anneal multiplicatively from
                # eps_train -> eps_train_final each select_action_with_mask call.
                "eps_test": 0.05,
                "eps_train": 1.0,
                "eps_train_final": 0.05,
                "eps_decay": 0.995,
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

                # Pull the configured hidden width (the legacy hard-coded
                # 512/256/128 stack is now derived from this base width so the
                # ``dqn_hidden_size`` knob is honored end-to-end).
                base_hidden = max(8, int(self.algorithm_config.get("dqn_hidden_size", 64)))

                # Create Q-network that handles 1D, 2D, and 3D observations
                class AdaptiveQNet(nn.Module):
                    def __init__(self, observation_shape, num_actions, device, hidden_size: int):
                        super().__init__()
                        self.observation_shape = observation_shape
                        # Only treat as spatial if we have 3D observations (C, H, W)
                        # 2D observations are flattened feature vectors, not spatial
                        self.is_spatial = len(observation_shape) == 3
                        h1 = max(8, hidden_size * 4)
                        h2 = max(8, hidden_size * 2)
                        h3 = max(8, hidden_size)

                        if self.is_spatial:
                            # CNN layers for spatial processing (3D observations: C, H, W)
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
                                nn.Linear(self.flattened_size, h1),
                                nn.ReLU(),
                                nn.Linear(h1, h2),
                                nn.ReLU(),
                                nn.Linear(h2, num_actions),
                            )
                        else:
                            # Fully connected layers for 1D and 2D observations
                            # 2D observations are flattened feature vectors
                            input_size = observation_shape[0]
                            self.q_layers = nn.Sequential(
                                nn.Linear(input_size, h1),
                                nn.ReLU(),
                                nn.Linear(h1, h2),
                                nn.ReLU(),
                                nn.Linear(h2, h3),
                                nn.ReLU(),
                                nn.Linear(h3, num_actions),
                            )

                    def forward(self, obs, state=None, info=None):
                        # Handle different observation dimensions
                        if obs.dim() == 1:  # Single 1D observation
                            obs = obs.unsqueeze(0)  # Add batch dimension
                        elif obs.dim() == 2:  # Single 2D observation (batch_size, features) or (features,)
                            if obs.shape[0] == self.observation_shape[0]:
                                # This is a single observation with shape (features,)
                                obs = obs.unsqueeze(0)  # Add batch dimension
                            # Otherwise, it's already batched (batch_size, features)
                        elif obs.dim() == 3:  # Single 3D observation (C, H, W)
                            obs = obs.unsqueeze(0)  # Add batch dimension

                        if self.is_spatial:
                            x = self.conv_layers(obs)
                            q_values = self.q_layers(x)
                        else:
                            q_values = self.q_layers(obs)

                        # Tianshou DQNPolicy expects model(obs, state, info) -> (logits, state).
                        return q_values, state

                # Create adaptive Q-network
                q_net = AdaptiveQNet(
                    self.observation_shape,
                    self.num_actions,
                    self.algorithm_config["device"],
                    hidden_size=base_hidden,
                )

                # Move to device
                q_net.to(self.algorithm_config["device"])

                # Create optimizer
                optim = torch.optim.Adam(
                    q_net.parameters(), lr=self.algorithm_config["lr"]
                )

                # Create policy.  ``target_update_freq`` and ``estimation_step``
                # are first-class DQNPolicy constructor args, so we pass them
                # explicitly rather than letting the EXCLUDED_PARAMS filter
                # drop them.  This makes the Chromosome A
                # ``target_update_freq`` gene control the hard target sync
                # cadence end-to-end.
                target_update_freq = int(
                    self.algorithm_config.get("target_update_freq", 500)
                )
                if target_update_freq <= 0:
                    raise ValueError(
                        "target_update_freq must be a positive integer; got "
                        f"{target_update_freq}."
                    )
                estimation_step = int(self.algorithm_config.get("n_step", 1))

                self.policy = DQNPolicy(
                    model=q_net,
                    optim=optim,
                    action_space=gymnasium.spaces.Discrete(self.num_actions),
                    target_update_freq=target_update_freq,
                    estimation_step=estimation_step,
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

    def _state_to_batched_tensor(self, state: Union[np.ndarray, Any]):
        """Convert an observation to a batched float tensor on the policy device."""
        state_np = batch_observation(state, self.observation_shape)
        device = self.algorithm_config.get("device", "cpu")
        return torch.from_numpy(state_np).float().to(device)

    @staticmethod
    def _tensor_to_1d_numpy(value: Any) -> np.ndarray:
        """Extract a 1D numpy vector from a model output tensor."""
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, torch.Tensor):
            if value.ndim == 2:
                value = value[0]
            return value.detach().cpu().numpy().reshape(-1)
        return np.asarray(value, dtype=np.float64).reshape(-1)

    # Sharpness of the soft logits produced from a SAC/DDPG continuous actor
    # output. Smaller values keep the implied distribution closer to uniform
    # so the chromosome ``action_weights`` multiplier can still meaningfully
    # shift sampling. ``CONTINUOUS_ACTOR_LOGIT_SCALE = 2.0`` corresponds to
    # roughly an exp(2) ≈ 7.4× preference for the actor's chosen bin over a
    # bin one unit away, which keeps the discrete approximation expressive
    # without collapsing to a one-hot.
    CONTINUOUS_ACTOR_LOGIT_SCALE = 2.0

    def _continuous_actor_to_logits(self, action_out: Any) -> np.ndarray:
        """Map a continuous actor output to discrete action logits.

        SAC actors in Tianshou return ``(mean, log_std)``; DDPG actors return
        a deterministic action mean. We treat the first scalar of either form
        as the actor's preferred continuous action in ``[-1, 1]`` (post-tanh)
        and project it onto the discrete action space with a soft, distance-
        based logit profile. This is an approximation — see
        ``docs/devlog/2026-05-21-baldwinian-vs-lamarckian-ab-harness.md`` for
        the rationale and known limitations: continuous-actor algorithms
        cannot natively express a full categorical distribution, so the
        ``policy_probs × action_weights × mask`` composition in
        :meth:`DecisionModule.decide_action` is necessarily less expressive
        for SAC/DDPG than for DQN/PPO/A2C.
        """
        if isinstance(action_out, tuple):
            # SAC: (mean, log_std) — use the deterministic mean for the
            # discretization target. The log_std is intentionally discarded
            # because the discrete softmax already supplies sampling noise.
            action_out = action_out[0]
        if isinstance(action_out, torch.Tensor):
            raw = action_out.detach().cpu().numpy().reshape(-1)
        else:
            raw = np.asarray(action_out, dtype=np.float64).reshape(-1)
        if raw.size == 0:
            return np.zeros(self.num_actions, dtype=np.float64)

        # Map a scalar in [-1, 1] (post-tanh continuous action) to an index
        # in [0, num_actions - 1] as a float so we can compute a distance
        # profile rather than a one-hot.
        scalar = float(np.clip(raw[0], -1.0, 1.0))
        center = (scalar + 1.0) * 0.5 * (self.num_actions - 1)
        bins = np.arange(self.num_actions, dtype=np.float64)
        # Negative-squared-distance gives a smooth Gaussian-shaped preference
        # around ``center``; scale controls how peaked the resulting softmax
        # is. See ``CONTINUOUS_ACTOR_LOGIT_SCALE`` docstring for the choice.
        return -self.CONTINUOUS_ACTOR_LOGIT_SCALE * (bins - center) ** 2

    def _policy_q_values(self, state: Union[np.ndarray, Any]) -> np.ndarray:
        """Return per-action logits or Q-values with shape ``(num_actions,)``."""
        if self.policy is None:
            return np.ones(self.num_actions, dtype=np.float64)

        state_tensor = self._state_to_batched_tensor(state)
        with torch.no_grad():
            if self.algorithm_name in ("PPO", "A2C") and hasattr(self.policy, "actor"):
                logits = self._tensor_to_1d_numpy(self.policy.actor(state_tensor))
            elif self.algorithm_name in ("SAC", "DDPG") and hasattr(self.policy, "actor"):
                logits = self._continuous_actor_to_logits(self.policy.actor(state_tensor))
            elif hasattr(self.policy, "model"):
                logits = self._tensor_to_1d_numpy(self.policy.model(state_tensor))
            else:
                raise RuntimeError(
                    f"No supported policy forward path for algorithm {self.algorithm_name!r}"
                )

        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        if logits.shape[0] != self.num_actions:
            raise ValueError(
                f"Expected {self.num_actions} action scores, got {logits.shape[0]}"
            )
        return logits

    @staticmethod
    def _masked_softmax(logits: np.ndarray, action_mask: Optional[np.ndarray]) -> np.ndarray:
        """Softmax logits with optional invalid-action masking."""
        masked = logits.astype(np.float64, copy=True)
        if action_mask is not None:
            if len(action_mask) != len(masked):
                raise ValueError("action_mask length must match num_actions")
            masked[~action_mask] = -np.inf
        finite = np.isfinite(masked)
        if not finite.any():
            uniform = np.zeros_like(masked, dtype=np.float64)
            if action_mask is not None and action_mask.any():
                uniform[action_mask] = 1.0 / action_mask.sum()
            else:
                uniform[:] = 1.0 / len(uniform)
            return uniform
        shifted = masked - np.max(masked[finite])
        exp_logits = np.zeros_like(shifted, dtype=np.float64)
        exp_logits[finite] = np.exp(shifted[finite])
        total = exp_logits.sum()
        if total <= 0:
            return np.ones(len(masked), dtype=np.float64) / len(masked)
        return exp_logits / total

    def _select_greedy_action(self, logits: np.ndarray, action_mask: Optional[np.ndarray]) -> int:
        """Return the masked argmax action index."""
        masked = logits.astype(np.float64, copy=True)
        if action_mask is not None:
            masked[~action_mask] = -np.inf
        if not np.isfinite(masked).any():
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                if len(valid) > 0:
                    return int(valid[0])
            return 0
        return int(np.argmax(masked))

    def _sample_from_logits(
        self,
        logits: np.ndarray,
        action_mask: Optional[np.ndarray],
    ) -> int:
        """Sample an action from masked logits (used by on-policy algorithms)."""
        probs = self._masked_softmax(logits, action_mask)
        return int(np.random.choice(self.num_actions, p=probs))

    def select_action(self, state: np.ndarray) -> int:
        """Select an action using the Tianshou policy.

        Args:
            state: Current state observation

        Returns:
            Selected action index
        """
        return self.select_action_with_mask(state, action_mask=None)

    def select_action_with_mask(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        *,
        apply_epsilon_decay: bool = True,
    ) -> int:
        """Select an action using the Tianshou policy with action masking support."""
        if self.policy is None:
            raise RuntimeError("Policy not initialized")

        logits = self._policy_q_values(state)
        if action_mask is None:
            valid_mask = np.ones(self.num_actions, dtype=bool)
        else:
            valid_mask = np.asarray(action_mask, dtype=bool)

        if self.algorithm_name == "DQN" and self._train_mode:
            valid_actions = np.where(valid_mask)[0]
            if valid_actions.size == 0:
                action = 0
            elif np.random.random() < self._eps_current:
                action = int(np.random.choice(valid_actions))
            else:
                action = self._select_greedy_action(logits, valid_mask)
        else:
            action = self._sample_from_logits(logits, valid_mask)

        if apply_epsilon_decay:
            self._decay_eps()
        return int(action)

    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        """Predict action probabilities from policy logits/Q-values.

        Returns a length-``num_actions`` softmax over the policy's per-action
        scores. No temperature scaling is applied — the raw logits/Q-values
        are passed straight to ``_masked_softmax``. If a future caller needs
        a hotter or colder distribution, route that through
        :class:`DecisionConfig` so the knob is explicit and persisted in
        experiment metadata.
        """
        if self.policy is None:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=float)

        logits = self._policy_q_values(state)
        return self._masked_softmax(logits, action_mask=None)

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
        self.update_step_count()

    def _estimate_td_errors(
        self,
        result: Dict[str, Any],
        rewards: np.ndarray,
        dones: np.ndarray,
        states: np.ndarray,
        next_states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Estimate TD errors for PER priority updates.

        Preference order:
        1) Use explicit TD errors from policy outputs if present.
        2) For DQN, compute one-step Bellman residual directly from policy networks.
        3) Fallback to absolute centered rewards for algorithms that do not expose
           per-sample TD errors.
        """
        for key in ("td_errors", "td_error", "per_td_errors", "per_td_error"):
            if key not in result:
                continue
            values = np.asarray(result[key], dtype=np.float64).reshape(-1)
            if values.size == rewards.size:
                return np.abs(values)
            if values.size == 1:
                return np.full(rewards.shape[0], float(np.abs(values[0])), dtype=np.float64)

        if self.algorithm_name == "DQN" and self.policy is not None:
            try:
                import torch

                with torch.no_grad():
                    states_tensor = torch.as_tensor(states, dtype=torch.float32)
                    next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32)
                    actions_tensor = torch.as_tensor(actions, dtype=torch.long).reshape(-1)
                    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
                    dones_tensor = torch.as_tensor(dones, dtype=torch.float32).reshape(-1)

                    q_values = self.policy.model(states_tensor)
                    if isinstance(q_values, tuple):
                        q_values = q_values[0]
                    selected_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                    target_model = getattr(self.policy, "model_old", self.policy.model)
                    next_q_values = target_model(next_states_tensor)
                    if isinstance(next_q_values, tuple):
                        next_q_values = next_q_values[0]
                    next_max_q = next_q_values.max(dim=1)[0]

                    gamma = float(self.algorithm_config.get("gamma", 0.99))
                    td_errors = rewards_tensor + (1.0 - dones_tensor) * gamma * next_max_q - selected_q
                    return np.abs(td_errors.cpu().numpy())
            except Exception as exc:
                logger.debug("Failed to compute DQN TD-errors for PER: %s", exc)

        # Fallback for policy-gradient style learners that do not expose per-sample TD errors.
        centered_rewards = rewards - float(np.mean(rewards))
        return np.abs(centered_rewards) + 1e-6

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
            from tianshou.data import Batch

            if len(self.replay_buffer) >= self.batch_size:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                indices = np.asarray(sampled_batch["indices"])
                is_weights = np.asarray(sampled_batch["is_weights"], dtype=np.float32)

                states_np = np.asarray(sampled_batch["state"])
                actions_np = np.asarray(sampled_batch["action"])
                rewards_np = np.asarray(sampled_batch["reward"], dtype=np.float32)
                next_states_np = np.asarray(sampled_batch["next_state"])
                dones_np = np.asarray(sampled_batch["done"], dtype=np.float32)
                device = torch.device(self.algorithm_config.get("device", "cpu"))

                # Convert to tensors
                states = torch.tensor(states_np, dtype=torch.float32, device=device)
                actions = torch.tensor(actions_np, dtype=torch.long, device=device)
                rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)
                dones = torch.tensor(dones_np, dtype=torch.float32, device=device)

                # Use Tianshou Batch so policy.learn can access attributes like batch.info.
                tianshou_batch = Batch(
                    obs=states,
                    act=actions,
                    rew=rewards,
                    obs_next=next_states,
                    done=dones,
                    # Tianshou policies that support PER consume this key to
                    # importance-weight per-sample losses.
                    weight=torch.tensor(is_weights, dtype=torch.float32, device=device),
                    terminated=dones,  # Required by RolloutBatchProtocol
                    truncated=torch.zeros_like(dones),  # Required by RolloutBatchProtocol
                    info=Batch(),
                )

                if self.algorithm_name == "DQN":
                    # DQNPolicy.learn expects precomputed return targets for this
                    # ad-hoc replay path (the standard trainer computes these in process_fn).
                    with torch.no_grad():
                        target_model = getattr(self.policy, "model_old", self.policy.model)
                        next_q_values = target_model(next_states)
                        if isinstance(next_q_values, tuple):
                            next_q_values = next_q_values[0]
                        next_max_q = next_q_values.max(dim=1)[0]
                        gamma = float(self.algorithm_config.get("gamma", 0.99))
                        tianshou_batch.returns = rewards + (1.0 - dones) * gamma * next_max_q

                # Train the policy
                result = self.policy.learn(
                    tianshou_batch, batch_size=self.batch_size, repeat=1  # type: ignore
                )

                # Update PER priorities after optimizer step (uniform sampling ignores them).
                if self.replay_strategy == "prioritized":
                    td_errors = self._estimate_td_errors(
                        result if isinstance(result, dict) else {},
                        rewards=rewards_np,
                        dones=dones_np,
                        states=states_np,
                        next_states=next_states_np,
                        actions=actions_np,
                    )
                    self.replay_buffer.update_priorities(indices, td_errors)
                    self.replay_buffer.update_beta()

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

                # Replay diagnostics for observability/tuning.
                replay_diag = self.replay_buffer.diagnostics()
                metrics.update(
                    {
                        "replay_beta": float(replay_diag["beta"]),
                        "replay_priority_min": float(replay_diag["priority_min"]),
                        "replay_priority_max": float(replay_diag["priority_max"]),
                        "replay_priority_mean": float(replay_diag["priority_mean"]),
                        "replay_is_weight_mean": float(np.mean(is_weights)),
                    }
                )
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

    def get_model_state(
        self,
        include_optimizer_state: bool = False,
        include_replay_buffer: bool = False,
        replay_buffer_limit: Optional[int] = None,
        include_plasticity_state: bool = False,
    ) -> Dict[str, Any]:
        """Get the current model state for saving.

        Optional payloads are excluded by default so the existing
        weights-only/Baldwinian paths keep their current behavior unless an
        inheritance mode explicitly opts into richer module state.
        """
        if self.policy is None:
            return {}

        try:
            state = {
                "policy_state_dict": self.policy.state_dict(),
                "step_count": self.step_count,
                "buffer_size": len(self.replay_buffer),
                "algorithm_name": self.algorithm_name,
                "algorithm_config": self.algorithm_config,
            }
            if include_optimizer_state:
                optimizer_state = {
                    attr_name: optimizer.state_dict()
                    for attr_name, optimizer in self._get_policy_optimizers().items()
                }
                if optimizer_state:
                    state["optimizer_state"] = optimizer_state
            if include_replay_buffer:
                state["replay_buffer_state"] = self._get_replay_buffer_state(
                    limit=replay_buffer_limit
                )
            if include_plasticity_state:
                learning_rates = self._get_learning_rates()
                state["plasticity_state"] = {
                    "epsilon": float(self.epsilon),
                    "eps_current": float(self._eps_current),
                    "eps_test": float(self._eps_test),
                    "train_mode": bool(self._train_mode),
                    "learning_rate": (
                        float(next(iter(learning_rates.values())))
                        if learning_rates
                        else float(self.algorithm_config.get("lr", 0.0))
                    ),
                    "learning_rates": learning_rates,
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

        optimizer_state = state.get("optimizer_state")
        if isinstance(optimizer_state, dict):
            for attr_name, optimizer_snapshot in optimizer_state.items():
                optimizer = self._get_policy_optimizers().get(attr_name)
                if optimizer is None:
                    continue
                try:
                    optimizer.load_state_dict(optimizer_snapshot)
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state for {attr_name}: {e}")

        replay_buffer_state = state.get("replay_buffer_state")
        if isinstance(replay_buffer_state, dict):
            try:
                self._load_replay_buffer_state(replay_buffer_state)
            except Exception as e:
                logger.warning(f"Failed to load replay buffer state: {e}")

        plasticity_state = state.get("plasticity_state")
        if isinstance(plasticity_state, dict):
            if "train_mode" in plasticity_state:
                self._train_mode = bool(plasticity_state["train_mode"])
            if "eps_test" in plasticity_state:
                self._eps_test = float(plasticity_state["eps_test"])
            if "eps_current" in plasticity_state:
                self._eps_current = float(plasticity_state["eps_current"])
            elif "epsilon" in plasticity_state:
                self._eps_current = float(plasticity_state["epsilon"])

            learning_rates = plasticity_state.get("learning_rates")
            if isinstance(learning_rates, dict) and learning_rates:
                self._set_learning_rates(learning_rates)
            elif "learning_rate" in plasticity_state:
                self._set_learning_rates(
                    {
                        attr_name: plasticity_state["learning_rate"]
                        for attr_name in self._get_policy_optimizers()
                    }
                )
            self._apply_eps_to_policy()

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
