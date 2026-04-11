"""Shared CartPole-v1 DQN parent training for cartpole pipeline scripts."""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import random
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
except ImportError as exc:
    raise SystemExit("gymnasium is required: pip install gymnasium") from exc

from farm.core.decision.base_dqn import BaseQNetwork


@dataclasses.dataclass(frozen=True)
class CartPoleParentTrainResult:
    """Paths and summary stats after training one parent."""

    checkpoint_path: str
    metadata_path: str
    replay_states_path: str
    mean_reward_last_50: float
    replay_states_shape: Tuple[int, ...]


class CartPoleDQN:
    """Minimal DQN trainer wired to a ``BaseQNetwork`` for CartPole-v1.

    Implements experience replay, epsilon-greedy exploration, Double Q-Learning
    updates, and soft target-network updates.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        tau: float,
        memory_size: int,
        batch_size: int,
        seed: Optional[int],
        device: torch.device,
    ) -> None:
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.q_net = BaseQNetwork(input_dim, output_dim, hidden_size).to(device)
        self.target_net = BaseQNetwork(input_dim, output_dim, hidden_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.memory: Deque[Tuple] = collections.deque(maxlen=memory_size)
        self._states_seen: List[np.ndarray] = []

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        s = torch.from_numpy(state).float().to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(s)
        self.q_net.train()
        return int(q.argmax().item())

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))
        self._states_seen.append(state.astype("float32"))

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(self.device)
        actions = torch.tensor([b[1] for b in batch], device=self.device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        self.q_net.eval()
        current_q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        self.q_net.train()

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        for tp, lp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def replay_states(self) -> np.ndarray:
        if not self._states_seen:
            return np.empty((0, self.input_dim), dtype="float32")
        return np.stack(self._states_seen, axis=0).astype("float32")


def train_cartpole_parent(
    label: str,
    episodes: int,
    hidden_size: int,
    lr: float,
    gamma: float,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay: float,
    tau: float,
    memory_size: int,
    batch_size: int,
    seed: Optional[int],
    output_dir: str,
    log_every: int,
    device: torch.device,
    env_reset_seed: Optional[int] = None,
) -> CartPoleParentTrainResult:
    """Train one CartPole-v1 parent, write checkpoint/metadata/replay states."""
    env = gym.make("CartPole-v1")
    input_dim = int(env.observation_space.shape[0])
    output_dim = int(env.action_space.n)

    agent = CartPoleDQN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=hidden_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        tau=tau,
        memory_size=memory_size,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    episode_rewards: List[float] = []
    recent: Deque[float] = collections.deque(maxlen=100)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=env_reset_seed)
        state = np.array(obs, dtype="float32")
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            obs2, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(obs2, dtype="float32")
            done = terminated or truncated
            agent.store(state, action, float(reward), next_state, done)
            agent.train_step()
            state = next_state
            total_reward += float(reward)
        episode_rewards.append(total_reward)
        recent.append(total_reward)
        if ep % log_every == 0 or ep == episodes:
            print(
                f"  ep {ep:>5}/{episodes}  reward={total_reward:6.1f}"
                f"  mean100={np.mean(recent):6.2f}  ε={agent.epsilon:.4f}"
            )

    env.close()

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"parent_{label}.pt")
    agent.q_net.eval()
    torch.save(agent.q_net.state_dict(), ckpt_path)

    mean_last50 = float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0
    meta = {
        "label": label,
        "env": "CartPole-v1",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_size": hidden_size,
        "episodes_trained": episodes,
        "seed": seed,
        "final_epsilon": round(agent.epsilon, 6),
        "mean_reward_last_50_episodes": round(mean_last50, 4),
        "episode_rewards": [round(r, 4) for r in episode_rewards],
    }
    meta_path = ckpt_path + ".json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    states_path = os.path.join(output_dir, "replay_states.npy")
    replay = agent.replay_states()
    np.save(states_path, replay)
    replay_shape = tuple(int(x) for x in replay.shape)

    return CartPoleParentTrainResult(
        checkpoint_path=ckpt_path,
        metadata_path=meta_path,
        replay_states_path=states_path,
        mean_reward_last_50=mean_last50,
        replay_states_shape=replay_shape,
    )
