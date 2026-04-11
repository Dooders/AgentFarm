#!/usr/bin/env python3
"""Train two parent BaseQNetwork models on the CartPole-v1 task.

Each parent is trained independently via DQN with epsilon-greedy exploration.
Their ``q_network`` state dicts are saved as ``parent_A.pt`` and ``parent_B.pt``
(plus companion metadata JSON files) in the output directory.

CartPole-v1 state space
-----------------------
4-dimensional observation: cart position, cart velocity, pole angle, pole
angular velocity.  Two discrete actions: push left (0) or push right (1).

Episode ends when the pole tilts beyond ±12°, the cart moves more than ±2.4
units from the centre, or after 500 timesteps.  A reward of +1 is given each
step the pole stays upright.

How to run
----------
::

    # Train both parents with defaults (200 episodes each)
    python scripts/train_cartpole_parents.py

    # Custom run
    python scripts/train_cartpole_parents.py \\
        --episodes 500 \\
        --hidden-size 64 \\
        --seed-a 1 --seed-b 2 \\
        --output-dir checkpoints/cartpole

    # Train only one parent
    python scripts/train_cartpole_parents.py --pair A --episodes 300

Outputs
-------
``<output-dir>/parent_A.pt``
    ``BaseQNetwork`` state dict saved with ``torch.save(model.state_dict(), …)``.
``<output-dir>/parent_A.pt.json``
    Companion metadata: input/output dims, hidden size, seed, final epsilon,
    mean reward of the last 50 episodes.
``<output-dir>/parent_B.pt``  (and ``.pt.json``) — same for parent B.
``<output-dir>/replay_states.npy``
    Concatenated experience states (float32, shape ``(N, 4)``) collected during
    the *last* training run (parent B, or parent A if only pair A is trained).
    Useful as a real-distribution state buffer for downstream pipeline stages.

The script also writes per-episode rewards to stdout so progress can be
monitored in real time.

Architecture
------------
Both parents share the same architecture (``BaseQNetwork`` with configurable
``hidden_size``), but are trained independently from different random seeds.
This intentional diversity means the two parents will have learnt slightly
different policies, providing meaningful signal for the crossover stage.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow running directly from repo root without installing the package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import gymnasium as gym
except ImportError as exc:
    raise SystemExit(
        "gymnasium is required: pip install gymnasium"
    ) from exc

from farm.core.decision.base_dqn import BaseQNetwork  # noqa: E402

# CartPole-v1 dimensions used as fallback (inferred at runtime from environment).
_CARTPOLE_INPUT_DIM = 4
_CARTPOLE_OUTPUT_DIM = 2


# ---------------------------------------------------------------------------
# DQN trainer (self-contained, no SimulationDatabase dependency)
# ---------------------------------------------------------------------------


class _CartPoleDQN:
    """Minimal DQN trainer wired to a ``BaseQNetwork`` for CartPole-v1.

    Implements:
    - Experience replay buffer
    - Epsilon-greedy exploration with linear/exponential decay
    - Double Q-Learning updates
    - Soft target-network updates
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

        # Collected states for replay buffer export
        self._states_seen: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # Soft target update
        for tp, lp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def replay_states(self) -> np.ndarray:
        """Return all states seen during training as float32 (N, input_dim)."""
        if not self._states_seen:
            return np.empty((0, 4), dtype="float32")
        return np.stack(self._states_seen, axis=0).astype("float32")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_one_parent(
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
) -> None:
    """Run a full DQN training loop for one parent and save the checkpoint."""
    print(f"\n{'=' * 60}")
    print(f"Training parent_{label}  (CartPole-v1, {episodes} episodes)")
    if seed is not None:
        print(f"  Seed       : {seed}")
    print(f"  Hidden     : {hidden_size}")
    print(f"  lr         : {lr}  gamma={gamma}  eps_decay={epsilon_decay}")
    print(f"{'=' * 60}")

    env = gym.make("CartPole-v1")
    input_dim = int(env.observation_space.shape[0])   # _CARTPOLE_INPUT_DIM = 4
    output_dim = int(env.action_space.n)               # _CARTPOLE_OUTPUT_DIM = 2

    agent = _CartPoleDQN(
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
        obs, _ = env.reset(seed=None)
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
            mean100 = float(np.mean(recent))
            print(
                f"  ep {ep:>5}/{episodes}  reward={total_reward:6.1f}"
                f"  mean100={mean100:6.2f}  ε={agent.epsilon:.4f}"
            )

    env.close()

    # Save state dict checkpoint
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"parent_{label}.pt")
    agent.q_net.eval()
    torch.save(agent.q_net.state_dict(), ckpt_path)

    # Companion metadata
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

    print(f"\n  ✓ Checkpoint   : {ckpt_path}")
    print(f"  ✓ Metadata     : {meta_path}")
    print(f"  Mean reward (last 50 eps): {mean_last50:.2f}")

    # Save replay states (overwritten each time – caller picks up the last one)
    states_path = os.path.join(output_dir, "replay_states.npy")
    replay = agent.replay_states()
    np.save(states_path, replay)
    print(f"  ✓ Replay states: {states_path}  shape={replay.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train two parent BaseQNetwork models on CartPole-v1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pair",
        choices=["A", "B", "both"],
        default="both",
        help="Which parent(s) to train.",
    )
    # Architecture
    p.add_argument("--hidden-size", type=int, default=64, help="Hidden layer width.")
    # Training
    p.add_argument("--episodes", type=int, default=200, help="Training episodes per parent.")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon.")
    p.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon.")
    p.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Per-step epsilon decay factor.",
    )
    p.add_argument("--tau", type=float, default=0.005, help="Soft target-update rate.")
    p.add_argument("--memory-size", type=int, default=10000, help="Replay buffer capacity.")
    p.add_argument("--batch-size", type=int, default=64, help="Training mini-batch size.")
    # Seeds (each parent gets its own seed so policies diverge)
    p.add_argument("--seed-a", type=int, default=42, help="RNG seed for parent A.")
    p.add_argument("--seed-b", type=int, default=99, help="RNG seed for parent B.")
    # Output
    p.add_argument(
        "--output-dir",
        default="checkpoints/cartpole",
        help="Directory to write parent checkpoints.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress every N episodes.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cpu")

    common = dict(
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        tau=args.tau,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        log_every=args.log_every,
        device=device,
    )

    if args.pair in ("A", "both"):
        _train_one_parent(
            label="A",
            episodes=args.episodes,
            seed=args.seed_a,
            **common,
        )

    if args.pair in ("B", "both"):
        _train_one_parent(
            label="B",
            episodes=args.episodes,
            seed=args.seed_b,
            **common,
        )

    print("\nDone.  Parent checkpoints written to:", args.output_dir)


if __name__ == "__main__":
    main()
