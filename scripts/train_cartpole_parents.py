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
import os
import sys
from typing import Optional

import torch

# Allow running directly from repo root without installing the package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scripts_dir = os.path.join(_repo_root, "scripts")
for _p in (_repo_root, _scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cartpole_dqn_training import train_cartpole_parent  # noqa: E402


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

    result = train_cartpole_parent(
        label=label,
        episodes=episodes,
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
        output_dir=output_dir,
        log_every=log_every,
        device=device,
        env_reset_seed=None,
    )

    print(f"\n  ✓ Checkpoint   : {result.checkpoint_path}")
    print(f"  ✓ Metadata     : {result.metadata_path}")
    print(f"  Mean reward (last 50 eps): {result.mean_reward_last_50:.2f}")
    print(
        f"  ✓ Replay states: {result.replay_states_path}  "
        f"shape={result.replay_states_shape}"
    )


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
