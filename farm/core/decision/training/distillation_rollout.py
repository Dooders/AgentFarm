"""Seeded synthetic MDP rollouts for parent vs student Q-policy comparison.

This module supports GitHub issue #597-style *online* checks without wiring
checkpoints into the full AgentFarm simulation.  Dynamics and rewards are
deterministic given a base seed; returns compare greedy-Q policies from parent
and student networks on the **same** trajectories only in the sense of same
MDP and episode seeds — visitation differs if policies disagree.

For evaluation against real simulation behaviour, use replay / feature states
from production training and/or integrate :class:`~farm.core.decision.training.collector.ExperienceCollector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class SeededLinearMDP:
    """Linear-Gaussian-style MDP with fixed random dynamics (seeded at construction)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        base_seed: int,
        max_steps: int,
    ) -> None:
        if obs_dim <= 0 or n_actions <= 0:
            raise ValueError("obs_dim and n_actions must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        rng = np.random.default_rng(base_seed)
        scale = 0.12
        self._w = (rng.standard_normal((n_actions, obs_dim, obs_dim)) * scale).astype(
            np.float32
        )
        self._b = (rng.standard_normal((n_actions, obs_dim)) * scale).astype(np.float32)
        self._rw = (rng.standard_normal(obs_dim) * scale).astype(np.float32)
        self._ra = (rng.standard_normal(n_actions) * scale).astype(np.float32)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.max_steps = max_steps
        self._state = np.zeros(obs_dim, dtype=np.float32)
        self._t = 0

    def reset(self, episode_seed: int) -> np.ndarray:
        ep_rng = np.random.default_rng(episode_seed)
        self._state = ep_rng.standard_normal(self.obs_dim).astype(np.float32)
        self._t = 0
        return self._state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool]:
        a = int(action)
        if not (0 <= a < self.n_actions):
            raise ValueError(
                f"action must be in [0, {self.n_actions}), got {action!r}"
            )
        next_s = self._w[a] @ self._state + self._b[a]
        reward = float(np.dot(self._rw, self._state) + float(self._ra[a]))
        self._state = next_s.astype(np.float32)
        self._t += 1
        truncated = self._t >= self.max_steps
        return self._state.copy(), reward, False, truncated


def _episode_return(
    model: nn.Module,
    env: SeededLinearMDP,
    *,
    episode_seed: int,
    device: torch.device,
) -> float:
    was_training = model.training
    try:
        model.eval()
        obs = env.reset(episode_seed)
        total = 0.0
        done = False
        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = model(x)
                act = int(q.argmax(dim=-1).item())
            obs, reward, terminated, truncated = env.step(act)
            total += reward
            done = terminated or truncated
        return float(total)
    finally:
        model.train(was_training)


def _assert_models_match_action_space(
    parent: nn.Module,
    student: nn.Module,
    *,
    obs_dim: int,
    n_actions: int,
    device: torch.device,
) -> None:
    """Fail fast if Q heads do not emit ``n_actions`` logits."""
    was_p = parent.training
    was_s = student.training
    try:
        parent.eval()
        student.eval()
        x = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
        with torch.no_grad():
            pq = parent(x)
            sq = student(x)
    finally:
        parent.train(was_p)
        student.train(was_s)

    def _out_dim(name: str, q: torch.Tensor) -> None:
        if q.dim() != 2 or q.size(0) != 1:
            raise ValueError(
                f"{name} Q output must be 2D with batch size 1 for probe, "
                f"got shape {tuple(q.shape)}"
            )
        if int(q.size(-1)) != n_actions:
            raise ValueError(
                f"{name} Q output dim {int(q.size(-1))} does not match n_actions={n_actions}"
            )

    _out_dim("parent", pq)
    _out_dim("student", sq)


def _rollout_passed(
    parent_mean: float,
    student_mean: float,
    max_relative_return_drop: float,
) -> bool:
    """True if student mean return is within allowed drop vs parent."""
    if parent_mean > 1e-8:
        return student_mean >= parent_mean * (1.0 - max_relative_return_drop)
    if parent_mean < -1e-8:
        return student_mean >= parent_mean * (1.0 + max_relative_return_drop)
    return True


def relative_return_drop(parent_mean: float, student_mean: float) -> Optional[float]:
    """Scalar summary: positive when student is worse than parent (typical cases)."""
    if parent_mean > 1e-8:
        return float(max(0.0, 1.0 - student_mean / parent_mean))
    if parent_mean < -1e-8:
        return float(max(0.0, (parent_mean - student_mean) / abs(parent_mean)))
    return None


@dataclass(frozen=True)
class RolloutComparisonResult:
    parent_mean_return: float
    student_mean_return: float
    relative_drop: Optional[float]
    n_episodes: int
    max_steps: int
    base_seed: int
    passed: Optional[bool]

    def to_dict(self) -> dict:
        return {
            "parent_mean_return": self.parent_mean_return,
            "student_mean_return": self.student_mean_return,
            "relative_drop": self.relative_drop,
            "n_episodes": self.n_episodes,
            "max_steps": self.max_steps,
            "base_seed": self.base_seed,
            "passed": self.passed,
        }


def compare_parent_student_rollouts(
    parent: nn.Module,
    student: nn.Module,
    *,
    obs_dim: int,
    n_actions: int,
    base_seed: int,
    n_episodes: int,
    max_steps: int,
    device: torch.device,
    max_relative_return_drop: Optional[float] = None,
) -> RolloutComparisonResult:
    """Run greedy-Q episodes for parent and student on the same seeded MDP.

    Episode seeds are ``base_seed + i * 1_000_003`` for ``i`` in ``range(n_episodes)``.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    _assert_models_match_action_space(
        parent,
        student,
        obs_dim=obs_dim,
        n_actions=n_actions,
        device=device,
    )
    env = SeededLinearMDP(
        obs_dim,
        n_actions,
        base_seed=base_seed,
        max_steps=max_steps,
    )
    parent_returns: list[float] = []
    student_returns: list[float] = []
    for i in range(n_episodes):
        ep_seed = int(base_seed + i * 1_000_003)
        parent_returns.append(_episode_return(parent, env, episode_seed=ep_seed, device=device))
        student_returns.append(_episode_return(student, env, episode_seed=ep_seed, device=device))
    p_mean = float(np.mean(parent_returns))
    s_mean = float(np.mean(student_returns))
    rel = relative_return_drop(p_mean, s_mean)
    passed: Optional[bool]
    if max_relative_return_drop is None:
        passed = None
    else:
        passed = _rollout_passed(p_mean, s_mean, max_relative_return_drop)
    return RolloutComparisonResult(
        parent_mean_return=p_mean,
        student_mean_return=s_mean,
        relative_drop=rel,
        n_episodes=n_episodes,
        max_steps=max_steps,
        base_seed=base_seed,
        passed=passed,
    )
