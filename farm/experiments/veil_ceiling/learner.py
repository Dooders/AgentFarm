"""Compact per-agent Q-learner used by the veil-ceiling agents.

A two-layer tanh MLP maps the observation vector to one Q-value per action
and is trained online by one-step Q-learning on a small replay buffer. The
class exposes the ``get_model_state`` / ``load_model_state`` / ``policy``
surface expected by :func:`farm.core.policy_inheritance.apply_lamarckian_policy_warmstart`,
so Lamarckian offspring are warm-started through the same transfer helper
used by the inheritance-ladder experiments.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from farm.experiments.veil_ceiling.config import LearnerConfig


class MLPQPolicy:
    """Weights of the Q-network with a ``state_dict`` interface.

    ``Q(x) = tanh(x W1 + b1) W2 + x Ws + b2``. The linear skip path lets the
    network represent payoff differences that are already linear in the
    observation (yields, cue) without waiting for the hidden layer to align.
    ``b2`` starts at ``optimistic_init`` so every action is tried until its
    value is learned.
    """

    KEYS = ("w1", "b1", "w2", "ws", "b2")

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        hidden: int,
        rng: np.random.Generator,
        optimistic_init: float = 0.0,
    ) -> None:
        scale1 = 1.0 / np.sqrt(n_features)
        scale2 = 1.0 / np.sqrt(hidden)
        self.w1 = rng.normal(0.0, scale1, size=(n_features, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, scale2, size=(hidden, n_actions))
        self.ws = np.zeros((n_features, n_actions))
        self.b2 = np.full(n_actions, float(optimistic_init))

    def state_dict(self) -> dict[str, np.ndarray]:
        return {key: getattr(self, key).copy() for key in self.KEYS}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for key in self.KEYS:
            value = np.asarray(state[key], dtype=float)
            if value.shape != getattr(self, key).shape:
                raise ValueError(f"shape mismatch for {key}: {value.shape} vs {getattr(self, key).shape}")
            setattr(self, key, value.copy())

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.w1 + self.b1)
        return h, h @ self.w2 + x @ self.ws + self.b2


class MLPQLearner:
    """Online Q-learning with a tiny replay buffer."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        cfg: LearnerConfig,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.n_features = n_features
        self.n_actions = n_actions
        self._rng = rng
        self.policy = MLPQPolicy(n_features, n_actions, cfg.hidden_size, rng, cfg.optimistic_init)
        self._target = MLPQPolicy(n_features, n_actions, cfg.hidden_size, rng, cfg.optimistic_init)
        self._target.load_state_dict(self.policy.state_dict())
        self._updates = 0
        self.step_count = 0
        self.learning_enabled = True
        self._buf_s = np.zeros((cfg.replay_size, n_features))
        self._buf_a = np.zeros(cfg.replay_size, dtype=np.int64)
        self._buf_r = np.zeros(cfg.replay_size)
        self._buf_s2 = np.zeros((cfg.replay_size, n_features))
        self._buf_done = np.zeros(cfg.replay_size)
        self._buf_len = 0
        self._buf_pos = 0

    # ── acting ────────────────────────────────────────────────────────────
    def epsilon(self) -> float:
        cfg = self.cfg
        frac = min(1.0, self.step_count / max(1, cfg.epsilon_decay_ticks))
        return cfg.epsilon_start + (cfg.epsilon_min - cfg.epsilon_start) * frac

    def q_values(self, features: np.ndarray) -> np.ndarray:
        _, q = self.policy.forward(features)
        return q

    def select_action(self, features: np.ndarray) -> int:
        self.step_count += 1
        if self._rng.random() < self.epsilon():
            return int(self._rng.integers(self.n_actions))
        q = self.q_values(features)
        best = np.flatnonzero(q >= q.max() - 1e-12)
        if best.size == 1:
            return int(best[0])
        return int(self._rng.choice(best))

    # ── learning ──────────────────────────────────────────────────────────
    def observe(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        if not self.learning_enabled:
            return
        i = self._buf_pos
        self._buf_s[i] = s
        self._buf_a[i] = a
        self._buf_r[i] = r * self.cfg.reward_scale
        self._buf_s2[i] = s2
        self._buf_done[i] = 1.0 if done else 0.0
        self._buf_pos = (i + 1) % self.cfg.replay_size
        self._buf_len = min(self._buf_len + 1, self.cfg.replay_size)
        for _ in range(self.cfg.train_steps_per_observation):
            self._train_step()

    def _train_step(self) -> None:
        cfg = self.cfg
        if self._buf_len < cfg.batch_size:
            return
        idx = self._rng.integers(self._buf_len, size=cfg.batch_size)
        s, a, r = self._buf_s[idx], self._buf_a[idx], self._buf_r[idx]
        s2, done = self._buf_s2[idx], self._buf_done[idx]
        p = self.policy
        _, q_next = self._target.forward(s2)
        target = r + cfg.gamma * (1.0 - done) * q_next.max(axis=1)
        h, q = p.forward(s)
        rows = np.arange(cfg.batch_size)
        err = np.zeros_like(q)
        err[rows, a] = (q[rows, a] - target) / cfg.batch_size
        grad_w2 = h.T @ err
        grad_ws = s.T @ err
        grad_b2 = err.sum(axis=0)
        dh = (err @ p.w2.T) * (1.0 - h * h)
        grad_w1 = s.T @ dh
        grad_b1 = dh.sum(axis=0)
        lr = cfg.learning_rate
        p.w2 -= lr * grad_w2
        p.ws -= lr * grad_ws
        p.b2 -= lr * grad_b2
        p.w1 -= lr * grad_w1
        p.b1 -= lr * grad_b1
        self._updates += 1
        if self._updates % cfg.target_sync_steps == 0:
            self._target.load_state_dict(p.state_dict())

    # ── inheritance surface (mirrors farm.core.decision algorithms) ───────
    def get_model_state(self) -> dict[str, Any]:
        return {"policy_state_dict": self.policy.state_dict(), "step_count": self.step_count}

    def load_model_state(self, state: dict[str, Any]) -> None:
        policy_state: dict[str, np.ndarray] | None = state.get("policy_state_dict")
        if policy_state:
            self.policy.load_state_dict(policy_state)
            self._target.load_state_dict(policy_state)
