"""Adapter for parent/student rollout validation against a sim-like environment.

This module provides :class:`PolicyRolloutAdapter` — a clean injection point for
checkpoint-loaded parent/student Q-networks that evaluates them through the same
episode harness used in production training, rather than the synthetic linear MDP
in :mod:`~farm.core.decision.training.distillation_rollout`.

Key features
------------
- **Env-agnostic** – the adapter depends on a lightweight
  :class:`EpisodeEnvProtocol` interface (reset + step) rather than the full
  AgentFarm :class:`~farm.core.environment.Environment`, so it can be tested with
  minimal mock environments and wired to the real sim when needed.
- **Feature-pipeline injection** – a user-supplied callable maps raw observation
  numpy arrays (as returned by ``env.reset`` / ``env.step``) to the fixed-length
  float32 feature vectors expected by the Q-networks.  The default identity
  pipeline passes observations through unchanged, which is correct when the
  environment already returns properly shaped feature vectors.
- **Greedy-Q rollouts** – both parent and student run deterministic greedy-Q
  policies (``argmax`` over Q-values) so results are directly comparable across
  seeds.
- **Seeded comparison** – episode seeds are derived from a configurable
  ``base_seed``, giving reproducible and fair parent-vs-student comparisons.

Typical usage
-------------
::

    import torch
    from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork
    from farm.core.decision.training.sim_rollout_adapter import (
        PolicyRolloutAdapter,
        SimRolloutConfig,
    )

    parent = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    student = StudentQNetwork(input_dim=8, output_dim=4, parent_hidden_size=64)

    # Load checkpoints
    parent.load_state_dict(torch.load("parent.pt", weights_only=True))
    student.load_state_dict(torch.load("student.pt", weights_only=True))

    config = SimRolloutConfig(
        n_episodes=20,
        max_steps=200,
        base_seed=42,
    )
    adapter = PolicyRolloutAdapter(parent, student, config=config)
    result = adapter.run(env_factory=my_env_factory)
    print(result.passed, result.to_dict())

See also: :mod:`~farm.core.decision.training.distillation_rollout` for fast
seeded-synthetic-MDP smoke tests that do **not** require a real environment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn

from farm.utils.logging import get_logger

logger = get_logger(__name__)

# Episode seed stride — matches the stride used in distillation_rollout for consistency.
# Using a large prime prevents accidental seed aliasing across episodes.
_EPISODE_SEED_STRIDE: int = 1_000_003

# Tolerance for treating a mean return as effectively zero when computing relative drop.
_ZERO_THRESHOLD: float = 1e-8

# ---------------------------------------------------------------------------
# Protocols / interfaces
# ---------------------------------------------------------------------------


class EpisodeEnvProtocol(Protocol):
    """Minimal environment interface required by :class:`PolicyRolloutAdapter`.

    Implementations must support ``reset(seed=…)`` and ``step(action)``.  The
    real AgentFarm :class:`~farm.core.environment.Environment` satisfies this
    interface through the PettingZoo / Gymnasium ``reset``/``step`` signatures.
    """

    def reset(self, *, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and return ``(obs, info)``.

        Parameters
        ----------
        seed:
            Optional integer seed for deterministic resets.

        Returns
        -------
        obs:
            Initial observation (numpy array or compatible type).
        info:
            Auxiliary info dict (may be empty).
        """
        ...

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take *action* and return ``(obs, reward, terminated, truncated, info)``.

        Parameters
        ----------
        action:
            Integer action index.

        Returns
        -------
        obs:
            Next observation.
        reward:
            Scalar reward.
        terminated:
            ``True`` if the episode ended naturally (agent died, goal reached,
            etc.).
        truncated:
            ``True`` if the episode was cut short by a time-limit.
        info:
            Auxiliary info dict (may be empty).
        """
        ...


# A callable that converts a raw env observation to a float32 feature vector.
FeaturePipeline = Callable[[Any], np.ndarray]

# A zero-argument factory that returns a fresh EpisodeEnvProtocol-compatible env.
EnvFactory = Callable[[], EpisodeEnvProtocol]


def _identity_feature_pipeline(obs: Any) -> np.ndarray:
    """Return obs unchanged as a float32 numpy array (default feature pipeline)."""
    return np.asarray(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Configuration / result data classes
# ---------------------------------------------------------------------------


@dataclass
class SimRolloutConfig:
    """Configuration for :class:`PolicyRolloutAdapter` episode rollouts.

    Attributes
    ----------
    n_episodes:
        Number of episodes to run for each policy (parent and student).
        Must be positive.
    max_steps:
        Maximum number of steps per episode before truncation.  Must be
        positive.
    base_seed:
        Base seed used to derive per-episode seeds.  Episode *i* uses seed
        ``base_seed + i * _EPISODE_SEED_STRIDE`` (same stride as
        :func:`~farm.core.decision.training.distillation_rollout.compare_parent_student_rollouts`
        for consistency).
    max_relative_return_drop:
        When set, the rollout passes only if::

            student_mean_return >= parent_mean_return * (1 - max_relative_return_drop)

        for positive parent return, and symmetrically for negative.  Set to
        ``None`` (default) to skip the pass/fail check.
    """

    n_episodes: int = 10
    max_steps: int = 200
    base_seed: int = 42
    max_relative_return_drop: Optional[float] = None


@dataclass
class SimEpisodeStats:
    """Per-episode statistics for one policy in a sim rollout.

    Attributes
    ----------
    returns:
        Episode cumulative rewards.
    step_counts:
        Number of steps taken each episode (includes the terminal step).
    survival_rates:
        Fraction of ``max_steps`` survived per episode (1.0 means survived
        the full horizon without natural termination).
    wall_time_s:
        Total wall-clock time (seconds) spent on rollouts.
    """

    returns: List[float] = field(default_factory=list)
    step_counts: List[int] = field(default_factory=list)
    survival_rates: List[float] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns)) if self.returns else 0.0

    @property
    def mean_steps(self) -> float:
        return float(np.mean(self.step_counts)) if self.step_counts else 0.0

    @property
    def mean_survival_rate(self) -> float:
        return float(np.mean(self.survival_rates)) if self.survival_rates else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_return": self.mean_return,
            "mean_steps": self.mean_steps,
            "mean_survival_rate": self.mean_survival_rate,
            "returns": self.returns,
            "step_counts": self.step_counts,
            "survival_rates": self.survival_rates,
            "wall_time_s": self.wall_time_s,
        }


@dataclass
class SimRolloutResult:
    """Aggregate result for a parent vs student sim rollout comparison.

    Attributes
    ----------
    parent_stats:
        Episode statistics for the parent policy.
    student_stats:
        Episode statistics for the student policy.
    relative_drop:
        Relative return drop ``max(0, 1 - student_mean / parent_mean)`` for
        positive parent return; analogous for negative.  ``None`` when the
        parent mean return is near zero (undefined ratio).
    n_episodes:
        Number of episodes run per policy.
    max_steps:
        Configured episode horizon.
    base_seed:
        Base seed used for episode seeding.
    passed:
        ``True`` / ``False`` when ``max_relative_return_drop`` was set and the
        check ran; ``None`` otherwise.
    """

    parent_stats: SimEpisodeStats
    student_stats: SimEpisodeStats
    relative_drop: Optional[float]
    n_episodes: int
    max_steps: int
    base_seed: int
    passed: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent": self.parent_stats.to_dict(),
            "student": self.student_stats.to_dict(),
            "parent_mean_return": self.parent_stats.mean_return,
            "student_mean_return": self.student_stats.mean_return,
            "relative_drop": self.relative_drop,
            "n_episodes": self.n_episodes,
            "max_steps": self.max_steps,
            "base_seed": self.base_seed,
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------


class PolicyRolloutAdapter:
    """Run parent and student Q-policies through a sim-like environment.

    The adapter evaluates two Q-networks — *parent* (teacher) and *student*
    (distilled) — on identical episode seeds so that differences in episode
    return can be attributed solely to policy quality rather than stochasticity.

    Both networks use a **greedy-Q** policy (argmax over Q-values).

    Parameters
    ----------
    parent:
        Parent (teacher) Q-network.  ``forward(x)`` must accept a ``(1,
        input_dim)`` float32 tensor and return a ``(1, n_actions)`` tensor of
        Q-values.  The network is placed in eval mode during rollouts.
    student:
        Student (distilled) Q-network with the same input/output dimensions as
        *parent*.
    config:
        :class:`SimRolloutConfig` controlling episode count, horizon, seeding,
        and optional pass/fail threshold.
    feature_pipeline:
        Callable that maps a raw environment observation (as returned by
        ``env.reset`` / ``env.step``) to a 1-D float32 numpy array of length
        ``input_dim``.  Defaults to an identity cast (suitable when the env
        already returns properly shaped float32 vectors).
    device:
        PyTorch device for network inference.  Defaults to CPU.

    Example
    -------
    ::

        import torch
        from unittest.mock import MagicMock
        import numpy as np
        from farm.core.decision.base_dqn import BaseQNetwork
        from farm.core.decision.training.sim_rollout_adapter import (
            PolicyRolloutAdapter,
            SimRolloutConfig,
        )

        net = BaseQNetwork(input_dim=4, output_dim=2, hidden_size=16)

        env = MagicMock()
        env.reset.return_value = (np.zeros(4), {})
        env.step.return_value = (np.zeros(4), 1.0, False, True, {})

        cfg = SimRolloutConfig(n_episodes=5, max_steps=10, base_seed=0)
        adapter = PolicyRolloutAdapter(net, net, config=cfg)
        result = adapter.run(env_factory=lambda: env)
        assert result.parent_stats.mean_return == result.student_stats.mean_return
    """

    def __init__(
        self,
        parent: nn.Module,
        student: nn.Module,
        *,
        config: Optional[SimRolloutConfig] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if config is None:
            config = SimRolloutConfig()
        if config.n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {config.n_episodes}")
        if config.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {config.max_steps}")
        if (
            config.max_relative_return_drop is not None
            and not (0.0 <= config.max_relative_return_drop <= 1.0)
        ):
            raise ValueError(
                "max_relative_return_drop must be in [0, 1] when set, "
                f"got {config.max_relative_return_drop}"
            )

        self.parent = parent
        self.student = student
        self.config = config
        self.feature_pipeline: FeaturePipeline = (
            feature_pipeline if feature_pipeline is not None else _identity_feature_pipeline
        )
        self.device = device or torch.device("cpu")
        self.parent.to(self.device)
        self.student.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, env_factory: EnvFactory) -> SimRolloutResult:
        """Run greedy-Q rollouts for parent and student on identical episode seeds.

        A **fresh environment** is created via *env_factory* at the start of
        each episode so that per-episode resets are fully isolated.

        Parameters
        ----------
        env_factory:
            Zero-argument callable that returns a new :class:`EpisodeEnvProtocol`
            compatible environment.  Called once per episode for each policy
            (``2 * n_episodes`` total calls).

        Returns
        -------
        SimRolloutResult
            Aggregate statistics and pass/fail result.
        """
        cfg = self.config
        parent_stats = self._rollout_policy(self.parent, env_factory, label="parent")
        student_stats = self._rollout_policy(self.student, env_factory, label="student")

        p_mean = parent_stats.mean_return
        s_mean = student_stats.mean_return
        rel = _relative_return_drop(p_mean, s_mean)

        passed: Optional[bool] = None
        if cfg.max_relative_return_drop is not None:
            passed = _rollout_passed(p_mean, s_mean, cfg.max_relative_return_drop)

        result = SimRolloutResult(
            parent_stats=parent_stats,
            student_stats=student_stats,
            relative_drop=rel,
            n_episodes=cfg.n_episodes,
            max_steps=cfg.max_steps,
            base_seed=cfg.base_seed,
            passed=passed,
        )
        logger.info(
            "sim_rollout_complete",
            n_episodes=cfg.n_episodes,
            parent_mean_return=round(p_mean, 6),
            student_mean_return=round(s_mean, 6),
            relative_drop=round(rel, 6) if rel is not None else None,
            passed=passed,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rollout_policy(
        self,
        model: nn.Module,
        env_factory: EnvFactory,
        *,
        label: str,
    ) -> SimEpisodeStats:
        """Collect *n_episodes* greedy-Q episodes for *model*."""
        cfg = self.config
        stats = SimEpisodeStats()
        was_training = model.training
        try:
            model.eval()
            t0 = time.perf_counter()
            for i in range(cfg.n_episodes):
                ep_seed = int(cfg.base_seed + i * _EPISODE_SEED_STRIDE)
                ep_return, ep_steps = self._run_episode(
                    model, env_factory(), episode_seed=ep_seed
                )
                stats.returns.append(ep_return)
                stats.step_counts.append(ep_steps)
                stats.survival_rates.append(ep_steps / cfg.max_steps)
            stats.wall_time_s = time.perf_counter() - t0
        finally:
            model.train(was_training)

        logger.info(
            "sim_rollout_policy_done",
            policy=label,
            n_episodes=cfg.n_episodes,
            mean_return=round(stats.mean_return, 6),
            mean_steps=round(stats.mean_steps, 2),
            wall_time_s=round(stats.wall_time_s, 3),
        )
        return stats

    def _run_episode(
        self,
        model: nn.Module,
        env: EpisodeEnvProtocol,
        *,
        episode_seed: int,
    ) -> Tuple[float, int]:
        """Run a single greedy-Q episode; return ``(total_return, steps_taken)``."""
        obs, _ = env.reset(seed=episode_seed)
        total = 0.0
        done = False
        steps = 0
        max_steps = self.config.max_steps

        while not done and steps < max_steps:
            features = self.feature_pipeline(obs)
            x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q = model(x)
                action = int(q.argmax(dim=-1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated

        return total, steps


# ---------------------------------------------------------------------------
# Helpers (mirror distillation_rollout conventions)
# ---------------------------------------------------------------------------


def _relative_return_drop(parent_mean: float, student_mean: float) -> Optional[float]:
    """Scalar summary of return drop (positive when student is worse)."""
    if parent_mean > _ZERO_THRESHOLD:
        return float(max(0.0, 1.0 - student_mean / parent_mean))
    if parent_mean < -_ZERO_THRESHOLD:
        return float(max(0.0, (parent_mean - student_mean) / abs(parent_mean)))
    return None


def _rollout_passed(
    parent_mean: float,
    student_mean: float,
    max_relative_return_drop: float,
) -> bool:
    """Return ``True`` if student return is within the allowed drop vs parent."""
    if parent_mean > _ZERO_THRESHOLD:
        return student_mean >= parent_mean * (1.0 - max_relative_return_drop)
    if parent_mean < -_ZERO_THRESHOLD:
        return student_mean >= parent_mean * (1.0 + max_relative_return_drop)
    return True
