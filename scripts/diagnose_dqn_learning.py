"""DQN learning diagnostic.

Runs a real AgentFarm simulation with the default configuration, instruments
every learning agent's ``DecisionModule`` (and the underlying Tianshou DQN
policy) to record:

* every ``store_experience`` call (replay-buffer fill rate per agent)
* every ``train_if_ready`` outcome (whether a gradient step was taken)
* every ``policy.learn`` call (independent confirmation that gradient
  steps actually run end-to-end)
* a tracked Q-network parameter snapshot so we can verify that weights
  *actually move* across training (not just that ``learn`` returned)
* per-agent lifespan in steps

Then prints a per-agent table plus aggregate diagnostics so a human can see
whether agents are learning at all and, if not, why.

Run with ``PYTHONHASHSEED=0 python scripts/diagnose_dqn_learning.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch

from farm.config import SimulationConfig
from farm.core.decision.algorithms.tianshou import TianshouWrapper
from farm.core.decision.decision import DecisionModule
from farm.core.simulation import run_simulation


# ---------------------------------------------------------------------------
# Stats container
# ---------------------------------------------------------------------------


class AgentStats:
    """Per-agent learning instrumentation counters."""

    __slots__ = (
        "agent_id",
        "store_calls",
        "train_ready_true",
        "train_ready_false",
        "policy_learn_calls",
        "policy_learn_succeeded",
        "first_seen_step",
        "last_seen_step",
        "initial_param_sample",
        "final_param_sample",
        "param_change_l2",
        "buffer_size_at_end",
        "epsilon_at_end",
    )

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.store_calls = 0
        self.train_ready_true = 0
        self.train_ready_false = 0
        self.policy_learn_calls = 0
        self.policy_learn_succeeded = 0
        self.first_seen_step: Optional[int] = None
        self.last_seen_step: Optional[int] = None
        self.initial_param_sample: Optional[np.ndarray] = None
        self.final_param_sample: Optional[np.ndarray] = None
        self.param_change_l2: float = 0.0
        self.buffer_size_at_end: int = 0
        self.epsilon_at_end: Optional[float] = None


_STATS: Dict[str, AgentStats] = defaultdict(lambda: AgentStats("?"))


def _stats_for(agent_id: str) -> AgentStats:
    s = _STATS.get(agent_id)
    if s is None:
        s = AgentStats(agent_id)
        _STATS[agent_id] = s
    return s


def _sample_policy_params(policy: Any) -> Optional[np.ndarray]:
    """Return a flat numpy snapshot of the first parameter tensor we find."""
    model = getattr(policy, "model", None)
    if model is None:
        return None
    for p in model.parameters():
        return p.detach().flatten().cpu().numpy().copy()
    return None


# ---------------------------------------------------------------------------
# Monkey-patches that instrument the DecisionModule + TianshouWrapper
# ---------------------------------------------------------------------------


def install_instrumentation() -> None:
    """Wrap key methods to count calls without changing behavior."""

    orig_store = TianshouWrapper.store_experience
    orig_should_train = TianshouWrapper.should_train
    orig_train_on_batch = TianshouWrapper.train_on_batch
    orig_decision_train_if_ready = DecisionModule.train_if_ready

    def patched_store(self, state, action, reward, next_state, done, **kwargs):
        agent_id = getattr(self, "_diag_agent_id", "?")
        stats = _stats_for(agent_id)
        stats.store_calls += 1
        # Take an initial parameter sample on the very first store so that
        # comparisons later show real movement (or the absence of it).
        if stats.initial_param_sample is None and self.policy is not None:
            sample = _sample_policy_params(self.policy)
            if sample is not None:
                stats.initial_param_sample = sample
        return orig_store(self, state, action, reward, next_state, done, **kwargs)

    def patched_should_train(self) -> bool:
        return orig_should_train(self)

    def patched_train_on_batch(self, batch, **kwargs):
        agent_id = getattr(self, "_diag_agent_id", "?")
        stats = _stats_for(agent_id)
        stats.policy_learn_calls += 1
        result = orig_train_on_batch(self, batch, **kwargs)
        # Treat any non-None metrics dict that has a finite loss as a success.
        if isinstance(result, dict):
            stats.policy_learn_succeeded += 1
        return result

    def patched_decision_train_if_ready(self) -> bool:
        agent_id = getattr(self, "agent_id", "?")
        stats = _stats_for(agent_id)
        ready = self._should_train_algorithm() if self.algorithm is not None else False
        if not ready:
            stats.train_ready_false += 1
            return False
        stats.train_ready_true += 1
        return orig_decision_train_if_ready(self)

    TianshouWrapper.store_experience = patched_store  # type: ignore[assignment]
    TianshouWrapper.should_train = patched_should_train  # type: ignore[assignment]
    TianshouWrapper.train_on_batch = patched_train_on_batch  # type: ignore[assignment]
    DecisionModule.train_if_ready = patched_decision_train_if_ready  # type: ignore[assignment]

    # Tag the wrapper with the agent id so the patched methods can attribute
    # their counts. We do this by patching DecisionModule.__init__ so that
    # right after the algorithm is constructed we stash ``_diag_agent_id``
    # on it.
    orig_dm_init = DecisionModule.__init__

    def patched_dm_init(self, agent, action_space, observation_space, *args, **kwargs):
        orig_dm_init(self, agent, action_space, observation_space, *args, **kwargs)
        if self.algorithm is not None:
            self.algorithm._diag_agent_id = self.agent_id  # type: ignore[attr-defined]
        # Track the agent so we can sample parameters at the end too.
        _stats_for(self.agent_id).first_seen_step = 0

    DecisionModule.__init__ = patched_dm_init  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# End-of-run snapshotting
# ---------------------------------------------------------------------------


def collect_final_state(env) -> None:
    """Sample final parameters/buffer size/epsilon for every learning agent."""
    agent_objs = list(getattr(env, "agent_objects", []) or [])
    # Some agents may already be dead; pull from the agent registry too if present
    for agent in agent_objs:
        agent_id = getattr(agent, "agent_id", None)
        if not agent_id:
            continue
        behavior = getattr(agent, "behavior", None)
        decision_module = getattr(behavior, "decision_module", None)
        if decision_module is None:
            continue
        algorithm = getattr(decision_module, "algorithm", None)
        if algorithm is None:
            continue
        stats = _stats_for(agent_id)
        # Buffer size / epsilon snapshot
        try:
            stats.buffer_size_at_end = len(algorithm.replay_buffer)
        except Exception:
            stats.buffer_size_at_end = -1
        try:
            policy = algorithm.policy
            stats.epsilon_at_end = float(getattr(policy, "eps", float("nan")))
        except Exception:
            stats.epsilon_at_end = None
        # Parameter snapshot for L2 movement
        sample = _sample_policy_params(getattr(algorithm, "policy", None))
        if sample is not None:
            stats.final_param_sample = sample
            if stats.initial_param_sample is not None and stats.initial_param_sample.shape == sample.shape:
                stats.param_change_l2 = float(
                    np.linalg.norm(sample - stats.initial_param_sample)
                )


# ---------------------------------------------------------------------------
# Lifespan harvesting from the simulation database
# ---------------------------------------------------------------------------


def harvest_lifespans(env) -> Dict[str, int]:
    """Return ``agent_id -> lifespan_in_steps`` from the in-memory env."""
    out: Dict[str, int] = {}
    final_time = int(getattr(env, "time", 0))
    for agent in getattr(env, "agent_objects", []) or []:
        agent_id = getattr(agent, "agent_id", None)
        if not agent_id:
            continue
        birth = int(getattr(agent, "birth_time", 0))
        # Use death_time if present, else current time (still alive at end of run).
        death = getattr(agent, "death_time", None)
        if death is None:
            # Some implementations don't set death_time on alive agents
            death = final_time
        out[str(agent_id)] = max(0, int(death) - birth)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _percentiles(values, qs=(0, 25, 50, 75, 100)) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=float)
    return ", ".join(f"p{q}={np.percentile(arr, q):.1f}" for q in qs)


def report(num_steps: int, lifespans: Dict[str, int]) -> None:
    print("\n" + "=" * 78)
    print("DQN learning diagnostic")
    print("=" * 78)
    if not _STATS:
        print("No DecisionModule instances were instrumented. Did the simulation start?")
        return

    rows = []
    for agent_id, s in _STATS.items():
        rows.append(
            {
                "agent": agent_id,
                "lifespan": lifespans.get(agent_id, -1),
                "stores": s.store_calls,
                "train_ready_T": s.train_ready_true,
                "train_ready_F": s.train_ready_false,
                "policy_learn_calls": s.policy_learn_calls,
                "buf_end": s.buffer_size_at_end,
                "param_l2": s.param_change_l2,
                "eps_end": s.epsilon_at_end,
            }
        )

    rows.sort(key=lambda r: r["agent"])

    print(f"\nSimulation steps requested: {num_steps}")
    print(f"Total agents instrumented:  {len(rows)}\n")

    header = (
        f"{'agent_id':<14} {'life':>5} {'stores':>7} {'train_ok':>9} "
        f"{'train_skip':>11} {'learn_called':>13} {'buf_end':>8} "
        f"{'|Δw|_2':>9} {'eps_end':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows[:50]:  # cap the printout
        eps = "n/a" if r["eps_end"] is None else f"{r['eps_end']:.3f}"
        life = "?" if r["lifespan"] < 0 else str(r["lifespan"])
        print(
            f"{r['agent'][:14]:<14} {life:>5} {r['stores']:>7} "
            f"{r['train_ready_T']:>9} {r['train_ready_F']:>11} "
            f"{r['policy_learn_calls']:>13} {r['buf_end']:>8} "
            f"{r['param_l2']:>9.4f} {eps:>9}"
        )
    if len(rows) > 50:
        print(f"... ({len(rows) - 50} additional agents truncated)")

    # Aggregates
    stores = [r["stores"] for r in rows]
    learns = [r["policy_learn_calls"] for r in rows]
    skipped = [r["train_ready_F"] for r in rows]
    successful_train_calls = [r["train_ready_T"] for r in rows]
    param_moves = [r["param_l2"] for r in rows]
    lifespan_vals = [r["lifespan"] for r in rows if r["lifespan"] >= 0]

    print("\n--- Aggregates ---")
    print(f"Stores per agent:                  {_percentiles(stores)}")
    print(f"Train-ready=True calls per agent:  {_percentiles(successful_train_calls)}")
    print(f"Train-ready=False calls per agent: {_percentiles(skipped)}")
    print(f"policy.learn() calls per agent:    {_percentiles(learns)}")
    print(f"|Δw|_2 of first param per agent:   {_percentiles(param_moves)}")
    if lifespan_vals:
        print(f"Lifespan in steps per agent:       {_percentiles(lifespan_vals)}")
        print(f"  -> mean lifespan: {np.mean(lifespan_vals):.1f} steps")
    else:
        print("Lifespan in steps per agent:       n/a (no DB rows)")

    n_agents = len(rows)
    n_stored_anything = sum(1 for s in stores if s > 0)
    n_ever_trained = sum(1 for L in learns if L > 0)
    n_weights_moved = sum(1 for v in param_moves if v > 1e-8)
    n_filled_batch = sum(
        1 for r in rows if r["buf_end"] >= 32  # default rl_batch_size
    )

    print("\n--- Summary ---")
    print(f"Agents that stored ≥1 experience:           {n_stored_anything}/{n_agents}")
    print(f"Agents whose buffer reached batch size 32:  {n_filled_batch}/{n_agents}")
    print(f"Agents that ran ≥1 training gradient step:  {n_ever_trained}/{n_agents}")
    print(f"Agents whose Q-net weights actually moved:  {n_weights_moved}/{n_agents}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument(
        "--no-defer",
        action="store_true",
        help="Force defer_learning_training=False so each store_experience trains immediately",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Override max_learning_updates_per_step",
    )
    parser.add_argument("--output", default="simulations/diagnostic", help="Output directory")
    args = parser.parse_args()

    install_instrumentation()

    config = SimulationConfig.from_centralized_config(environment="development")
    perf = getattr(config, "performance", None)
    if perf is not None:
        if args.no_defer:
            perf.defer_learning_training = False
        if args.max_updates is not None:
            perf.max_learning_updates_per_step = int(args.max_updates)

    print(
        f"[diag] config: defer_learning_training={getattr(perf, 'defer_learning_training', None)}, "
        f"max_learning_updates_per_step={getattr(perf, 'max_learning_updates_per_step', None)}, "
        f"system_agents={config.population.system_agents}, "
        f"independent_agents={config.population.independent_agents}, "
        f"control_agents={config.population.control_agents}, "
        f"steps={args.steps}"
    )
    print(
        f"[diag] DecisionConfig defaults expected: rl_batch_size=32, rl_train_freq=4, rl_buffer_size=10000"
    )

    env = run_simulation(
        num_steps=args.steps,
        config=config,
        path=args.output,
        save_config=False,
        disable_console_logging=True,
    )

    collect_final_state(env)
    lifespans = harvest_lifespans(env)
    report(args.steps, lifespans)

    try:
        env.cleanup()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    if "PYTHONHASHSEED" not in os.environ or os.environ["PYTHONHASHSEED"] != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    raise SystemExit(main())
