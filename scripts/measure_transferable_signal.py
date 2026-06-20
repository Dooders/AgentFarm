#!/usr/bin/env python3
"""Transferable-signal precondition gate for richer inheritance payloads (#904).

Before building P2-P4 inherited payloads (#848), we must answer a hard
precondition: *in the chosen regime, does an end-of-life policy measurably
outperform a freshly-initialized one?* If within-life learning produces no
transferable signal, none of P1-P4 can beat the Baldwinian baseline and the
richer-payload work should stop with a clean negative result.

This script measures the **transferable-signal budget** two complementary ways
in a learning-positive regime (small population, long horizon, no inheritance):

Tier 1 - probe-state drift (cheap, design-prescribed)
    Cache a fixed set of observation tensors early in the run, snapshot each
    learning agent's Q-net at birth and end-of-life, and report argmax / max-Q
    drift over a lifetime on the frozen probe set. This confirms whether the
    policy *moved* at all - but movement alone is not improvement.

Tier 2 - grounded rollout differential (the decision metric)
    For each harvested (birth, end-of-life) weight pair, run paired held-out
    evaluation episodes and measure several deltas ``end-of-life - init``:

    - full-episode net reward (kept for continuity, but **survival-dominated**:
      it is ~survival_steps x foraging-rate, so it mostly measures "stayed
      alive longer", not finer decisions);
    - survival;
    - per-step reward rate (survival-length-normalized); and
    - **early-age net reward** at fixed ages (default 10/25/50). Because every
      survivor accrues over the same short window, this isolates foraging /
      decision quality from survival length and matches the "net early RL
      reward at ages N" readout #904's A/B is graded on.

    The eval baseline (init arm) runs the **real action-weighted stochastic
    policy** (``softmax(Q) x action_weights`` sampling) by default, not pure
    greedy: greedy-argmax on an untrained net is a degenerate fixed-action
    controller and overstates the gap. Under the weighted policy a random-init
    net collapses to the chromosome action-prior (Baldwinian P0) policy, which
    is exactly the baseline #904 compares against. ``--eval-policy greedy``
    restores the old pure-greedy read.

The gate keys off a **survival-decoupled** decision metric (``--gate-metric``,
default the early-age net-reward budget), so a PASS means end-of-life policies
make robustly better fine-grained decisions - the quantity P2-P4 transfer - not
merely that they survive. Cohort aggregation uses the project robustness gate:
paired 95% CI excludes zero AND within-profile sign agreement >= 0.75.

Run with ``PYTHONHASHSEED=0 python scripts/measure_transferable_signal.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# The regime trains many tiny per-agent DQNs. With the default BLAS/OpenMP
# thread pools, torch oversubscribes (dozens of threads thrash across cores)
# and a single step balloons to ~seconds, which previously read as the run
# "hanging". Tiny networks are fastest single-threaded, so pin the math
# backends to one thread *before* numpy/torch import their pools.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from farm.config import SimulationConfig  # noqa: E402
from farm.core.agent.core import AgentCore  # noqa: E402
from farm.core.decision.algorithms.tianshou import TianshouWrapper  # noqa: E402
from farm.core.decision.decision import DecisionModule  # noqa: E402
from farm.core.simulation import run_simulation  # noqa: E402
from farm.runners.intrinsic_evolution_experiment import (  # noqa: E402
    STABLE_SUB_PROFILES,
)
from scripts.analyze_stable_profile_seed_sweep import (  # noqa: E402
    PROFILE_ORDER,
    SIGN_AGREEMENT_THRESHOLD,
    _mean,
    _sign_agreement,
    _t_ci,
    _variance,
)

DEFAULT_SEEDS: List[int] = [42, 7, 19, 101, 137, 256]
DEFAULT_PROFILES: List[str] = ["conservative", "balanced", "buffered"]

# Mapping from STABLE_SUB_PROFILES override keys to SimulationConfig fields.
# Mirrors InitialConditionsConfig application in
# farm/runners/intrinsic_evolution_experiment.py so the gate runs in the same
# ecology the #848 experiment will use.
_PROFILE_FIELD_MAP = {
    "initial_agent_resource_level": ("agent_behavior", "initial_resource_level"),
    "initial_resource_count": ("resources", "initial_resources"),
    "resource_regen_rate": ("resources", "resource_regen_rate"),
    "resource_regen_amount": ("resources", "resource_regen_amount"),
}


def _configure_torch_threads() -> None:
    """Pin Torch intra/inter-op thread counts to 1.

    Tiny networks are fastest single-threaded. Calling this at __main__ time
    (rather than at import time) avoids unexpectedly overriding the thread
    settings of any process that imports this module as a library.
    """
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already initialized (e.g. torch used before import); intra-op cap still applies.
        pass


StateDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Recorder: shared instrumentation state for capture + greedy override
# ---------------------------------------------------------------------------


@dataclass
class _PolicySnapshots:
    """Birth / end-of-life weight snapshots and lifespan bookkeeping per agent."""

    birth: Optional[StateDict] = None
    eol: Optional[StateDict] = None
    birth_time: int = 0
    first_seen: int = -1
    last_seen: int = -1

    @property
    def lifespan(self) -> int:
        if self.last_seen < 0:
            return 0
        return max(0, self.last_seen - self.birth_time)


@dataclass
class _Recorder:
    """Process-global recorder driving probe capture and greedy evaluation.

    A single instance is shared by the monkey-patched ``DecisionModule`` methods.
    ``mode`` gates behavior: ``"train"`` captures probe states, ``"eval"`` forces
    pure-greedy action selection with injected weights, ``"idle"`` is a no-op.
    """

    mode: str = "idle"
    current_step: int = -1

    # Probe capture (train mode)
    probes: List[np.ndarray] = field(default_factory=list)
    capture_lo: int = 20
    capture_hi: int = 120
    max_probes: int = 256

    # Per-agent snapshots (train mode). Memory is bounded: ``alive`` holds only
    # currently-living agents (<= max_population), and ``dead`` keeps just the
    # ``snapshot_cap`` longest-lived agents that have already died. Without this
    # bound, a high-churn / high-population profile (e.g. ``buffered``) would
    # accumulate one cloned Q-net state-dict per agent ever born and exhaust RAM.
    alive: Dict[str, _PolicySnapshots] = field(default_factory=dict)
    dead: List[_PolicySnapshots] = field(default_factory=list)
    snapshot_cap: int = 48
    scratch_algo: Optional[Any] = None

    # Eval mode
    eval_weights: Optional[StateDict] = None
    # "greedy" forces pure-argmax (degenerate for an untrained net); "weighted"
    # falls through to the real ``decide_action`` (softmax(Q) x action_weights
    # sampling), i.e. the policy the agent actually runs in the sim. The latter
    # is the non-degenerate baseline: with a random-init net it collapses to the
    # chromosome action-prior (Baldwinian) policy, which is exactly the P0
    # baseline #904 compares against.
    eval_policy: str = "weighted"

    def reset(self) -> None:
        self.mode = "idle"
        self.current_step = -1
        self.probes = []
        self.alive = {}
        self.dead = []
        self.scratch_algo = None
        self.eval_weights = None
        self.eval_policy = "weighted"

    # -- train-mode hooks --------------------------------------------------

    def begin_train(
        self,
        capture_lo: int,
        capture_hi: int,
        max_probes: int,
        snapshot_cap: int = 48,
    ) -> None:
        self.mode = "train"
        self.current_step = -1
        self.capture_lo = capture_lo
        self.capture_hi = capture_hi
        self.max_probes = max_probes
        self.snapshot_cap = max(1, snapshot_cap)
        self.probes = []
        self.alive = {}
        self.dead = []
        self.scratch_algo = None

    def on_init(self, decision_module: "DecisionModule") -> None:
        """Register a new learning agent: birth snapshot + scratch evaluator."""
        if self.mode != "train":
            return
        algo = getattr(decision_module, "algorithm", None)
        if algo is None or not hasattr(algo, "_policy_q_values"):
            return
        agent_id = str(getattr(decision_module, "agent_id", "") or "")
        if not agent_id:
            return
        snap = self.alive.setdefault(agent_id, _PolicySnapshots())
        if snap.birth is None:
            snap.birth = _clone_policy_state(algo)
        if self.scratch_algo is None:
            self.scratch_algo = algo

    def _retire(self, snap: "_PolicySnapshots") -> None:
        """Move a dead agent's snapshot into the bounded top-K ``dead`` list."""
        if snap.birth is None or snap.eol is None:
            return
        self.dead.append(snap)
        if len(self.dead) > self.snapshot_cap:
            self.dead.sort(key=lambda s: s.lifespan, reverse=True)
            del self.dead[self.snapshot_cap:]

    def on_train_step_end(self, env: Any, step: int, snapshot_interval: int) -> None:
        self.current_step = step
        take_snapshot = snapshot_interval > 0 and (step % snapshot_interval == 0)

        current_ids = set()
        for agent in list(getattr(env, "agent_objects", []) or []):
            agent_id = str(getattr(agent, "agent_id", "") or "")
            if not agent_id:
                continue
            algo = _agent_algorithm(agent)
            if algo is None:
                continue
            current_ids.add(agent_id)
            snap = self.alive.setdefault(agent_id, _PolicySnapshots())
            if snap.first_seen < 0:
                snap.first_seen = step
                snap.birth_time = int(getattr(agent, "birth_time", step) or 0)
                if snap.birth is None:
                    snap.birth = _clone_policy_state(algo)
            snap.last_seen = step
            if take_snapshot or snap.eol is None:
                snap.eol = _clone_policy_state(algo)

        # Retire agents that vanished this step (died) to bound memory.
        for agent_id in list(self.alive.keys()):
            if agent_id not in current_ids:
                self._retire(self.alive.pop(agent_id))

    def finalize_train(self, env: Any) -> None:
        """Snapshot end-of-life weights for every agent still alive at run end."""
        for agent in list(getattr(env, "agent_objects", []) or []):
            agent_id = str(getattr(agent, "agent_id", "") or "")
            if not agent_id:
                continue
            algo = _agent_algorithm(agent)
            if algo is None:
                continue
            snap = self.alive.setdefault(agent_id, _PolicySnapshots())
            snap.eol = _clone_policy_state(algo)
            if snap.last_seen < 0:
                snap.last_seen = self.current_step

    def on_decide(
        self,
        decision_module: "DecisionModule",
        state: Any,
        enabled_actions: Optional[List[int]],
    ) -> Optional[int]:
        """Return a greedy action in eval mode; capture a probe in train mode.

        Returns ``None`` when the caller should fall through to the original
        (stochastic) ``decide_action`` implementation.
        """
        algo = getattr(decision_module, "algorithm", None)
        if algo is None or not hasattr(algo, "_policy_q_values"):
            return None

        if self.mode == "eval":
            if self.eval_policy == "greedy":
                return _greedy_decide(decision_module, state, enabled_actions)
            # "weighted": fall through to the real action-weighted stochastic
            # policy so the baseline is non-degenerate.
            return None

        if self.mode == "train":
            if (
                self.capture_lo <= self.current_step <= self.capture_hi
                and len(self.probes) < self.max_probes
            ):
                self.probes.append(
                    np.asarray(
                        decision_module._normalize_state_for_algorithm(state),
                        dtype=np.float32,
                    ).copy()
                )
        return None

    # -- eval-mode hooks ---------------------------------------------------

    def begin_eval(self, weights: StateDict, eval_policy: str = "weighted") -> None:
        self.mode = "eval"
        self.eval_weights = weights
        self.eval_policy = eval_policy

    def end_eval(self) -> None:
        self.mode = "idle"
        self.eval_weights = None

    def harvested(self, min_lifespan: int, top_k: int) -> List["_PolicySnapshots"]:
        """Return the longest-lived agents with usable birth+eol snapshots."""
        candidates = list(self.alive.values()) + list(self.dead)
        usable = [
            s
            for s in candidates
            if s.birth is not None and s.eol is not None and s.lifespan >= min_lifespan
        ]
        usable.sort(key=lambda s: s.lifespan, reverse=True)
        if top_k > 0:
            usable = usable[:top_k]
        return usable


_REC = _Recorder()
_INSTALLED = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _agent_algorithm(agent: Any) -> Optional[Any]:
    behavior = getattr(agent, "behavior", None)
    decision_module = getattr(behavior, "decision_module", None)
    return getattr(decision_module, "algorithm", None)


def _clone_policy_state(algo: Any) -> Optional[StateDict]:
    """Detach + clone the policy weights so later training cannot mutate them."""
    policy = getattr(algo, "policy", None)
    if policy is None:
        return None
    try:
        return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    except Exception:
        return None


def _greedy_decide(
    decision_module: "DecisionModule",
    state: Any,
    enabled_actions: Optional[List[int]],
) -> int:
    """Pure-greedy (epsilon=0, no chromosome prior) action via masked argmax."""
    algo = decision_module.algorithm
    state_np = decision_module._normalize_state_for_algorithm(state)
    q_values = algo._policy_q_values(state_np)
    mask = decision_module._create_action_mask(enabled_actions)
    full_action = algo._select_greedy_action(q_values, mask)
    return decision_module._map_to_enabled_index(full_action, enabled_actions)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(logits) / len(logits)
    return exp / total


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with a small floor to avoid divide-by-zero."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def install_instrumentation() -> None:
    """Patch ``DecisionModule`` once to route capture / greedy through ``_REC``."""
    global _INSTALLED
    if _INSTALLED:
        return

    orig_init = DecisionModule.__init__
    orig_decide = DecisionModule.decide_action
    orig_reproduce = AgentCore.reproduce

    def patched_init(self, agent, action_space, observation_space, *args, **kwargs):
        orig_init(self, agent, action_space, observation_space, *args, **kwargs)
        try:
            _REC.on_init(self)
        except Exception:
            # Instrumentation must never break the simulation.
            pass

    def patched_decide(self, state, enabled_actions=None, action_weights=None):
        override = _REC.on_decide(self, state, enabled_actions)
        if override is not None:
            return override
        return orig_decide(self, state, enabled_actions, action_weights)

    def patched_reproduce(self):
        # Reproduction cannot be disabled via config: ``reproduce`` is in the
        # agent's action set (DecisionConfig.reproduce_weight) and gated by
        # ``decision.curriculum_phases`` (not the sim-level curriculum), while
        # ``reproduce_action`` enforces no population cap. Block it outright
        # while the recorder is active so the population stays fixed - the gate
        # measures within-life learning, which is independent of reproduction.
        if _REC.mode in ("train", "eval"):
            return False
        return orig_reproduce(self)

    DecisionModule.__init__ = patched_init  # type: ignore[assignment]
    DecisionModule.decide_action = patched_decide  # type: ignore[assignment]
    AgentCore.reproduce = patched_reproduce  # type: ignore[assignment]
    _INSTALLED = True


# ---------------------------------------------------------------------------
# Regime + eval config builders
# ---------------------------------------------------------------------------


def _apply_profile_overrides(config: SimulationConfig, profile: str) -> Dict[str, Any]:
    """Apply STABLE_SUB_PROFILES[profile] ecology to a SimulationConfig."""
    overrides = STABLE_SUB_PROFILES[profile]
    for key, value in overrides.items():
        mapping = _PROFILE_FIELD_MAP.get(key)
        if mapping is None:
            # Unknown key — forward-compat: skip rather than crash.  Log so
            # callers notice when STABLE_SUB_PROFILES gains keys not yet in
            # _PROFILE_FIELD_MAP.
            print(
                f"_apply_profile_overrides: ignoring unknown profile key {key!r}",
                file=sys.stderr,
            )
            continue
        section_name, field_name = mapping
        section = getattr(config, section_name, None)
        if section is not None and hasattr(section, field_name):
            setattr(section, field_name, value)
    return dict(overrides)


def _use_disk_database(config: SimulationConfig) -> None:
    """Force disk-backed, persisted SQLite (never in-memory).

    In-memory SQLite schema is per-connection and proved fragile over long
    runs (lost ``agent_actions`` table mid-flush), so all simulation data is
    persisted to disk.
    """
    config.database.use_in_memory_db = False
    if hasattr(config.database, "persist_db_on_completion"):
        config.database.persist_db_on_completion = True


def build_regime_config(
    profile: str,
    *,
    environment: str = "development",
    population: int = 8,
    max_population: int = 64,
) -> SimulationConfig:
    """Learning-positive training regime: small fixed population, fixed ecology.

    Reproduction is blocked process-wide by ``install_instrumentation`` (it
    patches ``AgentCore.reproduce`` to a no-op while the recorder is active),
    so the population starts at ``population`` and only ever shrinks. This is
    essential: ``reproduce_action`` enforces no population cap, and a
    high-resource ecology (e.g. ``buffered``) otherwise explodes to 100+ agents
    and exhausts RAM. The gate measures within-life learning, which is
    independent of reproduction.
    """
    config = SimulationConfig.from_centralized_config(environment=environment)
    pop = config.population
    pop.system_agents = 0
    pop.independent_agents = int(population)
    pop.control_agents = 0
    for attr in ("order_agents", "chaos_agents"):
        if hasattr(pop, attr):
            setattr(pop, attr, 0)
    pop.max_population = int(max_population)
    _use_disk_database(config)
    _apply_profile_overrides(config, profile)
    return config


def build_eval_config(
    profile: str, *, environment: str = "development"
) -> SimulationConfig:
    """Single-agent, reproduction-free held-out evaluation ecology.

    Reproduction is blocked by the ``AgentCore.reproduce`` no-op patch installed
    by ``install_instrumentation`` (active in eval mode), so the single agent
    cannot spawn a colony during a rollout.
    """
    config = SimulationConfig.from_centralized_config(environment=environment)
    pop = config.population
    pop.system_agents = 0
    pop.independent_agents = 1
    pop.control_agents = 0
    for attr in ("order_agents", "chaos_agents"):
        if hasattr(pop, attr):
            setattr(pop, attr, 0)
    pop.max_population = 1
    _use_disk_database(config)
    _apply_profile_overrides(config, profile)
    # Frozen-policy rollouts: disable deferred gradient steps so injected
    # snapshot weights stay fixed for the full held-out episode.
    config.performance.max_learning_updates_per_step = -1
    return config


# ---------------------------------------------------------------------------
# Tier 1: probe-state drift
# ---------------------------------------------------------------------------


def compute_drift(
    scratch_algo: Optional[Any],
    harvested: Sequence["_PolicySnapshots"],
    probes: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Per-policy argmax / max-Q / KL drift between birth and end-of-life nets."""
    empty = {
        "n_policies": 0,
        "n_probes": len(probes),
        "argmax_disagreement": float("nan"),
        "max_q_drift": float("nan"),
        "softmax_kl": float("nan"),
        "entropy_delta": float("nan"),
    }
    if scratch_algo is None or not probes or not harvested:
        return empty

    disagreements: List[float] = []
    maxq_drifts: List[float] = []
    kls: List[float] = []
    entropy_deltas: List[float] = []

    for snap in harvested:
        try:
            scratch_algo.load_model_state({"policy_state_dict": snap.birth})
            q_init = np.stack([scratch_algo._policy_q_values(p) for p in probes])
            scratch_algo.load_model_state({"policy_state_dict": snap.eol})
            q_eol = np.stack([scratch_algo._policy_q_values(p) for p in probes])
        except Exception:
            continue

        argmax_init = q_init.argmax(axis=1)
        argmax_eol = q_eol.argmax(axis=1)
        disagreements.append(float(np.mean(argmax_init != argmax_eol)))
        maxq_drifts.append(
            float(np.mean(np.abs(q_eol.max(axis=1) - q_init.max(axis=1))))
        )

        probe_kls: List[float] = []
        ent_init: List[float] = []
        ent_eol: List[float] = []
        for i in range(q_init.shape[0]):
            p_init = _softmax(q_init[i])
            p_eol = _softmax(q_eol[i])
            probe_kls.append(_kl(p_eol, p_init))
            ent_init.append(_entropy(p_init))
            ent_eol.append(_entropy(p_eol))
        kls.append(float(np.mean(probe_kls)))
        entropy_deltas.append(float(np.mean(ent_eol) - np.mean(ent_init)))

    if not disagreements:
        return empty
    return {
        "n_policies": len(disagreements),
        "n_probes": len(probes),
        "argmax_disagreement": _mean(disagreements),
        "max_q_drift": _mean(maxq_drifts),
        "softmax_kl": _mean(kls),
        "entropy_delta": _mean(entropy_deltas),
    }


# ---------------------------------------------------------------------------
# Tier 2: grounded greedy-rollout differential
# ---------------------------------------------------------------------------


def eval_episode_return(
    eval_config: SimulationConfig,
    weights: StateDict,
    eval_seed: int,
    n_steps: int,
    reward_ages: Sequence[int],
    eval_policy: str = "weighted",
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one single-agent episode; return full + early-age reward and survival.

    ``reward_ages`` are agent ages (steps since birth) at which to snapshot the
    focal agent's cumulative reward. Early-age reward controls survival length
    (every survivor accrues over the same short window), so it isolates
    foraging / decision quality from the "stayed alive longer" effect that
    dominates the full-episode total. If the agent dies before an age, that
    age's reward is frozen at death (dying early is still penalized, but the
    metric is no longer survival-length-saturated).
    """
    focal: Dict[str, Any] = {
        "id": None,
        "reward": 0.0,
        "last_step": -1,
        "birth_time": 0,
        "reward_at_age": {int(a): None for a in reward_ages},
    }

    def on_ready(env: Any) -> None:
        for agent in list(getattr(env, "agent_objects", []) or []):
            algo = _agent_algorithm(agent)
            if algo is None or not hasattr(algo, "load_model_state"):
                continue
            algo.load_model_state({"policy_state_dict": weights})
            if hasattr(algo, "set_train_mode"):
                algo.set_train_mode(False)
            if focal["id"] is None:
                focal["id"] = str(getattr(agent, "agent_id", "") or "")
                focal["birth_time"] = int(getattr(agent, "birth_time", 0) or 0)

    def on_step(env: Any, step: int) -> None:
        for agent in list(getattr(env, "agent_objects", []) or []):
            if str(getattr(agent, "agent_id", "") or "") == focal["id"]:
                reward = float(getattr(agent, "total_reward", 0.0) or 0.0)
                focal["reward"] = reward
                focal["last_step"] = step
                age = step - focal["birth_time"]
                for a, recorded in focal["reward_at_age"].items():
                    if recorded is None and age >= a:
                        focal["reward_at_age"][a] = reward
                break

    _REC.begin_eval(weights, eval_policy=eval_policy)
    try:
        env = run_simulation(
            num_steps=n_steps,
            config=eval_config,
            path=path,
            save_config=False,
            seed=eval_seed,
            on_environment_ready=on_ready,
            on_step_end=on_step,
            disable_console_logging=True,
        )
    finally:
        _REC.end_eval()

    survived = focal["last_step"] >= (n_steps - 1)
    survived_steps = int(focal["last_step"] + 1)
    # Freeze any unreached age at the final (death) reward.
    reward_at_age = {
        a: (focal["reward"] if v is None else float(v))
        for a, v in focal["reward_at_age"].items()
    }
    net_reward = float(focal["reward"])
    try:
        env.cleanup()
    except Exception as exc:
        # Best-effort cleanup: evaluation metrics are already computed, so do
        # not fail the run on teardown errors. Emit a warning for visibility.
        print(f"[measure_transferable_signal] warning: env.cleanup() failed: {exc}", file=sys.stderr)
    return {
        "net_reward": net_reward,
        "survived": bool(survived),
        "survived_steps": survived_steps,
        "reward_at_age": reward_at_age,
        "reward_rate": net_reward / max(1, survived_steps),
    }


def eval_policy_budget(
    eval_config: SimulationConfig,
    birth_weights: StateDict,
    eol_weights: StateDict,
    eval_seeds: Sequence[int],
    n_steps: int,
    reward_ages: Sequence[int],
    eval_policy: str = "weighted",
    path: Optional[str] = None,
) -> Dict[str, float]:
    """Paired (end-of-life - init) deltas over eval seeds.

    Reports the full-episode reward delta (survival-dominated, kept for
    continuity), the survival delta, and the survival-decoupled decision-quality
    metrics: per-step reward rate and early-age net reward at ``reward_ages``.
    """
    reward_deltas: List[float] = []
    survival_deltas: List[float] = []
    rate_deltas: List[float] = []
    age_deltas: Dict[int, List[float]] = {int(a): [] for a in reward_ages}
    for seed in eval_seeds:
        init = eval_episode_return(
            eval_config, birth_weights, seed, n_steps, reward_ages,
            eval_policy=eval_policy, path=path,
        )
        eol = eval_episode_return(
            eval_config, eol_weights, seed, n_steps, reward_ages,
            eval_policy=eval_policy, path=path,
        )
        reward_deltas.append(eol["net_reward"] - init["net_reward"])
        survival_deltas.append(float(eol["survived"]) - float(init["survived"]))
        rate_deltas.append(eol["reward_rate"] - init["reward_rate"])
        for a in age_deltas:
            age_deltas[a].append(eol["reward_at_age"][a] - init["reward_at_age"][a])
    budget = {
        "reward_delta": _mean(reward_deltas),
        "survival_delta": _mean(survival_deltas),
        "reward_rate_delta": _mean(rate_deltas),
    }
    for a, vals in age_deltas.items():
        budget[f"reward_age_{a}_delta"] = _mean(vals)
    return budget


# ---------------------------------------------------------------------------
# Per-cell driver + cohort aggregation
# ---------------------------------------------------------------------------


def run_cell(profile: str, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    """Run one (profile, seed) cell: train, then Tier-1 drift + Tier-2 budget.

    Writes the per-cell result to ``{output_dir}/stable_{profile}/seed_{seed}/
    cell_result.json`` and, when ``--resume`` is set, returns a previously
    completed checkpoint instead of recomputing. The training and evaluation
    simulation databases are persisted under the same cell directory.
    """
    cell_dir = Path(args.output_dir) / f"stable_{profile}" / f"seed_{seed}"
    checkpoint = cell_dir / "cell_result.json"
    if args.resume and checkpoint.is_file():
        try:
            return json.loads(checkpoint.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    cell_dir.mkdir(parents=True, exist_ok=True)

    install_instrumentation()
    _REC.reset()
    _REC.begin_train(
        args.probe_capture_start,
        args.probe_capture_end,
        args.max_probes,
        snapshot_cap=max(args.top_k * 8, 32),
    )

    config = build_regime_config(
        profile,
        environment=args.environment,
        population=args.population,
        max_population=args.max_population,
    )

    def on_step(env: Any, step: int) -> None:
        _REC.on_train_step_end(env, step, args.snapshot_interval)

    env = run_simulation(
        num_steps=args.num_steps,
        config=config,
        path=str(cell_dir),
        save_config=False,
        seed=seed,
        on_step_end=on_step,
        disable_console_logging=True,
    )
    _REC.finalize_train(env)

    harvested = _REC.harvested(args.min_lifespan, args.top_k)
    drift = compute_drift(_REC.scratch_algo, harvested, _REC.probes)

    eval_profile = args.cross_eval_profile or profile
    eval_config = build_eval_config(eval_profile, environment=args.environment)
    eval_dir = cell_dir / "eval"
    per_policy: List[Dict[str, float]] = []
    if not args.skip_rollout:
        for snap in harvested:
            budget = eval_policy_budget(
                eval_config,
                snap.birth,
                snap.eol,
                args.eval_seeds,
                args.eval_steps,
                args.reward_ages,
                eval_policy=args.eval_policy,
                path=str(eval_dir),
            )
            budget["lifespan"] = snap.lifespan
            per_policy.append(budget)

    try:
        env.cleanup()
    except Exception as exc:
        # Cleanup is best-effort: do not fail the run teardown path, but
        # surface the failure so it is not silently swallowed.
        print(f"Warning: env.cleanup() failed: {exc}", file=sys.stderr)

    # Cell-level budget = mean per-policy delta, for every metric we tracked.
    metric_keys = ["reward_delta", "survival_delta", "reward_rate_delta"] + [
        f"reward_age_{int(a)}_delta" for a in args.reward_ages
    ]
    budgets = {
        f"{k.replace('_delta', '')}_budget": (
            _mean([p[k] for p in per_policy]) if per_policy else float("nan")
        )
        for k in metric_keys
    }
    cell = {
        "profile": profile,
        "seed": seed,
        "eval_profile": eval_profile,
        "eval_policy": args.eval_policy,
        "n_probes": len(_REC.probes),
        "n_harvested": len(harvested),
        "drift": drift,
        # Back-compat aliases (existing plots / readers).
        "reward_budget": budgets["reward_budget"],
        "survival_budget": budgets["survival_budget"],
        "budgets": budgets,
        "per_policy": per_policy,
    }
    _REC.reset()

    checkpoint.write_text(
        json.dumps(_json_safe(cell), indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    return cell


def _verdict(deltas: Sequence[float]) -> Dict[str, Any]:
    """Project robustness gate for a list of paired deltas (one per seed)."""
    vals = [d for d in deltas if d is not None and not math.isnan(d)]
    if len(vals) < 2:
        return {
            "n": len(vals),
            "mean_delta": _mean(vals) if vals else float("nan"),
            "ci95": [float("nan"), float("nan")],
            "sign_agreement": float("nan"),
            "ci_excludes_zero": False,
            "robust": False,
        }
    lo, hi = _t_ci(vals)
    sign_agreement = _sign_agreement(vals)
    ci_excludes_zero = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    return {
        "n": len(vals),
        "mean_delta": _mean(vals),
        "variance": _variance(vals),
        "ci95": [lo, hi],
        "sign_agreement": sign_agreement,
        "ci_excludes_zero": ci_excludes_zero,
        "robust": ci_excludes_zero and sign_agreement >= SIGN_AGREEMENT_THRESHOLD,
    }


def _metric_budget_keys(cells: Sequence[Dict[str, Any]]) -> List[str]:
    """Stable, ordered list of budget metric keys present across cells."""
    for c in cells:
        if c.get("budgets"):
            return list(c["budgets"].keys())
    return ["reward_budget", "survival_budget"]


def aggregate(
    cells: Sequence[Dict[str, Any]],
    drift_threshold: float,
    gate_metric: str = "reward_age_10_budget",
) -> Dict[str, Any]:
    """Aggregate per-cell budgets per profile and decide the gate verdict.

    The gate keys off ``gate_metric`` (default the early-age net-reward budget),
    which is the survival-decoupled decision-quality metric that #904's A/B is
    graded on - not the full-episode reward, which is survival-dominated.
    Verdicts for every tracked metric are reported alongside for context.
    """
    profiles = sorted({c["profile"] for c in cells}, key=_profile_rank)
    metric_keys = _metric_budget_keys(cells)
    if gate_metric not in metric_keys:
        gate_metric = "reward_budget" if "reward_budget" in metric_keys else metric_keys[0]
    per_profile: Dict[str, Any] = {}
    all_disagreements: List[float] = []

    for profile in profiles:
        rows = sorted(
            (c for c in cells if c["profile"] == profile), key=lambda c: c["seed"]
        )
        disagreements = [
            c["drift"].get("argmax_disagreement", float("nan")) for c in rows
        ]
        all_disagreements.extend(
            d for d in disagreements if d is not None and not math.isnan(d)
        )

        def _budget_series(key: str) -> List[float]:
            return [c.get("budgets", {}).get(key, c.get(key, float("nan"))) for c in rows]

        verdicts = {key: _verdict(_budget_series(key)) for key in metric_keys}
        per_profile[profile] = {
            "seeds": [c["seed"] for c in rows],
            "budget_per_seed": {key: _budget_series(key) for key in metric_keys},
            "verdicts": verdicts,
            # Back-compat: existing plots/readers expect these top-level keys.
            "reward_budget_per_seed": _budget_series(gate_metric),
            "reward_verdict": verdicts.get(gate_metric, _verdict([])),
            "full_reward_verdict": verdicts.get("reward_budget", _verdict([])),
            "survival_verdict": verdicts.get("survival_budget", _verdict([])),
            "mean_argmax_disagreement": _mean(
                [d for d in disagreements if d is not None and not math.isnan(d)]
            ),
        }

    robust_positive = [
        p
        for p, v in per_profile.items()
        if v["verdicts"].get(gate_metric, {}).get("robust")
        and v["verdicts"].get(gate_metric, {}).get("mean_delta", 0.0) > 0
    ]
    drift_nontrivial = _mean(all_disagreements) > drift_threshold if all_disagreements else False
    gate = "PASS" if (robust_positive and drift_nontrivial) else "NULL"

    return {
        "gate_metric": gate_metric,
        "metric_keys": metric_keys,
        "per_profile": per_profile,
        "robust_positive_profiles": robust_positive,
        "mean_argmax_disagreement": _mean(all_disagreements)
        if all_disagreements
        else float("nan"),
        "drift_threshold": drift_threshold,
        "drift_nontrivial": bool(drift_nontrivial),
        "gate": gate,
    }


def _profile_rank(profile: str) -> int:
    return PROFILE_ORDER.index(profile) if profile in PROFILE_ORDER else 99


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{decimals}f}"
    if isinstance(value, (list, tuple)):
        return f"[{_fmt(value[0], decimals)}, {_fmt(value[1], decimals)}]"
    return str(value)


_METRIC_LABELS = {
    "reward_budget": "Full-episode reward (survival-dominated)",
    "survival_budget": "Survival (fraction)",
    "reward_rate_budget": "Per-step reward rate (survival-normalized)",
}


def _metric_label(key: str) -> str:
    if key in _METRIC_LABELS:
        return _METRIC_LABELS[key]
    if key.startswith("reward_age_") and key.endswith("_budget"):
        age = key[len("reward_age_"):-len("_budget")]
        return f"Net reward at age {age} (decision quality)"
    return key


def build_markdown(summary: Dict[str, Any]) -> str:
    agg = summary["aggregate"]
    gate_metric = agg.get("gate_metric", "reward_budget")
    eval_policy = summary["config"].get("eval_policy", "weighted")
    cross = summary["config"].get("cross_eval_profile")
    lines: List[str] = [
        "# Transferable-signal precondition gate (#904)",
        "",
        f"**Gate verdict: {agg['gate']}** (decision metric: "
        f"`{gate_metric}` = {_metric_label(gate_metric)})",
        "",
        "Measures whether an end-of-life policy outperforms a freshly-initialized "
        "one in a learning-positive regime (small population, long horizon, no "
        "inheritance). The gate keys off a **survival-decoupled** decision-quality "
        "metric (early-age net reward), which is what #904's A/B is graded on - "
        "not the full-episode reward, which is dominated by how long the agent "
        "survives. The baseline (init) arm runs the "
        f"**{eval_policy}** action policy, gated per profile with the project "
        "rule (95% CI excludes zero AND sign agreement >= 0.75). Tier-1 probe "
        "drift confirms the policy moved.",
        "",
        f"- Regime: population={summary['config']['population']}, "
        f"num_steps={summary['config']['num_steps']}, "
        f"profiles={summary['config']['profiles']}, "
        f"seeds={summary['config']['seeds']}",
        f"- Eval: policy={eval_policy}, steps={summary['config']['eval_steps']}, "
        f"reward_ages={summary['config'].get('reward_ages')}"
        + (f", cross_eval_profile={cross}" if cross else ""),
        f"- Mean argmax disagreement (probe drift): "
        f"{_fmt(agg['mean_argmax_disagreement'])} "
        f"(threshold {_fmt(agg['drift_threshold'])}, "
        f"non-trivial={agg['drift_nontrivial']})",
        "",
        f"## Decision metric per profile (`{gate_metric}`)",
        "",
        "| Profile | Mean Δ | 95% CI | Sign agr. | Robust | Mean drift |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for profile, block in agg["per_profile"].items():
        rv = block["verdicts"].get(gate_metric, {})
        lines.append(
            f"| {profile} "
            f"| {_fmt(rv.get('mean_delta'))} "
            f"| {_fmt(rv.get('ci95'))} "
            f"| {_fmt(rv.get('sign_agreement'), 2)} "
            f"| {'yes' if rv.get('robust') else 'no'} "
            f"| {_fmt(block.get('mean_argmax_disagreement'))} |"
        )
    lines.append("")

    # All-metric overview: mean budget + robust flag per profile, per metric.
    metric_keys = agg.get("metric_keys", [])
    if metric_keys:
        lines += [
            "## All metrics (mean Δ; ✓ = robust per project gate)",
            "",
            "| Metric | " + " | ".join(agg["per_profile"].keys()) + " |",
            "| --- | " + " | ".join("---" for _ in agg["per_profile"]) + " |",
        ]
        for key in metric_keys:
            cells_md = []
            for block in agg["per_profile"].values():
                v = block["verdicts"].get(key, {})
                flag = "✓" if v.get("robust") else "·"
                cells_md.append(f"{_fmt(v.get('mean_delta'), 2)} {flag}")
            lines.append(f"| {_metric_label(key)} | " + " | ".join(cells_md) + " |")
        lines.append("")

    lines += ["## Interpretation", ""]
    if agg["gate"] == "PASS":
        lines.append(
            "There is measurable transferable **decision-quality** signal: on the "
            f"survival-decoupled metric `{gate_metric}`, end-of-life policies "
            f"robustly beat init in {agg['robust_positive_profiles']}. This is the "
            "signal P2-P4 are designed to transfer, so proceed to the inheritance "
            "experiment in this regime."
        )
    else:
        lines.append(
            "**No survival-decoupled transferable signal** in this regime: the "
            f"decision metric `{gate_metric}` does not clear the robustness gate "
            "(or the policy barely drifts). The end-of-life policy may keep the "
            "agent alive longer, but it does not make robustly better fine-grained "
            "decisions - which is the quantity richer inheritance (P2-P4) would "
            "carry. Per result #1/#2 in the inherited-payload design, richer "
            "payloads are unlikely to help here; record the negative before "
            "building P2-P4."
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--environment", type=str, default="development")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=DEFAULT_PROFILES,
        choices=list(STABLE_SUB_PROFILES),
        metavar="PROFILE",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output-dir", type=str, default="experiments/transferable_signal"
    )
    parser.add_argument("--num-steps", type=int, default=3000)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--max-population", type=int, default=64)
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--probe-capture-start", type=int, default=20)
    parser.add_argument("--probe-capture-end", type=int, default=120)
    parser.add_argument("--max-probes", type=int, default=256)
    parser.add_argument("--min-lifespan", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--eval-steps", type=int, default=150)
    parser.add_argument(
        "--reward-ages",
        nargs="+",
        type=int,
        default=[10, 25, 50],
        help=(
            "Agent ages (steps since birth) at which to snapshot net reward. "
            "Early-age reward isolates decision quality from survival length."
        ),
    )
    parser.add_argument(
        "--eval-policy",
        choices=["weighted", "greedy"],
        default="weighted",
        help=(
            "Action policy for both eval arms. 'weighted' = the real "
            "softmax(Q) x action_weights sampling the agent uses (non-degenerate "
            "baseline; a random-init net collapses to the chromosome-prior P0 "
            "policy). 'greedy' = pure argmax (degenerate for an untrained net)."
        ),
    )
    parser.add_argument(
        "--gate-metric",
        type=str,
        default="reward_age_10_budget",
        help=(
            "Budget metric the gate keys off. Default is the survival-decoupled "
            "early-age net reward (decision quality), matching #904's A/B. "
            "Examples: reward_age_10_budget, reward_rate_budget, reward_budget."
        ),
    )
    parser.add_argument(
        "--cross-eval-profile",
        type=str,
        default=None,
        choices=list(STABLE_SUB_PROFILES),
        help=(
            "Evaluate every harvested policy in this (different) ecology instead "
            "of its training profile - the cross-ecology overfit discriminator."
        ),
    )
    parser.add_argument("--drift-threshold", type=float, default=0.05)
    parser.add_argument(
        "--skip-rollout",
        action="store_true",
        help="Skip Tier-2 greedy rollouts (Tier-1 drift only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip cells whose {output_dir}/stable_{profile}/seed_{seed}/"
            "cell_result.json checkpoint already exists."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned (profile, seed) matrix and regime, then exit.",
    )
    return parser


def _print_dry_run(args: argparse.Namespace, output_dir: Path) -> None:
    print("Transferable-signal precondition gate - DRY RUN")
    print(f"  output_dir : {output_dir}")
    print(f"  profiles   : {args.profiles}")
    print(f"  seeds      : {args.seeds}")
    print(f"  regime     : population={args.population}, max_pop={args.max_population}, "
          f"num_steps={args.num_steps}")
    print(f"  probes     : capture [{args.probe_capture_start}, "
          f"{args.probe_capture_end}], max {args.max_probes}")
    print(f"  harvest    : min_lifespan={args.min_lifespan}, top_k={args.top_k}")
    print(f"  rollout    : eval_seeds={args.eval_seeds}, eval_steps={args.eval_steps}, "
          f"policy={args.eval_policy}, skip={args.skip_rollout}")
    print(f"  metrics    : reward_ages={args.reward_ages}, gate_metric={args.gate_metric}, "
          f"cross_eval_profile={args.cross_eval_profile}")
    print(f"  total cells: {len(args.profiles) * len(args.seeds)}")
    for profile in args.profiles:
        print(f"  ecology stable_{profile}: {STABLE_SUB_PROFILES[profile]}")


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)

    if args.dry_run:
        _print_dry_run(args, output_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    cells: List[Dict[str, Any]] = []
    total = len(args.profiles) * len(args.seeds)
    idx = 0
    for profile in args.profiles:
        for seed in args.seeds:
            idx += 1
            print(f"[{idx}/{total}] cell profile={profile} seed={seed} ...")
            cell = run_cell(profile, seed, args)
            cells.append(cell)
            gate_budget = cell.get("budgets", {}).get(args.gate_metric)
            print(
                f"    probes={cell['n_probes']} harvested={cell['n_harvested']} "
                f"drift_disagree={_fmt(cell['drift'].get('argmax_disagreement'))} "
                f"reward_budget={_fmt(cell['reward_budget'])} "
                f"{args.gate_metric}={_fmt(gate_budget)}"
            )

    agg = aggregate(cells, args.drift_threshold, gate_metric=args.gate_metric)
    summary = {
        "config": {
            "environment": args.environment,
            "profiles": args.profiles,
            "seeds": args.seeds,
            "num_steps": args.num_steps,
            "population": args.population,
            "max_population": args.max_population,
            "eval_seeds": args.eval_seeds,
            "eval_steps": args.eval_steps,
            "eval_policy": args.eval_policy,
            "reward_ages": args.reward_ages,
            "gate_metric": args.gate_metric,
            "cross_eval_profile": args.cross_eval_profile,
            "min_lifespan": args.min_lifespan,
            "top_k": args.top_k,
        },
        "cells": cells,
        "aggregate": agg,
    }

    summary_path = output_dir / "signal_budget_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    md_path = output_dir / "signal_budget_summary.md"
    md_path.write_text(build_markdown(summary), encoding="utf-8")

    print(f"\nGate verdict: {agg['gate']}")
    print(f"  summary JSON : {summary_path}")
    print(f"  summary MD   : {md_path}")
    return 0


if __name__ == "__main__":
    _configure_torch_threads()
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    raise SystemExit(main())
