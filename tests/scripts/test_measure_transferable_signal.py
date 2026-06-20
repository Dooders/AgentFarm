"""Tests for scripts/measure_transferable_signal.py (the #904 precondition gate).

Fast unit tests cover the pure logic: the robustness verdict, cohort
aggregation / gate decision, probe-drift computation, the deterministic
pure-greedy override, profile-ecology application, and the disk-persistence /
single-agent guarantees of the config builders.

A single end-to-end smoke test (marked ``slow``) runs a tiny real simulation
to confirm probe capture, snapshot harvesting, and JSON/verdict schema. It is
excluded from the default ``pytest`` run (see ``pytest.ini``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.measure_transferable_signal import (  # noqa: E402
    _PolicySnapshots,
    _entropy,
    _greedy_decide,
    _kl,
    _softmax,
    _verdict,
    aggregate,
    build_eval_config,
    build_regime_config,
    compute_drift,
    _apply_profile_overrides,
)
from farm.config import SimulationConfig  # noqa: E402
from farm.runners.intrinsic_evolution_experiment import STABLE_SUB_PROFILES  # noqa: E402


# ── Verdict ──────────────────────────────────────────────────────────────────


def test_verdict_too_few_samples():
    out = _verdict([0.5])
    assert out["n"] == 1
    assert out["robust"] is False
    assert out["ci_excludes_zero"] is False


def test_verdict_robust_positive():
    out = _verdict([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
    assert out["ci_excludes_zero"] is True
    assert out["sign_agreement"] == 1.0
    assert out["robust"] is True
    assert out["mean_delta"] > 0


def test_verdict_mixed_signs_not_robust():
    out = _verdict([1.0, -1.0, 0.8, -0.9, 1.2, -1.1])
    # CI straddles zero; sign agreement well below 0.75.
    assert out["robust"] is False


def test_verdict_drops_nans():
    out = _verdict([1.0, float("nan"), 1.1, 0.9])
    assert out["n"] == 3


# ── Softmax / entropy / KL ───────────────────────────────────────────────────


def test_softmax_sums_to_one():
    p = _softmax(np.array([1.0, 2.0, 3.0]))
    assert math.isclose(p.sum(), 1.0, rel_tol=1e-9)
    assert np.argmax(p) == 2


def test_kl_self_is_zero():
    p = _softmax(np.array([0.2, 1.0, -0.5]))
    assert _kl(p, p) == pytest.approx(0.0, abs=1e-9)


def test_entropy_uniform_is_max():
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    assert _entropy(uniform) > _entropy(peaked)


# ── Aggregation / gate verdict ────────────────────────────────────────────────


def _cell(profile, seed, reward_budget, disagreement, survival=0.0):
    return {
        "profile": profile,
        "seed": seed,
        "n_probes": 256,
        "n_harvested": 5,
        "drift": {"argmax_disagreement": disagreement},
        "reward_budget": reward_budget,
        "survival_budget": survival,
        "per_policy": [],
    }


def test_aggregate_gate_pass():
    # Tight, same-sign positive budgets per seed + clear drift -> PASS.
    cells = [
        _cell("balanced", s, b, 0.6)
        for s, b in zip([42, 7, 19, 101, 137, 256], [10, 11, 9, 10.5, 9.5, 10])
    ]
    agg = aggregate(cells, drift_threshold=0.05)
    assert agg["gate"] == "PASS"
    assert "balanced" in agg["robust_positive_profiles"]
    assert agg["drift_nontrivial"] is True


def test_aggregate_gate_null_no_signal():
    # Budgets straddle zero -> not robust -> NULL even though drift is large.
    cells = [
        _cell("balanced", s, b, 0.9)
        for s, b in zip([42, 7, 19, 101, 137, 256], [1, -1, 0.5, -0.7, 1.2, -1.1])
    ]
    agg = aggregate(cells, drift_threshold=0.05)
    assert agg["gate"] == "NULL"
    assert agg["robust_positive_profiles"] == []


def test_aggregate_gate_null_no_drift():
    # Robustly positive budgets but the policy barely drifts -> NULL.
    cells = [
        _cell("balanced", s, b, 0.001)
        for s, b in zip([42, 7, 19, 101, 137, 256], [10, 11, 9, 10.5, 9.5, 10])
    ]
    agg = aggregate(cells, drift_threshold=0.05)
    assert agg["drift_nontrivial"] is False
    assert agg["gate"] == "NULL"


def _cell_with_budgets(profile, seed, budgets, disagreement=0.6):
    return {
        "profile": profile,
        "seed": seed,
        "n_probes": 256,
        "n_harvested": 5,
        "drift": {"argmax_disagreement": disagreement},
        "reward_budget": budgets.get("reward_budget", float("nan")),
        "survival_budget": budgets.get("survival_budget", float("nan")),
        "budgets": budgets,
        "per_policy": [],
    }


def test_aggregate_gate_keys_off_decision_metric_not_full_reward():
    """Full-episode reward robustly positive, but the survival-decoupled
    decision metric straddles zero -> gate must read NULL on the decision
    metric. This is the core #904 fix: a survival-driven win must not pass."""
    seeds = [42, 7, 19, 101, 137, 256]
    full = [800, 810, 790, 805, 795, 800]          # robust positive (survival)
    early = [5, -6, 3, -4, 6, -5]                  # straddles zero (decisions)
    cells = [
        _cell_with_budgets(
            "balanced", s,
            {"reward_budget": f, "survival_budget": 0.7, "reward_age_10_budget": e},
        )
        for s, f, e in zip(seeds, full, early)
    ]
    # Gate on the early-age decision metric -> NULL despite the huge full reward.
    agg = aggregate(cells, drift_threshold=0.05, gate_metric="reward_age_10_budget")
    assert agg["gate_metric"] == "reward_age_10_budget"
    assert agg["gate"] == "NULL"
    assert agg["robust_positive_profiles"] == []
    # The full-episode reward verdict is still reported (and is robust).
    assert agg["per_profile"]["balanced"]["full_reward_verdict"]["robust"] is True


def test_aggregate_gate_pass_on_decision_metric():
    seeds = [42, 7, 19, 101, 137, 256]
    early = [12, 11, 13, 10, 14, 12]
    cells = [
        _cell_with_budgets(
            "balanced", s,
            {"reward_budget": 800, "survival_budget": 0.7, "reward_age_10_budget": e},
        )
        for s, e in zip(seeds, early)
    ]
    agg = aggregate(cells, drift_threshold=0.05, gate_metric="reward_age_10_budget")
    assert agg["gate"] == "PASS"
    assert "balanced" in agg["robust_positive_profiles"]


def test_aggregate_unknown_gate_metric_falls_back_to_reward_budget():
    cells = [
        _cell("balanced", s, b, 0.6)
        for s, b in zip([42, 7, 19, 101, 137, 256], [10, 11, 9, 10.5, 9.5, 10])
    ]
    agg = aggregate(cells, drift_threshold=0.05, gate_metric="does_not_exist_budget")
    assert agg["gate_metric"] == "reward_budget"
    assert agg["gate"] == "PASS"


# ── Probe-state drift ─────────────────────────────────────────────────────────


class _FakeAlgo:
    """Minimal scratch evaluator: Q depends on the loaded scalar bias."""

    def __init__(self):
        self.bias = 0.0

    def load_model_state(self, state):
        self.bias = float(state["policy_state_dict"]["b"])

    def _policy_q_values(self, probe):
        base = float(np.sum(probe))
        return np.array([base, base + self.bias, base - 1.0], dtype=np.float64)


def test_compute_drift_detects_argmax_shift():
    probes = [np.ones((1, 4), dtype=np.float32) for _ in range(8)]
    snap = _PolicySnapshots(birth={"b": 0.0}, eol={"b": 5.0})
    snap.first_seen = 0
    snap.last_seen = 200
    drift = compute_drift(_FakeAlgo(), [snap], probes)
    assert drift["n_policies"] == 1
    assert drift["n_probes"] == 8
    # Bias 5 flips the argmax from action 0 to action 1 on every probe.
    assert drift["argmax_disagreement"] == pytest.approx(1.0)
    assert drift["max_q_drift"] > 0
    assert drift["softmax_kl"] > 0


def test_compute_drift_empty_inputs():
    drift = compute_drift(None, [], [])
    assert drift["n_policies"] == 0
    assert math.isnan(drift["argmax_disagreement"])


# ── Pure-greedy override determinism ──────────────────────────────────────────


class _StubGreedyAlgo:
    def __init__(self, q):
        self._q = np.asarray(q, dtype=np.float64)

    def _policy_q_values(self, state):
        return self._q

    def _select_greedy_action(self, logits, mask):
        masked = logits.astype(np.float64, copy=True)
        if mask is not None:
            masked[~mask] = -np.inf
        return int(np.argmax(masked))


class _StubDecisionModule:
    def __init__(self, q, num_actions):
        self.algorithm = _StubGreedyAlgo(q)
        self.num_actions = num_actions

    def _normalize_state_for_algorithm(self, state):
        return np.asarray(state, dtype=np.float32)

    def _create_action_mask(self, enabled_actions):
        if not enabled_actions:
            return np.ones(self.num_actions, dtype=bool)
        mask = np.zeros(self.num_actions, dtype=bool)
        mask[enabled_actions] = True
        return mask

    def _map_to_enabled_index(self, full_action, enabled_actions):
        if not enabled_actions:
            return int(full_action)
        return enabled_actions.index(full_action)


def test_greedy_decide_is_deterministic_and_argmax():
    dm = _StubDecisionModule(q=[0.1, 0.9, 0.3, 0.2], num_actions=4)
    state = np.zeros((1, 4), dtype=np.float32)
    a1 = _greedy_decide(dm, state, enabled_actions=None)
    a2 = _greedy_decide(dm, state, enabled_actions=None)
    assert a1 == a2 == 1  # argmax of the Q vector


def test_greedy_decide_respects_enabled_mask():
    dm = _StubDecisionModule(q=[0.9, 0.8, 0.3, 0.2], num_actions=4)
    state = np.zeros((1, 4), dtype=np.float32)
    # Action 0 (the global argmax) is disabled -> falls to action 1.
    idx = _greedy_decide(dm, state, enabled_actions=[1, 2, 3])
    assert idx == 0  # index within the enabled list, i.e. full action 1


def test_eval_policy_weighted_falls_through_greedy_overrides():
    """In 'weighted' eval mode on_decide returns None (fall through to the real
    action-weighted policy); in 'greedy' mode it returns a forced argmax."""
    from scripts.measure_transferable_signal import _REC

    _REC.reset()
    dm = _StubDecisionModule(q=[0.1, 0.9, 0.3, 0.2], num_actions=4)
    # Give the stub a _policy_q_values attr check the recorder relies on.
    assert hasattr(dm.algorithm, "_policy_q_values")
    state = np.zeros((1, 4), dtype=np.float32)

    _REC.begin_eval({"w": 0}, eval_policy="weighted")
    assert _REC.on_decide(dm, state, None) is None  # fall through

    _REC.begin_eval({"w": 0}, eval_policy="greedy")
    assert _REC.on_decide(dm, state, None) == 1  # forced argmax
    _REC.end_eval()
    _REC.reset()


# ── Config builders ───────────────────────────────────────────────────────────


def test_apply_profile_overrides_sets_ecology():
    config = SimulationConfig.from_centralized_config(environment="testing")
    _apply_profile_overrides(config, "buffered")
    overrides = STABLE_SUB_PROFILES["buffered"]
    assert config.agent_behavior.initial_resource_level == overrides[
        "initial_agent_resource_level"
    ]
    assert config.resources.initial_resources == overrides["initial_resource_count"]
    assert config.resources.resource_regen_rate == overrides["resource_regen_rate"]
    assert config.resources.resource_regen_amount == overrides["resource_regen_amount"]


def test_regime_config_is_disk_backed_and_small():
    config = build_regime_config(
        "balanced", environment="testing", population=3, max_population=20
    )
    assert config.database.use_in_memory_db is False
    assert config.population.independent_agents == 3
    assert config.population.system_agents == 0
    assert config.population.control_agents == 0
    assert config.population.max_population == 20


def test_eval_config_is_single_agent():
    config = build_eval_config("conservative", environment="testing")
    assert config.database.use_in_memory_db is False
    assert config.population.independent_agents == 1
    assert config.population.max_population == 1


def test_reproduction_is_blocked_in_train_and_eval_modes():
    """The reproduce no-op patch is the only reliable population cap.

    ``reproduce_action`` enforces no ``max_population`` cap and the action stays
    in every agent's set, so the gate blocks reproduction by patching
    ``AgentCore.reproduce`` while the recorder is active. Without this the
    population explodes (100+ agents on high-resource ecologies) and exhausts
    RAM / slows each step to seconds.
    """
    from scripts.measure_transferable_signal import _REC, install_instrumentation
    from farm.core.agent.core import AgentCore

    install_instrumentation()
    _REC.reset()

    class _DummyAgent:
        pass

    for mode in ("train", "eval"):
        _REC.mode = mode
        assert AgentCore.reproduce(_DummyAgent()) is False
    _REC.reset()


# ── End-to-end smoke (slow) ──────────────────────────────────────────────────


@pytest.mark.slow
def test_run_cell_end_to_end_smoke():
    from scripts.measure_transferable_signal import run_cell

    with TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            environment="testing",
            output_dir=tmp,
            num_steps=60,
            population=3,
            max_population=12,
            snapshot_interval=20,
            probe_capture_start=5,
            probe_capture_end=40,
            max_probes=32,
            min_lifespan=10,
            top_k=2,
            eval_seeds=[0],
            eval_steps=20,
            reward_ages=[5, 10],
            eval_policy="weighted",
            gate_metric="reward_age_10_budget",
            cross_eval_profile=None,
            skip_rollout=False,
            resume=False,
        )
        cell = run_cell("balanced", 42, args)

        assert cell["n_probes"] > 0
        assert "drift" in cell and "argmax_disagreement" in cell["drift"]
        assert isinstance(cell["reward_budget"], float)
        assert "budgets" in cell and "reward_age_10_budget" in cell["budgets"]
        assert "reward_rate_budget" in cell["budgets"]
        if cell["per_policy"]:
            p0 = cell["per_policy"][0]
            assert "reward_rate_delta" in p0 and "reward_age_10_delta" in p0
        # Checkpoint written for --resume.
        checkpoint = Path(tmp) / "stable_balanced" / "seed_42" / "cell_result.json"
        assert checkpoint.is_file()
        reloaded = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert reloaded["profile"] == "balanced"
        # Training DB persisted to disk (not in-memory).
        assert list((Path(tmp) / "stable_balanced" / "seed_42").glob("*.db"))
