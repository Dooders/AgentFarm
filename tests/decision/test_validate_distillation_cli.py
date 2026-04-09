"""Unit tests for ``scripts/validate_distillation.py`` CLI validation."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "validate_distillation.py"


def _load_validate_module():
    spec = importlib.util.spec_from_file_location("validate_distillation_mod", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = None


def _get_mod():
    global _mod
    if _mod is None:
        _mod = _load_validate_module()
    return _mod


def _default_cli_ns(**overrides: object) -> argparse.Namespace:
    base = {
        "min_action_agreement": 0.85,
        "max_kl_divergence": 0.5,
        "max_mse": 2.0,
        "min_cosine_similarity": 0.8,
        "max_param_ratio": 0.9,
        "max_mae": None,
        "max_latency_ratio": None,
        "min_robustness_action_agreement": None,
        "rollout_episodes": 0,
        "rollout_max_steps": 50,
        "max_relative_return_drop": None,
        "n_states": 1000,
        "latency_warmup": 5,
        "latency_repeats": 50,
        # sim rollout adapter fields
        "sim_rollout": False,
        "sim_rollout_episodes": 0,
        "sim_rollout_max_steps": 200,
        "sim_max_relative_return_drop": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_validate_cli_args_accepts_defaults():
    mod = _get_mod()
    mod._validate_cli_args(_default_cli_ns())


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("min_action_agreement", 1.1, "min_action_agreement"),
        ("max_kl_divergence", -0.1, "max_kl_divergence"),
        ("max_mse", -1.0, "max_mse"),
        ("min_cosine_similarity", 2.0, "min_cosine_similarity"),
        ("max_param_ratio", 0.0, "max_param_ratio"),
        ("max_mae", -0.01, "max_mae"),
        ("max_latency_ratio", 0.0, "max_latency_ratio"),
        ("min_robustness_action_agreement", -0.1, "min_robustness_action_agreement"),
        ("rollout_episodes", -1, "rollout_episodes"),
        ("rollout_max_steps", 0, "rollout_max_steps"),
        ("max_relative_return_drop", 1.5, "max_relative_return_drop"),
        ("n_states", 0, "n_states"),
        ("latency_warmup", -1, "latency_warmup"),
        ("sim_rollout_episodes", -1, "sim_rollout_episodes"),
        ("sim_rollout_max_steps", 0, "sim_rollout_max_steps"),
        ("sim_max_relative_return_drop", 1.5, "sim_max_relative_return_drop"),
    ],
)
def test_validate_cli_args_rejects_invalid(field: str, value: object, match: str):
    mod = _get_mod()
    kwargs = {field: value}
    # sim_max_relative_return_drop requires sim_rollout_episodes > 0 for validation to trigger
    if field == "sim_max_relative_return_drop" and isinstance(value, float) and value > 1.0:
        kwargs["sim_rollout_episodes"] = 5
        kwargs["sim_rollout"] = True
    with pytest.raises(ValueError, match=match):
        mod._validate_cli_args(_default_cli_ns(**kwargs))


# ---------------------------------------------------------------------------
# _load_env_factory tests
# ---------------------------------------------------------------------------


def test_load_env_factory_empty_returns_shim():
    """Empty sim_env_factory string should return a factory satisfying EpisodeEnvProtocol."""
    mod = _get_mod()
    factory = mod._load_env_factory(
        "",
        input_dim=4,
        output_dim=2,
        sim_rollout_max_steps=200,
        sim_rollout_base_seed=0,
    )
    assert callable(factory)
    env = factory()
    # Verify protocol compliance by actually calling both required methods
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    step_result = env.step(0)
    assert len(step_result) == 5  # (obs, reward, terminated, truncated, info)


def test_load_env_factory_shim_reset_returns_array():
    """Shim env reset must return (obs, info) tuple with array of correct size."""
    import numpy as np

    mod = _get_mod()
    factory = mod._load_env_factory(
        "",
        input_dim=4,
        output_dim=2,
        sim_rollout_max_steps=200,
        sim_rollout_base_seed=0,
    )
    env = factory()
    result = env.reset(seed=1)
    # Must return a (obs, info) tuple per EpisodeEnvProtocol
    assert isinstance(result, tuple)
    assert len(result) == 2
    obs, info = result
    assert hasattr(obs, "shape")
    assert obs.shape == (4,)
    assert isinstance(info, dict)


def test_load_env_factory_shim_step_returns_five_tuple():
    """Shim env step must return (obs, reward, terminated, truncated, info)."""
    mod = _get_mod()
    factory = mod._load_env_factory(
        "",
        input_dim=4,
        output_dim=2,
        sim_rollout_max_steps=200,
        sim_rollout_base_seed=0,
    )
    env = factory()
    env.reset(seed=0)
    result = env.step(0)
    assert isinstance(result, tuple)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_load_env_factory_invalid_format_raises():
    """Factory path without ':' separator must raise ValueError."""
    mod = _get_mod()
    with pytest.raises(ValueError, match="module.path:attr"):
        mod._load_env_factory(
            "invalid_no_colon",
            input_dim=4,
            output_dim=2,
            sim_rollout_max_steps=200,
            sim_rollout_base_seed=0,
        )


def test_load_env_factory_valid_import():
    """A valid 'module:attr' path to a callable should be resolved correctly."""
    import os

    mod = _get_mod()
    # Use os.getcwd as a trivially importable callable
    factory = mod._load_env_factory(
        "os:getcwd",
        input_dim=4,
        output_dim=2,
        sim_rollout_max_steps=200,
        sim_rollout_base_seed=0,
    )
    assert callable(factory)
    assert factory is os.getcwd


def test_load_env_factory_noncallable_raises():
    """Non-callable attribute should raise ValueError."""
    mod = _get_mod()
    with pytest.raises(ValueError, match="not callable"):
        mod._load_env_factory(
            "os:sep",
            input_dim=4,
            output_dim=2,
            sim_rollout_max_steps=200,
            sim_rollout_base_seed=0,
        )


def test_load_env_factory_shim_respects_max_steps():
    """Shim env must truncate according to sim_rollout_max_steps."""
    mod = _get_mod()
    factory = mod._load_env_factory(
        "",
        input_dim=4,
        output_dim=2,
        sim_rollout_max_steps=3,
        sim_rollout_base_seed=0,
    )
    env = factory()
    env.reset(seed=0)
    # First two steps should not truncate; third should.
    assert env.step(0)[3] is False
    assert env.step(0)[3] is False
    assert env.step(0)[3] is True


def test_load_env_factory_shim_mdp_seed_changes_dynamics():
    """SeededLinearMDP weights depend on sim_rollout_base_seed (matches PolicyRolloutAdapter)."""
    mod = _get_mod()

    def _first_reward(seed: int) -> float:
        f = mod._load_env_factory(
            "",
            input_dim=4,
            output_dim=2,
            sim_rollout_max_steps=50,
            sim_rollout_base_seed=seed,
        )
        env = f()
        env.reset(seed=12345)
        return env.step(0)[1]

    assert _first_reward(0) != _first_reward(1)


def test_validate_cli_args_rejects_sim_rollout_flag_without_episodes():
    mod = _get_mod()
    with pytest.raises(ValueError, match="sim_rollout requires sim_rollout_episodes > 0"):
        mod._validate_cli_args(_default_cli_ns(sim_rollout=True, sim_rollout_episodes=0))


def test_validate_cli_args_rejects_sim_rollout_episodes_without_flag():
    mod = _get_mod()
    with pytest.raises(ValueError, match="sim_rollout_episodes > 0 requires --sim-rollout"):
        mod._validate_cli_args(_default_cli_ns(sim_rollout=False, sim_rollout_episodes=5))

