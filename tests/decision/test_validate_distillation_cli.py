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
    ],
)
def test_validate_cli_args_rejects_invalid(field: str, value: object, match: str):
    mod = _get_mod()
    kwargs = {field: value}
    with pytest.raises(ValueError, match=match):
        mod._validate_cli_args(_default_cli_ns(**kwargs))
