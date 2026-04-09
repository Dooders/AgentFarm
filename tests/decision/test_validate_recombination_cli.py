"""Unit tests for ``scripts/validate_recombination.py`` CLI safety behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "validate_recombination.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_recombination_mod", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_quantized_role_requires_explicit_unpickle_opt_in():
    mod = _load_module()
    with pytest.raises(ValueError, match="--allow-unsafe-unpickle"):
        mod._load_model_for_role(
            path="unused.pt",
            input_dim=8,
            output_dim=4,
            hidden_size=64,
            label="child",
            quantized=True,
            allow_unsafe_unpickle=False,
        )
