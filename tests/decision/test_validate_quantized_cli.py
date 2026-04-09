"""Unit tests for ``scripts/validate_quantized.py`` CLI safety guards."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "validate_quantized.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_quantized_mod", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_guard_rejects_without_opt_in():
    mod = _load_module()
    with pytest.raises(ValueError, match="--allow-unsafe-unpickle"):
        mod._ensure_unsafe_unpickle_allowed(False)


def test_guard_allows_with_opt_in():
    mod = _load_module()
    mod._ensure_unsafe_unpickle_allowed(True)
