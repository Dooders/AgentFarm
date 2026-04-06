"""Tests for built-in analysis suite resolution."""

import pytest

from farm.analysis.exceptions import ConfigurationError
from farm.analysis.registry import registry
from farm.analysis.suites import (
    BUILTIN_SUITE_MODULES,
    list_builtin_suite_names,
    resolve_suite_module_names,
)


def test_list_builtin_suite_names_includes_full():
    names = list_builtin_suite_names()
    assert "full" in names
    assert "system_dynamics" in names
    assert "agent_behavior" in names
    assert "social" in names


def test_list_builtin_suite_names_is_sorted():
    names = list_builtin_suite_names()
    assert names == sorted(names), "list_builtin_suite_names() must return a fully sorted list"


def test_list_builtin_suite_names_full_position():
    names = list_builtin_suite_names()
    # 'full' should be placed in sorted order, not always at the end
    assert names.index("full") == sorted(names).index("full")


def test_resolve_explicit_modules_order_and_dedupe():
    out = resolve_suite_module_names(modules=["a", "b", "a", "c"])
    assert out == ["a", "b", "c"]


def test_resolve_system_dynamics_matches_spec():
    assert BUILTIN_SUITE_MODULES["system_dynamics"] == ("population", "resources", "temporal")
    assert resolve_suite_module_names(suite="system_dynamics") == [
        "population",
        "resources",
        "temporal",
    ]


def test_resolve_full_uses_registry():
    registry.clear()
    assert resolve_suite_module_names(suite="full") == []


def test_resolve_unknown_suite():
    with pytest.raises(ConfigurationError, match="Unknown analysis suite"):
        resolve_suite_module_names(suite="not_a_real_suite")


def test_resolve_requires_suite_or_modules():
    with pytest.raises(ConfigurationError, match="requires either"):
        resolve_suite_module_names()
