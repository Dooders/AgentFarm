"""Tests for SQLite pragma documentation helpers."""

from farm.database.pragma_docs import (
    analyze_pragma_value,
    get_pragma_info,
    get_pragma_profile,
)


def test_get_pragma_info_known():
    info = get_pragma_info("synchronous")
    assert "description" in info
    assert "values" in info


def test_get_pragma_info_unknown():
    info = get_pragma_info("nonexistent_pragma_xyz")
    assert info["description"] == "No information available"


def test_analyze_pragma_value_with_values_map():
    out = analyze_pragma_value("synchronous", "off")
    assert "description" in out


def test_analyze_pragma_value_unknown_value():
    out = analyze_pragma_value("synchronous", "not_a_real_mode")
    assert out["description"] == "Unknown value"


def test_analyze_pragma_value_performance_impact_only():
    out = analyze_pragma_value("cache_size", -1000)
    assert "description" in out


def test_get_pragma_profile_known():
    perf = get_pragma_profile("performance")
    assert perf["synchronous"] == "OFF"


def test_get_pragma_profile_unknown_defaults_to_balanced():
    bal = get_pragma_profile("balanced")
    assert get_pragma_profile("not_a_real_profile_name") == bal
