"""Tests for EnvironmentalFactorsConfig validation behavior."""

import pytest

from farm.config.config import EnvironmentalFactorsConfig


def test_environmental_factors_accepts_in_range_values():
    cfg = EnvironmentalFactorsConfig(
        temperature=0.2,
        moisture=0.4,
        light=0.6,
        soil_quality=0.8,
        optimal_temperature=0.3,
        optimal_moisture=0.5,
        optimal_light=0.7,
        optimal_soil=0.9,
        tolerance_width=0.1,
    )
    assert cfg.temperature == pytest.approx(0.2)
    assert cfg.optimal_soil == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("temperature", -0.1),
        ("temperature", 1.1),
        ("moisture", -0.1),
        ("moisture", 1.1),
        ("light", -0.1),
        ("light", 1.1),
        ("soil_quality", -0.1),
        ("soil_quality", 1.1),
        ("optimal_temperature", -0.1),
        ("optimal_temperature", 1.1),
        ("optimal_moisture", -0.1),
        ("optimal_moisture", 1.1),
        ("optimal_light", -0.1),
        ("optimal_light", 1.1),
        ("optimal_soil", -0.1),
        ("optimal_soil", 1.1),
    ],
)
def test_environmental_factors_rejects_out_of_range_values(field_name, bad_value):
    kwargs = {field_name: bad_value}
    with pytest.raises(ValueError, match=field_name):
        EnvironmentalFactorsConfig(**kwargs)


def test_environmental_factors_rejects_non_positive_tolerance_width():
    with pytest.raises(ValueError, match="tolerance_width"):
        EnvironmentalFactorsConfig(tolerance_width=0.0)
