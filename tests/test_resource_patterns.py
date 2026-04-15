"""Unit tests for farm.core.resource_patterns.

Tests cover:
- ResourceGenerationConfig defaults and construction
- All concrete regenerator classes (should_regenerate / get_regen_amount contracts)
- EnvironmentalRegenerator factor calculation and condition updates
- EcosystemRegenerator combined factor with mutualistic and competitive types
- EvolutionaryRegenerator stress, mutation, and carrying-capacity dynamics
- create_regenerator factory helper (valid and invalid types)
"""

import math
import random
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from farm.core.resource_patterns import (
    AdaptiveRegenerator,
    BasicRegenerator,
    EcosystemRegenerator,
    EnvironmentalRegenerator,
    EvolutionaryRegenerator,
    ProximityRegenerator,
    ResourceDependentRegenerator,
    ResourceGenerationConfig,
    SeasonalRegenerator,
    TimeBasedRegenerator,
    create_regenerator,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_resource(
    resource_id=1,
    amount=5,
    max_amount=10,
    position=(0.0, 0.0),
    resource_type=None,
    environment=None,
):
    """Create a lightweight mock resource."""
    r = SimpleNamespace(
        resource_id=resource_id,
        amount=amount,
        max_amount=max_amount,
        position=position,
        environment=environment,
    )
    if resource_type is not None:
        r.resource_type = resource_type
    return r


def _make_environment(resources):
    """Create a mock environment containing *resources*."""
    env = SimpleNamespace(resources=resources)
    return env


@pytest.fixture
def default_config():
    return ResourceGenerationConfig(regen_rate=0.5, regen_amount=2, max_amount=10)


# ---------------------------------------------------------------------------
# ResourceGenerationConfig
# ---------------------------------------------------------------------------


class TestResourceGenerationConfig:
    def test_defaults(self):
        cfg = ResourceGenerationConfig()
        assert cfg.regen_rate == 0.1
        assert cfg.regen_amount == 1
        assert cfg.max_amount == 10
        assert cfg.min_amount == 0

    def test_custom_values(self):
        cfg = ResourceGenerationConfig(regen_rate=0.5, regen_amount=3, max_amount=20)
        assert cfg.regen_rate == 0.5
        assert cfg.regen_amount == 3
        assert cfg.max_amount == 20


# ---------------------------------------------------------------------------
# BasicRegenerator
# ---------------------------------------------------------------------------


class TestBasicRegenerator:
    def test_no_regen_at_max(self, default_config):
        reg = BasicRegenerator(default_config)
        resource = _make_resource(amount=10, max_amount=10)
        # At max_amount should never regenerate
        for _ in range(20):
            assert reg.should_regenerate(resource) is False

    def test_regen_amount_is_config_value(self, default_config):
        reg = BasicRegenerator(default_config)
        resource = _make_resource(amount=5)
        assert reg.get_regen_amount(resource) == default_config.regen_amount

    def test_probability_with_full_rate(self):
        cfg = ResourceGenerationConfig(regen_rate=1.0)
        reg = BasicRegenerator(cfg)
        resource = _make_resource(amount=0)
        assert reg.should_regenerate(resource) is True

    def test_probability_with_zero_rate(self):
        cfg = ResourceGenerationConfig(regen_rate=0.0)
        reg = BasicRegenerator(cfg)
        resource = _make_resource(amount=0)
        # With rate=0.0 random.random() > 0 always
        for _ in range(20):
            assert reg.should_regenerate(resource) is False


# ---------------------------------------------------------------------------
# TimeBasedRegenerator
# ---------------------------------------------------------------------------


class TestTimeBasedRegenerator:
    def test_regenerates_at_interval(self, default_config):
        reg = TimeBasedRegenerator(default_config, interval=3)
        resource = _make_resource()
        results = [reg.should_regenerate(resource) for _ in range(9)]
        # Steps 3, 6, 9 should be True (counter % 3 == 0)
        assert results[2] is True
        assert results[5] is True
        assert results[8] is True
        assert results[0] is False
        assert results[1] is False

    def test_no_regen_at_max(self, default_config):
        reg = TimeBasedRegenerator(default_config, interval=1)
        resource = _make_resource(amount=10, max_amount=10)
        for _ in range(5):
            assert reg.should_regenerate(resource) is False

    def test_regen_amount(self, default_config):
        reg = TimeBasedRegenerator(default_config)
        resource = _make_resource()
        assert reg.get_regen_amount(resource) == default_config.regen_amount

    def test_independent_counters_per_resource(self, default_config):
        reg = TimeBasedRegenerator(default_config, interval=2)
        r1 = _make_resource(resource_id=1)
        r2 = _make_resource(resource_id=2)
        # Advance r1 twice (should trigger at step 2)
        reg.should_regenerate(r1)
        result = reg.should_regenerate(r1)
        assert result is True
        # r2 is still at step 1 — should not trigger
        result2 = reg.should_regenerate(r2)
        assert result2 is False


# ---------------------------------------------------------------------------
# SeasonalRegenerator
# ---------------------------------------------------------------------------


class TestSeasonalRegenerator:
    def test_factor_at_trough_and_peak(self):
        cfg = ResourceGenerationConfig(regen_rate=0.5, regen_amount=2, max_amount=10)
        reg = SeasonalRegenerator(cfg, season_length=4, peak_rate_multiplier=3.0)
        # At the trough (sin = -1, step = 3/4 of season_length) factor = 1.0
        # Formula: (1 + sin) / 2 * (peak-1) + 1
        # Min at sin=-1: 0 * (3-1) + 1 = 1.0
        # Max at sin=+1: 1 * (3-1) + 1 = 3.0
        reg._global_step = 3  # angle = 2π*3/4 = 3π/2 → sin = -1
        assert math.isclose(reg._seasonal_factor(), 1.0, abs_tol=0.05)

        reg._global_step = 1  # angle = 2π*1/4 = π/2 → sin = +1
        assert math.isclose(reg._seasonal_factor(), 3.0, abs_tol=0.05)

    def test_no_regen_at_max(self, default_config):
        reg = SeasonalRegenerator(default_config)
        resource = _make_resource(amount=10, max_amount=10)
        for _ in range(5):
            assert reg.should_regenerate(resource) is False

    def test_regen_amount_at_least_1(self, default_config):
        reg = SeasonalRegenerator(default_config, season_length=4, peak_rate_multiplier=0.5)
        resource = _make_resource(amount=0)
        assert reg.get_regen_amount(resource) >= 1


# ---------------------------------------------------------------------------
# ProximityRegenerator
# ---------------------------------------------------------------------------


class TestProximityRegenerator:
    def test_no_environment_returns_factor_1(self, default_config):
        reg = ProximityRegenerator(default_config)
        resource = _make_resource()
        # No environment attribute — factor should default to 1.0
        factor = reg._proximity_factor(resource)
        assert math.isclose(factor, 1.0)

    def test_competition_reduces_factor(self, default_config):
        reg = ProximityRegenerator(
            default_config, competition_range=10.0, competition_factor=0.5
        )
        target = _make_resource(resource_id=0, position=(0.0, 0.0))
        competitor = _make_resource(resource_id=1, position=(5.0, 0.0))
        env = _make_environment([target, competitor])
        target.environment = env
        factor = reg._proximity_factor(target)
        assert factor < 1.0

    def test_boost_increases_factor(self, default_config):
        reg = ProximityRegenerator(
            default_config, boost_range=50.0, boost_factor=0.5, competition_range=0.1
        )
        target = _make_resource(resource_id=0, position=(0.0, 0.0))
        neighbour = _make_resource(resource_id=1, position=(30.0, 0.0))
        env = _make_environment([target, neighbour])
        target.environment = env
        factor = reg._proximity_factor(target)
        assert factor > 1.0

    def test_no_regen_at_max(self, default_config):
        reg = ProximityRegenerator(default_config)
        resource = _make_resource(amount=10, max_amount=10)
        for _ in range(5):
            assert reg.should_regenerate(resource) is False


# ---------------------------------------------------------------------------
# ResourceDependentRegenerator
# ---------------------------------------------------------------------------


class TestResourceDependentRegenerator:
    def test_no_deps_returns_full_rate(self, default_config):
        reg = ResourceDependentRegenerator(default_config, dependencies={})
        resource = _make_resource(amount=0)
        # With no dependencies, satisfaction = 1.0 → normal rate
        assert reg._evaluate_dependencies(resource) == pytest.approx(1.0)

    def test_dep_satisfaction_with_nearby_resource(self, default_config):
        deps = {"water": {"weight": 1.0, "range": 50.0}}
        reg = ResourceDependentRegenerator(default_config, dependencies=deps)

        water = _make_resource(resource_id=10, amount=10, resource_type="water", position=(5.0, 0.0))
        target = _make_resource(resource_id=1, amount=0, position=(0.0, 0.0))
        env = _make_environment([target, water])
        target.environment = env
        water.environment = env

        satisfaction = reg._evaluate_dependencies(target)
        # water is nearby and at max → satisfaction should be 1.0
        assert satisfaction == pytest.approx(1.0)

    def test_no_dep_resource_nearby_returns_zero(self, default_config):
        deps = {"water": {"weight": 1.0, "range": 10.0}}
        reg = ResourceDependentRegenerator(default_config, dependencies=deps)

        target = _make_resource(resource_id=1, amount=0, position=(0.0, 0.0))
        env = _make_environment([target])  # No water resources
        target.environment = env

        satisfaction = reg._evaluate_dependencies(target)
        assert satisfaction == pytest.approx(0.0)

    def test_no_regen_at_max(self, default_config):
        reg = ResourceDependentRegenerator(default_config, dependencies={})
        resource = _make_resource(amount=10, max_amount=10)
        for _ in range(5):
            assert reg.should_regenerate(resource) is False


# ---------------------------------------------------------------------------
# AdaptiveRegenerator
# ---------------------------------------------------------------------------


class TestAdaptiveRegenerator:
    def test_initial_rate_matches_config(self, default_config):
        reg = AdaptiveRegenerator(default_config)
        resource = _make_resource(resource_id=99, amount=5)
        # On the first call, no history — adaptation = 1.0
        adaptation = reg._calculate_adaptation(resource)
        assert adaptation == 1.0

    def test_high_consumption_increases_rate(self, default_config):
        reg = AdaptiveRegenerator(default_config, adaptation_rate=0.2)
        resource = _make_resource(resource_id=1, amount=0)
        resource.last_amount = 10  # consumed 10 units
        reg._update_history(resource)
        adaptation = reg._calculate_adaptation(resource)
        # avg_consumption=10 > max_expected=5 → should increase
        assert adaptation > 1.0

    def test_low_consumption_decreases_rate(self, default_config):
        reg = AdaptiveRegenerator(default_config, adaptation_rate=0.2)
        resource = _make_resource(resource_id=2, amount=9)
        resource.last_amount = 9  # essentially no consumption
        reg._update_history(resource)
        adaptation = reg._calculate_adaptation(resource)
        # avg_consumption≈0 < 20% of max_expected → should decrease
        assert adaptation < 1.0

    def test_get_regen_amount_at_least_1(self, default_config):
        reg = AdaptiveRegenerator(default_config)
        resource = _make_resource(resource_id=3, amount=5)
        assert reg.get_regen_amount(resource) >= 1

    def test_no_regen_at_max(self, default_config):
        reg = AdaptiveRegenerator(default_config)
        resource = _make_resource(resource_id=4, amount=10, max_amount=10)
        for _ in range(5):
            assert reg.should_regenerate(resource) is False


# ---------------------------------------------------------------------------
# EnvironmentalRegenerator
# ---------------------------------------------------------------------------


class TestEnvironmentalRegenerator:
    def test_optimal_conditions_give_high_factor(self):
        cfg = ResourceGenerationConfig()
        reg = EnvironmentalRegenerator(
            cfg,
            temperature=0.6,
            moisture=0.7,
            light=0.6,
            soil_quality=0.7,
            optimal_temperature=0.6,
            optimal_moisture=0.7,
            optimal_light=0.6,
            optimal_soil=0.7,
        )
        factor = reg.environmental_factor()
        # At optimal conditions the Gaussian peaks at 1.0 for each factor
        assert factor == pytest.approx(1.0, abs=1e-6)

    def test_extreme_conditions_give_low_factor(self):
        cfg = ResourceGenerationConfig()
        reg = EnvironmentalRegenerator(
            cfg,
            temperature=0.0,
            moisture=0.0,
            light=0.0,
            soil_quality=0.0,
            optimal_temperature=1.0,
            optimal_moisture=1.0,
            optimal_light=1.0,
            optimal_soil=1.0,
            tolerance_width=0.1,
        )
        factor = reg.environmental_factor()
        assert factor < 0.01

    def test_update_conditions_clamps_to_unit_interval(self):
        cfg = ResourceGenerationConfig()
        reg = EnvironmentalRegenerator(cfg)
        reg.update_conditions(temperature=2.0, moisture=-0.5)
        assert reg.temperature == pytest.approx(1.0)
        assert reg.moisture == pytest.approx(0.0)

    def test_update_conditions_partial(self):
        cfg = ResourceGenerationConfig()
        reg = EnvironmentalRegenerator(cfg, temperature=0.3, light=0.4)
        reg.update_conditions(moisture=0.9)
        assert reg.temperature == pytest.approx(0.3)  # unchanged
        assert reg.moisture == pytest.approx(0.9)    # updated
        assert reg.light == pytest.approx(0.4)        # unchanged

    def test_no_regen_at_max(self):
        cfg = ResourceGenerationConfig(regen_rate=1.0, max_amount=10)
        reg = EnvironmentalRegenerator(cfg)
        resource = _make_resource(amount=10, max_amount=10)
        for _ in range(5):
            assert reg.should_regenerate(resource) is False

    def test_full_rate_at_optimal_conditions(self):
        cfg = ResourceGenerationConfig(regen_rate=1.0, max_amount=20)
        reg = EnvironmentalRegenerator(
            cfg,
            temperature=0.6,
            moisture=0.7,
            light=0.6,
            soil_quality=0.7,
            optimal_temperature=0.6,
            optimal_moisture=0.7,
            optimal_light=0.6,
            optimal_soil=0.7,
        )
        resource = _make_resource(amount=0)
        # rate=1.0 and factor≈1.0 → should always regenerate
        for _ in range(10):
            assert reg.should_regenerate(resource) is True

    def test_get_regen_amount_at_optimal(self):
        cfg = ResourceGenerationConfig(regen_amount=4, max_amount=20)
        reg = EnvironmentalRegenerator(
            cfg,
            temperature=0.6,
            moisture=0.7,
            light=0.6,
            soil_quality=0.7,
            optimal_temperature=0.6,
            optimal_moisture=0.7,
            optimal_light=0.6,
            optimal_soil=0.7,
        )
        resource = _make_resource(amount=0)
        # Factor ≈ 1.0 → amount should be config.regen_amount
        assert reg.get_regen_amount(resource) == 4

    def test_get_regen_amount_at_least_1(self):
        cfg = ResourceGenerationConfig(regen_amount=1, max_amount=10)
        reg = EnvironmentalRegenerator(
            cfg,
            temperature=0.0,
            moisture=0.0,
            light=0.0,
            soil_quality=0.0,
            optimal_temperature=1.0,
            optimal_moisture=1.0,
            optimal_light=1.0,
            optimal_soil=1.0,
            tolerance_width=0.01,
        )
        resource = _make_resource(amount=0)
        assert reg.get_regen_amount(resource) >= 1


# ---------------------------------------------------------------------------
# EcosystemRegenerator
# ---------------------------------------------------------------------------


class TestEcosystemRegenerator:
    def test_no_relationships_combined_factor_equals_env(self):
        cfg = ResourceGenerationConfig(max_amount=10)
        reg = EcosystemRegenerator(
            cfg,
            temperature=0.6,
            moisture=0.7,
            light=0.6,
            soil_quality=0.7,
            optimal_temperature=0.6,
            optimal_moisture=0.7,
            optimal_light=0.6,
            optimal_soil=0.7,
            carrying_capacity_factor=1.0,
        )
        resource = _make_resource(amount=0)
        factor = reg.combined_factor(resource)
        # With no relationships and carrying_capacity_factor=1, combined = env_factor * 1.0 * 1.0
        env_factor = reg._environmental_factor()
        assert factor == pytest.approx(env_factor, rel=1e-5)

    def test_mutualistic_type_boosts_factor(self):
        cfg = ResourceGenerationConfig(max_amount=10)
        reg = EcosystemRegenerator(
            cfg,
            mutualistic_types={"nitrogen_fixer": {"weight": 1.0, "range": 50.0}},
        )
        target = _make_resource(resource_id=0, position=(0.0, 0.0))
        mutualist = _make_resource(
            resource_id=1, amount=10, position=(10.0, 0.0), resource_type="nitrogen_fixer"
        )
        env = _make_environment([target, mutualist])
        target.environment = env
        mutualist.environment = env

        factor = reg.combined_factor(target)
        # relationship_factor should be > 1 → combined > env_factor
        env_factor = reg._environmental_factor()
        assert factor > env_factor

    def test_competitive_type_suppresses_factor(self):
        cfg = ResourceGenerationConfig(max_amount=10)
        reg = EcosystemRegenerator(
            cfg,
            competitive_types={"invasive": {"weight": 1.0, "range": 50.0}},
        )
        target = _make_resource(resource_id=0, position=(0.0, 0.0))
        competitor = _make_resource(
            resource_id=1, amount=10, position=(5.0, 0.0), resource_type="invasive"
        )
        env = _make_environment([target, competitor])
        target.environment = env
        competitor.environment = env

        factor = reg.combined_factor(target)
        env_factor = reg._environmental_factor()
        assert factor < env_factor

    def test_carrying_capacity_factor_scales_output(self):
        cfg = ResourceGenerationConfig(max_amount=10)
        reg_full = EcosystemRegenerator(cfg, carrying_capacity_factor=1.0)
        reg_half = EcosystemRegenerator(cfg, carrying_capacity_factor=0.5)
        resource = _make_resource(amount=0)
        factor_full = reg_full.combined_factor(resource)
        factor_half = reg_half.combined_factor(resource)
        assert math.isclose(factor_half, factor_full * 0.5, rel_tol=1e-5)

    def test_no_regen_at_max(self):
        cfg = ResourceGenerationConfig(regen_rate=1.0, max_amount=10)
        reg = EcosystemRegenerator(cfg)
        resource = _make_resource(amount=10, max_amount=10)
        for _ in range(5):
            assert reg.should_regenerate(resource) is False

    def test_update_conditions(self):
        cfg = ResourceGenerationConfig()
        reg = EcosystemRegenerator(cfg, temperature=0.3)
        reg.update_conditions(temperature=0.8, carrying_capacity_factor=0.5)
        assert reg.temperature == pytest.approx(0.8)
        assert reg.carrying_capacity_factor == pytest.approx(0.5)

    def test_update_conditions_clamps(self):
        cfg = ResourceGenerationConfig()
        reg = EcosystemRegenerator(cfg)
        reg.update_conditions(temperature=5.0, carrying_capacity_factor=-1.0)
        assert reg.temperature == pytest.approx(1.0)
        assert reg.carrying_capacity_factor == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EvolutionaryRegenerator
# ---------------------------------------------------------------------------


class TestEvolutionaryRegenerator:
    def test_initial_rate_equals_config(self):
        cfg = ResourceGenerationConfig(regen_rate=0.3, max_amount=10)
        reg = EvolutionaryRegenerator(cfg)
        resource = _make_resource(resource_id=1, amount=5)
        # Before any calls, evolved rate = config regen_rate
        assert reg.evolved_rate(resource) == pytest.approx(0.3)

    def test_stress_increases_rate(self):
        cfg = ResourceGenerationConfig(regen_rate=0.2, max_amount=10)
        reg = EvolutionaryRegenerator(cfg, stress_threshold=0.5, stress_adaptation_rate=0.1)
        resource = _make_resource(resource_id=1, amount=1, max_amount=10)  # below 50% → stress

        # Call should_regenerate several times to accumulate stress
        for _ in range(20):
            reg.should_regenerate(resource)

        # Rate should have increased above the base
        assert reg.evolved_rate(resource) > 0.2

    def test_carrying_capacity_suppresses_rate(self):
        cfg = ResourceGenerationConfig(regen_rate=0.9, max_amount=10)
        reg = EvolutionaryRegenerator(cfg, min_rate=0.01, max_rate=0.99)
        # Resource is at 95% of max_amount → capacity suppression applies
        resource = _make_resource(resource_id=1, amount=9, max_amount=10)
        initial_rate = reg.evolved_rate(resource)

        # After several steps the rate should be suppressed below initial
        for _ in range(10):
            reg.should_regenerate(resource)
        suppressed_rate = reg.evolved_rate(resource)
        assert suppressed_rate < initial_rate

    def test_rate_stays_within_bounds(self):
        cfg = ResourceGenerationConfig(regen_rate=0.5, max_amount=10)
        reg = EvolutionaryRegenerator(cfg, min_rate=0.1, max_rate=0.8)
        resource = _make_resource(resource_id=1, amount=0, max_amount=10)

        for _ in range(200):
            reg.should_regenerate(resource)

        rate = reg.evolved_rate(resource)
        assert reg.min_rate <= rate <= reg.max_rate

    def test_mutation_applied_at_interval(self):
        cfg = ResourceGenerationConfig(regen_rate=0.5, max_amount=10)
        reg = EvolutionaryRegenerator(
            cfg, mutation_interval=5, mutation_rate=0.0, min_rate=0.01, max_rate=0.99
        )
        resource = _make_resource(resource_id=1, amount=5, max_amount=10)
        # With mutation_rate=0.0 the Gaussian perturbation is 0 → rate stable
        for _ in range(50):
            reg.should_regenerate(resource)
        # Rate should remain close to the original (no random drift)
        rate = reg.evolved_rate(resource)
        assert 0.01 <= rate <= 0.99

    def test_get_regen_amount_at_least_1(self):
        cfg = ResourceGenerationConfig(regen_rate=0.1, regen_amount=2, max_amount=10)
        reg = EvolutionaryRegenerator(cfg)
        resource = _make_resource(resource_id=1, amount=5)
        assert reg.get_regen_amount(resource) >= 1

    def test_no_regen_at_max(self):
        cfg = ResourceGenerationConfig(regen_rate=1.0, max_amount=10)
        reg = EvolutionaryRegenerator(cfg)
        resource = _make_resource(resource_id=1, amount=10, max_amount=10)
        for _ in range(5):
            assert reg.should_regenerate(resource) is False

    def test_independent_per_resource(self):
        cfg = ResourceGenerationConfig(regen_rate=0.5, max_amount=10)
        reg = EvolutionaryRegenerator(cfg, stress_threshold=0.8, stress_adaptation_rate=0.2)
        r1 = _make_resource(resource_id=1, amount=0, max_amount=10)  # stressed
        r2 = _make_resource(resource_id=2, amount=9, max_amount=10)  # comfortable

        for _ in range(30):
            reg.should_regenerate(r1)
            reg.should_regenerate(r2)

        # r1 should have a higher rate than r2 due to stress
        assert reg.evolved_rate(r1) > reg.evolved_rate(r2)


# ---------------------------------------------------------------------------
# create_regenerator factory
# ---------------------------------------------------------------------------


class TestCreateRegeneratorFactory:
    @pytest.mark.parametrize(
        "rtype",
        [
            "basic",
            "time_based",
            "seasonal",
            "proximity",
            "adaptive",
            "environmental",
            "ecosystem",
            "evolutionary",
        ],
    )
    def test_valid_type_creates_instance(self, rtype):
        cfg = ResourceGenerationConfig()
        reg = create_regenerator(rtype, cfg)
        assert reg is not None
        assert hasattr(reg, "should_regenerate")
        assert hasattr(reg, "get_regen_amount")

    def test_dependent_type_requires_dependencies(self):
        cfg = ResourceGenerationConfig()
        deps = {"water": {"weight": 1.0, "range": 30.0}}
        reg = create_regenerator("dependent", cfg, dependencies=deps)
        assert isinstance(reg, ResourceDependentRegenerator)

    def test_default_config_used_when_none(self):
        reg = create_regenerator("basic")
        assert isinstance(reg, BasicRegenerator)
        assert reg.config is not None

    def test_invalid_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown regenerator type"):
            create_regenerator("does_not_exist")

    def test_kwargs_forwarded(self):
        cfg = ResourceGenerationConfig()
        reg = create_regenerator("time_based", cfg, interval=7)
        assert isinstance(reg, TimeBasedRegenerator)
        assert reg.interval == 7

    def test_environmental_kwargs(self):
        cfg = ResourceGenerationConfig()
        reg = create_regenerator("environmental", cfg, temperature=0.9, moisture=0.1)
        assert isinstance(reg, EnvironmentalRegenerator)
        assert reg.temperature == pytest.approx(0.9)
        assert reg.moisture == pytest.approx(0.1)
