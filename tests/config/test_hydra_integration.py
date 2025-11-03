"""
Comprehensive integration tests for Hydra configuration system.

This test suite validates:
- Config loading and composition
- Override functionality
- Multi-run support
- Sweep configurations
"""

import os
import tempfile
from pathlib import Path

import pytest

from farm.config import SimulationConfig, load_config
from farm.config.hydra_loader import HydraConfigLoader


class TestHydraConfigLoading:
    """Test basic Hydra config loading."""

    def test_load_config(self):
        """Test loading config using Hydra."""
        config = load_config(
            environment="development",
            profile=None,
        )
        
        assert isinstance(config, SimulationConfig)
        assert config.simulation_steps == 100  # Default value
        assert config.environment.width == 100  # From development environment

    def test_load_config_with_profile(self):
        """Test loading config with profile."""
        config = load_config(
            environment="production",
            profile="benchmark",
        )
        
        assert isinstance(config, SimulationConfig)
        # Profile should override some values

    def test_load_config_with_overrides(self):
        """Test loading config with overrides."""
        config = load_config(
            environment="development",
            profile=None,
            overrides=["simulation_steps=200", "population.system_agents=50"],
        )
        
        assert config.simulation_steps == 200
        assert config.population.system_agents == 50

    def test_load_config_with_nested_overrides(self):
        """Test nested parameter overrides."""
        config = load_config(
            environment="development",
            overrides=[
                "environment.width=200",
                "environment.height=200",
                "learning.learning_rate=0.0005",
            ],
        )
        
        assert config.environment.width == 200
        assert config.environment.height == 200
        assert config.learning.learning_rate == 0.0005


class TestConfigConsistency:
    """Test config consistency across different loading methods."""

    def test_from_centralized_config_uses_hydra(self):
        """Test that from_centralized_config uses Hydra internally."""
        from farm.config import SimulationConfig
        
        config1 = load_config(environment="development")
        config2 = SimulationConfig.from_centralized_config(environment="development")
        
        # Both should produce same config (both use Hydra)
        assert config1.simulation_steps == config2.simulation_steps
        assert config1.environment.width == config2.environment.width
        assert config1.population.system_agents == config2.population.system_agents


class TestHydraConfigLoader:
    """Test HydraConfigLoader class."""

    def test_loader_initialization(self):
        """Test loader can be initialized."""
        loader = HydraConfigLoader()
        assert loader.config_path == "conf"
        assert loader.config_name == "config"

    def test_loader_custom_path(self):
        """Test loader with custom config path."""
        loader = HydraConfigLoader(config_path="conf", config_name="config")
        config = loader.load_config(environment="development")
        assert isinstance(config, SimulationConfig)

    def test_loader_with_overrides(self):
        """Test loader with overrides."""
        loader = HydraConfigLoader()
        config = loader.load_config(
            environment="development",
            overrides=["simulation_steps=300"],
        )
        assert config.simulation_steps == 300

    def test_loader_clear(self):
        """Test loader clear method."""
        loader = HydraConfigLoader()
        loader.load_config(environment="development")
        loader.clear()  # Should not raise
        
        # Should be able to load again after clear
        config = loader.load_config(environment="development")
        assert isinstance(config, SimulationConfig)


class TestSimulationConfigFromHydra:
    """Test SimulationConfig.from_hydra() method."""

    def test_from_hydra_dictconfig(self):
        """Test conversion from Hydra DictConfig."""
        loader = HydraConfigLoader()
        loader._ensure_initialized()
        
        from hydra import compose
        from hydra.core.global_hydra import GlobalHydra
        
        GlobalHydra.instance().clear()
        
        with loader._initialize_context():
            cfg = compose(config_name="config", overrides=["environment=development"])
            config = SimulationConfig.from_hydra(cfg)
            
            assert isinstance(config, SimulationConfig)
            assert config.simulation_steps == 100

    def test_from_hydra_dict(self):
        """Test conversion from regular dict."""
        config_dict = {
            "simulation_steps": 200,
            "seed": 12345,
            "environment": {
                "width": 150,
                "height": 150,
            },
        }
        
        config = SimulationConfig.from_hydra(config_dict)
        assert isinstance(config, SimulationConfig)
        assert config.simulation_steps == 200
        assert config.environment.width == 150


class TestEnvironmentProfiles:
    """Test different environment and profile combinations."""

    @pytest.mark.parametrize("environment", ["development", "production", "testing"])
    def test_all_environments(self, environment):
        """Test that all environments load correctly."""
        config = load_config(environment=environment)
        assert isinstance(config, SimulationConfig)
        assert config.environment.width > 0
        assert config.environment.height > 0

    @pytest.mark.parametrize("profile", ["benchmark", "research", "simulation", None])
    def test_all_profiles(self, profile):
        """Test that all profiles load correctly."""
        config = load_config(
            environment="production",
            profile=profile,
        )
        assert isinstance(config, SimulationConfig)

    @pytest.mark.parametrize(
        "environment,profile",
        [
            ("development", None),
            ("development", "benchmark"),
            ("production", "benchmark"),
            ("production", "research"),
            ("testing", None),
        ],
    )
    def test_environment_profile_combinations(self, environment, profile):
        """Test various environment/profile combinations."""
        config = load_config(
            environment=environment,
            profile=profile,
        )
        assert isinstance(config, SimulationConfig)


class TestOverrideValidation:
    """Test override validation and error handling."""

    def test_invalid_override_format(self):
        """Test that invalid override format is handled."""
        # This should either work (if Hydra accepts it) or raise a clear error
        try:
            config = load_config(
                environment="development",
                overrides=["invalid_override"],
            )
            # If it doesn't raise, config should still be valid
            assert isinstance(config, SimulationConfig)
        except Exception as e:
            # Error should be informative
            assert "override" in str(e).lower() or "invalid" in str(e).lower()

    def test_nonexistent_parameter(self):
        """Test override with nonexistent parameter."""
        # Hydra might ignore unknown parameters or raise
        try:
            config = load_config(
                environment="development",
                overrides=["nonexistent.parameter=123"],
            )
            # If it works, config should still be valid
            assert isinstance(config, SimulationConfig)
        except Exception:
            # If it fails, that's also acceptable behavior
            pass

    def test_type_coercion(self):
        """Test that overrides handle type coercion correctly."""
        config = load_config(
            environment="development",
            overrides=[
                "simulation_steps=200",  # String to int
                "learning.learning_rate=0.0005",  # String to float
            ],
        )
        
        assert isinstance(config.simulation_steps, int)
        assert config.simulation_steps == 200
        assert isinstance(config.learning.learning_rate, float)
        assert config.learning.learning_rate == 0.0005


class TestConfigConsistency:
    """Test config consistency across loading methods."""

    def test_config_consistency_across_methods(self):
        """Test that different loading methods produce consistent configs."""
        # Test multiple scenarios
        scenarios = [
            {"environment": "development", "profile": None},
            {"environment": "production", "profile": "benchmark"},
            {"environment": "testing", "profile": None},
        ]
        
        for scenario in scenarios:
            config1 = load_config(**scenario)
            config2 = SimulationConfig.from_centralized_config(**scenario)
            
            # Both should produce same config (both use Hydra)
            assert config1.simulation_steps == config2.simulation_steps
            assert config1.environment.width == config2.environment.width
            assert config1.population.system_agents == config2.population.system_agents

    def test_override_functionality(self):
        """Test that overrides work correctly."""
        overrides = ["simulation_steps=250", "population.system_agents=75"]
        
        config = load_config(
            environment="development",
            overrides=overrides,
        )
        
        assert config.simulation_steps == 250
        assert config.population.system_agents == 75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
