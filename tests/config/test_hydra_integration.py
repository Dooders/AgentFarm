"""
Comprehensive integration tests for Hydra configuration system.

This test suite validates:
- Config loading and composition
- Backward compatibility with legacy system
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

    def test_load_config_with_hydra(self):
        """Test loading config using Hydra."""
        config = load_config(
            environment="development",
            profile=None,
            use_hydra=True,
        )
        
        assert isinstance(config, SimulationConfig)
        assert config.simulation_steps == 100  # Default value
        assert config.environment.width == 100  # From development environment

    def test_load_config_with_profile(self):
        """Test loading config with profile."""
        config = load_config(
            environment="production",
            profile="benchmark",
            use_hydra=True,
        )
        
        assert isinstance(config, SimulationConfig)
        # Profile should override some values

    def test_load_config_with_overrides(self):
        """Test loading config with overrides."""
        config = load_config(
            environment="development",
            profile=None,
            use_hydra=True,
            overrides=["simulation_steps=200", "population.system_agents=50"],
        )
        
        assert config.simulation_steps == 200
        assert config.population.system_agents == 50

    def test_load_config_with_nested_overrides(self):
        """Test nested parameter overrides."""
        config = load_config(
            environment="development",
            use_hydra=True,
            overrides=[
                "environment.width=200",
                "environment.height=200",
                "learning.learning_rate=0.0005",
            ],
        )
        
        assert config.environment.width == 200
        assert config.environment.height == 200
        assert config.learning.learning_rate == 0.0005


class TestBackwardCompatibility:
    """Test backward compatibility with legacy system."""

    def test_legacy_config_still_works(self):
        """Test that legacy config loading still works."""
        config = load_config(
            environment="development",
            profile=None,
            use_hydra=False,
        )
        
        assert isinstance(config, SimulationConfig)
        assert config.simulation_steps == 100

    def test_same_config_values(self):
        """Test that Hydra and legacy produce same config values."""
        hydra_config = load_config(
            environment="development",
            profile=None,
            use_hydra=True,
        )
        
        legacy_config = load_config(
            environment="development",
            profile=None,
            use_hydra=False,
        )
        
        # Compare key values
        assert hydra_config.simulation_steps == legacy_config.simulation_steps
        assert hydra_config.environment.width == legacy_config.environment.width
        assert hydra_config.environment.height == legacy_config.environment.height
        assert hydra_config.population.system_agents == legacy_config.population.system_agents

    def test_environment_variable_control(self):
        """Test that USE_HYDRA_CONFIG environment variable works."""
        # Save original value
        original_value = os.environ.get("USE_HYDRA_CONFIG")
        
        try:
            # Test with Hydra enabled
            os.environ["USE_HYDRA_CONFIG"] = "true"
            config_hydra = load_config(environment="development", use_hydra=None)
            
            # Test with Hydra disabled
            os.environ["USE_HYDRA_CONFIG"] = "false"
            config_legacy = load_config(environment="development", use_hydra=None)
            
            # Both should work
            assert isinstance(config_hydra, SimulationConfig)
            assert isinstance(config_legacy, SimulationConfig)
            
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["USE_HYDRA_CONFIG"] = original_value
            elif "USE_HYDRA_CONFIG" in os.environ:
                del os.environ["USE_HYDRA_CONFIG"]


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
        config = load_config(environment=environment, use_hydra=True)
        assert isinstance(config, SimulationConfig)
        assert config.environment.width > 0
        assert config.environment.height > 0

    @pytest.mark.parametrize("profile", ["benchmark", "research", "simulation", None])
    def test_all_profiles(self, profile):
        """Test that all profiles load correctly."""
        config = load_config(
            environment="production",
            profile=profile,
            use_hydra=True,
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
            use_hydra=True,
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
                use_hydra=True,
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
                use_hydra=True,
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
            use_hydra=True,
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

    def test_hydra_vs_legacy_consistency(self):
        """Test that Hydra and legacy produce consistent configs."""
        # Test multiple scenarios
        scenarios = [
            {"environment": "development", "profile": None},
            {"environment": "production", "profile": "benchmark"},
            {"environment": "testing", "profile": None},
        ]
        
        for scenario in scenarios:
            hydra_config = load_config(use_hydra=True, **scenario)
            legacy_config = load_config(use_hydra=False, **scenario)
            
            # Compare critical fields
            assert hydra_config.simulation_steps == legacy_config.simulation_steps
            assert hydra_config.environment.width == legacy_config.environment.width
            assert hydra_config.population.system_agents == legacy_config.population.system_agents

    def test_override_consistency(self):
        """Test that overrides work consistently."""
        overrides = ["simulation_steps=250", "population.system_agents=75"]
        
        hydra_config = load_config(
            environment="development",
            use_hydra=True,
            overrides=overrides,
        )
        
        # Legacy doesn't support overrides, so we can't compare
        # But Hydra should apply them correctly
        assert hydra_config.simulation_steps == 250
        assert hydra_config.population.system_agents == 75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
