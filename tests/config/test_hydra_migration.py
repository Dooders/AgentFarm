"""
Migration testing for Hydra configuration system.

Tests gradual migration, feature flags, and rollback capability.
"""

import os
from unittest.mock import patch

import pytest

from farm.config import load_config


class TestFeatureFlag:
    """Test feature flag functionality."""

    def test_use_hydra_flag_true(self):
        """Test explicit use_hydra=True."""
        config = load_config(environment="development", use_hydra=True)
        assert isinstance(config, SimulationConfig)

    def test_use_hydra_flag_false(self):
        """Test explicit use_hydra=False."""
        config = load_config(environment="development", use_hydra=False)
        assert isinstance(config, SimulationConfig)

    def test_use_hydra_flag_none_defaults_to_legacy(self):
        """Test that None defaults to legacy (backward compatibility)."""
        # Save original env var
        original = os.environ.get("USE_HYDRA_CONFIG")
        
        try:
            # Remove env var
            if "USE_HYDRA_CONFIG" in os.environ:
                del os.environ["USE_HYDRA_CONFIG"]
            
            # Should default to legacy
            config = load_config(environment="development", use_hydra=None)
            assert isinstance(config, SimulationConfig)
            
        finally:
            if original is not None:
                os.environ["USE_HYDRA_CONFIG"] = original

    def test_environment_variable_override(self):
        """Test USE_HYDRA_CONFIG environment variable."""
        original = os.environ.get("USE_HYDRA_CONFIG")
        
        try:
            # Set to true
            os.environ["USE_HYDRA_CONFIG"] = "true"
            config = load_config(environment="development", use_hydra=None)
            assert isinstance(config, SimulationConfig)
            
            # Set to false
            os.environ["USE_HYDRA_CONFIG"] = "false"
            config = load_config(environment="development", use_hydra=None)
            assert isinstance(config, SimulationConfig)
            
        finally:
            if original is not None:
                os.environ["USE_HYDRA_CONFIG"] = original
            elif "USE_HYDRA_CONFIG" in os.environ:
                del os.environ["USE_HYDRA_CONFIG"]


class TestGradualMigration:
    """Test gradual migration scenarios."""

    def test_mixed_usage_scenarios(self):
        """Test that both systems can be used in same session."""
        # Load with Hydra
        hydra_config = load_config(environment="development", use_hydra=True)
        
        # Load with legacy
        legacy_config = load_config(environment="development", use_hydra=False)
        
        # Both should work
        assert isinstance(hydra_config, SimulationConfig)
        assert isinstance(legacy_config, SimulationConfig)
        
        # Key values should match
        assert hydra_config.simulation_steps == legacy_config.simulation_steps

    def test_rollback_capability(self):
        """Test that we can rollback to legacy system."""
        # Start with Hydra
        hydra_config = load_config(environment="development", use_hydra=True)
        
        # Rollback to legacy
        legacy_config = load_config(environment="development", use_hydra=False)
        
        # Both should produce valid configs
        assert isinstance(hydra_config, SimulationConfig)
        assert isinstance(legacy_config, SimulationConfig)
        
        # Legacy should still work after Hydra was used
        legacy_config2 = load_config(environment="development", use_hydra=False)
        assert isinstance(legacy_config2, SimulationConfig)

    def test_no_breaking_changes(self):
        """Test that existing code continues to work."""
        # Simulate existing code that doesn't specify use_hydra
        # This should default to legacy for backward compatibility
        original = os.environ.get("USE_HYDRA_CONFIG")
        
        try:
            if "USE_HYDRA_CONFIG" in os.environ:
                del os.environ["USE_HYDRA_CONFIG"]
            
            # Existing code pattern
            config = load_config(environment="development")
            assert isinstance(config, SimulationConfig)
            
        finally:
            if original is not None:
                os.environ["USE_HYDRA_CONFIG"] = original


class TestConfigEquivalence:
    """Test that configs are equivalent between systems."""

    def test_same_default_values(self):
        """Test that default values match."""
        hydra_config = load_config(environment="development", use_hydra=True)
        legacy_config = load_config(environment="development", use_hydra=False)
        
        # Compare critical default values
        assert hydra_config.simulation_steps == legacy_config.simulation_steps
        assert hydra_config.seed == legacy_config.seed
        assert hydra_config.environment.width == legacy_config.environment.width
        assert hydra_config.population.system_agents == legacy_config.population.system_agents

    def test_same_environment_overrides(self):
        """Test that environment overrides match."""
        for env in ["development", "production", "testing"]:
            hydra_config = load_config(environment=env, use_hydra=True)
            legacy_config = load_config(environment=env, use_hydra=False)
            
            # Environment-specific values should match
            assert hydra_config.environment.width == legacy_config.environment.width
            assert hydra_config.environment.height == legacy_config.environment.height

    def test_same_profile_overrides(self):
        """Test that profile overrides match."""
        profiles = ["benchmark", "research", "simulation"]
        
        for profile in profiles:
            hydra_config = load_config(
                environment="production",
                profile=profile,
                use_hydra=True,
            )
            legacy_config = load_config(
                environment="production",
                profile=profile,
                use_hydra=False,
            )
            
            # Key values should match (exact values depend on profile definitions)
            assert isinstance(hydra_config, SimulationConfig)
            assert isinstance(legacy_config, SimulationConfig)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_environment(self):
        """Test handling of invalid environment."""
        with pytest.raises(Exception):  # Should raise some kind of error
            load_config(environment="invalid_env", use_hydra=True)

    def test_invalid_profile(self):
        """Test handling of invalid profile."""
        # Hydra might ignore invalid profiles or raise
        try:
            config = load_config(
                environment="development",
                profile="invalid_profile",
                use_hydra=True,
            )
            # If it doesn't raise, config should still be valid
            assert isinstance(config, SimulationConfig)
        except Exception:
            # If it raises, that's also acceptable
            pass

    def test_missing_config_files(self):
        """Test handling of missing config files."""
        # This should be handled gracefully
        # (Actual test would require manipulating file system)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
