"""
Tests for Hydra multi-run and sweep functionality.

These tests validate:
- Multi-run parameter generation
- Sweep configuration loading
- Parameter combination generation
"""

import os
from pathlib import Path

import pytest

from farm.config.hydra_loader import HydraConfigLoader
from farm.config.sweep_runner import (
    generate_sweep_combinations,
    get_sweep_combinations,
    load_sweep_config,
    parse_sweep_range,
)


class TestSweepRangeParsing:
    """Test parsing of sweep range strings."""

    def test_parse_range_numeric(self):
        """Test parsing numeric range."""
        values = parse_sweep_range("range(0.0001, 0.001, 0.0001)")
        assert len(values) > 0
        assert 0.0001 in values
        assert 0.001 in values
        assert all(isinstance(v, float) for v in values)

    def test_parse_range_integer(self):
        """Test parsing integer range."""
        values = parse_sweep_range("range(10, 50, 10)")
        assert values == [10, 20, 30, 40, 50]

    def test_parse_choice(self):
        """Test parsing choice syntax."""
        values = parse_sweep_range("choice(0.9, 0.95, 0.99)")
        assert values == [0.9, 0.95, 0.99]

    def test_parse_comma_separated(self):
        """Test parsing comma-separated values."""
        values = parse_sweep_range("0.0001, 0.0005, 0.001")
        assert len(values) == 3
        assert 0.0001 in values

    def test_parse_single_value(self):
        """Test parsing single value."""
        values = parse_sweep_range("0.0001")
        assert values == [0.0001]


class TestSweepConfigLoading:
    """Test loading sweep configurations."""

    def test_load_sweep_config(self):
        """Test loading a sweep config file."""
        if not os.path.exists("conf/sweeps/learning_rate_sweep.yaml"):
            pytest.skip("Sweep config file not found")
        
        config = load_sweep_config("learning_rate_sweep")
        assert config is not None
        assert "hydra" in config or hasattr(config, "hydra")

    def test_load_nonexistent_sweep(self):
        """Test loading nonexistent sweep config."""
        with pytest.raises(FileNotFoundError):
            load_sweep_config("nonexistent_sweep")

    def test_get_sweep_combinations(self):
        """Test getting sweep combinations."""
        if not os.path.exists("conf/sweeps/learning_rate_sweep.yaml"):
            pytest.skip("Sweep config file not found")
        
        combinations = get_sweep_combinations("learning_rate_sweep")
        assert len(combinations) > 0
        assert all(isinstance(combo, list) for combo in combinations)
        assert all(len(combo) > 0 for combo in combinations)


class TestSweepCombinationGeneration:
    """Test generation of parameter combinations."""

    def test_single_parameter_sweep(self):
        """Test sweep with single parameter."""
        from omegaconf import DictConfig, OmegaConf
        
        sweep_config = OmegaConf.create({
            "hydra": {
                "sweeper": {
                    "params": {
                        "learning.learning_rate": "choice(0.0001, 0.0005, 0.001)",
                    }
                }
            }
        })
        
        combinations = generate_sweep_combinations(sweep_config)
        assert len(combinations) == 3
        assert all("learning.learning_rate=" in combo[0] for combo in combinations)

    def test_multiple_parameter_sweep(self):
        """Test sweep with multiple parameters."""
        from omegaconf import DictConfig, OmegaConf
        
        sweep_config = OmegaConf.create({
            "hydra": {
                "sweeper": {
                    "params": {
                        "learning.learning_rate": "choice(0.0001, 0.001)",
                        "learning.gamma": "choice(0.9, 0.99)",
                    }
                }
            }
        })
        
        combinations = generate_sweep_combinations(sweep_config)
        assert len(combinations) == 4  # 2 ? 2 = 4 combinations
        assert all(len(combo) == 2 for combo in combinations)

    def test_empty_sweep_config(self):
        """Test empty sweep config."""
        from omegaconf import DictConfig, OmegaConf
        
        sweep_config = OmegaConf.create({
            "hydra": {
                "sweeper": {
                    "params": {}
                }
            }
        })
        
        combinations = generate_sweep_combinations(sweep_config)
        assert len(combinations) == 1
        assert combinations[0] == []

    def test_no_sweeper_config(self):
        """Test config without sweeper."""
        from omegaconf import DictConfig, OmegaConf
        
        sweep_config = OmegaConf.create({})
        
        combinations = generate_sweep_combinations(sweep_config)
        assert len(combinations) == 1
        assert combinations[0] == []


class TestMultiRunIntegration:
    """Test multi-run integration (requires actual Hydra setup)."""

    @pytest.mark.skipif(
        not os.path.exists("conf/config.yaml"),
        reason="Hydra config not available"
    )
    def test_multiple_config_loads(self):
        """Test loading multiple configs with different overrides."""
        loader = HydraConfigLoader()
        
        configs = []
        overrides_list = [
            ["simulation_steps=100"],
            ["simulation_steps=200"],
            ["simulation_steps=300"],
        ]
        
        for overrides in overrides_list:
            config = loader.load_config(
                environment="development",
                overrides=overrides,
            )
            configs.append(config)
        
        # All configs should be valid
        assert all(isinstance(c, type(configs[0])) for c in configs)
        
        # Values should differ
        assert configs[0].simulation_steps == 100
        assert configs[1].simulation_steps == 200
        assert configs[2].simulation_steps == 300

    @pytest.mark.skipif(
        not os.path.exists("conf/sweeps/learning_rate_sweep.yaml"),
        reason="Sweep config not available"
    )
    def test_sweep_config_loading(self):
        """Test that sweep configs can be loaded."""
        loader = HydraConfigLoader()
        
        # Load base config
        base_config = loader.load_config(environment="development")
        
        # Try loading with sweep-style overrides
        sweep_config = loader.load_config(
            environment="development",
            overrides=["learning.learning_rate=0.0005"],
        )
        
        assert isinstance(sweep_config, type(base_config))
        assert sweep_config.learning.learning_rate == 0.0005


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
