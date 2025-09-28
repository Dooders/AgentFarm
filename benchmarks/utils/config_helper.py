"""
Configuration helper utilities for benchmarks.
"""

from typing import Any, Dict, Optional

from config_hydra import HydraSimulationConfig


def configure_for_performance_with_persistence(
    config: Optional[HydraSimulationConfig] = None,
) -> HydraSimulationConfig:
    """
    Configure a simulation for optimal performance while maintaining data persistence.

    This configuration uses an in-memory database with persistence enabled,
    which provides a good balance between performance and data durability
    for post-simulation analysis.

    Parameters
    ----------
    config : HydraSimulationConfig, optional
        Existing configuration to modify. If None, a new configuration is created.

    Returns
    -------
    HydraSimulationConfig
        Configuration with in-memory database and persistence enabled
    """
    if config is None:
        config = HydraSimulationConfig()

    # Enable in-memory database with persistence
    config.use_in_memory_db = True
    config.persist_db_on_completion = True

    return config


def get_recommended_config(
    num_agents: int = 30,
    num_steps: int = 100,
    additional_params: Optional[Dict[str, Any]] = None,
) -> HydraSimulationConfig:
    """
    Get a recommended configuration for simulations that need post-simulation analysis.

    Parameters
    ----------
    num_agents : int
        Total number of agents (divided equally among agent types)
    num_steps : int
        Number of simulation steps to run
    additional_params : Dict[str, Any], optional
        Additional parameters to set on the configuration

    Returns
    -------
    HydraSimulationConfig
        Recommended configuration for simulations
    """
    config = HydraSimulationConfig()

    # Set basic simulation parameters
    config.width = 100
    config.height = 100

    config.system_agents = num_agents // 3
    config.independent_agents = num_agents // 3
    config.control_agents = num_agents - (
        2 * (num_agents // 3)
    )  # Ensure total is num_agents

    config.initial_resources = 20
    config.simulation_steps = num_steps

    # Set additional parameters
    if additional_params:
        for key, value in additional_params.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Configure for performance with persistence
    return configure_for_performance_with_persistence(config)


def print_config_recommendations() -> None:
    """
    Print recommendations for configuring simulations for post-simulation analysis.
    """
    print("=" * 80)
    print("RECOMMENDED CONFIGURATION FOR POST-SIMULATION ANALYSIS")
    print("=" * 80)
    print(
        "For optimal performance while maintaining data for post-simulation analysis,"
    )
    print("use an in-memory database with persistence enabled:")
    print()
    print("    config.use_in_memory_db = True")
    print("    config.persist_db_on_completion = True")
    print()
    print("This configuration provides:")
    print("  - 33.6% faster execution than disk-based database")
    print("  - Full data persistence for post-simulation analysis")
    print("  - Good balance between performance and data durability")
    print()
    print("You can use the helper function to apply these settings:")
    print()
    print(
        "    from benchmarks.utils.config_helper import configure_for_performance_with_persistence"
    )
    print("    config = configure_for_performance_with_persistence(config)")
    print("=" * 80)
