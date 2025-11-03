"""
Sweep runner for Hydra-based multi-run experiments.

This module provides functionality to run parameter sweeps programmatically
using Hydra configurations.
"""

import os
from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .config import SimulationConfig
from .hydra_loader import HydraConfigLoader


def parse_sweep_range(range_str: str) -> List[float]:
    """
    Parse a range string into a list of values.
    
    Supports:
    - range(start, end, step) - numeric range
    - choice(val1, val2, val3) - discrete choices
    
    Args:
        range_str: Range string like "range(0.0001, 0.001, 0.0001)"
        
    Returns:
        List of values
    """
    range_str = range_str.strip()
    
    if range_str.startswith("range("):
        # Parse range(start, end, step)
        content = range_str[6:-1]  # Remove "range(" and ")"
        parts = [p.strip() for p in content.split(",")]
        start = float(parts[0])
        end = float(parts[1])
        step = float(parts[2]) if len(parts) > 2 else 1.0
        
        values = []
        current = start
        while current <= end:
            values.append(current)
            current += step
        return values
    
    elif range_str.startswith("choice("):
        # Parse choice(val1, val2, val3)
        content = range_str[7:-1]  # Remove "choice(" and ")"
        parts = [p.strip() for p in content.split(",")]
        return [float(p) if "." in p or "e" in p.lower() else int(p) for p in parts]
    
    else:
        # Single value or comma-separated list
        parts = [p.strip() for p in range_str.split(",")]
        return [float(p) if "." in p or "e" in p.lower() else int(p) for p in parts]


def generate_sweep_combinations(sweep_config: DictConfig) -> List[List[str]]:
    """
    Generate all parameter combinations for a sweep.
    
    Args:
        sweep_config: Hydra config with sweep parameters
        
    Returns:
        List of override lists, each representing one run
    """
    sweeper_params = sweep_config.get("hydra", {}).get("sweeper", {}).get("params", {})
    
    if not sweeper_params:
        return [[]]
    
    # Parse all parameter ranges
    param_values = {}
    for param_name, range_str in sweeper_params.items():
        if isinstance(range_str, str):
            param_values[param_name] = parse_sweep_range(range_str)
        elif isinstance(range_str, (list, tuple)):
            param_values[param_name] = list(range_str)
        else:
            param_values[param_name] = [range_str]
    
    # Generate cartesian product
    import itertools
    
    param_names = list(param_values.keys())
    param_value_lists = [param_values[name] for name in param_names]
    
    combinations = []
    for combo in itertools.product(*param_value_lists):
        overrides = [f"{name}={value}" for name, value in zip(param_names, combo)]
        combinations.append(overrides)
    
    return combinations


def load_sweep_config(sweep_name: str, config_path: str = "conf") -> DictConfig:
    """
    Load a sweep configuration file.
    
    Args:
        sweep_name: Name of the sweep config (without .yaml extension)
        config_path: Path to config directory
        
    Returns:
        Hydra DictConfig with sweep parameters
    """
    sweep_path = os.path.join(config_path, "sweeps", f"{sweep_name}.yaml")
    
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(f"Sweep config not found: {sweep_path}")
    
    # Initialize Hydra and load sweep config
    GlobalHydra.instance().clear()
    
    with initialize_config_dir(config_path=config_path, version_base=None):
        # Load the sweep config
        cfg = compose(config_name=f"sweeps/{sweep_name}")
    
    return cfg


def get_sweep_combinations(sweep_name: str, config_path: str = "conf") -> List[List[str]]:
    """
    Get all parameter combinations for a sweep.
    
    Args:
        sweep_name: Name of the sweep config
        config_path: Path to config directory
        
    Returns:
        List of override lists, each representing one run
    """
    sweep_config = load_sweep_config(sweep_name, config_path)
    return generate_sweep_combinations(sweep_config)
