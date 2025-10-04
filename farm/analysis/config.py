"""
Configuration constants for analysis modules.

This module centralizes magic numbers and configuration values used across
analysis modules, making them easier to tune and maintain.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpatialAnalysisConfig:
    """Configuration for spatial analysis module."""
    
    # Resource clustering threshold (units)
    resource_clustering_threshold: float = 20.0
    
    # Agent gathering range (units)
    gathering_range: float = 30.0
    
    # Hotspot detection threshold (multiples of standard deviation above mean)
    hotspot_threshold_stds: float = 1.0
    
    # Maximum number of clusters to try in K-means
    max_clusters: int = 10
    
    # Minimum data points required for clustering
    min_clustering_points: int = 3
    
    # Spatial density estimation bins
    density_bins: int = 20


@dataclass
class GenesisAnalysisConfig:
    """Configuration for genesis analysis module."""
    
    # Resource proximity threshold (units)
    resource_proximity_threshold: float = 30.0
    
    # Critical period end step
    critical_period_end: int = 100
    
    # Dominance metric weights
    auc_weight: float = 0.2
    recency_weighted_auc_weight: float = 0.3
    dominance_duration_weight: float = 0.2
    growth_trend_weight: float = 0.1
    final_ratio_weight: float = 0.2


@dataclass
class AgentAnalysisConfig:
    """Configuration for agent analysis module."""
    
    # Learning curve smoothing window
    learning_curve_window: int = 10
    
    # Number of clusters for behavior clustering
    behavior_clusters: int = 3
    
    # Performance score weights
    success_rate_weight: float = 0.4
    reward_rate_weight: float = 0.4
    lifespan_weight: float = 0.2
    
    # Top performers count
    top_performers_count: int = 5


@dataclass
class PopulationAnalysisConfig:
    """Configuration for population analysis module."""
    
    # Stability calculation window (steps)
    stability_window: int = 50
    
    # Growth rate calculation window (steps)
    growth_window: int = 20


@dataclass
class LearningAnalysisConfig:
    """Configuration for learning analysis module."""
    
    # Moving average window for learning curves
    moving_average_window: int = 10
    
    # Convergence detection window (steps)
    convergence_window: int = 20


@dataclass
class AnalysisGlobalConfig:
    """Global configuration for all analysis modules."""
    
    # Default database filename
    default_db_filename: str = "simulation.db"
    
    # Default output subdirectories
    output_subdirs: list = None
    
    # Plot DPI
    plot_dpi: int = 300
    
    # Plot style
    plot_style: str = "default"
    
    # Default figure size
    default_figsize: tuple = (8, 6)
    
    # Logging verbosity
    verbose_logging: bool = False
    
    # Cache analysis results
    enable_caching: bool = True
    
    # Parallel processing
    enable_parallel: bool = False
    max_workers: Optional[int] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.output_subdirs is None:
            self.output_subdirs = ["plots", "data", "reports"]


# Global configuration instances
spatial_config = SpatialAnalysisConfig()
genesis_config = GenesisAnalysisConfig()
agent_config = AgentAnalysisConfig()
population_config = PopulationAnalysisConfig()
learning_config = LearningAnalysisConfig()
global_config = AnalysisGlobalConfig()


def reset_to_defaults():
    """Reset all configurations to default values."""
    global spatial_config, genesis_config, agent_config, population_config, learning_config, global_config
    
    spatial_config = SpatialAnalysisConfig()
    genesis_config = GenesisAnalysisConfig()
    agent_config = AgentAnalysisConfig()
    population_config = PopulationAnalysisConfig()
    learning_config = LearningAnalysisConfig()
    global_config = AnalysisGlobalConfig()


def get_config(module_name: str):
    """Get configuration for a specific module.
    
    Args:
        module_name: Name of the analysis module
        
    Returns:
        Configuration instance for the module
        
    Raises:
        ValueError: If module name is not recognized
    """
    config_map = {
        'spatial': spatial_config,
        'genesis': genesis_config,
        'agents': agent_config,
        'population': population_config,
        'learning': learning_config,
        'global': global_config,
    }
    
    if module_name not in config_map:
        raise ValueError(
            f"Unknown module '{module_name}'. "
            f"Available modules: {', '.join(config_map.keys())}"
        )
    
    return config_map[module_name]
