"""
Modern configuration system for AgentFarm.

This module provides a clean, profile-based configuration system that eliminates
duplication and provides high configurability through composition.
"""

import copy
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from farm.core.profiles import (
    DQNProfile, AgentBehaviorProfile, EnvironmentProfile,
    get_dqn_profile, get_behavior_profile, get_environment_profile
)


@dataclass
class ActionConfig:
    """Configuration for a specific action type.
    
    Uses a profile as base and allows specific overrides for rewards/costs.
    """
    dqn_profile: str = "default"
    rewards: Dict[str, float] = field(default_factory=dict)
    costs: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    
    def get_dqn_config(self) -> DQNProfile:
        """Get the resolved DQN configuration with overrides applied."""
        profile = get_dqn_profile(self.dqn_profile)
        return profile.with_overrides(**self.overrides)


@dataclass
class AgentTypeConfig:
    """Configuration for a specific agent type."""
    behavior_profile: str = "balanced"
    dqn_profile: str = "default" 
    count: int = 10
    initial_resources: int = 5
    overrides: Dict[str, Any] = field(default_factory=dict)
    action_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_behavior_config(self) -> AgentBehaviorProfile:
        """Get the resolved behavior configuration."""
        profile = get_behavior_profile(self.behavior_profile)
        return profile.with_overrides(**self.overrides)


@dataclass
class VisualizationConfig:
    """Visualization settings (kept minimal)."""
    canvas_size: Tuple[int, int] = (400, 400)
    background_color: str = "black"
    agent_colors: Dict[str, str] = field(default_factory=lambda: {
        "SystemAgent": "blue",
        "IndependentAgent": "red", 
        "ControlAgent": "#DAA520"
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class SimulationConfig:
    """
    Main simulation configuration using profile-based composition.
    
    This eliminates the massive duplication from the old system by using
    reusable profiles that can be mixed and matched.
    """
    
    # Profiles (specify by name, resolved at runtime)
    environment_profile: str = "default"
    default_dqn_profile: str = "default"
    
    # Agent configurations
    agents: Dict[str, AgentTypeConfig] = field(default_factory=lambda: {
        "system": AgentTypeConfig(behavior_profile="cooperative"),
        "independent": AgentTypeConfig(behavior_profile="balanced"),
        "control": AgentTypeConfig(behavior_profile="balanced")
    })
    
    # Action configurations (only specify if different from defaults)
    actions: Dict[str, ActionConfig] = field(default_factory=lambda: {
        "move": ActionConfig(
            costs={"base": -0.1},
            rewards={"approach_resource": 0.3, "retreat_penalty": -0.2}
        ),
        "attack": ActionConfig(
            costs={"base": -0.2}, 
            rewards={"success": 1.0, "kill": 5.0, "failure_penalty": -0.3},
            thresholds={"defense": 0.3, "defense_boost": 2.0}
        ),
        "gather": ActionConfig(
            rewards={"success": 0.5, "failure_penalty": -0.1, "efficiency_bonus": 0.3},
            costs={"base": -0.05}
        ),
        "share": ActionConfig(
            rewards={"success": 0.3, "altruism_bonus": 0.2},
            costs={"failure_penalty": -0.1},
            thresholds={"range": 30.0, "min_amount": 1}
        ),
        "reproduce": ActionConfig(
            rewards={"success": 1.0, "offspring_survival": 0.5},
            costs={"failure_penalty": -0.2, "offspring_cost": 3},
            thresholds={"min_health": 0.5, "min_resources": 8}
        )
    })
    
    # Global simulation settings
    max_population: int = 3000
    simulation_steps: int = 1000
    starvation_threshold: int = 0
    max_starvation_time: int = 15
    
    # Combat settings
    starting_health: float = 100.0
    attack_range: float = 20.0
    perception_radius: int = 2
    
    # Other settings
    seed: Optional[int] = 1234567890
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Database settings
    use_in_memory_db: bool = False
    persist_db_on_completion: bool = True
    db_profile: str = "balanced"  # fast, balanced, safe
    
    # Performance optimization
    auto_optimize: bool = True
    
    def __post_init__(self):
        """Validate and optimize configuration after creation."""
        if self.auto_optimize:
            self._auto_optimize()
        self._validate()
    
    def _auto_optimize(self):
        """Automatically optimize config based on system resources."""
        if not PSUTIL_AVAILABLE:
            print("psutil not available - skipping auto-optimization")
            return
            
        available_mb = psutil.virtual_memory().available // (1024 * 1024)
        
        # Adjust DQN profiles based on available memory
        if available_mb < 4000:  # Less than 4GB
            print("Low memory detected - using memory efficient profiles")
            for action_config in self.actions.values():
                if action_config.dqn_profile == "default":
                    action_config.dqn_profile = "memory_efficient"
                    
        elif available_mb > 16000:  # More than 16GB  
            print("High memory detected - using high performance profiles")
            for action_config in self.actions.values():
                if action_config.dqn_profile == "default":
                    action_config.dqn_profile = "high_performance"
        
        # Adjust simulation size for very large worlds
        env_profile = get_environment_profile(self.environment_profile)
        world_size = env_profile.width * env_profile.height
        total_agents = sum(agent.count for agent in self.agents.values())
        
        if world_size > 40000 and total_agents < 50:
            print("Large world with few agents - consider increasing agent count")
            
    def _validate(self):
        """Validate configuration for common issues."""
        errors = []
        
        # Check agent counts
        total_agents = sum(agent.count for agent in self.agents.values())
        if total_agents == 0:
            errors.append("No agents configured")
        elif total_agents > self.max_population:
            errors.append(f"Total agents ({total_agents}) exceeds max_population ({self.max_population})")
            
        # Check profile references
        try:
            get_environment_profile(self.environment_profile)
        except ValueError as e:
            errors.append(f"Invalid environment profile: {e}")
            
        for agent_name, agent_config in self.agents.items():
            try:
                get_behavior_profile(agent_config.behavior_profile)
                get_dqn_profile(agent_config.dqn_profile)
            except ValueError as e:
                errors.append(f"Invalid profile for agent {agent_name}: {e}")
                
        for action_name, action_config in self.actions.items():
            try:
                get_dqn_profile(action_config.dqn_profile)
            except ValueError as e:
                errors.append(f"Invalid DQN profile for action {action_name}: {e}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    def get_environment_config(self) -> EnvironmentProfile:
        """Get the resolved environment configuration."""
        return get_environment_profile(self.environment_profile)
    
    def get_agent_config(self, agent_type: str) -> AgentTypeConfig:
        """Get configuration for a specific agent type."""
        if agent_type not in self.agents:
            raise ValueError(f"No configuration for agent type: {agent_type}")
        return self.agents[agent_type]
    
    def get_action_config(self, action_type: str) -> ActionConfig:
        """Get configuration for a specific action type.""" 
        if action_type not in self.actions:
            # Return default config if not specified
            return ActionConfig(dqn_profile=self.default_dqn_profile)
        return self.actions[action_type]
    
    def with_environment(self, profile: str, **overrides) -> "SimulationConfig":
        """Create a copy with a different environment profile."""
        new_config = copy.deepcopy(self)
        new_config.environment_profile = profile
        return new_config
    
    def with_agents(self, **agent_configs: AgentTypeConfig) -> "SimulationConfig":
        """Create a copy with different agent configurations."""
        new_config = copy.deepcopy(self)
        new_config.agents.update(agent_configs)
        return new_config
    
    def with_dqn_profile(self, profile: str) -> "SimulationConfig":
        """Create a copy with a different default DQN profile."""
        new_config = copy.deepcopy(self)
        new_config.default_dqn_profile = profile
        # Update all actions that use default profile
        for action_config in new_config.actions.values():
            if action_config.dqn_profile == "default":
                action_config.dqn_profile = profile
        return new_config
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "SimulationConfig":
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Create configuration from dictionary."""
        # Handle nested configurations
        if "agents" in data:
            agents_data = data["agents"]
            agents = {}
            for name, config_data in agents_data.items():
                agents[name] = AgentTypeConfig(**config_data)
            data["agents"] = agents
            
        if "actions" in data:
            actions_data = data["actions"] 
            actions = {}
            for name, config_data in actions_data.items():
                actions[name] = ActionConfig(**config_data)
            data["actions"] = actions
            
        if "visualization" in data:
            data["visualization"] = VisualizationConfig(**data["visualization"])
            
        return cls(**data)
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        data = self.to_dict()
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
                
            if key == "agents":
                result[key] = {name: config.__dict__ for name, config in value.items()}
            elif key == "actions":
                result[key] = {name: config.__dict__ for name, config in value.items()}
            elif key == "visualization":
                result[key] = value.to_dict()
            else:
                result[key] = value
                
        return result
    
    def copy(self) -> "SimulationConfig":
        """Create a deep copy of the configuration."""
        return copy.deepcopy(self)


# ===== PRESET CONFIGURATIONS =====

def create_cooperative_simulation() -> SimulationConfig:
    """Create a simulation focused on cooperative behavior."""
    return SimulationConfig(
        environment_profile="resource_rich",
        agents={
            "cooperative": AgentTypeConfig(
                behavior_profile="cooperative", 
                count=15
            ),
            "balanced": AgentTypeConfig(
                behavior_profile="balanced",
                count=5
            )
        }
    )

def create_competitive_simulation() -> SimulationConfig:
    """Create a simulation focused on competitive behavior."""
    return SimulationConfig(
        environment_profile="resource_scarce", 
        agents={
            "aggressive": AgentTypeConfig(
                behavior_profile="aggressive",
                count=10
            ),
            "survivor": AgentTypeConfig(
                behavior_profile="survivor", 
                count=10
            )
        }
    )

def create_large_scale_simulation() -> SimulationConfig:
    """Create a large-scale simulation with many agents.""" 
    return (SimulationConfig()
        .with_environment("large_world")
        .with_agents(
            system=AgentTypeConfig(behavior_profile="cooperative", count=50),
            independent=AgentTypeConfig(behavior_profile="aggressive", count=50),
            control=AgentTypeConfig(behavior_profile="gatherer", count=25)
        )
        .with_dqn_profile("fast_learning")
    )

def create_memory_efficient_simulation() -> SimulationConfig:
    """Create a simulation optimized for low memory usage."""
    return (SimulationConfig()
        .with_environment("small_world") 
        .with_dqn_profile("memory_efficient")
    )


# Default configurations for easy access
DEFAULT_CONFIG = SimulationConfig()
COOPERATIVE_CONFIG = create_cooperative_simulation()
COMPETITIVE_CONFIG = create_competitive_simulation()
LARGE_SCALE_CONFIG = create_large_scale_simulation()
MEMORY_EFFICIENT_CONFIG = create_memory_efficient_simulation()