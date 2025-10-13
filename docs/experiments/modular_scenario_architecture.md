# Modular Scenario Architecture for AgentFarm

## Overview

This document describes a modular scenario system that allows you to easily create, register, and swap between different simulation scenarios (flocking, predator-prey, resource competition, etc.) using standard interfaces.

---

## Design Principles

1. **Standard Interfaces**: All scenarios implement the same protocol
2. **Pluggable Components**: Agents, metrics, and visualizations are swappable
3. **Configuration-Driven**: Select scenarios via YAML config
4. **Registry Pattern**: Automatic scenario discovery and registration
5. **Minimal Boilerplate**: Creating new scenarios is straightforward
6. **AgentFarm Integration**: Leverages existing services and patterns

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Scenario Registry                         │
│  (Discovers and manages all available scenarios)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Scenario Protocol                          │
│  (Standard interface all scenarios must implement)           │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
        ┌───────────┐  ┌───────────┐  ┌───────────┐
        │ Flocking  │  │ Predator  │  │ Resource  │
        │ Scenario  │  │   Prey    │  │Competition│
        └───────────┘  └───────────┘  └───────────┘
                │             │             │
        ┌───────┴─────────────┴─────────────┴───────┐
        ▼                                             ▼
┌─────────────┐                               ┌─────────────┐
│   Agents    │                               │   Metrics   │
│  Component  │                               │  Component  │
└─────────────┘                               └─────────────┘
```

---

## Core Components

### 1. Scenario Protocol

**File**: `farm/core/scenarios/protocol.py`

```python
"""Protocol for modular simulation scenarios."""

from typing import Protocol, Dict, Any, List, Type, Optional
from abc import abstractmethod
import numpy as np

from farm.core.agent import BaseAgent
from farm.core.environment import Environment


class ScenarioMetrics(Protocol):
    """Protocol for scenario-specific metrics."""
    
    def update(self, environment: Environment, step: int) -> None:
        """Update metrics for current step.
        
        Args:
            environment: Current simulation environment
            step: Current simulation step number
        """
        ...
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values.
        
        Returns:
            Dictionary mapping metric names to current values
        """
        ...
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get complete metric history.
        
        Returns:
            Dictionary mapping metric names to value histories
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary.
        
        Returns:
            Complete metrics data as dictionary
        """
        ...


class ScenarioVisualizer(Protocol):
    """Protocol for scenario visualization."""
    
    def plot_metrics(
        self, 
        metrics: ScenarioMetrics, 
        save_path: Optional[str] = None
    ) -> Any:
        """Create visualization of metrics.
        
        Args:
            metrics: Metrics to visualize
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure or similar visualization object
        """
        ...
    
    def create_animation(
        self,
        environment: Environment,
        save_path: Optional[str] = None,
        frames: int = 500
    ) -> Any:
        """Create animation of simulation.
        
        Args:
            environment: Environment to animate
            save_path: Optional path to save animation
            frames: Number of frames to render
            
        Returns:
            Animation object
        """
        ...


class Scenario(Protocol):
    """Protocol that all scenarios must implement.
    
    This defines the standard interface for modular simulation scenarios.
    Each scenario is responsible for:
    - Creating appropriate agents
    - Defining scenario-specific logic
    - Providing metrics tracking
    - Offering visualization tools
    """
    
    # Scenario metadata
    name: str
    description: str
    version: str
    
    @abstractmethod
    def setup(
        self, 
        environment: Environment, 
        config: Any
    ) -> List[BaseAgent]:
        """Initialize scenario and create agents.
        
        Args:
            environment: The simulation environment
            config: Scenario-specific configuration
            
        Returns:
            List of created agents
        """
        ...
    
    @abstractmethod
    def step_hook(
        self, 
        environment: Environment, 
        step: int
    ) -> None:
        """Called after each simulation step.
        
        Use this for scenario-specific logic that runs each step
        (e.g., resource spawning, events, phase transitions).
        
        Args:
            environment: Current environment
            step: Current step number
        """
        ...
    
    @abstractmethod
    def get_metrics(self) -> ScenarioMetrics:
        """Get scenario-specific metrics tracker.
        
        Returns:
            Metrics object for this scenario
        """
        ...
    
    @abstractmethod
    def get_visualizer(self) -> ScenarioVisualizer:
        """Get scenario-specific visualizer.
        
        Returns:
            Visualizer object for this scenario
        """
        ...
    
    @abstractmethod
    def validate_config(self, config: Any) -> bool:
        """Validate scenario configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid with explanation
        """
        ...
    
    def get_agent_types(self) -> List[Type[BaseAgent]]:
        """Get list of agent types used in this scenario.
        
        Returns:
            List of agent class types
        """
        ...
    
    def cleanup(self, environment: Environment) -> None:
        """Cleanup scenario resources.
        
        Called when simulation ends. Optional to implement.
        
        Args:
            environment: Environment to cleanup
        """
        pass
```

---

### 2. Scenario Registry

**File**: `farm/core/scenarios/registry.py`

```python
"""Registry for discovering and managing simulation scenarios."""

from typing import Dict, Type, Optional, List
import inspect
from pathlib import Path

from farm.core.scenarios.protocol import Scenario
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class ScenarioRegistry:
    """Central registry for simulation scenarios.
    
    Provides automatic discovery and registration of scenario implementations.
    Scenarios can be registered manually or auto-discovered from a directory.
    """
    
    _instance: Optional['ScenarioRegistry'] = None
    _scenarios: Dict[str, Type[Scenario]] = {}
    
    def __new__(cls):
        """Singleton pattern - only one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(
        cls, 
        name: str, 
        scenario_class: Type[Scenario]
    ) -> None:
        """Register a scenario class.
        
        Args:
            name: Unique identifier for the scenario
            scenario_class: Scenario class to register
            
        Raises:
            ValueError: If name already registered or class invalid
        """
        if name in cls._scenarios:
            logger.warning(
                f"Scenario '{name}' already registered, overwriting"
            )
        
        # Validate that class implements Scenario protocol
        if not cls._validates_protocol(scenario_class):
            raise ValueError(
                f"Class {scenario_class.__name__} does not implement "
                "Scenario protocol"
            )
        
        cls._scenarios[name] = scenario_class
        logger.info(f"Registered scenario: {name}")
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a scenario.
        
        Args:
            name: Scenario name to unregister
        """
        if name in cls._scenarios:
            del cls._scenarios[name]
            logger.info(f"Unregistered scenario: {name}")
    
    @classmethod
    def get(cls, name: str) -> Type[Scenario]:
        """Get a registered scenario class.
        
        Args:
            name: Scenario name
            
        Returns:
            Scenario class
            
        Raises:
            KeyError: If scenario not found
        """
        if name not in cls._scenarios:
            available = ", ".join(cls._scenarios.keys())
            raise KeyError(
                f"Scenario '{name}' not found. "
                f"Available scenarios: {available}"
            )
        
        return cls._scenarios[name]
    
    @classmethod
    def list_scenarios(cls) -> List[str]:
        """List all registered scenario names.
        
        Returns:
            List of scenario names
        """
        return list(cls._scenarios.keys())
    
    @classmethod
    def get_scenario_info(cls, name: str) -> Dict[str, str]:
        """Get information about a scenario.
        
        Args:
            name: Scenario name
            
        Returns:
            Dictionary with scenario metadata
        """
        scenario_class = cls.get(name)
        instance = scenario_class()
        
        return {
            'name': instance.name,
            'description': instance.description,
            'version': instance.version,
            'class': scenario_class.__name__,
            'module': scenario_class.__module__,
        }
    
    @classmethod
    def discover_scenarios(cls, directory: Path) -> int:
        """Auto-discover scenarios from a directory.
        
        Searches for Python files containing Scenario implementations
        and automatically registers them.
        
        Args:
            directory: Directory to search for scenarios
            
        Returns:
            Number of scenarios discovered and registered
        """
        discovered = 0
        
        for py_file in directory.glob("**/*_scenario.py"):
            try:
                # Import module
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(
                    module_name, py_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find Scenario implementations
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (cls._validates_protocol(obj) and 
                        obj.__module__ == module.__name__):
                        
                        scenario_instance = obj()
                        cls.register(scenario_instance.name, obj)
                        discovered += 1
                        
            except Exception as e:
                logger.error(
                    f"Failed to load scenario from {py_file}: {e}"
                )
        
        logger.info(
            f"Discovered {discovered} scenarios from {directory}"
        )
        return discovered
    
    @classmethod
    def _validates_protocol(cls, scenario_class: Type) -> bool:
        """Check if class implements Scenario protocol.
        
        Args:
            scenario_class: Class to validate
            
        Returns:
            True if class implements protocol
        """
        required_methods = [
            'setup', 'step_hook', 'get_metrics', 
            'get_visualizer', 'validate_config'
        ]
        
        return all(
            hasattr(scenario_class, method) 
            for method in required_methods
        )


# Decorator for easy registration
def register_scenario(name: str):
    """Decorator to register a scenario class.
    
    Usage:
        @register_scenario("my_scenario")
        class MyScenario:
            ...
    """
    def decorator(cls):
        ScenarioRegistry.register(name, cls)
        return cls
    return decorator
```

---

### 3. Base Scenario Implementation

**File**: `farm/core/scenarios/base.py`

```python
"""Base implementation of Scenario protocol with common functionality."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Optional
import numpy as np

from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.core.scenarios.protocol import (
    Scenario, ScenarioMetrics, ScenarioVisualizer
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class BaseScenario(ABC):
    """Base class providing common scenario functionality.
    
    Inherit from this class to create new scenarios. You must implement:
    - setup()
    - create_agents()
    - get_metrics()
    - get_visualizer()
    
    Optional to override:
    - step_hook()
    - validate_config()
    - get_agent_types()
    - cleanup()
    """
    
    # Subclasses must set these
    name: str = "base_scenario"
    description: str = "Base scenario implementation"
    version: str = "1.0.0"
    
    def __init__(self):
        """Initialize base scenario."""
        self._environment: Optional[Environment] = None
        self._config: Optional[Any] = None
        self._agents: List[BaseAgent] = []
    
    def setup(
        self, 
        environment: Environment, 
        config: Any
    ) -> List[BaseAgent]:
        """Setup scenario in environment.
        
        Args:
            environment: Simulation environment
            config: Scenario configuration
            
        Returns:
            List of created agents
        """
        # Validate configuration
        self.validate_config(config)
        
        # Store references
        self._environment = environment
        self._config = config
        
        # Create agents (delegated to subclass)
        self._agents = self.create_agents(environment, config)
        
        # Add agents to environment
        for agent in self._agents:
            environment.add_agent(agent)
        
        logger.info(
            f"Scenario '{self.name}' setup complete: "
            f"{len(self._agents)} agents created"
        )
        
        return self._agents
    
    @abstractmethod
    def create_agents(
        self,
        environment: Environment,
        config: Any
    ) -> List[BaseAgent]:
        """Create agents for this scenario.
        
        Subclasses must implement this to create scenario-specific agents.
        
        Args:
            environment: Simulation environment
            config: Scenario configuration
            
        Returns:
            List of created agents (not yet added to environment)
        """
        ...
    
    def step_hook(
        self, 
        environment: Environment, 
        step: int
    ) -> None:
        """Called after each simulation step.
        
        Default implementation does nothing. Override for scenario-specific
        per-step logic.
        
        Args:
            environment: Current environment
            step: Current step number
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> ScenarioMetrics:
        """Get scenario metrics tracker.
        
        Subclasses must implement to return scenario-specific metrics.
        
        Returns:
            Metrics tracker instance
        """
        ...
    
    @abstractmethod
    def get_visualizer(self) -> ScenarioVisualizer:
        """Get scenario visualizer.
        
        Subclasses must implement to return scenario-specific visualizer.
        
        Returns:
            Visualizer instance
        """
        ...
    
    def validate_config(self, config: Any) -> bool:
        """Validate scenario configuration.
        
        Default implementation checks for required scenario config section.
        Override to add scenario-specific validation.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration invalid
        """
        if not hasattr(config, 'scenario'):
            raise ValueError("Config must have 'scenario' section")
        
        return True
    
    def get_agent_types(self) -> List[Type[BaseAgent]]:
        """Get agent types used in scenario.
        
        Default implementation extracts types from created agents.
        Override if you need different behavior.
        
        Returns:
            List of agent types
        """
        if not self._agents:
            return []
        
        return list(set(type(agent) for agent in self._agents))
    
    def cleanup(self, environment: Environment) -> None:
        """Cleanup scenario resources.
        
        Default implementation does nothing. Override if needed.
        
        Args:
            environment: Environment to cleanup
        """
        pass
```

---

### 4. Scenario Factory

**File**: `farm/core/scenarios/factory.py`

```python
"""Factory for creating and running scenarios."""

from typing import Optional, Any
from pathlib import Path

from farm.core.environment import Environment
from farm.core.scenarios.registry import ScenarioRegistry
from farm.core.scenarios.protocol import Scenario
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class ScenarioFactory:
    """Factory for creating scenario instances from configuration."""
    
    @staticmethod
    def create(
        scenario_name: str,
        environment: Environment,
        config: Any
    ) -> Scenario:
        """Create and setup a scenario.
        
        Args:
            scenario_name: Name of registered scenario
            environment: Environment to setup scenario in
            config: Scenario configuration
            
        Returns:
            Initialized scenario instance
            
        Raises:
            KeyError: If scenario not found
            ValueError: If configuration invalid
        """
        # Get scenario class from registry
        scenario_class = ScenarioRegistry.get(scenario_name)
        
        # Create instance
        scenario = scenario_class()
        
        # Setup scenario
        scenario.setup(environment, config)
        
        logger.info(f"Created scenario: {scenario_name}")
        
        return scenario
    
    @staticmethod
    def create_from_config(
        config: Any,
        environment: Optional[Environment] = None
    ) -> tuple[Scenario, Environment]:
        """Create scenario and environment from configuration.
        
        Args:
            config: Configuration with scenario and environment settings
            environment: Optional pre-created environment
            
        Returns:
            Tuple of (scenario, environment)
        """
        # Create environment if not provided
        if environment is None:
            environment = Environment(
                width=config.environment.width,
                height=config.environment.height,
                resource_distribution={
                    "type": "random",
                    "amount": config.resources.initial_resources,
                },
                config=config,
                seed=getattr(config, 'seed', None)
            )
        
        # Get scenario name from config
        scenario_name = config.scenario.type
        
        # Create scenario
        scenario = ScenarioFactory.create(
            scenario_name, 
            environment, 
            config
        )
        
        return scenario, environment
    
    @staticmethod
    def discover_scenarios(directory: Optional[Path] = None) -> int:
        """Discover scenarios from directory.
        
        Args:
            directory: Directory to search (default: farm/scenarios/)
            
        Returns:
            Number of scenarios discovered
        """
        if directory is None:
            # Default to scenarios directory
            directory = Path(__file__).parent.parent / "scenarios"
        
        return ScenarioRegistry.discover_scenarios(directory)
```

---

### 5. Scenario Runner

**File**: `farm/core/scenarios/runner.py`

```python
"""High-level runner for executing scenarios."""

from typing import Optional, Any
from tqdm import tqdm

from farm.core.environment import Environment
from farm.core.scenarios.protocol import Scenario
from farm.core.scenarios.factory import ScenarioFactory
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class ScenarioRunner:
    """High-level runner for scenario simulations."""
    
    def __init__(
        self,
        scenario: Scenario,
        environment: Environment,
        config: Any
    ):
        """Initialize runner.
        
        Args:
            scenario: Scenario to run
            environment: Environment to run in
            config: Configuration
        """
        self.scenario = scenario
        self.environment = environment
        self.config = config
        self.metrics = scenario.get_metrics()
    
    def run(
        self, 
        steps: int, 
        progress_bar: bool = True,
        log_interval: int = 100
    ) -> dict:
        """Run scenario simulation.
        
        Args:
            steps: Number of steps to run
            progress_bar: Show progress bar
            log_interval: Steps between status logs
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(
            f"Starting scenario '{self.scenario.name}' "
            f"for {steps} steps"
        )
        
        # Create progress bar if requested
        iterator = range(steps)
        if progress_bar:
            iterator = tqdm(iterator, desc=self.scenario.name)
        
        # Run simulation
        for step in iterator:
            # Step environment
            self.environment.step()
            
            # Scenario-specific step logic
            self.scenario.step_hook(self.environment, step)
            
            # Update metrics
            self.metrics.update(self.environment, step)
            
            # Periodic logging
            if step % log_interval == 0:
                current_metrics = self.metrics.get_current_metrics()
                logger.info(
                    f"Step {step}/{steps}: {current_metrics}"
                )
        
        # Finalize
        self.environment.finalize()
        self.scenario.cleanup(self.environment)
        
        logger.info(f"Scenario '{self.scenario.name}' complete")
        
        # Return results
        return {
            'scenario': self.scenario.name,
            'steps': steps,
            'metrics': self.metrics.to_dict(),
            'environment': self.environment,
        }
    
    @classmethod
    def run_from_config(
        cls,
        config: Any,
        steps: Optional[int] = None,
        **kwargs
    ) -> dict:
        """Create and run scenario from configuration.
        
        Args:
            config: Scenario configuration
            steps: Number of steps (uses config.max_steps if None)
            **kwargs: Additional arguments for run()
            
        Returns:
            Simulation results
        """
        # Create scenario and environment
        scenario, environment = ScenarioFactory.create_from_config(config)
        
        # Create runner
        runner = cls(scenario, environment, config)
        
        # Determine steps
        if steps is None:
            steps = getattr(config, 'max_steps', 1000)
        
        # Run
        return runner.run(steps, **kwargs)
```

---

## Usage Examples

### Defining a New Scenario

**File**: `farm/scenarios/flocking_scenario.py`

```python
"""Thermodynamic flocking scenario implementation."""

from typing import List, Any
import numpy as np

from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.core.scenarios.base import BaseScenario
from farm.core.scenarios.registry import register_scenario
from farm.core.flocking_agent import FlockingAgent
from farm.analysis.flocking_metrics import FlockingMetrics
from farm.visualization.flocking_viz import FlockingVisualizer


@register_scenario("thermodynamic_flocking")
class FlockingScenario(BaseScenario):
    """Thermodynamic flocking simulation scenario."""
    
    name = "thermodynamic_flocking"
    description = "Flocking with energy costs (ESDP Principle 2)"
    version = "1.0.0"
    
    def create_agents(
        self,
        environment: Environment,
        config: Any
    ) -> List[BaseAgent]:
        """Create flocking agents."""
        agents = []
        
        # Get scenario config
        flocking_config = config.scenario.flocking
        n_agents = flocking_config.n_agents
        
        for i in range(n_agents):
            # Random position
            position = (
                np.random.uniform(0, environment.width),
                np.random.uniform(0, environment.height)
            )
            
            # Varied initial energy
            initial_energy = np.random.uniform(
                flocking_config.initial_energy_min,
                flocking_config.initial_energy_max
            )
            
            # Create agent
            agent = FlockingAgent(
                agent_id=environment.get_next_agent_id(),
                position=position,
                resource_level=initial_energy,
                spatial_service=environment.spatial_service,
                environment=environment,
                config=config
            )
            
            agents.append(agent)
        
        return agents
    
    def step_hook(
        self, 
        environment: Environment, 
        step: int
    ) -> None:
        """Per-step scenario logic."""
        # Could add dynamic events, phase transitions, etc.
        pass
    
    def get_metrics(self):
        """Get flocking metrics."""
        return FlockingMetrics()
    
    def get_visualizer(self):
        """Get flocking visualizer."""
        return FlockingVisualizer()
    
    def validate_config(self, config: Any) -> bool:
        """Validate flocking configuration."""
        super().validate_config(config)
        
        # Check for required flocking config
        if not hasattr(config.scenario, 'flocking'):
            raise ValueError(
                "Config must have 'scenario.flocking' section"
            )
        
        flocking = config.scenario.flocking
        
        # Validate parameters
        if flocking.n_agents <= 0:
            raise ValueError("n_agents must be positive")
        
        if flocking.max_speed <= 0:
            raise ValueError("max_speed must be positive")
        
        return True
```

### Configuration

**File**: `farm/config/scenarios/flocking.yaml`

```yaml
# Scenario selection
scenario:
  type: "thermodynamic_flocking"  # Registry name
  
  # Scenario-specific config
  flocking:
    n_agents: 50
    initial_energy_min: 30.0
    initial_energy_max: 100.0
    max_speed: 2.0
    max_force: 0.5
    perception_radius: 10.0
    separation_radius: 5.0
    alignment_weight: 1.0
    cohesion_weight: 1.0
    separation_weight: 1.5

# Standard environment config
environment:
  width: 100
  height: 100
  
  spatial_index:
    enable_spatial_hash_indices: true

resources:
  initial_resources: 8
  resource_regen_rate: 0.02

# Simulation settings
max_steps: 1000
seed: 42
```

### Running a Scenario

**Option 1: Direct API**

```python
from farm.config import load_config
from farm.core.scenarios.runner import ScenarioRunner

# Load config
config = load_config("farm/config/scenarios/flocking.yaml")

# Run scenario
results = ScenarioRunner.run_from_config(
    config,
    steps=1000,
    progress_bar=True
)

# Visualize
visualizer = results['scenario'].get_visualizer()
visualizer.plot_metrics(results['metrics'])
```

**Option 2: CLI**

```bash
python -m farm.scenarios.cli run \
    --config farm/config/scenarios/flocking.yaml \
    --steps 1000 \
    --visualize
```

**Option 3: Programmatic**

```python
from farm.core.scenarios.factory import ScenarioFactory
from farm.core.scenarios.runner import ScenarioRunner
from farm.core.environment import Environment
from farm.config import load_config

# Load config
config = load_config("farm/config/scenarios/flocking.yaml")

# Create environment
env = Environment(
    width=100, height=100,
    resource_distribution={"type": "random", "amount": 8},
    config=config
)

# Create scenario
scenario, env = ScenarioFactory.create_from_config(config, env)

# Run
runner = ScenarioRunner(scenario, env, config)
results = runner.run(steps=1000)
```

### Swapping Scenarios

Just change the config:

```yaml
# flocking.yaml
scenario:
  type: "thermodynamic_flocking"
  # ...

# predator_prey.yaml  
scenario:
  type: "predator_prey"
  # ...

# resource_competition.yaml
scenario:
  type: "resource_competition"
  # ...
```

Same runner code works for all scenarios!

---

## Directory Structure

```
farm/
├── core/
│   └── scenarios/
│       ├── __init__.py
│       ├── protocol.py        # Scenario protocol definition
│       ├── registry.py        # Scenario registry
│       ├── base.py           # Base scenario class
│       ├── factory.py        # Scenario factory
│       └── runner.py         # Scenario runner
│
├── scenarios/               # Scenario implementations
│   ├── __init__.py
│   ├── flocking_scenario.py
│   ├── predator_prey_scenario.py
│   ├── resource_competition_scenario.py
│   └── ...
│
├── config/
│   └── scenarios/          # Scenario configs
│       ├── flocking.yaml
│       ├── predator_prey.yaml
│       └── ...
│
└── analysis/              # Scenario-specific metrics
    ├── flocking_metrics.py
    ├── predator_prey_metrics.py
    └── ...
```

---

## Benefits

1. **Easy Swapping**: Change one line in config to switch scenarios
2. **Consistent Interface**: All scenarios work the same way
3. **Auto-Discovery**: Registry automatically finds scenarios
4. **Reusable Components**: Metrics and visualizers are pluggable
5. **Type Safety**: Protocol ensures all scenarios implement required methods
6. **Minimal Boilerplate**: BaseScenario handles common logic
7. **Testing**: Easy to test scenarios in isolation

---

## Next Steps

See [Flocking Scenario Implementation](flocking_scenario_modular.md) for the refactored flocking implementation using this architecture.
