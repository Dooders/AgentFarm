# Thermodynamic Flocking - Modular Implementation

This document shows how to implement the thermodynamic flocking simulation using the new modular scenario architecture.

---

## Complete Modular Implementation

### 1. Flocking Scenario

**File**: `farm/scenarios/flocking_scenario.py`

```python
"""Thermodynamic flocking scenario implementation."""

from typing import List, Any, Type
import numpy as np

from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.core.scenarios.base import BaseScenario
from farm.core.scenarios.registry import register_scenario
from farm.core.scenarios.protocol import ScenarioMetrics, ScenarioVisualizer


@register_scenario("thermodynamic_flocking")
class FlockingScenario(BaseScenario):
    """Flocking with thermodynamic constraints.
    
    Implements Reynolds' flocking rules (alignment, cohesion, separation)
    with energy-based movement costs, demonstrating ESDP Principle 2
    (Thermodynamic Realism).
    
    Features:
    - Local flocking rules
    - Energy consumption (E ∝ v²)
    - Optional sparse resource foraging
    - Multiple modes (classic, adaptive, evolutionary)
    """
    
    name = "thermodynamic_flocking"
    description = "Flocking with energy costs (ESDP Principle 2)"
    version = "1.0.0"
    
    def __init__(self):
        """Initialize flocking scenario."""
        super().__init__()
        self._metrics = None
        self._visualizer = None
    
    def create_agents(
        self,
        environment: Environment,
        config: Any
    ) -> List[BaseAgent]:
        """Create flocking agents based on configuration.
        
        Args:
            environment: Simulation environment
            config: Scenario configuration
            
        Returns:
            List of FlockingAgent instances
        """
        # Import here to avoid circular dependencies
        from farm.core.flocking_agent import (
            FlockingAgent,
            AwareFlockingAgent,
            EvoFlockingAgent
        )
        
        # Get scenario config
        flocking_config = config.scenario.flocking
        n_agents = flocking_config.n_agents
        
        # Select agent class based on mode
        mode = getattr(flocking_config, 'mode', 'classic')
        if mode == 'evolutionary':
            agent_class = EvoFlockingAgent
        elif mode == 'adaptive':
            agent_class = AwareFlockingAgent
        else:
            agent_class = FlockingAgent
        
        agents = []
        
        for i in range(n_agents):
            # Random position
            position = (
                np.random.uniform(0, environment.width),
                np.random.uniform(0, environment.height)
            )
            
            # Varied initial energy (thermodynamic diversity)
            if getattr(flocking_config, 'classic_mode', False):
                initial_energy = 100.0
            else:
                initial_energy = np.random.uniform(
                    flocking_config.initial_energy_min,
                    flocking_config.initial_energy_max
                )
            
            # Create agent
            agent = agent_class(
                agent_id=environment.get_next_agent_id(),
                position=position,
                resource_level=initial_energy,
                spatial_service=environment.spatial_service,
                environment=environment,
                agent_type=f"{agent_class.__name__}",
                config=config,
                generation=0
            )
            
            agents.append(agent)
        
        return agents
    
    def step_hook(
        self, 
        environment: Environment, 
        step: int
    ) -> None:
        """Called after each simulation step.
        
        Could add dynamic events like:
        - Phase transitions
        - Environmental changes
        - Predator spawning
        - Energy source migration
        
        Args:
            environment: Current environment
            step: Current step number
        """
        # Example: Spawn new resource every 100 steps
        flocking_config = self._config.scenario.flocking
        
        if (step % 100 == 0 and 
            getattr(flocking_config, 'dynamic_resources', False)):
            
            # Spawn new energy source
            from farm.core.resources import Resource
            
            position = (
                np.random.uniform(0, environment.width),
                np.random.uniform(0, environment.height)
            )
            
            resource = Resource(
                resource_id=environment.identity.resource_id(),
                position=position,
                amount=50.0
            )
            
            environment.add_resource(resource)
    
    def get_metrics(self) -> ScenarioMetrics:
        """Get flocking metrics tracker.
        
        Returns:
            FlockingMetrics instance
        """
        if self._metrics is None:
            from farm.analysis.flocking_metrics import FlockingMetrics
            self._metrics = FlockingMetrics()
        
        return self._metrics
    
    def get_visualizer(self) -> ScenarioVisualizer:
        """Get flocking visualizer.
        
        Returns:
            FlockingVisualizer instance
        """
        if self._visualizer is None:
            from farm.visualization.flocking_viz import FlockingVisualizer
            self._visualizer = FlockingVisualizer()
        
        return self._visualizer
    
    def validate_config(self, config: Any) -> bool:
        """Validate flocking configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration invalid
        """
        super().validate_config(config)
        
        # Check for required flocking config
        if not hasattr(config.scenario, 'flocking'):
            raise ValueError(
                "Config must have 'scenario.flocking' section"
            )
        
        flocking = config.scenario.flocking
        
        # Validate parameters
        required_params = [
            'n_agents', 'max_speed', 'max_force',
            'perception_radius', 'separation_radius'
        ]
        
        for param in required_params:
            if not hasattr(flocking, param):
                raise ValueError(
                    f"Flocking config missing required parameter: {param}"
                )
        
        # Validate ranges
        if flocking.n_agents <= 0:
            raise ValueError("n_agents must be positive")
        
        if flocking.max_speed <= 0:
            raise ValueError("max_speed must be positive")
        
        if flocking.perception_radius <= 0:
            raise ValueError("perception_radius must be positive")
        
        return True
    
    def get_agent_types(self) -> List[Type[BaseAgent]]:
        """Get agent types used in this scenario.
        
        Returns:
            List of possible agent types
        """
        from farm.core.flocking_agent import (
            FlockingAgent,
            AwareFlockingAgent,
            EvoFlockingAgent
        )
        
        return [FlockingAgent, AwareFlockingAgent, EvoFlockingAgent]
```

---

### 2. Flocking Metrics (Scenario-Specific)

**File**: `farm/analysis/flocking_metrics.py`

```python
"""Metrics for thermodynamic flocking scenario."""

from typing import List, Dict, Any
import numpy as np

from farm.core.environment import Environment
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class FlockingMetrics:
    """Track emergence metrics for flocking behavior.
    
    Implements ScenarioMetrics protocol for flocking scenario.
    """
    
    def __init__(self):
        """Initialize metric storage."""
        self.time = []
        self.alive_count = []
        self.avg_energy = []
        self.avg_speed = []
        self.alignment = []
        self.cohesion = []
        self.entropy_production = []
        self.phase_synchrony = []
    
    def update(self, environment: Environment, step: int) -> None:
        """Update metrics for current step.
        
        Args:
            environment: Current simulation environment
            step: Current simulation step number
        """
        # Get alive agents
        all_agents = list(environment._agent_objects.values())
        alive_agents = [a for a in all_agents if a.alive]
        
        # Record time
        self.time.append(step)
        self.alive_count.append(len(alive_agents))
        
        if alive_agents:
            # Basic metrics
            self.avg_energy.append(
                np.mean([a.resource_level for a in alive_agents])
            )
            self.avg_speed.append(
                np.mean([
                    np.linalg.norm(a.velocity) 
                    for a in alive_agents
                ])
            )
            
            # Flocking metrics
            self.alignment.append(self.compute_alignment(alive_agents))
            self.cohesion.append(self.compute_cohesion(alive_agents))
            
            # Thermodynamic metrics
            self.entropy_production.append(
                self.compute_entropy_production(alive_agents)
            )
            self.phase_synchrony.append(
                self.compute_phase_synchrony(alive_agents)
            )
        else:
            # No agents alive
            self.avg_energy.append(0.0)
            self.avg_speed.append(0.0)
            self.alignment.append(0.0)
            self.cohesion.append(0.0)
            self.entropy_production.append(0.0)
            self.phase_synchrony.append(0.0)
    
    def compute_alignment(self, agents) -> float:
        """Compute velocity coherence (0-1)."""
        if not agents:
            return 0.0
        
        velocities = np.array([a.velocity for a in agents])
        avg_velocity = np.mean(velocities, axis=0)
        avg_speed = np.mean([np.linalg.norm(v) for v in velocities])
        
        if avg_speed < 1e-6:
            return 0.0
        
        alignment = np.linalg.norm(avg_velocity) / avg_speed
        return float(alignment)
    
    def compute_cohesion(self, agents) -> float:
        """Compute spatial clustering (0-1)."""
        if not agents:
            return 0.0
        
        positions = np.array([a.position for a in agents])
        center = np.mean(positions, axis=0)
        avg_distance = np.mean([
            np.linalg.norm(p - center) for p in positions
        ])
        
        cohesion = 1.0 / (1.0 + avg_distance / 10.0)
        return float(cohesion)
    
    def compute_entropy_production(self, agents) -> float:
        """Compute entropy production rate: σ = Σv²/N."""
        if not agents:
            return 0.0
        
        total_dissipation = sum(
            np.linalg.norm(a.velocity) ** 2 
            for a in agents
        )
        
        entropy_rate = total_dissipation / len(agents)
        return float(entropy_rate)
    
    def compute_phase_synchrony(self, agents) -> float:
        """Compute Kuramoto order parameter (0-1)."""
        if len(agents) < 2:
            return 0.0
        
        # Calculate velocity angles
        angles = [
            np.arctan2(a.velocity[1], a.velocity[0]) 
            for a in agents
        ]
        
        # Kuramoto order parameter
        complex_phases = np.exp(1j * np.array(angles))
        order_param = np.abs(np.mean(complex_phases))
        
        return float(order_param)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values.
        
        Returns:
            Dictionary of current metrics
        """
        if not self.time:
            return {}
        
        return {
            'alive_count': self.alive_count[-1],
            'avg_energy': self.avg_energy[-1],
            'avg_speed': self.avg_speed[-1],
            'alignment': self.alignment[-1],
            'cohesion': self.cohesion[-1],
            'entropy_production': self.entropy_production[-1],
            'phase_synchrony': self.phase_synchrony[-1],
        }
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get complete metric history.
        
        Returns:
            Dictionary of metric histories
        """
        return {
            'time': self.time,
            'alive_count': self.alive_count,
            'avg_energy': self.avg_energy,
            'avg_speed': self.avg_speed,
            'alignment': self.alignment,
            'cohesion': self.cohesion,
            'entropy_production': self.entropy_production,
            'phase_synchrony': self.phase_synchrony,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary.
        
        Returns:
            Complete metrics data
        """
        return self.get_history()
```

---

### 3. Flocking Visualizer

**File**: `farm/visualization/flocking_viz.py`

```python
"""Visualization for thermodynamic flocking scenario."""

from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from farm.core.environment import Environment
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class FlockingVisualizer:
    """Visualizer for flocking scenario.
    
    Implements ScenarioVisualizer protocol.
    """
    
    def plot_metrics(
        self, 
        metrics, 
        save_path: Optional[str] = None
    ) -> Any:
        """Create 3x2 grid of flocking metrics.
        
        Args:
            metrics: FlockingMetrics instance
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        
        history = metrics.get_history()
        times = history['time']
        
        # Population
        axes[0, 0].plot(times, history['alive_count'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Alive Agents')
        axes[0, 0].set_title('Population Dynamics')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy
        axes[0, 1].plot(times, history['avg_energy'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Average Energy')
        axes[0, 1].set_title('Energy Dynamics')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Alignment
        axes[1, 0].plot(times, history['alignment'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Alignment (0-1)')
        axes[1, 0].set_title('Velocity Coherence')
        axes[1, 0].set_ylim([0, 1.1])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cohesion
        axes[1, 1].plot(times, history['cohesion'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Cohesion (0-1)')
        axes[1, 1].set_title('Spatial Clustering')
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].grid(True, alpha=0.3)
        
        # Entropy Production
        axes[2, 0].plot(
            times, history['entropy_production'], 
            'orange', linewidth=2
        )
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].set_ylabel('Entropy Production (σ)')
        axes[2, 0].set_title('Dissipative Structure Formation')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Peak annotation
        if history['entropy_production']:
            max_entropy = max(history['entropy_production'])
            max_idx = history['entropy_production'].index(max_entropy)
            max_time = times[max_idx]
            axes[2, 0].axvline(
                max_time, color='red', linestyle='--', 
                alpha=0.5, label=f'Peak: {max_entropy:.2f}'
            )
            axes[2, 0].legend()
        
        # Phase Synchrony
        axes[2, 1].plot(
            times, history['phase_synchrony'], 
            'purple', linewidth=2
        )
        axes[2, 1].set_xlabel('Time')
        axes[2, 1].set_ylabel('Phase Synchrony (0-1)')
        axes[2, 1].set_title('Velocity Coordination (Kuramoto)')
        axes[2, 1].set_ylim([0, 1.1])
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved metrics plot to {save_path}")
        
        return fig
    
    def create_animation(
        self,
        environment: Environment,
        save_path: Optional[str] = None,
        frames: int = 500
    ) -> Any:
        """Create animation of flocking simulation.
        
        Args:
            environment: Environment to animate
            save_path: Optional path to save animation
            frames: Number of frames to render
            
        Returns:
            Animation object
        """
        # This would create an animation from database logs
        # For now, placeholder
        logger.warning("Animation creation not yet implemented")
        return None
```

---

### 4. Updated Configuration

**File**: `farm/config/scenarios/flocking.yaml`

```yaml
# Scenario selection and configuration
scenario:
  type: "thermodynamic_flocking"  # Must match registered name
  
  # Flocking-specific settings
  flocking:
    # Population
    n_agents: 50
    mode: "classic"  # Options: classic, adaptive, evolutionary
    
    # Initial conditions
    initial_energy_min: 30.0
    initial_energy_max: 100.0
    
    # Movement parameters
    max_speed: 2.0
    max_force: 0.5
    
    # Perception radii
    perception_radius: 10.0
    separation_radius: 5.0
    
    # Flocking weights
    alignment_weight: 1.0
    cohesion_weight: 1.0
    separation_weight: 1.5
    
    # Energy system
    classic_mode: false
    velocity_cost: 0.25
    base_cost: 0.03
    ambient_replenishment: 0.85
    use_sparse_resources: false
    
    # Dynamic features
    dynamic_resources: false

# Standard environment configuration
environment:
  width: 100
  height: 100
  
  spatial_index:
    enable_spatial_hash_indices: true
    spatial_hash_cell_size: 15.0

# Resources
resources:
  initial_resources: 8
  resource_regen_rate: 0.02
  resource_regen_amount: 50
  max_resource_amount: 50

# Simulation settings
max_steps: 1000
seed: 42

# Database
database:
  use_in_memory_db: false
  db_pragma_profile: "balanced"
```

---

### 5. Simple Runner Script

**File**: `scripts/run_scenario.py`

```python
"""Universal scenario runner script."""

import argparse
from pathlib import Path

from farm.config import load_config
from farm.core.scenarios.runner import ScenarioRunner
from farm.core.scenarios.factory import ScenarioFactory
from farm.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main():
    """Run a simulation scenario."""
    parser = argparse.ArgumentParser(
        description="Run AgentFarm simulation scenario"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to scenario configuration file'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of steps (overrides config)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualizations after simulation'
    )
    parser.add_argument(
        '--save-plots',
        type=str,
        default=None,
        help='Path to save plots'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(environment="development", log_level="INFO")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Discover scenarios
    ScenarioFactory.discover_scenarios()
    
    # Run scenario
    logger.info(f"Running scenario: {config.scenario.type}")
    results = ScenarioRunner.run_from_config(
        config,
        steps=args.steps,
        progress_bar=True,
        log_interval=100
    )
    
    # Visualize if requested
    if args.visualize:
        logger.info("Creating visualizations...")
        
        # Get scenario and visualizer
        scenario, _ = ScenarioFactory.create_from_config(config)
        visualizer = scenario.get_visualizer()
        
        # Get metrics from results
        from farm.analysis.flocking_metrics import FlockingMetrics
        metrics = FlockingMetrics()
        # Reconstruct metrics from results
        for key, values in results['metrics'].items():
            setattr(metrics, key, values)
        
        # Create plots
        save_path = args.save_plots or "scenario_results.png"
        visualizer.plot_metrics(metrics, save_path=save_path)
        
        logger.info(f"Saved plots to {save_path}")
    
    logger.info("Scenario complete!")


if __name__ == "__main__":
    main()
```

---

## Usage Examples

### Run Flocking Scenario

```bash
# Run with default settings
python scripts/run_scenario.py \
    --config farm/config/scenarios/flocking.yaml \
    --visualize

# Run with custom steps
python scripts/run_scenario.py \
    --config farm/config/scenarios/flocking.yaml \
    --steps 2000 \
    --visualize \
    --save-plots flocking_results.png
```

### Swap to Different Scenario

Just change config file:

```bash
# Run predator-prey instead
python scripts/run_scenario.py \
    --config farm/config/scenarios/predator_prey.yaml \
    --visualize
```

### Programmatic Usage

```python
from farm.config import load_config
from farm.core.scenarios.runner import ScenarioRunner

# Load any scenario config
config = load_config("farm/config/scenarios/flocking.yaml")

# Run (same code for all scenarios!)
results = ScenarioRunner.run_from_config(
    config,
    steps=1000,
    progress_bar=True
)

# Get metrics and visualize
scenario = results['scenario']
metrics = results['metrics']

visualizer = scenario.get_visualizer()
visualizer.plot_metrics(metrics, save_path="results.png")
```

---

## Benefits of Modular Approach

1. **Easy Swapping**: Change one line in config to switch scenarios
2. **Consistent Interface**: All scenarios work identically
3. **Reusable Code**: Share components across scenarios
4. **Clean Separation**: Scenario logic is isolated
5. **Testable**: Each scenario can be tested independently
6. **Discoverable**: Registry automatically finds scenarios
7. **Extensible**: Add new scenarios without modifying existing code

---

## Comparison: Before vs After

### Before (Monolithic)

```python
# Separate script for each simulation type
python run_flocking_simulation.py
python run_predator_prey_simulation.py
python run_resource_competition.py

# Each with different interfaces, configs, and metrics
```

### After (Modular)

```python
# One universal runner
python scripts/run_scenario.py --config flocking.yaml
python scripts/run_scenario.py --config predator_prey.yaml
python scripts/run_scenario.py --config resource_competition.yaml

# Same interface, metrics, and visualization for all
```

---

## Next Scenario Example

Creating a new scenario is simple:

```python
@register_scenario("my_new_scenario")
class MyScenario(BaseScenario):
    name = "my_new_scenario"
    description = "Description here"
    version = "1.0.0"
    
    def create_agents(self, environment, config):
        # Create your agents
        return agents
    
    def get_metrics(self):
        return MyMetrics()
    
    def get_visualizer(self):
        return MyVisualizer()
```

That's it! The scenario automatically works with the entire system.
