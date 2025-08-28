# Usage Examples and Tutorials

This document provides practical examples and tutorials for using AgentFarm effectively. Each example builds on the previous ones, starting with basic usage and progressing to advanced implementations.

## Tutorial 1: Basic Simulation Setup

### Creating Your First Simulation

```python
#!/usr/bin/env python3
"""
Basic AgentFarm simulation example.
This tutorial shows how to create and run a simple simulation.
"""

import random
import torch
from farm.core.environment import Environment
from farm.core.observations import ObservationConfig
from farm.core.agent import BaseAgent

def create_basic_simulation():
    """Create a basic simulation with 10 agents."""

    # 1. Configure observations
    obs_config = ObservationConfig(
        R=6,                    # Observation radius (6 cells in each direction)
        fov_radius=5,           # Field-of-view radius
        decay_factors={
            'trails': 0.95,     # Movement trails decay slowly
            'damage_heat': 0.90 # Combat heat decays moderately
        }
    )

    # 2. Create environment
    environment = Environment(
        width=50,               # 50x50 grid world
        height=50,
        resource_distribution="uniform",  # Resources distributed evenly
        obs_config=obs_config,
        initial_resource_count=200,       # Start with 200 resource units
        resource_regeneration_rate=0.02   # 2% regeneration per step
    )

    # 3. Add agents
    for i in range(10):
        agent = BaseAgent(
            agent_id=f"agent_{i:02d}",
            position=(random.randint(0, 49), random.randint(0, 49)),
            resource_level=100,  # Start with 100 resource units
            environment=environment,
            learning_rate=0.001,
            memory_size=5000
        )
        environment.add_agent(agent)

    return environment

def run_basic_simulation():
    """Run the basic simulation for 100 steps."""

    print("Creating basic simulation...")
    environment = create_basic_simulation()

    print(f"Simulation initialized with {len(environment.agents)} agents")
    print(f"Environment size: {environment.width}x{environment.height}")
    print(f"Initial resources: {environment.resources.total_count()}")

    # Run simulation
    for step in range(100):
        # Get actions from all active agents
        actions = {}
        for agent_id, agent in environment.agents.items():
            if not agent.is_terminated:
                action = agent.decide_action()
                actions[agent_id] = action

        # Execute simulation step
        results = environment.step(actions)

        # Print progress every 10 steps
        if step % 10 == 0:
            alive_agents = sum(1 for agent in environment.agents.values()
                             if not agent.is_terminated)
            print(f"Step {step:3d}: {alive_agents} agents alive, "
                  f"{environment.resources.total_count()} resources")

    print("Simulation completed!")

    # Print final statistics
    final_alive = sum(1 for agent in environment.agents.values()
                     if not agent.is_terminated)
    print(f"Final state: {final_alive} agents survived")
    print(f"Final resources: {environment.resources.total_count()}")

if __name__ == "__main__":
    run_basic_simulation()
```

### Running the Example

```bash
cd /path/to/AgentFarm
python basic_simulation.py
```

**Expected Output:**
```
Creating basic simulation...
Simulation initialized with 10 agents
Environment size: 50x50
Initial resources: 200
Step   0: 10 agents alive, 200 resources
Step  10: 10 agents alive, 198 resources
Step  20: 10 agents alive, 196 resources
...
Step  90: 9 agents alive, 185 resources
Step 100: 8 agents alive, 182 resources
Simulation completed!
Final state: 8 agents survived
Final resources: 182 resources
```

## Tutorial 2: Custom Agent Behaviors

### Implementing a Cooperative Agent

```python
#!/usr/bin/env python3
"""
Custom agent implementation example.
This tutorial shows how to create agents with specialized behaviors.
"""

import random
import numpy as np
from typing import Tuple, Optional
from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.core.observations import ObservationConfig
from farm.actions.share import share_action

class CooperativeAgent(BaseAgent):
    """An agent that prioritizes cooperation and resource sharing."""

    def __init__(self, agent_id: str, position: Tuple[int, int],
                 resource_level: int, environment: Environment, **kwargs):
        super().__init__(agent_id, position, resource_level, environment, **kwargs)

        # Cooperative behavior parameters
        self.sharing_threshold = kwargs.get('sharing_threshold', 150)
        self.cooperation_radius = kwargs.get('cooperation_radius', 3)
        self.generosity_factor = kwargs.get('generosity_factor', 0.3)

    def decide_action(self):
        """Decide action with cooperative priorities."""

        # 1. Check if we should share resources
        if self._should_share_resources():
            ally_id = self._find_needy_ally()
            if ally_id:
                return {
                    'action_type': 'share',
                    'target_agent': ally_id,
                    'resource_amount': int(self.resource_level * self.generosity_factor)
                }

        # 2. Otherwise, use default decision making
        return super().decide_action()

    def _should_share_resources(self) -> bool:
        """Determine if agent should share resources."""
        return (self.resource_level > self.sharing_threshold and
                random.random() < 0.7)  # 70% chance when above threshold

    def _find_needy_ally(self) -> Optional[str]:
        """Find a nearby ally that needs resources."""
        nearby_agents = self.environment.get_nearby_agents(
            self.position, self.cooperation_radius
        )

        needy_allies = []
        for agent_id in nearby_agents:
            if agent_id == self.agent_id:
                continue

            agent = self.environment.agents[agent_id]
            if not agent.is_terminated and agent.resource_level < 80:
                needy_allies.append((agent_id, agent.resource_level))

        # Return ally with lowest resources
        if needy_allies:
            needy_allies.sort(key=lambda x: x[1])  # Sort by resource level
            return needy_allies[0][0]

        return None

class CompetitiveAgent(BaseAgent):
    """An agent that prioritizes competition and resource hoarding."""

    def __init__(self, agent_id: str, position: Tuple[int, int],
                 resource_level: int, environment: Environment, **kwargs):
        super().__init__(agent_id, position, resource_level, environment, **kwargs)

        self.competitive_radius = kwargs.get('competitive_radius', 4)
        self.attack_threshold = kwargs.get('attack_threshold', 120)

    def decide_action(self):
        """Decide action with competitive priorities."""

        # 1. Check if we should attack weak neighbors
        if self._should_attack():
            target_id = self._find_weak_target()
            if target_id:
                return {
                    'action_type': 'attack',
                    'target_agent': target_id
                }

        # 2. Otherwise, use default decision making
        return super().decide_action()

    def _should_attack(self) -> bool:
        """Determine if agent should attack."""
        return (self.resource_level > self.attack_threshold and
                random.random() < 0.4)  # 40% chance when strong

    def _find_weak_target(self) -> Optional[str]:
        """Find a nearby weak agent to attack."""
        nearby_agents = self.environment.get_nearby_agents(
            self.position, self.competitive_radius
        )

        weak_targets = []
        for agent_id in nearby_agents:
            if agent_id == self.agent_id:
                continue

            agent = self.environment.agents[agent_id]
            if not agent.is_terminated and agent.resource_level < self.resource_level * 0.7:
                weak_targets.append((agent_id, agent.resource_level))

        # Return weakest target
        if weak_targets:
            weak_targets.sort(key=lambda x: x[1])  # Sort by resource level
            return weak_targets[0][0]

        return None

def create_mixed_simulation():
    """Create simulation with different agent types."""

    obs_config = ObservationConfig(R=6, fov_radius=5)

    environment = Environment(
        width=60, height=60,
        resource_distribution="clustered",
        obs_config=obs_config,
        initial_resource_count=300
    )

    # Add cooperative agents
    for i in range(5):
        agent = CooperativeAgent(
            agent_id=f"coop_{i:02d}",
            position=(random.randint(10, 25), random.randint(10, 25)),
            resource_level=100,
            environment=environment,
            sharing_threshold=140,
            generosity_factor=0.4
        )
        environment.add_agent(agent)

    # Add competitive agents
    for i in range(5):
        agent = CompetitiveAgent(
            agent_id=f"comp_{i:02d}",
            position=(random.randint(35, 50), random.randint(35, 50)),
            resource_level=100,
            environment=environment,
            attack_threshold=130
        )
        environment.add_agent(agent)

    return environment

def run_behavior_comparison():
    """Compare cooperative vs competitive behaviors."""

    print("Running behavior comparison simulation...")
    environment = create_mixed_simulation()

    coop_agents = [aid for aid in environment.agents.keys() if aid.startswith('coop')]
    comp_agents = [aid for aid in environment.agents.keys() if aid.startswith('comp')]

    print(f"Cooperative agents: {len(coop_agents)}")
    print(f"Competitive agents: {len(comp_agents)}")

    # Track behavior metrics
    sharing_events = 0
    attack_events = 0

    for step in range(200):
        actions = {}
        for agent_id, agent in environment.agents.items():
            if not agent.is_terminated:
                action = agent.decide_action()
                actions[agent_id] = action

                # Track special actions
                if action.get('action_type') == 'share':
                    sharing_events += 1
                elif action.get('action_type') == 'attack':
                    attack_events += 1

        environment.step(actions)

        # Print progress
        if step % 50 == 0:
            alive_coop = sum(1 for aid in coop_agents
                           if aid in environment.agents and not environment.agents[aid].is_terminated)
            alive_comp = sum(1 for aid in comp_agents
                           if aid in environment.agents and not environment.agents[aid].is_terminated)

            print(f"Step {step:3d}: Coop={alive_coop}, Comp={alive_comp}, "
                  f"Sharing={sharing_events}, Attacks={attack_events}")

    print("\nBehavior comparison completed!")
    print(f"Total sharing events: {sharing_events}")
    print(f"Total attack events: {attack_events}")

if __name__ == "__main__":
    run_behavior_comparison()
```

## Tutorial 3: Custom Observation Channels

### Implementing Environmental Awareness

```python
#!/usr/bin/env python3
"""
Custom observation channels example.
This tutorial shows how to extend the observation system with custom channels.
"""

import torch
import random
import numpy as np
from typing import Tuple
from farm.core.channels import ChannelHandler, ChannelBehavior, register_channel
from farm.core.environment import Environment
from farm.core.observations import ObservationConfig
from farm.core.agent import BaseAgent

class WeatherChannel(ChannelHandler):
    """Channel representing dynamic weather conditions."""

    def __init__(self, weather_system):
        super().__init__("WEATHER", ChannelBehavior.DYNAMIC, gamma=0.98)
        self.weather_system = weather_system

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        """Update weather information in observation."""
        x, y = agent_world_pos

        # Get weather intensity at agent position
        weather_intensity = self.weather_system.get_weather_at(x, y)

        # Encode weather as channel values
        obs_size = observation.shape[-1]
        center = obs_size // 2

        # Weather affects visibility and movement
        observation[channel_idx, center, center] = weather_intensity

        # Add some spatial variation based on weather patterns
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if abs(dx) + abs(dy) <= 2:  # Diamond pattern
                    px, py = center + dx, center + dy
                    if 0 <= px < obs_size and 0 <= py < obs_size:
                        local_weather = self.weather_system.get_weather_at(
                            x + dx, y + dy
                        )
                        observation[channel_idx, px, py] = local_weather * 0.7

class WeatherSystem:
    """Simulates dynamic weather patterns."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.weather_map = np.zeros((height, width))
        self.weather_centers = []  # Storm centers

        # Initialize with random weather patterns
        self._initialize_weather()

    def _initialize_weather(self):
        """Create initial weather patterns."""
        # Add some random storm centers
        for _ in range(3):
            center_x = random.randint(5, self.width - 5)
            center_y = random.randint(5, self.height - 5)
            intensity = random.uniform(0.3, 0.8)
            self.weather_centers.append((center_x, center_y, intensity))

    def update_weather(self):
        """Update weather patterns over time."""
        # Slowly move weather centers
        for i, (x, y, intensity) in enumerate(self.weather_centers):
            # Random walk for weather centers
            new_x = x + random.randint(-1, 1)
            new_y = y + random.randint(-1, 1)

            # Keep within bounds
            new_x = max(0, min(self.width - 1, new_x))
            new_y = max(0, min(self.height - 1, new_y))

            self.weather_centers[i] = (new_x, new_y, intensity)

        # Update weather map
        self.weather_map.fill(0)
        for x, y, intensity in self.weather_centers:
            # Create radial weather pattern
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    px, py = x + dx, y + dy
                    if 0 <= px < self.width and 0 <= py < self.height:
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance <= 5:
                            weather_effect = intensity * (1 - distance/5)
                            self.weather_map[py, px] = max(
                                self.weather_map[py, px], weather_effect
                            )

    def get_weather_at(self, x: int, y: int) -> float:
        """Get weather intensity at specific location."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.weather_map[y, x]
        return 0.0

class ResourceDensityChannel(ChannelHandler):
    """Channel showing resource density in the environment."""

    def __init__(self):
        super().__init__("RESOURCE_DENSITY", ChannelBehavior.INSTANT)

    def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        """Update resource density information."""
        environment = kwargs.get('environment')
        if not environment:
            return

        x, y = agent_world_pos
        obs_size = observation.shape[-1]
        radius = obs_size // 2

        # Count resources in observation area
        total_resources = 0
        count = 0

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                px, py = x + dx, y + dy
                if (0 <= px < environment.width and
                    0 <= py < environment.height):
                    resources_at_pos = environment.resources.get_count_at(px, py)
                    total_resources += resources_at_pos
                    count += 1

        # Normalize by area and maximum possible resources
        if count > 0:
            density = total_resources / count
            # Normalize to 0-1 range (assuming max 10 resources per cell)
            normalized_density = min(density / 10.0, 1.0)

            # Fill channel with density information
            observation[channel_idx].fill_(normalized_density)

class WeatherAwareAgent(BaseAgent):
    """Agent that uses weather information for decision making."""

    def __init__(self, agent_id: str, position: Tuple[int, int],
                 resource_level: int, environment: Environment, **kwargs):
        super().__init__(agent_id, position, resource_level, environment, **kwargs)
        self.weather_aversion = kwargs.get('weather_aversion', 0.5)

    def decide_action(self):
        """Make decisions considering weather conditions."""

        # Get weather at current position
        weather_system = self.environment.weather_system
        current_weather = weather_system.get_weather_at(*self.position)

        # If weather is bad, consider moving to better area
        if current_weather > 0.6 and random.random() < self.weather_aversion:
            return self._find_better_weather_position()

        # Otherwise, use normal decision making
        return super().decide_action()

    def _find_better_weather_position(self):
        """Find a nearby position with better weather."""
        best_position = None
        best_weather = float('inf')
        weather_system = self.environment.weather_system

        # Check nearby positions
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue

                new_x = self.position[0] + dx
                new_y = self.position[1] + dy

                if (0 <= new_x < self.environment.width and
                    0 <= new_y < self.environment.height):

                    weather = weather_system.get_weather_at(new_x, new_y)
                    if weather < best_weather:
                        best_weather = weather
                        best_position = (new_x, new_y)

        if best_position:
            return {
                'action_type': 'move',
                'target_position': best_position
            }

        return super().decide_action()

def create_weather_simulation():
    """Create simulation with weather and custom channels."""

    # Create weather system
    weather_system = WeatherSystem(width=50, height=50)

    # Configure observations
    obs_config = ObservationConfig(R=8, fov_radius=6)

    # Create environment with weather system
    environment = Environment(
        width=50, height=50,
        resource_distribution="clustered",
        obs_config=obs_config,
        initial_resource_count=250
    )

    # Add weather system to environment
    environment.weather_system = weather_system

    # Register custom channels
    weather_channel_idx = register_channel(WeatherChannel(weather_system))
    resource_density_idx = register_channel(ResourceDensityChannel())

    print(f"Registered weather channel at index: {weather_channel_idx}")
    print(f"Registered resource density channel at index: {resource_density_idx}")

    # Add weather-aware agents
    for i in range(8):
        agent = WeatherAwareAgent(
            agent_id=f"weather_agent_{i:02d}",
            position=(random.randint(0, 49), random.randint(0, 49)),
            resource_level=100,
            environment=environment,
            weather_aversion=0.6
        )
        environment.add_agent(agent)

    return environment, weather_system

def run_weather_simulation():
    """Run simulation with weather dynamics."""

    print("Creating weather-aware simulation...")
    environment, weather_system = create_weather_simulation()

    print(f"Simulation initialized with {len(environment.agents)} weather-aware agents")

    # Run simulation with weather updates
    for step in range(150):
        # Update weather patterns
        weather_system.update_weather()

        # Get actions from agents
        actions = {}
        for agent_id, agent in environment.agents.items():
            if not agent.is_terminated:
                action = agent.decide_action()
                actions[agent_id] = action

        # Execute simulation step
        results = environment.step(actions)

        # Print progress
        if step % 30 == 0:
            alive_agents = sum(1 for agent in environment.agents.values()
                             if not agent.is_terminated)
            total_weather = np.mean(weather_system.weather_map)
            print(".3f")

    print("Weather simulation completed!")

if __name__ == "__main__":
    run_weather_simulation()
```

## Tutorial 4: Experiment Management

### Running Parameter Studies

```python
#!/usr/bin/env python3
"""
Experiment management example.
This tutorial shows how to set up and run systematic parameter studies.
"""

import json
import random
from typing import Dict, List, Any
from farm.core.environment import Environment
from farm.core.observations import ObservationConfig
from farm.core.agent import BaseAgent
from farm.core.config import ExperimentConfig
from farm.runners.experiment_runner import ExperimentRunner

class ExperimentRunner:
    """Custom experiment runner for parameter studies."""

    def __init__(self):
        self.results = []

    def run_parameter_study(self, base_config: Dict[str, Any],
                           parameter_ranges: Dict[str, List[Any]],
                           num_replications: int = 3,
                           steps_per_run: int = 500) -> List[Dict[str, Any]]:
        """Run parameter study with multiple configurations."""

        print(f"Starting parameter study with {len(parameter_ranges)} parameters")
        print(f"Running {num_replications} replications per configuration")

        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)

        print(f"Total configurations to test: {len(param_combinations)}")

        results = []
        for i, params in enumerate(param_combinations):
            print(f"\nRunning configuration {i+1}/{len(param_combinations)}")
            print(f"Parameters: {params}")

            # Run multiple replications for statistical significance
            replication_results = []
            for rep in range(num_replications):
                print(f"  Replication {rep+1}/{num_replications}")

                # Merge base config with current parameters
                config = base_config.copy()
                config.update(params)

                # Run simulation
                result = self._run_single_simulation(config, steps_per_run)
                replication_results.append(result)

            # Aggregate results across replications
            aggregated_result = self._aggregate_results(replication_results, params)
            results.append(aggregated_result)

        return results

    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        if not parameter_ranges:
            return [{}]

        # Simple parameter combination (could use itertools.product for more complex cases)
        combinations = []

        def generate_combinations(current: Dict[str, Any], remaining_params: List[str]):
            if not remaining_params:
                combinations.append(current.copy())
                return

            param_name = remaining_params[0]
            remaining = remaining_params[1:]

            for value in parameter_ranges[param_name]:
                current[param_name] = value
                generate_combinations(current, remaining)

        generate_combinations({}, list(parameter_ranges.keys()))
        return combinations

    def _run_single_simulation(self, config: Dict[str, Any], steps: int) -> Dict[str, Any]:
        """Run a single simulation with given configuration."""

        # Create observation configuration
        obs_config = ObservationConfig(
            R=config.get('observation_radius', 6),
            fov_radius=config.get('fov_radius', 5)
        )

        # Create environment
        environment = Environment(
            width=config.get('world_width', 50),
            height=config.get('world_height', 50),
            resource_distribution=config.get('resource_distribution', 'uniform'),
            obs_config=obs_config,
            initial_resource_count=config.get('initial_resources', 200)
        )

        # Add agents
        num_agents = config.get('num_agents', 10)
        for i in range(num_agents):
            agent = BaseAgent(
                agent_id=f"agent_{i:02d}",
                position=(random.randint(0, environment.width-1),
                         random.randint(0, environment.height-1)),
                resource_level=config.get('initial_agent_resources', 100),
                environment=environment,
                learning_rate=config.get('learning_rate', 0.001)
            )
            environment.add_agent(agent)

        # Track metrics
        survival_counts = []
        resource_counts = []

        # Run simulation
        for step in range(steps):
            actions = {}
            for agent_id, agent in environment.agents.items():
                if not agent.is_terminated:
                    action = agent.decide_action()
                    actions[agent_id] = action

            environment.step(actions)

            # Record metrics every 50 steps
            if step % 50 == 0:
                alive_count = sum(1 for agent in environment.agents.values()
                                if not agent.is_terminated)
                resource_count = environment.resources.total_count()

                survival_counts.append(alive_count)
                resource_counts.append(resource_count)

        # Calculate final metrics
        final_alive = sum(1 for agent in environment.agents.values()
                         if not agent.is_terminated)

        return {
            'final_survival_rate': final_alive / num_agents,
            'avg_survival_over_time': survival_counts,
            'resource_trajectory': resource_counts,
            'final_resources': environment.resources.total_count(),
            'steps_completed': steps
        }

    def _aggregate_results(self, replication_results: List[Dict[str, Any]],
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results across replications."""

        aggregated = {
            'parameters': params,
            'num_replications': len(replication_results),
            'avg_survival_rate': 0,
            'std_survival_rate': 0,
            'avg_final_resources': 0,
            'survival_trajectories': [],
            'resource_trajectories': []
        }

        # Collect all survival rates
        survival_rates = [r['final_survival_rate'] for r in replication_results]

        # Calculate statistics
        aggregated['avg_survival_rate'] = sum(survival_rates) / len(survival_rates)
        aggregated['std_survival_rate'] = (sum((x - aggregated['avg_survival_rate'])**2
                                               for x in survival_rates) / len(survival_rates))**0.5

        # Aggregate trajectories
        for result in replication_results:
            aggregated['survival_trajectories'].append(result['avg_survival_over_time'])
            aggregated['resource_trajectories'].append(result['resource_trajectory'])

        return aggregated

def run_resource_distribution_study():
    """Study how resource distribution affects agent behavior."""

    # Base configuration
    base_config = {
        'world_width': 60,
        'world_height': 60,
        'num_agents': 15,
        'initial_agent_resources': 100,
        'initial_resources': 300,
        'observation_radius': 7,
        'fov_radius': 5,
        'learning_rate': 0.001
    }

    # Parameter ranges to study
    parameter_ranges = {
        'resource_distribution': ['uniform', 'clustered', 'scattered'],
        'num_agents': [10, 15, 20],
        'initial_resources': [200, 300, 400]
    }

    # Run experiment
    runner = ExperimentRunner()
    results = runner.run_parameter_study(
        base_config=base_config,
        parameter_ranges=parameter_ranges,
        num_replications=3,
        steps_per_run=300
    )

    # Save results
    with open('resource_study_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n=== Resource Distribution Study Results ===")
    for result in results:
        params = result['parameters']
        print(".3f"
              f"±{result['std_survival_rate']:.3f}")

    return results

def run_learning_parameter_study():
    """Study how learning parameters affect agent performance."""

    base_config = {
        'world_width': 50,
        'world_height': 50,
        'num_agents': 12,
        'initial_agent_resources': 100,
        'initial_resources': 250,
        'resource_distribution': 'clustered',
        'observation_radius': 6,
        'fov_radius': 5
    }

    # Focus on learning parameters
    parameter_ranges = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'memory_size': [1000, 5000, 10000]
    }

    runner = ExperimentRunner()
    results = runner.run_parameter_study(
        base_config=base_config,
        parameter_ranges=parameter_ranges,
        num_replications=5,
        steps_per_run=400
    )

    # Save results
    with open('learning_study_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== Learning Parameter Study Results ===")
    for result in results:
        params = result['parameters']
        print(".4f"
              f"±{result['std_survival_rate']:.3f}")

    return results

if __name__ == "__main__":
    print("Running resource distribution study...")
    resource_results = run_resource_distribution_study()

    print("\nRunning learning parameter study...")
    learning_results = run_learning_parameter_study()

    print("\nExperiment studies completed!")
    print("Results saved to JSON files for further analysis.")
```

## Tutorial 5: Analysis and Visualization

### Creating Custom Analysis Scripts

```python
#!/usr/bin/env python3
"""
Analysis and visualization example.
This tutorial shows how to analyze simulation results and create visualizations.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

class SimulationAnalyzer:
    """Comprehensive analyzer for AgentFarm simulation results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def load_experiment_results(self, filename: str) -> List[Dict[str, Any]]:
        """Load experiment results from JSON file."""
        filepath = self.results_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)

    def create_survival_analysis(self, results: List[Dict[str, Any]],
                                title: str = "Survival Analysis"):
        """Create survival rate analysis plot."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Prepare data
        survival_data = []
        for result in results:
            params = result['parameters']
            avg_survival = result['avg_survival_rate']
            std_survival = result['std_survival_rate']

            survival_data.append({
                'parameters': str(params),
                'avg_survival': avg_survival,
                'std_survival': std_survival,
                'resource_dist': params.get('resource_distribution', 'unknown'),
                'num_agents': params.get('num_agents', 0)
            })

        df = pd.DataFrame(survival_data)

        # Plot 1: Survival by resource distribution
        if 'resource_dist' in df.columns:
            ax = axes[0, 0]
            resource_survival = df.groupby('resource_dist')['avg_survival'].mean()
            resource_survival.plot(kind='bar', ax=ax, yerr=df.groupby('resource_dist')['std_survival'].std())
            ax.set_title('Survival Rate by Resource Distribution')
            ax.set_ylabel('Survival Rate')
            ax.tick_params(axis='x', rotation=45)

        # Plot 2: Survival by number of agents
        if 'num_agents' in df.columns:
            ax = axes[0, 1]
            agent_survival = df.groupby('num_agents')['avg_survival'].mean()
            agent_survival.plot(kind='bar', ax=ax, yerr=df.groupby('num_agents')['std_survival'].std())
            ax.set_title('Survival Rate by Agent Count')
            ax.set_ylabel('Survival Rate')

        # Plot 3: Parameter correlation heatmap
        ax = axes[1, 0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation = df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Parameter Correlation')

        # Plot 4: Survival distribution
        ax = axes[1, 1]
        ax.hist(df['avg_survival'], bins=10, alpha=0.7, edgecolor='black')
        ax.axvline(df['avg_survival'].mean(), color='red', linestyle='--',
                  label='.3f')
        ax.set_title('Survival Rate Distribution')
        ax.set_xlabel('Survival Rate')
        ax.set_ylabel('Frequency')
        ax.legend()

        plt.tight_layout()
        return fig

    def create_trajectory_analysis(self, results: List[Dict[str, Any]],
                                  title: str = "Trajectory Analysis"):
        """Analyze how metrics change over time."""

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Collect trajectory data
        survival_trajectories = []
        resource_trajectories = []
        labels = []

        for result in results:
            params = result['parameters']
            label = f"{params.get('resource_distribution', 'unknown')}_agents{params.get('num_agents', 0)}"

            if 'survival_trajectories' in result:
                # Average across replications
                survival_traj = np.mean(result['survival_trajectories'], axis=0)
                survival_trajectories.append(survival_traj)

            if 'resource_trajectories' in result:
                resource_traj = np.mean(result['resource_trajectories'], axis=0)
                resource_trajectories.append(resource_traj)

            labels.append(label)

        # Plot survival trajectories
        ax = axes[0]
        for i, trajectory in enumerate(survival_trajectories):
            steps = len(trajectory)
            ax.plot(range(steps), trajectory, label=labels[i], marker='o', markersize=2)

        ax.set_title('Agent Survival Over Time')
        ax.set_xlabel('Simulation Steps')
        ax.set_ylabel('Number of Surviving Agents')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot resource trajectories
        ax = axes[1]
        for i, trajectory in enumerate(resource_trajectories):
            steps = len(trajectory)
            ax.plot(range(steps), trajectory, label=labels[i], marker='s', markersize=2)

        ax.set_title('Resource Count Over Time')
        ax.set_xlabel('Simulation Steps')
        ax.set_ylabel('Total Resources')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_comparative_report(self, results: List[Dict[str, Any]],
                                 filename: str = "comparative_report"):
        """Create comprehensive comparative report."""

        # Generate plots
        survival_fig = self.create_survival_analysis(results, "Comparative Survival Analysis")
        trajectory_fig = self.create_trajectory_analysis(results, "Comparative Trajectory Analysis")

        # Save plots
        survival_fig.savefig(self.results_dir / f"{filename}_survival.png", dpi=300, bbox_inches='tight')
        trajectory_fig.savefig(self.results_dir / f"{filename}_trajectory.png", dpi=300, bbox_inches='tight')

        # Generate statistical summary
        summary = self._generate_statistical_summary(results)

        # Save summary
        with open(self.results_dir / f"{filename}_summary.txt", 'w') as f:
            f.write(summary)

        print(f"Comparative report saved to {self.results_dir}")
        print(f"- Survival analysis: {filename}_survival.png")
        print(f"- Trajectory analysis: {filename}_trajectory.png")
        print(f"- Statistical summary: {filename}_summary.txt")

        return {
            'survival_plot': survival_fig,
            'trajectory_plot': trajectory_fig,
            'summary': summary
        }

    def _generate_statistical_summary(self, results: List[Dict[str, Any]]) -> str:
        """Generate statistical summary of results."""

        summary_lines = ["=== Statistical Summary ===\n"]

        # Overall statistics
        survival_rates = [r['avg_survival_rate'] for r in results]
        summary_lines.append(f"Total configurations tested: {len(results)}")
        summary_lines.append(".3f")
        summary_lines.append(".3f")

        # Best and worst performers
        best_idx = np.argmax(survival_rates)
        worst_idx = np.argmin(survival_rates)

        summary_lines.append(f"\nBest performing configuration:")
        summary_lines.append(f"  Parameters: {results[best_idx]['parameters']}")
        summary_lines.append(".3f")

        summary_lines.append(f"\nWorst performing configuration:")
        summary_lines.append(f"  Parameters: {results[worst_idx]['parameters']}")
        summary_lines.append(".3f")

        # Group by resource distribution
        resource_groups = {}
        for result in results:
            dist = result['parameters'].get('resource_distribution', 'unknown')
            if dist not in resource_groups:
                resource_groups[dist] = []
            resource_groups[dist].append(result['avg_survival_rate'])

        summary_lines.append(f"\nSurvival by Resource Distribution:")
        for dist, rates in resource_groups.items():
            avg_rate = np.mean(rates)
            std_rate = np.std(rates)
            summary_lines.append(".3f")

        return "\n".join(summary_lines)

def analyze_experiment_results():
    """Analyze results from experiment studies."""

    analyzer = SimulationAnalyzer()

    try:
        # Load resource study results
        print("Loading resource distribution study results...")
        resource_results = analyzer.load_experiment_results("resource_study_results.json")

        # Create comparative report
        print("Generating resource study report...")
        analyzer.create_comparative_report(
            resource_results,
            filename="resource_study_analysis"
        )

    except FileNotFoundError:
        print("Resource study results not found. Run experiment first.")

    try:
        # Load learning study results
        print("Loading learning parameter study results...")
        learning_results = analyzer.load_experiment_results("learning_study_results.json")

        # Create comparative report
        print("Generating learning study report...")
        analyzer.create_comparative_report(
            learning_results,
            filename="learning_study_analysis"
        )

    except FileNotFoundError:
        print("Learning study results not found. Run experiment first.")

def create_custom_visualization():
    """Create custom visualizations for specific analysis."""

    analyzer = SimulationAnalyzer()

    # Create a custom comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Example: Compare learning rates
    learning_rates = [0.0001, 0.001, 0.01]
    survival_rates = [0.65, 0.78, 0.72]  # Example data
    std_errors = [0.05, 0.03, 0.04]

    ax = axes[0]
    ax.errorbar(learning_rates, survival_rates, yerr=std_errors,
               marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Survival Rate')
    ax.set_title('Survival Rate vs Learning Rate')
    ax.grid(True, alpha=0.3)

    # Example: Resource distribution comparison
    distributions = ['Uniform', 'Clustered', 'Scattered']
    survival_by_dist = [0.71, 0.82, 0.68]

    ax = axes[1]
    bars = ax.bar(distributions, survival_by_dist, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Resource Distribution')
    ax.set_ylabel('Survival Rate')
    ax.set_title('Survival Rate by Resource Distribution')

    # Add value labels on bars
    for bar, rate in zip(bars, survival_by_dist):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               '.3f', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(analyzer.results_dir / "custom_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Running experiment analysis...")
    analyze_experiment_results()

    print("\nCreating custom visualization...")
    create_custom_visualization()

    print("\nAnalysis completed!")
```

These tutorials provide a comprehensive introduction to using AgentFarm effectively, from basic simulations to advanced customizations and analysis.
