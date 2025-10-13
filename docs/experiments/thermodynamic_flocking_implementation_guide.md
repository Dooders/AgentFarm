# Thermodynamic Flocking Simulation - Implementation Guide for AgentFarm

## Executive Summary

This document provides a comprehensive guide for implementing the **Thermodynamic Flocking Simulation** using the existing AgentFarm library. The simulation demonstrates ESDP Principle 2 (Thermodynamic Realism) applied to flocking behavior, where agents follow local alignment, cohesion, and separation rules but incur energy costs for movement, leading to emergent efficient swarms that balance cohesion and dissipation.

**Implementation Approach**: The AgentFarm library provides excellent infrastructure for this simulation. We can implement it primarily through **configuration** and **custom agent classes**, leveraging existing systems for spatial queries, metrics tracking, and visualization.

---

## Table of Contents

1. [Conceptual Mapping](#conceptual-mapping)
2. [Implementation Architecture](#implementation-architecture)
3. [Detailed Implementation Plan](#detailed-implementation-plan)
4. [Code Examples](#code-examples)
5. [Configuration](#configuration)
6. [Metrics and Analysis](#metrics-and-analysis)
7. [Visualization](#visualization)
8. [Extensions and Variants](#extensions-and-variants)

---

## Conceptual Mapping

### Original Simulation → AgentFarm Equivalents

| Original Component | AgentFarm Equivalent | Implementation Strategy |
|-------------------|---------------------|------------------------|
| **FlockingAgent** | `BaseAgent` subclass | Create `FlockingAgent(BaseAgent)` with custom attributes |
| **Position/Velocity** | `agent.position` + custom `velocity` | Add velocity as agent state, update in custom `act()` |
| **Energy Budget** | `agent.resource_level` | Repurpose resources as "energy units" |
| **Perception Radius** | `SpatialIndex.get_nearby()` | Use existing spatial queries for neighbor detection |
| **Flocking Rules** | Custom actions | Implement as new action types or within custom `decide_action()` |
| **Energy Sources** | `Resource` objects | Use existing resource system as energy replenishment |
| **Toroidal Boundaries** | Custom boundary logic | Override `apply_boundary()` method |
| **Metrics Tracking** | `MetricsTracker` + custom | Extend metrics system with flocking-specific measurements |
| **Animation** | Database + external viz | Log to DB, then create visualization script |

---

## Implementation Architecture

### Three-Layer Approach

#### **Layer 1: Core Flocking Agent**
- Extend `BaseAgent` with flocking-specific attributes
- Implement velocity-based movement
- Add energy consumption mechanics
- Override action decision-making

#### **Layer 2: Flocking Actions**
- Create custom actions: `AlignmentAction`, `CohesionAction`, `SeparationAction`
- Integrate with existing action system
- Energy cost calculation per action

#### **Layer 3: Environment Configuration**
- Configure spatial settings for efficient neighbor queries
- Set up energy resource distribution
- Define simulation parameters via YAML

---

## Detailed Implementation Plan

### Step 1: Create FlockingAgent Class

**Location**: `farm/core/flocking_agent.py`

**Key Components**:

```python
class FlockingAgent(BaseAgent):
    """Agent with velocity-based flocking behavior and energy constraints."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Flocking-specific attributes
        self.velocity = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0)
        ], dtype=float)
        
        self.max_speed = 2.0
        self.max_force = 0.5
        
        # Energy system (using resource_level as energy)
        # self.resource_level serves as energy
        
        # Flocking weights (configurable)
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        self.separation_weight = 1.5
        self.perception_radius = 10.0
        self.separation_radius = 5.0
```

**Key Methods**:
1. `compute_alignment()` - steer toward average velocity of neighbors
2. `compute_cohesion()` - steer toward center of mass of neighbors
3. `compute_separation()` - avoid crowding neighbors
4. `apply_flocking_forces()` - combine all steering forces
5. `update_energy()` - calculate energy cost based on velocity
6. `act()` - override to implement flocking behavior

### Step 2: Implement Flocking Steering Behaviors

The flocking behaviors can be implemented using AgentFarm's spatial query system:

```python
def compute_alignment(self):
    """Steer towards average heading of neighbors."""
    # Use AgentFarm's spatial service
    nearby = self.spatial_service.get_nearby(
        self.position, 
        self.perception_radius, 
        ["agents"]
    )
    neighbors = nearby.get("agents", [])
    
    if not neighbors:
        return np.zeros(2)
    
    # Average velocity of neighbors
    avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
    desired = avg_velocity - self.velocity
    return desired

def compute_cohesion(self):
    """Steer towards average position of neighbors."""
    nearby = self.spatial_service.get_nearby(
        self.position, 
        self.perception_radius, 
        ["agents"]
    )
    neighbors = nearby.get("agents", [])
    
    if not neighbors:
        return np.zeros(2)
    
    # Center of mass
    center = np.mean([n.position for n in neighbors], axis=0)
    desired = (center - np.array(self.position)) * 0.01  # Gentle steering
    return desired

def compute_separation(self):
    """Steer away from neighbors that are too close."""
    nearby = self.spatial_service.get_nearby(
        self.position, 
        self.separation_radius, 
        ["agents"]
    )
    neighbors = nearby.get("agents", [])
    
    if not neighbors:
        return np.zeros(2)
    
    steering = np.zeros(2)
    for neighbor in neighbors:
        diff = np.array(self.position) - np.array(neighbor.position)
        distance = np.linalg.norm(diff)
        if distance > 0:
            # Weight by inverse distance squared
            steering += diff / (distance ** 2)
    
    return steering
```

### Step 3: Energy Consumption Mechanics

Integrate energy costs into the agent's `act()` method:

```python
def act(self):
    """Execute flocking behavior with energy constraints."""
    if not self.alive:
        return
    
    # Calculate flocking forces
    align = self.compute_alignment() * self.alignment_weight
    cohere = self.compute_cohesion() * self.cohesion_weight
    separate = self.compute_separation() * self.separation_weight
    
    # Combine forces
    total_force = align + cohere + separate
    
    # Apply force (limit magnitude)
    force_mag = np.linalg.norm(total_force)
    if force_mag > self.max_force:
        total_force = total_force / force_mag * self.max_force
    
    # Update velocity
    self.velocity += total_force
    
    # Limit speed
    speed = np.linalg.norm(self.velocity)
    if speed > self.max_speed:
        self.velocity = self.velocity / speed * self.max_speed
    
    # Update position
    new_position = (
        self.position[0] + self.velocity[0],
        self.position[1] + self.velocity[1]
    )
    
    # Apply toroidal boundaries
    new_position = (
        new_position[0] % self.environment.width,
        new_position[1] % self.environment.height
    )
    
    self.update_position(new_position)
    
    # Energy consumption (thermodynamic costs)
    if not self.config.classic_mode:
        # Quadratic cost for realism: E ∝ v²
        energy_cost = 0.25 * speed ** 2
        
        # Base metabolic cost
        base_cost = 0.03
        
        # Deduct energy (using resource_level as energy)
        self.resource_level -= (energy_cost + base_cost)
        
        # Check for death by energy depletion
        if self.resource_level <= 0:
            self.terminate()
            return
    
    # Energy replenishment (ambient or from resources)
    # This can be handled by environment step or custom logic
```

### Step 4: Energy Resource System

**Option A: Ambient Energy Replenishment**
- Use Environment's existing step logic to add small energy increments
- Configure `resource_regen_rate` in YAML

**Option B: Sparse Energy Sources** (more interesting)
- Use existing `Resource` objects as energy pickups
- Agents collect energy when near resources
- Resources respawn after depletion

```python
def collect_energy_from_resources(self):
    """Collect energy from nearby resource nodes."""
    nearby = self.spatial_service.get_nearby(
        self.position,
        radius=3.0,  # Collection radius
        types=["resources"]
    )
    
    resources = nearby.get("resources", [])
    for resource in resources:
        if resource.amount > 0:
            # Collect energy
            collected = min(5.0, resource.amount)
            self.resource_level = min(
                self.resource_level + collected, 
                100.0  # Max energy cap
            )
            resource.amount -= collected
            
            # Log collection event
            if self.logging_service:
                self.logging_service.log_interaction_edge(
                    step_number=self.time_service.current_time(),
                    source_agent_id=self.agent_id,
                    target_agent_id=f"resource_{id(resource)}",
                    interaction_type="energy_collection",
                    value=collected
                )
```

### Step 5: Metrics Tracking

Extend `MetricsTracker` to capture flocking-specific metrics:

```python
# Add to custom environment or agent tracking
class FlockingMetrics:
    """Track flocking-specific emergence metrics."""
    
    def __init__(self):
        self.alignment_history = []
        self.cohesion_history = []
        self.entropy_production_history = []
        self.phase_synchrony_history = []
    
    def compute_alignment(self, agents):
        """Measure velocity coherence (0-1)."""
        if not agents:
            return 0.0
        
        velocities = np.array([a.velocity for a in agents])
        avg_vel = np.mean(velocities, axis=0)
        avg_speed = np.mean([np.linalg.norm(v) for v in velocities])
        
        if avg_speed < 1e-6:
            return 0.0
        
        alignment = np.linalg.norm(avg_vel) / avg_speed
        return alignment
    
    def compute_cohesion(self, agents):
        """Measure spatial clustering (0-1)."""
        if not agents:
            return 0.0
        
        positions = np.array([a.position for a in agents])
        center = np.mean(positions, axis=0)
        avg_dist = np.mean([
            np.linalg.norm(p - center) for p in positions
        ])
        
        # Normalized (arbitrary scaling)
        cohesion = 1.0 / (1.0 + avg_dist / 10.0)
        return cohesion
    
    def compute_entropy_production(self, agents):
        """Calculate entropy production rate (dissipation)."""
        if not agents:
            return 0.0
        
        # σ ≈ sum(v²) / N
        total_dissipation = sum(
            np.linalg.norm(a.velocity) ** 2 
            for a in agents
        )
        
        entropy_rate = total_dissipation / len(agents)
        return entropy_rate
    
    def compute_phase_synchrony(self, agents):
        """Kuramoto order parameter for velocity coordination."""
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
        
        return order_param
    
    def update(self, agents):
        """Compute and store all metrics."""
        self.alignment_history.append(self.compute_alignment(agents))
        self.cohesion_history.append(self.compute_cohesion(agents))
        self.entropy_production_history.append(
            self.compute_entropy_production(agents)
        )
        self.phase_synchrony_history.append(
            self.compute_phase_synchrony(agents)
        )
```

### Step 6: Environment Configuration

Create a custom environment or configure via YAML:

**Configuration File**: `farm/config/flocking_config.yaml`

```yaml
# Flocking Simulation Configuration
simulation:
  name: "thermodynamic_flocking"
  description: "Flocking with energy constraints"
  max_steps: 1000
  seed: 42

environment:
  width: 100
  height: 100
  position_discretization_method: "floor"
  
  spatial_index:
    enable_batch_updates: true
    region_size: 50.0
    enable_spatial_hash_indices: true
    spatial_hash_cell_size: 15.0

population:
  system_agents: 0
  independent_agents: 0
  control_agents: 0
  # Use custom agent initialization

resources:
  initial_resources: 8  # Energy sources
  resource_regen_rate: 0.02  # Respawn rate
  resource_regen_amount: 50  # Energy per source
  max_resource_amount: 50

# Flocking-specific parameters
flocking:
  n_agents: 50
  initial_energy_min: 30.0
  initial_energy_max: 100.0
  
  # Movement parameters
  max_speed: 2.0
  max_force: 0.5
  
  # Flocking weights
  alignment_weight: 1.0
  cohesion_weight: 1.0
  separation_weight: 1.5
  
  # Perception radii
  perception_radius: 10.0
  separation_radius: 5.0
  
  # Energy costs
  velocity_cost_coefficient: 0.25  # E = 0.25 * v²
  base_metabolic_cost: 0.03
  
  # Energy replenishment
  ambient_replenishment: 0.85  # Per step
  use_sparse_resources: true  # Use resource nodes
  energy_collection_radius: 3.0
  energy_collection_rate: 5.0
  
  # Modes
  classic_mode: false  # Disable thermodynamics
  aware_mode: false    # Enable adaptation
  evo_mode: false      # Enable evolution

database:
  use_in_memory_db: false
  db_pragma_profile: "balanced"
```

### Step 7: Simulation Runner

Create a simulation script that initializes agents and runs the simulation:

**File**: `scripts/run_flocking_simulation.py`

```python
import numpy as np
from farm.config import SimulationConfig
from farm.core.environment import Environment
from farm.core.flocking_agent import FlockingAgent  # Custom agent

def run_flocking_simulation(config_path="farm/config/flocking_config.yaml"):
    """Run thermodynamic flocking simulation."""
    
    # Load configuration
    from farm.config import load_config
    config = load_config(config_path)
    
    # Create environment
    env = Environment(
        width=config.environment.width,
        height=config.environment.height,
        resource_distribution={
            "type": "random",
            "amount": config.resources.initial_resources,
        },
        config=config,
        seed=config.seed
    )
    
    # Create flocking agents
    n_agents = config.flocking.n_agents
    
    for i in range(n_agents):
        # Random position
        position = (
            np.random.uniform(0, env.width),
            np.random.uniform(0, env.height)
        )
        
        # Random initial velocity
        velocity = np.random.uniform(-2.0, 2.0, 2)
        
        # Varied initial energy (thermodynamic diversity)
        if config.flocking.classic_mode:
            initial_energy = 100.0
        else:
            initial_energy = np.random.uniform(
                config.flocking.initial_energy_min,
                config.flocking.initial_energy_max
            )
        
        # Create agent
        agent = FlockingAgent(
            agent_id=env.get_next_agent_id(),
            position=position,
            resource_level=initial_energy,
            spatial_service=env.spatial_service,
            environment=env,
            agent_type="FlockingAgent",
            config=config
        )
        
        # Set velocity
        agent.velocity = velocity
        
        # Set flocking parameters from config
        agent.max_speed = config.flocking.max_speed
        agent.max_force = config.flocking.max_force
        agent.alignment_weight = config.flocking.alignment_weight
        agent.cohesion_weight = config.flocking.cohesion_weight
        agent.separation_weight = config.flocking.separation_weight
        agent.perception_radius = config.flocking.perception_radius
        agent.separation_radius = config.flocking.separation_radius
        
        # Add to environment
        env.add_agent(agent)
    
    # Initialize flocking metrics
    from flocking_metrics import FlockingMetrics
    metrics = FlockingMetrics()
    
    # Run simulation
    from tqdm import tqdm
    for step in tqdm(range(config.max_steps), desc="Flocking Simulation"):
        # Step all agents
        env.step()
        
        # Update flocking metrics
        alive_agents = [
            a for a in env.get_all_agents() 
            if a.alive
        ]
        metrics.update(alive_agents)
        
        # Optional: Log to database
        if step % 10 == 0:
            log_flocking_step(env, metrics, step)
    
    # Finalize and save results
    env.finalize()
    
    # Generate analysis plots
    from flocking_visualization import plot_flocking_metrics
    plot_flocking_metrics(
        metrics,
        save_path=f"results/flocking_{config.simulation_id}.png"
    )
    
    return env, metrics

if __name__ == "__main__":
    env, metrics = run_flocking_simulation()
```

---

## Code Examples

### Complete FlockingAgent Implementation

**File**: `farm/core/flocking_agent.py`

```python
"""
Flocking Agent with Thermodynamic Constraints

Implements Reynolds' flocking rules (alignment, cohesion, separation)
with energy-based movement costs for thermodynamically realistic behavior.
"""

import numpy as np
from farm.core.agent import BaseAgent
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class FlockingAgent(BaseAgent):
    """Agent implementing flocking behavior with energy constraints."""
    
    def __init__(self, *args, velocity=None, **kwargs):
        """Initialize flocking agent.
        
        Args:
            velocity: Initial velocity vector (default: random)
            *args, **kwargs: Passed to BaseAgent
        """
        super().__init__(*args, **kwargs)
        
        # Initialize velocity
        if velocity is not None:
            self.velocity = np.array(velocity, dtype=float)
        else:
            self.velocity = np.random.uniform(-2.0, 2.0, 2).astype(float)
        
        # Flocking parameters (can be overridden from config)
        self.max_speed = getattr(
            self.config.flocking if hasattr(self.config, 'flocking') else self,
            'max_speed',
            2.0
        )
        self.max_force = getattr(
            self.config.flocking if hasattr(self.config, 'flocking') else self,
            'max_force',
            0.5
        )
        
        # Weights
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        self.separation_weight = 1.5
        
        # Radii
        self.perception_radius = 10.0
        self.separation_radius = 5.0
        
        # Energy costs
        self.velocity_cost_coeff = 0.25
        self.base_metabolic_cost = 0.03
    
    def get_neighbors(self, radius):
        """Get nearby agents within radius."""
        nearby = self.spatial_service.get_nearby(
            self.position,
            radius,
            ["agents"]
        )
        
        neighbors = []
        for agent in nearby.get("agents", []):
            if agent.agent_id != self.agent_id and agent.alive:
                neighbors.append(agent)
        
        return neighbors
    
    def compute_alignment(self):
        """Steer towards average velocity of neighbors."""
        neighbors = self.get_neighbors(self.perception_radius)
        
        if not neighbors:
            return np.zeros(2)
        
        # Average velocity
        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        
        # Desired change
        desired = avg_velocity - self.velocity
        
        return desired
    
    def compute_cohesion(self):
        """Steer towards center of mass of neighbors."""
        neighbors = self.get_neighbors(self.perception_radius)
        
        if not neighbors:
            return np.zeros(2)
        
        # Center of mass
        center = np.mean([n.position for n in neighbors], axis=0)
        
        # Desired direction (gentle steering)
        desired = (center - np.array(self.position)) * 0.01
        
        return desired
    
    def compute_separation(self):
        """Steer away from neighbors that are too close."""
        neighbors = self.get_neighbors(self.separation_radius)
        
        if not neighbors:
            return np.zeros(2)
        
        steering = np.zeros(2)
        
        for neighbor in neighbors:
            diff = np.array(self.position) - np.array(neighbor.position)
            distance = np.linalg.norm(diff)
            
            if distance > 0:
                # Weight by inverse distance squared
                steering += diff / (distance ** 2)
        
        return steering
    
    def apply_toroidal_boundary(self, position):
        """Apply wrap-around boundary conditions."""
        return (
            position[0] % self.environment.width,
            position[1] % self.environment.height
        )
    
    def collect_energy_from_resources(self):
        """Collect energy from nearby resource nodes."""
        if not hasattr(self.config, 'flocking') or \
           not self.config.flocking.use_sparse_resources:
            return
        
        nearby = self.spatial_service.get_nearby(
            self.position,
            radius=3.0,
            types=["resources"]
        )
        
        for resource in nearby.get("resources", []):
            if resource.amount > 0:
                # Collect energy
                collected = min(5.0, resource.amount)
                self.resource_level = min(
                    self.resource_level + collected,
                    100.0
                )
                resource.amount -= collected
    
    def act(self):
        """Execute flocking behavior with energy constraints."""
        if not self.alive:
            return
        
        # Compute flocking forces
        align = self.compute_alignment() * self.alignment_weight
        cohere = self.compute_cohesion() * self.cohesion_weight
        separate = self.compute_separation() * self.separation_weight
        
        # Combine forces
        total_force = align + cohere + separate
        
        # Limit force magnitude
        force_mag = np.linalg.norm(total_force)
        if force_mag > self.max_force:
            total_force = total_force / force_mag * self.max_force
        
        # Update velocity
        self.velocity += total_force
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        
        # Update position
        new_position = np.array(self.position) + self.velocity
        new_position = self.apply_toroidal_boundary(new_position)
        
        self.update_position(tuple(new_position))
        
        # Energy consumption (thermodynamic costs)
        classic_mode = getattr(
            self.config.flocking if hasattr(self.config, 'flocking') else self,
            'classic_mode',
            False
        )
        
        if not classic_mode:
            # Quadratic cost: E ∝ v²
            energy_cost = self.velocity_cost_coeff * (speed ** 2)
            
            # Base metabolic cost
            base_cost = self.base_metabolic_cost
            
            # Deduct energy
            self.resource_level -= (energy_cost + base_cost)
            
            # Check for death
            if self.resource_level <= 0:
                logger.info(
                    f"Agent {self.agent_id} died from energy depletion"
                )
                self.terminate()
                return
        
        # Energy collection
        self.collect_energy_from_resources()
        
        # Ambient energy replenishment
        if hasattr(self.config, 'flocking') and \
           not self.config.flocking.use_sparse_resources:
            replenishment = self.config.flocking.ambient_replenishment
            self.resource_level = min(
                self.resource_level + replenishment,
                100.0
            )
```

---

## Configuration

### Complete YAML Configuration

**File**: `farm/config/flocking_config.yaml`

```yaml
# Thermodynamic Flocking Simulation Configuration
simulation:
  name: "thermodynamic_flocking"
  description: "Flocking with energy costs demonstrating ESDP Principle 2"
  max_steps: 1000
  seed: 42

environment:
  width: 100
  height: 100
  position_discretization_method: "floor"
  
  spatial_index:
    enable_batch_updates: true
    region_size: 50.0
    max_batch_size: 100
    enable_spatial_hash_indices: true
    spatial_hash_cell_size: 15.0

population:
  system_agents: 0
  independent_agents: 0
  control_agents: 0

resources:
  initial_resources: 8
  resource_regen_rate: 0.02
  resource_regen_amount: 50
  max_resource_amount: 50

database:
  use_in_memory_db: false
  db_pragma_profile: "balanced"
  log_buffer_size: 1000
  commit_interval_seconds: 30

# Flocking-specific configuration
flocking:
  # Population
  n_agents: 50
  initial_energy_min: 30.0
  initial_energy_max: 100.0
  
  # Movement
  max_speed: 2.0
  max_force: 0.5
  
  # Flocking weights
  alignment_weight: 1.0
  cohesion_weight: 1.0
  separation_weight: 1.5
  
  # Perception
  perception_radius: 10.0
  separation_radius: 5.0
  
  # Energy costs
  velocity_cost_coefficient: 0.25
  base_metabolic_cost: 0.03
  
  # Energy replenishment
  ambient_replenishment: 0.85
  use_sparse_resources: true
  energy_collection_radius: 3.0
  energy_collection_rate: 5.0
  
  # Simulation modes
  classic_mode: false   # Classic Reynolds flocking (no energy)
  aware_mode: false     # Adaptive flocking
  evo_mode: false       # Evolutionary flocking
```

---

## Metrics and Analysis

### Flocking Metrics Module

**File**: `farm/analysis/flocking_metrics.py`

```python
"""Metrics for analyzing flocking behavior and thermodynamic properties."""

import numpy as np
from typing import List
from farm.core.flocking_agent import FlockingAgent


class FlockingMetrics:
    """Track and compute flocking emergence metrics."""
    
    def __init__(self):
        self.time = []
        self.alive_count = []
        self.avg_energy = []
        self.avg_speed = []
        self.alignment = []
        self.cohesion = []
        self.entropy_production = []
        self.phase_synchrony = []
    
    def compute_alignment(self, agents: List[FlockingAgent]) -> float:
        """Velocity coherence: how aligned are agent velocities?"""
        if not agents:
            return 0.0
        
        velocities = np.array([a.velocity for a in agents])
        avg_vel = np.mean(velocities, axis=0)
        avg_speed = np.mean([np.linalg.norm(v) for v in velocities])
        
        if avg_speed < 1e-6:
            return 0.0
        
        alignment = np.linalg.norm(avg_vel) / avg_speed
        return float(alignment)
    
    def compute_cohesion(self, agents: List[FlockingAgent]) -> float:
        """Spatial clustering: how close are agents to center of mass?"""
        if not agents:
            return 0.0
        
        positions = np.array([a.position for a in agents])
        center = np.mean(positions, axis=0)
        avg_dist = np.mean([np.linalg.norm(p - center) for p in positions])
        
        # Normalized cohesion (0-1)
        cohesion = 1.0 / (1.0 + avg_dist / 10.0)
        return float(cohesion)
    
    def compute_entropy_production(self, agents: List[FlockingAgent]) -> float:
        """Entropy production rate: σ ≈ sum(v²) / N"""
        if not agents:
            return 0.0
        
        total_dissipation = sum(
            np.linalg.norm(a.velocity) ** 2 
            for a in agents
        )
        
        entropy_rate = total_dissipation / len(agents)
        return float(entropy_rate)
    
    def compute_phase_synchrony(self, agents: List[FlockingAgent]) -> float:
        """Kuramoto order parameter for velocity coordination."""
        if len(agents) < 2:
            return 0.0
        
        # Velocity angles
        angles = [
            np.arctan2(a.velocity[1], a.velocity[0]) 
            for a in agents
        ]
        
        # Kuramoto order parameter
        complex_phases = np.exp(1j * np.array(angles))
        order_param = np.abs(np.mean(complex_phases))
        
        return float(order_param)
    
    def update(self, agents: List[FlockingAgent], step: int):
        """Compute and record all metrics."""
        self.time.append(step)
        self.alive_count.append(len(agents))
        
        if agents:
            self.avg_energy.append(np.mean([a.resource_level for a in agents]))
            self.avg_speed.append(np.mean([
                np.linalg.norm(a.velocity) for a in agents
            ]))
        else:
            self.avg_energy.append(0.0)
            self.avg_speed.append(0.0)
        
        self.alignment.append(self.compute_alignment(agents))
        self.cohesion.append(self.compute_cohesion(agents))
        self.entropy_production.append(self.compute_entropy_production(agents))
        self.phase_synchrony.append(self.compute_phase_synchrony(agents))
    
    def to_dict(self):
        """Export metrics as dictionary."""
        return {
            'time': self.time,
            'alive_count': self.alive_count,
            'avg_energy': self.avg_energy,
            'avg_speed': self.avg_speed,
            'alignment': self.alignment,
            'cohesion': self.cohesion,
            'entropy_production': self.entropy_production,
            'phase_synchrony': self.phase_synchrony
        }
```

---

## Visualization

### Flocking Visualization Script

**File**: `scripts/visualize_flocking.py`

```python
"""Visualization tools for flocking simulation results."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from farm.database.database import SimulationDatabase


def plot_flocking_metrics(metrics, save_path=None):
    """Create 3x2 grid of flocking metrics plots."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    times = metrics.time
    
    # Population
    axes[0, 0].plot(times, metrics.alive_count, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Alive Agents')
    axes[0, 0].set_title('Population Dynamics')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy
    axes[0, 1].plot(times, metrics.avg_energy, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Average Energy')
    axes[0, 1].set_title('Energy Dynamics')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Alignment
    axes[1, 0].plot(times, metrics.alignment, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Alignment (0-1)')
    axes[1, 0].set_title('Velocity Coherence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cohesion
    axes[1, 1].plot(times, metrics.cohesion, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Cohesion (0-1)')
    axes[1, 1].set_title('Spatial Clustering')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Entropy Production
    axes[2, 0].plot(times, metrics.entropy_production, 'orange', linewidth=2)
    axes[2, 0].set_xlabel('Time')
    axes[2, 0].set_ylabel('Entropy Production (σ)')
    axes[2, 0].set_title('Dissipative Structure Formation')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Peak annotation
    if metrics.entropy_production:
        max_entropy = max(metrics.entropy_production)
        max_idx = metrics.entropy_production.index(max_entropy)
        max_time = times[max_idx]
        axes[2, 0].axvline(
            max_time, color='red', linestyle='--', 
            alpha=0.5, label=f'Peak: {max_entropy:.2f}'
        )
        axes[2, 0].legend()
    
    # Phase Synchrony
    axes[2, 1].plot(times, metrics.phase_synchrony, 'purple', linewidth=2)
    axes[2, 1].set_xlabel('Time')
    axes[2, 1].set_ylabel('Phase Synchrony (0-1)')
    axes[2, 1].set_title('Velocity Coordination (Kuramoto)')
    axes[2, 1].set_ylim([0, 1.1])
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def create_flocking_animation(db_path, simulation_id, save_path=None):
    """Create animation from database-logged simulation data."""
    
    # Load simulation data
    db = SimulationDatabase(db_path)
    
    # Get agent states per step
    states = db.get_agent_states_by_simulation(simulation_id)
    
    # Extract positions and energies per step
    steps = sorted(set(s.step_number for s in states))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Main view
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')
    ax1.set_title('Flocking Behavior with Energy')
    ax1.grid(True, alpha=0.3)
    
    # Metrics view
    ax2.set_xlim(0, len(steps))
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Emergence Metrics')
    ax2.grid(True, alpha=0.3)
    
    scatter = ax1.scatter([], [], c=[], cmap='RdYlGn', vmin=0, vmax=100)
    
    def animate(frame_idx):
        step = steps[frame_idx]
        
        # Get agents at this step
        step_states = [s for s in states if s.step_number == step]
        
        if step_states:
            positions = np.array([
                [s.position_x, s.position_y] 
                for s in step_states
            ])
            energies = np.array([s.resource_level for s in step_states])
            
            scatter.set_offsets(positions)
            scatter.set_array(energies)
        
        return scatter,
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(steps), 
        interval=50, blit=False
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=20)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=20)
        print(f"Saved animation to {save_path}")
    
    return anim
```

---

## Extensions and Variants

### 1. Adaptive Flocking (Aware Mode)

Agents adjust their behavior based on local conditions:

```python
class AwareFlockingAgent(FlockingAgent):
    """Flocking agent with adaptive behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.density_threshold = 5
        self.low_energy_threshold = 30.0
    
    def act(self):
        """Adapt flocking weights based on local conditions."""
        
        # Sense local density
        neighbors = self.get_neighbors(self.perception_radius)
        density = len(neighbors)
        
        # Increase separation in crowded areas
        if density > self.density_threshold:
            self.separation_weight = 1.5 * 1.5
        else:
            self.separation_weight = 1.5
        
        # Slow down when low on energy
        if self.resource_level < self.low_energy_threshold:
            self.max_speed = 1.0
        else:
            self.max_speed = 2.0
        
        # Execute normal flocking
        super().act()
```

### 2. Evolutionary Flocking (Evo Mode)

Agents reproduce and evolve flocking traits:

```python
class EvoFlockingAgent(AwareFlockingAgent):
    """Flocking agent with reproduction and genetic traits."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Genetic traits
        self.traits = {
            'max_speed': self.max_speed,
            'separation_weight': self.separation_weight,
            'energy_efficiency': 1.0
        }
        
        self.reproduction_threshold = 80.0
        self.mutation_rate = 0.1
        self.mutation_strength = 0.2
    
    def can_reproduce(self):
        """Check if agent has enough energy to reproduce."""
        return (
            self.alive and 
            self.resource_level > self.reproduction_threshold
        )
    
    def mutate_traits(self, traits):
        """Apply random mutations to traits."""
        mutated = traits.copy()
        
        for trait in mutated:
            if np.random.random() < self.mutation_rate:
                change = np.random.uniform(
                    -self.mutation_strength, 
                    self.mutation_strength
                )
                mutated[trait] *= (1 + change)
                
                # Clamp to reasonable ranges
                if trait == 'max_speed':
                    mutated[trait] = np.clip(mutated[trait], 1.0, 3.0)
                elif trait == 'separation_weight':
                    mutated[trait] = np.clip(mutated[trait], 1.0, 3.0)
                elif trait == 'energy_efficiency':
                    mutated[trait] = np.clip(mutated[trait], 0.5, 1.5)
        
        return mutated
    
    def reproduce(self):
        """Create offspring with inherited and mutated traits."""
        if not self.can_reproduce():
            return False
        
        # Split energy
        child_energy = self.resource_level / 2
        self.resource_level /= 2
        
        # Inherit and mutate traits
        child_traits = self.mutate_traits(self.traits)
        
        # Create offspring
        offset = np.random.uniform(-5, 5, 2)
        child_pos = (self.position + offset) % [
            self.environment.width,
            self.environment.height
        ]
        
        child = EvoFlockingAgent(
            agent_id=self.environment.get_next_agent_id(),
            position=tuple(child_pos),
            resource_level=child_energy,
            spatial_service=self.spatial_service,
            environment=self.environment,
            config=self.config,
            generation=self.generation + 1
        )
        
        child.traits = child_traits
        child.max_speed = child_traits['max_speed']
        
        self.environment.add_agent(child)
        return True
```

### 3. Configurable Flocking Simulation

Grid search over different feature combinations:

```python
class ConfigurableFlockingAgent(FlockingAgent):
    """Flocking agent with toggleable features."""
    
    def __init__(self, 
                 enable_energy=True,
                 enable_adaptation=False,
                 enable_evolution=False,
                 *args, **kwargs):
        
        self.enable_energy = enable_energy
        self.enable_adaptation = enable_adaptation
        self.enable_evolution = enable_evolution
        
        super().__init__(*args, **kwargs)
        
        if enable_evolution:
            self._init_evolution()
        
        if enable_adaptation:
            self._init_adaptation()
    
    def act(self):
        """Execute flocking with conditional features."""
        
        # Adaptation
        if self.enable_adaptation:
            self._apply_adaptation()
        
        # Base flocking (with or without energy)
        super().act()
        
        # Evolution
        if self.enable_evolution:
            self._check_reproduction()
```

---

## Summary

### Implementation Checklist

- [ ] Create `FlockingAgent` class in `farm/core/flocking_agent.py`
- [ ] Implement flocking steering behaviors (alignment, cohesion, separation)
- [ ] Add energy consumption mechanics
- [ ] Configure energy resource system (ambient or sparse)
- [ ] Create `FlockingMetrics` class for emergence tracking
- [ ] Set up configuration in `farm/config/flocking_config.yaml`
- [ ] Create simulation runner script `scripts/run_flocking_simulation.py`
- [ ] Implement visualization tools
- [ ] Add database logging for positions, velocities, energy
- [ ] Create analysis notebooks

### Key Advantages of AgentFarm

1. **Spatial Indexing**: Efficient neighbor queries via KD-tree/Quadtree
2. **Database Logging**: Automatic tracking of agent states
3. **Metrics System**: Built-in infrastructure for emergence metrics
4. **Configuration**: YAML-based parameter management
5. **Extensibility**: Easy to add variants (awareness, evolution)

### Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Velocity not in BaseAgent | Add as custom attribute in FlockingAgent |
| Energy vs Resources | Repurpose `resource_level` as energy |
| Toroidal boundaries | Implement custom boundary logic |
| Continuous movement | Override `act()` for velocity-based updates |
| Metrics tracking | Extend with custom FlockingMetrics class |

---

## Next Steps

1. **Prototype**: Start with minimal `FlockingAgent` implementation
2. **Test**: Run small simulation (10 agents, 100 steps)
3. **Iterate**: Add energy costs and metrics
4. **Validate**: Compare with original simulation behavior
5. **Extend**: Add awareness and evolution modes
6. **Optimize**: Profile and optimize for larger populations
7. **Visualize**: Create animations and analysis plots

---

**Estimated Implementation Time**: 2-3 days for full implementation including variants and visualization.
