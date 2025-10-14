# Scenario Implementation Guide for AgentFarm

## Quick Start (5 minutes)

**Just want it working?**

```bash
# Run the thermodynamic flocking simulation
python examples/flocking_simulation_starter.py
```

That's it! You now have a working flocking simulation with energy constraints.

---

## Overview

This guide shows how to implement simulation scenarios in AgentFarm. Two approaches:

1. **Standalone**: Fast, self-contained (use for single scenarios)
2. **Modular**: Reusable, swappable (use for multiple scenarios)

**Decision**: Building multiple scenarios? → Modular. Just one? → Standalone.

---

## Part 1: Standalone Approach

### When to Use
- ✅ Single scenario
- ✅ Quick prototype
- ✅ Learning AgentFarm
- ✅ Time-constrained

### Implementation (4 hours)

The starter code (`examples/flocking_simulation_starter.py`) shows the complete pattern:

```python
# 1. Create custom agent class
class FlockingAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity = np.random.uniform(-2.0, 2.0, 2)
        # Flocking parameters
        self.max_speed = 2.0
        self.perception_radius = 10.0
    
    def act(self):
        # Compute flocking forces (alignment, cohesion, separation)
        neighbors = self.get_neighbors(self.perception_radius)
        forces = self.compute_flocking_forces(neighbors)
        
        # Update velocity and position
        self.velocity += forces
        self.update_position(self.position + self.velocity)
        
        # Energy costs (thermodynamic)
        speed = np.linalg.norm(self.velocity)
        self.resource_level -= 0.25 * speed**2 + 0.03
        
        if self.resource_level <= 0:
            self.terminate()

# 2. Create metrics tracker
class FlockingMetrics:
    def update(self, environment, step):
        alive = [a for a in environment.get_all_agents() if a.alive]
        self.alignment.append(self.compute_alignment(alive))
        self.cohesion.append(self.compute_cohesion(alive))

# 3. Run simulation
def run_simulation(n_agents=50, n_steps=1000):
    env = Environment(width=100, height=100, ...)
    
    # Create agents
    for i in range(n_agents):
        agent = FlockingAgent(...)
        env.add_agent(agent)
    
    # Run
    metrics = FlockingMetrics()
    for step in range(n_steps):
        env.step()
        metrics.update(env, step)
    
    return env, metrics
```

**That's the complete pattern!** See `examples/flocking_simulation_starter.py` for full working code.

---

## Part 2: Modular Approach

### When to Use
- ✅ Multiple scenarios
- ✅ Need to compare scenarios
- ✅ Production system
- ✅ Research with variants

### Architecture

**Core Concept**: Define standard interfaces, scenarios implement them.

```python
# Standard interface all scenarios implement
class Scenario(Protocol):
    name: str
    
    def setup(environment, config) -> List[BaseAgent]
    def step_hook(environment, step) -> None
    def get_metrics() -> ScenarioMetrics
    def get_visualizer() -> ScenarioVisualizer

# Register scenarios
@register_scenario("thermodynamic_flocking")
class FlockingScenario(BaseScenario):
    name = "thermodynamic_flocking"
    
    def create_agents(self, environment, config):
        # Create FlockingAgent instances
        return agents
    
    def get_metrics(self):
        return FlockingMetrics()

# Run any scenario
python scripts/run_scenario.py --config flocking.yaml
```

### Implementation (8 hours initial, 4-6 hours per scenario)

#### Phase 1: Infrastructure (4 hours)

**Create**: `farm/core/scenarios/protocol.py`

```python
from typing import Protocol, List, Dict, Any
from farm.core.agent import BaseAgent
from farm.core.environment import Environment

class ScenarioMetrics(Protocol):
    def update(self, environment: Environment, step: int) -> None: ...
    def get_current_metrics(self) -> Dict[str, float]: ...

class ScenarioVisualizer(Protocol):
    def plot_metrics(self, metrics, save_path=None) -> Any: ...

class Scenario(Protocol):
    name: str
    description: str
    
    def setup(self, environment, config) -> List[BaseAgent]: ...
    def step_hook(self, environment, step) -> None: ...
    def get_metrics(self) -> ScenarioMetrics: ...
    def get_visualizer(self) -> ScenarioVisualizer: ...
```

**Create**: `farm/core/scenarios/registry.py`

```python
class ScenarioRegistry:
    _scenarios = {}
    
    @classmethod
    def register(cls, name, scenario_class):
        cls._scenarios[name] = scenario_class
    
    @classmethod
    def get(cls, name):
        return cls._scenarios[name]

def register_scenario(name):
    def decorator(cls):
        ScenarioRegistry.register(name, cls)
        return cls
    return decorator
```

**Create**: `farm/core/scenarios/base.py`

```python
class BaseScenario(ABC):
    name: str = "base_scenario"
    
    def setup(self, environment, config):
        agents = self.create_agents(environment, config)
        for agent in agents:
            environment.add_agent(agent)
        return agents
    
    @abstractmethod
    def create_agents(self, environment, config):
        """Subclasses implement this."""
        ...
```

**Create**: `farm/core/scenarios/runner.py`

```python
class ScenarioRunner:
    def __init__(self, scenario, environment, config):
        self.scenario = scenario
        self.environment = environment
        self.metrics = scenario.get_metrics()
    
    def run(self, steps):
        for step in range(steps):
            self.environment.step()
            self.scenario.step_hook(self.environment, step)
            self.metrics.update(self.environment, step)
        
        return {'metrics': self.metrics.to_dict()}
```

#### Phase 2: Flocking Scenario (4 hours)

**Create**: `farm/scenarios/flocking_scenario.py`

```python
@register_scenario("thermodynamic_flocking")
class FlockingScenario(BaseScenario):
    name = "thermodynamic_flocking"
    description = "Flocking with energy costs"
    
    def create_agents(self, environment, config):
        agents = []
        for i in range(config.scenario.flocking.n_agents):
            agent = FlockingAgent(
                agent_id=environment.get_next_agent_id(),
                position=(random.uniform(0, 100), random.uniform(0, 100)),
                resource_level=random.uniform(30, 100),
                spatial_service=environment.spatial_service,
                environment=environment,
                config=config
            )
            agents.append(agent)
        return agents
    
    def get_metrics(self):
        return FlockingMetrics()
    
    def get_visualizer(self):
        return FlockingVisualizer()
```

**Create**: `farm/config/scenarios/flocking.yaml`

```yaml
scenario:
  type: "thermodynamic_flocking"
  flocking:
    n_agents: 50
    max_speed: 2.0
    perception_radius: 10.0

environment:
  width: 100
  height: 100

max_steps: 1000
```

**Create**: `scripts/run_scenario.py`

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    results = ScenarioRunner.run_from_config(config)
```

#### Usage

```bash
# Run flocking
python scripts/run_scenario.py --config flocking.yaml

# Run different scenario (same command!)
python scripts/run_scenario.py --config predator_prey.yaml
```

---

## Implementation Checklist

### Standalone
- [ ] Copy `examples/flocking_simulation_starter.py`
- [ ] Modify parameters as needed
- [ ] Run and verify
- [ ] Customize agent behavior
- [ ] Add metrics as needed

### Modular
- [ ] Create `farm/core/scenarios/` directory
- [ ] Implement protocol.py, registry.py, base.py, runner.py
- [ ] Create scenario class
- [ ] Create metrics and visualizer classes
- [ ] Create config file
- [ ] Create universal runner script
- [ ] Test scenario switching

---

## Key Flocking Components

### Flocking Agent

```python
class FlockingAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity = np.random.uniform(-2.0, 2.0, 2)
    
    def compute_alignment(self, neighbors):
        """Steer toward average velocity."""
        if not neighbors:
            return np.zeros(2)
        avg_vel = np.mean([n.velocity for n in neighbors], axis=0)
        return avg_vel - self.velocity
    
    def compute_cohesion(self, neighbors):
        """Steer toward center of mass."""
        if not neighbors:
            return np.zeros(2)
        center = np.mean([n.position for n in neighbors], axis=0)
        return (center - np.array(self.position)) * 0.01
    
    def compute_separation(self, neighbors):
        """Avoid crowding."""
        steering = np.zeros(2)
        for n in neighbors:
            diff = np.array(self.position) - np.array(n.position)
            dist = np.linalg.norm(diff)
            if dist > 0:
                steering += diff / (dist ** 2)
        return steering
```

### Flocking Metrics

```python
class FlockingMetrics:
    def compute_alignment(self, agents):
        """Velocity coherence (0-1)."""
        velocities = np.array([a.velocity for a in agents])
        avg_vel = np.mean(velocities, axis=0)
        avg_speed = np.mean([np.linalg.norm(v) for v in velocities])
        return np.linalg.norm(avg_vel) / avg_speed if avg_speed > 0 else 0
    
    def compute_cohesion(self, agents):
        """Spatial clustering (0-1)."""
        positions = np.array([a.position for a in agents])
        center = np.mean(positions, axis=0)
        avg_dist = np.mean([np.linalg.norm(p - center) for p in positions])
        return 1.0 / (1.0 + avg_dist / 10.0)
    
    def compute_entropy_production(self, agents):
        """Energy dissipation rate."""
        return sum(np.linalg.norm(a.velocity)**2 for a in agents) / len(agents)
    
    def compute_phase_synchrony(self, agents):
        """Kuramoto order parameter."""
        angles = [np.arctan2(a.velocity[1], a.velocity[0]) for a in agents]
        return np.abs(np.mean(np.exp(1j * np.array(angles))))
```

---

## Configuration

### Standalone Parameters

```python
# In your script
n_agents = 50
max_speed = 2.0
perception_radius = 10.0
energy_cost_coefficient = 0.25
base_metabolic_cost = 0.03
```

### Modular YAML

```yaml
scenario:
  type: "thermodynamic_flocking"
  flocking:
    n_agents: 50
    max_speed: 2.0
    max_force: 0.5
    perception_radius: 10.0
    separation_radius: 5.0
    alignment_weight: 1.0
    cohesion_weight: 1.0
    separation_weight: 1.5
    velocity_cost: 0.25
    base_cost: 0.03

environment:
  width: 100
  height: 100
  spatial_index:
    enable_spatial_hash_indices: true

max_steps: 1000
seed: 42
```

---

## Variants

### Adaptive Flocking

```python
class AwareFlockingAgent(FlockingAgent):
    def act(self):
        # Adapt based on local conditions
        neighbors = self.get_neighbors(self.perception_radius)
        if len(neighbors) > 5:
            self.separation_weight = 2.5  # More separation in crowds
        
        if self.resource_level < 30:
            self.max_speed = 1.0  # Slow down to conserve energy
        
        super().act()
```

### Evolutionary Flocking

```python
class EvoFlockingAgent(AwareFlockingAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traits = {
            'max_speed': self.max_speed,
            'energy_efficiency': 1.0
        }
    
    def reproduce(self):
        if self.resource_level > 80:
            child_traits = self.mutate_traits(self.traits)
            child = EvoFlockingAgent(..., traits=child_traits)
            self.environment.add_agent(child)
```

### Configuration Switching

```yaml
# Classic mode (no energy costs)
scenario:
  flocking:
    mode: "classic"

# Adaptive mode
scenario:
  flocking:
    mode: "adaptive"

# Evolutionary mode
scenario:
  flocking:
    mode: "evolutionary"
```

---

## Time Estimates

| Task | Standalone | Modular |
|------|-----------|---------|
| Infrastructure | - | 4 hours |
| First scenario | 4 hours | 4 hours |
| Second scenario | 25 hours | 6 hours |
| Third scenario | 25 hours | 6 hours |
| **Total (3)** | **54 hours** | **20 hours** |

**Break-even point**: 2 scenarios

---

## File Structure

### Standalone
```
examples/
└── flocking_simulation_starter.py  (complete implementation)
```

### Modular
```
farm/
├── core/
│   └── scenarios/
│       ├── protocol.py
│       ├── registry.py
│       ├── base.py
│       └── runner.py
├── scenarios/
│   └── flocking_scenario.py
├── analysis/
│   └── flocking_metrics.py
└── visualization/
    └── flocking_viz.py

config/scenarios/
└── flocking.yaml

scripts/
└── run_scenario.py
```

---

## Migration: Standalone → Modular

If you start standalone and want to migrate:

**Step 1: Wrap in Scenario**

```python
@register_scenario("thermodynamic_flocking")
class FlockingScenario(BaseScenario):
    def create_agents(self, environment, config):
        # Call your existing code
        return create_flocking_agents_from_starter(environment, config)
```

**Step 2: Extract Components**

- Move metrics to `FlockingMetrics` class
- Move visualizer to `FlockingVisualizer` class
- Create config YAML

**Time**: ~2 hours

---

## Debugging

### Common Issues

**Agents not flocking**
- Check `perception_radius` (try 10-15)
- Verify `get_neighbors()` returns agents
- Ensure forces are being applied

**All agents dying**
- Reduce energy costs (try 0.1 instead of 0.25)
- Increase energy replenishment
- Check initial energy distribution

**Slow performance**
- Enable spatial hash grid
- Reduce perception radius
- Decrease agent count for testing

---

## Next Steps

### Standalone Path
1. Run `python examples/flocking_simulation_starter.py`
2. Modify parameters to experiment
3. Add custom behaviors as needed
4. Extend metrics if desired

### Modular Path
1. Create infrastructure (Phase 1)
2. Implement flocking scenario (Phase 2)
3. Test with universal runner
4. Add second scenario to prove modularity

### Resources
- Working code: `examples/flocking_simulation_starter.py`
- AgentFarm docs: `docs/generic_simulation_scenario_howto.md`
- Agent guide: `docs/design/Agent.md`

---

## Summary

**Standalone**: Fast, self-contained, ~4 hours  
**Modular**: Reusable, swappable, ~8 hours initial, then ~6 hours per scenario

**Choose standalone if**: Single scenario, prototyping, learning  
**Choose modular if**: Multiple scenarios, production, research

**Start here**: Run `examples/flocking_simulation_starter.py` to see it working!
