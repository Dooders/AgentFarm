# Updated Implementation Plan - Modular Scenario Architecture

## Overview

This is the **revised implementation plan** for the thermodynamic flocking simulation using the new modular scenario architecture. This approach makes scenarios pluggable, reusable, and easy to swap.

---

## What Changed?

### Original Plan
- Standalone flocking implementation
- Custom runner scripts
- Scenario-specific everything
- Hard to reuse or extend

### New Plan
- **Standard scenario interface**
- **Pluggable components** (agents, metrics, visualizers)
- **Universal runner** (works for all scenarios)
- **Configuration-driven** scenario selection
- **Registry pattern** for automatic discovery

---

## Implementation Phases (Revised)

### Phase 0: Modular Infrastructure (NEW - 3-4 hours)

**Goal**: Set up the modular scenario system

#### Tasks

**0.1 Create Scenario Protocol** (1 hour)

Create `farm/core/scenarios/protocol.py`:

```python
from typing import Protocol, List, Dict, Any
from farm.core.agent import BaseAgent
from farm.core.environment import Environment

class ScenarioMetrics(Protocol):
    def update(self, environment: Environment, step: int) -> None: ...
    def get_current_metrics(self) -> Dict[str, float]: ...
    def get_history(self) -> Dict[str, List[float]]: ...
    def to_dict(self) -> Dict[str, Any]: ...

class ScenarioVisualizer(Protocol):
    def plot_metrics(self, metrics, save_path=None) -> Any: ...
    def create_animation(self, environment, save_path=None) -> Any: ...

class Scenario(Protocol):
    name: str
    description: str
    version: str
    
    def setup(self, environment, config) -> List[BaseAgent]: ...
    def step_hook(self, environment, step) -> None: ...
    def get_metrics(self) -> ScenarioMetrics: ...
    def get_visualizer(self) -> ScenarioVisualizer: ...
    def validate_config(self, config) -> bool: ...
```

**0.2 Create Scenario Registry** (1 hour)

Create `farm/core/scenarios/registry.py`:

```python
class ScenarioRegistry:
    """Central registry for scenarios."""
    _scenarios = {}
    
    @classmethod
    def register(cls, name, scenario_class):
        cls._scenarios[name] = scenario_class
    
    @classmethod
    def get(cls, name):
        return cls._scenarios[name]
    
    @classmethod
    def list_scenarios(cls):
        return list(cls._scenarios.keys())

# Decorator for easy registration
def register_scenario(name):
    def decorator(cls):
        ScenarioRegistry.register(name, cls)
        return cls
    return decorator
```

**0.3 Create Base Scenario Class** (1 hour)

Create `farm/core/scenarios/base.py`:

```python
from abc import ABC, abstractmethod

class BaseScenario(ABC):
    """Base class for scenarios with common functionality."""
    
    name: str = "base_scenario"
    description: str = ""
    version: str = "1.0.0"
    
    def setup(self, environment, config):
        self.validate_config(config)
        self._environment = environment
        self._config = config
        
        agents = self.create_agents(environment, config)
        
        for agent in agents:
            environment.add_agent(agent)
        
        return agents
    
    @abstractmethod
    def create_agents(self, environment, config):
        """Subclasses must implement."""
        ...
    
    # ... other common methods
```

**0.4 Create Scenario Runner** (30 min)

Create `farm/core/scenarios/runner.py`:

```python
class ScenarioRunner:
    """Universal runner for any scenario."""
    
    def __init__(self, scenario, environment, config):
        self.scenario = scenario
        self.environment = environment
        self.config = config
        self.metrics = scenario.get_metrics()
    
    def run(self, steps, progress_bar=True):
        for step in range(steps):
            self.environment.step()
            self.scenario.step_hook(self.environment, step)
            self.metrics.update(self.environment, step)
        
        return {
            'scenario': self.scenario.name,
            'metrics': self.metrics.to_dict(),
            'environment': self.environment
        }
```

**Deliverable**: Modular scenario infrastructure ready

---

### Phase 1: Flocking Scenario (4-6 hours)

**Goal**: Implement flocking using modular architecture

#### Tasks

**1.1 Create FlockingAgent** (2 hours)

Create `farm/core/flocking_agent.py` (same as before):

```python
class FlockingAgent(BaseAgent):
    """Agent with flocking behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity = np.random.uniform(-2.0, 2.0, 2)
        # ... flocking parameters
    
    def act(self):
        # Compute flocking forces
        # Update velocity and position
        # Apply energy costs
        ...
```

**1.2 Create FlockingMetrics** (1 hour)

Create `farm/analysis/flocking_metrics.py` implementing `ScenarioMetrics`:

```python
class FlockingMetrics:
    """Implements ScenarioMetrics protocol."""
    
    def update(self, environment, step):
        # Update all metrics
        ...
    
    def get_current_metrics(self):
        return {'alignment': ..., 'cohesion': ...}
    
    def get_history(self):
        return {'time': ..., 'alignment': ...}
    
    def to_dict(self):
        return self.get_history()
```

**1.3 Create FlockingVisualizer** (1 hour)

Create `farm/visualization/flocking_viz.py` implementing `ScenarioVisualizer`:

```python
class FlockingVisualizer:
    """Implements ScenarioVisualizer protocol."""
    
    def plot_metrics(self, metrics, save_path=None):
        # Create 3x2 subplot
        ...
    
    def create_animation(self, environment, save_path=None):
        # Create animation
        ...
```

**1.4 Create FlockingScenario** (1 hour)

Create `farm/scenarios/flocking_scenario.py`:

```python
@register_scenario("thermodynamic_flocking")
class FlockingScenario(BaseScenario):
    name = "thermodynamic_flocking"
    description = "Flocking with energy costs"
    version = "1.0.0"
    
    def create_agents(self, environment, config):
        agents = []
        for i in range(config.scenario.flocking.n_agents):
            agent = FlockingAgent(...)
            agents.append(agent)
        return agents
    
    def get_metrics(self):
        return FlockingMetrics()
    
    def get_visualizer(self):
        return FlockingVisualizer()
    
    def validate_config(self, config):
        # Validate flocking config
        ...
```

**1.5 Create Configuration** (30 min)

Create `farm/config/scenarios/flocking.yaml`:

```yaml
scenario:
  type: "thermodynamic_flocking"
  flocking:
    n_agents: 50
    max_speed: 2.0
    # ... all parameters

environment:
  width: 100
  height: 100

max_steps: 1000
```

**Deliverable**: Working flocking scenario using modular system

---

### Phase 2: Universal Runner (1-2 hours)

**Goal**: Create universal scenario runner script

#### Tasks

**2.1 Create Runner Script** (1 hour)

Create `scripts/run_scenario.py`:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run scenario (works for ANY scenario!)
    results = ScenarioRunner.run_from_config(
        config,
        steps=args.steps
    )
    
    # Visualize
    if args.visualize:
        scenario.get_visualizer().plot_metrics(results['metrics'])
```

**2.2 Test with Multiple Configs** (30 min)

Create different config files and verify same runner works:

```bash
python scripts/run_scenario.py --config flocking.yaml --visualize
python scripts/run_scenario.py --config flocking_classic.yaml --visualize
python scripts/run_scenario.py --config flocking_adaptive.yaml --visualize
```

**Deliverable**: Universal runner that works for all scenarios

---

### Phase 3: Flocking Variants (3-4 hours)

**Goal**: Add adaptive and evolutionary modes

#### Tasks

**3.1 Create AwareFlockingAgent** (1 hour)

```python
class AwareFlockingAgent(FlockingAgent):
    """Flocking agent with adaptation."""
    
    def act(self):
        # Adapt parameters based on local conditions
        self.adapt_behavior()
        super().act()
    
    def adapt_behavior(self):
        neighbors = self.get_neighbors(self.perception_radius)
        if len(neighbors) > 5:
            self.separation_weight = 2.5  # More separation
```

**3.2 Create EvoFlockingAgent** (2 hours)

```python
class EvoFlockingAgent(AwareFlockingAgent):
    """Flocking agent with evolution."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traits = {'max_speed': 2.0, ...}
    
    def reproduce(self):
        # Create offspring with mutated traits
        ...
```

**3.3 Update FlockingScenario** (30 min)

```python
class FlockingScenario(BaseScenario):
    def create_agents(self, environment, config):
        # Select agent class based on mode
        mode = config.scenario.flocking.mode
        if mode == 'evolutionary':
            agent_class = EvoFlockingAgent
        elif mode == 'adaptive':
            agent_class = AwareFlockingAgent
        else:
            agent_class = FlockingAgent
        
        # Create agents
        ...
```

**3.4 Create Mode Configs** (30 min)

```yaml
# flocking_adaptive.yaml
scenario:
  type: "thermodynamic_flocking"
  flocking:
    mode: "adaptive"
    # ...

# flocking_evo.yaml  
scenario:
  type: "thermodynamic_flocking"
  flocking:
    mode: "evolutionary"
    # ...
```

**Deliverable**: Three flocking modes accessible via config

---

### Phase 4: Additional Scenarios (Optional)

**Goal**: Demonstrate modularity with another scenario

#### Create Predator-Prey Scenario (3-4 hours)

```python
@register_scenario("predator_prey")
class PredatorPreyScenario(BaseScenario):
    name = "predator_prey"
    
    def create_agents(self, environment, config):
        # Create prey agents
        # Create predator agents
        ...
    
    def get_metrics(self):
        return PredatorPreyMetrics()
    
    def get_visualizer(self):
        return PredatorPreyVisualizer()
```

**Test swapping**:

```bash
# Flocking
python scripts/run_scenario.py --config flocking.yaml

# Predator-Prey (same runner!)
python scripts/run_scenario.py --config predator_prey.yaml
```

**Deliverable**: Multiple scenarios using same infrastructure

---

## File Structure

```
farm/
├── core/
│   ├── scenarios/              # NEW
│   │   ├── __init__.py
│   │   ├── protocol.py        # Scenario protocols
│   │   ├── registry.py        # Scenario registry
│   │   ├── base.py           # Base scenario class
│   │   ├── factory.py        # Scenario factory
│   │   └── runner.py         # Scenario runner
│   │
│   ├── flocking_agent.py     # Flocking agents
│   └── ...
│
├── scenarios/                  # NEW - Scenario implementations
│   ├── __init__.py
│   ├── flocking_scenario.py
│   ├── predator_prey_scenario.py
│   └── ...
│
├── analysis/
│   ├── flocking_metrics.py   # Scenario-specific metrics
│   └── ...
│
├── visualization/              # NEW - Scenario visualizers
│   ├── flocking_viz.py
│   └── ...
│
├── config/
│   └── scenarios/             # Scenario configs
│       ├── flocking.yaml
│       ├── flocking_adaptive.yaml
│       ├── flocking_evo.yaml
│       ├── predator_prey.yaml
│       └── ...
│
└── scripts/
    └── run_scenario.py        # Universal runner
```

---

## Timeline Comparison

### Original Plan
- Phase 1-9: ~25-37 hours
- Each scenario needs custom implementation
- Limited reusability

### New Modular Plan
- Phase 0: 3-4 hours (one-time infrastructure)
- Phase 1: 4-6 hours (first scenario)
- Phase 2: 1-2 hours (universal runner)
- Phase 3: 3-4 hours (variants)
- **Total: ~11-16 hours for flocking**

**But then**:
- Each additional scenario: ~4-6 hours (vs ~25-37 hours)
- All scenarios share infrastructure
- Consistent interface and testing

**Long-term savings**: Massive!

---

## Benefits of Modular Approach

### 1. Easy Scenario Swapping

```bash
# Before: Different scripts for each
python run_flocking.py
python run_predator_prey.py

# After: Same script, different config
python scripts/run_scenario.py --config flocking.yaml
python scripts/run_scenario.py --config predator_prey.yaml
```

### 2. Reusable Components

```python
# Metrics can be reused across scenarios
class PopulationMetrics:
    # Used by multiple scenarios
    ...

# Visualizers can be composed
class CompositeVisualizer:
    def __init__(self, *visualizers):
        self.visualizers = visualizers
```

### 3. Testing

```python
# Test any scenario the same way
def test_scenario(scenario_class):
    scenario = scenario_class()
    env = create_test_environment()
    scenario.setup(env, test_config)
    # ... test
```

### 4. Discovery

```python
# Automatically find all scenarios
ScenarioRegistry.discover_scenarios("farm/scenarios/")

# List available
print(ScenarioRegistry.list_scenarios())
# Output: ['thermodynamic_flocking', 'predator_prey', ...]
```

### 5. Comparison

```python
# Easy to compare scenarios
for scenario_name in ['flocking', 'predator_prey']:
    config = load_config(f"{scenario_name}.yaml")
    results = ScenarioRunner.run_from_config(config)
    compare_metrics(results['metrics'])
```

---

## Migration from Original Plan

If you've already started with the original approach:

### Step 1: Create Infrastructure (Phase 0)
- Add scenario protocols
- Add registry
- Add base scenario class
- Add runner

### Step 2: Wrap Existing Code

```python
# If you have standalone flocking code:
@register_scenario("thermodynamic_flocking")
class FlockingScenario(BaseScenario):
    def create_agents(self, environment, config):
        # Call your existing code
        return create_flocking_agents(environment, config)
    
    def get_metrics(self):
        # Wrap existing metrics
        return FlockingMetrics()
```

### Step 3: Update Configs

```yaml
# Add scenario section
scenario:
  type: "thermodynamic_flocking"
  
# Move scenario-specific settings under scenario
scenario:
  flocking:
    n_agents: 50
    # ...
```

### Step 4: Use Universal Runner

```bash
# Instead of custom script
python scripts/run_scenario.py --config flocking.yaml
```

---

## Checklist

### Infrastructure
- [ ] Create `farm/core/scenarios/` directory
- [ ] Implement `protocol.py` (Scenario protocols)
- [ ] Implement `registry.py` (Scenario registry)
- [ ] Implement `base.py` (Base scenario class)
- [ ] Implement `factory.py` (Scenario factory)
- [ ] Implement `runner.py` (Scenario runner)

### Flocking Scenario
- [ ] Create `farm/core/flocking_agent.py` (FlockingAgent)
- [ ] Create `farm/scenarios/flocking_scenario.py` (FlockingScenario)
- [ ] Create `farm/analysis/flocking_metrics.py` (FlockingMetrics)
- [ ] Create `farm/visualization/flocking_viz.py` (FlockingVisualizer)
- [ ] Create `farm/config/scenarios/flocking.yaml` (Config)

### Runner
- [ ] Create `scripts/run_scenario.py` (Universal runner)
- [ ] Test with flocking scenario
- [ ] Create variant configs (adaptive, evo)

### Testing
- [ ] Unit tests for scenario infrastructure
- [ ] Integration test for flocking scenario
- [ ] Test scenario swapping

### Documentation
- [ ] Document scenario protocol
- [ ] Document how to create scenarios
- [ ] Create example scenarios
- [ ] Update user guide

---

## Next Steps

1. **Start with Phase 0**: Build the modular infrastructure
2. **Implement Phase 1**: Create flocking scenario
3. **Test thoroughly**: Ensure scenario works with universal runner
4. **Add variants**: Implement adaptive and evolutionary modes
5. **Create second scenario**: Prove modularity with different scenario
6. **Optimize and polish**: Clean up code and documentation

---

## Questions & Answers

**Q: Do I need to rewrite everything?**  
A: No! The modular approach wraps existing code. You can migrate incrementally.

**Q: Is this more work upfront?**  
A: Yes, ~4 hours for infrastructure. But you save 20+ hours on each subsequent scenario.

**Q: Can I still use the standalone approach?**  
A: Yes! But you'll lose the benefits of modularity and reusability.

**Q: What if I only want one scenario?**  
A: The infrastructure is still valuable for:
- Clean separation of concerns
- Easy testing
- Configuration-driven behavior
- Variant support (adaptive, evo modes)

**Q: How do I add a new scenario?**  
A: 
1. Create scenario class extending `BaseScenario`
2. Implement required methods
3. Add `@register_scenario("name")` decorator
4. Create config file
5. Run with `scripts/run_scenario.py --config your_scenario.yaml`

That's it!

---

## Summary

The modular scenario architecture provides:

✅ **Easy swapping** between scenarios  
✅ **Reusable components** (metrics, visualizers)  
✅ **Consistent interface** for all scenarios  
✅ **Configuration-driven** behavior  
✅ **Automatic discovery** of scenarios  
✅ **Universal runner** script  
✅ **Clean testing** and maintenance  

**Initial investment**: ~4 hours  
**Payoff**: Massive time savings on subsequent scenarios  
**Best for**: Anyone planning multiple scenarios or variants
