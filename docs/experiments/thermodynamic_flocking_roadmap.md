# Thermodynamic Flocking Implementation Roadmap

## Overview

This roadmap provides a step-by-step plan to implement the thermodynamic flocking simulation in AgentFarm, from minimal prototype to full-featured implementation with all variants.

**Total Estimated Time**: 2-3 days  
**Difficulty**: Moderate  
**Prerequisites**: Basic Python, NumPy, understanding of AgentFarm structure

---

## Phase 1: Minimal Prototype (2-4 hours)

**Goal**: Get basic flocking working without advanced features

### Tasks

#### 1.1 Copy Starter Template ✅ (5 min)
- [x] Copy `examples/flocking_simulation_starter.py` to your workspace
- [x] Verify imports work
- [x] Run to check for errors

```bash
python examples/flocking_simulation_starter.py
```

**Expected Output**: Simulation runs, creates DB and plot

#### 1.2 Understand Core Components (30 min)
- [ ] Read through `FlockingAgent` class
- [ ] Understand flocking rules (alignment, cohesion, separation)
- [ ] Review energy consumption mechanics
- [ ] Examine metrics tracking

**Action Items**:
- Add print statements to understand flow
- Modify parameters and observe effects
- Plot different metrics

#### 1.3 Test and Validate (1 hour)
- [ ] Run with different agent counts (10, 25, 50, 100)
- [ ] Vary initial energy ranges
- [ ] Test boundary wrapping
- [ ] Verify agents die when energy depleted

**Validation Checklist**:
- ✓ Agents move and flock together
- ✓ Energy decreases over time
- ✓ Some agents die from energy depletion
- ✓ Metrics show emergent patterns

#### 1.4 Debug Common Issues (1 hour)
- [ ] All agents dying too fast → Reduce energy costs
- [ ] No flocking visible → Check neighbor detection
- [ ] Agents stuck → Verify velocity updates
- [ ] Performance slow → Enable spatial hash

**Deliverable**: Working minimal flocking simulation

---

## Phase 2: Energy System Enhancement (4-6 hours)

**Goal**: Implement sophisticated energy mechanics

### Tasks

#### 2.1 Ambient Energy Replenishment (1 hour)

Add gradual energy recovery:

```python
def act(self):
    # ... existing flocking code ...
    
    # Ambient energy replenishment
    replenishment = 0.85  # Per step
    self.resource_level = min(
        self.resource_level + replenishment,
        100.0  # Max energy cap
    )
```

**Test**: Verify agents survive longer

#### 2.2 Sparse Energy Resources (2 hours)

Implement energy collection from resource nodes:

```python
def collect_energy_from_resources(self):
    """Collect energy from nearby resources."""
    nearby = self.spatial_service.get_nearby(
        self.position, radius=3.0, types=["resources"]
    )
    
    for resource in nearby.get("resources", []):
        if resource.amount > 0:
            collected = min(5.0, resource.amount)
            self.resource_level = min(
                self.resource_level + collected, 100.0
            )
            resource.amount -= collected

# Call in act()
def act(self):
    # ... flocking and movement ...
    self.collect_energy_from_resources()
```

**Test**: Agents should forage for energy

#### 2.3 Energy Cost Tuning (1 hour)

Experiment with different cost models:

```python
# Quadratic cost (realistic)
energy_cost = 0.25 * speed ** 2

# Linear cost (simpler)
energy_cost = 0.5 * speed

# Cubic cost (severe)
energy_cost = 0.1 * speed ** 3
```

**Test**: Find balance between realism and survival

#### 2.4 Classic Mode Toggle (30 min)

Add flag to disable energy costs:

```python
def __init__(self, *args, classic_mode=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.classic_mode = classic_mode
    # ...

def act(self):
    # ... flocking ...
    
    if not self.classic_mode:
        # Apply energy costs
        self.resource_level -= energy_cost + base_cost
```

**Test**: Compare classic vs thermodynamic modes

**Deliverable**: Flexible energy system with multiple modes

---

## Phase 3: Advanced Metrics (3-4 hours)

**Goal**: Track thermodynamic and coordination metrics

### Tasks

#### 3.1 Entropy Production (1 hour)

Measure dissipation rate:

```python
def compute_entropy_production(self, agents):
    """σ = sum(v²) / N"""
    if not agents:
        return 0.0
    
    total_dissipation = sum(
        np.linalg.norm(a.velocity) ** 2 
        for a in agents
    )
    
    return total_dissipation / len(agents)
```

**Test**: Entropy should peak during active flocking

#### 3.2 Phase Synchrony (1 hour)

Implement Kuramoto order parameter:

```python
def compute_phase_synchrony(self, agents):
    """R = |1/N * Σ exp(iθ)|"""
    if len(agents) < 2:
        return 0.0
    
    angles = [
        np.arctan2(a.velocity[1], a.velocity[0]) 
        for a in agents
    ]
    
    complex_phases = np.exp(1j * np.array(angles))
    order_param = np.abs(np.mean(complex_phases))
    
    return float(order_param)
```

**Test**: Phase synchrony should increase as flock forms

#### 3.3 Enhanced Visualization (1 hour)

Add new metric plots:

```python
# Add to plot_metrics()
axes[2, 0].plot(metrics.time, metrics.entropy_production)
axes[2, 0].set_title('Entropy Production Rate')

axes[2, 1].plot(metrics.time, metrics.phase_synchrony)
axes[2, 1].set_title('Phase Synchrony (Kuramoto)')
```

**Test**: All 6 metrics should display properly

#### 3.4 Database Logging (1 hour)

Log velocity and custom metrics to database:

```python
# After each step in main loop
if step % 10 == 0:
    for agent in alive_agents:
        env.db.logger.log_custom_metric(
            step_number=step,
            agent_id=agent.agent_id,
            metric_name="velocity_x",
            value=float(agent.velocity[0])
        )
```

**Test**: Query database to verify logging

**Deliverable**: Comprehensive metrics tracking and visualization

---

## Phase 4: Configuration System (2-3 hours)

**Goal**: Make simulation fully configurable via YAML

### Tasks

#### 4.1 Create Configuration File (30 min)

Create `farm/config/flocking.yaml`:

```yaml
simulation:
  name: "thermodynamic_flocking"
  max_steps: 1000
  seed: 42

environment:
  width: 100
  height: 100
  
  spatial_index:
    enable_spatial_hash_indices: true

flocking:
  n_agents: 50
  max_speed: 2.0
  perception_radius: 10.0
  alignment_weight: 1.0
  cohesion_weight: 1.0
  separation_weight: 1.5
  velocity_cost: 0.25
  base_cost: 0.03
```

#### 4.2 Update Agent to Use Config (1 hour)

```python
class FlockingAgent(BaseAgent):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load from config if available
        if config and hasattr(config, 'flocking'):
            self.max_speed = config.flocking.max_speed
            self.perception_radius = config.flocking.perception_radius
            # ... etc
```

#### 4.3 Configuration Loading (30 min)

```python
from farm.config import load_config

config = load_config("farm/config/flocking.yaml")
```

#### 4.4 Parameter Sweep Support (1 hour)

Create configs for different experiments:

```python
# Generate config variants
configs = []
for alignment_weight in [0.5, 1.0, 1.5, 2.0]:
    config = load_config("farm/config/flocking.yaml")
    config.flocking.alignment_weight = alignment_weight
    configs.append(config)

# Run batch
for config in configs:
    run_flocking_simulation(config)
```

**Test**: Run same simulation with different configs

**Deliverable**: Fully configurable simulation system

---

## Phase 5: Adaptive Flocking (3-4 hours)

**Goal**: Implement aware agents that adapt behavior

### Tasks

#### 5.1 Create AwareFlockingAgent (1 hour)

```python
class AwareFlockingAgent(FlockingAgent):
    """Flocking agent with adaptive behavior."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.density_threshold = 5
        self.low_energy_threshold = 30.0
    
    def adapt_behavior(self):
        """Adjust parameters based on local conditions."""
        neighbors = self.get_neighbors(self.perception_radius)
        density = len(neighbors)
        
        # Increase separation in crowded areas
        if density > self.density_threshold:
            self.separation_weight = 2.5
        else:
            self.separation_weight = 1.5
        
        # Slow down when low on energy
        if self.resource_level < self.low_energy_threshold:
            self.max_speed = 1.0
        else:
            self.max_speed = 2.0
    
    def act(self):
        """Act with adaptation."""
        self.adapt_behavior()
        super().act()
```

#### 5.2 Test Adaptation (1 hour)

Compare aware vs unaware agents:

```python
# Run two simulations
env_basic, metrics_basic = run_flocking_simulation(
    agent_class=FlockingAgent
)

env_aware, metrics_aware = run_flocking_simulation(
    agent_class=AwareFlockingAgent
)

# Compare metrics
compare_metrics(metrics_basic, metrics_aware)
```

**Test**: Aware agents should survive longer

#### 5.3 Additional Adaptations (1 hour)

Implement more sophisticated adaptations:

- Energy-based speed regulation
- Neighbor count-based cohesion
- Resource-seeking behavior
- Danger avoidance

#### 5.4 Analysis (1 hour)

- [ ] Compare survival rates
- [ ] Measure adaptation effectiveness
- [ ] Visualize behavioral changes over time

**Deliverable**: Adaptive flocking agents with improved survival

---

## Phase 6: Evolutionary Flocking (4-6 hours)

**Goal**: Implement reproduction and trait evolution

### Tasks

#### 6.1 Create EvoFlockingAgent (2 hours)

```python
class EvoFlockingAgent(AwareFlockingAgent):
    """Flocking agent with reproduction and genetics."""
    
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
    
    def can_reproduce(self):
        return (
            self.alive and 
            self.resource_level > self.reproduction_threshold
        )
    
    def reproduce(self):
        """Create offspring with mutated traits."""
        if not self.can_reproduce():
            return False
        
        # Split energy
        child_energy = self.resource_level / 2
        self.resource_level /= 2
        
        # Mutate traits
        child_traits = self.mutate_traits(self.traits)
        
        # Create child
        child = EvoFlockingAgent(
            agent_id=self.environment.get_next_agent_id(),
            position=self.position,
            resource_level=child_energy,
            spatial_service=self.spatial_service,
            environment=self.environment,
            generation=self.generation + 1
        )
        
        child.traits = child_traits
        child.apply_traits()
        
        self.environment.add_agent(child)
        return True
    
    def act(self):
        super().act()
        
        # Try to reproduce
        if self.can_reproduce():
            self.reproduce()
```

#### 6.2 Trait Mutation System (1 hour)

```python
def mutate_traits(self, traits):
    """Apply random mutations."""
    mutated = traits.copy()
    
    for trait_name, value in mutated.items():
        if np.random.random() < self.mutation_rate:
            # Random change ±20%
            change = np.random.uniform(-0.2, 0.2)
            mutated[trait_name] *= (1 + change)
            
            # Clamp to reasonable ranges
            mutated[trait_name] = self.clamp_trait(
                trait_name, mutated[trait_name]
            )
    
    return mutated

def clamp_trait(self, name, value):
    """Ensure traits stay in valid ranges."""
    ranges = {
        'max_speed': (1.0, 3.0),
        'separation_weight': (1.0, 3.0),
        'energy_efficiency': (0.5, 1.5)
    }
    min_val, max_val = ranges.get(name, (0, 100))
    return np.clip(value, min_val, max_val)
```

#### 6.3 Evolution Metrics (1 hour)

Track evolutionary progress:

```python
class EvolutionMetrics:
    def __init__(self):
        self.generation_count = []
        self.avg_fitness = []
        self.trait_distributions = {}
    
    def update(self, agents, step):
        alive = [a for a in agents if a.alive]
        
        if alive:
            self.generation_count.append(
                np.mean([a.generation for a in alive])
            )
            
            # Track trait evolution
            for trait_name in ['max_speed', 'energy_efficiency']:
                values = [a.traits[trait_name] for a in alive]
                if trait_name not in self.trait_distributions:
                    self.trait_distributions[trait_name] = []
                self.trait_distributions[trait_name].append(
                    np.mean(values)
                )
```

#### 6.4 Evolution Analysis (1 hour)

- [ ] Plot trait evolution over time
- [ ] Analyze selection pressures
- [ ] Measure adaptation rate
- [ ] Compare evolved vs initial populations

**Test**: Traits should evolve toward optimal values

**Deliverable**: Fully evolutionary flocking simulation

---

## Phase 7: Visualization and Animation (3-4 hours)

**Goal**: Create compelling visualizations of flocking behavior

### Tasks

#### 7.1 Real-time Animation (2 hours)

```python
import matplotlib.animation as animation

def create_flocking_animation(env, save_path="flocking.gif"):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def animate(frame):
        ax.clear()
        
        agents = env.get_all_agents()
        alive = [a for a in agents if a.alive]
        
        if alive:
            positions = np.array([a.position for a in alive])
            energies = np.array([a.resource_level for a in alive])
            
            # Scatter plot colored by energy
            ax.scatter(
                positions[:, 0], positions[:, 1],
                c=energies, cmap='RdYlGn',
                vmin=0, vmax=100, s=50
            )
        
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_title(f'Step {frame}')
    
    anim = animation.FuncAnimation(
        fig, animate, frames=1000, interval=50
    )
    
    anim.save(save_path, writer='pillow', fps=20)
```

#### 7.2 Velocity Field Visualization (1 hour)

Add velocity arrows to animation:

```python
def plot_velocity_field(agents, ax):
    positions = np.array([a.position for a in agents])
    velocities = np.array([a.velocity for a in agents])
    
    ax.quiver(
        positions[:, 0], positions[:, 1],
        velocities[:, 0], velocities[:, 1],
        scale=20, width=0.003, alpha=0.6
    )
```

#### 7.3 Interactive Dashboard (1 hour)

Create dashboard with real-time metrics:

```python
def create_dashboard(metrics):
    from ipywidgets import interact, IntSlider
    
    @interact(step=IntSlider(min=0, max=len(metrics.time)-1))
    def plot_step(step):
        # Plot state at specific step
        ...
```

**Deliverable**: Publication-quality visualizations

---

## Phase 8: Testing and Validation (2-3 hours)

**Goal**: Ensure correctness and robustness

### Tasks

#### 8.1 Unit Tests (1 hour)

```python
import unittest

class TestFlockingAgent(unittest.TestCase):
    def test_alignment_computation(self):
        # Test alignment force calculation
        ...
    
    def test_energy_consumption(self):
        # Verify energy decreases correctly
        ...
    
    def test_boundary_wrapping(self):
        # Check toroidal boundaries
        ...
```

#### 8.2 Integration Tests (1 hour)

```python
def test_full_simulation():
    """Test complete simulation run."""
    env, metrics = run_flocking_simulation(
        n_agents=10, n_steps=100
    )
    
    assert len(metrics.time) == 100
    assert all(m >= 0 for m in metrics.alignment)
    # ... more assertions
```

#### 8.3 Performance Benchmarks (1 hour)

```python
import time

def benchmark_simulation():
    start = time.time()
    run_flocking_simulation(n_agents=100, n_steps=1000)
    duration = time.time() - start
    
    print(f"Duration: {duration:.2f}s")
    print(f"Steps/second: {1000/duration:.1f}")
```

**Deliverable**: Validated, tested implementation

---

## Phase 9: Documentation (2-3 hours)

**Goal**: Create comprehensive documentation

### Tasks

#### 9.1 Code Documentation (1 hour)
- [ ] Add docstrings to all classes and methods
- [ ] Include usage examples in docstrings
- [ ] Document parameters and return values

#### 9.2 User Guide (1 hour)
- [ ] Write getting started guide
- [ ] Create parameter tuning guide
- [ ] Document common issues and solutions

#### 9.3 API Reference (30 min)
- [ ] Generate API docs with Sphinx
- [ ] Document configuration options
- [ ] List all metrics and their meanings

#### 9.4 Example Gallery (30 min)
- [ ] Create example notebooks
- [ ] Show different parameter configurations
- [ ] Demonstrate analysis workflows

**Deliverable**: Complete documentation

---

## Phase 10: Advanced Features (Optional, 4-6 hours)

**Goal**: Implement additional interesting features

### Optional Extensions

#### 10.1 Predator-Prey Dynamics
- Add predator agents that hunt flockers
- Implement escape behaviors
- Track survival strategies

#### 10.2 Multi-Species Flocking
- Different species with different parameters
- Inter-species interactions
- Competitive exclusion dynamics

#### 10.3 Environmental Gradients
- Energy density varies spatially
- Agents learn optimal foraging areas
- Migration patterns emerge

#### 10.4 Learning-Based Flocking
- Use RL to learn flocking weights
- Adaptive rule discovery
- Compare learned vs hardcoded rules

---

## Success Criteria

### Minimal Success (Phase 1-3)
- ✓ Basic flocking behavior visible
- ✓ Energy costs implemented
- ✓ Metrics tracked and plotted
- ✓ Database logging works

### Complete Success (Phase 1-9)
- ✓ All three modes (classic, aware, evo) implemented
- ✓ Comprehensive metrics (including thermodynamic)
- ✓ Full configuration system
- ✓ Visualizations and animations
- ✓ Tests passing
- ✓ Documentation complete

### Excellence (Phase 1-10)
- ✓ Advanced features implemented
- ✓ Performance optimized for 1000+ agents
- ✓ Publication-ready visualizations
- ✓ Research-grade analysis tools

---

## Troubleshooting Guide

### Issue: Agents not flocking
**Solutions**:
1. Check `perception_radius` is large enough (>= 10)
2. Verify `get_neighbors()` returns agents
3. Ensure forces are being applied to velocity
4. Check spatial index is working

### Issue: All agents dying
**Solutions**:
1. Reduce `velocity_cost_coefficient`
2. Increase `ambient_replenishment`
3. Increase `initial_energy_max`
4. Add more energy resources

### Issue: Slow performance
**Solutions**:
1. Enable spatial hash grid
2. Reduce `perception_radius`
3. Decrease agent count
4. Optimize neighbor queries

### Issue: No emergent patterns
**Solutions**:
1. Increase agent count (> 30)
2. Adjust flocking weights
3. Increase simulation steps
4. Check energy balance

---

## Resources

- **Full Guide**: `docs/experiments/thermodynamic_flocking_implementation_guide.md`
- **Quick Reference**: `docs/experiments/thermodynamic_flocking_quick_reference.md`
- **Starter Code**: `examples/flocking_simulation_starter.py`
- **Generic Guide**: `docs/generic_simulation_scenario_howto.md`

---

## Timeline Summary

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1. Minimal Prototype | 2-4 hours | 4 hours |
| 2. Energy System | 4-6 hours | 10 hours |
| 3. Advanced Metrics | 3-4 hours | 14 hours |
| 4. Configuration | 2-3 hours | 17 hours |
| 5. Adaptive Flocking | 3-4 hours | 21 hours |
| 6. Evolutionary | 4-6 hours | 27 hours |
| 7. Visualization | 3-4 hours | 31 hours |
| 8. Testing | 2-3 hours | 34 hours |
| 9. Documentation | 2-3 hours | 37 hours |
| **Total (Phases 1-9)** | **25-37 hours** | **~2-3 days** |

---

## Next Actions

**Immediate (Today)**:
1. Run starter template
2. Understand code flow
3. Modify parameters and observe

**Short-term (This Week)**:
1. Complete Phase 1-3 (minimal working version)
2. Add configuration system
3. Create basic visualizations

**Medium-term (This Month)**:
1. Implement all three modes
2. Complete testing
3. Write documentation

**Long-term**:
1. Optimize for large populations
2. Add advanced features
3. Publish results

---

Good luck with your implementation! 🚀
