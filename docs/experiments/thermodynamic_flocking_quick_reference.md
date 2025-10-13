# Thermodynamic Flocking - Quick Implementation Reference

## TL;DR

**Goal**: Implement Reynolds flocking with energy costs in AgentFarm  
**Approach**: Custom agent class + configuration  
**Time**: ~2-3 days  
**Difficulty**: Moderate

---

## Minimal Implementation (30 minutes)

### 1. Create FlockingAgent (`farm/core/flocking_agent.py`)

```python
from farm.core.agent import BaseAgent
import numpy as np

class FlockingAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity = np.random.uniform(-2.0, 2.0, 2)
        self.max_speed = 2.0
        self.perception_radius = 10.0
    
    def act(self):
        if not self.alive:
            return
        
        # Get neighbors
        nearby = self.spatial_service.get_nearby(
            self.position, self.perception_radius, ["agents"]
        )
        neighbors = [a for a in nearby.get("agents", []) 
                    if a.agent_id != self.agent_id and a.alive]
        
        # Flocking forces
        if neighbors:
            # Alignment
            avg_vel = np.mean([n.velocity for n in neighbors], axis=0)
            align = (avg_vel - self.velocity) * 1.0
            
            # Cohesion
            center = np.mean([n.position for n in neighbors], axis=0)
            cohere = (center - np.array(self.position)) * 0.01
            
            # Separation
            separate = np.zeros(2)
            for n in neighbors:
                diff = np.array(self.position) - np.array(n.position)
                dist = np.linalg.norm(diff)
                if dist > 0 and dist < 5.0:
                    separate += diff / (dist ** 2)
            separate *= 1.5
            
            # Apply
            force = align + cohere + separate
            self.velocity += force
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        
        # Move
        new_pos = np.array(self.position) + self.velocity
        new_pos = (new_pos[0] % self.environment.width, 
                   new_pos[1] % self.environment.height)
        self.update_position(tuple(new_pos))
        
        # Energy cost
        self.resource_level -= 0.25 * speed ** 2 + 0.03
        if self.resource_level <= 0:
            self.terminate()
```

### 2. Create Simulation Script

```python
# scripts/run_flocking_simple.py
from farm.core.environment import Environment
from farm.core.flocking_agent import FlockingAgent
import numpy as np

# Create environment
env = Environment(
    width=100, height=100,
    resource_distribution={"type": "random", "amount": 8},
    db_path="flocking.db"
)

# Add agents
for i in range(50):
    pos = (np.random.uniform(0, 100), np.random.uniform(0, 100))
    agent = FlockingAgent(
        agent_id=env.get_next_agent_id(),
        position=pos,
        resource_level=np.random.uniform(30, 100),
        spatial_service=env.spatial_service,
        environment=env
    )
    env.add_agent(agent)

# Run
for step in range(1000):
    env.step()
    if step % 100 == 0:
        alive = sum(1 for a in env.get_all_agents() if a.alive)
        print(f"Step {step}: {alive} agents alive")

env.finalize()
```

### 3. Run

```bash
python scripts/run_flocking_simple.py
```

---

## Core Concepts Mapping

| Original Concept | AgentFarm Equivalent |
|-----------------|---------------------|
| `FlockingAgent.position` | `BaseAgent.position` |
| `FlockingAgent.velocity` | Custom attribute |
| `FlockingAgent.energy` | `BaseAgent.resource_level` |
| `get_neighbors()` | `spatial_service.get_nearby()` |
| `EnergySource` | `Resource` objects |
| Toroidal wrap | `position % (width, height)` |

---

## Metrics to Track

### Built-in (via AgentFarm)
- Population count
- Average energy
- Agent positions (logged to DB)

### Custom (add to FlockingMetrics)
```python
class FlockingMetrics:
    def compute_alignment(self, agents):
        velocities = [a.velocity for a in agents]
        avg_vel = np.mean(velocities, axis=0)
        return np.linalg.norm(avg_vel) / np.mean([np.linalg.norm(v) for v in velocities])
    
    def compute_entropy(self, agents):
        speeds_squared = [np.linalg.norm(a.velocity)**2 for a in agents]
        return sum(speeds_squared) / len(agents)
    
    def compute_phase_sync(self, agents):
        angles = [np.arctan2(a.velocity[1], a.velocity[0]) for a in agents]
        return np.abs(np.mean(np.exp(1j * np.array(angles))))
```

---

## Configuration Template

```yaml
# farm/config/flocking.yaml
environment:
  width: 100
  height: 100
  
  spatial_index:
    enable_spatial_hash_indices: true
    spatial_hash_cell_size: 15.0

resources:
  initial_resources: 8
  resource_regen_rate: 0.02

flocking:
  n_agents: 50
  max_speed: 2.0
  perception_radius: 10.0
  separation_radius: 5.0
  
  # Weights
  alignment_weight: 1.0
  cohesion_weight: 1.0
  separation_weight: 1.5
  
  # Energy
  velocity_cost: 0.25  # E = 0.25 * v²
  base_cost: 0.03
  ambient_replenishment: 0.85
```

---

## Key Methods to Implement

### FlockingAgent

```python
compute_alignment()      # Steer toward avg velocity
compute_cohesion()       # Steer toward center of mass
compute_separation()     # Avoid crowding
apply_flocking_forces()  # Combine above
update_energy()          # E -= cost(v²)
act()                    # Override BaseAgent.act()
```

### FlockingMetrics

```python
compute_alignment()        # Velocity coherence
compute_entropy_production()  # Dissipation rate
compute_phase_synchrony()  # Kuramoto parameter
update()                   # Record all metrics
```

---

## Visualization

### Using Matplotlib

```python
import matplotlib.pyplot as plt

def plot_metrics(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Alignment
    axes[0,0].plot(metrics.time, metrics.alignment)
    axes[0,0].set_title('Velocity Coherence')
    
    # Energy
    axes[0,1].plot(metrics.time, metrics.avg_energy)
    axes[0,1].set_title('Average Energy')
    
    # Entropy
    axes[1,0].plot(metrics.time, metrics.entropy)
    axes[1,0].set_title('Entropy Production')
    
    # Phase Sync
    axes[1,1].plot(metrics.time, metrics.phase_sync)
    axes[1,1].set_title('Phase Synchrony')
    
    plt.tight_layout()
    plt.savefig('flocking_metrics.png')
```

---

## Variants

### 1. Classic Mode (No Energy Costs)
```python
def act(self):
    # ... flocking logic ...
    
    if not self.config.classic_mode:
        # Apply energy costs
        self.resource_level -= energy_cost
```

### 2. Adaptive Mode (Aware Agents)
```python
def act(self):
    # Adapt based on local conditions
    neighbors = self.get_neighbors(self.perception_radius)
    if len(neighbors) > 5:
        self.separation_weight = 2.5  # Increase separation
    
    # ... normal flocking ...
```

### 3. Evolutionary Mode
```python
def act(self):
    # ... normal flocking ...
    
    # Reproduction
    if self.resource_level > 80.0:
        self.reproduce()

def reproduce(self):
    # Split energy, create offspring with mutated traits
    child_energy = self.resource_level / 2
    self.resource_level /= 2
    # ... create child with mutated parameters ...
```

---

## Debugging Tips

### Common Issues

1. **Agents not moving**
   - Check `velocity` is being updated
   - Verify `update_position()` is called
   - Ensure forces aren't zero

2. **All agents dying**
   - Energy cost too high → reduce `velocity_cost`
   - Energy replenishment too low → increase `ambient_replenishment`
   - Check initial energy distribution

3. **No flocking behavior**
   - Verify `get_nearby()` returns neighbors
   - Check `perception_radius` isn't too small
   - Ensure forces are being applied

4. **Performance issues**
   - Enable spatial hash: `enable_spatial_hash_indices: true`
   - Reduce `n_agents`
   - Decrease `perception_radius`

### Logging

```python
# Add to FlockingAgent.act()
logger.debug(f"Agent {self.agent_id}: "
            f"neighbors={len(neighbors)}, "
            f"speed={speed:.2f}, "
            f"energy={self.resource_level:.1f}")
```

---

## Testing Checklist

- [ ] Agents move with velocity
- [ ] Flocking behavior visible (clustering)
- [ ] Energy decreases over time
- [ ] Agents die when energy depleted
- [ ] Resources respawn
- [ ] Metrics tracked correctly
- [ ] Database logging works
- [ ] Visualization renders

---

## Performance Targets

| Agents | Steps | Time (estimate) |
|--------|-------|-----------------|
| 50 | 1000 | ~30 seconds |
| 100 | 1000 | ~60 seconds |
| 500 | 1000 | ~5 minutes |

**Optimization**: Enable spatial hash grid for >100 agents

---

## File Structure

```
farm/
├── core/
│   └── flocking_agent.py          # FlockingAgent class
├── config/
│   └── flocking.yaml               # Configuration
├── analysis/
│   └── flocking_metrics.py         # Metrics tracking
scripts/
├── run_flocking_simple.py          # Basic runner
├── run_flocking_full.py            # Full featured
└── visualize_flocking.py           # Plotting
docs/
└── experiments/
    ├── thermodynamic_flocking_implementation_guide.md  # Full guide
    └── thermodynamic_flocking_quick_reference.md       # This file
```

---

## Next Steps

1. **Start Simple**: Implement minimal FlockingAgent (30 min)
2. **Test**: Run with 10 agents for 100 steps
3. **Add Energy**: Implement thermodynamic costs
4. **Add Metrics**: Track alignment, entropy, phase sync
5. **Visualize**: Create plots and animations
6. **Extend**: Add awareness or evolution variants
7. **Optimize**: Profile and tune for larger populations

---

## Resources

- **Full Guide**: `docs/experiments/thermodynamic_flocking_implementation_guide.md`
- **Original Code**: User-provided flocking simulation script
- **AgentFarm Docs**: `docs/generic_simulation_scenario_howto.md`
- **Agent Design**: `docs/design/Agent.md`

---

## Questions?

Check the full implementation guide for:
- Detailed code examples
- Complete class implementations
- Database integration
- Animation creation
- Evolutionary variants
- Metrics computation algorithms
