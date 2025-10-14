# Scenario Implementation Documentation

## Quick Start

**Want to run thermodynamic flocking simulation right now?**

```bash
python examples/flocking_simulation_starter.py
```

Done! The simulation runs, creates metrics, and saves a plot.

---

## Documentation

### [SCENARIO_IMPLEMENTATION_GUIDE.md](SCENARIO_IMPLEMENTATION_GUIDE.md)

**Complete guide for implementing scenarios in AgentFarm**

Covers two approaches:

1. **Standalone**: Fast, self-contained (~4 hours)
   - Single scenario
   - Quick prototyping
   - See: `examples/flocking_simulation_starter.py`

2. **Modular**: Reusable, swappable (~8 hours initial, ~6 hours per additional)
   - Multiple scenarios
   - Standard interfaces
   - Configuration-driven

**Contents**:
- Quick start (5 minutes)
- Complete implementation patterns
- Flocking example with energy constraints
- Metrics and visualization
- Variants (adaptive, evolutionary)
- Migration guide
- Debugging tips

---

## Working Code

### `examples/flocking_simulation_starter.py`

**Ready-to-run thermodynamic flocking simulation**

Features:
- 50 flocking agents with energy constraints
- Reynolds rules: alignment, cohesion, separation
- Energy costs: E ∝ v² (thermodynamic realism)
- Metrics: alignment, cohesion, entropy, phase synchrony
- Automatic visualization

Modify parameters:
```python
run_flocking_simulation(
    n_agents=100,      # Number of agents
    n_steps=2000,      # Simulation steps
    initial_energy_min=20.0,
    initial_energy_max=100.0
)
```

---

## Files

```
docs/experiments/
└── SCENARIO_IMPLEMENTATION_GUIDE.md    # Complete guide

examples/
└── flocking_simulation_starter.py      # Working code
```

That's it! Just 2 files.

---

## Which Approach?

**Choose Standalone if**:
- Single scenario
- Learning AgentFarm
- Quick prototype
- Time-constrained

**Choose Modular if**:
- Multiple scenarios
- Need scenario comparison
- Production system
- Research platform

**Not sure?** Start with standalone (run the example), migrate to modular later if needed (~2 hours).

---

## Next Steps

1. **Try it**: `python examples/flocking_simulation_starter.py`
2. **Read**: [SCENARIO_IMPLEMENTATION_GUIDE.md](SCENARIO_IMPLEMENTATION_GUIDE.md)
3. **Customize**: Modify the starter code
4. **Expand**: Add your own scenarios
