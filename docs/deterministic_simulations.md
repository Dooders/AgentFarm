# Deterministic Simulations in AgentFarm

This document provides guidance on how to ensure your AgentFarm simulations are deterministic - meaning they produce exactly the same results given the same initial conditions and configuration.

## Why Determinism Matters

Determinism is essential for:

1. **Scientific Reproducibility**: Other researchers should be able to reproduce your experiments exactly
2. **Debugging**: Makes it easier to track down issues when the same sequence of events occurs each time
3. **A/B Testing**: When comparing different agent strategies or configurations, you want to isolate the impact of your changes from random variation
4. **Validation**: Ensures the system is behaving consistently as expected

## Key Factors Affecting Determinism

### 1. Random Number Generation

The most important factor for determinism is controlling all sources of randomness:

- **Setting a Seed**: Always set a specific seed value at the beginning of your simulation
- **Global Seed Management**: Ensure all random number generators are seeded properly:
  - Python's `random` module
  - NumPy's random number generators
  - PyTorch's random number generators (for neural network components)

### 2. Floating Point Determinism

Floating point calculations can introduce non-determinism:

- **Hardware Differences**: Different CPUs or GPUs may produce slightly different results
- **Parallelism**: Multi-threaded or distributed computations might execute in different orders
- **Optimization Levels**: Compiler optimizations can affect floating point precision

### 3. External State

- **Database Interactions**: Database operations might introduce ordering issues
- **File I/O**: Reading and writing to files can introduce timing dependencies
- **Network Communications**: Any external API calls introduce non-determinism

## How to Ensure Determinism in AgentFarm

### Simulation Configuration

When running simulations, always:

1. **Set a Seed**: Use the `seed` parameter in `run_simulation()` or set `config.seed`:

```python
# Setting seed in the configuration
config = SimulationConfig.from_centralized_config()
config.seed = 42

# Or passing it directly to run_simulation
environment = run_simulation(
    num_steps=1000,
    config=config,
    seed=42
)
```

2. **Control PyTorch Randomness**: For DQN and learning components:

```python
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True  # May reduce performance but increases determinism
torch.backends.cudnn.benchmark = False
```

3. **Control NumPy Randomness**:

```python
import numpy as np
np.random.seed(42)
```

4. **Disable Multi-threading**: When needed, run simulations with a single thread to avoid race conditions.

### Testing for Determinism

Use the `deterministic_test.py` script to verify that your simulations are deterministic:

```bash
python deterministic_test.py --environment testing --steps 100 --seed 42
```

This script runs two identical simulations with the same seed and compares their final states to verify determinism.

## Common Issues and Solutions

### Non-deterministic Behavior

If your simulation is not deterministic, consider these potential causes:

1. **Uncontrolled Random Sources**: Look for places where random numbers are generated without using the seeded RNG
2. **Floating Point Precision**: Reduce reliance on exact floating point comparisons
3. **Race Conditions**: Look for multi-threading or parallel code sections
4. **External Dependencies**: Identify and eliminate or control external data sources

### Performance vs. Determinism

Sometimes enforcing determinism can reduce performance:

- `torch.backends.cudnn.deterministic = True` can slow down neural network operations
- Avoiding parallelism can increase runtime
- Consider the tradeoffs based on your specific needs

## Best Practices

1. **Version Your Seeds**: Keep track of which seed was used for which experiment
2. **Document Dependencies**: Record all external dependencies that might affect determinism
3. **Test Determinism Regularly**: Use the deterministic_test.py script to verify determinism
4. **Save Initial Conditions**: Save the complete initial state to allow future reproduction

## Advanced Topics

### Deterministic Multi-Agent Behavior

When running simulations with multiple agents, order of execution can matter:

- **Fixed Action Order**: Process agents in a consistent order (e.g., by agent ID)
- **Time Step Resolution**: Ensure the time step is fine-grained enough that order effects are minimized

### Determinism Across Different Machines

To achieve determinism across different computing environments:

- **Fix Software Versions**: Use exact same versions of Python, NumPy, and PyTorch
- **Container-Based Deployment**: Use Docker to create a consistent environment
- **Avoid GPU-Specific Optimizations**: GPU operations can vary across different hardware

## Recent Enhancements for Full Determinism

We recently made several important improvements to ensure complete determinism in our simulations:

### 1. Centralized Random Seed Management

- Added `init_random_seeds()` function in `farm/core/simulation.py` to properly initialize all random number generators:
  - Python's `random` module
  - NumPy's random generators
  - PyTorch's random generators (when available)

### 2. Deterministic Environment Initialization

- `Environment` class now accepts a `seed` parameter and uses it to:
  - Initialize random number generators
  - Generate consistent agent IDs
  - Create reproducible resource distributions

### 3. Deterministic Agent IDs

- Agent IDs are now generated deterministically based on a counter and the simulation seed
- This ensures agents have the same IDs across simulation runs with the same seed

### 4. Deterministic Resource Regeneration

- Resource regeneration decisions are now determined by hash-based seeding
- Each resource has a unique hash based on its ID, position, and current simulation time
- This enables consistent resource behaviors between runs

### 5. Testing for Determinism

We've added a new script `deterministic_test.py` that:
- Runs two identical simulations with the same seed
- Compares their final states to verify determinism
- Provides detailed comparison of any differences

## Implementation Details

### Generating Deterministic Agent IDs

```python
def get_next_agent_id(self):
    """Generate a unique short ID for an agent using environment's seed."""
    if hasattr(self, 'seed_value') and self.seed_value is not None:
        # For deterministic mode, create IDs based on a counter
        if not hasattr(self, 'agent_id_counter'):
            self.agent_id_counter = 0
        
        # Use agent counter and seed to create a deterministic ID
        agent_seed = f"{self.seed_value}_{self.agent_id_counter}"
        # Increment counter for next agent
        self.agent_id_counter += 1
        
        # Use a deterministic hash function
        import hashlib
        agent_hash = hashlib.md5(agent_seed.encode()).hexdigest()[:10]
        return f"agent_{agent_hash}"
    else:
        # Non-deterministic mode uses random short ID
        return self.seed.id()
```

### Deterministic Resource Regeneration

```python
def update(self):
    """Update environment state for current time step."""
    try:
        # Update resources with deterministic regeneration if seed is set
        if hasattr(self, 'seed_value') and self.seed_value is not None:
            # Create deterministic RNG based on seed and current time
            rng = random.Random(self.seed_value + self.time)
            
            # Deterministically decide which resources regenerate
            for resource in self.resources:
                # Use resource ID and position as additional entropy sources
                decision_seed = hash((resource.resource_id, resource.position[0], 
                                    resource.position[1], self.time))
                # Mix with simulation seed
                combined_seed = (self.seed_value * 100000) + decision_seed
                # Create a deterministic random generator for this resource
                resource_rng = random.Random(combined_seed)
                
                # Check if this resource should regenerate
                if resource_rng.random() < self.config.resource_regen_rate:
                    # Regeneration logic...
```

## Testing Determinism

You can validate that your changes maintain determinism by running:

```bash
python deterministic_test.py --steps 100 --seed 42
```

This script will:
1. Run two identical simulations with the same seed
2. Compare their final states
3. Report whether they are identical

## Conclusion

With these enhancements, AgentFarm simulations are now fully deterministic when provided with the same seed. This allows for reproducible experiments and consistent behavior across runs, which is essential for scientific rigor and effective debugging.

Remember that for complete determinism across different machines, you should still use the same versions of Python, NumPy, and PyTorch, as implementation details of these libraries can sometimes affect results. 