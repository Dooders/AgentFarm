# Context Manager System for Farm Simulation

## Overview
The context manager system provides a structured way to manage agent lifecycles and relationships within the simulation. It ensures proper resource cleanup and maintains consistency in agent states.

## Key Components

### 1. Agent Context **Management**
```**python**
with BaseAgent("agent1", (0,0), 10, env) as agent:
    agent.act()  # Agent is automatically cleaned up after context
```

Agents implement the context manager protocol through:
- `__enter__`: Activates agent and registers with environment
- `__exit__`: Cleans up agent resources and relationships
- `validate_context`: Ensures agent is in valid context state

### 2. Parent-Child Relationships
Agents can establish hierarchical relationships:

```python
with BaseAgent("parent", (0,0), 10, env) as parent:
    with BaseAgent("child", (1,1), 5, env) as child:
        parent.create_child_context(child)
        parent.act()
        child.act()
```

Key features:
- Parents track child contexts
- Children reference their parent context
- Automatic cleanup of relationships
- Validation of relationship consistency

### 3. Environment Context Tracking
The environment maintains thread-safe tracking of active contexts:

```python
class Environment:
    def __init__(self):
        self._active_contexts = set()
        self._context_lock = threading.Lock()

    def register_active_context(self, agent):
        with self._context_lock:
            self._active_contexts.add(agent)
```

### 4. Multiple Agent Management
For managing multiple agents simultaneously:

```python
from contextlib import ExitStack

with ExitStack() as stack:
    agents = [
        stack.enter_context(BaseAgent(f"agent_{i}", (i,i), 10, env))
        for i in range(10)
    ]
    
    for agent in agents:
        agent.act()
```

## Benefits

1. **Resource Management**
   - Automatic cleanup of agent resources
   - Proper handling of database connections
   - Memory leak prevention

2. **State Consistency**
   - Validates agent state before actions
   - Maintains relationship integrity
   - Prevents orphaned agents

3. **Error Handling**
   - Graceful cleanup on exceptions
   - Context-specific logging
   - Clear error messages

4. **Relationship Management**
   - Hierarchical agent relationships
   - Automatic cleanup of child contexts
   - Validation of relationship consistency

## Controller Integration

### SimulationController
The SimulationController also implements the context manager protocol:

```python
with SimulationController(config, "sim.db") as sim:
    sim.initialize_simulation()
    sim.start()
    # Controller and resources automatically cleaned up
```

Benefits:
- Automatic cleanup of simulation resources
- Proper database connection handling
- Thread cleanup on exit

### ExperimentController
The ExperimentController provides context management for experiment runs:

```python
with ExperimentController("exp1", "Test", config) as exp:
    exp.run_experiment(
        num_iterations=10,
        variations=variations,
        num_steps=1000
    )
    # Experiment resources and analysis automatically cleaned up
```

Benefits:
- Manages experiment lifecycle
- Handles iteration cleanup
- Ensures analysis completion

## Usage Examples

### Basic Agent Usage
```python
with BaseAgent("agent1", (0,0), 10, env) as agent:
    agent.act()
```

### Parent-Child Relationship
```python
with BaseAgent("parent", (0,0), 10, env) as parent:
    with BaseAgent("child", (1,1), 5, env) as child:
        parent.create_child_context(child)
        # Both agents cleaned up properly
```

### Multiple Agents
```python
with ExitStack() as stack:
    agents = [
        stack.enter_context(BaseAgent(f"agent_{i}", (i,i), 10, env))
        for i in range(10)
    ]
    # All agents managed together
```

### Full Simulation Example
```python
with SimulationController(config, "sim.db") as sim:
    with ExitStack() as stack:
        agents = [
            stack.enter_context(BaseAgent(f"agent_{i}", (i,i), 10, sim.environment))
            for i in range(10)
        ]
        
        sim.initialize_simulation()
        sim.start()
        
        while sim.is_running:
            for agent in agents:
                agent.act()
```

## Key Methods

### BaseAgent
- `__enter__()`: Activates agent in environment
- `__exit__()`: Cleans up agent resources
- `validate_context()`: Ensures valid context state
- `create_child_context()`: Establishes parent-child relationship

### Environment
- `register_active_context()`: Tracks active agent contexts
- `unregister_active_context()`: Removes inactive contexts
- `get_active_contexts()`: Returns current active contexts

### SimulationController
- `__enter__()`: Initializes simulation resources
- `__exit__()`: Ensures simulation cleanup
- `cleanup()`: Handles resource cleanup

### ExperimentController
- `__enter__()`: Sets up experiment environment
- `__exit__()`: Cleans up experiment resources
- `cleanup()`: Handles analysis and data cleanup

## Best Practices

1. Always use agents within context managers
2. Validate context before critical operations
3. Clean up parent-child relationships explicitly
4. Use ExitStack for multiple agents
5. Handle exceptions appropriately
6. Use controllers within context managers
7. Nest contexts appropriately for complex operations

## Common Patterns

1. **Single Agent Operations**
   ```python
   with agent as a:
       a.act()
   ```

2. **Parent-Child Operations**
   ```python
   with parent as p, child as c:
       p.create_child_context(c)
       p.act()
       c.act()
   ```

3. **Batch Operations**
   ```python
   with ExitStack() as stack:
       agents = [stack.enter_context(agent) for agent in agent_list]
       for agent in agents:
           agent.act()
   ```

4. **Full Experiment**
   ```python
   with ExperimentController("exp1", "Test", config) as exp:
       exp.run_experiment(num_iterations=10)
       exp.analyze_results()
   ```

## Error Handling

Always use try/finally blocks for complex operations:

```python
with SimulationController(config, "sim.db") as sim:
    try:
        sim.initialize_simulation()
        sim.start()
    except SimulationError as e:
        logger.error(f"Simulation failed: {e}")
        raise
    finally:
        # Context manager ensures cleanup
        pass
```

## Thread Safety

The context manager system is thread-safe:
- Environment uses locks for context tracking
- Controllers handle thread cleanup
- Agents validate context state before operations

## Conclusion

The context manager system provides a robust way to manage simulation resources and relationships. Using context managers ensures proper cleanup and state consistency throughout the simulation lifecycle. 