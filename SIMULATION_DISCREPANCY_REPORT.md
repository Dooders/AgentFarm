# Simulation Result Discrepancy Investigation Report

## Issue Summary
The simulation on the `dev` branch produces different results compared to `main` branch:
- **Main branch**: 59 final agents after 100 steps
- **Dev branch**: 30 final agents after 100 steps  

## Root Cause

### Primary Issue: Broken Reproduction System

The dev branch has a **broken reproduction mechanism** that prevents agents from creating offspring:

1. **Missing `reproduce()` method in `AgentCore`**
   - The `reproduce_action()` function in `farm/core/action.py` calls `agent.reproduce()` (line 1179)
   - However, `AgentCore` class (which replaced `BaseAgent`) does not implement a `reproduce()` method
   - The `ReproductionComponent` has a `reproduce()` method but it always returns `None` (line 111 of `farm/core/agent/components/reproduction.py`)
   - The component's reproduce method explicitly states it's a "template method" and actual offspring creation should be implemented by the factory or lifecycle service
   - **No integration exists** between the action system and the reproduction component

2. **Result**: No offspring are ever created in the dev branch
   - All 30 agents in the final count are the initial agents (10 system + 10 independent + 10 control)
   - Zero reproduction events occur during the simulation
   - On main branch, agents successfully reproduce, growing from 30 initial agents to 59 final agents

### Secondary Issue: Different Initial Resources (Not the Primary Cause)

The dev branch also changed how agents are initialized:

1. **Main branch**: Agents start with `resource_level=1` (hardcoded in `farm/core/simulation.py`)
2. **Dev branch**: Agents start with `initial_resources=100` (default fallback in `farm/core/simulation.py` line 137 and `farm/core/agent/factory.py` line 48)

This change affects agent behavior but is **overshadowed** by the broken reproduction system.

## Code References

### Dev Branch Issues

**Missing reproduce method in AgentCore** (`farm/core/agent/core.py`):
```python
# AgentCore class has NO reproduce() method
# Properties and methods exist for other capabilities, but reproduce is missing
```

**Incomplete ReproductionComponent** (`farm/core/agent/components/reproduction.py:69-111`):
```python
def reproduce(self) -> Optional["AgentCore"]:
    """
    Create offspring agent.
    
    This is a template method - actual offspring creation should be
    implemented by the factory or lifecycle service.
    
    Returns:
        New agent instance or None if reproduction failed
    """
    if not self.can_reproduce():
        return None
    
    if not self.core:
        return None
    
    # Deduct cost from parent
    resource_component = self.core.get_component("resource")
    if resource_component:
        resource_component.remove(self.config.offspring_cost)
    
    self.offspring_created += 1
    
    # ... logging code ...
    
    return None  # ← ALWAYS RETURNS NONE!
```

**Action system expects reproduce method** (`farm/core/action.py:1179`):
```python
# Attempt reproduction using the agent's method
success = agent.reproduce()  # ← Calls agent.reproduce() but method doesn't exist!
```

### Main Branch (Working)

**BaseAgent has functional reproduce** (`farm/core/agent.py`):
```python
def reproduce(self) -> bool:
    """Create offspring agent. Assumes resource requirements already checked by action."""
    # Store initial resources for tracking
    initial_resources = self.resource_level

    try:
        # Attempt to create offspring
        new_agent = self._create_offspring()  # ← Actually creates offspring
        
        # Record successful reproduction
        # ... logging code ...
        
        return True
```

## Recommendations

### Immediate Fix Required

1. **Implement `reproduce()` method in `AgentCore`**:
   - Add a `reproduce()` method to `AgentCore` that:
     - Checks reproduction eligibility via `ReproductionComponent.can_reproduce()`
     - Calls the reproduction component to deduct resources
     - Uses the `AgentFactory` to create offspring
     - Returns success/failure status

2. **Update `ReproductionComponent`**:
   - Either have it return the actual offspring (requires factory access)
   - Or have it coordinate with a lifecycle service that can create offspring
   - Remove the hardcoded `return None`

3. **Verify Initial Resources**:
   - Decide on the correct initial resource level (1, 5, or 100)
   - Update config loading to ensure `agent_behavior.initial_resource_level` is properly read from YAML
   - The YAML has `initial_resource_level: 5` but code may be using fallback of 100

### Testing Required

After fixing:
1. Run the same simulation on both branches
2. Verify agent counts match (or document expected differences)
3. Verify reproduction events are logged in dev branch
4. Compare performance metrics (speed, final population, etc.)

## Additional Notes

- The dev branch simulation runs faster (6.77s vs 18.06s) but this is likely because:
  - Fewer agents to process (30 vs 59)
  - No reproduction logic being executed
  - Database validation added but that's a minor overhead

- The agent refactor introduced a component-based architecture which is a good design, but the integration with the action system was not completed for reproduction

- Other actions (move, gather, attack, share) may work correctly as they don't require factory-level agent creation
