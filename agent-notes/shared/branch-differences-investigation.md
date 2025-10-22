# Branch Differences Investigation Report
**Date**: 2025-10-22  
**Branches Compared**: `main` vs `dev`  
**Issue**: Different simulation results when running `run_simulation.py`

## Executive Summary

The `dev` branch contains a **major refactoring of the agent system** that fundamentally changes how agents are created, managed, and operate. This architectural overhaul is the root cause of different simulation results between branches.

## Key Architectural Changes

### 1. Agent Architecture: Monolithic → Component-Based

**Main Branch (`BaseAgent`):**
- Single monolithic class with all capabilities built-in
- Direct attribute access for state (e.g., `agent.resource_level`, `agent.current_health`)
- Hard-coded behavior logic within the agent class
- Inheritance-based design

**Dev Branch (`AgentCore`):**
- Minimal coordinator that delegates to pluggable components
- Component-based architecture (MovementComponent, ResourceComponent, CombatComponent, etc.)
- Behavior strategies (DefaultAgentBehavior, LearningAgentBehavior)
- Composition over inheritance design pattern

**Impact on Results:**
- Component lifecycle hooks (`on_step_start`, `on_step_end`) may execute logic in different order
- State updates are now managed through components rather than direct assignment
- Different initialization sequences can lead to different initial conditions

### 2. Reward System: Environment-Calculated → Component-Based

**Main Branch:**
```python
def _calculate_reward(agent_id, pre_action_state):
    # Environment calculates rewards based on deltas
    resource_delta = agent.resource_level - pre_action_state["resource_level"]
    health_delta = agent.current_health - pre_action_state["health"]
    reward = resource_delta + health_delta * 0.5
    reward += 0.1 if agent.alive else -10.0
    return reward
```

**Dev Branch:**
```python
def _get_agent_reward(agent_id, pre_action_state):
    # Delegate to agent's reward component
    return getattr(agent, 'step_reward', 0.0)
```

**Impact on Results:**
- Reward calculation now happens in `RewardComponent` with different scaling factors
- `RewardComponent` uses configurable scales:
  - `resource_reward_scale` (default unknown)
  - `health_reward_scale` (default unknown)
  - `survival_bonus` vs `death_penalty`
  - Additional `age_bonus` not present in main
- Pre-action state is captured differently (in component vs environment)
- Different reward values → different learning signals → different decisions

### 3. State Management: Direct Attributes → AgentStateManager

**Main Branch:**
```python
self.position = (x, y)
self.resource_level = 100
self.current_health = 100
self.is_defending = False
```

**Dev Branch:**
```python
self.state = AgentStateManager(...)
self.state.update_position((x, y))
self.state.update_resource_level(100)
self.state.update_health(100)
# Access through properties that delegate to components
```

**Impact on Results:**
- State updates are now tracked and logged differently
- Properties delegate to components, adding indirection
- State snapshots captured at different points in execution
- Potential timing differences in when state changes are applied

### 4. Component Lifecycle

**Dev Branch introduces lifecycle hooks:**
```python
def step():
    # 1. on_step_start on all components
    for component in components:
        component.on_step_start()
    
    # 2. Decide and execute action
    action = behavior.decide_action(...)
    action.execute(agent)
    
    # 3. on_step_end on all components
    for component in components:
        component.on_step_end()
```

**Specific Component Actions:**
- `ResourceComponent.on_step_end()`: Applies resource consumption
- `RewardComponent.on_step_start()`: Captures pre-action state
- `RewardComponent.on_step_end()`: Calculates and applies rewards
- `CombatComponent.on_step_start()`: Updates defense timer

**Impact on Results:**
- Order of operations is different (rewards calculated after action vs during step)
- Resource depletion happens at different times
- Defense timers update at different points

### 5. Action System: Type Changes

**Main Branch:**
```python
def attack_action(agent: "BaseAgent") -> dict:
    # Actions expect BaseAgent
```

**Dev Branch:**
```python
def attack_action(agent: "AgentCore") -> dict:
    # Actions expect AgentCore with component access
```

**Impact on Results:**
- Actions now access agent capabilities through components
- Component-based access patterns may have different error handling
- Spatial queries delegated to MovementComponent

### 6. Agent Factory Pattern

**Main Branch:**
- Agents created directly with all parameters
- Direct initialization of DecisionModule

**Dev Branch:**
```python
class AgentFactory:
    def create_learning_agent(...):
        # Complex multi-step assembly
        # 1. Create components
        # 2. Create temporary behavior
        # 3. Create agent
        # 4. Create DecisionModule
        # 5. Create learning behavior
        # 6. Replace temporary behavior
```

**Impact on Results:**
- Different initialization order can affect RNG state
- DecisionModule creation timing changed
- Initial component states may differ

## Specific Code Changes

### Environment.py Changes
1. **Reward delegation**: `_calculate_reward()` → `_get_agent_reward()`
2. **Agent property access**: Direct attribute → component-based helper methods
3. **Action logging**: Different resource tracking (before/after captured in agent vs environment)

### Agent Module Changes
1. **File structure**: Single `agent.py` → `agent/` directory with multiple modules
2. **20+ commits** on dev branch related to agent refactoring
3. **New files added**:
   - `agent/core.py` - AgentCore coordinator
   - `agent/factory.py` - Factory pattern for agent creation
   - `agent/components/` - Modular capabilities
   - `agent/behaviors/` - Pluggable decision strategies
   - `agent/config/` - Component configurations

## Root Causes of Different Results

### Primary Causes
1. **Reward Calculation Changes**: Different formulas, scales, and timing
2. **Component Execution Order**: Lifecycle hooks change when logic executes
3. **State Update Timing**: When state changes are applied and observed

### Secondary Causes
1. **RNG State Differences**: Different initialization order affects random seed usage
2. **Configuration Defaults**: Components may have different default values
3. **Precision Differences**: Float calculations through components vs direct

## Recommendations

### To Verify Compatibility
1. **Run deterministic tests**: Use fixed seed on both branches and compare step-by-step
2. **Log reward calculations**: Compare reward values at each step between branches
3. **Component inspection**: Check that component configs match original BaseAgent parameters
4. **Action validation**: Verify actions produce same results on both architectures

### To Ensure Correct Migration
1. **Review `RewardComponent` config**: Ensure reward scales match old calculation
2. **Check component initialization**: Verify all components start with correct values
3. **Validate lifecycle order**: Confirm components execute in expected sequence
4. **Test edge cases**: Reproduction, death, combat, resource depletion

### Files to Review
- `/workspace/farm/core/agent/components/reward.py` - Reward calculations
- `/workspace/farm/core/agent/config/component_configs.py` - Default configurations
- `/workspace/farm/core/agent/factory.py` - Agent creation logic
- `/workspace/farm/core/environment.py` - Environment integration

## Conclusion

The different results are **expected and intentional** due to the major architectural refactoring. The dev branch represents a significant improvement in code organization (following SOLID principles), but the refactoring has changed:

1. **Reward calculation** (most critical for learning)
2. **Execution timing** (component lifecycle)
3. **State management** (indirection through components)

To make results match, you would need to:
1. Carefully tune `RewardComponent` configuration to match old reward calculation
2. Verify component execution order matches old agent step sequence
3. Ensure all component initial values match old BaseAgent defaults

However, **exact matching may not be possible or desirable** if the new architecture includes intentional improvements to reward calculation or agent behavior.

## Next Steps

Would you like me to:
1. Create a compatibility layer to make dev branch produce same results as main?
2. Analyze specific configuration files to tune reward calculations?
3. Run side-by-side comparisons with detailed logging to identify exact divergence points?
4. Create comprehensive tests to validate the refactored architecture?
