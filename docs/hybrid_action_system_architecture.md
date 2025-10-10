# Hybrid Action System Architecture

## Overview

The hybrid action system provides a clean separation between external interfaces (for RL frameworks like PettingZoo) and internal agent capabilities (component-based architecture). This design maintains compatibility while enabling flexible agent behaviors.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Environment Layer                        │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   PettingZoo    │    │        Action Registry          │ │
│  │   Interface     │───▶│  ┌─────┬─────┬─────┬─────────┐  │ │
│  └─────────────────┘    │  │move │attack│gather│  ...   │  │ │
│                         │  └─────┴─────┴─────┴─────────┘  │ │
│                         └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Delegation Layer                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Action Functions                           │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │ │
│  │  │move_    │  │attack_  │  │gather_  │  │share_   │   │ │
│  │  │action() │  │action() │  │action() │  │action() │   │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  AgentCore                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
│  │  │Movement     │  │Combat       │  │Resource     │    │ │
│  │  │Component    │  │Component    │  │Component    │    │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
│  │  │Reproduction │  │Perception   │  │Other        │    │ │
│  │  │Component    │  │Component    │  │Components   │    │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Separation of Concerns

- **Environment**: Owns action registry for external compatibility
- **Agents**: Own components for internal capabilities
- **Actions**: Act as thin delegation layer

### 2. Dual Access Patterns

The system supports two ways to execute actions:

#### Registry-Based (External)
```python
# For PettingZoo/RL frameworks
action = action_registry.get("move")
result = action.execute(agent)
```

#### Component-Based (Internal)
```python
# For agent behaviors
movement = agent.get_component("movement")
result = movement.move_by(5.0, 0.0)
```

### 3. Consistent Result Format

All actions return standardized results:
```python
{
    "success": bool,
    "error": str | None,
    "details": Dict[str, Any]
}
```

## Component Mapping

| Action | Component | Method | Purpose |
|--------|-----------|--------|---------|
| `move` | `MovementComponent` | `move_by()` | Agent movement |
| `attack` | `CombatComponent` | `attack()` | Combat actions |
| `defend` | `CombatComponent` | `start_defense()` | Defensive stance |
| `gather` | `ResourceComponent` | `add()` | Resource collection |
| `share` | `ResourceComponent` | `consume()` + target `add()` | Resource sharing |
| `reproduce` | `ReproductionComponent` | `reproduce()` | Agent reproduction |
| `pass` | None | No-op | Strategic inaction |

## Execution Flow

### 1. Environment Integration

```python
# Environment.step(action_index)
action = action_registry.get(action_name)
result = action.execute(agent)
```

### 2. Action Delegation

```python
def move_action(agent: "AgentCore") -> Dict[str, Any]:
    movement = agent.get_component("movement")
    if not movement:
        return {"success": False, "error": "No movement component", "details": {}}
    
    return movement.move_by(5.0, 0.0)
```

### 3. Component Execution

```python
class MovementComponent(AgentComponent):
    def move_by(self, dx: float, dy: float) -> Dict[str, Any]:
        # Perform movement logic
        new_position = (self.position[0] + dx, self.position[1] + dy)
        self.position = new_position
        
        return {
            "success": True,
            "error": None,
            "details": {
                "old_position": old_position,
                "new_position": new_position
            }
        }
```

## Benefits

### 1. **Maintainability**
- Clear separation between external interfaces and internal logic
- Components can be modified without affecting external APIs
- Actions serve as stable delegation points

### 2. **Flexibility**
- Behaviors can choose between registry-based or direct component access
- Components can be composed in different ways
- Easy to add new actions or modify existing ones

### 3. **Compatibility**
- Maintains PettingZoo integration
- Preserves existing environment interfaces
- Backward compatible with existing code

### 4. **Testability**
- Components can be tested independently
- Actions can be tested with mock components
- Clear interfaces make mocking straightforward

## Implementation Details

### Action Function Structure

All action functions follow this pattern:

```python
def action_name(agent: "AgentCore") -> Dict[str, Any]:
    """Action that delegates to appropriate component."""
    try:
        # 1. Get required component
        component = agent.get_component("component_name")
        if not component:
            return {
                "success": False,
                "error": "Agent has no component_name component",
                "details": {}
            }
        
        # 2. Validate prerequisites (optional)
        if not component.can_perform_action():
            return {
                "success": False,
                "error": "Prerequisites not met",
                "details": {}
            }
        
        # 3. Delegate to component
        result = component.perform_action(params)
        
        # 4. Log action (optional)
        logger.info(f"Action {action_name} executed", agent_id=agent.agent_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Action {action_name} failed", agent_id=agent.agent_id, error=str(e))
        return {
            "success": False,
            "error": f"Exception during action execution: {str(e)}",
            "details": {"exception_type": type(e).__name__}
        }
```

### Component Interface

Components should implement a consistent interface:

```python
class AgentComponent:
    def __init__(self, name: str, config: ComponentConfig):
        self.name = name
        self.config = config
        self._state = {}
    
    def can_perform_action(self) -> bool:
        """Check if action can be performed."""
        return True
    
    def perform_action(self, **kwargs) -> Dict[str, Any]:
        """Perform the component's primary action."""
        raise NotImplementedError
```

## Migration Guide

### From Old Action System

1. **Remove direct state manipulation** from action functions
2. **Delegate to components** instead of accessing `agent.config`
3. **Use component methods** for behavior implementation
4. **Maintain result format** for compatibility

### Example Migration

**Before (Old System):**
```python
def move_action(agent):
    max_movement = agent.config.max_movement
    dx = random.uniform(-max_movement, max_movement)
    dy = random.uniform(-max_movement, max_movement)
    agent.position = (agent.position[0] + dx, agent.position[1] + dy)
```

**After (Hybrid System):**
```python
def move_action(agent: "AgentCore") -> Dict[str, Any]:
    movement = agent.get_component("movement")
    if not movement:
        return {"success": False, "error": "No movement component", "details": {}}
    
    return movement.move_by(5.0, 0.0)
```

## Testing Strategy

### Unit Tests
- Test each component independently
- Test action delegation logic
- Test error handling and edge cases

### Integration Tests
- Test environment → action → component flow
- Test behavior → component direct access
- Test result format consistency

### Compatibility Tests
- Verify PettingZoo integration still works
- Test existing environment code
- Validate action registry functionality

## Future Extensions

### 1. **Action Composition**
Allow actions to call multiple components:
```python
def complex_action(agent):
    movement = agent.get_component("movement")
    combat = agent.get_component("combat")
    
    # Move then attack
    move_result = movement.move_by(5.0, 0.0)
    if move_result["success"]:
        return combat.attack()
    return move_result
```

### 2. **Conditional Actions**
Actions that adapt based on agent state:
```python
def adaptive_action(agent):
    if agent.resource_level < 10:
        return gather_action(agent)
    else:
        return move_action(agent)
```

### 3. **Action Chaining**
Support for action sequences:
```python
def chained_action(agent):
    results = []
    results.append(move_action(agent))
    results.append(gather_action(agent))
    return combine_results(results)
```

This architecture provides a solid foundation for the hybrid action system while maintaining flexibility for future enhancements.
