# Integration Points with Existing Code

Based on my analysis of the codebase, here's how to integrate the new AgentMemory system with the existing agent framework:

## 1. Integration Strategy

The existing codebase already has a memory system in `farm/memory/redis_memory.py` that provides basic Redis-backed memory functionality. The new AgentMemory system will need to coexist with this system initially and eventually replace it. Here's a strategic approach:

### 1.1 Phased Integration

1. **Phase 1: Parallel Systems**
   - Implement the new AgentMemory system in `farm/memory/agent_memory/` as designed
   - Add a compatibility layer that allows the new system to be used with the existing `BaseAgent` class
   - Both systems can coexist, with new agents using the new system

2. **Phase 2: Feature Migration**
   - Gradually move agents to use the new system
   - Deprecate methods in the old system in favor of the new API

3. **Phase 3: Full Replacement**
   - Complete the transition to the new system
   - Update all agent implementations to use the new system
   - Keep a thin compatibility layer for backward compatibility if needed

## 2. Specific Hooks for BaseAgent.py

The existing `BaseAgent` class already has memory-related hooks that can be leveraged for the new system:

### 2.1 Update in `__init__` Method

```python
def __init__(self, ..., use_memory=False, memory_config=None):
    # Existing code...
    
    # Replace the current memory initialization
    self.memory = None
    if use_memory:
        memory_config = memory_config or {}
        
        # Check if we should use the new memory system
        use_new_memory = memory_config.pop("use_new_memory", False)
        
        if use_new_memory:
            # Initialize the new memory system
            from farm.memory.agent_memory.api.memory_api import AgentMemoryAPI
            from farm.memory.agent_memory.config import MemoryConfig
            
            # Convert dict to MemoryConfig if provided
            if isinstance(memory_config, dict):
                memory_config = MemoryConfig(**memory_config)
            
            # Create memory API instance
            memory_api = AgentMemoryAPI(memory_config)
            
            # Store the memory API instance
            self.memory_api = memory_api
            
            # For backward compatibility, set self.memory to a compatibility wrapper
            self.memory = AgentMemoryLegacyAdapter(self.agent_id, memory_api)
        else:
            # Use the existing memory system
            self._init_memory(memory_config)
```

### 2.2 Hook in `act` Method

```python
def act(self) -> None:
    if not self.alive:
        return

    # Reset defense status at start of turn
    self.is_defending = False

    # Resource consumption
    self.resource_level -= self.config.base_consumption_rate

    # Check starvation state - exit early if agent dies
    if self.check_starvation():
        return

    # Get current state before action
    current_state = self.get_state()
    
    # NEW: Store state before action if using new memory system
    if hasattr(self, 'memory_api'):
        self.memory_api.store_agent_state(
            self.agent_id,
            current_state.__dict__,  # Convert to dict
            self.environment.time
        )

    # Select and execute action
    action = self.decide_action()
    
    # NEW: Capture state before action execution for action memory
    state_before = self.get_state().__dict__ if hasattr(self, 'memory_api') else None
    
    # Execute the action
    action_result = action.execute(self)
    
    # NEW: Store action in new memory system
    if hasattr(self, 'memory_api'):
        # Get state after action
        state_after = self.get_state().__dict__
        
        # Create action data
        action_data = {
            "action_type": action.name,
            "action_params": getattr(action, "params", {}),
            "state_before": state_before,
            "state_after": state_after,
            "reward": getattr(action_result, "reward", 0.0),
        }
        
        # Store in memory
        self.memory_api.store_agent_action(
            self.agent_id,
            action_data,
            self.environment.time
        )
    
    # Existing memory storage (for backward compatibility)
    if self.memory and not hasattr(self, 'memory_api'):
        self.remember_experience(
            action_name=action.name,
            reward=getattr(action_result, "reward", 0.0),
            perception_data=self.get_perception()
        )

    # Store state for learning
    self.previous_state = current_state
    self.previous_action = action
```

### 2.3 Memory Hooks for Interactions

Add a new method to track agent interactions:

```python
def record_interaction(self, target_agent_id, interaction_type, outcome, metadata=None):
    """Record an interaction between this agent and another.
    
    Args:
        target_agent_id (str): ID of the agent interacted with
        interaction_type (str): Type of interaction (attack, share, etc.)
        outcome (dict): Result of the interaction
        metadata (dict, optional): Additional data about the interaction
    
    Returns:
        bool: True if recording was successful
    """
    if not hasattr(self, 'memory_api'):
        return False
        
    # Create interaction data
    interaction_data = {
        "target_agent_id": target_agent_id,
        "interaction_type": interaction_type,
        "outcome": outcome,
        "agent_state": self.get_state().__dict__,
    }
    
    # Add metadata if provided
    if metadata:
        interaction_data["metadata"] = metadata
    
    # Store in memory
    return self.memory_api.store_agent_interaction(
        self.agent_id,
        interaction_data,
        self.environment.time
    )
```

### 2.4 Memory-Enhanced Decision Making

Enhance the `decide_action` method to use memory for better decisions:

```python
def decide_action(self):
    """Select an action based on current state and memory.
    
    Uses both current perceptions and relevant memories to make decisions.
    
    Returns:
        Action: The selected action to perform
    """
    # Get current perception and state
    perception = self.get_perception()
    current_state = self.get_state()
    
    # If using new memory system, retrieve relevant memories
    relevant_memories = []
    if hasattr(self, 'memory_api'):
        # Try to get similar states
        try:
            # Convert current state to format suitable for similarity search
            current_state_dict = current_state.__dict__
            
            # Retrieve similar past states
            relevant_memories = self.memory_api.retrieve_states_by_similarity(
                self.agent_id,
                current_state_dict,
                count=5
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve memories for decision making: {e}")
    
    # Create selection state with memory augmentation
    selection_state = create_selection_state(
        perception=perception,
        agent=self,
        memories=relevant_memories
    )
    
    # Use select module to choose action
    action_index = self.select_module.select_action(selection_state)
    return self.actions[action_index]
```

## 3. Compatibility Layer

To ensure backward compatibility with the existing memory system, implement an adapter class:

```python
class AgentMemoryLegacyAdapter:
    """Adapter to make the new memory API compatible with the old interface.
    
    This allows agents using the old memory interface to work with the new system.
    """
    
    def __init__(self, agent_id, memory_api):
        """Initialize the adapter.
        
        Args:
            agent_id (str): Agent ID
            memory_api (AgentMemoryAPI): New memory API instance
        """
        self.agent_id = agent_id
        self.memory_api = memory_api
    
    def remember_state(self, step, state, action=None, reward=None, perception=None, metadata=None, ttl=None):
        """Adapter for the old remember_state method.
        
        Maps parameters from old API to new and calls the appropriate methods.
        """
        # Convert AgentState object to dictionary if needed
        state_dict = state.__dict__ if not isinstance(state, dict) else state
        
        # Store the state
        success = self.memory_api.store_agent_state(
            self.agent_id,
            state_dict,
            step
        )
        
        # If action is provided, store action data as well
        if action is not None and success:
            action_data = {
                "action_type": action,
                "state_before": state_dict,
                "reward": reward
            }
            if metadata:
                action_data["metadata"] = metadata
                
            success = self.memory_api.store_agent_action(
                self.agent_id,
                action_data,
                step
            )
        
        return success
    
    # Implement other adapter methods as needed to match the old API
    # ...
```

## 4. Memory Hooks from todo.md

Regarding the memory hooks mentioned in your todo.md file (`Should I keep the [memory hooks](memory_agent.md#51-adding-memory-agent-to-an-agent)`), my recommendation is:

**Yes, keep the memory hooks.**

Memory hooks provide a flexible and event-driven way to form memories based on significant events, rather than just storing every state. This aligns well with how human memory works - we form stronger memories around emotionally significant or unexpected events.

### 4.1 Implementing Memory Hooks

Create a hooks system in the new memory implementation:

```python
class MemoryHookManager:
    """Manages memory hooks that trigger memory formation.
    
    Hooks are functions that determine if an event should be stored as a critical
    memory, and with what importance level.
    """
    
    def __init__(self):
        """Initialize the hook manager."""
        self.hooks = []  # List of (name, function, priority) tuples
    
    def register_hook(self, name, hook_function, priority=0):
        """Register a new memory hook.
        
        Args:
            name (str): Name of the hook for identification
            hook_function (callable): Function that evaluates events
            priority (int): Priority level (higher = evaluated earlier)
        """
        self.hooks.append((name, hook_function, priority))
        # Sort hooks by priority (descending)
        self.hooks.sort(key=lambda x: x[2], reverse=True)
    
    def process_event(self, event_data):
        """Process an event through all registered hooks.
        
        Args:
            event_data (dict): Event data to evaluate
            
        Returns:
            dict or False: Dict with memory parameters if critical, False otherwise
        """
        for name, hook_func, _ in self.hooks:
            try:
                result = hook_func(event_data)
                if result:  # If hook returns a truthy value
                    # Add hook name to result
                    if isinstance(result, dict):
                        result["hook_name"] = name
                    return result
            except Exception as e:
                logger.error(f"Error in memory hook '{name}': {e}")
        
        return False  # No hook triggered for this event
```

### 4.2 Default Memory Hooks

Implement default memory hooks for common critical events:

```python
# In memory_agent.py implementation

# Initialize hook manager
self.hook_manager = MemoryHookManager()

# Register default hooks
self.hook_manager.register_hook(
    "resource_change",
    lambda event: {
        "is_critical": True,
        "importance": 0.8
    } if abs(event.get("resource_change", 0)) > 0.25 * event.get("resource_level", 1) else False,
    priority=5
)

self.hook_manager.register_hook(
    "health_damage",
    lambda event: {
        "is_critical": True,
        "importance": 0.9
    } if (event.get("previous_health", 1.0) - event.get("current_health", 0)) > 0.3 else False,
    priority=10
)

self.hook_manager.register_hook(
    "novelty_detection",
    lambda event: {
        "is_critical": True,
        "importance": 0.75
    } if event.get("novelty_score", 0) > 0.7 else False,
    priority=7
)
```

### 4.3 Integration with Event System

Connect this with the agent event system:

```python
# In BaseAgent.act()

# After executing an action, generate event data
event_data = {
    "agent_id": self.agent_id,
    "previous_health": previous_health,
    "current_health": self.current_health,
    "previous_resources": previous_resources,
    "current_resources": self.resource_level,
    "resource_change": self.resource_level - previous_resources,
    "action_type": action.name,
    "step_number": self.environment.time,
    # Add any other relevant data
}

# Process event through memory hooks if using new memory system
if hasattr(self, 'memory_api') and hasattr(self.memory_api, 'memory_system'):
    memory_agent = self.memory_api.memory_system.get_memory_agent(self.agent_id)
    
    # If hooks manager exists, process the event
    if hasattr(memory_agent, 'hook_manager'):
        hook_result = memory_agent.hook_manager.process_event(event_data)
        
        # If event is critical, store with higher importance
        if hook_result and isinstance(hook_result, dict):
            importance = hook_result.get("importance", 0.5)
            
            # Store as critical memory with higher importance
            self.memory_api.store_agent_state(
                self.agent_id,
                self.get_state().__dict__,
                self.environment.time,
                importance=importance
            )
```