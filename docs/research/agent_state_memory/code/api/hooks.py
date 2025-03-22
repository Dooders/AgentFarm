"""Memory hooks for integrating with agent lifecycle events."""

import functools
import logging
import time
from typing import Type, Dict, Any, Optional

from farm.agents.base_agent import BaseAgent
from ..config import MemoryConfig
from ..core import AgentMemorySystem

logger = logging.getLogger(__name__)


def install_memory_hooks(agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
    """Install memory hooks on an agent class.
    
    This is a class decorator that adds memory hooks to BaseAgent subclasses.
    Hooks are only installed if enable_memory_hooks is True in the memory config.
    
    Args:
        agent_class: The agent class to install hooks on
        
    Returns:
        The modified agent class
    """
    original_init = agent_class.__init__
    original_act = agent_class.act
    original_get_state = agent_class.get_state
    
    @functools.wraps(original_init)
    def init_with_memory(self, *args, **kwargs):
        """Initialize with memory system support."""
        original_init(self, *args, **kwargs)
        
        # Get memory system
        memory_config = getattr(self.config, "memory_config", None)
        if isinstance(memory_config, dict):
            memory_config = MemoryConfig(**memory_config)
        
        # Early return if memory hooks are disabled
        if memory_config and not memory_config.enable_memory_hooks:
            logger.info(f"Memory hooks disabled for agent {getattr(self, 'agent_id', 'unknown')}")
            return
            
        try:
            self.memory_system = AgentMemorySystem.get_instance(memory_config)
            # Track last error time to avoid spamming logs
            self._memory_last_error_time = 0
            # Track if we're currently recording to prevent duplicate stores
            self._memory_recording = False
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            # Create a dummy memory system to avoid attribute errors
            self.memory_system = None
    
    @functools.wraps(original_act)
    def act_with_memory(self, *args, **kwargs):
        """Act with memory integration.
        
        Captures state before and after action, calculates importance,
        and handles errors gracefully with fallback behavior.
        """
        if not hasattr(self, "memory_system") or self.memory_system is None:
            return original_act(self, *args, **kwargs)
            
        # Get state before action
        try:
            self._memory_recording = True
            state_before = self.get_state()
            step_number = getattr(self, "step_number", 0)
            
            # Call original act method
            start_time = time.time()
            result = original_act(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Get state after action
            state_after = self.get_state()
            
            # Calculate importance score based on state change and reward
            reward = getattr(result, "reward", 0.0)
            state_diff = self._calculate_state_difference(state_before, state_after)
            importance = min(1.0, (0.5 * abs(reward) + 0.5 * state_diff))
            
            # Create action record with metadata
            action_data = {
                "action_type": getattr(result, "action_type", "unknown"),
                "action_params": getattr(result, "params", {}),
                "state_before": state_before,
                "state_after": state_after,
                "reward": reward,
                "execution_time": execution_time,
                "step_number": step_number
            }
            
            # Store in memory with importance score for prioritization
            self.memory_system.store_agent_action(
                self.agent_id,
                action_data,
                step_number,
                priority=importance
            )
            
            # Reset recording flag
            self._memory_recording = False
            return result
            
        except Exception as e:
            # Avoid spamming logs with memory errors
            current_time = time.time()
            if current_time - getattr(self, "_memory_last_error_time", 0) > 60:
                logger.error(f"Memory error in act_with_memory: {e}")
                self._memory_last_error_time = current_time
                
            # Reset recording flag and fall back to original behavior
            self._memory_recording = False
            return original_act(self, *args, **kwargs)
    
    @functools.wraps(original_get_state)
    def get_state_with_memory(self, *args, **kwargs):
        """Get state with memory integration.
        
        Retrieves state and stores in memory if not already recording.
        """
        # Call original get_state method
        state = original_get_state(self, *args, **kwargs)
        
        # Skip storage if we're already in a recording process or memory system is unavailable
        if not hasattr(self, "memory_system") or self.memory_system is None:
            return state
            
        # Store in memory if not already storing in act method
        if not getattr(self, "_memory_recording", False):
            try:
                step_number = getattr(self, "step_number", 0)
                
                # Calculate importance based on state attributes 
                # (agents with low health or high rewards are more important to track)
                importance = 0.5  # Default importance
                if "health" in state:
                    # Low health is more important to remember
                    importance = max(importance, 1.0 - state["health"])
                if "reward" in state:
                    # High rewards (positive or negative) are important 
                    importance = max(importance, min(1.0, abs(state["reward"]) / 10.0))
                
                self.memory_system.store_agent_state(
                    self.agent_id,
                    state,
                    step_number,
                    priority=importance
                )
            except Exception as e:
                # Avoid spamming logs with memory errors
                current_time = time.time()
                if current_time - getattr(self, "_memory_last_error_time", 0) > 60:
                    logger.error(f"Memory error in get_state_with_memory: {e}")
                    self._memory_last_error_time = current_time
        
        return state
    
    def _calculate_state_difference(self, state_before: Dict[str, Any], state_after: Dict[str, Any]) -> float:
        """Calculate a normalized difference between two states.
        
        Args:
            state_before: State before action
            state_after: State after action
            
        Returns:
            Normalized difference score (0.0-1.0)
        """
        # Handle empty states
        if not state_before or not state_after:
            return 1.0
            
        # Track differences in numeric values
        diff_sum = 0
        diff_count = 0
        
        # Compare common numeric keys
        for key in set(state_before.keys()) & set(state_after.keys()):
            before_val = state_before[key]
            after_val = state_after[key]
            
            # Only compare numeric values
            if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                # Normalize difference based on value scale
                value_scale = max(1.0, abs(before_val))
                diff = abs(after_val - before_val) / value_scale
                diff_sum += min(1.0, diff)  # Cap at 1.0
                diff_count += 1
                
        # Return average difference, defaulting to 0.5 if no comparable values
        return diff_sum / diff_count if diff_count > 0 else 0.5
    
    # Replace methods only if this is the first time applying hooks to this class
    if not hasattr(agent_class, '_memory_hooks_installed'):
        agent_class.__init__ = init_with_memory
        agent_class.act = act_with_memory
        agent_class.get_state = get_state_with_memory
        agent_class._calculate_state_difference = _calculate_state_difference
        agent_class._memory_hooks_installed = True
    
    return agent_class


def with_memory(agent_instance: BaseAgent) -> BaseAgent:
    """Add memory capabilities to an existing agent instance.
    
    Args:
        agent_instance: The agent instance to add memory to
        
    Returns:
        The agent with memory capabilities
    """
    # Skip if memory hooks are disabled in config
    memory_config = getattr(agent_instance.config, "memory_config", None)
    if memory_config:
        if isinstance(memory_config, dict):
            memory_config = MemoryConfig(**memory_config)
        if not memory_config.enable_memory_hooks:
            logger.info(f"Memory hooks disabled for agent {getattr(agent_instance, 'agent_id', 'unknown')}")
            return agent_instance
    
    # Create a dynamic subclass with memory hooks
    agent_class = type(agent_instance)
    memory_class = type(
        f"{agent_class.__name__}WithMemory",
        (agent_class,),
        {}
    )
    
    # Install hooks on the new class
    memory_class = install_memory_hooks(memory_class)
    
    # Update the instance's class
    agent_instance.__class__ = memory_class
    
    return agent_instance