# **Memory Agent Implementation**  

## **1. Introduction**  
This document describes the implementation details of the Memory Agent component within the Agent State Memory system. The Memory Agent is responsible for managing how agents store, retrieve, and use their memory to make decisions.

For a comprehensive overview of the hierarchical memory architecture and core concepts used in this document, please refer to the [Core Concepts](core_concepts.md) document.

## **2. Implementation Objectives**  

1. **Implement Hierarchical Memory Management**  
   - Build on the hierarchical memory system defined in [Core Concepts](core_concepts.md#2-hierarchical-memory-architecture)
   - Develop transition mechanics between memory tiers  

2. **Create Memory Compression Algorithms**  
   - Implement compression techniques as outlined in [Core Concepts](core_concepts.md#4-memory-compression-techniques)
   - Optimize for minimal information loss  

3. **Develop Advanced Retrieval Mechanisms**  
   - Implement the retrieval methods described in [Core Concepts](core_concepts.md#5-memory-retrieval-methods)
   - Optimize for retrieval speed and relevance accuracy  

4. **Integrate with Agent Decision Systems**  
   - Create interfaces to allow agents to use memory in decision-making
   - Develop event hooks for memory formation during critical experiences

## **3. Implementation Details**  

### **3.1 Memory Agent Class Structure**  

```python
class MemoryAgent:
    """Manages an agent's memory across hierarchical storage tiers."""
    
    def __init__(self, agent_id, config=None):
        self.agent_id = agent_id
        self.config = config or DefaultMemoryConfig()
        self.stm_store = RedisSTMStore(agent_id, config.stm_settings)
        self.im_store = RedisIMStore(agent_id, config.im_settings)
        self.ltm_store = SQLiteLTMStore(agent_id, config.ltm_settings)
        self.compression_engine = CompressionEngine(config.compression_settings)
        
        # Optional: Initialize neural embedding engine for advanced vectorization
        # See custom_autoencoder.md for implementation details
        if config.use_neural_embeddings:
            self.embedding_engine = AutoencoderEmbeddingEngine(
                model_path=config.autoencoder_model_path,
                input_dim=config.input_dim,
                stm_dim=config.stm_dim,
                im_dim=config.im_dim,
                ltm_dim=config.ltm_dim
            )
        
    def store_experience(self, experience_data, importance=None):
        """Store a new experience in STM."""
        # Calculate importance if not provided
        if importance is None:
            importance = self._calculate_importance(experience_data)
            
        # Create memory entry using format from Core Concepts
        memory_entry = self._create_memory_entry(experience_data, importance)
        
        # Store in STM
        self.stm_store.store(memory_entry)
        
        # Check if STM requires maintenance
        if self.stm_store.needs_maintenance():
            self._maintain_memory_tiers()
            
    def retrieve_relevant_memories(self, context, limit=5):
        """Retrieve memories relevant to the current context."""
        # Generate query vector from context
        query_vector = self._vectorize_context(context)
        
        # Search across memory tiers with priority
        memories = []
        memories.extend(self.stm_store.retrieve_similar(query_vector, limit))
        
        if len(memories) < limit:
            # If we need more, check intermediate memory
            im_limit = limit - len(memories)
            memories.extend(self.im_store.retrieve_similar(query_vector, im_limit))
            
        if len(memories) < limit:
            # If still need more, check long-term memory
            ltm_limit = limit - len(memories)
            memories.extend(self.ltm_store.retrieve_similar(query_vector, ltm_limit))
            
        # Rank and return combined results
        return self._rank_memories(memories, context)
```

### **3.2 Memory Maintenance Process**  

The memory agent periodically performs maintenance to transition memories between tiers:

1. **STM → IM Transition**
   - Identifies oldest or least important STM memories
   - Applies intermediate compression using autoencoders
   - Transfers to IM storage with updated metadata

2. **IM → LTM Transition**
   - Identifies oldest IM memories or those exceeding TTL
   - Applies high-level compression
   - Transfers to LTM with updated abstract vectors

```python
def _maintain_memory_tiers(self):
    """Perform maintenance across memory tiers."""
    # Identify STM candidates for transition
    stm_candidates = self.stm_store.get_transition_candidates()
    
    # Compress and move to IM
    for memory in stm_candidates:
        compressed_memory = self.compression_engine.compress_for_im(memory)
        self.im_store.store(compressed_memory)
        self.stm_store.delete(memory.memory_id)
        
    # Identify IM candidates for transition
    im_candidates = self.im_store.get_transition_candidates()
    
    # Highly compress and move to LTM
    for memory in im_candidates:
        compressed_memory = self.compression_engine.compress_for_ltm(memory)
        self.ltm_store.store(compressed_memory)
        self.im_store.delete(memory.memory_id)
```

### **3.3 Event Hooks for Critical Experience Memory Formation**

The Memory Agent implements a specialized event hook system to automatically identify and preserve critical experiences in an agent's lifecycle. These hooks allow for enhanced memory formation during particularly important moments, without requiring explicit calls from the agent's decision systems.

```python
class MemoryAgent:
    # ... existing methods ...
    
    def __init__(self, agent_id, config=None):
        # ... existing initialization ...
        
        # Initialize event hooks registry
        self.event_hooks = {
            "critical_resource_change": [],
            "health_threshold": [],
            "novel_observation": [],
            "goal_achievement": [],
            "unexpected_outcome": [],
            "agent_interaction": [],
            "environment_change": []
        }
        
        # Register default hooks if enabled in config
        if config and config.use_default_hooks:
            self._register_default_hooks()
    
    def register_hook(self, event_type, hook_function, priority=5):
        """Register a hook function to be called when a specific event type occurs.
        
        Args:
            event_type (str): Type of event to hook into
            hook_function (callable): Function to call when event is triggered
            priority (int): Priority level (1-10, 10 being highest)
        """
        if event_type not in self.event_hooks:
            self.event_hooks[event_type] = []
            
        self.event_hooks[event_type].append({
            "function": hook_function,
            "priority": priority
        })
        
        # Sort hooks by priority (highest first)
        self.event_hooks[event_type] = sorted(
            self.event_hooks[event_type], 
            key=lambda h: h["priority"], 
            reverse=True
        )
        
    def trigger_event(self, event_type, event_data):
        """Trigger all hooks registered for a specific event type.
        
        Args:
            event_type (str): Type of event that occurred
            event_data (dict): Data related to the event
        """
        if event_type not in self.event_hooks:
            return
            
        # Add timestamp to event data
        event_data["timestamp"] = time.time()
        
        # Track if any hook has marked this as a critical experience
        is_critical = False
        importance_override = None
        
        # Execute all hooks for this event type
        for hook in self.event_hooks[event_type]:
            result = hook["function"](event_data, self)
            
            # Hook can return None (no opinion), True (critical), or a dict with options
            if result is True:
                is_critical = True
            elif isinstance(result, dict):
                if result.get("is_critical", False):
                    is_critical = True
                if "importance" in result:
                    importance_override = result["importance"]
        
        # If any hook marked this as critical, store with high importance
        if is_critical:
            self.store_experience(
                event_data,
                importance=importance_override or 0.9  # High importance by default
            )
            
    def _register_default_hooks(self):
        """Register the default set of hooks for critical experience detection."""
        # Resource change hooks
        self.register_hook(
            "critical_resource_change",
            self._hook_significant_resource_change,
            priority=7
        )
        
        # Health/status hooks
        self.register_hook(
            "health_threshold", 
            self._hook_health_critical,
            priority=9
        )
        
        # Novel observation hooks
        self.register_hook(
            "novel_observation",
            self._hook_novelty_detection,
            priority=8
        )
        
        # Goal achievement hooks
        self.register_hook(
            "goal_achievement",
            self._hook_goal_progress,
            priority=9
        )
        
        # Unexpected outcome hooks
        self.register_hook(
            "unexpected_outcome",
            self._hook_expectation_violation,
            priority=8
        )
        
    def _hook_significant_resource_change(self, event_data, memory_agent):
        """Hook function to detect significant resource changes."""
        if "previous_resources" not in event_data or "current_resources" not in event_data:
            return False
            
        prev = event_data["previous_resources"]
        curr = event_data["current_resources"]
        
        # Calculate percent change
        if prev > 0:  # Avoid division by zero
            percent_change = abs((curr - prev) / prev)
            
            # If change is significant (>25%), mark as critical
            if percent_change > 0.25:
                return {
                    "is_critical": True,
                    "importance": min(0.7 + percent_change, 0.95)
                }
                
        return False
        
    def _hook_health_critical(self, event_data, memory_agent):
        """Hook function to detect critical health situations."""
        if "health" not in event_data:
            return False
            
        health = event_data["health"]
        
        # Critical low health threshold (below 20%)
        if health < 0.2:
            return {
                "is_critical": True, 
                "importance": 0.95
            }
            
        return False
        
    def _hook_novelty_detection(self, event_data, memory_agent):
        """Hook function to detect novel observations."""
        if "observation" not in event_data:
            return False
            
        # Calculate novelty score using a vector similarity search
        # against recent memories to see if this is significantly different
        observation_vector = self._vectorize_context(event_data["observation"])
        recent_memories = memory_agent.stm_store.retrieve_similar(
            observation_vector, 
            limit=5
        )
        
        if not recent_memories:
            # No similar memories, this is novel
            return {
                "is_critical": True,
                "importance": 0.85
            }
            
        # Calculate average similarity
        similarities = [
            self._calculate_similarity(observation_vector, memory["embeddings"]["full_vector"]) 
            for memory in recent_memories
        ]
        avg_similarity = sum(similarities) / len(similarities)
        
        # If similarity is low (below 0.3), this is novel
        if avg_similarity < 0.3:
            novelty_score = 1.0 - avg_similarity  # Higher novelty = lower similarity
            return {
                "is_critical": True,
                "importance": min(0.7 + (novelty_score * 0.2), 0.9)
            }
            
        return False
        
    def _hook_goal_progress(self, event_data, memory_agent):
        """Hook function to detect significant goal progress or achievement."""
        if "goal_status" not in event_data:
            return False
            
        goal_status = event_data["goal_status"]
        
        # Significant progress or goal completion
        if goal_status.get("achieved", False) or goal_status.get("progress_change", 0) > 0.25:
            return {
                "is_critical": True,
                "importance": 0.9 if goal_status.get("achieved", False) else 0.75
            }
            
        return False
        
    def _hook_expectation_violation(self, event_data, memory_agent):
        """Hook function to detect when outcomes violate agent expectations."""
        if "expected_outcome" not in event_data or "actual_outcome" not in event_data:
            return False
            
        expected = event_data["expected_outcome"]
        actual = event_data["actual_outcome"]
        
        # For numerical outcomes, check percentage difference
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if expected != 0:  # Avoid division by zero
                percent_diff = abs((actual - expected) / expected)
                if percent_diff > 0.5:  # 50% or more difference
                    return {
                        "is_critical": True,
                        "importance": min(0.7 + (percent_diff * 0.2), 0.9)
                    }
        # For categorical outcomes, check if they match
        elif expected != actual:
            return {
                "is_critical": True,
                "importance": 0.8
            }
            
        return False
        
    def _calculate_similarity(self, vector1, vector2):
        """Calculate cosine similarity between two vectors."""
        # Simple cosine similarity implementation
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))
        
        if magnitude1 * magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)
```

### **3.4 Integration with Agent Perception System**

To leverage the event hooks system, the agent perception and action loops need to trigger appropriate events. This integration allows for automatic memory formation without requiring explicit calls within the agent's decision logic.

```python
# Example of integration with agent perception system
class Agent:
    # ... existing methods ...
    
    def observe_environment(self, environment):
        """Process observations from the environment and trigger appropriate hooks."""
        # Get current observation
        observation = environment.get_observation_for_agent(self.agent_id)
        
        # Process basic observation and store in STM
        self.memory.store_experience({
            "observation": observation,
            "agent_state": self.current_state,
            "timestamp": time.time()
        })
        
        # Check for significant health changes
        if "health" in observation and "health" in self.current_state:
            # If health dropped below critical threshold
            if observation["health"] < 0.2 and self.current_state["health"] >= 0.2:
                self.memory.trigger_event("health_threshold", {
                    "health": observation["health"],
                    "previous_health": self.current_state["health"],
                    "observation": observation
                })
        
        # Check for resource changes
        if "resources" in observation and "resources" in self.current_state:
            self.memory.trigger_event("critical_resource_change", {
                "previous_resources": self.current_state["resources"],
                "current_resources": observation["resources"],
                "observation": observation
            })
        
        # Check for novelty in observation
        # This leverages the memory system's novelty detection
        self.memory.trigger_event("novel_observation", {
            "observation": observation
        })
        
        # Update current state with new observation
        self.current_state.update(observation)
        
    def perform_action(self, action, expected_outcome=None):
        """Execute an action and trigger appropriate hooks based on the outcome."""
        # Execute the action
        result = self._execute_action_internal(action)
        
        # Store base action memory
        self.memory.store_agent_action({
            "action_type": action.type,
            "action_params": action.params,
            "result": result,
            "timestamp": time.time()
        })
        
        # If there was an expected outcome, check for expectation violations
        if expected_outcome is not None:
            self.memory.trigger_event("unexpected_outcome", {
                "action": action,
                "expected_outcome": expected_outcome,
                "actual_outcome": result,
                "agent_state": self.current_state
            })
        
        # Check if this action satisfied any goals
        updated_goals = self._update_goals_based_on_action(action, result)
        for goal_id, status in updated_goals.items():
            if status["progress_change"] > 0:
                self.memory.trigger_event("goal_achievement", {
                    "goal_id": goal_id,
                    "goal_status": status,
                    "action": action,
                    "result": result
                })
        
        return result
```

### **3.5 Custom Hook Configuration**

Agents can be configured with custom hooks specific to their domain or objectives. This example shows how to configure different types of agents with specialized critical experience detection:

```python
# Configure a resource-gathering agent with specialized hooks
def configure_resource_agent_memory(agent):
    """Configure memory hooks for a resource-gathering agent."""
    # Register custom hook for resource deposits
    agent.memory.register_hook(
        "critical_resource_change",
        lambda event, memory: {
            "is_critical": True,
            "importance": 0.9
        } if event.get("current_resources", 0) - event.get("previous_resources", 0) > 100 else False,
        priority=8
    )
    
    # Register custom hook for resource locations
    agent.memory.register_hook(
        "environment_change",
        lambda event, memory: {
            "is_critical": True,
            "importance": 0.85
        } if "resource_location" in event.get("observation", {}) else False,
        priority=7
    )

# Configure a combat agent with specialized hooks
def configure_combat_agent_memory(agent):
    """Configure memory hooks for a combat-oriented agent."""
    # Register custom hook for combat encounters
    agent.memory.register_hook(
        "agent_interaction",
        lambda event, memory: {
            "is_critical": True,
            "importance": 0.9
        } if event.get("interaction_type") == "combat" else False,
        priority=9
    )
    
    # Register custom hook for significant damage events
    agent.memory.register_hook(
        "health_threshold",
        lambda event, memory: {
            "is_critical": True,
            "importance": 0.95
        } if (event.get("previous_health", 1.0) - event.get("health", 0)) > 0.3 else False,
        priority=10
    )
```

This event hook system provides a flexible, extensible framework for automatically capturing critical experiences without requiring explicit memory storage calls throughout the agent codebase. By centralizing the logic for determining what constitutes a "critical experience," the system can evolve independently of the agent decision logic while still providing rich, contextualized memories for future decision-making.

## **4. Experimental Results**  

### **4.1 Compression Efficiency**  

**[ ! ]** ***need to replace with actual results*** **[ ! ]**

| Memory Tier | Original Size | Compressed Size | Compression Ratio | Reconstruction Error |
|-------------|---------------|-----------------|-------------------|----------------------|
| STM → IM    | 500d vectors  | 100d vectors    | 5:1               | 0.12 MSE             |
| IM → LTM    | 100d vectors  | 20d vectors     | 5:1               | 0.28 MSE             |

### **4.2 Retrieval Performance**  

| Memory Tier | Avg Query Time | Relevance Score | Cache Hit Rate |
|-------------|----------------|-----------------|----------------|
| STM         | 5ms            | 0.92            | 95%            |
| IM          | 15ms           | 0.84            | 80%            |
| LTM         | 45ms           | 0.76            | 70%            |

## **5. Integration Guide**  

### **5.1 Adding Memory Agent to an Agent**  

```python
# Agent initialization
agent = Agent(agent_id="agent-123")
agent.memory = MemoryAgent(agent_id="agent-123")

# During agent perception/action cycle
def act(self, observation):
    # Store current observation in memory
    self.memory.store_experience({
        "observation": observation,
        "agent_state": self.current_state,
        "timestamp": time.time()
    })
    
    # Retrieve relevant past experiences
    relevant_memories = self.memory.retrieve_relevant_memories(
        context=observation, 
        limit=5
    )
    
    # Use memories in decision making
    action = self.decision_model.decide(observation, relevant_memories)
    return action
```

### **5.2 Using Action State Memory**

```python
# During agent action execution
def execute_action(self, action_type, action_params):
    # Record the initial context
    initial_context = {
        "agent_state": self.current_state,
        "observation": self.current_observation
    }
    
    # Execute the action
    result = self._perform_action(action_type, action_params)
    
    # Store the action in memory with context and outcome
    self.memory.store_agent_action({
        "action_type": action_type,
        "action_params": action_params,
        "initial_context": initial_context,
        "outcome": result,
        "timestamp": time.time()
    })
    
    # Later, retrieve similar past actions to inform decisions
    def make_decision(self, action_options):
        best_option = None
        highest_score = float('-inf')
        
        for option in action_options:
            # Find similar past actions
            similar_actions = self.memory.retrieve_relevant_memories(
                context=option,
                memory_type="action",
                limit=3
            )
            
            # Evaluate option based on past outcomes
            score = self._evaluate_option_from_history(option, similar_actions)
            
            if score > highest_score:
                highest_score = score
                best_option = option
                
        return best_option
```

## **6. Future Improvements**  

- **Attention Mechanisms**: Prioritize retrieval based on learned attention patterns
- **Emotional Tagging**: Add emotional valence to memories to prioritize impactful experiences
- **Episodic Chunking**: Group related experiences into episodes for more coherent retrieval

See [Future Enhancements](future_enhancements.md) for more details on planned improvements.

---

**See Also:**
- [Core Concepts](core_concepts.md) - Fundamental architecture and data structures
- [Agent State Storage](agent_state_storage.md) - Persistent state storage implementation
- [Redis Integration](redis_integration.md) - Redis configuration for memory tiers
