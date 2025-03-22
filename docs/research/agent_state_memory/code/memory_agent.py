"""Memory Agent implementation for agent state management."""

import logging
import time
from typing import Dict, Any, List, Optional, Union

from config import MemoryConfig
from storage.redis_stm import RedisSTMStore
from storage.redis_im import RedisIMStore
from storage.sqlite_ltm import SQLiteLTMStore
from embeddings.compression import CompressionEngine
from embeddings.autoencoder import AutoencoderEmbeddingEngine

logger = logging.getLogger(__name__)


class MemoryAgent:
    """Manages an agent's memory across hierarchical storage tiers.
    
    This class provides a unified interface for storing and retrieving
    agent memories across different storage tiers with varying levels
    of compression and resolution.
    
    Attributes:
        agent_id: Unique identifier for the agent
        config: Configuration for the memory agent
        stm_store: Short-Term Memory store (Redis)
        im_store: Intermediate Memory store (Redis with TTL)
        ltm_store: Long-Term Memory store (SQLite)
        compression_engine: Engine for compressing memory entries
        embedding_engine: Optional neural embedding engine
    """
    
    def __init__(self, agent_id: str, config: MemoryConfig):
        """Initialize the MemoryAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration for the memory agent
        """
        self.agent_id = agent_id
        self.config = config
        
        # Initialize memory stores
        self.stm_store = RedisSTMStore(agent_id, config.stm_config)
        self.im_store = RedisIMStore(agent_id, config.im_config)
        self.ltm_store = SQLiteLTMStore(agent_id, config.ltm_config)
        
        # Initialize compression engine
        self.compression_engine = CompressionEngine(config.autoencoder_config)
        
        # Optional: Initialize neural embedding engine for advanced vectorization
        if config.autoencoder_config.use_neural_embeddings:
            self.embedding_engine = AutoencoderEmbeddingEngine(
                model_path=config.autoencoder_config.model_path,
                input_dim=config.autoencoder_config.input_dim,
                stm_dim=config.autoencoder_config.stm_dim,
                im_dim=config.autoencoder_config.im_dim,
                ltm_dim=config.autoencoder_config.ltm_dim
            )
        else:
            self.embedding_engine = None
        
        # Internal state
        self._insert_count = 0
        
        logger.debug("MemoryAgent initialized for agent %s", agent_id)
    
    def store_state(
        self, 
        state_data: Dict[str, Any], 
        step_number: int,
        priority: float = 1.0
    ) -> bool:
        """Store an agent state in memory.
        
        Args:
            state_data: Dictionary containing agent state attributes
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            True if storage was successful
        """
        # Create a standardized memory entry
        memory_entry = self._create_memory_entry(
            state_data, 
            step_number, 
            "state", 
            priority
        )
        
        # Store in Short-Term Memory first
        success = self.stm_store.store(memory_entry)
        
        # Increment insert count and check if cleanup is needed
        self._insert_count += 1
        if self._insert_count % self.config.cleanup_interval == 0:
            self._check_memory_transition()
        
        return success
    
    def store_interaction(
        self, 
        interaction_data: Dict[str, Any], 
        step_number: int,
        priority: float = 1.0
    ) -> bool:
        """Store an agent interaction in memory.
        
        Args:
            interaction_data: Dictionary containing interaction details
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            True if storage was successful
        """
        # Create a standardized memory entry
        memory_entry = self._create_memory_entry(
            interaction_data, 
            step_number, 
            "interaction", 
            priority
        )
        
        # Store in Short-Term Memory first
        success = self.stm_store.store(memory_entry)
        
        # Increment insert count and check if cleanup is needed
        self._insert_count += 1
        if self._insert_count % self.config.cleanup_interval == 0:
            self._check_memory_transition()
        
        return success
    
    def store_action(
        self, 
        action_data: Dict[str, Any], 
        step_number: int,
        priority: float = 1.0
    ) -> bool:
        """Store an agent action in memory.
        
        Args:
            action_data: Dictionary containing action details
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            True if storage was successful
        """
        # Create a standardized memory entry
        memory_entry = self._create_memory_entry(
            action_data, 
            step_number, 
            "action", 
            priority
        )
        
        # Store in Short-Term Memory first
        success = self.stm_store.store(memory_entry)
        
        # Increment insert count and check if cleanup is needed
        self._insert_count += 1
        if self._insert_count % self.config.cleanup_interval == 0:
            self._check_memory_transition()
        
        return success
    
    def _create_memory_entry(
        self, 
        data: Dict[str, Any], 
        step_number: int,
        memory_type: str,
        priority: float
    ) -> Dict[str, Any]:
        """Create a standardized memory entry.
        
        Args:
            data: Raw data to store
            step_number: Current simulation step
            memory_type: Type of memory ("state", "interaction", "action")
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            Formatted memory entry
        """
        # Generate unique memory ID
        timestamp = int(time.time())
        memory_id = f"{self.agent_id}-{step_number}-{timestamp}"
        
        # Generate embeddings if available
        embeddings = {}
        if self.embedding_engine:
            embeddings = {
                "full_vector": self.embedding_engine.encode_stm(data),
                "compressed_vector": self.embedding_engine.encode_im(data),
                "abstract_vector": self.embedding_engine.encode_ltm(data)
            }
        
        # Create standardized memory entry
        return {
            "memory_id": memory_id,
            "agent_id": self.agent_id,
            "step_number": step_number,
            "timestamp": timestamp,
            
            "contents": data,
            
            "metadata": {
                "creation_time": timestamp,
                "last_access_time": timestamp,
                "compression_level": 0,
                "importance_score": priority,
                "retrieval_count": 0,
                "memory_type": memory_type
            },
            
            "embeddings": embeddings
        }
    
    def _check_memory_transition(self) -> None:
        """Check if memories need to be transitioned between tiers.
        
        This method implements a hybrid age-importance based memory transition
        mechanism that determines when memories should move between tiers
        based on both capacity constraints and importance scores.
        """
        current_time = time.time()
        
        # Check if STM is at capacity
        stm_count = self.stm_store.count()
        if stm_count > self.config.stm_config.memory_limit:
            # Calculate transition scores for all STM memories
            stm_memories = self.stm_store.get_all()
            transition_candidates = []
            
            for memory in stm_memories:
                # Calculate memory importance
                importance_score = self._calculate_importance(memory)
                
                # Calculate age factor (normalized by TTL)
                age = (current_time - memory["metadata"]["creation_time"]) / self.config.stm_config.ttl
                age_factor = min(1.0, max(0.0, age))
                
                # Calculate transition score: higher means more likely to transition
                transition_score = age_factor * (1.0 - importance_score)
                
                transition_candidates.append((memory, transition_score))
            
            # Sort by transition score (highest first)
            transition_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Get overflow count memories with highest transition scores
            overflow = stm_count - self.config.stm_config.memory_limit
            to_transition = transition_candidates[:overflow]
            
            # Compress and move to IM
            for memory, _ in to_transition:
                # Apply level 1 compression
                compressed_entry = self.compression_engine.compress(memory, level=1)
                compressed_entry["metadata"]["compression_level"] = 1
                compressed_entry["metadata"]["last_transition_time"] = current_time
                
                # Store in IM
                self.im_store.store(compressed_entry)
                
                # Remove from STM
                self.stm_store.delete(memory["memory_id"])
            
            logger.debug("Transitioned %d memories from STM to IM for agent %s", 
                        overflow, self.agent_id)
        
        # Check if IM is at capacity
        im_count = self.im_store.count()
        if im_count > self.config.im_config.memory_limit:
            # Calculate transition scores for all IM memories
            im_memories = self.im_store.get_all()
            transition_candidates = []
            
            for memory in im_memories:
                # Calculate memory importance
                importance_score = self._calculate_importance(memory)
                
                # Calculate age factor (normalized by TTL)
                age = (current_time - memory["metadata"]["creation_time"]) / self.config.im_config.ttl
                age_factor = min(1.0, max(0.0, age))
                
                # Calculate transition score: higher means more likely to transition
                transition_score = age_factor * (1.0 - importance_score)
                
                transition_candidates.append((memory, transition_score))
            
            # Sort by transition score (highest first)
            transition_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Get overflow count memories with highest transition scores
            overflow = im_count - self.config.im_config.memory_limit
            to_transition = transition_candidates[:overflow]
            
            # Compress and move to LTM
            batch = []
            for memory, _ in to_transition:
                # Apply level 2 compression
                compressed_entry = self.compression_engine.compress(memory, level=2)
                compressed_entry["metadata"]["compression_level"] = 2
                compressed_entry["metadata"]["last_transition_time"] = current_time
                
                # Add to batch
                batch.append(compressed_entry)
                
                # Remove from IM
                self.im_store.delete(memory["memory_id"])
                
                # Process in batches
                if len(batch) >= self.config.ltm_config.batch_size:
                    self.ltm_store.store_batch(batch)
                    batch = []
            
            # Store any remaining entries
            if batch:
                self.ltm_store.store_batch(batch)
            
            logger.debug("Transitioned %d memories from IM to LTM for agent %s", 
                        overflow, self.agent_id)
    
    def _calculate_importance(self, memory: Dict[str, Any]) -> float:
        """Calculate importance score for a memory entry.
        
        The importance score determines how likely a memory is to be retained
        in its current tier versus being transitioned to a lower tier.
        
        Args:
            memory: Memory entry to calculate importance for
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        # Reward magnitude component (40%)
        reward = memory.get("contents", {}).get("reward", 0)
        reward_importance = min(1.0, abs(reward) / 10.0) * 0.4
        
        # Retrieval frequency component (30%)
        retrieval_count = memory["metadata"].get("retrieval_count", 0)
        retrieval_factor = min(1.0, retrieval_count / 5.0) * 0.3
        
        # Recency component (20%)
        current_time = time.time()
        creation_time = memory["metadata"].get("creation_time", current_time)
        time_diff = current_time - creation_time
        recency = max(0.0, 1.0 - (time_diff / 1000)) * 0.2
        
        # Surprise factor component (10%)
        # This measures how different the outcome was from the expected outcome
        # For now, we use a simplified placeholder implementation
        surprise = memory["metadata"].get("surprise_factor", 0.5) * 0.1
        
        # Combine all factors
        importance = reward_importance + retrieval_factor + recency + surprise
        
        return min(1.0, max(0.0, importance))
    
    def clear_memory(self) -> bool:
        """Clear all memory data for this agent.
        
        Returns:
            True if clearing was successful
        """
        stm_success = self.stm_store.clear()
        im_success = self.im_store.clear()
        ltm_success = self.ltm_store.clear()
        
        return stm_success and im_success and ltm_success

    def retrieve_similar_states(
        self,
        query_state: Dict[str, Any],
        k: int = 5,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve most similar past states to the provided query state.
        
        Args:
            query_state: The state to find similar states for
            k: Number of results to return
            memory_type: Optional filter for specific memory types
            
        Returns:
            List of memory entries sorted by similarity to query state
        """
        # Generate query embedding
        if self.embedding_engine:
            query_embedding = self.embedding_engine.encode_stm(query_state)
        else:
            raise RuntimeError("Neural embeddings required for similarity search")

        # Search in each tier with appropriate embedding level
        results = []
        
        # Search STM (full resolution)
        stm_results = self.stm_store.search_similar(
            query_embedding,
            k=k,
            memory_type=memory_type
        )
        results.extend(stm_results)
        
        # If we need more results, search IM
        if len(results) < k:
            remaining = k - len(results)
            im_query = self.embedding_engine.encode_im(query_state)
            im_results = self.im_store.search_similar(
                im_query,
                k=remaining,
                memory_type=memory_type
            )
            results.extend(im_results)
        
        # If still need more, search LTM
        if len(results) < k:
            remaining = k - len(results)
            ltm_query = self.embedding_engine.encode_ltm(query_state)
            ltm_results = self.ltm_store.search_similar(
                ltm_query,
                k=remaining,
                memory_type=memory_type
            )
            results.extend(ltm_results)
        
        # Sort by similarity score
        results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return results[:k]

    def retrieve_by_time_range(
        self,
        start_step: int,
        end_step: int,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve memories within a specific time/step range.
        
        Args:
            start_step: Beginning of time range
            end_step: End of time range
            memory_type: Optional filter for specific memory types
            
        Returns:
            List of memory entries within the specified time range
        """
        results = []
        
        # Search each tier
        stm_results = self.stm_store.search_by_step_range(
            start_step,
            end_step,
            memory_type
        )
        results.extend(stm_results)
        
        im_results = self.im_store.search_by_step_range(
            start_step,
            end_step,
            memory_type
        )
        results.extend(im_results)
        
        ltm_results = self.ltm_store.search_by_step_range(
            start_step,
            end_step,
            memory_type
        )
        results.extend(ltm_results)
        
        # Sort by step number
        results.sort(key=lambda x: x["step_number"])
        return results

    def retrieve_by_attributes(
        self,
        attributes: Dict[str, Any],
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve memories matching specific attribute values.
        
        Args:
            attributes: Dictionary of attribute-value pairs to match
            memory_type: Optional filter for specific memory types
            
        Returns:
            List of memory entries matching the specified attributes
        """
        results = []
        
        # Search each tier
        stm_results = self.stm_store.search_by_attributes(
            attributes,
            memory_type
        )
        results.extend(stm_results)
        
        im_results = self.im_store.search_by_attributes(
            attributes,
            memory_type
        )
        results.extend(im_results)
        
        ltm_results = self.ltm_store.search_by_attributes(
            attributes,
            memory_type
        )
        results.extend(ltm_results)
        
        # Sort by recency (most recent first)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory usage and performance.
        
        Returns:
            Dictionary containing memory statistics including:
            - Counts per tier
            - Average importance scores
            - Compression ratios
            - Access patterns
            - Memory transitions
        """
        stats = {
            "timestamp": int(time.time()),
            "tiers": {
                "stm": {
                    "count": self.stm_store.count(),
                    "size_bytes": self.stm_store.get_size(),
                    "avg_importance": self._calculate_tier_importance("stm"),
                },
                "im": {
                    "count": self.im_store.count(),
                    "size_bytes": self.im_store.get_size(),
                    "avg_importance": self._calculate_tier_importance("im"),
                    "compression_ratio": self._calculate_compression_ratio("im"),
                },
                "ltm": {
                    "count": self.ltm_store.count(),
                    "size_bytes": self.ltm_store.get_size(),
                    "avg_importance": self._calculate_tier_importance("ltm"),
                    "compression_ratio": self._calculate_compression_ratio("ltm"),
                }
            },
            "total_memories": (
                self.stm_store.count() + 
                self.im_store.count() + 
                self.ltm_store.count()
            ),
            "memory_types": self._get_memory_type_distribution(),
            "access_patterns": self._get_access_patterns(),
        }
        
        return stats

    def force_maintenance(self) -> bool:
        """Force memory tier transitions and cleanup operations.
        
        Returns:
            True if maintenance was successful
        """
        try:
            self._check_memory_transition()
            return True
        except Exception as e:
            logger.error("Failed to perform maintenance: %s", e)
            return False

    def search_by_embedding(
        self,
        query_embedding: List[float],
        k: int = 5,
        memory_tiers: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find memories by raw embedding vector similarity.
        
        Args:
            query_embedding: Embedding vector to search with
            k: Number of results to return
            memory_tiers: Optional list of tiers to search
            
        Returns:
            List of memory entries sorted by similarity
        """
        results = []
        tiers = memory_tiers or ["stm", "im", "ltm"]
        
        for tier in tiers:
            if tier == "stm" and "stm" in tiers:
                stm_results = self.stm_store.search_by_embedding(
                    query_embedding,
                    k=k
                )
                results.extend(stm_results)
            
            if tier == "im" and "im" in tiers:
                # Compress query for IM search
                im_query = self.compression_engine.compress_embedding(
                    query_embedding,
                    level=1
                )
                im_results = self.im_store.search_by_embedding(
                    im_query,
                    k=k
                )
                results.extend(im_results)
            
            if tier == "ltm" and "ltm" in tiers:
                # Compress query for LTM search
                ltm_query = self.compression_engine.compress_embedding(
                    query_embedding,
                    level=2
                )
                ltm_results = self.ltm_store.search_by_embedding(
                    ltm_query,
                    k=k
                )
                results.extend(ltm_results)
        
        # Sort by similarity score
        results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return results[:k]

    def search_by_content(
        self,
        content_query: Union[str, Dict[str, Any]],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for memories based on content text/attributes.
        
        Args:
            content_query: String or dict to search for in memory contents
            k: Number of results to return
            
        Returns:
            List of memory entries matching the content query
        """
        results = []
        
        # Convert string query to dict if needed
        if isinstance(content_query, str):
            content_query = {"text": content_query}
        
        # Search each tier
        stm_results = self.stm_store.search_by_content(content_query, k)
        results.extend(stm_results)
        
        if len(results) < k:
            remaining = k - len(results)
            im_results = self.im_store.search_by_content(content_query, remaining)
            results.extend(im_results)
        
        if len(results) < k:
            remaining = k - len(results)
            ltm_results = self.ltm_store.search_by_content(content_query, remaining)
            results.extend(ltm_results)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:k]

    def register_hook(
        self,
        event_type: str,
        hook_function: callable,
        priority: int = 5
    ) -> bool:
        """Register a hook function for memory formation events.
        
        Args:
            event_type: Type of event to hook into
            hook_function: Function to call when event is triggered
            priority: Priority level (1-10, 10 being highest)
            
        Returns:
            True if hook was registered successfully
        """
        if not hasattr(self, "_event_hooks"):
            self._event_hooks = {}
        
        if event_type not in self._event_hooks:
            self._event_hooks[event_type] = []
        
        self._event_hooks[event_type].append({
            "function": hook_function,
            "priority": priority
        })
        
        # Sort hooks by priority (highest first)
        self._event_hooks[event_type].sort(
            key=lambda x: x["priority"],
            reverse=True
        )
        
        return True

    def trigger_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """Trigger memory formation event hooks.
        
        Args:
            event_type: Type of event that occurred
            event_data: Data related to the event
            
        Returns:
            True if event was processed successfully
        """
        if not hasattr(self, "_event_hooks") or event_type not in self._event_hooks:
            return False
        
        success = True
        event_data["timestamp"] = time.time()
        
        for hook in self._event_hooks[event_type]:
            try:
                result = hook["function"](event_data, self)
                
                # Process hook results
                if isinstance(result, dict):
                    if result.get("store_memory", False):
                        self.store_state(
                            result.get("memory_data", event_data),
                            result.get("step_number", 0),
                            result.get("priority", 1.0)
                        )
            except Exception as e:
                logger.error("Hook execution failed: %s", e)
                success = False
        
        return success

    def _calculate_tier_importance(self, tier: str) -> float:
        """Calculate average importance score for a memory tier."""
        store = getattr(self, f"{tier}_store")
        memories = store.get_all()
        if not memories:
            return 0.0
        
        total_importance = sum(
            memory["metadata"]["importance_score"]
            for memory in memories
        )
        return total_importance / len(memories)

    def _calculate_compression_ratio(self, tier: str) -> float:
        """Calculate compression ratio for a memory tier."""
        store = getattr(self, f"{tier}_store")
        compressed_size = store.get_size()
        if compressed_size == 0:
            return 0.0
        
        # Get original size from metadata
        memories = store.get_all()
        if not memories:
            return 0.0
        
        original_size = sum(
            memory["metadata"].get("original_size", compressed_size)
            for memory in memories
        )
        
        return original_size / compressed_size if compressed_size > 0 else 0.0

    def _get_memory_type_distribution(self) -> Dict[str, int]:
        """Get distribution of memory types across all tiers."""
        distribution = {}
        
        for store in [self.stm_store, self.im_store, self.ltm_store]:
            memories = store.get_all()
            for memory in memories:
                memory_type = memory["metadata"]["memory_type"]
                distribution[memory_type] = distribution.get(memory_type, 0) + 1
        
        return distribution

    def _get_access_patterns(self) -> Dict[str, Any]:
        """Get statistics about memory access patterns."""
        patterns = {
            "most_accessed": [],
            "least_accessed": [],
            "avg_accesses": 0,
            "total_accesses": 0
        }
        
        all_memories = []
        for store in [self.stm_store, self.im_store, self.ltm_store]:
            all_memories.extend(store.get_all())
        
        if not all_memories:
            return patterns
        
        # Calculate access statistics
        access_counts = [
            (memory["metadata"].get("retrieval_count", 0), memory)
            for memory in all_memories
        ]
        
        patterns["total_accesses"] = sum(count for count, _ in access_counts)
        patterns["avg_accesses"] = patterns["total_accesses"] / len(all_memories)
        
        # Get most and least accessed
        access_counts.sort(key=lambda x: x[0], reverse=True)
        patterns["most_accessed"] = [m for _, m in access_counts[:5]]
        patterns["least_accessed"] = [m for _, m in access_counts[-5:]]
        
        return patterns