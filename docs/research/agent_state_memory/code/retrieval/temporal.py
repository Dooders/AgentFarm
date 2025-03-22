"""Temporal memory retrieval mechanisms.

This module provides methods for retrieving memories based on temporal
characteristics such as recency, specific time periods, and step ranges.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from ..storage.redis_stm import RedisSTMStore
from ..storage.redis_im import RedisIMStore
from ..storage.sqlite_ltm import SQLiteLTMStore

logger = logging.getLogger(__name__)


class TemporalRetrieval:
    """Retrieval mechanisms based on temporal characteristics.
    
    This class provides methods for retrieving memories based on
    time-related attributes like recency, specific time periods,
    or simulation step numbers.
    
    Attributes:
        stm_store: Short-Term Memory store
        im_store: Intermediate Memory store
        ltm_store: Long-Term Memory store
    """
    
    def __init__(
        self,
        stm_store: RedisSTMStore,
        im_store: RedisIMStore,
        ltm_store: SQLiteLTMStore,
    ):
        """Initialize the temporal retrieval.
        
        Args:
            stm_store: Short-Term Memory store
            im_store: Intermediate Memory store
            ltm_store: Long-Term Memory store
        """
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store
    
    def retrieve_recent(
        self, 
        count: int = 10,
        memory_type: Optional[str] = None,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve the most recent memories.
        
        Args:
            count: Maximum number of memories to retrieve
            memory_type: Optional filter for memory type
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of recent memories
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get recent memories
        memories = store.get_recent(count=count*2)  # Get extra to account for filtering
        
        # Filter by memory type if specified
        if memory_type and memories:
            memories = [
                m for m in memories 
                if m.get("metadata", {}).get("memory_type") == memory_type
            ]
        
        # Limit to requested count
        return memories[:count]
    
    def retrieve_by_step(
        self, 
        step: int,
        tier: str = "stm"
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a memory for a specific step.
        
        Args:
            step: Simulation step number
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            Memory for the specified step or None if not found
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get memory by step
        return store.get_by_step(step)
    
    def retrieve_step_range(
        self, 
        start_step: int,
        end_step: int,
        memory_type: Optional[str] = None,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories within a range of steps.
        
        Args:
            start_step: Starting step number (inclusive)
            end_step: Ending step number (inclusive)
            memory_type: Optional filter for memory type
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories within the step range
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get memories in step range
        memories = store.get_step_range(start_step, end_step)
        
        # Filter by memory type if specified
        if memory_type and memories:
            memories = [
                m for m in memories 
                if m.get("metadata", {}).get("memory_type") == memory_type
            ]
        
        return memories
    
    def retrieve_time_range(
        self, 
        start_time: int,
        end_time: int,
        memory_type: Optional[str] = None,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories within a time range.
        
        Args:
            start_time: Starting Unix timestamp (inclusive)
            end_time: Ending Unix timestamp (inclusive)
            memory_type: Optional filter for memory type
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories within the time range
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get memories in time range
        memories = store.get_time_range(start_time, end_time)
        
        # Filter by memory type if specified
        if memory_type and memories:
            memories = [
                m for m in memories 
                if m.get("metadata", {}).get("memory_type") == memory_type
            ]
        
        return memories
    
    def retrieve_last_n_minutes(
        self, 
        minutes: int = 60,
        memory_type: Optional[str] = None,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories from the last N minutes.
        
        Args:
            minutes: Number of minutes to look back
            memory_type: Optional filter for memory type
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories from the specified time period
        """
        current_time = int(time.time())
        start_time = current_time - (minutes * 60)
        
        return self.retrieve_time_range(
            start_time=start_time,
            end_time=current_time,
            memory_type=memory_type,
            tier=tier
        )
    
    def retrieve_oldest(
        self, 
        count: int = 10,
        memory_type: Optional[str] = None,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve the oldest memories.
        
        Args:
            count: Maximum number of memories to retrieve
            memory_type: Optional filter for memory type
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of oldest memories
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get oldest memories
        memories = store.get_oldest(count=count*2)  # Get extra to account for filtering
        
        # Filter by memory type if specified
        if memory_type and memories:
            memories = [
                m for m in memories 
                if m.get("metadata", {}).get("memory_type") == memory_type
            ]
        
        # Limit to requested count
        return memories[:count]
    
    def retrieve_narrative_sequence(
        self, 
        memory_id: str,
        context_before: int = 3,
        context_after: int = 3,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve a sequence of memories forming a narrative context.
        
        This retrieves memories before and after a central memory to
        provide context for understanding a sequence of events.
        
        Args:
            memory_id: ID of the central memory
            context_before: Number of preceding memories to include
            context_after: Number of following memories to include
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories forming the narrative sequence
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get the central memory
        central_memory = store.get(memory_id)
        if not central_memory:
            logger.warning("Central memory %s not found in %s", memory_id, tier)
            return []
        
        # Get the step number
        step_number = central_memory.get("step_number")
        if step_number is None:
            logger.warning("Central memory %s has no step number", memory_id)
            return []
        
        # Get memories before
        before_memories = store.get_step_range(
            step_number - context_before,
            step_number - 1
        )
        
        # Get memories after
        after_memories = store.get_step_range(
            step_number + 1,
            step_number + context_after
        )
        
        # Combine in sequence
        sequence = before_memories + [central_memory] + after_memories
        
        # Sort by step number to ensure correct order
        sequence.sort(key=lambda m: m.get("step_number", 0))
        
        return sequence
    
    def _get_store_for_tier(self, tier: str):
        """Get the appropriate store for the specified tier.
        
        Args:
            tier: Memory tier ("stm", "im", or "ltm")
            
        Returns:
            Memory store for the tier
        """
        if tier == "im":
            return self.im_store
        elif tier == "ltm":
            return self.ltm_store
        else:  # Default to STM
            return self.stm_store 