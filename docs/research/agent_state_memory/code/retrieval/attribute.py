"""Attribute-based memory retrieval mechanisms.

This module provides methods for retrieving memories based on specific
attributes, metadata, and content patterns.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from ..storage.redis_stm import RedisSTMStore
from ..storage.redis_im import RedisIMStore
from ..storage.sqlite_ltm import SQLiteLTMStore

logger = logging.getLogger(__name__)


class AttributeRetrieval:
    """Retrieval mechanisms based on memory attributes.
    
    This class provides methods for retrieving memories based on specific
    attributes, metadata values, and content patterns.
    
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
        """Initialize the attribute retrieval.
        
        Args:
            stm_store: Short-Term Memory store
            im_store: Intermediate Memory store
            ltm_store: Long-Term Memory store
        """
        self.stm_store = stm_store
        self.im_store = im_store
        self.ltm_store = ltm_store
    
    def retrieve_by_memory_type(
        self, 
        memory_type: str,
        limit: int = 10,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories of a specific type.
        
        Args:
            memory_type: Type of memory (e.g., "state", "action", "interaction")
            limit: Maximum number of memories to retrieve
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories of the specified type
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get memories by type
        return store.get_by_type(memory_type, limit=limit)
    
    def retrieve_by_importance(
        self, 
        min_importance: float = 0.7,
        limit: int = 10,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with importance score above threshold.
        
        Args:
            min_importance: Minimum importance score (0.0-1.0)
            limit: Maximum number of memories to retrieve
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of important memories
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get important memories
        return store.get_by_importance(min_importance, limit=limit)
    
    def retrieve_by_metadata(
        self, 
        metadata_filters: Dict[str, Any],
        limit: int = 10,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories matching metadata filters.
        
        Args:
            metadata_filters: Dictionary of metadata key-value pairs to match
            limit: Maximum number of memories to retrieve
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories matching the metadata filters
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get memories by metadata
        memories = store.get_all()
        
        # Filter by metadata constraints
        filtered_memories = []
        for memory in memories:
            if self._matches_metadata_filters(memory, metadata_filters):
                filtered_memories.append(memory)
                if len(filtered_memories) >= limit:
                    break
        
        return filtered_memories
    
    def retrieve_by_content_value(
        self, 
        path: str,
        value: Any,
        limit: int = 10,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with a specific value at a content path.
        
        Args:
            path: Dot-notation path to the content field (e.g., "location.name")
            value: Value to match at the specified path
            limit: Maximum number of memories to retrieve
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories with the specified content value
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get all memories (this is inefficient but needed without direct index)
        memories = store.get_all()
        
        # Filter by content path value
        filtered_memories = []
        for memory in memories:
            if "contents" not in memory:
                continue
            
            # Get the value at the specified path
            path_value = self._get_value_at_path(memory["contents"], path)
            
            # Check if value matches
            if path_value == value:
                filtered_memories.append(memory)
                if len(filtered_memories) >= limit:
                    break
        
        return filtered_memories
    
    def retrieve_by_content_pattern(
        self, 
        path: str,
        pattern: str,
        limit: int = 10,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with content matching a regex pattern.
        
        Args:
            path: Dot-notation path to the content field (e.g., "dialog.text")
            pattern: Regex pattern to match against the field value
            limit: Maximum number of memories to retrieve
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories with content matching the pattern
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Compile regex pattern
        try:
            regex = re.compile(pattern)
        except re.error as e:
            logger.error("Invalid regex pattern: %s", str(e))
            return []
        
        # Get all memories
        memories = store.get_all()
        
        # Filter by content pattern
        filtered_memories = []
        for memory in memories:
            if "contents" not in memory:
                continue
            
            # Get the value at the specified path
            path_value = self._get_value_at_path(memory["contents"], path)
            
            # Check if value matches pattern
            if isinstance(path_value, str) and regex.search(path_value):
                filtered_memories.append(memory)
                if len(filtered_memories) >= limit:
                    break
        
        return filtered_memories
    
    def retrieve_by_custom_filter(
        self, 
        filter_fn: Callable[[Dict[str, Any]], bool],
        limit: int = 10,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories using a custom filter function.
        
        Args:
            filter_fn: Function that takes a memory entry and returns a boolean
            limit: Maximum number of memories to retrieve
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories matching the filter function
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get all memories
        memories = store.get_all()
        
        # Apply custom filter
        filtered_memories = []
        for memory in memories:
            try:
                if filter_fn(memory):
                    filtered_memories.append(memory)
                    if len(filtered_memories) >= limit:
                        break
            except Exception as e:
                logger.warning("Filter function error for memory: %s", str(e))
        
        return filtered_memories
    
    def retrieve_by_tag(
        self, 
        tag: str,
        limit: int = 10,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with a specific tag.
        
        Args:
            tag: Tag to search for
            limit: Maximum number of memories to retrieve
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories with the specified tag
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get memories by tag
        return store.get_by_tag(tag, limit=limit)
    
    def retrieve_by_compound_query(
        self, 
        conditions: List[Dict[str, Any]],
        operator: str = "AND",
        limit: int = 10,
        tier: str = "stm"
    ) -> List[Dict[str, Any]]:
        """Retrieve memories matching a compound query with multiple conditions.
        
        Args:
            conditions: List of condition dictionaries
            operator: "AND" or "OR" to determine how conditions are combined
            limit: Maximum number of memories to retrieve
            tier: Memory tier to search ("stm", "im", or "ltm")
            
        Returns:
            List of memories matching the compound query
        """
        # Select the appropriate store
        store = self._get_store_for_tier(tier)
        
        # Get all memories
        memories = store.get_all()
        
        # Filter by compound conditions
        filtered_memories = []
        for memory in memories:
            # Check all conditions against this memory
            matches = []
            for condition in conditions:
                if condition.get("type") == "metadata":
                    # Metadata condition
                    meta_key = condition.get("key")
                    meta_value = condition.get("value")
                    meta_actual = memory.get("metadata", {}).get(meta_key)
                    matches.append(meta_actual == meta_value)
                
                elif condition.get("type") == "content":
                    # Content condition
                    content_path = condition.get("path")
                    content_value = condition.get("value")
                    content_actual = self._get_value_at_path(
                        memory.get("contents", {}), 
                        content_path
                    )
                    matches.append(content_actual == content_value)
                
                elif condition.get("type") == "pattern":
                    # Pattern condition
                    pattern_path = condition.get("path")
                    pattern = condition.get("pattern")
                    try:
                        regex = re.compile(pattern)
                        pattern_actual = self._get_value_at_path(
                            memory.get("contents", {}), 
                            pattern_path
                        )
                        if isinstance(pattern_actual, str):
                            matches.append(bool(regex.search(pattern_actual)))
                        else:
                            matches.append(False)
                    except:
                        matches.append(False)
            
            # Determine if memory matches based on operator
            memory_matches = False
            if operator.upper() == "AND" and all(matches):
                memory_matches = True
            elif operator.upper() == "OR" and any(matches):
                memory_matches = True
            
            if memory_matches:
                filtered_memories.append(memory)
                if len(filtered_memories) >= limit:
                    break
        
        return filtered_memories
    
    def _matches_metadata_filters(self, memory: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a memory matches all metadata filters.
        
        Args:
            memory: Memory entry to check
            filters: Dictionary of metadata key-value pairs to match
            
        Returns:
            True if the memory matches all filters
        """
        metadata = memory.get("metadata", {})
        
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        
        return True
    
    def _get_value_at_path(self, obj: Dict[str, Any], path: str) -> Any:
        """Get a value from a nested dictionary using dot notation path.
        
        Args:
            obj: Dictionary to navigate
            path: Dot-notation path (e.g., "location.name")
            
        Returns:
            Value at the specified path or None if not found
        """
        parts = path.split(".")
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
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