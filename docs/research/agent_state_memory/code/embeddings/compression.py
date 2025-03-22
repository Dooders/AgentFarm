"""Compression utilities for agent memory entries at different resolution levels.

This module provides functionality for compressing memory entries as they
move through the memory hierarchy, reducing detail while preserving
essential information.
"""

import copy
import json
import logging
import zlib
from typing import Dict, Any, Set

from ..config import AutoencoderConfig

logger = logging.getLogger(__name__)


class CompressionEngine:
    """Engine for compressing memory entries at different resolution levels.
    
    This class provides methods to compress memory entries as they move
    from Short-Term Memory (STM) to Intermediate Memory (IM) to Long-Term
    Memory (LTM), with increasing levels of compression.
    
    Attributes:
        config: Configuration for compression settings
    """
    
    def __init__(self, config: AutoencoderConfig):
        """Initialize the compression engine.
        
        Args:
            config: Configuration for the autoencoder and compression
        """
        self.config = config
        
        # Initialize compression level configurations
        self.level_configs = {
            # Level 0: No compression (STM)
            0: {
                "attribute_precision": None,  # Full precision
                "content_filter_keys": set(),  # Keep all keys
                "binary_compression": False,   # No binary compression
            },
            # Level 1: Moderate compression (IM)
            1: {
                "attribute_precision": 3,  # Numeric precision (decimal places)
                "content_filter_keys": {   # Keys to exclude from content
                    "raw_observation", "full_observation", "detailed_state",
                    "debug_info", "internal_state", "temporary_data"
                },
                "binary_compression": False,  # No binary compression
            },
            # Level 2: High compression (LTM)
            2: {
                "attribute_precision": 1,  # Low numeric precision
                "content_filter_keys": {   # Only keep essential keys
                    "raw_observation", "full_observation", "detailed_state", 
                    "debug_info", "internal_state", "temporary_data",
                    "intermediate_results", "processing_steps", "trace",
                    "interaction_details", "sensory_data", "raw_action_data"
                },
                "binary_compression": True,  # Apply binary compression
            }
        }
    
    def compress(
        self, 
        memory_entry: Dict[str, Any], 
        level: int = 0
    ) -> Dict[str, Any]:
        """Compress a memory entry to the specified level.
        
        Args:
            memory_entry: Memory entry to compress
            level: Compression level (0=none, 1=moderate, 2=high)
            
        Returns:
            Compressed memory entry
        """
        # If level is 0, return a deep copy of the original entry
        if level == 0:
            return copy.deepcopy(memory_entry)
        
        # Get compression config for the requested level
        if level not in self.level_configs:
            logger.warning("Unknown compression level %d, using level 1", level)
            level = 1
        
        level_config = self.level_configs[level]
        
        # Start with a copy of the original entry
        compressed_entry = copy.deepcopy(memory_entry)
        
        # Apply content filtering (remove non-essential keys)
        if "contents" in compressed_entry and level_config["content_filter_keys"]:
            self._filter_content_keys(
                compressed_entry["contents"], 
                level_config["content_filter_keys"]
            )
        
        # Apply numeric precision reduction
        if level_config["attribute_precision"] is not None:
            self._reduce_numeric_precision(
                compressed_entry, 
                level_config["attribute_precision"]
            )
        
        # Update metadata
        if "metadata" in compressed_entry:
            compressed_entry["metadata"]["compression_level"] = level
        
        # Apply binary compression if enabled
        if level_config["binary_compression"]:
            if "contents" in compressed_entry:
                compressed_entry["contents"] = self._binary_compress(compressed_entry["contents"])
        
        return compressed_entry
    
    def decompress(self, memory_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to decompress a memory entry.
        
        This performs partial decompression, recovering what information
        is available after compression. Note that some information loss
        is inevitable at higher compression levels.
        
        Args:
            memory_entry: Compressed memory entry
            
        Returns:
            Decompressed memory entry
        """
        # Make a copy to avoid modifying the original
        decompressed_entry = copy.deepcopy(memory_entry)
        
        # Check if binary compression was applied
        if "metadata" in decompressed_entry and decompressed_entry["metadata"].get("compression_level", 0) > 1:
            if "contents" in decompressed_entry and isinstance(decompressed_entry["contents"], str):
                try:
                    decompressed_entry["contents"] = self._binary_decompress(decompressed_entry["contents"])
                except Exception as e:
                    logger.error("Failed to decompress memory contents: %s", str(e))
        
        return decompressed_entry
    
    def _filter_content_keys(self, content: Dict[str, Any], filter_keys: Set[str]) -> None:
        """Remove non-essential keys from content dictionary.
        
        Args:
            content: Content dictionary to filter
            filter_keys: Set of keys to remove
        """
        if not isinstance(content, dict):
            return
        
        # Get keys to remove (cannot modify during iteration)
        keys_to_remove = [key for key in content if key in filter_keys]
        
        # Remove keys
        for key in keys_to_remove:
            del content[key]
        
        # Recursively filter nested dictionaries
        for key, value in content.items():
            if isinstance(value, dict):
                self._filter_content_keys(value, filter_keys)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._filter_content_keys(item, filter_keys)
    
    def _reduce_numeric_precision(self, obj: Any, precision: int) -> None:
        """Recursively reduce precision of numeric values.
        
        Args:
            obj: Object to process
            precision: Number of decimal places to keep
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if isinstance(value, float):
                        obj[key] = round(value, precision)
                else:
                    self._reduce_numeric_precision(value, precision)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if isinstance(value, float):
                        obj[i] = round(value, precision)
                else:
                    self._reduce_numeric_precision(value, precision)
    
    def _binary_compress(self, obj: Any) -> str:
        """Compress an object using binary compression.
        
        Args:
            obj: Object to compress
            
        Returns:
            Base64-encoded compressed string
        """
        import base64
        
        # Convert to JSON
        json_str = json.dumps(obj)
        
        # Compress with zlib
        compressed = zlib.compress(json_str.encode('utf-8'), level=9)
        
        # Encode as base64
        b64_str = base64.b64encode(compressed).decode('ascii')
        
        return b64_str
    
    def _binary_decompress(self, compressed_str: str) -> Any:
        """Decompress a binary compressed string.
        
        Args:
            compressed_str: Base64-encoded compressed string
            
        Returns:
            Decompressed object
        """
        import base64
        
        # Decode base64
        compressed = base64.b64decode(compressed_str)
        
        # Decompress with zlib
        json_str = zlib.decompress(compressed).decode('utf-8')
        
        # Parse JSON
        return json.loads(json_str)


class AbstractionEngine:
    """Engine for creating abstract representations of memory content.
    
    This class provides methods to extract key information and create
    summarized versions of memory entries, preserving semantic meaning
    while reducing detail.
    """
    
    def __init__(self):
        """Initialize the abstraction engine."""
        pass
    
    def abstract_memory(
        self, 
        memory_entry: Dict[str, Any], 
        level: int = 1
    ) -> Dict[str, Any]:
        """Create an abstracted version of a memory entry.
        
        Args:
            memory_entry: Memory entry to abstract
            level: Abstraction level (1=light, 2=heavy)
            
        Returns:
            Abstracted memory entry
        """
        # Make a copy to avoid modifying the original
        abstract_entry = copy.deepcopy(memory_entry)
        
        # Extract key fields
        if "contents" in abstract_entry and isinstance(abstract_entry["contents"], dict):
            if level == 1:
                abstract_entry["contents"] = self._light_abstraction(abstract_entry["contents"])
            else:
                abstract_entry["contents"] = self._heavy_abstraction(abstract_entry["contents"])
        
        # Update metadata
        if "metadata" in abstract_entry:
            abstract_entry["metadata"]["abstraction_level"] = level
        
        return abstract_entry
    
    def _light_abstraction(self, contents: Dict[str, Any]) -> Dict[str, Any]:
        """Create a lightly abstracted version of memory contents.
        
        This preserves most semantic content but removes details.
        
        Args:
            contents: Memory contents to abstract
            
        Returns:
            Abstracted contents
        """
        # Extract a subset of key fields that represent core information
        abstract = {}
        
        # Extract key aspects based on content type
        if "memory_type" in contents:
            abstract["memory_type"] = contents["memory_type"]
        
        # For state memories
        if contents.get("memory_type") == "state":
            # Extract key state attributes
            for key in ["location", "status", "goals", "resources", "relationships"]:
                if key in contents:
                    abstract[key] = contents[key]
        
        # For action memories
        elif contents.get("memory_type") == "action":
            # Extract key action attributes
            for key in ["action_type", "target", "outcome", "success"]:
                if key in contents:
                    abstract[key] = contents[key]
        
        # For interaction memories
        elif contents.get("memory_type") == "interaction":
            # Extract key interaction attributes
            for key in ["interaction_type", "entities", "dialog", "outcome"]:
                if key in contents:
                    abstract[key] = contents[key]
        
        # Copy common fields relevant for all types
        for key in ["timestamp", "step_number", "agent_id", "importance"]:
            if key in contents:
                abstract[key] = contents[key]
        
        return abstract
    
    def _heavy_abstraction(self, contents: Dict[str, Any]) -> Dict[str, Any]:
        """Create a heavily abstracted version of memory contents.
        
        This preserves only the most essential semantic content.
        
        Args:
            contents: Memory contents to abstract
            
        Returns:
            Heavily abstracted contents
        """
        # Start with light abstraction
        abstract = self._light_abstraction(contents)
        
        # Further reduce content based on type
        if "memory_type" in abstract:
            if abstract["memory_type"] == "state":
                # Keep only the most essential state information
                essential_keys = ["location", "status"]
                abstract = {k: v for k, v in abstract.items() if k in essential_keys or k == "memory_type"}
            
            elif abstract["memory_type"] == "action":
                # Keep only the most essential action information
                essential_keys = ["action_type", "outcome"]
                abstract = {k: v for k, v in abstract.items() if k in essential_keys or k == "memory_type"}
            
            elif abstract["memory_type"] == "interaction":
                # Keep only the most essential interaction information
                essential_keys = ["interaction_type", "outcome"]
                abstract = {k: v for k, v in abstract.items() if k in essential_keys or k == "memory_type"}
        
        # Always keep timestamp and step info
        for key in ["timestamp", "step_number"]:
            if key in contents:
                abstract[key] = contents[key]
        
        return abstract 