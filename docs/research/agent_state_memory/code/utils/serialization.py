"""Serialization utilities for the agent memory system.

This module provides functions for serializing and deserializing memory
entries and related data structures for storage and transmission.
"""

import json
import logging
import base64
import pickle
import datetime
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class MemorySerializer:
    """Serialize and deserialize memory entries and related objects.
    
    This class provides methods for converting memory entries and their
    components to and from various serialized formats.
    """
    
    @staticmethod
    def serialize_memory(memory_entry: Dict[str, Any], format: str = "json") -> str:
        """Serialize a memory entry to a string.
        
        Args:
            memory_entry: Memory entry to serialize
            format: Format to use ("json" or "pickle")
            
        Returns:
            Serialized string representation
            
        Raises:
            ValueError: If serialization fails
        """
        if format == "json":
            return MemorySerializer.to_json(memory_entry)
        elif format == "pickle":
            return MemorySerializer.to_pickle(memory_entry)
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
    
    @staticmethod
    def deserialize_memory(data: str, format: str = "json") -> Dict[str, Any]:
        """Deserialize a memory entry from a string.
        
        Args:
            data: Serialized memory entry
            format: Format to use ("json" or "pickle")
            
        Returns:
            Deserialized memory entry
            
        Raises:
            ValueError: If deserialization fails
        """
        if format == "json":
            return MemorySerializer.from_json(data)
        elif format == "pickle":
            return MemorySerializer.from_pickle(data)
        else:
            raise ValueError(f"Unsupported deserialization format: {format}")
    
    @staticmethod
    def to_json(obj: Any) -> str:
        """Convert an object to a JSON string with special type handling.
        
        This method handles special types like datetime objects that are
        not natively serializable to JSON.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON string representation
            
        Raises:
            ValueError: If serialization fails
        """
        try:
            return json.dumps(obj, cls=MemoryJSONEncoder)
        except Exception as e:
            logger.error("Failed to serialize to JSON: %s", str(e))
            raise ValueError(f"JSON serialization failed: {str(e)}")
    
    @staticmethod
    def from_json(json_str: str) -> Any:
        """Convert a JSON string to an object with special type handling.
        
        This method handles special types like datetime objects that are
        not natively deserializable from JSON.
        
        Args:
            json_str: JSON string to deserialize
            
        Returns:
            Deserialized object
            
        Raises:
            ValueError: If deserialization fails
        """
        try:
            return json.loads(json_str, object_hook=memory_json_decoder)
        except Exception as e:
            logger.error("Failed to deserialize from JSON: %s", str(e))
            raise ValueError(f"JSON deserialization failed: {str(e)}")
    
    @staticmethod
    def to_pickle(obj: Any) -> str:
        """Convert an object to a base64-encoded pickle string.
        
        This method serializes an object using Python's pickle protocol
        and encodes the result as a base64 string for storage.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Base64-encoded pickle string
            
        Raises:
            ValueError: If serialization fails
        """
        try:
            pickle_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return base64.b64encode(pickle_bytes).decode('ascii')
        except Exception as e:
            logger.error("Failed to serialize to pickle: %s", str(e))
            raise ValueError(f"Pickle serialization failed: {str(e)}")
    
    @staticmethod
    def from_pickle(pickle_str: str) -> Any:
        """Convert a base64-encoded pickle string to an object.
        
        This method decodes a base64 string and deserializes it using
        Python's pickle protocol.
        
        Args:
            pickle_str: Base64-encoded pickle string
            
        Returns:
            Deserialized object
            
        Raises:
            ValueError: If deserialization fails
        """
        try:
            pickle_bytes = base64.b64decode(pickle_str.encode('ascii'))
            return pickle.loads(pickle_bytes)
        except Exception as e:
            logger.error("Failed to deserialize from pickle: %s", str(e))
            raise ValueError(f"Pickle deserialization failed: {str(e)}")
    
    @staticmethod
    def serialize_vector(vector: List[float]) -> str:
        """Serialize a vector to a string.
        
        This method creates a compact string representation of a vector
        for efficient storage.
        
        Args:
            vector: List of float values representing a vector
            
        Returns:
            String representation of the vector
            
        Raises:
            ValueError: If serialization fails
        """
        try:
            return json.dumps(vector)
        except Exception as e:
            logger.error("Failed to serialize vector: %s", str(e))
            raise ValueError(f"Vector serialization failed: {str(e)}")
    
    @staticmethod
    def deserialize_vector(vector_str: str) -> List[float]:
        """Deserialize a vector from a string.
        
        This method converts a string representation back to a vector.
        
        Args:
            vector_str: String representation of a vector
            
        Returns:
            List of float values representing a vector
            
        Raises:
            ValueError: If deserialization fails
        """
        try:
            return json.loads(vector_str)
        except Exception as e:
            logger.error("Failed to deserialize vector: %s", str(e))
            raise ValueError(f"Vector deserialization failed: {str(e)}")


class MemoryJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for memory entries.
    
    This encoder handles special types like datetime objects and
    numpy arrays that are not natively serializable to JSON.
    """
    
    def default(self, obj):
        """Convert special types to JSON-serializable objects.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable representation of the object
        """
        # Handle datetime objects
        if isinstance(obj, datetime.datetime):
            return {
                "__type__": "datetime",
                "value": obj.isoformat()
            }
        
        # Handle sets
        if isinstance(obj, set):
            return {
                "__type__": "set",
                "value": list(obj)
            }
        
        # Handle bytes
        if isinstance(obj, bytes):
            return {
                "__type__": "bytes",
                "value": base64.b64encode(obj).decode('ascii')
            }
        
        # Handle numpy arrays if available
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return {
                    "__type__": "ndarray",
                    "value": obj.tolist()
                }
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
        except ImportError:
            pass
        
        # Let the base class handle it or raise TypeError
        return super().default(obj)


def memory_json_decoder(obj):
    """Custom JSON decoder for memory entries.
    
    This decoder handles special types like datetime objects that
    were encoded by MemoryJSONEncoder.
    
    Args:
        obj: Object to decode
        
    Returns:
        Decoded object
    """
    if "__type__" in obj:
        obj_type = obj["__type__"]
        
        # Handle datetime objects
        if obj_type == "datetime":
            return datetime.datetime.fromisoformat(obj["value"])
        
        # Handle sets
        if obj_type == "set":
            return set(obj["value"])
        
        # Handle bytes
        if obj_type == "bytes":
            return base64.b64decode(obj["value"].encode('ascii'))
        
        # Handle numpy arrays if available
        if obj_type == "ndarray":
            try:
                import numpy as np
                return np.array(obj["value"])
            except ImportError:
                # Return as list if numpy is not available
                return obj["value"]
    
    # Return the object unchanged
    return obj


# Shorthand functions for external use
serialize_memory = MemorySerializer.serialize_memory
deserialize_memory = MemorySerializer.deserialize_memory
to_json = MemorySerializer.to_json
from_json = MemorySerializer.from_json 