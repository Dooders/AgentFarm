"""
Utility functions for the AgentFarm MCP Server.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
    
    Returns:
        Unique identifier string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_part = str(uuid.uuid4())[:8]
    
    if prefix:
        return f"{prefix}_{timestamp}_{unique_part}"
    else:
        return f"{timestamp}_{unique_part}"

def safe_json_loads(json_str: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON string, returning None on error.
    
    Args:
        json_str: JSON string to parse
    
    Returns:
        Parsed dictionary or None if parsing fails
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return None

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely serialize object to JSON string.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
    
    Returns:
        JSON string representation
    """
    try:
        return json.dumps(obj, default=str, **kwargs)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize to JSON: {e}")
        return json.dumps({"error": f"Serialization failed: {str(e)}"})

def format_error_response(tool_name: str, error_message: str, error_code: Optional[str] = None) -> str:
    """Format a standardized error response.
    
    Args:
        tool_name: Name of the tool that encountered the error
        error_message: Description of the error
        error_code: Optional error code for categorization
    
    Returns:
        JSON string with error details
    """
    error_response = {
        "error": True,
        "tool": tool_name,
        "message": error_message,
        "timestamp": datetime.now().isoformat()
    }
    
    if error_code:
        error_response["error_code"] = error_code
    
    return json.dumps(error_response, indent=2)

def format_success_response(tool_name: str, data: Dict[str, Any], message: Optional[str] = None) -> str:
    """Format a standardized success response.
    
    Args:
        tool_name: Name of the tool that succeeded
        data: Response data
        message: Optional success message
    
    Returns:
        JSON string with success details
    """
    success_response = {
        "success": True,
        "tool": tool_name,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    if message:
        success_response["message"] = message
    
    return safe_json_dumps(success_response, indent=2)

def validate_simulation_config(config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate simulation configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = []  # No strictly required fields for flexibility
    
    # Check for reasonable values
    validations = [
        ("width", lambda x: isinstance(x, int) and 10 <= x <= 1000, "Width must be between 10 and 1000"),
        ("height", lambda x: isinstance(x, int) and 10 <= x <= 1000, "Height must be between 10 and 1000"),
        ("simulation_steps", lambda x: isinstance(x, int) and 1 <= x <= 10000, "Steps must be between 1 and 10000"),
        ("system_agents", lambda x: isinstance(x, int) and x >= 0, "System agents must be non-negative"),
        ("independent_agents", lambda x: isinstance(x, int) and x >= 0, "Independent agents must be non-negative"),
        ("control_agents", lambda x: isinstance(x, int) and x >= 0, "Control agents must be non-negative"),
        ("initial_resources", lambda x: isinstance(x, int) and x >= 0, "Initial resources must be non-negative"),
    ]
    
    for field, validator, error_msg in validations:
        if field in config:
            if not validator(config[field]):
                return False, error_msg
    
    # Check total agents is reasonable
    total_agents = sum(config.get(f"{t}_agents", 0) for t in ["system", "independent", "control"])
    if total_agents > 1000:
        return False, "Total agents exceeds reasonable limit (1000)"
    
    if total_agents == 0:
        return False, "At least one agent must be specified"
    
    return True, None

def extract_experiment_id(result_json: str) -> Optional[str]:
    """Extract experiment ID from a tool result.
    
    Args:
        result_json: JSON result string from create_experiment
    
    Returns:
        Experiment ID if found, None otherwise
    """
    try:
        result_data = safe_json_loads(result_json)
        if result_data and "experiment_id" in result_data:
            return result_data["experiment_id"]
        
        # Fallback: try to extract from text
        import re
        match = re.search(r"exp_\d+_\d+_[a-f0-9]+", result_json)
        return match.group() if match else None
        
    except Exception as e:
        logger.warning(f"Failed to extract experiment ID: {e}")
        return None

def extract_simulation_id(result_json: str) -> Optional[str]:
    """Extract simulation ID from a tool result.
    
    Args:
        result_json: JSON result string from create_simulation
    
    Returns:
        Simulation ID if found, None otherwise
    """
    try:
        result_data = safe_json_loads(result_json)
        if result_data and "simulation_id" in result_data:
            return result_data["simulation_id"]
        
        # Fallback: try to extract from text
        import re
        match = re.search(r"sim_\d+_\d+_[a-f0-9]+", result_json)
        return match.group() if match else None
        
    except Exception as e:
        logger.warning(f"Failed to extract simulation ID: {e}")
        return None

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem usage.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    import re
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized or "unnamed"

def get_system_info() -> Dict[str, Any]:
    """Get system information for diagnostics.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_usage": psutil.disk_usage('.').free
    }