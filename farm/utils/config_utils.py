def get_config_value(config, attr_name, default_value, expected_types=(int, float)):
    """Safely retrieve a configuration value with type validation.
    
    Args:
        config: Configuration object or None
        attr_name: Name of the attribute to retrieve
        default_value: Default value if attribute is missing or invalid
        expected_types: Tuple of acceptable types for the value
        
    Returns:
        The configuration value or default_value if invalid/missing
    """
    if not config:
        return default_value
        
    try:
        value = getattr(config, attr_name, default_value)
        if not isinstance(value, expected_types):
            return default_value
        return value
    except (AttributeError, TypeError):
        return default_value 


def get_nested_then_flat(
    *,
    config,
    nested_parent_attr: str | None,
    nested_attr_name: str | None,
    flat_attr_name: str | None,
    default_value,
    expected_types: tuple = (int, float, bool),
):
    """Retrieve a configuration value using nested-then-flat pattern.

    This helper reduces duplication when a value might be present under a nested
    config object (e.g., config.agent_behavior.perception_radius) or a legacy
    flat attribute (e.g., config.perception_radius).

    Resolution order:
    1) If nested_parent_attr exists on config and nested_attr_name exists on it,
       and the value is of an expected type, return it.
    2) Else if flat_attr_name exists on config and is of an expected type, return it.
    3) Else return default_value.

    Args:
        config: Root configuration object (may be None)
        nested_parent_attr: Name of the nested parent (e.g., "agent_behavior") or None
        nested_attr_name: Attribute under the nested parent (e.g., "perception_radius") or None
        flat_attr_name: Legacy flat attribute on root config (e.g., "perception_radius") or None
        default_value: Value to return if none resolved
        expected_types: Acceptable types for the value

    Returns:
        The resolved configuration value, or default_value if not present/invalid.
    """
    try:
        if config is not None and nested_parent_attr and nested_attr_name:
            parent = getattr(config, nested_parent_attr, None)
            if parent is not None and hasattr(parent, nested_attr_name):
                value = getattr(parent, nested_attr_name)
                if isinstance(value, expected_types):
                    return value
        if config is not None and flat_attr_name:
            value = getattr(config, flat_attr_name, None)
            if isinstance(value, expected_types):
                return value
    except Exception:
        # Fall through to default
        pass

    return default_value