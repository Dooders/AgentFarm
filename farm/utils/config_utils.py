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