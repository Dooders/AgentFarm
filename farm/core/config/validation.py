"""
Configuration validation system for hierarchical configurations.

This module provides comprehensive validation capabilities for configuration
values, including type checking, constraint validation, and schema validation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable

from .exceptions import ValidationException
from .hierarchical import HierarchicalConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation.
    
    Contains validation results including errors, warnings, and overall validity.
    """
    
    errors: List[str] = None
    warnings: List[str] = None
    is_valid: bool = True
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        
        # Update validity based on errors
        self.is_valid = len(self.errors) == 0
    
    def add_error(self, error: str) -> None:
        """Add a validation error.
        
        Args:
            error: Error message to add
        """
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a validation warning.
        
        Args:
            warning: Warning message to add
        """
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another validation result into this one.
        
        Args:
            other: Another ValidationResult to merge
            
        Returns:
            New ValidationResult with merged errors and warnings.
        """
        return ValidationResult(
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            is_valid=self.is_valid and other.is_valid
        )


class FieldValidator:
    """Validator for individual configuration fields."""
    
    def __init__(self, field_name: str, rules: Dict[str, Any]):
        """Initialize field validator.
        
        Args:
            field_name: Name of the field to validate
            rules: Validation rules for the field
        """
        self.field_name = field_name
        self.rules = rules
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate a field value against its rules.
        
        Args:
            value: Value to validate
            
        Returns:
            ValidationResult containing validation errors and warnings.
        """
        result = ValidationResult()
        
        # Check if field is required
        if self.rules.get('required', False) and value is None:
            result.add_error(f"Required field '{self.field_name}' is missing")
            return result
        
        # Skip validation if value is None and not required
        if value is None:
            return result
        
        # Validate type
        expected_type = self.rules.get('type')
        if expected_type and not isinstance(value, expected_type):
            result.add_error(
                f"Field '{self.field_name}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            return result
        
        # Validate constraints
        self._validate_constraints(value, result)
        
        # Validate custom validators
        self._validate_custom(value, result)
        
        return result
    
    def _validate_constraints(self, value: Any, result: ValidationResult) -> None:
        """Validate field constraints.
        
        Args:
            value: Value to validate
            result: ValidationResult to add errors/warnings to
        """
        # Numeric constraints
        if isinstance(value, (int, float)):
            min_val = self.rules.get('min')
            max_val = self.rules.get('max')
            
            if min_val is not None and value < min_val:
                result.add_error(
                    f"Field '{self.field_name}' must be >= {min_val}, got {value}"
                )
            
            if max_val is not None and value > max_val:
                result.add_error(
                    f"Field '{self.field_name}' must be <= {max_val}, got {value}"
                )
        
        # String constraints
        if isinstance(value, str):
            min_length = self.rules.get('min_length')
            max_length = self.rules.get('max_length')
            pattern = self.rules.get('pattern')
            
            if min_length is not None and len(value) < min_length:
                result.add_error(
                    f"Field '{self.field_name}' must be at least {min_length} characters, "
                    f"got {len(value)}"
                )
            
            if max_length is not None and len(value) > max_length:
                result.add_error(
                    f"Field '{self.field_name}' must be at most {max_length} characters, "
                    f"got {len(value)}"
                )
            
            if pattern and not self._matches_pattern(value, pattern):
                result.add_error(
                    f"Field '{self.field_name}' must match pattern '{pattern}'"
                )
        
        # List/array constraints
        if isinstance(value, (list, tuple)):
            min_items = self.rules.get('min_items')
            max_items = self.rules.get('max_items')
            item_type = self.rules.get('item_type')
            
            if min_items is not None and len(value) < min_items:
                result.add_error(
                    f"Field '{self.field_name}' must have at least {min_items} items, "
                    f"got {len(value)}"
                )
            
            if max_items is not None and len(value) > max_items:
                result.add_error(
                    f"Field '{self.field_name}' must have at most {max_items} items, "
                    f"got {len(value)}"
                )
            
            if item_type:
                for i, item in enumerate(value):
                    if not isinstance(item, item_type):
                        result.add_error(
                            f"Field '{self.field_name}[{i}]' must be of type "
                            f"{item_type.__name__}, got {type(item).__name__}"
                        )
        
        # Enum/choice constraints
        choices = self.rules.get('choices')
        if choices and value not in choices:
            result.add_error(
                f"Field '{self.field_name}' must be one of {choices}, got {value}"
            )
    
    def _validate_custom(self, value: Any, result: ValidationResult) -> None:
        """Validate using custom validator functions.
        
        Args:
            value: Value to validate
            result: ValidationResult to add errors/warnings to
        """
        custom_validators = self.rules.get('validators', [])
        
        for validator in custom_validators:
            try:
                if isinstance(validator, dict):
                    # Dictionary-based validator
                    func = validator.get('function')
                    message = validator.get('message', 'Custom validation failed')
                    
                    if callable(func) and not func(value):
                        result.add_error(f"Field '{self.field_name}': {message}")
                elif callable(validator):
                    # Direct function validator
                    if not validator(value):
                        result.add_error(f"Field '{self.field_name}': Custom validation failed")
            except Exception as e:
                result.add_error(
                    f"Field '{self.field_name}': Custom validator error: {str(e)}"
                )
    
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Check if string matches regex pattern.
        
        Args:
            value: String to check
            pattern: Regex pattern to match
            
        Returns:
            True if string matches pattern, False otherwise.
        """
        import re
        try:
            return bool(re.match(pattern, value))
        except re.error:
            logger.warning(f"Invalid regex pattern '{pattern}' for field '{self.field_name}'")
            return False


class ConfigurationValidator:
    """Runtime configuration validation at startup.
    
    This class provides comprehensive validation of hierarchical configurations
    against defined schemas and constraints.
    """
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """Initialize configuration validator.
        
        Args:
            schema: Validation schema defining field rules and constraints
        """
        self.schema = schema or {}
        self.field_validators = self._build_field_validators()
    
    def _build_field_validators(self) -> Dict[str, FieldValidator]:
        """Build field validators from schema.
        
        Returns:
            Dictionary mapping field names to their validators.
        """
        validators = {}
        
        fields = self.schema.get('fields', {})
        for field_name, rules in fields.items():
            validators[field_name] = FieldValidator(field_name, rules)
        
        return validators
    
    def validate_config(self, config: HierarchicalConfig) -> ValidationResult:
        """Validate configuration against schema.
        
        Args:
            config: HierarchicalConfig to validate
            
        Returns:
            ValidationResult containing validation errors and warnings.
        """
        result = ValidationResult()
        
        # Validate required fields
        self._validate_required_fields(config, result)
        
        # Validate field values
        self._validate_field_values(config, result)
        
        # Validate dependencies
        self._validate_dependencies(config, result)
        
        # Validate cross-field constraints
        self._validate_cross_field_constraints(config, result)
        
        logger.debug(f"Configuration validation completed: {len(result.errors)} errors, "
                    f"{len(result.warnings)} warnings")
        
        return result
    
    def _validate_required_fields(self, config: HierarchicalConfig, result: ValidationResult) -> None:
        """Validate that all required fields are present.
        
        Args:
            config: Configuration to validate
            result: ValidationResult to add errors to
        """
        required_fields = self.schema.get('required', [])
        
        for field in required_fields:
            if not config.has(field):
                result.add_error(f"Required field '{field}' is missing")
    
    def _validate_field_values(self, config: HierarchicalConfig, result: ValidationResult) -> None:
        """Validate individual field values.
        
        Args:
            config: Configuration to validate
            result: ValidationResult to add errors/warnings to
        """
        for field_name, validator in self.field_validators.items():
            value = config.get(field_name)
            field_result = validator.validate(value)
            
            # Merge field validation results
            result.errors.extend(field_result.errors)
            result.warnings.extend(field_result.warnings)
    
    def _validate_dependencies(self, config: HierarchicalConfig, result: ValidationResult) -> None:
        """Validate field dependencies.
        
        Args:
            config: Configuration to validate
            result: ValidationResult to add errors to
        """
        dependencies = self.schema.get('dependencies', {})
        
        for field, deps in dependencies.items():
            field_value = config.get(field)
            
            if field_value is not None:
                for dep_field, dep_condition in deps.items():
                    if isinstance(dep_condition, dict):
                        # Complex dependency condition
                        dep_value = config.get(dep_field)
                        if dep_value is not None and not self._evaluate_dependency_condition(dep_value, dep_condition):
                            result.add_error(
                                f"Field '{field}' requires '{dep_field}' to satisfy condition: {dep_condition}"
                            )
                    else:
                        # Simple dependency (field must exist)
                        if not config.has(dep_field):
                            result.add_error(
                                f"Field '{field}' requires field '{dep_field}' to be present"
                            )
    
    def _validate_cross_field_constraints(self, config: HierarchicalConfig, result: ValidationResult) -> None:
        """Validate constraints that involve multiple fields.
        
        Args:
            config: Configuration to validate
            result: ValidationResult to add errors to
        """
        cross_constraints = self.schema.get('cross_constraints', [])
        
        for constraint in cross_constraints:
            try:
                if not self._evaluate_cross_constraint(config, constraint):
                    result.add_error(constraint.get('message', 'Cross-field constraint violated'))
            except Exception as e:
                result.add_error(f"Error evaluating cross-constraint: {str(e)}")
    
    def _evaluate_dependency_condition(self, value: Any, condition: Dict[str, Any]) -> bool:
        """Evaluate a dependency condition.
        
        Args:
            value: Value to check against condition
            condition: Condition to evaluate
            
        Returns:
            True if condition is satisfied, False otherwise.
        """
        if 'equals' in condition:
            return value == condition['equals']
        elif 'not_equals' in condition:
            return value != condition['not_equals']
        elif 'in' in condition:
            return value in condition['in']
        elif 'not_in' in condition:
            return value not in condition['not_in']
        elif 'greater_than' in condition:
            return value > condition['greater_than']
        elif 'less_than' in condition:
            return value < condition['less_than']
        elif 'greater_equal' in condition:
            return value >= condition['greater_equal']
        elif 'less_equal' in condition:
            return value <= condition['less_equal']
        
        return True
    
    def _evaluate_cross_constraint(self, config: HierarchicalConfig, constraint: Dict[str, Any]) -> bool:
        """Evaluate a cross-field constraint.
        
        Args:
            config: Configuration to validate
            constraint: Constraint definition
            
        Returns:
            True if constraint is satisfied, False otherwise.
        """
        fields = constraint.get('fields', [])
        condition = constraint.get('condition')
        
        if not condition or not fields:
            return True
        
        # Get values for all fields
        values = [config.get(field) for field in fields]
        
        # Evaluate condition
        if condition == 'all_present':
            return all(v is not None for v in values)
        elif condition == 'any_present':
            return any(v is not None for v in values)
        elif condition == 'sum_equals':
            target = constraint.get('target')
            return sum(v for v in values if isinstance(v, (int, float))) == target
        elif condition == 'custom':
            func = constraint.get('function')
            return callable(func) and func(*values)
        
        return True
    
    def add_field_validator(self, field_name: str, rules: Dict[str, Any]) -> None:
        """Add or update a field validator.
        
        Args:
            field_name: Name of the field to validate
            rules: Validation rules for the field
        """
        self.field_validators[field_name] = FieldValidator(field_name, rules)
        logger.debug(f"Added validator for field '{field_name}'")
    
    def remove_field_validator(self, field_name: str) -> None:
        """Remove a field validator.
        
        Args:
            field_name: Name of the field to remove validator for
        """
        if field_name in self.field_validators:
            del self.field_validators[field_name]
            logger.debug(f"Removed validator for field '{field_name}'")
    
    def get_validation_schema(self) -> Dict[str, Any]:
        """Get the current validation schema.
        
        Returns:
            Dictionary containing the validation schema.
        """
        return self.schema.copy()
    
    def set_validation_schema(self, schema: Dict[str, Any]) -> None:
        """Set a new validation schema.
        
        Args:
            schema: New validation schema
        """
        self.schema = schema
        self.field_validators = self._build_field_validators()
        logger.debug("Updated validation schema")


# Predefined validation schemas for common configuration types
DEFAULT_SIMULATION_SCHEMA = {
    'required': [
        'simulation_id',
        'max_steps',
        'environment',
        'width',
        'height'
    ],
    'fields': {
        'simulation_id': {
            'type': str,
            'required': True,
            'min_length': 1,
            'max_length': 100
        },
        'max_steps': {
            'type': int,
            'required': True,
            'min': 1,
            'max': 1000000
        },
        'width': {
            'type': int,
            'required': True,
            'min': 10,
            'max': 1000
        },
        'height': {
            'type': int,
            'required': True,
            'min': 10,
            'max': 1000
        },
        'learning_rate': {
            'type': float,
            'min': 0.0001,
            'max': 1.0
        },
        'gamma': {
            'type': float,
            'min': 0.0,
            'max': 1.0
        },
        'epsilon_start': {
            'type': float,
            'min': 0.0,
            'max': 1.0
        },
        'epsilon_min': {
            'type': float,
            'min': 0.0,
            'max': 1.0
        },
        'batch_size': {
            'type': int,
            'min': 1,
            'max': 10000
        },
        'memory_size': {
            'type': int,
            'min': 100,
            'max': 1000000
        }
    },
    'cross_constraints': [
        {
            'fields': ['width', 'height'],
            'condition': 'all_present',
            'message': 'Both width and height must be specified'
        },
        {
            'fields': ['epsilon_min', 'epsilon_start'],
            'condition': 'custom',
            'function': lambda min_val, start_val: min_val <= start_val,
            'message': 'epsilon_min must be less than or equal to epsilon_start'
        }
    ]
}