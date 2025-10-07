"""
Configuration validation and error handling for production readiness.

This module provides comprehensive validation, error handling, and recovery
mechanisms for the configuration system.
"""

import os
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple
from .config import SimulationConfig


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.config_path = config_path
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ConfigurationError):
    """Raised when configuration validation fails."""


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration files are missing."""


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration data is invalid or malformed."""


class ConfigurationValidator:
    """
    Comprehensive configuration validator with schema validation and business rule checking.
    """

    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []

    def validate_config(
        self, config: SimulationConfig, strict: bool = False
    ) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate a configuration comprehensively.

        Args:
            config: Configuration to validate
            strict: Whether to treat warnings as errors

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Run all validation checks
        self._validate_basic_properties(config)
        self._validate_agent_settings(config)
        self._validate_resource_settings(config)
        self._validate_learning_parameters(config)
        self._validate_environment_settings(config)
        self._validate_performance_settings(config)
        self._validate_business_rules(config)

        # Determine if valid
        is_valid = len(self.errors) == 0
        if strict and self.warnings:
            is_valid = False
            self.errors.extend(self.warnings)
            self.warnings = []

        return is_valid, self.errors.copy(), self.warnings.copy()

    def _validate_basic_properties(self, config: SimulationConfig) -> None:
        """Validate basic configuration properties."""
        # Environment dimensions
        if config.environment.width <= 0 or config.environment.height <= 0:
            self.errors.append(
                {
                    "field": "width/height",
                    "error": "Environment dimensions must be positive",
                    "value": f"{config.environment.width}x{config.environment.height}",
                }
            )

        if config.environment.width > 10000 or config.environment.height > 10000:
            self.warnings.append(
                {
                    "field": "width/height",
                    "warning": "Very large environment dimensions may impact performance",
                    "value": f"{config.environment.width}x{config.environment.height}",
                }
            )

        # Population limits
        if config.population.max_population <= 0:
            self.errors.append(
                {
                    "field": "max_population",
                    "error": "Maximum population must be positive",
                    "value": config.population.max_population,
                }
            )

        if config.population.max_population > 100000:
            self.warnings.append(
                {
                    "field": "max_population",
                    "warning": "Very large population may impact performance",
                    "value": config.population.max_population,
                }
            )

    def _validate_agent_settings(self, config: SimulationConfig) -> None:
        """Validate agent-related settings."""
        total_agents = (
            config.population.system_agents + config.population.independent_agents + config.population.control_agents
        )

        if total_agents == 0:
            self.errors.append(
                {
                    "field": "agent_counts",
                    "error": "At least one agent type must be configured",
                    "value": f"system={config.population.system_agents}, independent={config.population.independent_agents}, control={config.population.control_agents}",
                }
            )

        if total_agents > config.population.max_population:
            self.warnings.append(
                {
                    "field": "agent_counts",
                    "warning": "Total initial agents exceeds max population limit",
                    "value": f"total={total_agents}, max={config.population.max_population}",
                }
            )

        # Agent type ratios
        total_ratio = sum(config.population.agent_type_ratios.values())
        if abs(total_ratio - 1.0) > 0.001:  # Allow small floating point errors
            self.errors.append(
                {
                    "field": "agent_type_ratios",
                    "error": "Agent type ratios must sum to 1.0",
                    "value": f"sum={total_ratio:.3f}",
                }
            )

        # Individual ratios
        for agent_type, ratio in config.population.agent_type_ratios.items():
            if ratio < 0.0 or ratio > 1.0:
                self.errors.append(
                    {
                        "field": f"agent_type_ratios.{agent_type}",
                        "error": "Agent type ratios must be between 0.0 and 1.0",
                        "value": ratio,
                    }
                )

    def _validate_resource_settings(self, config: SimulationConfig) -> None:
        """Validate resource-related settings."""
        if config.resources.initial_resources < 0:
            self.errors.append(
                {
                    "field": "initial_resources",
                    "error": "Initial resources cannot be negative",
                    "value": config.resources.initial_resources,
                }
            )

        if config.resources.max_resource_amount <= 0:
            self.errors.append(
                {
                    "field": "max_resource_amount",
                    "error": "Maximum resource amount must be positive",
                    "value": config.resources.max_resource_amount,
                }
            )

        if config.resources.resource_regen_rate < 0.0 or config.resources.resource_regen_rate > 1.0:
            self.errors.append(
                {
                    "field": "resource_regen_rate",
                    "error": "Resource regeneration rate must be between 0.0 and 1.0",
                    "value": config.resources.resource_regen_rate,
                }
            )

        if config.resources.resource_regen_amount < 0:
            self.errors.append(
                {
                    "field": "resource_regen_amount",
                    "error": "Resource regeneration amount cannot be negative",
                    "value": config.resources.resource_regen_amount,
                }
            )

    def _validate_learning_parameters(self, config: SimulationConfig) -> None:
        """Validate reinforcement learning parameters."""
        # Learning rate
        if config.learning.learning_rate <= 0.0 or config.learning.learning_rate > 1.0:
            self.errors.append(
                {
                    "field": "learning_rate",
                    "error": "Learning rate must be between 0.0 and 1.0",
                    "value": config.learning.learning_rate,
                }
            )

        # Discount factor
        if config.learning.gamma < 0.0 or config.learning.gamma > 1.0:
            self.errors.append(
                {
                    "field": "gamma",
                    "error": "Discount factor (gamma) must be between 0.0 and 1.0",
                    "value": config.learning.gamma,
                }
            )

        # Epsilon parameters
        for param_name in ["epsilon_start", "epsilon_min", "epsilon_decay"]:
            param_value = getattr(config.learning, param_name)
            if param_value < 0.0 or param_value > 1.0:
                self.errors.append(
                    {
                        "field": param_name,
                        "error": f"{param_name} must be between 0.0 and 1.0",
                        "value": param_value,
                    }
                )

        if config.learning.epsilon_start < config.learning.epsilon_min:
            self.warnings.append(
                {
                    "field": "epsilon_start/epsilon_min",
                    "warning": "Epsilon start should typically be >= epsilon min",
                    "value": f"start={config.learning.epsilon_start}, min={config.learning.epsilon_min}",
                }
            )

        # Memory size
        if config.learning.memory_size <= 0:
            self.errors.append(
                {
                    "field": "memory_size",
                    "error": "Memory size must be positive",
                    "value": config.learning.memory_size,
                }
            )

        # Batch size
        if config.learning.batch_size <= 0:
            self.errors.append(
                {
                    "field": "batch_size",
                    "error": "Batch size must be positive",
                    "value": config.learning.batch_size,
                }
            )

        if config.learning.batch_size > config.learning.memory_size:
            self.warnings.append(
                {
                    "field": "batch_size/memory_size",
                    "warning": "Batch size should not exceed memory size",
                    "value": f"batch={config.learning.batch_size}, memory={config.learning.memory_size}",
                }
            )

    def _validate_environment_settings(self, config: SimulationConfig) -> None:
        """Validate environment and physics settings."""
        # Perception radius
        if config.agent_behavior.perception_radius <= 0:
            self.errors.append(
                {
                    "field": "perception_radius",
                    "error": "Perception radius must be positive",
                    "value": config.agent_behavior.perception_radius,
                }
            )

        if (
            config.agent_behavior.perception_radius > config.environment.width
            or config.agent_behavior.perception_radius > config.environment.height
        ):
            self.warnings.append(
                {
                    "field": "perception_radius",
                    "warning": "Perception radius larger than environment may impact performance",
                    "value": f"radius={config.agent_behavior.perception_radius}, env={config.environment.width}x{config.environment.height}",
                }
            )

        # Movement and gathering ranges
        for param_name in ["max_movement", "gathering_range", "territory_range"]:
            param_value = getattr(config.agent_behavior, param_name)
            if param_value <= 0:
                self.errors.append(
                    {
                        "field": param_name,
                        "error": f"{param_name} must be positive",
                        "value": param_value,
                    }
                )

        # Consumption rate
        if config.agent_behavior.base_consumption_rate <= 0.0 or config.agent_behavior.base_consumption_rate > 1.0:
            self.errors.append(
                {
                    "field": "base_consumption_rate",
                    "error": "Base consumption rate must be between 0.0 and 1.0",
                    "value": config.agent_behavior.base_consumption_rate,
                }
            )

    def _validate_performance_settings(self, config: SimulationConfig) -> None:
        """Validate performance and database settings."""
        # Database settings
        if config.database.db_cache_size_mb <= 0:
            self.errors.append(
                {
                    "field": "db_cache_size_mb",
                    "error": "Database cache size must be positive",
                    "value": config.database.db_cache_size_mb,
                }
            )

        if config.database.db_cache_size_mb > 10000:  # 10GB limit
            self.warnings.append(
                {
                    "field": "db_cache_size_mb",
                    "warning": "Very large database cache size may impact system performance",
                    "value": config.database.db_cache_size_mb,
                }
            )

        # In-memory database settings
        if config.database.use_in_memory_db and config.database.in_memory_db_memory_limit_mb is not None:
            if config.database.in_memory_db_memory_limit_mb <= 0:
                self.errors.append(
                    {
                        "field": "in_memory_db_memory_limit_mb",
                        "error": "In-memory database memory limit must be positive",
                        "value": config.database.in_memory_db_memory_limit_mb,
                    }
                )

    def _validate_business_rules(self, config: SimulationConfig) -> None:
        """Validate business rules and consistency checks."""
        # Reproduction settings
        if config.agent_behavior.min_reproduction_resources <= 0:
            self.errors.append(
                {
                    "field": "min_reproduction_resources",
                    "error": "Minimum reproduction resources must be positive",
                    "value": config.agent_behavior.min_reproduction_resources,
                }
            )

        if config.agent_behavior.offspring_cost <= 0:
            self.errors.append(
                {
                    "field": "offspring_cost",
                    "error": "Offspring cost must be positive",
                    "value": config.agent_behavior.offspring_cost,
                }
            )

        if config.agent_behavior.min_reproduction_resources < config.agent_behavior.offspring_cost:
            self.warnings.append(
                {
                    "field": "min_reproduction_resources/offspring_cost",
                    "warning": "Minimum reproduction resources should typically exceed offspring cost",
                    "value": f"min={config.agent_behavior.min_reproduction_resources}, cost={config.agent_behavior.offspring_cost}",
                }
            )

        # Starvation threshold
        if config.agent_behavior.starvation_threshold <= 0:
            self.errors.append(
                {
                    "field": "starvation_threshold",
                    "error": "Starvation threshold must be positive",
                    "value": config.agent_behavior.starvation_threshold,
                }
            )

        # Simulation steps
        if config.max_steps <= 0:
            self.errors.append(
                {
                    "field": "max_steps",
                    "error": "Maximum steps must be positive",
                    "value": config.max_steps,
                }
            )

        if config.simulation_steps > config.max_steps:
            self.warnings.append(
                {
                    "field": "simulation_steps/max_steps",
                    "warning": "Simulation steps should not exceed maximum steps",
                    "value": f"simulation={config.simulation_steps}, max={config.max_steps}",
                }
            )


class ConfigurationRecovery:
    """
    Configuration recovery and fallback mechanisms.
    """

    @staticmethod
    def create_fallback_config(error_details: Dict[str, Any]) -> SimulationConfig:
        """
        Create a minimal working configuration as fallback.

        Args:
            error_details: Information about the configuration error

        Returns:
            Minimal working configuration
        """
        from .config import (
            EnvironmentConfig,
            PopulationConfig,
            ResourceConfig,
            LearningConfig,
            VisualizationConfig,
            RedisMemoryConfig,
            DatabaseConfig,
        )

        # Create a minimal working configuration using nested config objects
        return SimulationConfig(
            environment=EnvironmentConfig(width=50, height=50),
            population=PopulationConfig(
                system_agents=5,
                independent_agents=5,
                control_agents=5,
                max_population=50,
            ),
            resources=ResourceConfig(
                initial_resources=20,
                max_resource_amount=30,
            ),
            learning=LearningConfig(
                learning_rate=0.001,
                gamma=0.95,
                epsilon_start=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                memory_size=1000,
                batch_size=32,
            ),
            max_steps=100,
            database=DatabaseConfig(use_in_memory_db=True),
            visualization=VisualizationConfig(),
            redis=RedisMemoryConfig(),
        )

    @staticmethod
    def attempt_config_repair(
        config: SimulationConfig, errors: List[Dict[str, Any]]
    ) -> Tuple[SimulationConfig, List[str]]:
        """
        Attempt to automatically repair configuration errors.

        Args:
            config: Configuration with errors
            errors: List of validation errors

        Returns:
            Tuple of (repaired_config, repair_actions_taken)
        """
        repaired_config = config.copy()
        repair_actions = []

        for error in errors:
            field = error["field"]

            # Apply automatic fixes for common issues
            if field in ["width", "height"] and error.get("value", 0) <= 0:
                setattr(repaired_config, field, 50)
                repair_actions.append(f"Set {field} to 50 (was {error.get('value', 0)})")

            elif field == "max_population" and error.get("value", 0) <= 0:
                repaired_config.max_population = 100
                repair_actions.append("Set max_population to 100")

            elif field == "learning_rate":
                if error.get("value", 0) <= 0:
                    repaired_config.learning_rate = 0.001
                    repair_actions.append("Set learning_rate to 0.001")
                elif error.get("value", 0) > 1.0:
                    repaired_config.learning_rate = 0.1
                    repair_actions.append("Set learning_rate to 0.1")

            elif field in ["gamma", "epsilon_start", "epsilon_min", "epsilon_decay"]:
                value = error.get("value", 0)
                if value < 0.0 or value > 1.0:
                    setattr(repaired_config, field, max(0.0, min(1.0, value)))
                    repair_actions.append(f"Clamped {field} to [0.0, 1.0] (was {value})")

            elif field == "memory_size" and error.get("value", 0) <= 0:
                repaired_config.memory_size = 1000
                repair_actions.append("Set memory_size to 1000")

            elif field == "batch_size" and error.get("value", 0) <= 0:
                repaired_config.batch_size = 32
                repair_actions.append("Set batch_size to 32")

        return repaired_config, repair_actions


class SafeConfigLoader:
    """
    Safe configuration loader with validation and automatic recovery.
    """

    def __init__(self, validator: Optional[ConfigurationValidator] = None):
        """
        Initialize safe loader.

        Args:
            validator: Configuration validator to use
        """
        self.validator = validator or ConfigurationValidator()

    def validate_config_dict(
        self,
        config_dict: Dict[str, Any],
        strict_validation: bool = False,
        auto_repair: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validate a configuration dict with comprehensive error handling and recovery.

        Args:
            config_dict: Configuration dictionary to validate
            strict_validation: Whether to treat warnings as errors
            auto_repair: Whether to attempt automatic repair of errors

        Returns:
            Tuple of (validated_config_dict, status_info)

        Raises:
            ValidationError: If validation fails and auto_repair is disabled
        """
        status_info = {
            "success": False,
            "errors": [],
            "warnings": [],
            "repair_actions": [],
            "fallback_used": False,
            "validation_method": "dict",
        }

        try:
            # Convert dict to SimulationConfig for validation
            from .config import SimulationConfig

            config = SimulationConfig.from_dict(config_dict)

            # Validate configuration
            is_valid, errors, warnings = self.validator.validate_config(config, strict=strict_validation)

            status_info["errors"] = errors
            status_info["warnings"] = warnings

            validated_config_dict = config_dict  # Start with original

            if not is_valid and auto_repair:
                # Attempt to repair configuration
                repaired_config, repair_actions = ConfigurationRecovery.attempt_config_repair(config, errors)
                status_info["repair_actions"] = repair_actions

                # Re-validate repaired configuration
                is_valid, errors, warnings = self.validator.validate_config(repaired_config, strict=strict_validation)

                if is_valid:
                    validated_config_dict = repaired_config.to_dict()
                    status_info["errors"] = errors
                    status_info["warnings"] = warnings
                else:
                    # Repair failed, use fallback
                    fallback_config = ConfigurationRecovery.create_fallback_config(
                        {"original_errors": errors, "config_dict": config_dict}
                    )
                    validated_config_dict = fallback_config.to_dict()
                    status_info["fallback_used"] = True
                    status_info["errors"] = errors

            if not is_valid and not auto_repair:
                raise ValidationError(
                    f"Configuration validation failed: {len(errors)} errors",
                    details={"errors": errors, "warnings": warnings},
                )

            status_info["success"] = True
            return validated_config_dict, status_info

        except Exception as e:
            # Unexpected error during validation
            fallback_config = ConfigurationRecovery.create_fallback_config(
                {
                    "error": "validation_error",
                    "exception": str(e),
                    "config_dict": config_dict,
                }
            )
            validated_config_dict = fallback_config.to_dict()
            status_info["fallback_used"] = True
            status_info["errors"] = [{"error": f"Validation error: {e}"}]

            if strict_validation:
                raise ValidationError(
                    f"Failed to validate configuration: {e}",
                    details={"original_error": str(e), "config_dict": config_dict},
                ) from e

            return validated_config_dict, status_info


def validate_config_files(config_dir: str = "config") -> Dict[str, Any]:
    """
    Validate all configuration files in a directory.

    Args:
        config_dir: Configuration directory to validate

    Returns:
        Validation report
    """
    validator = ConfigurationValidator()
    report = {
        "config_dir": config_dir,
        "files_checked": [],
        "total_errors": 0,
        "total_warnings": 0,
        "file_reports": {},
    }

    # Check base configuration
    base_path = os.path.join(config_dir, "default.yaml")
    if os.path.exists(base_path):
        try:
            config = SimulationConfig.from_yaml(base_path)
            is_valid, errors, warnings = validator.validate_config(config)
            report["files_checked"].append("default.yaml")
            report["file_reports"]["default.yaml"] = {
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
            }
            report["total_errors"] += len(errors)
            report["total_warnings"] += len(warnings)
        except Exception as e:
            report["file_reports"]["default.yaml"] = {
                "valid": False,
                "load_error": str(e),
            }
            report["total_errors"] += 1

    # Check environment configurations
    env_dir = os.path.join(config_dir, "environments")
    if os.path.exists(env_dir):
        for filename in os.listdir(env_dir):
            if filename.endswith(".yaml"):
                filepath = os.path.join(env_dir, filename)
                try:
                    config = SimulationConfig.from_yaml(filepath)
                    is_valid, errors, warnings = validator.validate_config(config)
                    report["files_checked"].append(f"environments/{filename}")
                    report["file_reports"][f"environments/{filename}"] = {
                        "valid": is_valid,
                        "errors": errors,
                        "warnings": warnings,
                    }
                    report["total_errors"] += len(errors)
                    report["total_warnings"] += len(warnings)
                except Exception as e:
                    report["file_reports"][f"environments/{filename}"] = {
                        "valid": False,
                        "load_error": str(e),
                    }
                    report["total_errors"] += 1

    # Check profile configurations
    profile_dir = os.path.join(config_dir, "profiles")
    if os.path.exists(profile_dir):
        for filename in os.listdir(profile_dir):
            if filename.endswith(".yaml"):
                filepath = os.path.join(profile_dir, filename)
                try:
                    config = SimulationConfig.from_yaml(filepath)
                    is_valid, errors, warnings = validator.validate_config(config)
                    report["files_checked"].append(f"profiles/{filename}")
                    report["file_reports"][f"profiles/{filename}"] = {
                        "valid": is_valid,
                        "errors": errors,
                        "warnings": warnings,
                    }
                    report["total_errors"] += len(errors)
                    report["total_warnings"] += len(warnings)
                except Exception as e:
                    report["file_reports"][f"profiles/{filename}"] = {
                        "valid": False,
                        "load_error": str(e),
                    }
                    report["total_errors"] += 1

    return report
