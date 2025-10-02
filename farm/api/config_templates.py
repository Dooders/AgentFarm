"""Configuration templates and validation for the unified AgentFarm API.

This module provides predefined configuration templates and validation
logic for simulations and experiments, making it easy for agentic systems
to create valid configurations.
"""

from typing import Any, Dict, List, Optional

from farm.api.models import ConfigCategory, ConfigTemplate, ValidationResult
from farm.config import SimulationConfig
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


class ConfigTemplateManager:
    """Manages configuration templates and validation."""

    def __init__(self):
        """Initialize the template manager."""
        self._templates = self._create_default_templates()

    def _create_default_templates(self) -> Dict[str, ConfigTemplate]:
        """Create default configuration templates."""
        templates = {}

        # Basic simulation template
        templates["basic_simulation"] = ConfigTemplate(
            name="basic_simulation",
            description="Basic simulation with default parameters",
            category=ConfigCategory.SIMULATION,
            parameters={
                "name": "Basic Simulation",
                "steps": 1000,
                "environment": {"width": 100, "height": 100, "resources": 50},
                "agents": {
                    "system_agents": 10,
                    "independent_agents": 10,
                    "control_agents": 0,
                },
                "learning": {"enabled": True, "algorithm": "dqn"},
            },
            required_fields=["name", "steps"],
            optional_fields=["environment", "agents", "learning"],
            examples=[
                {
                    "name": "Quick Test",
                    "steps": 100,
                    "agents": {"system_agents": 5, "independent_agents": 5},
                },
                {
                    "name": "Long Run",
                    "steps": 5000,
                    "environment": {"width": 200, "height": 200},
                },
            ],
        )

        # Combat simulation template
        templates["combat_simulation"] = ConfigTemplate(
            name="combat_simulation",
            description="Combat-focused simulation with fighting mechanics",
            category=ConfigCategory.SIMULATION,
            parameters={
                "name": "Combat Simulation",
                "steps": 2000,
                "environment": {"width": 150, "height": 150, "resources": 30},
                "agents": {
                    "system_agents": 15,
                    "independent_agents": 15,
                    "control_agents": 0,
                },
                "combat": {
                    "enabled": True,
                    "damage_multiplier": 1.0,
                    "defense_strength": 0.5,
                },
                "learning": {"enabled": True, "algorithm": "dqn"},
            },
            required_fields=["name", "steps"],
            optional_fields=["environment", "agents", "combat", "learning"],
            examples=[
                {
                    "name": "High Combat",
                    "steps": 1000,
                    "combat": {"damage_multiplier": 2.0},
                }
            ],
        )

        # Research simulation template
        templates["research_simulation"] = ConfigTemplate(
            name="research_simulation",
            description="Research-focused simulation with detailed logging",
            category=ConfigCategory.SIMULATION,
            parameters={
                "name": "Research Simulation",
                "steps": 5000,
                "environment": {"width": 200, "height": 200, "resources": 100},
                "agents": {
                    "system_agents": 25,
                    "independent_agents": 25,
                    "control_agents": 0,
                },
                "logging": {
                    "detailed_logging": True,
                    "save_observations": True,
                    "save_actions": True,
                },
                "learning": {"enabled": True, "algorithm": "ppo"},
            },
            required_fields=["name", "steps"],
            optional_fields=["environment", "agents", "logging", "learning"],
            examples=[
                {
                    "name": "Behavior Study",
                    "steps": 10000,
                    "logging": {"save_observations": True},
                }
            ],
        )

        # Basic experiment template
        templates["basic_experiment"] = ConfigTemplate(
            name="basic_experiment",
            description="Basic experiment with parameter variations",
            category=ConfigCategory.EXPERIMENT,
            parameters={
                "name": "Basic Experiment",
                "description": "Compare different agent populations",
                "iterations": 5,
                "base_config": {
                    "steps": 1000,
                    "environment": {"width": 100, "height": 100, "resources": 50},
                },
                "variations": [
                    {"agents": {"system_agents": 5, "independent_agents": 15}},
                    {"agents": {"system_agents": 10, "independent_agents": 10}},
                    {"agents": {"system_agents": 15, "independent_agents": 5}},
                ],
            },
            required_fields=["name", "iterations", "base_config"],
            optional_fields=["description", "variations"],
            examples=[
                {
                    "name": "Population Test",
                    "iterations": 3,
                    "base_config": {"steps": 500},
                }
            ],
        )

        # Parameter sweep experiment template
        templates["parameter_sweep"] = ConfigTemplate(
            name="parameter_sweep",
            description="Parameter sweep experiment with systematic variations",
            category=ConfigCategory.EXPERIMENT,
            parameters={
                "name": "Parameter Sweep",
                "description": "Systematic parameter variation study",
                "iterations": 10,
                "base_config": {
                    "steps": 2000,
                    "environment": {"width": 150, "height": 150, "resources": 75},
                    "agents": {"system_agents": 20, "independent_agents": 20},
                },
                "parameter_ranges": {
                    "learning_rate": [0.001, 0.01, 0.1],
                    "exploration_rate": [0.1, 0.3, 0.5],
                },
            },
            required_fields=["name", "iterations", "base_config"],
            optional_fields=["description", "parameter_ranges"],
            examples=[
                {
                    "name": "Learning Rate Study",
                    "iterations": 5,
                    "parameter_ranges": {"learning_rate": [0.001, 0.01]},
                }
            ],
        )

        return templates

    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """Get a configuration template by name.

        Args:
            name: Template name

        Returns:
            ConfigTemplate if found, None otherwise
        """
        return self._templates.get(name)

    def list_templates(
        self, category: Optional[ConfigCategory] = None
    ) -> List[ConfigTemplate]:
        """List available templates, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of ConfigTemplate objects
        """
        templates = list(self._templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    def create_config_from_template(
        self, template_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a configuration from a template.

        Args:
            template_name: Name of the template to use
            overrides: Optional parameter overrides

        Returns:
            Configuration dictionary if template found, None otherwise
        """
        template = self.get_template(template_name)
        if not template:
            return None

        # Start with template parameters
        config = template.parameters.copy()

        # Apply overrides
        if overrides:
            config.update(overrides)

        return config

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate a configuration.

        Args:
            config: Configuration to validate

        Returns:
            ValidationResult with validation status and messages
        """
        # Acquire logger at call time to support patching in tests
        local_logger = get_logger(__name__)
        local_logger.debug("Validating configuration", config_type=type(config).__name__)

        errors = []
        warnings = []
        suggestions = []

        # Handle None or invalid config types
        if config is None:
            errors.append("Configuration cannot be None")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
            )

        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
            )

        # Check required fields
        if "name" not in config:
            errors.append("Configuration must have a 'name' field")

        if "steps" not in config:
            errors.append("Configuration must have a 'steps' field")
        elif not isinstance(config["steps"], int) or config["steps"] <= 0:
            errors.append("'steps' must be a positive integer")

        # Validate environment settings
        if "environment" in config:
            env = config["environment"]
            if "width" in env and (
                not isinstance(env["width"], int) or env["width"] <= 0
            ):
                errors.append("Environment width must be a positive integer")
            if "height" in env and (
                not isinstance(env["height"], int) or env["height"] <= 0
            ):
                errors.append("Environment height must be a positive integer")
            if "resources" in env and (
                not isinstance(env["resources"], int) or env["resources"] < 0
            ):
                errors.append("Environment resources must be a non-negative integer")

        # Validate agent settings
        if "agents" in config:
            agents = config["agents"]
            total_agents = 0
            for agent_type in ["system_agents", "independent_agents", "control_agents"]:
                if agent_type in agents:
                    count = agents[agent_type]
                    if not isinstance(count, int) or count < 0:
                        errors.append(f"{agent_type} must be a non-negative integer")
                    else:
                        total_agents += count

            if total_agents == 0:
                warnings.append("No agents configured - simulation will be empty")
            elif total_agents > 100:
                warnings.append(
                    f"Large agent population ({total_agents}) may impact performance"
                )

        # Validate learning settings
        if "learning" in config:
            learning = config["learning"]
            if learning.get("enabled", False):
                algorithm = learning.get("algorithm", "dqn")
                valid_algorithms = ["dqn", "ppo", "sac", "a2c", "td3"]
                if algorithm not in valid_algorithms:
                    suggestions.append(
                        f"Consider using a supported algorithm: {valid_algorithms}"
                    )

        # Performance suggestions
        steps = config.get("steps", 0)
        if isinstance(steps, int) and steps > 10000:
            suggestions.append(
                "Consider using in-memory database for large simulations"
            )

        if (
            config.get("environment", {}).get("width", 100)
            * config.get("environment", {}).get("height", 100)
            > 40000
        ):
            suggestions.append("Large environment size may impact performance")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            validated_config=config if is_valid else None,
        )

    def convert_to_simulation_config(
        self, config: Dict[str, Any]
    ) -> Optional[SimulationConfig]:
        """Convert a configuration dictionary to a SimulationConfig object.

        Args:
            config: Configuration dictionary

        Returns:
            SimulationConfig object if conversion successful, None otherwise
        """
        try:
            # Validate the config first
            validation_result = self.validate_config(config)
            if not validation_result.is_valid:
                return None

            # Start with default config
            sim_config = SimulationConfig.from_centralized_config()

            # Apply configuration overrides
            if "steps" in config:
                sim_config.simulation_steps = config["steps"]

            if "environment" in config:
                env = config["environment"]
                if "width" in env:
                    sim_config.width = env["width"]
                if "height" in env:
                    sim_config.height = env["height"]
                if "resources" in env:
                    sim_config.initial_resources = env["resources"]

            if "agents" in config:
                agents = config["agents"]
                if "system_agents" in agents:
                    sim_config.population.system_agents = agents["system_agents"]
                if "independent_agents" in agents:
                    sim_config.population.independent_agents = agents[
                        "independent_agents"
                    ]
                if "control_agents" in agents:
                    sim_config.population.control_agents = agents["control_agents"]

            if "learning" in config:
                learning = config["learning"]
                if "enabled" in learning:
                    sim_config.learning.enabled = learning["enabled"]
                if "algorithm" in learning:
                    sim_config.learning.algorithm = learning["algorithm"]

            return sim_config

        except Exception as e:
            # Log via freshly acquired logger to support patching
            get_logger(__name__).error(f"Error converting config to SimulationConfig: {e}")
            return None

    def convert_to_experiment_config(
        self, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Convert a configuration dictionary to an experiment configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Experiment configuration dictionary if conversion successful, None otherwise
        """
        try:
            # Validate the config first
            validation_result = self.validate_config(config)
            if not validation_result.is_valid:
                return None

            # Convert to experiment format
            exp_config = {
                "name": config.get("name", "experiment"),
                "iterations": config.get("steps", 1000),
                "variations": config.get("variations", []),
                "environment": config.get("environment", {}),
                "agents": config.get("agents", {}),
                "learning": config.get("learning", {}),
            }

            return exp_config

        except Exception as e:
            get_logger(__name__).error(f"Error converting config to experiment config: {e}")
            return None

    def get_template_examples(
        self, template_name: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get examples for a specific template.

        Args:
            template_name: Name of the template

        Returns:
            List of example configurations, or None if template doesn't exist
        """
        template = self.get_template(template_name)
        if template:
            return template.examples
        return None

    def get_required_fields(self, template_name: str) -> Optional[List[str]]:
        """Get required fields for a specific template.

        Args:
            template_name: Name of the template

        Returns:
            List of required field names, or None if template doesn't exist
        """
        template = self.get_template(template_name)
        if template:
            return template.required_fields
        return None

    def get_optional_fields(self, template_name: str) -> Optional[List[str]]:
        """Get optional fields for a specific template.

        Args:
            template_name: Name of the template

        Returns:
            List of optional field names, or None if template doesn't exist
        """
        template = self.get_template(template_name)
        if template:
            return template.optional_fields
        return None

    def validate_against_template(
        self, template_name: str, config: Dict[str, Any]
    ) -> Optional[ValidationResult]:
        """Validate a configuration against a specific template.

        Args:
            template_name: Name of the template to validate against
            config: Configuration to validate

        Returns:
            ValidationResult if template exists, None otherwise
        """
        template = self.get_template(template_name)
        if not template:
            return None

        # Get basic validation result
        result = self.validate_config(config)

        # Check required fields specific to template
        for field in template.required_fields:
            if field not in config:
                result.errors.append(f"Required field '{field}' is missing")

        # Check for unknown fields
        allowed_fields = set(template.required_fields + template.optional_fields)
        for field in config.keys():
            if field not in allowed_fields:
                result.warnings.append(f"Unknown field '{field}' in configuration")

        result.is_valid = len(result.errors) == 0
        return result

    def get_template_categories(self) -> List[ConfigCategory]:
        """Get all available template categories.

        Returns:
            List of ConfigCategory values
        """
        return list(ConfigCategory)

    def list_templates_by_category(
        self, category: ConfigCategory
    ) -> List[ConfigTemplate]:
        """List templates by category.

        Args:
            category: Category to filter by

        Returns:
            List of templates in the category
        """
        templates = []
        for name, template in self._templates.items():
            if template.category == category:
                templates.append(template)
        return templates

    def add_template(self, template: ConfigTemplate) -> bool:
        """Add a custom template.

        Args:
            template: Template to add

        Returns:
            True if added successfully, False if template already exists
        """
        if template.name in self._templates:
            return False

        self._templates[template.name] = template
        return True

    def remove_template(self, template_name: str) -> bool:
        """Remove a custom template.

        Args:
            template_name: Name of template to remove

        Returns:
            True if removed successfully, False if template is protected or doesn't exist
        """
        if template_name not in self._templates:
            return False

        # Check if it's a default template (protected)
        if template_name in self._create_default_templates():
            return False

        del self._templates[template_name]
        return True
