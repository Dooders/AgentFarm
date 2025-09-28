"""
Configuration templates and generators system.

This module provides template-based configuration generation, validation
templates, and best practice configuration patterns for the hierarchical
configuration system.
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import yaml

from .exceptions import ConfigurationError
from .hierarchical import HierarchicalConfig

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of configuration templates."""
    
    BASE = "base"
    ENVIRONMENT = "environment"
    AGENT = "agent"
    VALIDATION = "validation"
    MIGRATION = "migration"
    BEST_PRACTICE = "best_practice"


class TemplateCategory(Enum):
    """Categories of configuration templates."""
    
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    DEMO = "demo"
    CUSTOM = "custom"


@dataclass
class TemplateMetadata:
    """Metadata for configuration templates."""
    
    name: str
    description: str
    version: str
    template_type: TemplateType
    category: TemplateCategory
    author: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_schema: Optional[Dict[str, Any]] = None


@dataclass
class TemplateParameter:
    """Parameter definition for template generation."""
    
    name: str
    description: str
    parameter_type: str
    default_value: Any = None
    required: bool = False
    validation_rules: Optional[Dict[str, Any]] = None
    choices: Optional[List[Any]] = None


class ConfigurationTemplate:
    """Base class for configuration templates."""
    
    def __init__(self, metadata: TemplateMetadata, template_content: Dict[str, Any]):
        """Initialize configuration template.
        
        Args:
            metadata: Template metadata
            template_content: Template content with placeholders
        """
        self.metadata = metadata
        self.template_content = template_content
        self.parameters: Dict[str, TemplateParameter] = {}
        self._parse_parameters()
    
    def _parse_parameters(self) -> None:
        """Parse parameters from template content."""
        # Extract parameters from template content
        self._extract_parameters_recursive(self.template_content)
    
    def _extract_parameters_recursive(self, content: Any, path: str = "") -> None:
        """Recursively extract parameters from template content.
        
        Args:
            content: Content to extract parameters from
            path: Current path in the content
        """
        if isinstance(content, dict):
            for key, value in content.items():
                current_path = f"{path}.{key}" if path else key
                self._extract_parameters_recursive(value, current_path)
        elif isinstance(content, list):
            for i, item in enumerate(content):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                self._extract_parameters_recursive(item, current_path)
        elif isinstance(content, str) and self._is_parameter(content):
            param_name = self._extract_parameter_name(content)
            if param_name not in self.parameters:
                self.parameters[param_name] = TemplateParameter(
                    name=param_name,
                    description=f"Parameter for {path}",
                    parameter_type="string"
                )
    
    def _is_parameter(self, value: str) -> bool:
        """Check if a string value is a parameter placeholder.
        
        Args:
            value: String value to check
            
        Returns:
            True if it's a parameter placeholder
        """
        return value.startswith("{{") and value.endswith("}}")
    
    def _extract_parameter_name(self, value: str) -> str:
        """Extract parameter name from placeholder.
        
        Args:
            value: Parameter placeholder string
            
        Returns:
            Parameter name
        """
        return value[2:-2].strip()
    
    def generate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration from template with parameters.
        
        Args:
            parameters: Parameter values for template generation
            
        Returns:
            Generated configuration
        """
        # Validate required parameters
        self._validate_parameters(parameters)
        
        # Generate configuration
        generated_config = self._generate_recursive(self.template_content, parameters)
        
        return generated_config
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate provided parameters.
        
        Args:
            parameters: Parameters to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        for param_name, param_def in self.parameters.items():
            if param_def.required and param_name not in parameters:
                raise ConfigurationError(f"Required parameter '{param_name}' is missing")
            
            if param_name in parameters:
                value = parameters[param_name]
                self._validate_parameter_value(param_name, value, param_def)
    
    def _validate_parameter_value(self, param_name: str, value: Any, param_def: TemplateParameter) -> None:
        """Validate individual parameter value.
        
        Args:
            param_name: Parameter name
            value: Parameter value
            param_def: Parameter definition
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Check type
        if param_def.parameter_type == "string" and not isinstance(value, str):
            raise ConfigurationError(f"Parameter '{param_name}' must be a string")
        elif param_def.parameter_type == "integer" and not isinstance(value, int):
            raise ConfigurationError(f"Parameter '{param_name}' must be an integer")
        elif param_def.parameter_type == "float" and not isinstance(value, (int, float)):
            raise ConfigurationError(f"Parameter '{param_name}' must be a number")
        elif param_def.parameter_type == "boolean" and not isinstance(value, bool):
            raise ConfigurationError(f"Parameter '{param_name}' must be a boolean")
        
        # Check choices
        if param_def.choices and value not in param_def.choices:
            raise ConfigurationError(f"Parameter '{param_name}' must be one of {param_def.choices}")
        
        # Check validation rules
        if param_def.validation_rules:
            self._apply_validation_rules(param_name, value, param_def.validation_rules)
    
    def _apply_validation_rules(self, param_name: str, value: Any, rules: Dict[str, Any]) -> None:
        """Apply validation rules to parameter value.
        
        Args:
            param_name: Parameter name
            value: Parameter value
            rules: Validation rules
            
        Raises:
            ConfigurationError: If validation fails
        """
        if "min" in rules and value < rules["min"]:
            raise ConfigurationError(f"Parameter '{param_name}' must be >= {rules['min']}")
        
        if "max" in rules and value > rules["max"]:
            raise ConfigurationError(f"Parameter '{param_name}' must be <= {rules['max']}")
        
        if "min_length" in rules and len(str(value)) < rules["min_length"]:
            raise ConfigurationError(f"Parameter '{param_name}' must have length >= {rules['min_length']}")
        
        if "max_length" in rules and len(str(value)) > rules["max_length"]:
            raise ConfigurationError(f"Parameter '{param_name}' must have length <= {rules['max_length']}")
        
        if "pattern" in rules and not re.match(rules["pattern"], str(value)):
            raise ConfigurationError(f"Parameter '{param_name}' does not match required pattern")
    
    def _generate_recursive(self, content: Any, parameters: Dict[str, Any]) -> Any:
        """Recursively generate configuration from template.
        
        Args:
            content: Template content
            parameters: Parameter values
            
        Returns:
            Generated content
        """
        if isinstance(content, dict):
            return {
                key: self._generate_recursive(value, parameters)
                for key, value in content.items()
            }
        elif isinstance(content, list):
            return [
                self._generate_recursive(item, parameters)
                for item in content
            ]
        elif isinstance(content, str) and self._is_parameter(content):
            param_name = self._extract_parameter_name(content)
            if param_name in parameters:
                return parameters[param_name]
            elif param_name in self.parameters and self.parameters[param_name].default_value is not None:
                return self.parameters[param_name].default_value
            else:
                raise ConfigurationError(f"Parameter '{param_name}' is required but not provided")
        else:
            return content
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about template parameters.
        
        Returns:
            Dictionary with parameter information
        """
        return {
            name: {
                "description": param.description,
                "type": param.parameter_type,
                "default": param.default_value,
                "required": param.required,
                "validation_rules": param.validation_rules,
                "choices": param.choices
            }
            for name, param in self.parameters.items()
        }


class TemplateGenerator:
    """Generator for creating configuration templates."""
    
    def __init__(self, templates_dir: str = "config/templates"):
        """Initialize template generator.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, ConfigurationTemplate] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all templates from templates directory."""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return
        
        # Load template files
        for template_file in self.templates_dir.rglob("*.yaml"):
            try:
                self._load_template_file(template_file)
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")
        
        for template_file in self.templates_dir.rglob("*.json"):
            try:
                self._load_template_file(template_file)
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")
    
    def _load_template_file(self, template_file: Path) -> None:
        """Load a single template file.
        
        Args:
            template_file: Path to template file
        """
        with open(template_file, 'r', encoding='utf-8') as f:
            if template_file.suffix.lower() in ['.yaml', '.yml']:
                template_data = yaml.safe_load(f)
            else:
                template_data = json.load(f)
        
        # Extract metadata
        metadata = TemplateMetadata(
            name=template_data.get('metadata', {}).get('name', template_file.stem),
            description=template_data.get('metadata', {}).get('description', ''),
            version=template_data.get('metadata', {}).get('version', '1.0.0'),
            template_type=TemplateType(template_data.get('metadata', {}).get('type', 'base')),
            category=TemplateCategory(template_data.get('metadata', {}).get('category', 'custom')),
            author=template_data.get('metadata', {}).get('author', ''),
            tags=template_data.get('metadata', {}).get('tags', []),
            dependencies=template_data.get('metadata', {}).get('dependencies', []),
            parameters=template_data.get('metadata', {}).get('parameters', {}),
            validation_schema=template_data.get('validation_schema')
        )
        
        # Extract template content
        template_content = template_data.get('template', {})
        
        # Create template
        template = ConfigurationTemplate(metadata, template_content)
        self.templates[metadata.name] = template
        
        logger.debug(f"Loaded template: {metadata.name}")
    
    def get_template(self, name: str) -> Optional[ConfigurationTemplate]:
        """Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template or None if not found
        """
        return self.templates.get(name)
    
    def list_templates(self, template_type: Optional[TemplateType] = None, category: Optional[TemplateCategory] = None) -> List[str]:
        """List available templates.
        
        Args:
            template_type: Filter by template type
            category: Filter by category
            
        Returns:
            List of template names
        """
        templates = []
        
        for name, template in self.templates.items():
            if template_type and template.metadata.template_type != template_type:
                continue
            
            if category and template.metadata.category != category:
                continue
            
            templates.append(name)
        
        return sorted(templates)
    
    def generate_config(self, template_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration from template.
        
        Args:
            template_name: Name of the template
            parameters: Parameter values
            
        Returns:
            Generated configuration
            
        Raises:
            ConfigurationError: If template not found or generation fails
        """
        template = self.get_template(template_name)
        if not template:
            raise ConfigurationError(f"Template '{template_name}' not found")
        
        return template.generate(parameters)
    
    def create_template(
        self,
        name: str,
        description: str,
        template_type: TemplateType,
        category: TemplateCategory,
        template_content: Dict[str, Any],
        author: str = "",
        tags: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> ConfigurationTemplate:
        """Create a new configuration template.
        
        Args:
            name: Template name
            description: Template description
            template_type: Type of template
            category: Template category
            template_content: Template content
            author: Template author
            tags: Template tags
            parameters: Template parameters
            
        Returns:
            Created template
        """
        metadata = TemplateMetadata(
            name=name,
            description=description,
            version="1.0.0",
            template_type=template_type,
            category=category,
            author=author,
            tags=tags or [],
            parameters=parameters or {}
        )
        
        template = ConfigurationTemplate(metadata, template_content)
        self.templates[name] = template
        
        return template
    
    def save_template(self, template: ConfigurationTemplate, filename: Optional[str] = None) -> None:
        """Save template to file.
        
        Args:
            template: Template to save
            filename: Optional filename (defaults to template name)
        """
        if not filename:
            filename = f"{template.metadata.name}.yaml"
        
        # Ensure templates directory exists
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare template data
        template_data = {
            'metadata': {
                'name': template.metadata.name,
                'description': template.metadata.description,
                'version': template.metadata.version,
                'type': template.metadata.template_type.value,
                'category': template.metadata.category.value,
                'author': template.metadata.author,
                'tags': template.metadata.tags,
                'dependencies': template.metadata.dependencies,
                'parameters': template.metadata.parameters
            },
            'template': template.template_content
        }
        
        if template.metadata.validation_schema:
            template_data['validation_schema'] = template.metadata.validation_schema
        
        # Save to file
        template_file = self.templates_dir / filename
        with open(template_file, 'w', encoding='utf-8') as f:
            yaml.dump(template_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved template: {template_file}")


class BestPracticeTemplates:
    """Collection of best practice configuration templates."""
    
    @staticmethod
    def get_development_template() -> Dict[str, Any]:
        """Get development environment template.
        
        Returns:
            Development template content
        """
        return {
            "metadata": {
                "name": "development",
                "description": "Development environment configuration template",
                "type": "environment",
                "category": "development",
                "author": "Configuration System",
                "tags": ["development", "debug", "local"]
            },
            "template": {
                "debug": True,
                "verbose_logging": True,
                "max_steps": "{{max_steps}}",
                "max_population": "{{max_population}}",
                "use_in_memory_db": True,
                "persist_db_on_completion": False,
                "visualization": {
                    "display_enabled": True,
                    "show_debug_info": True,
                    "frame_rate": 10
                },
                "learning_rate": "{{learning_rate}}",
                "epsilon_start": "{{epsilon_start}}",
                "epsilon_min": "{{epsilon_min}}"
            }
        }
    
    @staticmethod
    def get_production_template() -> Dict[str, Any]:
        """Get production environment template.
        
        Returns:
            Production template content
        """
        return {
            "metadata": {
                "name": "production",
                "description": "Production environment configuration template",
                "type": "environment",
                "category": "production",
                "author": "Configuration System",
                "tags": ["production", "optimized", "secure"]
            },
            "template": {
                "debug": False,
                "verbose_logging": False,
                "log_level": "ERROR",
                "max_steps": "{{max_steps}}",
                "max_population": "{{max_population}}",
                "use_in_memory_db": False,
                "persist_db_on_completion": True,
                "visualization": {
                    "display_enabled": False,
                    "show_debug_info": False,
                    "frame_rate": 0
                },
                "learning_rate": "{{learning_rate}}",
                "epsilon_start": "{{epsilon_start}}",
                "epsilon_min": "{{epsilon_min}}",
                "db_pragma_profile": "safety",
                "db_cache_size_mb": 200
            }
        }
    
    @staticmethod
    def get_system_agent_template() -> Dict[str, Any]:
        """Get system agent template.
        
        Returns:
            System agent template content
        """
        return {
            "metadata": {
                "name": "system_agent",
                "description": "System agent configuration template",
                "type": "agent",
                "category": "custom",
                "author": "Configuration System",
                "tags": ["agent", "system", "cooperative"]
            },
            "template": {
                "agent_parameters": {
                    "SystemAgent": {
                        "gather_efficiency_multiplier": "{{gather_efficiency}}",
                        "gather_cost_multiplier": "{{gather_cost}}",
                        "min_resource_threshold": "{{resource_threshold}}",
                        "share_weight": "{{share_weight}}",
                        "attack_weight": "{{attack_weight}}"
                    }
                },
                "learning_rate": "{{learning_rate}}",
                "gamma": "{{gamma}}",
                "epsilon_start": "{{epsilon_start}}",
                "epsilon_min": "{{epsilon_min}}"
            }
        }
    
    @staticmethod
    def get_validation_template() -> Dict[str, Any]:
        """Get validation schema template.
        
        Returns:
            Validation template content
        """
        return {
            "metadata": {
                "name": "simulation_validation",
                "description": "Simulation configuration validation schema",
                "type": "validation",
                "category": "custom",
                "author": "Configuration System",
                "tags": ["validation", "schema", "simulation"]
            },
            "template": {
                "required": [
                    "simulation_id",
                    "max_steps",
                    "environment",
                    "learning_rate"
                ],
                "fields": {
                    "simulation_id": {
                        "type": "string",
                        "min_length": 1,
                        "max_length": 50
                    },
                    "max_steps": {
                        "type": "integer",
                        "min": 1,
                        "max": 100000
                    },
                    "environment": {
                        "type": "string",
                        "choices": ["development", "staging", "production", "testing"]
                    },
                    "learning_rate": {
                        "type": "float",
                        "min": 0.00001,
                        "max": 0.1
                    }
                }
            }
        }


class ConfigurationTemplateManager:
    """Manager for configuration templates and generation."""
    
    def __init__(self, templates_dir: str = "config/templates"):
        """Initialize template manager.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.generator = TemplateGenerator(templates_dir)
        self._load_best_practice_templates()
    
    def _load_best_practice_templates(self) -> None:
        """Load best practice templates."""
        best_practices = [
            BestPracticeTemplates.get_development_template(),
            BestPracticeTemplates.get_production_template(),
            BestPracticeTemplates.get_system_agent_template(),
            BestPracticeTemplates.get_validation_template()
        ]
        
        for template_data in best_practices:
            metadata = TemplateMetadata(
                name=template_data["metadata"]["name"],
                description=template_data["metadata"]["description"],
                version="1.0.0",
                template_type=TemplateType(template_data["metadata"]["type"]),
                category=TemplateCategory(template_data["metadata"]["category"]),
                author=template_data["metadata"]["author"],
                tags=template_data["metadata"]["tags"]
            )
            
            template = ConfigurationTemplate(metadata, template_data["template"])
            self.generator.templates[metadata.name] = template
    
    def generate_environment_config(self, environment: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate environment configuration from template.
        
        Args:
            environment: Environment name
            parameters: Generation parameters
            
        Returns:
            Generated environment configuration
        """
        return self.generator.generate_config(environment, parameters)
    
    def generate_agent_config(self, agent_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent configuration from template.
        
        Args:
            agent_type: Agent type
            parameters: Generation parameters
            
        Returns:
            Generated agent configuration
        """
        return self.generator.generate_config(agent_type, parameters)
    
    def generate_validation_schema(self, schema_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation schema from template.
        
        Args:
            schema_name: Schema name
            parameters: Generation parameters
            
        Returns:
            Generated validation schema
        """
        return self.generator.generate_config(schema_name, parameters)
    
    def list_available_templates(self) -> Dict[str, List[str]]:
        """List all available templates by type.
        
        Returns:
            Dictionary with templates grouped by type
        """
        templates_by_type = {}
        
        for template_type in TemplateType:
            templates = self.generator.list_templates(template_type=template_type)
            if templates:
                templates_by_type[template_type.value] = templates
        
        return templates_by_type
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a template.
        
        Args:
            template_name: Template name
            
        Returns:
            Template information or None if not found
        """
        template = self.generator.get_template(template_name)
        if not template:
            return None
        
        return {
            "metadata": {
                "name": template.metadata.name,
                "description": template.metadata.description,
                "version": template.metadata.version,
                "type": template.metadata.template_type.value,
                "category": template.metadata.category.value,
                "author": template.metadata.author,
                "tags": template.metadata.tags,
                "dependencies": template.metadata.dependencies
            },
            "parameters": template.get_parameter_info()
        }