"""
Configuration templating system for Agent Farm.

This module provides functionality to create parameterized configuration templates
that can be instantiated with different values, enabling easy configuration
variation for experiments.
"""

import copy
import os
import re
from typing import Any, Dict, List, Optional, Union

import yaml

from farm.core.config import SimulationConfig


class ConfigTemplate:
    """
    A configuration template with parameterized placeholders.

    Templates use {{variable_name}} syntax for placeholders that can be
    substituted with actual values when instantiating configurations.
    """

    def __init__(self, template_dict: Dict[str, Any]):
        """
        Initialize a configuration template.

        Args:
            template_dict: Dictionary containing template configuration with placeholders
        """
        self.template_dict = copy.deepcopy(template_dict)
        self._validate_template()

    def _validate_template(self) -> None:
        """Validate that the template has proper placeholder syntax."""
        def _check_placeholders(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(key, str) and "{{" in key and "}}" in key:
                        # Check that key placeholders are properly formatted
                        if not re.match(r'^\{\{\w+\}\}$', key):
                            raise ValueError(f"Invalid placeholder format in key: {current_path}")
                    _check_placeholders(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _check_placeholders(item, f"{path}[{i}]")
            elif isinstance(obj, str):
                # Check string value placeholders
                placeholders = re.findall(r'\{\{(\w+)\}\}', obj)
                for placeholder in placeholders:
                    if not re.match(r'^\w+$', placeholder):
                        raise ValueError(f"Invalid placeholder name '{placeholder}' in {path}")

        _check_placeholders(self.template_dict)

    @classmethod
    def from_yaml(cls, file_path: str) -> 'ConfigTemplate':
        """
        Load a configuration template from a YAML file.

        Args:
            file_path: Path to the template YAML file

        Returns:
            ConfigTemplate: Loaded template
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            template_dict = yaml.safe_load(f)
        return cls(template_dict)

    @classmethod
    def from_config(cls, config: SimulationConfig) -> 'ConfigTemplate':
        """
        Create a template from an existing configuration.

        Args:
            config: Configuration to convert to template

        Returns:
            ConfigTemplate: Template based on the configuration
        """
        return cls(config.to_dict())

    def instantiate(self, variables: Dict[str, Any]) -> SimulationConfig:
        """
        Instantiate the template with specific variable values.

        Args:
            variables: Dictionary mapping variable names to values

        Returns:
            SimulationConfig: Instantiated configuration

        Raises:
            ValueError: If required variables are missing
        """
        # Deep copy the template
        instance_dict = copy.deepcopy(self.template_dict)

        # Replace placeholders in keys and values
        def _replace_placeholders(obj: Any) -> Any:
            if isinstance(obj, dict):
                new_dict = {}
                for key, value in obj.items():
                    # Replace placeholders in keys
                    if isinstance(key, str):
                        new_key = self._replace_string_placeholders(key, variables)
                    else:
                        new_key = key

                    # Replace placeholders in values
                    new_value = _replace_placeholders(value)
                    new_dict[new_key] = new_value
                return new_dict
            elif isinstance(obj, list):
                return [_replace_placeholders(item) for item in obj]
            elif isinstance(obj, str):
                return self._replace_string_placeholders(obj, variables)
            else:
                return obj

        instance_dict = _replace_placeholders(instance_dict)

        # Create SimulationConfig from the instantiated dict
        return SimulationConfig.from_dict(instance_dict)

    def _replace_string_placeholders(self, text: str, variables: Dict[str, Any]) -> str:
        """
        Replace placeholders in a string with variable values.

        Args:
            text: String containing placeholders
            variables: Variable values

        Returns:
            str: String with placeholders replaced
        """
        def _replace_match(match):
            var_name = match.group(1)
            if var_name not in variables:
                raise ValueError(f"Missing required variable: {var_name}")
            return str(variables[var_name])

        return re.sub(r'\{\{(\w+)\}\}', _replace_match, text)

    def get_required_variables(self) -> List[str]:
        """
        Get list of all required variables in the template.

        Returns:
            List[str]: List of variable names required by the template
        """
        variables = set()

        def _collect_placeholders(obj: Any) -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(key, str):
                        matches = re.findall(r'\{\{(\w+)\}\}', key)
                        variables.update(matches)
                    _collect_placeholders(value)
            elif isinstance(obj, list):
                for item in obj:
                    _collect_placeholders(item)
            elif isinstance(obj, str):
                matches = re.findall(r'\{\{(\w+)\}\}', obj)
                variables.update(matches)

        _collect_placeholders(self.template_dict)
        return sorted(list(variables))

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """
        Validate that all required variables are provided.

        Args:
            variables: Variables to validate

        Returns:
            List[str]: List of missing variables (empty if all present)
        """
        required = set(self.get_required_variables())
        provided = set(variables.keys())
        missing = required - provided
        return sorted(list(missing))

    def to_yaml(self, file_path: str) -> None:
        """
        Save the template to a YAML file.

        Args:
            file_path: Path to save the template
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.template_dict, f, default_flow_style=False)


class ConfigTemplateManager:
    """
    Manager for configuration templates.
    """

    def __init__(self, template_dir: str = "config/templates"):
        """
        Initialize template manager.

        Args:
            template_dir: Directory containing templates
        """
        self.template_dir = template_dir
        os.makedirs(template_dir, exist_ok=True)

    def save_template(self, name: str, template: ConfigTemplate, description: Optional[str] = None) -> str:
        """
        Save a template to the template directory.

        Args:
            name: Template name
            template: Template to save
            description: Optional description

        Returns:
            str: Path to saved template
        """
        template_data = {
            '_metadata': {
                'name': name,
                'description': description,
                'required_variables': template.get_required_variables()
            },
            'template': template.template_dict
        }

        filepath = os.path.join(self.template_dir, f"{name}.yaml")
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(template_data, f, default_flow_style=False)

        return filepath

    def load_template(self, name: str) -> ConfigTemplate:
        """
        Load a template from the template directory.

        Args:
            name: Template name

        Returns:
            ConfigTemplate: Loaded template

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        filepath = os.path.join(self.template_dir, f"{name}.yaml")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Template '{name}' not found in {self.template_dir}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if 'template' not in data:
            raise ValueError(f"Invalid template format in {filepath}")

        return ConfigTemplate(data['template'])

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.

        Returns:
            List[Dict]: List of template information
        """
        templates = []
        if not os.path.exists(self.template_dir):
            return templates

        for filename in os.listdir(self.template_dir):
            if filename.endswith('.yaml'):
                filepath = os.path.join(self.template_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)

                    metadata = data.get('_metadata', {})
                    templates.append({
                        'name': metadata.get('name', filename.replace('.yaml', '')),
                        'description': metadata.get('description'),
                        'required_variables': metadata.get('required_variables', []),
                        'filepath': filepath
                    })
                except Exception:
                    continue

        return templates

    def create_experiment_configs(
        self,
        template_name: str,
        variable_sets: List[Dict[str, Any]],
        output_dir: str = "config/experiments"
    ) -> List[str]:
        """
        Create multiple configuration files from a template with different variable sets.

        Args:
            template_name: Name of template to use
            variable_sets: List of variable dictionaries
            output_dir: Directory to save generated configs

        Returns:
            List[str]: List of paths to generated config files
        """
        template = self.load_template(template_name)
        os.makedirs(output_dir, exist_ok=True)

        config_paths = []
        for i, variables in enumerate(variable_sets):
            config = template.instantiate(variables)
            versioned_config = config.version_config(
                description=f"Generated from template '{template_name}' with variables: {variables}"
            )

            filename = f"experiment_{template_name}_{i:03d}_{versioned_config.config_version}.yaml"
            filepath = os.path.join(output_dir, filename)
            versioned_config.to_yaml(filepath)
            config_paths.append(filepath)

        return config_paths
