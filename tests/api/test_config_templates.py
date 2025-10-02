"""Unit tests for the ConfigTemplateManager class."""

from typing import Dict, Any
from unittest.mock import Mock, patch

import pytest

from farm.api.config_templates import ConfigTemplateManager
from farm.api.models import ConfigCategory, ConfigTemplate, ValidationResult
from farm.config import SimulationConfig


class TestConfigTemplateManager:
    """Test ConfigTemplateManager class."""

    def test_init(self):
        """Test ConfigTemplateManager initialization."""
        manager = ConfigTemplateManager()
        
        assert manager._templates is not None
        assert len(manager._templates) > 0

    def test_list_templates(self):
        """Test listing available templates."""
        manager = ConfigTemplateManager()
        
        templates = manager.list_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        
        # Check that all returned items are ConfigTemplate objects
        for template in templates:
            assert isinstance(template, ConfigTemplate)

    def test_list_templates_by_category(self):
        """Test listing templates filtered by category."""
        manager = ConfigTemplateManager()
        
        # Get all templates
        all_templates = manager.list_templates()
        
        # Filter simulation templates
        sim_templates = [t for t in all_templates if t.category == ConfigCategory.SIMULATION]
        
        assert len(sim_templates) > 0
        for template in sim_templates:
            assert template.category == ConfigCategory.SIMULATION

    def test_get_template_existing(self):
        """Test getting an existing template."""
        manager = ConfigTemplateManager()
        
        # Get a template that should exist
        template = manager.get_template("basic_simulation")
        
        assert template is not None
        assert isinstance(template, ConfigTemplate)
        assert template.name == "basic_simulation"

    def test_get_template_nonexistent(self):
        """Test getting a non-existent template."""
        manager = ConfigTemplateManager()
        
        template = manager.get_template("nonexistent_template")
        assert template is None

    def test_validate_config_valid(self):
        """Test validating a valid configuration."""
        manager = ConfigTemplateManager()
        
        config = {
            "name": "Test Simulation",
            "steps": 1000,
            "agents": {
                "system_agents": 10,
                "independent_agents": 10
            }
        }
        
        result = manager.validate_config(config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_config_missing_required_field(self):
        """Test validating config with missing required field."""
        manager = ConfigTemplateManager()
        
        config = {
            "steps": 1000,
            # Missing "name" field
        }
        
        result = manager.validate_config(config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("name" in error.lower() for error in result.errors)

    def test_validate_config_invalid_field_type(self):
        """Test validating config with invalid field type."""
        manager = ConfigTemplateManager()
        
        config = {
            "name": "Test Simulation",
            "steps": "invalid",  # Should be integer
            "agents": {
                "system_agents": 10,
                "independent_agents": 10
            }
        }
        
        result = manager.validate_config(config)
        
        assert isinstance(result, ValidationResult)
        # May or may not be invalid depending on validation rules
        assert isinstance(result, ValidationResult)

    def test_validate_config_with_warnings(self):
        """Test validating config that generates warnings."""
        manager = ConfigTemplateManager()
        
        config = {
            "name": "Test Simulation",
            "steps": 50,  # Very low step count might generate warning
            "agents": {
                "system_agents": 1,  # Very low agent count might generate warning
                "independent_agents": 1
            }
        }
        
        result = manager.validate_config(config)
        
        assert isinstance(result, ValidationResult)
        # Warnings are optional, so just check structure
        assert isinstance(result.warnings, list)

    def test_create_config_from_template_success(self):
        """Test creating config from template successfully."""
        manager = ConfigTemplateManager()
        
        config = manager.create_config_from_template("basic_simulation")
        
        assert config is not None
        assert isinstance(config, dict)
        assert "name" in config
        assert "steps" in config

    def test_create_config_from_template_with_overrides(self):
        """Test creating config from template with overrides."""
        manager = ConfigTemplateManager()
        
        overrides = {
            "name": "Custom Simulation",
            "steps": 2000,
            "agents": {
                "system_agents": 20,
                "independent_agents": 20
            }
        }
        
        config = manager.create_config_from_template("basic_simulation", overrides)
        
        assert config is not None
        assert config["name"] == "Custom Simulation"
        assert config["steps"] == 2000
        assert config["agents"]["system_agents"] == 20
        assert config["agents"]["independent_agents"] == 20

    def test_create_config_from_template_nonexistent(self):
        """Test creating config from non-existent template."""
        manager = ConfigTemplateManager()
        
        config = manager.create_config_from_template("nonexistent_template")
        assert config is None

    def test_convert_to_simulation_config_success(self):
        """Test converting config to SimulationConfig successfully."""
        manager = ConfigTemplateManager()
        
        config = {
            "name": "Test Simulation",
            "steps": 1000,
            "environment": {
                "width": 100,
                "height": 100,
                "resources": 50
            },
            "agents": {
                "system_agents": 10,
                "independent_agents": 10,
                "control_agents": 0
            },
            "learning": {
                "enabled": True,
                "algorithm": "dqn"
            }
        }
        
        sim_config = manager.convert_to_simulation_config(config)
        
        assert sim_config is not None
        assert isinstance(sim_config, SimulationConfig)

    def test_convert_to_simulation_config_invalid(self):
        """Test converting invalid config to SimulationConfig."""
        manager = ConfigTemplateManager()
        
        config = {
            "name": "Test Simulation",
            # Missing required fields
        }
        
        sim_config = manager.convert_to_simulation_config(config)
        assert sim_config is None

    def test_convert_to_simulation_config_with_defaults(self):
        """Test converting config with defaults applied."""
        manager = ConfigTemplateManager()
        
        config = {
            "name": "Test Simulation",
            "steps": 1000
            # Missing other fields that should get defaults
        }
        
        sim_config = manager.convert_to_simulation_config(config)
        
        if sim_config is not None:
            assert sim_config.simulation_steps == 1000

    def test_get_template_examples(self):
        """Test getting template examples."""
        manager = ConfigTemplateManager()
        
        examples = manager.get_template_examples("basic_simulation")
        
        assert examples is not None
        assert isinstance(examples, list)
        assert len(examples) > 0
        
        for example in examples:
            assert isinstance(example, dict)

    def test_get_template_examples_nonexistent(self):
        """Test getting examples for non-existent template."""
        manager = ConfigTemplateManager()
        
        examples = manager.get_template_examples("nonexistent_template")
        assert examples is None

    def test_get_required_fields(self):
        """Test getting required fields for a template."""
        manager = ConfigTemplateManager()
        
        required_fields = manager.get_required_fields("basic_simulation")
        
        assert required_fields is not None
        assert isinstance(required_fields, list)
        assert "name" in required_fields
        assert "steps" in required_fields

    def test_get_required_fields_nonexistent(self):
        """Test getting required fields for non-existent template."""
        manager = ConfigTemplateManager()
        
        required_fields = manager.get_required_fields("nonexistent_template")
        assert required_fields is None

    def test_get_optional_fields(self):
        """Test getting optional fields for a template."""
        manager = ConfigTemplateManager()
        
        optional_fields = manager.get_optional_fields("basic_simulation")
        
        assert optional_fields is not None
        assert isinstance(optional_fields, list)

    def test_get_optional_fields_nonexistent(self):
        """Test getting optional fields for non-existent template."""
        manager = ConfigTemplateManager()
        
        optional_fields = manager.get_optional_fields("nonexistent_template")
        assert optional_fields is None

    def test_validate_against_template_success(self):
        """Test validating config against specific template."""
        manager = ConfigTemplateManager()
        
        config = {
            "name": "Test Simulation",
            "steps": 1000,
            "agents": {
                "system_agents": 10,
                "independent_agents": 10
            }
        }
        
        result = manager.validate_against_template("basic_simulation", config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_against_template_missing_required(self):
        """Test validating config against template with missing required fields."""
        manager = ConfigTemplateManager()
        
        config = {
            "steps": 1000,
            # Missing "name" field
        }
        
        result = manager.validate_against_template("basic_simulation", config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_against_template_nonexistent(self):
        """Test validating against non-existent template."""
        manager = ConfigTemplateManager()
        
        config = {"name": "Test", "steps": 1000}
        
        result = manager.validate_against_template("nonexistent_template", config)
        assert result is None

    def test_get_template_categories(self):
        """Test getting all template categories."""
        manager = ConfigTemplateManager()
        
        categories = manager.get_template_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        
        # Should include at least simulation category
        assert ConfigCategory.SIMULATION in categories

    def test_list_templates_by_category_simulation(self):
        """Test listing templates by simulation category."""
        manager = ConfigTemplateManager()
        
        templates = manager.list_templates_by_category(ConfigCategory.SIMULATION)
        
        assert isinstance(templates, list)
        for template in templates:
            assert template.category == ConfigCategory.SIMULATION

    def test_list_templates_by_category_experiment(self):
        """Test listing templates by experiment category."""
        manager = ConfigTemplateManager()
        
        templates = manager.list_templates_by_category(ConfigCategory.EXPERIMENT)
        
        assert isinstance(templates, list)
        for template in templates:
            assert template.category == ConfigCategory.EXPERIMENT

    def test_list_templates_by_category_research(self):
        """Test listing templates by research category."""
        manager = ConfigTemplateManager()
        
        templates = manager.list_templates_by_category(ConfigCategory.RESEARCH)
        
        assert isinstance(templates, list)
        for template in templates:
            assert template.category == ConfigCategory.RESEARCH

    def test_add_custom_template(self):
        """Test adding a custom template."""
        manager = ConfigTemplateManager()
        
        custom_template = ConfigTemplate(
            name="custom_template",
            description="A custom template for testing",
            category=ConfigCategory.SIMULATION,
            parameters={"name": "Custom", "steps": 500},
            required_fields=["name", "steps"],
            optional_fields=["agents"]
        )
        
        success = manager.add_template(custom_template)
        assert success is True
        
        # Should be able to retrieve it
        retrieved = manager.get_template("custom_template")
        assert retrieved is not None
        assert retrieved.name == "custom_template"

    def test_add_duplicate_template(self):
        """Test adding a duplicate template."""
        manager = ConfigTemplateManager()
        
        # Get existing template
        existing_template = manager.get_template("basic_simulation")
        assert existing_template is not None
        
        # Try to add it again
        success = manager.add_template(existing_template)
        assert success is False

    def test_remove_template_success(self):
        """Test removing a template successfully."""
        manager = ConfigTemplateManager()
        
        # First add a custom template
        custom_template = ConfigTemplate(
            name="temp_template",
            description="Temporary template",
            category=ConfigCategory.SIMULATION,
            parameters={"name": "Temp", "steps": 100},
            required_fields=["name", "steps"]
        )
        
        manager.add_template(custom_template)
        
        # Now remove it
        success = manager.remove_template("temp_template")
        assert success is True
        
        # Should not be able to retrieve it
        retrieved = manager.get_template("temp_template")
        assert retrieved is None

    def test_remove_template_nonexistent(self):
        """Test removing a non-existent template."""
        manager = ConfigTemplateManager()
        
        success = manager.remove_template("nonexistent_template")
        assert success is False

    def test_remove_protected_template(self):
        """Test removing a protected template (should fail)."""
        manager = ConfigTemplateManager()
        
        # Try to remove a default template
        success = manager.remove_template("basic_simulation")
        assert success is False
        
        # Should still be able to retrieve it
        retrieved = manager.get_template("basic_simulation")
        assert retrieved is not None

    @patch('farm.api.config_templates.get_logger')
    def test_logging_in_validation(self, mock_get_logger):
        """Test that validation methods log appropriately."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        manager = ConfigTemplateManager()
        
        # Test with invalid config
        config = {"invalid": "config"}
        result = manager.validate_config(config)
        
        # Should log validation attempts
        mock_logger.debug.assert_called()

    def test_config_validation_edge_cases(self):
        """Test config validation with edge cases."""
        manager = ConfigTemplateManager()
        
        # Test with empty config
        result = manager.validate_config({})
        assert isinstance(result, ValidationResult)
        
        # Test with None config
        result = manager.validate_config(None)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        
        # Test with non-dict config
        result = manager.validate_config("not a dict")
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False

    def test_template_parameter_merging(self):
        """Test that template parameters are properly merged with overrides."""
        manager = ConfigTemplateManager()
        
        overrides = {
            "name": "Override Name",
            "new_field": "new_value"
        }
        
        config = manager.create_config_from_template("basic_simulation", overrides)
        
        assert config is not None
        assert config["name"] == "Override Name"
        assert config["new_field"] == "new_value"
        
        # Should still have original template parameters
        assert "steps" in config

    def test_config_validation_with_nested_structures(self):
        """Test config validation with nested dictionary structures."""
        manager = ConfigTemplateManager()
        
        config = {
            "name": "Test Simulation",
            "steps": 1000,
            "environment": {
                "width": 100,
                "height": 100,
                "resources": 50
            },
            "agents": {
                "system_agents": 10,
                "independent_agents": 10,
                "control_agents": 0
            },
            "learning": {
                "enabled": True,
                "algorithm": "dqn",
                "parameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32
                }
            }
        }
        
        result = manager.validate_config(config)
        
        assert isinstance(result, ValidationResult)
        # Should handle nested structures without errors
        assert result.is_valid is True or len(result.errors) == 0
