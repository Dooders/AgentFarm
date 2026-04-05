"""Tests for config template module (ConfigTemplate and ConfigTemplateManager)."""

import os
import tempfile
import unittest

import yaml

from farm.config.template import ConfigTemplate, ConfigTemplateManager


class TestConfigTemplate(unittest.TestCase):
    """Tests for ConfigTemplate class."""

    def test_init_basic(self):
        """Test basic template initialization."""
        template = ConfigTemplate({"key": "value", "count": 42})
        self.assertEqual(template.template_dict["key"], "value")

    def test_get_required_variables_none(self):
        """Template with no placeholders returns empty list."""
        template = ConfigTemplate({"key": "value"})
        self.assertEqual(template.get_required_variables(), [])

    def test_get_required_variables_single(self):
        """Template with one placeholder returns it."""
        template = ConfigTemplate({"key": "{{my_var}}"})
        self.assertEqual(template.get_required_variables(), ["my_var"])

    def test_get_required_variables_multiple(self):
        """Template with multiple placeholders returns all."""
        template = ConfigTemplate({"a": "{{x}}", "b": "{{y}}", "c": "static"})
        self.assertEqual(template.get_required_variables(), ["x", "y"])

    def test_get_required_variables_in_nested(self):
        """Placeholders in nested dicts and lists are discovered."""
        template = ConfigTemplate({"outer": {"inner": "{{z}}"}})
        self.assertIn("z", template.get_required_variables())

    def test_validate_variables_all_present(self):
        """No missing variables when all are provided."""
        template = ConfigTemplate({"a": "{{x}}", "b": "{{y}}"})
        missing = template.validate_variables({"x": 1, "y": 2})
        self.assertEqual(missing, [])

    def test_validate_variables_missing(self):
        """Missing variables are reported."""
        template = ConfigTemplate({"a": "{{x}}", "b": "{{y}}"})
        missing = template.validate_variables({"x": 1})
        self.assertEqual(missing, ["y"])

    def test_instantiate_string_replacement(self):
        """String placeholders are replaced with provided values."""
        template = ConfigTemplate({"simulation_steps": "{{steps}}", "max_steps": "{{steps}}"})
        # Use a dict-compatible template; SimulationConfig.from_dict needs full dict
        from farm.config import SimulationConfig

        base = SimulationConfig().to_dict()
        base["simulation_steps"] = "{{steps}}"
        base["max_steps"] = "{{steps}}"
        template2 = ConfigTemplate(base)
        config = template2.instantiate({"steps": 50})
        self.assertEqual(config.simulation_steps, 50)

    def test_instantiate_missing_variable_raises(self):
        """Instantiating with missing variable raises ValueError."""
        template = ConfigTemplate({"key": "{{missing_var}}"})
        with self.assertRaises((ValueError, Exception)):
            template.instantiate({})

    def test_from_yaml(self):
        """Template can be loaded from a YAML file."""
        data = {"simulation_steps": 10, "max_steps": 20}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            fname = f.name
        try:
            template = ConfigTemplate.from_yaml(fname)
            self.assertEqual(template.template_dict["simulation_steps"], 10)
        finally:
            os.unlink(fname)

    def test_from_config(self):
        """Template can be created from a SimulationConfig."""
        from farm.config import SimulationConfig

        config = SimulationConfig()
        template = ConfigTemplate.from_config(config)
        self.assertIsInstance(template, ConfigTemplate)
        # The template dict should reflect config keys
        self.assertIn("simulation_steps", template.template_dict)

    def test_to_yaml(self):
        """Template can be saved to a YAML file."""
        template = ConfigTemplate({"key": "value"})
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tpl.yaml")
            template.to_yaml(filepath)
            self.assertTrue(os.path.exists(filepath))
            with open(filepath) as f:
                loaded = yaml.safe_load(f)
            self.assertEqual(loaded["key"], "value")

    def test_placeholder_in_list(self):
        """Placeholders inside list items are discovered."""
        template = ConfigTemplate({"items": ["{{a}}", "static"]})
        self.assertIn("a", template.get_required_variables())


class TestConfigTemplateManager(unittest.TestCase):
    """Tests for ConfigTemplateManager class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager = ConfigTemplateManager(template_dir=self.tmpdir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_simple_template(self):
        """Return a ConfigTemplate based on a default SimulationConfig."""
        from farm.config import SimulationConfig

        return ConfigTemplate.from_config(SimulationConfig())

    def test_save_and_load_template(self):
        """Saved template can be loaded back."""
        template = self._make_simple_template()
        self.manager.save_template("basic", template, description="A basic template")

        loaded = self.manager.load_template("basic")
        self.assertIsInstance(loaded, ConfigTemplate)

    def test_load_nonexistent_raises(self):
        """Loading a non-existent template raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.manager.load_template("does_not_exist")

    def test_list_templates_empty(self):
        """Empty directory returns empty list."""
        templates = self.manager.list_templates()
        self.assertEqual(templates, [])

    def test_list_templates_returns_saved(self):
        """List templates includes saved templates."""
        template = self._make_simple_template()
        self.manager.save_template("my_template", template, description="desc")

        templates = self.manager.list_templates()
        names = [t["name"] for t in templates]
        self.assertIn("my_template", names)

    def test_list_templates_includes_description(self):
        """Template listing includes description."""
        template = self._make_simple_template()
        self.manager.save_template("t1", template, description="my description")

        templates = self.manager.list_templates()
        self.assertEqual(templates[0]["description"], "my description")

    def test_create_experiment_configs(self):
        """Batch experiment configs are generated from a template."""
        from farm.config import SimulationConfig

        base = SimulationConfig().to_dict()
        base["simulation_steps"] = "{{steps}}"
        base["max_steps"] = "{{steps}}"
        template = ConfigTemplate(base)
        self.manager.save_template("exp_tpl", template)

        variable_sets = [{"steps": 10}, {"steps": 20}]
        output_dir = os.path.join(self.tmpdir, "experiments")
        paths = self.manager.create_experiment_configs("exp_tpl", variable_sets, output_dir)

        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(os.path.exists(p))

    def test_save_creates_yaml_file(self):
        """save_template produces a .yaml file."""
        template = self._make_simple_template()
        filepath = self.manager.save_template("saved", template)
        self.assertTrue(filepath.endswith(".yaml"))
        self.assertTrue(os.path.exists(filepath))


if __name__ == "__main__":
    unittest.main()
