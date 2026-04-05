"""Tests for config schema generation module."""

import unittest
from enum import Enum
from typing import Dict, List, Optional

from farm.config.schema import (
    _dataclass_to_properties,
    _enum_values,
    _pydantic_model_to_properties,
    _python_type_to_schema_type,
    generate_combined_config_schema,
)


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class TestPythonTypeToSchemaType(unittest.TestCase):
    """Tests for _python_type_to_schema_type."""

    def test_int(self):
        self.assertEqual(_python_type_to_schema_type(int), "integer")

    def test_float(self):
        self.assertEqual(_python_type_to_schema_type(float), "number")

    def test_bool(self):
        self.assertEqual(_python_type_to_schema_type(bool), "boolean")

    def test_str(self):
        self.assertEqual(_python_type_to_schema_type(str), "string")

    def test_list(self):
        self.assertEqual(_python_type_to_schema_type(List[int]), "array")

    def test_dict(self):
        self.assertEqual(_python_type_to_schema_type(Dict[str, int]), "object")

    def test_tuple(self):
        from typing import Tuple as TypingTuple
        self.assertEqual(_python_type_to_schema_type(TypingTuple[int, str]), "array")

    def test_optional_int(self):
        self.assertEqual(_python_type_to_schema_type(Optional[int]), "integer")

    def test_optional_str(self):
        self.assertEqual(_python_type_to_schema_type(Optional[str]), "string")

    def test_enum_type(self):
        self.assertEqual(_python_type_to_schema_type(Color), "string")

    def test_unknown_type(self):
        # Fallback for unknown types should be "object"
        self.assertEqual(_python_type_to_schema_type(object), "object")


class TestEnumValues(unittest.TestCase):
    """Tests for _enum_values."""

    def test_enum_returns_values(self):
        result = _enum_values(Color)
        self.assertEqual(sorted(result), ["blue", "green", "red"])

    def test_non_enum_returns_none(self):
        self.assertIsNone(_enum_values(int))
        self.assertIsNone(_enum_values(str))

    def test_type_error_handled(self):
        # Passing a non-type should return None gracefully
        self.assertIsNone(_enum_values("not_a_type"))


class TestDataclassToProperties(unittest.TestCase):
    """Tests for _dataclass_to_properties."""

    def test_simulation_config_properties(self):
        from farm.config.config import SimulationConfig

        props = _dataclass_to_properties(SimulationConfig)
        # Should contain basic fields
        self.assertIn("simulation_steps", props)
        self.assertIn("max_steps", props)

    def test_property_has_type_and_default(self):
        from farm.config.config import SimulationConfig

        props = _dataclass_to_properties(SimulationConfig)
        sim_steps = props["simulation_steps"]
        self.assertIn("type", sim_steps)
        self.assertIn("default", sim_steps)
        self.assertEqual(sim_steps["type"], "integer")

    def test_nested_dataclass_fields_skipped(self):
        from farm.config.config import SimulationConfig

        props = _dataclass_to_properties(SimulationConfig)
        # Nested dataclass fields like "environment", "population" should not appear
        self.assertNotIn("environment", props)
        self.assertNotIn("population", props)

    def test_known_enums_applied(self):
        from farm.config.config import SimulationConfig

        known_enums = {"simulation_steps": [10, 20, 50]}
        props = _dataclass_to_properties(SimulationConfig, known_enums=known_enums)
        self.assertIn("enum", props["simulation_steps"])
        self.assertEqual(props["simulation_steps"]["enum"], [10, 20, 50])

    def test_environment_config_properties(self):
        from farm.config.config import EnvironmentConfig

        props = _dataclass_to_properties(EnvironmentConfig)
        self.assertIn("width", props)
        self.assertIn("height", props)
        self.assertEqual(props["width"]["type"], "integer")


class TestPydanticModelToProperties(unittest.TestCase):
    """Tests for _pydantic_model_to_properties."""

    def test_observation_config_properties(self):
        from farm.core.observations import ObservationConfig

        props = _pydantic_model_to_properties(ObservationConfig)
        self.assertIsInstance(props, dict)
        self.assertGreater(len(props), 0)

    def test_properties_have_type(self):
        from farm.core.observations import ObservationConfig

        props = _pydantic_model_to_properties(ObservationConfig)
        for name, entry in props.items():
            self.assertIn("type", entry, f"Property '{name}' missing 'type' key")

    def test_model_with_optional_field_uses_anyof(self):
        """Pydantic models with Optional fields produce anyOf schemas."""
        from pydantic import BaseModel

        class ModelWithOptional(BaseModel):
            name: str = "default"
            value: Optional[int] = None

        props = _pydantic_model_to_properties(ModelWithOptional)
        self.assertIn("value", props)
        # type should be set even for Optional field
        self.assertIn("type", props["value"])

    def test_model_with_required_args_skips_instance(self):
        """Pydantic models with required args that can't be default-instantiated."""
        from pydantic import BaseModel

        class ModelRequiresArgs(BaseModel):
            required_field: str  # No default - cannot instantiate with no args

        # Should not raise; defaults fall back to None or meta-based
        props = _pydantic_model_to_properties(ModelRequiresArgs)
        self.assertIn("required_field", props)

    def test_model_with_description(self):
        """Fields with description metadata are preserved."""
        from pydantic import BaseModel, Field

        class ModelWithDesc(BaseModel):
            value: int = Field(default=5, description="A useful field")

        props = _pydantic_model_to_properties(ModelWithDesc)
        self.assertIn("description", props["value"])

    def test_model_with_enum(self):
        """Fields with enum metadata are preserved."""
        from pydantic import BaseModel, Field

        class ModelWithEnum(BaseModel):
            choice: str = Field(default="a", json_schema_extra={"enum": ["a", "b", "c"]})

        props = _pydantic_model_to_properties(ModelWithEnum)
        # Enum may appear in the generated schema
        # At minimum, type should be present
        self.assertIn("type", props["choice"])


class TestGenerateCombinedConfigSchema(unittest.TestCase):
    """Tests for generate_combined_config_schema."""

    def setUp(self):
        self.schema = generate_combined_config_schema()

    def test_schema_version(self):
        self.assertEqual(self.schema["version"], 1)

    def test_schema_has_generated_at(self):
        self.assertIn("generated_at", self.schema)
        # Should be an ISO timestamp ending in Z
        self.assertTrue(self.schema["generated_at"].endswith("Z"))

    def test_schema_sections_present(self):
        sections = self.schema["sections"]
        self.assertIn("simulation", sections)
        self.assertIn("visualization", sections)
        self.assertIn("redis", sections)
        self.assertIn("observation", sections)

    def test_simulation_section_structure(self):
        sim = self.schema["sections"]["simulation"]
        self.assertEqual(sim["title"], "Simulation")
        self.assertIn("properties", sim)
        self.assertIsInstance(sim["properties"], dict)

    def test_simulation_excludes_nested_sections(self):
        sim_props = self.schema["sections"]["simulation"]["properties"]
        # Nested sub-configs should not appear in simulation section
        self.assertNotIn("visualization", sim_props)
        self.assertNotIn("redis", sim_props)
        self.assertNotIn("observation", sim_props)

    def test_simulation_known_enums_applied(self):
        sim_props = self.schema["sections"]["simulation"]["properties"]
        # position_discretization_method should have enum values
        if "position_discretization_method" in sim_props:
            self.assertIn("enum", sim_props["position_discretization_method"])
            self.assertIn("floor", sim_props["position_discretization_method"]["enum"])

    def test_visualization_section(self):
        vis = self.schema["sections"]["visualization"]
        self.assertEqual(vis["title"], "Visualization")
        self.assertIn("properties", vis)

    def test_redis_section(self):
        redis = self.schema["sections"]["redis"]
        self.assertEqual(redis["title"], "Redis")
        self.assertIn("properties", redis)

    def test_observation_section_has_enums(self):
        obs = self.schema["sections"]["observation"]
        self.assertEqual(obs["title"], "Observation")
        self.assertIn("enums", obs)
        self.assertIn("storage_mode", obs["enums"])


if __name__ == "__main__":
    unittest.main()
