"""Tests for farm/utils/config_utils.py (safe config attribute resolution)."""

from types import SimpleNamespace

from farm.utils.config_utils import (
    get_config_value,
    get_nested_then_flat,
    resolve_spatial_index_config,
)


class TestGetConfigValue:
    def test_returns_value_when_present_and_typed(self):
        config = SimpleNamespace(perception_radius=5)
        assert get_config_value(config, "perception_radius", 2) == 5

    def test_returns_default_when_config_is_none(self):
        assert get_config_value(None, "perception_radius", 2) == 2

    def test_returns_default_when_attribute_missing(self):
        assert get_config_value(SimpleNamespace(), "perception_radius", 2) == 2

    def test_returns_default_when_type_mismatch(self):
        config = SimpleNamespace(perception_radius="five")
        assert get_config_value(config, "perception_radius", 2) == 2

    def test_custom_expected_types(self):
        config = SimpleNamespace(name="agent")
        assert get_config_value(config, "name", "default", expected_types=(str,)) == "agent"

    def test_float_value_accepted_by_default(self):
        config = SimpleNamespace(ratio=0.5)
        assert get_config_value(config, "ratio", 1.0) == 0.5


class TestGetNestedThenFlat:
    def test_nested_value_wins_over_flat(self):
        config = SimpleNamespace(
            agent_behavior=SimpleNamespace(perception_radius=7),
            perception_radius=3,
        )
        result = get_nested_then_flat(
            config=config,
            nested_parent_attr="agent_behavior",
            nested_attr_name="perception_radius",
            flat_attr_name="perception_radius",
            default_value=1,
        )
        assert result == 7

    def test_falls_back_to_flat_when_nested_missing(self):
        config = SimpleNamespace(perception_radius=3)
        result = get_nested_then_flat(
            config=config,
            nested_parent_attr="agent_behavior",
            nested_attr_name="perception_radius",
            flat_attr_name="perception_radius",
            default_value=1,
        )
        assert result == 3

    def test_falls_back_to_default_when_neither_present(self):
        result = get_nested_then_flat(
            config=SimpleNamespace(),
            nested_parent_attr="agent_behavior",
            nested_attr_name="perception_radius",
            flat_attr_name="perception_radius",
            default_value=1,
        )
        assert result == 1

    def test_none_config_returns_default(self):
        result = get_nested_then_flat(
            config=None,
            nested_parent_attr="agent_behavior",
            nested_attr_name="perception_radius",
            flat_attr_name="perception_radius",
            default_value=1,
        )
        assert result == 1

    def test_nested_value_with_wrong_type_falls_through_to_flat(self):
        config = SimpleNamespace(
            agent_behavior=SimpleNamespace(perception_radius="seven"),
            perception_radius=3,
        )
        result = get_nested_then_flat(
            config=config,
            nested_parent_attr="agent_behavior",
            nested_attr_name="perception_radius",
            flat_attr_name="perception_radius",
            default_value=1,
        )
        assert result == 3

    def test_bool_values_resolved_by_default_types(self):
        config = SimpleNamespace(agent_behavior=SimpleNamespace(enabled=True))
        result = get_nested_then_flat(
            config=config,
            nested_parent_attr="agent_behavior",
            nested_attr_name="enabled",
            flat_attr_name=None,
            default_value=False,
        )
        assert result is True

    def test_skips_nested_lookup_when_parent_attr_none(self):
        config = SimpleNamespace(perception_radius=4)
        result = get_nested_then_flat(
            config=config,
            nested_parent_attr=None,
            nested_attr_name=None,
            flat_attr_name="perception_radius",
            default_value=1,
        )
        assert result == 4


class TestResolveSpatialIndexConfig:
    def test_returns_nested_spatial_index(self):
        spatial_index = SimpleNamespace(enabled=True)
        config = SimpleNamespace(environment=SimpleNamespace(spatial_index=spatial_index))
        assert resolve_spatial_index_config(config) is spatial_index

    def test_none_config_returns_none(self):
        assert resolve_spatial_index_config(None) is None

    def test_missing_environment_returns_none(self):
        assert resolve_spatial_index_config(SimpleNamespace()) is None

    def test_missing_spatial_index_returns_none(self):
        config = SimpleNamespace(environment=SimpleNamespace())
        assert resolve_spatial_index_config(config) is None
