"""
Tests for ConfigComparison class.
"""

import pytest
from deepdiff import DeepDiff

from farm.analysis.comparative.config_comparison import ConfigComparison


class TestConfigComparison:
    """Test cases for ConfigComparison."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparison = ConfigComparison()
    
    def test_compare_identical_configs(self):
        """Test comparison of identical configurations."""
        config1 = {
            "simulation": {
                "name": "test",
                "duration": 1000
            },
            "environment": {
                "width": 1000,
                "height": 1000
            }
        }
        config2 = config1.copy()
        
        result = self.comparison.compare_configurations(config1, config2)
        
        assert result["status"] == "compared"
        assert result["differences"] == {}
        assert result["summary"]["total_changes"] == 0
    
    def test_compare_different_configs(self):
        """Test comparison of different configurations."""
        config1 = {
            "simulation": {
                "name": "test1",
                "duration": 1000
            },
            "environment": {
                "width": 1000,
                "height": 1000
            }
        }
        config2 = {
            "simulation": {
                "name": "test2",
                "duration": 2000
            },
            "environment": {
                "width": 2000,
                "height": 1000
            },
            "new_section": {
                "param": "value"
            }
        }
        
        result = self.comparison.compare_configurations(config1, config2)
        
        assert result["status"] == "compared"
        assert "changed" in result["differences"]
        assert "added" in result["differences"]
        assert result["summary"]["total_changes"] > 0
        assert result["summary"]["changed_items"] > 0
        assert result["summary"]["added_items"] > 0
    
    def test_compare_empty_configs(self):
        """Test comparison of empty configurations."""
        result = self.comparison.compare_configurations({}, {})
        
        assert result["status"] == "both_empty"
        assert result["differences"] == {}
    
    def test_compare_one_empty_config(self):
        """Test comparison with one empty configuration."""
        config1 = {"simulation": {"name": "test"}}
        config2 = {}
        
        result = self.comparison.compare_configurations(config1, config2)
        
        assert result["status"] == "config2_empty"
        assert "removed" in result["differences"]
        assert "simulation" in result["differences"]["removed"][0]["path"]
    
    def test_compare_with_none_configs(self):
        """Test comparison with None configurations."""
        result = self.comparison.compare_configurations(None, None)
        
        assert result["status"] == "both_empty"
    
    def test_format_deepdiff_result_no_differences(self):
        """Test formatting DeepDiff result with no differences."""
        diff = DeepDiff({}, {})
        result = self.comparison._format_deepdiff_result(diff)
        
        assert result["status"] == "compared"
        assert result["differences"] == {}
        assert result["summary"]["total_changes"] == 0
    
    def test_format_deepdiff_result_with_differences(self):
        """Test formatting DeepDiff result with differences."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "b": 3, "c": 4}
        diff = DeepDiff(dict1, dict2)
        
        result = self.comparison._format_deepdiff_result(diff)
        
        assert result["status"] == "compared"
        assert "changed" in result["differences"]
        assert "added" in result["differences"]
        assert result["summary"]["changed_items"] == 1
        assert result["summary"]["added_items"] == 1
    
    def test_format_config_differences_no_differences(self):
        """Test formatting config differences with no differences."""
        differences = {
            "status": "compared",
            "differences": {},
            "summary": {"total_changes": 0}
        }
        
        formatted = self.comparison.format_config_differences(differences)
        
        assert "No configuration differences found" in formatted
    
    def test_format_config_differences_with_changes(self):
        """Test formatting config differences with changes."""
        differences = {
            "status": "compared",
            "differences": {
                "added": [{"path": "new_param", "value": "new_value"}],
                "removed": [{"path": "old_param", "value": "old_value"}],
                "changed": [{
                    "path": "simulation.duration",
                    "old_value": 1000,
                    "new_value": 2000
                }]
            },
            "summary": {
                "total_changes": 3,
                "added_items": 1,
                "removed_items": 1,
                "changed_items": 1
            }
        }
        
        formatted = self.comparison.format_config_differences(differences)
        
        assert "Configuration Differences:" in formatted
        assert "Total changes: 3" in formatted
        assert "Added Configuration Items:" in formatted
        assert "Removed Configuration Items:" in formatted
        assert "Changed Configuration Items:" in formatted
        assert "new_param" in formatted
        assert "old_param" in formatted
        assert "simulation.duration" in formatted
    
    def test_format_config_differences_error(self):
        """Test formatting config differences with error."""
        differences = {
            "status": "error",
            "error": "Test error message"
        }
        
        formatted = self.comparison.format_config_differences(differences)
        
        assert "Error comparing configurations: Test error message" in formatted
    
    def test_get_significant_changes(self):
        """Test identifying significant changes."""
        differences = {
            "status": "compared",
            "differences": {
                "added": [
                    {"path": "environment.new_param", "value": "value"},
                    {"path": "other_param", "value": "value"}
                ],
                "changed": [
                    {"path": "simulation.duration", "old_value": 1000, "new_value": 2000},
                    {"path": "unimportant.timestamp", "old_value": "old", "new_value": "new"}
                ]
            },
            "summary": {"total_changes": 4}
        }
        
        significant = self.comparison.get_significant_changes(differences)
        
        assert significant["status"] == "significant_changes"
        assert "added" in significant["differences"]
        assert "changed" in significant["differences"]
        
        # Should include environment and simulation changes
        added_paths = [item["path"] for item in significant["differences"]["added"]]
        changed_paths = [item["path"] for item in significant["differences"]["changed"]]
        
        assert "environment.new_param" in added_paths
        assert "simulation.duration" in changed_paths
        assert "unimportant.timestamp" not in changed_paths
    
    def test_get_significant_changes_custom_paths(self):
        """Test identifying significant changes with custom paths."""
        differences = {
            "status": "compared",
            "differences": {
                "changed": [
                    {"path": "custom.param", "old_value": 1, "new_value": 2}
                ]
            },
            "summary": {"total_changes": 1}
        }
        
        significant = self.comparison.get_significant_changes(
            differences, 
            significant_paths=["custom"]
        )
        
        assert "changed" in significant["differences"]
        assert len(significant["differences"]["changed"]) == 1
        assert significant["differences"]["changed"][0]["path"] == "custom.param"
    
    def test_excluded_paths(self):
        """Test that excluded paths are properly configured."""
        excluded = self.comparison._get_excluded_paths()
        
        assert "timestamp" in excluded
        assert "id" in excluded
        assert "simulation_id" in excluded
        assert "version" in excluded
    
    def test_excluded_regex_paths(self):
        """Test that excluded regex paths are properly configured."""
        regex_paths = self.comparison._get_excluded_regex_paths()
        
        assert any("timestamp" in path for path in regex_paths)
        assert any("id$" in path for path in regex_paths)
        assert any("_id$" in path for path in regex_paths)
    
    def test_format_path_changes(self):
        """Test formatting path-based changes."""
        changes = {
            "root.new_param": "new_value",
            "nested.param": "nested_value"
        }
        
        formatted = self.comparison._format_path_changes(changes)
        
        assert len(formatted) == 2
        assert any(item["path"] == "root.new_param" for item in formatted)
        assert any(item["path"] == "nested.param" for item in formatted)
        assert all("path_parts" in item for item in formatted)
    
    def test_format_value_changes(self):
        """Test formatting value changes."""
        changes = {
            "param1": {"old_value": 1, "new_value": 2},
            "param2": {"old_value": "old", "new_value": "new"}
        }
        
        formatted = self.comparison._format_value_changes(changes)
        
        assert len(formatted) == 2
        assert any(item["path"] == "param1" for item in formatted)
        assert any(item["path"] == "param2" for item in formatted)
        assert all("old_value" in item for item in formatted)
        assert all("new_value" in item for item in formatted)
    
    def test_format_type_changes(self):
        """Test formatting type changes."""
        changes = {
            "param": {
                "old_type": "int",
                "new_type": "str",
                "old_value": 1,
                "new_value": "1"
            }
        }
        
        formatted = self.comparison._format_type_changes(changes)
        
        assert len(formatted) == 1
        assert formatted[0]["path"] == "param"
        assert formatted[0]["old_type"] == "int"
        assert formatted[0]["new_type"] == "str"
        assert formatted[0]["old_value"] == 1
        assert formatted[0]["new_value"] == "1"