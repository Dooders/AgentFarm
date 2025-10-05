"""
Metrics Registry Utility

Loads and provides access to the global metrics registry from JSON.
Supports module-specific queries, tag-based filtering, and metric validation.
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path


class MetricsRegistry:
    """Utility class for accessing the global metrics registry."""

    def __init__(self, registry_path: Optional[str] = None):
        """Initialize the metrics registry.

        Args:
            registry_path: Path to the metrics registry JSON file.
                          Defaults to 'metrics_registry.json' in project root.
        """
        if registry_path is None:
            # Default to project root
            project_root = Path(__file__).parent.parent.parent
            registry_path = project_root / "metrics_registry.json"

        self.registry_path = Path(registry_path)
        self._registry = None

    @property
    def registry(self) -> Dict[str, Any]:
        """Lazy load the registry."""
        if self._registry is None:
            self._load_registry()
        return self._registry

    def _load_registry(self):
        """Load the registry from JSON file."""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Metrics registry not found at: {self.registry_path}")

        with open(self.registry_path, 'r') as f:
            self._registry = json.load(f)

    def get_metric_definition(self, category: str, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get definition for a specific metric.

        Args:
            category: Metric category (e.g., 'action_statistics')
            metric_name: Name of the specific metric

        Returns:
            Metric definition dictionary or None if not found
        """
        try:
            return self.registry["metric_categories"][category]["metrics"][metric_name]
        except KeyError:
            return None

    def get_category_metrics(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all metrics for a specific category.

        Args:
            category: Metric category name

        Returns:
            Dictionary of metric definitions
        """
        try:
            return self.registry["metric_categories"][category]["metrics"]
        except KeyError:
            return {}

    def get_all_categories(self) -> List[str]:
        """Get list of all metric categories.

        Returns:
            List of category names
        """
        return list(self.registry["metric_categories"].keys())

    def get_metrics_by_tag(self, tag: str) -> Dict[str, Dict[str, Any]]:
        """Get all metrics that have a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            Dictionary of matching metrics with their full definitions
        """
        results = {}
        for category_name, category_data in self.registry["metric_categories"].items():
            for metric_name, metric_def in category_data["metrics"].items():
                if tag in metric_def.get("tags", []):
                    results[f"{category_name}.{metric_name}"] = metric_def
        return results

    def get_metrics_by_module(self, module: str) -> Dict[str, Dict[str, Any]]:
        """Get all metrics from a specific analysis module.

        Args:
            module: Module name (e.g., 'actions', 'agents', 'learning')

        Returns:
            Dictionary of metrics grouped by category
        """
        results = {}
        for category_name, category_data in self.registry["metric_categories"].items():
            if category_data.get("module") == module:
                results[category_name] = category_data["metrics"]
        return results

    def get_all_modules(self) -> List[str]:
        """Get list of all available analysis modules.

        Returns:
            List of unique module names
        """
        modules = set()
        for category_data in self.registry["metric_categories"].values():
            module = category_data.get("module")
            if module:
                modules.add(module)
        return sorted(list(modules))

    def get_module_categories(self, module: str) -> List[str]:
        """Get all categories belonging to a specific module.

        Args:
            module: Module name

        Returns:
            List of category names
        """
        categories = []
        for category_name, category_data in self.registry["metric_categories"].items():
            if category_data.get("module") == module:
                categories.append(category_name)
        return categories

    def search_metrics(self, query: str, search_type: str = "any") -> Dict[str, Dict[str, Any]]:
        """Search metrics by name, description, or tags.

        Args:
            query: Search term (case-insensitive)
            search_type: Type of search - "name", "description", "tags", or "any"

        Returns:
            Dictionary of matching metrics
        """
        results = {}
        query_lower = query.lower()

        for category_name, category_data in self.registry["metric_categories"].items():
            for metric_name, metric_def in category_data["metrics"].items():
                matches = False

                if search_type in ["name", "any"]:
                    if query_lower in metric_name.lower():
                        matches = True

                if search_type in ["description", "any"] and not matches:
                    if query_lower in metric_def.get("description", "").lower():
                        matches = True

                if search_type in ["tags", "any"] and not matches:
                    tags = [tag.lower() for tag in metric_def.get("tags", [])]
                    if any(query_lower in tag for tag in tags):
                        matches = True

                if matches:
                    results[f"{category_name}.{metric_name}"] = metric_def

        return results

    def validate_metric_data(self, category: str, metric_name: str, value: Any) -> List[str]:
        """Validate a single metric value against its definition.

        Args:
            category: Metric category
            metric_name: Metric name
            value: Value to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        definition = self.get_metric_definition(category, metric_name)

        if definition is None:
            errors.append(f"Unknown metric: {category}.{metric_name}")
            return errors

        expected_type = definition["data_type"]

        # Type validation
        if expected_type == "float" and not isinstance(value, (int, float)):
            errors.append(f"Expected float, got {type(value).__name__}")
        elif expected_type == "int" and not isinstance(value, int):
            errors.append(f"Expected int, got {type(value).__name__}")
        elif expected_type == "str" and not isinstance(value, str):
            errors.append(f"Expected str, got {type(value).__name__}")
        elif expected_type == "dict" and not isinstance(value, dict):
            errors.append(f"Expected dict, got {type(value).__name__}")
        elif expected_type == "list" and not isinstance(value, list):
            errors.append(f"Expected list, got {type(value).__name__}")

        # Range validation for numeric types
        if expected_type in ["float", "int"] and "range_description" in definition:
            range_desc = definition["range_description"]
            if range_desc and " to " in range_desc:
                try:
                    min_val, max_val = map(float, range_desc.split(" to "))
                    if not (min_val <= float(value) <= max_val):
                        errors.append(f"Value {value} outside valid range [{min_val}, {max_val}]")
                except (ValueError, TypeError):
                    pass  # Skip range validation if parsing fails

        return errors

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire registry.

        Returns:
            Summary dictionary
        """
        summary = {
            "version": self.registry.get("version", "unknown"),
            "total_categories": len(self.registry["metric_categories"]),
            "categories": {}
        }

        for cat_name, cat_data in self.registry["metric_categories"].items():
            summary["categories"][cat_name] = {
                "description": cat_data["description"],
                "metric_count": len(cat_data["metrics"]),
                "metrics": list(cat_data["metrics"].keys())
            }

        return summary

    def export_category_to_json(self, category: str, output_path: str):
        """Export a specific category to a separate JSON file.

        Args:
            category: Category to export
            output_path: Path for the output file
        """
        category_data = self.get_category_metrics(category)
        if category_data:
            with open(output_path, 'w') as f:
                json.dump({
                    "category": category,
                    "description": self.registry["metric_categories"][category]["description"],
                    "metrics": category_data
                }, f, indent=2)

    def list_all_tags(self) -> List[str]:
        """Get list of all available tags.

        Returns:
            Sorted list of tag names
        """
        return sorted(self.registry.get("tags", {}).keys())


# Global instance for convenience
_default_registry = None

def get_metrics_registry() -> MetricsRegistry:
    """Get the default global metrics registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = MetricsRegistry()
    return _default_registry


# Convenience functions
def get_metric_definition(category: str, metric_name: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get metric definition."""
    return get_metrics_registry().get_metric_definition(category, metric_name)

def get_category_metrics(category: str) -> Dict[str, Dict[str, Any]]:
    """Convenience function to get category metrics."""
    return get_metrics_registry().get_category_metrics(category)

def validate_metric_data(category: str, metric_name: str, value: Any) -> List[str]:
    """Convenience function to validate metric data."""
    return get_metrics_registry().validate_metric_data(category, metric_name, value)

def get_registry_summary() -> Dict[str, Any]:
    """Convenience function to get registry summary."""
    return get_metrics_registry().get_registry_summary()

def get_metrics_by_module(module: str) -> Dict[str, Dict[str, Any]]:
    """Convenience function to get metrics by module."""
    return get_metrics_registry().get_metrics_by_module(module)

def get_all_modules() -> List[str]:
    """Convenience function to get all modules."""
    return get_metrics_registry().get_all_modules()

def get_module_categories(module: str) -> List[str]:
    """Convenience function to get categories for a module."""
    return get_metrics_registry().get_module_categories(module)

def search_metrics(query: str, search_type: str = "any") -> Dict[str, Dict[str, Any]]:
    """Convenience function to search metrics."""
    return get_metrics_registry().search_metrics(query, search_type)


if __name__ == "__main__":
    # Example usage
    registry = get_metrics_registry()

    print("Metrics Registry Summary:")
    summary = registry.get_registry_summary()
    print(json.dumps(summary, indent=2))

    # Example validation
    print("\nValidating example data:")
    test_value = 0.74
    errors = registry.validate_metric_data("decision_patterns", "avg_success_rate", test_value)
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print(f"✓ Value {test_value} is valid for avg_success_rate")

    # Show metrics by tag
    reward_metrics = registry.get_metrics_by_tag("reward")
    print(f"\nFound {len(reward_metrics)} reward-related metrics")

    # Show module-specific queries
    modules = registry.get_all_modules()
    print(f"\nAvailable modules: {modules}")

    agents_metrics = registry.get_metrics_by_module('agents')
    print(f"Agents module has {len(agents_metrics)} categories")

    # Example search
    performance_search = registry.search_metrics("performance")
    print(f"Found {len(performance_search)} metrics containing 'performance'")
