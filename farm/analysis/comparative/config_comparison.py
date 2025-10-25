"""
Configuration comparison utilities.

This module provides functionality to compare simulation configurations
using DeepDiff and format the results for reporting.
"""

from typing import Any, Dict, List, Optional
from deepdiff import DeepDiff
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigComparison:
    """Handles comparison of simulation configurations."""
    
    def __init__(self, ignore_order: bool = True, ignore_string_case: bool = True):
        """Initialize configuration comparison.
        
        Args:
            ignore_order: Whether to ignore order in lists/dicts
            ignore_string_case: Whether to ignore case in string comparisons
        """
        self.ignore_order = ignore_order
        self.ignore_string_case = ignore_string_case
    
    def compare_configurations(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two configuration dictionaries using DeepDiff.
        
        Args:
            config1: First configuration dictionary
            config2: Second configuration dictionary
            
        Returns:
            Dictionary containing comparison results
        """
        if not config1 and not config2:
            return {'status': 'both_empty', 'differences': {}}
        
        if not config1:
            return {'status': 'config1_empty', 'differences': {'added': list(config2.keys())}}
        
        if not config2:
            return {'status': 'config2_empty', 'differences': {'removed': list(config1.keys())}}
        
        try:
            # Configure DeepDiff options
            diff_options = {
                'ignore_order': self.ignore_order,
                'ignore_string_case': self.ignore_string_case,
                'exclude_paths': self._get_excluded_paths(),
                'exclude_regex_paths': self._get_excluded_regex_paths()
            }
            
            # Perform comparison
            diff = DeepDiff(config1, config2, **diff_options)
            
            # Convert DeepDiff result to our format
            result = self._format_deepdiff_result(diff)
            
            logger.info(f"Configuration comparison completed: {len(result.get('differences', {}))} differences found")
            return result
            
        except Exception as e:
            logger.error(f"Error comparing configurations: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'differences': {}
            }
    
    def _get_excluded_paths(self) -> List[str]:
        """Get paths to exclude from comparison (e.g., timestamps, IDs)."""
        return [
            'timestamp',
            'created_at',
            'updated_at',
            'id',
            'simulation_id',
            'run_id',
            'experiment_id',
            'version',
            'build_info',
            'git_hash',
            'git_commit'
        ]
    
    def _get_excluded_regex_paths(self) -> List[str]:
        """Get regex patterns for paths to exclude from comparison."""
        return [
            r'.*timestamp.*',
            r'.*time.*',
            r'.*date.*',
            r'.*id$',
            r'.*_id$',
            r'.*hash.*',
            r'.*version.*'
        ]
    
    def _format_deepdiff_result(self, diff: DeepDiff) -> Dict[str, Any]:
        """Format DeepDiff result into our standard format.
        
        Args:
            diff: DeepDiff result object
            
        Returns:
            Formatted comparison result
        """
        result = {
            'status': 'compared',
            'differences': {},
            'summary': {
                'total_changes': 0,
                'added_items': 0,
                'removed_items': 0,
                'changed_items': 0,
                'type_changes': 0
            }
        }
        
        if not diff:
            return result
        
        # Process different types of differences
        differences = {}
        
        # Dictionary item added
        if 'dictionary_item_added' in diff:
            added = diff['dictionary_item_added']
            differences['added'] = self._format_path_changes(added)
            result['summary']['added_items'] = len(added)
        
        # Dictionary item removed
        if 'dictionary_item_removed' in diff:
            removed = diff['dictionary_item_removed']
            differences['removed'] = self._format_path_changes(removed)
            result['summary']['removed_items'] = len(removed)
        
        # Values changed
        if 'values_changed' in diff:
            changed = diff['values_changed']
            differences['changed'] = self._format_value_changes(changed)
            result['summary']['changed_items'] = len(changed)
        
        # Type changes
        if 'type_changes' in diff:
            type_changes = diff['type_changes']
            differences['type_changes'] = self._format_type_changes(type_changes)
            result['summary']['type_changes'] = len(type_changes)
        
        # Iterable items added/removed
        if 'iterable_item_added' in diff:
            added = diff['iterable_item_added']
            differences['iterable_added'] = self._format_path_changes(added)
            result['summary']['added_items'] += len(added)
        
        if 'iterable_item_removed' in diff:
            removed = diff['iterable_item_removed']
            differences['iterable_removed'] = self._format_path_changes(removed)
            result['summary']['removed_items'] += len(removed)
        
        result['differences'] = differences
        result['summary']['total_changes'] = sum(result['summary'].values())
        
        return result
    
    def _format_path_changes(self, changes) -> List[Dict[str, Any]]:
        """Format path-based changes (added/removed items).
        
        Args:
            changes: Dictionary of path changes from DeepDiff
            
        Returns:
            List of formatted change dictionaries
        """
        formatted = []
        # Handle both dict and SetOrdered types from DeepDiff
        if hasattr(changes, 'items'):
            items = changes.items()
        else:
            # Handle SetOrdered or other iterable types
            items = [(str(item), item) for item in changes]
        
        for path, value in items:
            formatted.append({
                'path': path,
                'value': value,
                'path_parts': path.split('.') if isinstance(path, str) else [str(path)]
            })
        return formatted
    
    def _format_value_changes(self, changes: Dict) -> List[Dict[str, Any]]:
        """Format value changes.
        
        Args:
            changes: Dictionary of value changes from DeepDiff
            
        Returns:
            List of formatted change dictionaries
        """
        formatted = []
        for path, change_info in changes.items():
            formatted.append({
                'path': path,
                'old_value': change_info.get('old_value'),
                'new_value': change_info.get('new_value'),
                'path_parts': path.split('.') if isinstance(path, str) else [str(path)]
            })
        return formatted
    
    def _format_type_changes(self, changes: Dict) -> List[Dict[str, Any]]:
        """Format type changes.
        
        Args:
            changes: Dictionary of type changes from DeepDiff
            
        Returns:
            List of formatted change dictionaries
        """
        formatted = []
        for path, change_info in changes.items():
            formatted.append({
                'path': path,
                'old_type': change_info.get('old_type'),
                'new_type': change_info.get('new_type'),
                'old_value': change_info.get('old_value'),
                'new_value': change_info.get('new_value'),
                'path_parts': path.split('.') if isinstance(path, str) else [str(path)]
            })
        return formatted
    
    def format_config_differences(self, differences: Dict[str, Any]) -> str:
        """Format configuration differences for human-readable reporting.
        
        Args:
            differences: Comparison result from compare_configurations
            
        Returns:
            Formatted string describing the differences
        """
        if not differences or differences.get('status') == 'both_empty':
            return "No configuration differences found."
        
        if differences.get('status') == 'error':
            return f"Error comparing configurations: {differences.get('error', 'Unknown error')}"
        
        lines = []
        lines.append("Configuration Differences:")
        lines.append("=" * 50)
        
        # Summary
        summary = differences.get('summary', {})
        if summary:
            lines.append(f"Total changes: {summary.get('total_changes', 0)}")
            lines.append(f"Added items: {summary.get('added_items', 0)}")
            lines.append(f"Removed items: {summary.get('removed_items', 0)}")
            lines.append(f"Changed items: {summary.get('changed_items', 0)}")
            lines.append(f"Type changes: {summary.get('type_changes', 0)}")
            lines.append("")
        
        # Detailed differences
        diff_details = differences.get('differences', {})
        
        # Added items
        if 'added' in diff_details:
            lines.append("Added Configuration Items:")
            for item in diff_details['added']:
                lines.append(f"  + {item['path']}: {item['value']}")
            lines.append("")
        
        # Removed items
        if 'removed' in diff_details:
            lines.append("Removed Configuration Items:")
            for item in diff_details['removed']:
                lines.append(f"  - {item['path']}: {item['value']}")
            lines.append("")
        
        # Changed items
        if 'changed' in diff_details:
            lines.append("Changed Configuration Items:")
            for item in diff_details['changed']:
                lines.append(f"  ~ {item['path']}:")
                lines.append(f"    From: {item['old_value']}")
                lines.append(f"    To:   {item['new_value']}")
            lines.append("")
        
        # Type changes
        if 'type_changes' in diff_details:
            lines.append("Type Changes:")
            for item in diff_details['type_changes']:
                lines.append(f"  ! {item['path']}:")
                lines.append(f"    Type: {item['old_type']} -> {item['new_type']}")
                lines.append(f"    Value: {item['old_value']} -> {item['new_value']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_significant_changes(self, differences: Dict[str, Any], 
                              significant_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Identify significant configuration changes.
        
        Args:
            differences: Comparison result from compare_configurations
            significant_paths: List of paths to consider significant
            
        Returns:
            Dictionary containing only significant changes
        """
        if not differences or differences.get('status') != 'compared':
            return differences
        
        if not significant_paths:
            # Default significant paths
            significant_paths = [
                'environment',
                'agents',
                'simulation',
                'parameters',
                'settings',
                'config'
            ]
        
        significant = {
            'status': 'significant_changes',
            'differences': {},
            'summary': differences.get('summary', {})
        }
        
        diff_details = differences.get('differences', {})
        
        # Filter each type of change for significant paths
        for change_type in ['added', 'removed', 'changed', 'type_changes']:
            if change_type in diff_details:
                significant_items = []
                for item in diff_details[change_type]:
                    path = item.get('path', '')
                    if any(sig_path in path for sig_path in significant_paths):
                        significant_items.append(item)
                
                if significant_items:
                    significant['differences'][change_type] = significant_items
        
        return significant