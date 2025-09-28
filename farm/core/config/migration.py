"""
Configuration migration system for version compatibility.

This module provides the ConfigurationMigrator class and related components
for handling configuration migration between different versions, including
automated migration scripts and version compatibility checking.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from .exceptions import ConfigurationMigrationError, ConfigurationError
from .hierarchical import HierarchicalConfig

logger = logging.getLogger(__name__)


@dataclass
class MigrationTransformation:
    """Individual transformation within a configuration migration.
    
    Represents a single transformation operation that can be applied
    to a configuration during migration.
    """
    
    operation: str  # Type of transformation (rename, move, delete, add, modify)
    source_path: Optional[str] = None  # Source configuration path
    target_path: Optional[str] = None  # Target configuration path
    value: Any = None  # Value for add/modify operations
    condition: Optional[Dict[str, Any]] = None  # Condition for applying transformation
    description: str = ""  # Human-readable description
    
    def apply(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this transformation to a configuration.
        
        Args:
            config: Configuration dictionary to transform
            
        Returns:
            Transformed configuration dictionary
        """
        if self.condition and not self._evaluate_condition(config):
            return config
        
        try:
            if self.operation == "rename":
                return self._apply_rename(config)
            elif self.operation == "move":
                return self._apply_move(config)
            elif self.operation == "delete":
                return self._apply_delete(config)
            elif self.operation == "add":
                return self._apply_add(config)
            elif self.operation == "modify":
                return self._apply_modify(config)
            elif self.operation == "merge":
                return self._apply_merge(config)
            elif self.operation == "split":
                return self._apply_split(config)
            else:
                logger.warning(f"Unknown transformation operation: {self.operation}")
                return config
                
        except Exception as e:
            raise ConfigurationMigrationError(
                from_version="unknown",
                to_version="unknown",
                message=f"Failed to apply transformation {self.operation}: {str(e)}",
                migration_step=self.description
            )
    
    def _evaluate_condition(self, config: Dict[str, Any]) -> bool:
        """Evaluate the condition for applying this transformation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if condition is met, False otherwise
        """
        if not self.condition:
            return True
        
        for key, expected_value in self.condition.items():
            actual_value = self._get_nested_value(config, key)
            if actual_value != expected_value:
                return False
        
        return True
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get a nested value from configuration using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path to the value
            
        Returns:
            Value at the specified path, or None if not found
        """
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value in configuration using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path to set
            value: Value to set
        """
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _apply_rename(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rename transformation."""
        if not self.source_path or not self.target_path:
            return config
        
        source_value = self._get_nested_value(config, self.source_path)
        if source_value is not None:
            self._set_nested_value(config, self.target_path, source_value)
            self._delete_nested_value(config, self.source_path)
        
        return config
    
    def _apply_move(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply move transformation (same as rename)."""
        return self._apply_rename(config)
    
    def _apply_delete(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply delete transformation."""
        if self.source_path:
            self._delete_nested_value(config, self.source_path)
        return config
    
    def _apply_add(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply add transformation."""
        if self.target_path and self.value is not None:
            self._set_nested_value(config, self.target_path, self.value)
        return config
    
    def _apply_modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modify transformation."""
        if self.source_path and self.value is not None:
            self._set_nested_value(config, self.source_path, self.value)
        return config
    
    def _apply_merge(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply merge transformation."""
        if not self.source_path or not self.target_path:
            return config
        
        source_value = self._get_nested_value(config, self.source_path)
        target_value = self._get_nested_value(config, self.target_path)
        
        if source_value is not None:
            if target_value is None:
                self._set_nested_value(config, self.target_path, source_value)
            elif isinstance(target_value, dict) and isinstance(source_value, dict):
                merged = {**target_value, **source_value}
                self._set_nested_value(config, self.target_path, merged)
            else:
                # Overwrite target with source
                self._set_nested_value(config, self.target_path, source_value)
            
            self._delete_nested_value(config, self.source_path)
        
        return config
    
    def _apply_split(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply split transformation."""
        if not self.source_path or not self.target_path:
            return config
        
        source_value = self._get_nested_value(config, self.source_path)
        if source_value is not None and isinstance(source_value, dict):
            # Split dictionary into multiple paths
            for key, value in source_value.items():
                split_path = f"{self.target_path}.{key}"
                self._set_nested_value(config, split_path, value)
            
            self._delete_nested_value(config, self.source_path)
        
        return config
    
    def _delete_nested_value(self, config: Dict[str, Any], path: str) -> None:
        """Delete a nested value from configuration.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path to delete
        """
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return
        
        if isinstance(current, dict) and keys[-1] in current:
            del current[keys[-1]]


@dataclass
class ConfigurationMigration:
    """Individual configuration migration between two versions.
    
    Represents a complete migration from one configuration version
    to another, including all necessary transformations.
    """
    
    from_version: str
    to_version: str
    transformations: List[MigrationTransformation] = field(default_factory=list)
    description: str = ""
    author: str = ""
    date: str = ""
    
    def apply(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply migration transformations to configuration.
        
        Args:
            config: Configuration dictionary to migrate
            
        Returns:
            Migrated configuration dictionary
        """
        migrated_config = config.copy()
        
        logger.info(f"Applying migration from {self.from_version} to {self.to_version}")
        
        for i, transformation in enumerate(self.transformations):
            try:
                logger.debug(f"Applying transformation {i+1}/{len(self.transformations)}: {transformation.operation}")
                migrated_config = transformation.apply(migrated_config)
            except Exception as e:
                raise ConfigurationMigrationError(
                    from_version=self.from_version,
                    to_version=self.to_version,
                    message=f"Failed to apply transformation {i+1}: {str(e)}",
                    migration_step=transformation.description,
                    original_error=e
                )
        
        logger.info(f"Migration from {self.from_version} to {self.to_version} completed successfully")
        return migrated_config
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate that this migration can be applied to the configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check if configuration has required source paths for transformations
        for transformation in self.transformations:
            if transformation.source_path:
                source_value = self._get_nested_value(config, transformation.source_path)
                if source_value is None and transformation.operation in ["rename", "move", "delete", "merge", "split"]:
                    errors.append(f"Source path '{transformation.source_path}' not found for {transformation.operation} transformation")
        
        return errors
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get a nested value from configuration using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


class ConfigurationMigrator:
    """Handles configuration migration for version compatibility.
    
    This class manages the migration of configuration from one version
    to another, including loading migration scripts and applying
    transformations in the correct order.
    """
    
    def __init__(self, migrations_dir: Optional[str] = None):
        """Initialize configuration migrator.
        
        Args:
            migrations_dir: Directory containing migration scripts
                          If None, will use default migrations directory
        """
        self.migrations_dir = Path(migrations_dir) if migrations_dir else Path("config/migrations")
        self.migrations: Dict[str, ConfigurationMigration] = {}
        self.migration_chain: Dict[str, str] = {}  # Maps from_version to to_version
        
        self._load_migrations()
    
    def _load_migrations(self) -> None:
        """Load all migration scripts from the migrations directory."""
        if not self.migrations_dir.exists():
            logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            return
        
        migration_files = list(self.migrations_dir.glob("*.yaml")) + list(self.migrations_dir.glob("*.json"))
        
        for migration_file in migration_files:
            try:
                migration = self._load_migration_file(migration_file)
                if migration:
                    migration_key = f"{migration.from_version}_to_{migration.to_version}"
                    self.migrations[migration_key] = migration
                    self.migration_chain[migration.from_version] = migration.to_version
                    logger.debug(f"Loaded migration: {migration_key}")
            except Exception as e:
                logger.error(f"Failed to load migration file {migration_file}: {e}")
    
    def _load_migration_file(self, file_path: Path) -> Optional[ConfigurationMigration]:
        """Load a single migration file.
        
        Args:
            file_path: Path to the migration file
            
        Returns:
            ConfigurationMigration object or None if loading failed
        """
        try:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.warning(f"Unsupported migration file format: {file_path}")
                return None
            
            return self._parse_migration_data(data, file_path.stem)
            
        except Exception as e:
            logger.error(f"Failed to load migration file {file_path}: {e}")
            return None
    
    def _parse_migration_data(self, data: Dict[str, Any], filename: str) -> ConfigurationMigration:
        """Parse migration data from file content.
        
        Args:
            data: Migration data dictionary
            filename: Name of the migration file
            
        Returns:
            ConfigurationMigration object
        """
        from_version = data.get('from_version', '')
        to_version = data.get('to_version', '')
        description = data.get('description', f'Migration from {from_version} to {to_version}')
        author = data.get('author', '')
        date = data.get('date', '')
        
        transformations = []
        for transform_data in data.get('transformations', []):
            transformation = MigrationTransformation(
                operation=transform_data.get('operation', ''),
                source_path=transform_data.get('source_path'),
                target_path=transform_data.get('target_path'),
                value=transform_data.get('value'),
                condition=transform_data.get('condition'),
                description=transform_data.get('description', '')
            )
            transformations.append(transformation)
        
        return ConfigurationMigration(
            from_version=from_version,
            to_version=to_version,
            transformations=transformations,
            description=description,
            author=author,
            date=date
        )
    
    def migrate_config(self, config: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Migrate configuration from one version to another.
        
        Args:
            config: Configuration dictionary to migrate
            from_version: Source version
            to_version: Target version
            
        Returns:
            Migrated configuration dictionary
            
        Raises:
            ConfigurationMigrationError: If migration fails
        """
        if from_version == to_version:
            logger.debug(f"Source and target versions are the same: {from_version}")
            return config
        
        # Find migration path
        migration_path = self._find_migration_path(from_version, to_version)
        if not migration_path:
            raise ConfigurationMigrationError(
                from_version=from_version,
                to_version=to_version,
                message=f"No migration path found from {from_version} to {to_version}"
            )
        
        logger.info(f"Migration path: {' -> '.join(migration_path)}")
        
        # Apply migrations in sequence
        current_config = config.copy()
        current_version = from_version
        
        for next_version in migration_path[1:]:
            migration_key = f"{current_version}_to_{next_version}"
            migration = self.migrations.get(migration_key)
            
            if not migration:
                raise ConfigurationMigrationError(
                    from_version=current_version,
                    to_version=next_version,
                    message=f"Migration not found: {migration_key}"
                )
            
            # Validate migration can be applied
            validation_errors = migration.validate(current_config)
            if validation_errors:
                logger.warning(f"Migration validation warnings: {validation_errors}")
            
            # Apply migration
            current_config = migration.apply(current_config)
            current_version = next_version
        
        return current_config
    
    def _find_migration_path(self, from_version: str, to_version: str) -> Optional[List[str]]:
        """Find the shortest migration path between two versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            List of versions in the migration path, or None if no path exists
        """
        if from_version == to_version:
            return [from_version]
        
        # Use BFS to find shortest path
        queue = [(from_version, [from_version])]
        visited = {from_version}
        
        while queue:
            current_version, path = queue.pop(0)
            
            if current_version == to_version:
                return path
            
            # Find next version in migration chain
            next_version = self.migration_chain.get(current_version)
            if next_version and next_version not in visited:
                visited.add(next_version)
                queue.append((next_version, path + [next_version]))
        
        return None
    
    def get_available_versions(self) -> List[str]:
        """Get list of all available configuration versions.
        
        Returns:
            List of version strings
        """
        versions = set()
        for migration in self.migrations.values():
            versions.add(migration.from_version)
            versions.add(migration.to_version)
        return sorted(list(versions))
    
    def get_migration_chain(self) -> Dict[str, str]:
        """Get the complete migration chain.
        
        Returns:
            Dictionary mapping from_version to to_version
        """
        return self.migration_chain.copy()
    
    def validate_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Validate that a migration path exists between two versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        migration_path = self._find_migration_path(from_version, to_version)
        if not migration_path:
            errors.append(f"No migration path found from {from_version} to {to_version}")
        
        return errors
    
    def create_migration_script(self, from_version: str, to_version: str, output_path: str) -> None:
        """Create a template migration script.
        
        Args:
            from_version: Source version
            to_version: Target version
            output_path: Path to save the migration script
        """
        template = {
            'from_version': from_version,
            'to_version': to_version,
            'description': f'Migration from {from_version} to {to_version}',
            'author': '',
            'date': '',
            'transformations': [
                {
                    'operation': 'rename',
                    'source_path': 'old_key',
                    'target_path': 'new_key',
                    'description': 'Rename old_key to new_key'
                },
                {
                    'operation': 'add',
                    'target_path': 'new_setting',
                    'value': 'default_value',
                    'description': 'Add new setting with default value'
                }
            ]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(template, f, default_flow_style=False, indent=2)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2)
        
        logger.info(f"Created migration script template: {output_path}")


class ConfigurationVersionDetector:
    """Detects configuration version from configuration content.
    
    This class analyzes configuration content to determine its version,
    either from explicit version fields or by analyzing structure and content.
    """
    
    def __init__(self):
        """Initialize version detector."""
        self.version_patterns = {
            '1.0': self._detect_v1_0,
            '1.1': self._detect_v1_1,
            '1.2': self._detect_v1_2,
            '2.0': self._detect_v2_0
        }
    
    def detect_version(self, config: Dict[str, Any]) -> str:
        """Detect configuration version.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Detected version string
        """
        # First check for explicit version field
        explicit_version = config.get('config_version') or config.get('version')
        if explicit_version:
            return str(explicit_version)
        
        # Try to detect version by analyzing structure
        for version, detector in self.version_patterns.items():
            if detector(config):
                return version
        
        # Default to latest version if no pattern matches
        return '2.0'
    
    def _detect_v1_0(self, config: Dict[str, Any]) -> bool:
        """Detect version 1.0 configuration."""
        # Version 1.0 had flat structure with specific keys
        v1_0_keys = ['simulation_id', 'max_steps', 'learning_rate']
        return all(key in config for key in v1_0_keys) and 'agent_parameters' not in config
    
    def _detect_v1_1(self, config: Dict[str, Any]) -> bool:
        """Detect version 1.1 configuration."""
        # Version 1.1 added agent_parameters
        return 'agent_parameters' in config and 'visualization' not in config
    
    def _detect_v1_2(self, config: Dict[str, Any]) -> bool:
        """Detect version 1.2 configuration."""
        # Version 1.2 added visualization config
        return 'visualization' in config and 'redis' not in config
    
    def _detect_v2_0(self, config: Dict[str, Any]) -> bool:
        """Detect version 2.0 configuration."""
        # Version 2.0 added redis config and hierarchical structure
        return 'redis' in config and isinstance(config.get('redis'), dict)