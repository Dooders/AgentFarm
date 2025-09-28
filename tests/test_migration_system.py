"""
Comprehensive tests for the configuration migration system.

This module tests the migration framework, transformation operations,
version detection, and automated migration tools.
"""

import pytest
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from farm.core.config import (
    ConfigurationMigrator,
    ConfigurationMigration,
    MigrationTransformation,
    ConfigurationVersionDetector,
    MigrationTool,
    ConfigurationMigrationError,
    ConfigurationError
)


class TestMigrationTransformation:
    """Test cases for MigrationTransformation class."""
    
    def test_rename_transformation(self):
        """Test rename transformation operation."""
        config = {
            'old_key': 'old_value',
            'other_key': 'other_value'
        }
        
        transformation = MigrationTransformation(
            operation='rename',
            source_path='old_key',
            target_path='new_key',
            description='Rename old_key to new_key'
        )
        
        result = transformation.apply(config)
        
        assert 'old_key' not in result
        assert result['new_key'] == 'old_value'
        assert result['other_key'] == 'other_value'
    
    def test_move_transformation(self):
        """Test move transformation operation."""
        config = {
            'source': {
                'nested': 'value'
            },
            'other': 'other_value'
        }
        
        transformation = MigrationTransformation(
            operation='move',
            source_path='source.nested',
            target_path='target.nested',
            description='Move nested value'
        )
        
        result = transformation.apply(config)
        
        assert 'source' not in result
        assert result['target']['nested'] == 'value'
        assert result['other'] == 'other_value'
    
    def test_delete_transformation(self):
        """Test delete transformation operation."""
        config = {
            'key_to_delete': 'value',
            'key_to_keep': 'value'
        }
        
        transformation = MigrationTransformation(
            operation='delete',
            source_path='key_to_delete',
            description='Delete key_to_delete'
        )
        
        result = transformation.apply(config)
        
        assert 'key_to_delete' not in result
        assert result['key_to_keep'] == 'value'
    
    def test_add_transformation(self):
        """Test add transformation operation."""
        config = {
            'existing_key': 'existing_value'
        }
        
        transformation = MigrationTransformation(
            operation='add',
            target_path='new_key',
            value='new_value',
            description='Add new key'
        )
        
        result = transformation.apply(config)
        
        assert result['existing_key'] == 'existing_value'
        assert result['new_key'] == 'new_value'
    
    def test_modify_transformation(self):
        """Test modify transformation operation."""
        config = {
            'key_to_modify': 'old_value',
            'other_key': 'other_value'
        }
        
        transformation = MigrationTransformation(
            operation='modify',
            source_path='key_to_modify',
            value='new_value',
            description='Modify existing key'
        )
        
        result = transformation.apply(config)
        
        assert result['key_to_modify'] == 'new_value'
        assert result['other_key'] == 'other_value'
    
    def test_merge_transformation(self):
        """Test merge transformation operation."""
        config = {
            'source': {
                'key1': 'value1',
                'key2': 'value2'
            },
            'target': {
                'key2': 'existing_value2',
                'key3': 'value3'
            }
        }
        
        transformation = MigrationTransformation(
            operation='merge',
            source_path='source',
            target_path='target',
            description='Merge source into target'
        )
        
        result = transformation.apply(config)
        
        assert 'source' not in result
        assert result['target']['key1'] == 'value1'
        assert result['target']['key2'] == 'value2'  # Source overwrites target
        assert result['target']['key3'] == 'value3'
    
    def test_split_transformation(self):
        """Test split transformation operation."""
        config = {
            'source': {
                'key1': 'value1',
                'key2': 'value2'
            },
            'other': 'other_value'
        }
        
        transformation = MigrationTransformation(
            operation='split',
            source_path='source',
            target_path='split',
            description='Split source into multiple keys'
        )
        
        result = transformation.apply(config)
        
        assert 'source' not in result
        assert result['split']['key1'] == 'value1'
        assert result['split']['key2'] == 'value2'
        assert result['other'] == 'other_value'
    
    def test_conditional_transformation(self):
        """Test transformation with condition."""
        config = {
            'condition_key': 'expected_value',
            'key_to_rename': 'value'
        }
        
        transformation = MigrationTransformation(
            operation='rename',
            source_path='key_to_rename',
            target_path='new_key',
            condition={'condition_key': 'expected_value'},
            description='Conditional rename'
        )
        
        result = transformation.apply(config)
        
        assert 'key_to_rename' not in result
        assert result['new_key'] == 'value'
        
        # Test with condition not met
        config['condition_key'] = 'different_value'
        result = transformation.apply(config)
        
        assert 'key_to_rename' in result
        assert 'new_key' not in result
    
    def test_nested_path_operations(self):
        """Test operations with nested paths."""
        config = {
            'level1': {
                'level2': {
                    'level3': 'deep_value'
                }
            }
        }
        
        transformation = MigrationTransformation(
            operation='rename',
            source_path='level1.level2.level3',
            target_path='shallow.value',
            description='Move deep nested value to shallow'
        )
        
        result = transformation.apply(config)
        
        assert result['shallow']['value'] == 'deep_value'
        assert 'level1' not in result


class TestConfigurationMigration:
    """Test cases for ConfigurationMigration class."""
    
    def test_simple_migration(self):
        """Test simple migration with multiple transformations."""
        config = {
            'old_key1': 'value1',
            'old_key2': 'value2',
            'keep_key': 'keep_value'
        }
        
        migration = ConfigurationMigration(
            from_version='1.0',
            to_version='1.1',
            transformations=[
                MigrationTransformation(
                    operation='rename',
                    source_path='old_key1',
                    target_path='new_key1',
                    description='Rename key1'
                ),
                MigrationTransformation(
                    operation='delete',
                    source_path='old_key2',
                    description='Delete key2'
                ),
                MigrationTransformation(
                    operation='add',
                    target_path='new_key3',
                    value='new_value3',
                    description='Add key3'
                )
            ],
            description='Test migration'
        )
        
        result = migration.apply(config)
        
        assert 'old_key1' not in result
        assert 'old_key2' not in result
        assert result['new_key1'] == 'value1'
        assert result['new_key3'] == 'new_value3'
        assert result['keep_key'] == 'keep_value'
    
    def test_migration_validation(self):
        """Test migration validation."""
        config = {
            'existing_key': 'value',
            'other_key': 'other_value'
        }
        
        migration = ConfigurationMigration(
            from_version='1.0',
            to_version='1.1',
            transformations=[
                MigrationTransformation(
                    operation='rename',
                    source_path='existing_key',
                    target_path='new_key',
                    description='Rename existing key'
                ),
                MigrationTransformation(
                    operation='rename',
                    source_path='missing_key',
                    target_path='new_key2',
                    description='Rename missing key'
                )
            ]
        )
        
        errors = migration.validate(config)
        
        assert len(errors) == 1
        assert 'missing_key' in errors[0]
    
    def test_migration_error_handling(self):
        """Test migration error handling."""
        config = {'key': 'value'}
        
        # Create a migration with invalid transformation
        migration = ConfigurationMigration(
            from_version='1.0',
            to_version='1.1',
            transformations=[
                MigrationTransformation(
                    operation='invalid_operation',
                    source_path='key',
                    target_path='new_key',
                    description='Invalid operation'
                )
            ]
        )
        
        with pytest.raises(ConfigurationMigrationError) as exc_info:
            migration.apply(config)
        
        assert 'invalid_operation' in str(exc_info.value)


class TestConfigurationMigrator:
    """Test cases for ConfigurationMigrator class."""
    
    @pytest.fixture
    def temp_migrations_dir(self):
        """Create temporary directory with test migration files."""
        temp_dir = tempfile.mkdtemp()
        migrations_dir = Path(temp_dir) / "migrations"
        migrations_dir.mkdir()
        
        # Create test migration files
        migration1 = {
            'from_version': '1.0',
            'to_version': '1.1',
            'description': 'Test migration 1.0 to 1.1',
            'transformations': [
                {
                    'operation': 'rename',
                    'source_path': 'old_key',
                    'target_path': 'new_key',
                    'description': 'Rename old_key to new_key'
                }
            ]
        }
        
        migration2 = {
            'from_version': '1.1',
            'to_version': '1.2',
            'description': 'Test migration 1.1 to 1.2',
            'transformations': [
                {
                    'operation': 'add',
                    'target_path': 'version_1_2_key',
                    'value': 'version_1_2_value',
                    'description': 'Add version 1.2 key'
                }
            ]
        }
        
        # Write migration files
        with open(migrations_dir / "v1.0_to_v1.1.yaml", 'w') as f:
            yaml.dump(migration1, f)
        
        with open(migrations_dir / "v1.1_to_v1.2.yaml", 'w') as f:
            yaml.dump(migration2, f)
        
        yield migrations_dir
        
        shutil.rmtree(temp_dir)
    
    def test_migrator_initialization(self, temp_migrations_dir):
        """Test migrator initialization and migration loading."""
        migrator = ConfigurationMigrator(str(temp_migrations_dir))
        
        assert len(migrator.migrations) == 2
        assert '1.0_to_1.1' in migrator.migrations
        assert '1.1_to_1.2' in migrator.migrations
        assert migrator.migration_chain['1.0'] == '1.1'
        assert migrator.migration_chain['1.1'] == '1.2'
    
    def test_single_migration(self, temp_migrations_dir):
        """Test single step migration."""
        migrator = ConfigurationMigrator(str(temp_migrations_dir))
        
        config = {
            'old_key': 'old_value',
            'other_key': 'other_value'
        }
        
        result = migrator.migrate_config(config, '1.0', '1.1')
        
        assert 'old_key' not in result
        assert result['new_key'] == 'old_value'
        assert result['other_key'] == 'other_value'
    
    def test_multi_step_migration(self, temp_migrations_dir):
        """Test multi-step migration."""
        migrator = ConfigurationMigrator(str(temp_migrations_dir))
        
        config = {
            'old_key': 'old_value',
            'other_key': 'other_value'
        }
        
        result = migrator.migrate_config(config, '1.0', '1.2')
        
        assert 'old_key' not in result
        assert result['new_key'] == 'old_value'
        assert result['version_1_2_key'] == 'version_1_2_value'
        assert result['other_key'] == 'other_value'
    
    def test_no_migration_needed(self, temp_migrations_dir):
        """Test migration when source and target versions are the same."""
        migrator = ConfigurationMigrator(str(temp_migrations_dir))
        
        config = {'key': 'value'}
        
        result = migrator.migrate_config(config, '1.1', '1.1')
        
        assert result == config
    
    def test_migration_path_not_found(self, temp_migrations_dir):
        """Test migration when no path exists between versions."""
        migrator = ConfigurationMigrator(str(temp_migrations_dir))
        
        config = {'key': 'value'}
        
        with pytest.raises(ConfigurationMigrationError) as exc_info:
            migrator.migrate_config(config, '1.0', '2.0')
        
        assert 'No migration path found' in str(exc_info.value)
    
    def test_get_available_versions(self, temp_migrations_dir):
        """Test getting available versions."""
        migrator = ConfigurationMigrator(str(temp_migrations_dir))
        
        versions = migrator.get_available_versions()
        
        assert '1.0' in versions
        assert '1.1' in versions
        assert '1.2' in versions
        assert len(versions) == 3
    
    def test_validate_migration_path(self, temp_migrations_dir):
        """Test migration path validation."""
        migrator = ConfigurationMigrator(str(temp_migrations_dir))
        
        # Valid path
        errors = migrator.validate_migration_path('1.0', '1.2')
        assert len(errors) == 0
        
        # Invalid path
        errors = migrator.validate_migration_path('1.0', '2.0')
        assert len(errors) == 1
        assert 'No migration path found' in errors[0]
    
    def test_create_migration_script(self, temp_migrations_dir):
        """Test creating migration script template."""
        migrator = ConfigurationMigrator(str(temp_migrations_dir))
        
        output_path = temp_migrations_dir / "test_migration.yaml"
        migrator.create_migration_script('1.2', '1.3', str(output_path))
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            content = yaml.safe_load(f)
        
        assert content['from_version'] == '1.2'
        assert content['to_version'] == '1.3'
        assert 'transformations' in content


class TestConfigurationVersionDetector:
    """Test cases for ConfigurationVersionDetector class."""
    
    def test_explicit_version_detection(self):
        """Test detection of explicit version field."""
        detector = ConfigurationVersionDetector()
        
        config = {
            'config_version': '1.5',
            'other_key': 'value'
        }
        
        version = detector.detect_version(config)
        assert version == '1.5'
        
        config = {
            'version': '2.0',
            'other_key': 'value'
        }
        
        version = detector.detect_version(config)
        assert version == '2.0'
    
    def test_v1_0_detection(self):
        """Test detection of version 1.0 configuration."""
        detector = ConfigurationVersionDetector()
        
        config = {
            'simulation_id': 'test',
            'max_steps': 1000,
            'learning_rate': 0.001
        }
        
        version = detector.detect_version(config)
        assert version == '1.0'
    
    def test_v1_1_detection(self):
        """Test detection of version 1.1 configuration."""
        detector = ConfigurationVersionDetector()
        
        config = {
            'simulation_id': 'test',
            'max_steps': 1000,
            'learning_rate': 0.001,
            'agent_parameters': {
                'SystemAgent': {}
            }
        }
        
        version = detector.detect_version(config)
        assert version == '1.1'
    
    def test_v1_2_detection(self):
        """Test detection of version 1.2 configuration."""
        detector = ConfigurationVersionDetector()
        
        config = {
            'simulation_id': 'test',
            'max_steps': 1000,
            'learning_rate': 0.001,
            'agent_parameters': {
                'SystemAgent': {}
            },
            'visualization': {
                'canvas_size': [400, 400]
            }
        }
        
        version = detector.detect_version(config)
        assert version == '1.2'
    
    def test_v2_0_detection(self):
        """Test detection of version 2.0 configuration."""
        detector = ConfigurationVersionDetector()
        
        config = {
            'simulation_id': 'test',
            'max_steps': 1000,
            'learning_rate': 0.001,
            'agent_parameters': {
                'SystemAgent': {}
            },
            'visualization': {
                'canvas_size': [400, 400]
            },
            'redis': {
                'host': 'localhost',
                'port': 6379
            }
        }
        
        version = detector.detect_version(config)
        assert version == '2.0'
    
    def test_default_version_detection(self):
        """Test default version detection for unknown configurations."""
        detector = ConfigurationVersionDetector()
        
        config = {
            'unknown_key': 'unknown_value'
        }
        
        version = detector.detect_version(config)
        assert version == '2.0'  # Default to latest version


class TestMigrationTool:
    """Test cases for MigrationTool class."""
    
    @pytest.fixture
    def temp_migrations_dir(self):
        """Create temporary directory with test migration files."""
        temp_dir = tempfile.mkdtemp()
        migrations_dir = Path(temp_dir) / "migrations"
        migrations_dir.mkdir()
        
        # Create simple test migration
        migration = {
            'from_version': '1.0',
            'to_version': '1.1',
            'description': 'Test migration',
            'transformations': [
                {
                    'operation': 'rename',
                    'source_path': 'old_key',
                    'target_path': 'new_key',
                    'description': 'Rename old_key to new_key'
                }
            ]
        }
        
        with open(migrations_dir / "v1.0_to_v1.1.yaml", 'w') as f:
            yaml.dump(migration, f)
        
        yield migrations_dir
        
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory with test configuration files."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir)
        
        # Create test configuration
        config = {
            'old_key': 'old_value',
            'other_key': 'other_value'
        }
        
        with open(config_dir / "test_config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        yield config_dir
        
        shutil.rmtree(temp_dir)
    
    def test_migrate_file(self, temp_migrations_dir, temp_config_dir):
        """Test migrating a single configuration file."""
        tool = MigrationTool(str(temp_migrations_dir))
        
        input_file = temp_config_dir / "test_config.yaml"
        output_file = temp_config_dir / "migrated_config.yaml"
        
        result = tool.migrate_file(
            str(input_file),
            str(output_file),
            '1.1'
        )
        
        assert result['success'] == True
        assert result['source_version'] == '1.0'
        assert result['target_version'] == '1.1'
        assert len(result['errors']) == 0
        
        # Verify output file
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            migrated_config = yaml.safe_load(f)
        
        assert 'old_key' not in migrated_config
        assert migrated_config['new_key'] == 'old_value'
        assert migrated_config['other_key'] == 'other_value'
    
    def test_migrate_file_with_explicit_source_version(self, temp_migrations_dir, temp_config_dir):
        """Test migrating a file with explicit source version."""
        tool = MigrationTool(str(temp_migrations_dir))
        
        input_file = temp_config_dir / "test_config.yaml"
        output_file = temp_config_dir / "migrated_config.yaml"
        
        result = tool.migrate_file(
            str(input_file),
            str(output_file),
            '1.1',
            source_version='1.0'
        )
        
        assert result['success'] == True
        assert result['source_version'] == '1.0'
        assert result['target_version'] == '1.1'
    
    def test_migrate_file_no_migration_needed(self, temp_migrations_dir, temp_config_dir):
        """Test migrating a file when no migration is needed."""
        tool = MigrationTool(str(temp_migrations_dir))
        
        input_file = temp_config_dir / "test_config.yaml"
        output_file = temp_config_dir / "migrated_config.yaml"
        
        result = tool.migrate_file(
            str(input_file),
            str(output_file),
            '1.0'  # Same as source version
        )
        
        assert result['success'] == True
        assert len(result['warnings']) > 0
        assert 'same' in result['warnings'][0].lower()
    
    def test_migrate_file_migration_path_not_found(self, temp_migrations_dir, temp_config_dir):
        """Test migrating a file when migration path is not found."""
        tool = MigrationTool(str(temp_migrations_dir))
        
        input_file = temp_config_dir / "test_config.yaml"
        output_file = temp_config_dir / "migrated_config.yaml"
        
        result = tool.migrate_file(
            str(input_file),
            str(output_file),
            '2.0'  # No migration path from 1.0 to 2.0
        )
        
        assert result['success'] == False
        assert len(result['errors']) > 0
        assert 'No migration path found' in result['errors'][0]
    
    def test_migrate_directory(self, temp_migrations_dir, temp_config_dir):
        """Test migrating all files in a directory."""
        tool = MigrationTool(str(temp_migrations_dir))
        
        # Create another config file
        config2 = {
            'old_key': 'value2',
            'other_key': 'other_value2'
        }
        
        with open(temp_config_dir / "test_config2.yaml", 'w') as f:
            yaml.dump(config2, f)
        
        output_dir = temp_config_dir / "migrated"
        
        result = tool.migrate_directory(
            str(temp_config_dir),
            str(output_dir),
            '1.1'
        )
        
        assert result['success'] == True
        assert result['files_processed'] == 2
        assert result['files_successful'] == 2
        assert result['files_failed'] == 0
        
        # Verify output files
        assert (output_dir / "test_config.yaml").exists()
        assert (output_dir / "test_config2.yaml").exists()
    
    def test_validate_migration(self, temp_migrations_dir, temp_config_dir):
        """Test migration validation."""
        tool = MigrationTool(str(temp_migrations_dir))
        
        config_file = temp_config_dir / "test_config.yaml"
        
        result = tool.validate_migration(
            str(config_file),
            '1.1'
        )
        
        assert result['valid'] == True
        assert result['source_version'] == '1.0'
        assert result['target_version'] == '1.1'
        assert len(result['errors']) == 0
    
    def test_validate_migration_invalid_path(self, temp_migrations_dir, temp_config_dir):
        """Test migration validation with invalid path."""
        tool = MigrationTool(str(temp_migrations_dir))
        
        config_file = temp_config_dir / "test_config.yaml"
        
        result = tool.validate_migration(
            str(config_file),
            '2.0'  # No migration path
        )
        
        assert result['valid'] == False
        assert len(result['errors']) > 0
        assert 'No migration path found' in result['errors'][0]
    
    def test_create_migration_script(self, temp_migrations_dir, temp_config_dir):
        """Test creating migration script template."""
        tool = MigrationTool(str(temp_migrations_dir))
        
        output_path = temp_config_dir / "new_migration.yaml"
        
        tool.create_migration_script('1.1', '1.2', str(output_path))
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            content = yaml.safe_load(f)
        
        assert content['from_version'] == '1.1'
        assert content['to_version'] == '1.2'
        assert 'transformations' in content
    
    def test_generate_migration_report(self, temp_migrations_dir, temp_config_dir):
        """Test generating migration report."""
        tool = MigrationTool(str(temp_migrations_dir))
        
        # Create mock migration results
        results = [
            {
                'success': True,
                'input_path': 'input1.yaml',
                'output_path': 'output1.yaml',
                'source_version': '1.0',
                'target_version': '1.1',
                'errors': [],
                'warnings': []
            },
            {
                'success': False,
                'input_path': 'input2.yaml',
                'output_path': 'output2.yaml',
                'source_version': '1.0',
                'target_version': '1.1',
                'errors': ['Migration failed'],
                'warnings': []
            }
        ]
        
        report_path = temp_config_dir / "migration_report.yaml"
        tool.generate_migration_report(results, str(report_path))
        
        assert report_path.exists()
        
        with open(report_path, 'r') as f:
            report = yaml.safe_load(f)
        
        assert report['migration_summary']['total_files'] == 2
        assert report['migration_summary']['successful_migrations'] == 1
        assert report['migration_summary']['failed_migrations'] == 1
        assert report['migration_summary']['total_errors'] == 1


if __name__ == '__main__':
    pytest.main([__file__])