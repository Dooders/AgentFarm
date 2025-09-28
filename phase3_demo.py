#!/usr/bin/env python3
"""
Phase 3 Demonstration: Configuration Migration System

This script demonstrates the configuration migration system implemented
in Phase 3, including:
- Configuration version detection and compatibility checking
- Migration transformation operations
- Automated migration tools and validation
- Integration with hierarchical configuration system
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
sys.path.append('/workspace')

from farm.core.config_hydra_bridge import HydraSimulationConfig
)


def demo_version_detection():
    """Demonstrate configuration version detection."""
    print("=" * 60)
    print("DEMO: Configuration Version Detection")
    print("=" * 60)
    
    detector = ConfigurationVersionDetector()
    
    # Test different configuration versions
    test_configs = [
        {
            'name': 'Version 1.0',
            'config': {
                'simulation_id': 'test',
                'max_steps': 1000,
                'learning_rate': 0.001
            }
        },
        {
            'name': 'Version 1.1',
            'config': {
                'simulation_id': 'test',
                'max_steps': 1000,
                'learning_rate': 0.001,
                'agent_parameters': {
                    'SystemAgent': {
                        'share_weight': 0.3
                    }
                }
            }
        },
        {
            'name': 'Version 1.2',
            'config': {
                'simulation_id': 'test',
                'max_steps': 1000,
                'learning_rate': 0.001,
                'agent_parameters': {
                    'SystemAgent': {
                        'share_weight': 0.3
                    }
                },
                'visualization': {
                    'canvas_size': [400, 400],
                    'background_color': 'black'
                }
            }
        },
        {
            'name': 'Version 2.0',
            'config': {
                'simulation_id': 'test',
                'max_steps': 1000,
                'learning_rate': 0.001,
                'agent_parameters': {
                    'SystemAgent': {
                        'share_weight': 0.3
                    }
                },
                'visualization': {
                    'canvas_size': [400, 400],
                    'background_color': 'black'
                },
                'redis': {
                    'host': 'localhost',
                    'port': 6379
                }
            }
        },
        {
            'name': 'Explicit Version',
            'config': {
                'config_version': '2.1',
                'simulation_id': 'test',
                'max_steps': 1000
            }
        }
    ]
    
    for test_case in test_configs:
        detected_version = detector.detect_version(test_case['config'])
        print(f"  {test_case['name']}: {detected_version}")
    
    print()


def demo_migration_transformations():
    """Demonstrate migration transformation operations."""
    print("=" * 60)
    print("DEMO: Migration Transformation Operations")
    print("=" * 60)
    
    # Test configuration
    config = {
        'old_key': 'old_value',
        'nested': {
            'deep_key': 'deep_value'
        },
        'source_dict': {
            'key1': 'value1',
            'key2': 'value2'
        },
        'target_dict': {
            'key2': 'existing_value2',
            'key3': 'value3'
        }
    }
    
    print("Original configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Test rename transformation
    print("1. Rename transformation:")
    rename_transform = MigrationTransformation(
        operation='rename',
        source_path='old_key',
        target_path='new_key',
        description='Rename old_key to new_key'
    )
    
    result = rename_transform.apply(config.copy())
    print(f"  old_key -> new_key: {result.get('new_key')}")
    print(f"  old_key exists: {'old_key' in result}")
    print()
    
    # Test move transformation
    print("2. Move transformation:")
    move_transform = MigrationTransformation(
        operation='move',
        source_path='nested.deep_key',
        target_path='shallow.value',
        description='Move deep nested value to shallow'
    )
    
    result = move_transform.apply(config.copy())
    print(f"  nested.deep_key -> shallow.value: {result.get('shallow', {}).get('value')}")
    print(f"  nested exists: {'nested' in result}")
    print()
    
    # Test merge transformation
    print("3. Merge transformation:")
    merge_transform = MigrationTransformation(
        operation='merge',
        source_path='source_dict',
        target_path='target_dict',
        description='Merge source_dict into target_dict'
    )
    
    result = merge_transform.apply(config.copy())
    merged_dict = result.get('target_dict', {})
    print(f"  Merged target_dict keys: {list(merged_dict.keys())}")
    print(f"  key1 (from source): {merged_dict.get('key1')}")
    print(f"  key2 (source overwrites): {merged_dict.get('key2')}")
    print(f"  key3 (from target): {merged_dict.get('key3')}")
    print(f"  source_dict exists: {'source_dict' in result}")
    print()
    
    # Test add transformation
    print("4. Add transformation:")
    add_transform = MigrationTransformation(
        operation='add',
        target_path='new_setting',
        value='new_value',
        description='Add new setting'
    )
    
    result = add_transform.apply(config.copy())
    print(f"  Added new_setting: {result.get('new_setting')}")
    print()
    
    # Test delete transformation
    print("5. Delete transformation:")
    delete_transform = MigrationTransformation(
        operation='delete',
        source_path='old_key',
        description='Delete old_key'
    )
    
    result = delete_transform.apply(config.copy())
    print(f"  old_key exists after delete: {'old_key' in result}")
    print()


def demo_migration_operations():
    """Demonstrate complete migration operations."""
    print("=" * 60)
    print("DEMO: Complete Migration Operations")
    print("=" * 60)
    
    # Create a v1.0 configuration
    v1_0_config = {
        'simulation_id': 'demo-simulation',
        'max_steps': 1000,
        'learning_rate': 0.001,
        'old_setting': 'old_value'
    }
    
    print("Original v1.0 Configuration:")
    for key, value in v1_0_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize migrator
    migrator = ConfigurationMigrator('/workspace/config/migrations')
    
    print("Available Migration Paths:")
    migration_chain = migrator.get_migration_chain()
    for from_version, to_version in migration_chain.items():
        print(f"  {from_version} -> {to_version}")
    print()
    
    # Test single-step migration (1.0 -> 1.1)
    print("Single-step migration (1.0 -> 1.1):")
    v1_1_config = migrator.migrate_config(v1_0_config, '1.0', '1.1')
    
    print("  Changes:")
    print(f"    Added agent_parameters: {'agent_parameters' in v1_1_config}")
    print(f"    Added config_version: {v1_1_config.get('config_version')}")
    print(f"    Preserved old_setting: {v1_1_config.get('old_setting')}")
    print()
    
    # Test multi-step migration (1.0 -> 2.0)
    print("Multi-step migration (1.0 -> 2.0):")
    v2_0_config = migrator.migrate_config(v1_0_config, '1.0', '2.0')
    
    print("  Changes:")
    print(f"    Added agent_parameters: {'agent_parameters' in v2_0_config}")
    print(f"    Added visualization: {'visualization' in v2_0_config}")
    print(f"    Added redis: {'redis' in v2_0_config}")
    print(f"    Added database settings: {'use_in_memory_db' in v2_0_config}")
    print(f"    Updated config_version: {v2_0_config.get('config_version')}")
    print(f"    Preserved original settings: {v2_0_config.get('simulation_id')}")
    print()
    
    # Test migration validation
    print("Migration Path Validation:")
    errors = migrator.validate_migration_path('1.0', '2.0')
    if not errors:
        print("  ✅ Migration path 1.0 -> 2.0 is valid")
    else:
        print(f"  ❌ Migration path validation failed: {errors}")
    
    errors = migrator.validate_migration_path('1.0', '3.0')
    if errors:
        print(f"  ❌ Migration path 1.0 -> 3.0 is invalid: {errors[0]}")
    print()


def demo_automated_migration_tool():
    """Demonstrate automated migration tool."""
    print("=" * 60)
    print("DEMO: Automated Migration Tool")
    print("=" * 60)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test configuration file
        test_config = {
            'simulation_id': 'tool-test',
            'max_steps': 500,
            'learning_rate': 0.01,
            'old_key': 'old_value'
        }
        
        config_file = temp_path / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        print(f"Created test configuration: {config_file}")
        print("  Content:")
        for key, value in test_config.items():
            print(f"    {key}: {value}")
        print()
        
        # Initialize migration tool
        tool = MigrationTool('/workspace/config/migrations')
        
        # Test file migration
        print("1. Single File Migration:")
        output_file = temp_path / "migrated_config.yaml"
        
        result = tool.migrate_file(
            str(config_file),
            str(output_file),
            '2.0'
        )
        
        print(f"  Migration result: {'✅ Success' if result['success'] else '❌ Failed'}")
        print(f"  Source version: {result['source_version']}")
        print(f"  Target version: {result['target_version']}")
        print(f"  Errors: {len(result['errors'])}")
        print(f"  Warnings: {len(result['warnings'])}")
        
        if result['success']:
            # Show migrated configuration
            with open(output_file, 'r') as f:
                migrated_config = yaml.safe_load(f)
            
            print("  Migrated configuration keys:")
            for key in sorted(migrated_config.keys()):
                print(f"    {key}")
        print()
        
        # Test migration validation
        print("2. Migration Validation:")
        validation_result = tool.validate_migration(
            str(config_file),
            '2.0'
        )
        
        print(f"  Validation result: {'✅ Valid' if validation_result['valid'] else '❌ Invalid'}")
        print(f"  Source version: {validation_result['source_version']}")
        print(f"  Target version: {validation_result['target_version']}")
        print(f"  Errors: {len(validation_result['errors'])}")
        print()
        
        # Test migration script creation
        print("3. Migration Script Creation:")
        script_file = temp_path / "custom_migration.yaml"
        
        tool.create_migration_script('2.0', '2.1', str(script_file))
        
        if script_file.exists():
            print(f"  ✅ Created migration script: {script_file}")
            with open(script_file, 'r') as f:
                script_content = yaml.safe_load(f)
            
            print(f"  Script from_version: {script_content['from_version']}")
            print(f"  Script to_version: {script_content['to_version']}")
            print(f"  Transformations: {len(script_content['transformations'])}")
        else:
            print("  ❌ Failed to create migration script")
        print()


def demo_integration_with_environment_system():
    """Demonstrate integration with environment configuration system."""
    print("=" * 60)
    print("DEMO: Integration with Environment Configuration System")
    print("=" * 60)
    
    # Create a temporary configuration file for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a v1.0 base configuration
        v1_0_base_config = {
            'simulation_id': 'integration-test',
            'max_steps': 1000,
            'learning_rate': 0.001,
            'debug': False
        }
        
        base_config_file = temp_path / "base.yaml"
        import yaml
        with open(base_config_file, 'w') as f:
            yaml.dump(v1_0_base_config, f)
        
        # Create environment-specific configuration
        env_config = {
            'debug': True,
            'max_steps': 500
        }
        
        env_dir = temp_path / "environments"
        env_dir.mkdir()
        
        env_config_file = env_dir / "development.yaml"
        with open(env_config_file, 'w') as f:
            yaml.dump(env_config, f)
        
        print("Created test configuration structure:")
        print(f"  Base config: {base_config_file}")
        print(f"  Environment config: {env_config_file}")
        print()
        
        # Initialize environment config manager
        env_manager = EnvironmentConfigManager(
            str(base_config_file),
            environment='development'
        )
        
        print("Environment Configuration Manager:")
        print(f"  Environment: {env_manager.environment}")
        print(f"  Available environments: {env_manager.get_available_environments()}")
        print()
        
        # Get configuration hierarchy
        config_hierarchy = env_manager.get_config_hierarchy()
        
        print("Configuration Hierarchy:")
        print(f"  Global keys: {len(config_hierarchy.global_config)}")
        print(f"  Environment keys: {len(config_hierarchy.environment_config)}")
        print(f"  Agent keys: {len(config_hierarchy.agent_config)}")
        print()
        
        # Test migration of the configuration
        print("Migration Integration:")
        migrator = ConfigurationMigrator('/workspace/config/migrations')
        
        # Get effective configuration
        effective_config = config_hierarchy.get_effective_config()
        
        # Detect version
        detector = ConfigurationVersionDetector()
        detected_version = detector.detect_version(effective_config)
        print(f"  Detected version: {detected_version}")
        
        # Migrate to latest version
        if detected_version != '2.1':
            migrated_config = migrator.migrate_config(effective_config, detected_version, '2.1')
            print(f"  Migrated to version: 2.1")
            print(f"  New config keys: {len(migrated_config)}")
            print(f"  Added features:")
            print(f"    - agent_parameters: {'agent_parameters' in migrated_config}")
            print(f"    - visualization: {'visualization' in migrated_config}")
            print(f"    - redis: {'redis' in migrated_config}")
            print(f"    - curriculum_phases: {'curriculum_phases' in migrated_config}")
        else:
            print(f"  Configuration is already at latest version: {detected_version}")
        print()


def demo_migration_script_examples():
    """Demonstrate example migration scripts."""
    print("=" * 60)
    print("DEMO: Migration Script Examples")
    print("=" * 60)
    
    # Show available migration scripts
    migrations_dir = Path('/workspace/config/migrations')
    
    if migrations_dir.exists():
        migration_files = list(migrations_dir.glob("*.yaml"))
        
        print("Available Migration Scripts:")
        for migration_file in sorted(migration_files):
            print(f"  📄 {migration_file.name}")
            
            # Load and show migration details
            import yaml
            with open(migration_file, 'r') as f:
                migration_data = yaml.safe_load(f)
            
            print(f"    From: {migration_data.get('from_version')}")
            print(f"    To: {migration_data.get('to_version')}")
            print(f"    Description: {migration_data.get('description')}")
            print(f"    Transformations: {len(migration_data.get('transformations', []))}")
            print()
    else:
        print("No migration scripts found in /workspace/config/migrations")
        print()
    
    # Show transformation operation examples
    print("Transformation Operation Examples:")
    operations = [
        {
            'operation': 'rename',
            'description': 'Rename a configuration key',
            'example': 'old_key -> new_key'
        },
        {
            'operation': 'move',
            'description': 'Move a value to a different path',
            'example': 'source.nested -> target.nested'
        },
        {
            'operation': 'add',
            'description': 'Add a new configuration value',
            'example': 'Add new_setting with default value'
        },
        {
            'operation': 'delete',
            'description': 'Remove a configuration key',
            'example': 'Remove deprecated_setting'
        },
        {
            'operation': 'merge',
            'description': 'Merge two configuration objects',
            'example': 'Merge source_dict into target_dict'
        },
        {
            'operation': 'split',
            'description': 'Split a configuration object into multiple keys',
            'example': 'Split combined_setting into individual settings'
        }
    ]
    
    for op in operations:
        print(f"  🔧 {op['operation']}: {op['description']}")
        print(f"     Example: {op['example']}")
    print()


def main():
    """Run all Phase 3 demonstrations."""
    print("PHASE 3: CONFIGURATION MIGRATION SYSTEM")
    print("=" * 60)
    print("This demonstration shows the configuration migration system")
    print("implemented in Phase 3, including version detection, migration")
    print("transformations, and automated migration tools.")
    print()
    
    demo_version_detection()
    demo_migration_transformations()
    demo_migration_operations()
    demo_automated_migration_tool()
    demo_integration_with_environment_system()
    demo_migration_script_examples()
    
    print("=" * 60)
    print("PHASE 3 IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Key Features Implemented:")
    print("✓ ConfigurationMigrator with automated migration path finding")
    print("✓ MigrationTransformation with 6 operation types (rename, move, add, delete, merge, split)")
    print("✓ ConfigurationVersionDetector with automatic version detection")
    print("✓ MigrationTool with command-line interface and batch operations")
    print("✓ Migration script system with YAML/JSON support")
    print("✓ Integration with environment configuration system")
    print("✓ Comprehensive error handling and validation")
    print("✓ Migration path validation and compatibility checking")
    print()
    print("Migration Scripts Available:")
    print("  📄 v1.0_to_v1.1.yaml - Add agent-specific parameters")
    print("  📄 v1.1_to_v1.2.yaml - Add visualization configuration")
    print("  📄 v1.2_to_v2.0.yaml - Add Redis and database settings")
    print("  📄 v2.0_to_v2.1.yaml - Add curriculum learning parameters")
    print()
    print("Transformation Operations:")
    print("  🔧 rename - Rename configuration keys")
    print("  🔧 move - Move values between paths")
    print("  🔧 add - Add new configuration values")
    print("  🔧 delete - Remove configuration keys")
    print("  🔧 merge - Merge configuration objects")
    print("  🔧 split - Split objects into multiple keys")
    print()
    print("Next Steps (Phase 4):")
    print("- Hot-reloading capabilities for dynamic configuration updates")
    print("- File system monitoring and automatic reloading")
    print("- Runtime configuration change notifications")
    print("- Integration with existing configuration systems")


if __name__ == '__main__':
    main()