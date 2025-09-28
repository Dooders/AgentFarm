"""
Automated configuration migration tool.

This module provides command-line tools and utilities for automated
configuration migration, including batch migration, validation,
and migration script generation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .migration import ConfigurationMigrator, ConfigurationVersionDetector
from .environment import EnvironmentConfigManager
from .exceptions import ConfigurationMigrationError, ConfigurationError

logger = logging.getLogger(__name__)


class MigrationTool:
    """Automated configuration migration tool.
    
    Provides utilities for migrating configuration files, validating
    migrations, and generating migration reports.
    """
    
    def __init__(self, migrations_dir: Optional[str] = None):
        """Initialize migration tool.
        
        Args:
            migrations_dir: Directory containing migration scripts
        """
        self.migrator = ConfigurationMigrator(migrations_dir)
        self.version_detector = ConfigurationVersionDetector()
    
    def migrate_file(
        self, 
        input_path: str, 
        output_path: str, 
        target_version: str,
        source_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Migrate a single configuration file.
        
        Args:
            input_path: Path to input configuration file
            output_path: Path to output configuration file
            target_version: Target version to migrate to
            source_version: Source version (auto-detected if None)
            
        Returns:
            Migration result dictionary
        """
        result = {
            'success': False,
            'input_path': input_path,
            'output_path': output_path,
            'source_version': source_version,
            'target_version': target_version,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Load input configuration
            input_path = Path(input_path)
            if not input_path.exists():
                result['errors'].append(f"Input file not found: {input_path}")
                return result
            
            # Load configuration
            if input_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(input_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            elif input_path.suffix.lower() == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    config = json.load(f) or {}
            else:
                result['errors'].append(f"Unsupported file format: {input_path.suffix}")
                return result
            
            # Detect source version if not provided
            if not source_version:
                source_version = self.version_detector.detect_version(config)
                result['source_version'] = source_version
            
            # Check if migration is needed
            if source_version == target_version:
                result['warnings'].append(f"Source and target versions are the same: {source_version}")
                result['success'] = True
                return result
            
            # Validate migration path
            validation_errors = self.migrator.validate_migration_path(source_version, target_version)
            if validation_errors:
                result['errors'].extend(validation_errors)
                return result
            
            # Perform migration
            migrated_config = self.migrator.migrate_config(config, source_version, target_version)
            
            # Save migrated configuration
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(migrated_config, f, default_flow_style=False, indent=2)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(migrated_config, f, indent=2)
            
            result['success'] = True
            logger.info(f"Successfully migrated {input_path} from {source_version} to {target_version}")
            
        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"Migration failed: {e}")
        
        return result
    
    def migrate_directory(
        self, 
        input_dir: str, 
        output_dir: str, 
        target_version: str,
        source_version: Optional[str] = None,
        file_pattern: str = "*.yaml"
    ) -> Dict[str, Any]:
        """Migrate all configuration files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            target_version: Target version to migrate to
            source_version: Source version (auto-detected if None)
            file_pattern: File pattern to match (default: "*.yaml")
            
        Returns:
            Migration results summary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            return {
                'success': False,
                'errors': [f"Input directory not found: {input_dir}"],
                'files_processed': 0,
                'files_successful': 0,
                'files_failed': 0
            }
        
        # Find configuration files
        config_files = list(input_dir.glob(file_pattern))
        
        results = {
            'success': True,
            'files_processed': len(config_files),
            'files_successful': 0,
            'files_failed': 0,
            'file_results': [],
            'errors': [],
            'warnings': []
        }
        
        for config_file in config_files:
            # Determine output path
            relative_path = config_file.relative_to(input_dir)
            output_file = output_dir / relative_path
            
            # Migrate file
            file_result = self.migrate_file(
                str(config_file),
                str(output_file),
                target_version,
                source_version
            )
            
            results['file_results'].append(file_result)
            
            if file_result['success']:
                results['files_successful'] += 1
            else:
                results['files_failed'] += 1
                results['success'] = False
                results['errors'].extend(file_result['errors'])
            
            results['warnings'].extend(file_result['warnings'])
        
        return results
    
    def validate_migration(
        self, 
        config_path: str, 
        target_version: str,
        source_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate that a configuration can be migrated to a target version.
        
        Args:
            config_path: Path to configuration file
            target_version: Target version
            source_version: Source version (auto-detected if None)
            
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': False,
            'config_path': config_path,
            'source_version': source_version,
            'target_version': target_version,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Load configuration
            config_path = Path(config_path)
            if not config_path.exists():
                result['errors'].append(f"Configuration file not found: {config_path}")
                return result
            
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f) or {}
            else:
                result['errors'].append(f"Unsupported file format: {config_path.suffix}")
                return result
            
            # Detect source version if not provided
            if not source_version:
                source_version = self.version_detector.detect_version(config)
                result['source_version'] = source_version
            
            # Check if migration is needed
            if source_version == target_version:
                result['warnings'].append(f"Configuration is already at target version: {target_version}")
                result['valid'] = True
                return result
            
            # Validate migration path
            validation_errors = self.migrator.validate_migration_path(source_version, target_version)
            if validation_errors:
                result['errors'].extend(validation_errors)
                return result
            
            # Try to perform migration (dry run)
            try:
                self.migrator.migrate_config(config, source_version, target_version)
                result['valid'] = True
            except Exception as e:
                result['errors'].append(f"Migration validation failed: {str(e)}")
        
        except Exception as e:
            result['errors'].append(f"Validation failed: {str(e)}")
        
        return result
    
    def generate_migration_report(
        self, 
        results: List[Dict[str, Any]], 
        output_path: str
    ) -> None:
        """Generate a migration report.
        
        Args:
            results: List of migration results
            output_path: Path to save the report
        """
        report = {
            'migration_summary': {
                'total_files': len(results),
                'successful_migrations': sum(1 for r in results if r['success']),
                'failed_migrations': sum(1 for r in results if not r['success']),
                'total_errors': sum(len(r['errors']) for r in results),
                'total_warnings': sum(len(r['warnings']) for r in results)
            },
            'file_results': results,
            'errors': [],
            'warnings': []
        }
        
        # Collect all errors and warnings
        for result in results:
            report['errors'].extend(result['errors'])
            report['warnings'].extend(result['warnings'])
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(report, f, default_flow_style=False, indent=2)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
        
        logger.info(f"Migration report saved to: {output_path}")
    
    def create_migration_script(
        self, 
        from_version: str, 
        to_version: str, 
        output_path: str
    ) -> None:
        """Create a template migration script.
        
        Args:
            from_version: Source version
            to_version: Target version
            output_path: Path to save the migration script
        """
        self.migrator.create_migration_script(from_version, to_version, output_path)
        logger.info(f"Migration script template created: {output_path}")


def main():
    """Command-line interface for the migration tool."""
    parser = argparse.ArgumentParser(description="Configuration Migration Tool")
    parser.add_argument('--migrations-dir', help='Directory containing migration scripts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Migrate single file command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate a single configuration file')
    migrate_parser.add_argument('input', help='Input configuration file path')
    migrate_parser.add_argument('output', help='Output configuration file path')
    migrate_parser.add_argument('target_version', help='Target version')
    migrate_parser.add_argument('--source-version', help='Source version (auto-detected if not provided)')
    
    # Migrate directory command
    migrate_dir_parser = subparsers.add_parser('migrate-dir', help='Migrate all files in a directory')
    migrate_dir_parser.add_argument('input_dir', help='Input directory path')
    migrate_dir_parser.add_argument('output_dir', help='Output directory path')
    migrate_dir_parser.add_argument('target_version', help='Target version')
    migrate_dir_parser.add_argument('--source-version', help='Source version (auto-detected if not provided)')
    migrate_dir_parser.add_argument('--pattern', default='*.yaml', help='File pattern to match')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate migration path')
    validate_parser.add_argument('config', help='Configuration file path')
    validate_parser.add_argument('target_version', help='Target version')
    validate_parser.add_argument('--source-version', help='Source version (auto-detected if not provided)')
    
    # Create migration script command
    create_parser = subparsers.add_parser('create-script', help='Create migration script template')
    create_parser.add_argument('from_version', help='Source version')
    create_parser.add_argument('to_version', help='Target version')
    create_parser.add_argument('output', help='Output migration script path')
    
    # List versions command
    list_parser = subparsers.add_parser('list-versions', help='List available versions')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create migration tool
    tool = MigrationTool(args.migrations_dir)
    
    try:
        if args.command == 'migrate':
            result = tool.migrate_file(
                args.input,
                args.output,
                args.target_version,
                args.source_version
            )
            
            if result['success']:
                print(f"✅ Migration successful: {args.input} -> {args.output}")
            else:
                print(f"❌ Migration failed: {args.input}")
                for error in result['errors']:
                    print(f"  Error: {error}")
                sys.exit(1)
        
        elif args.command == 'migrate-dir':
            result = tool.migrate_directory(
                args.input_dir,
                args.output_dir,
                args.target_version,
                args.source_version,
                args.pattern
            )
            
            print(f"📁 Directory migration completed:")
            print(f"  Files processed: {result['files_processed']}")
            print(f"  Successful: {result['files_successful']}")
            print(f"  Failed: {result['files_failed']}")
            
            if not result['success']:
                sys.exit(1)
        
        elif args.command == 'validate':
            result = tool.validate_migration(
                args.config,
                args.target_version,
                args.source_version
            )
            
            if result['valid']:
                print(f"✅ Migration validation passed: {args.config}")
            else:
                print(f"❌ Migration validation failed: {args.config}")
                for error in result['errors']:
                    print(f"  Error: {error}")
                sys.exit(1)
        
        elif args.command == 'create-script':
            tool.create_migration_script(
                args.from_version,
                args.to_version,
                args.output
            )
            print(f"📝 Migration script template created: {args.output}")
        
        elif args.command == 'list-versions':
            versions = tool.migrator.get_available_versions()
            print("Available configuration versions:")
            for version in versions:
                print(f"  - {version}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()