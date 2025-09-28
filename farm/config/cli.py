#!/usr/bin/env python3
"""
Command-line interface for configuration management.

This tool provides commands for:
- Configuration versioning and management
- Template instantiation
- Configuration comparison
- File watching and reloading
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .config import SimulationConfig
from .template import ConfigTemplate, ConfigTemplateManager
from .watcher import create_reloadable_config, get_global_watcher


def cmd_version(args):
    """Handle configuration versioning commands."""
    if args.subcommand == "create":
        # Load configuration and create version
        config = SimulationConfig.from_centralized_config(
            environment=args.environment,
            profile=args.profile
        )

        versioned_config = config.version_config(args.description)
        filepath = versioned_config.save_versioned_config(args.output_dir, args.description)

        print(f"Created versioned configuration: {versioned_config.config_version}")
        print(f"Saved to: {filepath}")
        print(f"Description: {args.description}")

    elif args.subcommand == "list":
        # List all versions
        versions = SimulationConfig.list_config_versions(args.directory)

        if not versions:
            print(f"No configuration versions found in {args.directory}")
            return

        print(f"Configuration versions in {args.directory}:")
        print("-" * 80)
        for version in versions:
            print(f"Version: {version['version']}")
            print(f"Created: {version['created_at']}")
            print(f"Description: {version['description'] or 'No description'}")
            print("-" * 40)

    elif args.subcommand == "load":
        # Load specific version
        config = SimulationConfig.load_versioned_config(args.directory, args.version)
        print(f"Loaded configuration version: {args.version}")
        print(f"Created: {config.config_created_at}")
        print(f"Description: {config.config_description}")

        if args.output:
            config.to_yaml(args.output)
            print(f"Saved to: {args.output}")
        else:
            # Print to stdout
            print("\nConfiguration:")
            print(yaml.dump(config.to_dict(), default_flow_style=False))


def cmd_template(args):
    """Handle template commands."""
    manager = ConfigTemplateManager(args.template_dir)

    if args.subcommand == "create":
        # Create template from config
        config = SimulationConfig.from_centralized_config(
            environment=args.environment,
            profile=args.profile
        )
        template = ConfigTemplate.from_config(config)
        filepath = manager.save_template(args.name, template, args.description)

        print(f"Created template: {args.name}")
        print(f"Saved to: {filepath}")
        print(f"Required variables: {template.get_required_variables()}")

    elif args.subcommand == "list":
        # List templates
        templates = manager.list_templates()

        if not templates:
            print(f"No templates found in {args.template_dir}")
            return

        print(f"Templates in {args.template_dir}:")
        print("-" * 80)
        for template in templates:
            print(f"Name: {template['name']}")
            print(f"Description: {template['description'] or 'No description'}")
            print(f"Required variables: {template['required_variables']}")
            print("-" * 40)

    elif args.subcommand == "instantiate":
        # Instantiate template
        variables = {}
        if args.variables:
            for var_str in args.variables:
                key, value = var_str.split('=', 1)
                # Try to parse as JSON, fall back to string
                try:
                    variables[key] = json.loads(value)
                except json.JSONDecodeError:
                    variables[key] = value

        # Check for missing variables
        template = manager.load_template(args.name)
        missing = template.validate_variables(variables)
        if missing:
            print(f"Error: Missing required variables: {missing}")
            print(f"Required variables: {template.get_required_variables()}")
            return

        config = template.instantiate(variables)
        config.to_yaml(args.output)

        print(f"Instantiated template '{args.name}' to: {args.output}")
        print(f"Variables used: {variables}")

    elif args.subcommand == "batch":
        # Create batch of configs from template
        variable_sets = []

        # Read variable sets from file or command line
        if args.variable_file:
            with open(args.variable_file, 'r') as f:
                data = json.load(f)
                variable_sets = data.get('variable_sets', [])
        elif args.variables:
            # Single variable set from command line
            variables = {}
            for var_str in args.variables:
                key, value = var_str.split('=', 1)
                try:
                    variables[key] = json.loads(value)
                except json.JSONDecodeError:
                    variables[key] = value
            variable_sets = [variables]

        if not variable_sets:
            print("Error: No variable sets provided")
            return

        config_paths = manager.create_experiment_configs(
            args.name,
            variable_sets,
            args.output_dir
        )

        print(f"Created {len(config_paths)} configurations:")
        for path in config_paths:
            print(f"  {path}")


def cmd_diff(args):
    """Handle configuration diff commands."""
    # Load configurations
    if args.config1.endswith('.yaml'):
        config1 = SimulationConfig.from_yaml(args.config1)
    else:
        # Assume it's a version hash in current directory
        config1 = SimulationConfig.load_versioned_config(args.version_dir, args.config1)

    if args.config2.endswith('.yaml'):
        config2 = SimulationConfig.from_yaml(args.config2)
    else:
        # Assume it's a version hash
        config2 = SimulationConfig.load_versioned_config(args.version_dir, args.config2)

    # Generate diff
    diff = config1.diff_config(config2)

    if not diff:
        print("Configurations are identical")
        return

    print(f"Differences between {args.config1} and {args.config2}:")
    print("-" * 80)

    for key, change in diff.items():
        print(f"{key}:")
        if change['self'] is not None:
            print(f"  First config:  {change['self']}")
        else:
            print("  First config:  (missing)")
        if change['other'] is not None:
            print(f"  Second config: {change['other']}")
        else:
            print("  Second config: (missing)")
        print()


def cmd_watch(args):
    """Handle file watching commands."""
    watcher = get_global_watcher()

    if args.subcommand == "start":
        def on_change(config):
            print(f"\nConfiguration changed at {os.path.basename(args.filepath)}")
            if args.verbose:
                print(f"New config version: {config.generate_version_hash()}")
                if hasattr(config, 'config_version') and config.config_version:
                    print(f"Config version: {config.config_version}")

        watcher.watch_file(args.filepath, on_change)
        watcher.start()

        print(f"Watching {args.filepath} for changes...")
        print("Press Ctrl+C to stop")

        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping watcher...")
            watcher.stop()

    elif args.subcommand == "status":
        watched = watcher.get_watched_files()
        if not watched:
            print("No files are currently being watched")
        else:
            print("Currently watching:")
            for filepath, filehash in watched.items():
                print(f"  {filepath} (hash: {filehash[:8]}...)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Configuration management tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Version command
    version_parser = subparsers.add_parser("version", help="Configuration versioning")
    version_subparsers = version_parser.add_subparsers(dest="subcommand")

    # version create
    version_create = version_subparsers.add_parser("create", help="Create a versioned configuration")
    version_create.add_argument("--environment", default="development", help="Environment")
    version_create.add_argument("--profile", help="Profile")
    version_create.add_argument("--description", help="Version description")
    version_create.add_argument("--output-dir", default="farm/config/versions", help="Output directory")

    # version list
    version_list = version_subparsers.add_parser("list", help="List versioned configurations")
    version_list.add_argument("--directory", default="farm/config/versions", help="Directory to scan")

    # version load
    version_load = version_subparsers.add_parser("load", help="Load a specific version")
    version_load.add_argument("version", help="Version hash to load")
    version_load.add_argument("--directory", default="farm/config/versions", help="Version directory")
    version_load.add_argument("--output", help="Output file (prints to stdout if not specified)")

    # Template command
    template_parser = subparsers.add_parser("template", help="Configuration templating")
    template_subparsers = template_parser.add_subparsers(dest="subcommand")

    # template create
    template_create = template_subparsers.add_parser("create", help="Create a template from config")
    template_create.add_argument("name", help="Template name")
    template_create.add_argument("--environment", default="development", help="Environment")
    template_create.add_argument("--profile", help="Profile")
    template_create.add_argument("--description", help="Template description")
    template_create.add_argument("--template-dir", default="farm/config/templates", help="Template directory")

    # template list
    template_list = template_subparsers.add_parser("list", help="List available templates")
    template_list.add_argument("--template-dir", default="farm/config/templates", help="Template directory")

    # template instantiate
    template_inst = template_subparsers.add_parser("instantiate", help="Instantiate a template")
    template_inst.add_argument("name", help="Template name")
    template_inst.add_argument("output", help="Output configuration file")
    template_inst.add_argument("--variables", nargs="+", help="Variable assignments (key=value)")
    template_inst.add_argument("--template-dir", default="farm/config/templates", help="Template directory")

    # template batch
    template_batch = template_subparsers.add_parser("batch", help="Create batch of configs from template")
    template_batch.add_argument("name", help="Template name")
    template_batch.add_argument("--variables", nargs="+", help="Variable assignments for single config")
    template_batch.add_argument("--variable-file", help="JSON file with variable sets")
    template_batch.add_argument("--output-dir", default="farm/config/experiments", help="Output directory")
    template_batch.add_argument("--template-dir", default="farm/config/templates", help="Template directory")

    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare configurations")
    diff_parser.add_argument("config1", help="First config (file path or version hash)")
    diff_parser.add_argument("config2", help="Second config (file path or version hash)")
    diff_parser.add_argument("--version-dir", default="farm/config/versions", help="Directory for versioned configs")

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch configuration files")
    watch_subparsers = watch_parser.add_subparsers(dest="subcommand")

    # watch start
    watch_start = watch_subparsers.add_parser("start", help="Start watching a config file")
    watch_start.add_argument("filepath", help="File to watch")
    watch_start.add_argument("--verbose", action="store_true", help="Verbose output")

    # watch status
    watch_status = watch_subparsers.add_parser("status", help="Show watch status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "version":
            cmd_version(args)
        elif args.command == "template":
            cmd_template(args)
        elif args.command == "diff":
            cmd_diff(args)
        elif args.command == "watch":
            cmd_watch(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
