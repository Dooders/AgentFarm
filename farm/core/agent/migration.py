"""
Migration utilities for transitioning from BaseAgent to AgentCore.

Provides tools and helpers for gradually migrating existing code.
"""

import ast
import re
from typing import List, Dict, Set, Optional
from pathlib import Path


class MigrationAnalyzer:
    """
    Analyze code for BaseAgent usage and suggest migrations.

    This class helps identify where old BaseAgent code needs updating.
    """

    def __init__(self):
        """Initialize migration analyzer."""
        self.issues: List[Dict[str, any]] = []

    def analyze_file(self, file_path: str) -> Dict[str, any]:
        """
        Analyze a Python file for migration needs.

        Args:
            file_path: Path to Python file

        Returns:
            dict: Analysis results with issues and suggestions

        Example:
            >>> analyzer = MigrationAnalyzer()
            >>> results = analyzer.analyze_file("my_simulation.py")
            >>> for issue in results['issues']:
            ...     print(f"{issue['line']}: {issue['message']}")
        """
        issues = []
        suggestions = []

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Check for BaseAgent imports
            if 'from farm.core.agent import BaseAgent' in content:
                issues.append({
                    'type': 'import',
                    'severity': 'high',
                    'message': 'Uses old BaseAgent import',
                    'suggestion': 'Change to: from farm.core.agent import AgentFactory'
                })

            # Check for BaseAgent instantiation
            if re.search(r'BaseAgent\s*\(', content):
                issues.append({
                    'type': 'instantiation',
                    'severity': 'high',
                    'message': 'Creates BaseAgent instances',
                    'suggestion': 'Use AgentFactory.create_default_agent() instead'
                })

            # Check for direct attribute access that should use components
            patterns = [
                (r'agent\.resource_level', 'Use agent.get_component("resource").level'),
                (r'agent\.current_health', 'Use agent.get_component("combat").health'),
                (r'agent\.max_movement', 'Use agent.get_component("movement").max_movement'),
            ]

            for pattern, suggestion in patterns:
                if re.search(pattern, content):
                    issues.append({
                        'type': 'attribute_access',
                        'severity': 'medium',
                        'pattern': pattern,
                        'message': f'Direct attribute access: {pattern}',
                        'suggestion': suggestion
                    })

            return {
                'file': file_path,
                'issues': issues,
                'can_use_adapter': True,
                'can_migrate_directly': len(issues) < 5,
            }

        except Exception as e:
            return {
                'file': file_path,
                'error': str(e),
                'issues': [],
            }

    def analyze_directory(self, directory: str) -> List[Dict[str, any]]:
        """
        Analyze all Python files in a directory.

        Args:
            directory: Path to directory

        Returns:
            list: Analysis results for each file

        Example:
            >>> analyzer = MigrationAnalyzer()
            >>> results = analyzer.analyze_directory("./my_project")
            >>> print(f"Found {len(results)} files to migrate")
        """
        results = []
        path = Path(directory)

        for py_file in path.rglob('*.py'):
            if 'test' not in str(py_file):  # Skip test files
                result = self.analyze_file(str(py_file))
                if result.get('issues'):
                    results.append(result)

        return results

    def generate_report(self, results: List[Dict[str, any]]) -> str:
        """
        Generate migration report.

        Args:
            results: Analysis results from analyze_directory

        Returns:
            str: Formatted report

        Example:
            >>> report = analyzer.generate_report(results)
            >>> print(report)
        """
        report = ["# BaseAgent Migration Report\n"]

        total_files = len(results)
        total_issues = sum(len(r.get('issues', [])) for r in results)

        report.append(f"## Summary\n")
        report.append(f"- Files needing migration: {total_files}")
        report.append(f"- Total issues found: {total_issues}\n")

        report.append(f"## Files\n")
        for result in results:
            file = result['file']
            issues = result.get('issues', [])

            report.append(f"### {file}\n")
            report.append(f"Issues: {len(issues)}\n")

            for issue in issues:
                severity = issue.get('severity', 'info')
                message = issue.get('message', '')
                suggestion = issue.get('suggestion', '')

                report.append(f"- [{severity.upper()}] {message}")
                if suggestion:
                    report.append(f"  - Suggestion: {suggestion}")

            report.append("")

        return "\n".join(report)


class CodeMigrator:
    """
    Automated code migration helper.

    Provides automated transformations for common migration patterns.
    """

    @staticmethod
    def suggest_replacement(old_code: str) -> Optional[str]:
        """
        Suggest modern replacement for old code pattern.

        Args:
            old_code: Old code snippet

        Returns:
            str or None: Suggested replacement

        Example:
            >>> old = "agent = BaseAgent(...)"
            >>> new = CodeMigrator.suggest_replacement(old)
            >>> print(new)
            factory = AgentFactory(...)
            agent = factory.create_default_agent(...)
        """
        replacements = {
            # Import replacements
            'from farm.core.agent import BaseAgent':
                'from farm.core.agent import AgentFactory, AgentConfig',

            # Instantiation replacements
            'BaseAgent(': 'BaseAgentAdapter.from_old_style(',

            # Attribute access replacements
            'agent.resource_level': 'agent.get_component("resource").level',
            'agent.current_health': 'agent.get_component("combat").health',
            'agent.is_defending': 'agent.get_component("combat").is_defending',
        }

        for old, new in replacements.items():
            if old in old_code:
                return old_code.replace(old, new)

        return None

    @staticmethod
    def generate_adapter_code(
        agent_id: str = "agent",
        position: str = "(0, 0)",
        resources: str = "100"
    ) -> str:
        """
        Generate code snippet for using adapter.

        Args:
            agent_id: Agent ID expression
            position: Position expression
            resources: Resource level expression

        Returns:
            str: Code snippet

        Example:
            >>> code = CodeMigrator.generate_adapter_code()
            >>> print(code)
        """
        return f"""
# Migration: Use adapter for backward compatibility
from farm.core.agent.compat import BaseAgentAdapter

agent = BaseAgentAdapter.from_old_style(
    agent_id={agent_id},
    position={position},
    resource_level={resources},
    spatial_service=spatial_service,
    time_service=time_service,
)

# Access via old API (works)
print(agent.position)
print(agent.resource_level)

# Access new features via core
movement = agent.core.get_component("movement")
movement.move_to((100, 100))
"""

    @staticmethod
    def generate_direct_migration_code(
        agent_id: str = "agent",
        position: str = "(0, 0)",
        resources: str = "100"
    ) -> str:
        """
        Generate code snippet for direct AgentCore usage.

        Args:
            agent_id: Agent ID expression
            position: Position expression
            resources: Resource level expression

        Returns:
            str: Code snippet

        Example:
            >>> code = CodeMigrator.generate_direct_migration_code()
            >>> print(code)
        """
        return f"""
# Migration: Use AgentCore directly (recommended)
from farm.core.agent import AgentFactory, AgentConfig

# Create factory once
factory = AgentFactory(
    spatial_service=spatial_service,
    time_service=time_service,
    lifecycle_service=lifecycle_service,
)

# Create agent
agent = factory.create_default_agent(
    agent_id={agent_id},
    position={position},
    initial_resources={resources},
)

# Access via new API
print(agent.position)
resource = agent.get_component("resource")
print(resource.level)

# Use components
movement = agent.get_component("movement")
movement.move_to((100, 100))
"""


def create_migration_guide() -> str:
    """
    Create comprehensive migration guide.

    Returns:
        str: Markdown-formatted migration guide

    Example:
        >>> guide = create_migration_guide()
        >>> with open("MIGRATION.md", "w") as f:
        ...     f.write(guide)
    """
    return """# Migration Guide: BaseAgent to AgentCore

## Overview

This guide helps you migrate from the old monolithic BaseAgent to the new
component-based AgentCore system.

## Migration Strategies

### Strategy 1: Use Adapter (Quick, Low Risk)

Best for: Large codebases, gradual migration

```python
# Before
from farm.core.agent import BaseAgent

agent = BaseAgent(
    agent_id="agent_001",
    position=(10, 20),
    resource_level=100,
    spatial_service=spatial_service
)

# After (minimal changes)
from farm.core.agent.compat import BaseAgentAdapter

agent = BaseAgentAdapter.from_old_style(
    agent_id="agent_001",
    position=(10, 20),
    resource_level=100,
    spatial_service=spatial_service
)

# All old code works!
print(agent.position)
print(agent.resource_level)
agent.act()
```

### Strategy 2: Direct Migration (Clean, Recommended)

Best for: New code, small modules

```python
# Before
from farm.core.agent import BaseAgent

agent = BaseAgent(
    agent_id="agent_001",
    position=(10, 20),
    resource_level=100,
    spatial_service=spatial_service
)

# After (use new API)
from farm.core.agent import AgentFactory

factory = AgentFactory(
    spatial_service=spatial_service,
    time_service=time_service,
)

agent = factory.create_default_agent(
    agent_id="agent_001",
    position=(10, 20),
    initial_resources=100
)

# New API
resource = agent.get_component("resource")
print(resource.level)
```

## Common Patterns

### Pattern 1: Attribute Access

```python
# Old
agent.resource_level
agent.current_health
agent.position

# New (via adapter)
agent.resource_level  # Still works
agent.current_health  # Still works
agent.position  # Still works

# New (direct)
agent.get_component("resource").level
agent.get_component("combat").health
agent.position  # Direct property
```

### Pattern 2: Movement

```python
# Old
agent.position = new_position
agent.update_position(new_position)

# New
movement = agent.get_component("movement")
movement.move_to(new_position)
```

### Pattern 3: Combat

```python
# Old
agent.handle_combat(attacker, damage)
agent.take_damage(damage)

# New
combat = agent.get_component("combat")
combat.take_damage(damage)
```

### Pattern 4: Resources

```python
# Old
agent.resource_level += 50
agent.resource_level -= 20

# New
resource = agent.get_component("resource")
resource.add(50)
resource.consume(20)
```

## Testing Migration

```python
import pytest
from farm.core.agent.compat import BaseAgentAdapter, is_new_agent

def test_adapter_compatibility():
    \"\"\"Test that adapter works like old BaseAgent.\"\"\"
    agent = BaseAgentAdapter.from_old_style(
        agent_id="test",
        position=(0, 0),
        resource_level=100,
        spatial_service=mock_spatial_service
    )

    # Old API should work
    assert agent.resource_level == 100
    assert agent.position == (0, 0)
    assert agent.alive is True

    # Can access new features
    assert is_new_agent(agent)
    movement = agent.core.get_component("movement")
    assert movement is not None
```

## Step-by-Step Migration Process

1. **Analyze**: Use MigrationAnalyzer to find all BaseAgent usage
2. **Choose Strategy**: Adapter for quick fix, direct for clean code
3. **Update Imports**: Change imports to new system
4. **Update Instantiation**: Use adapter or factory
5. **Test**: Verify behavior unchanged
6. **Refactor**: Gradually move to direct component access
7. **Cleanup**: Remove adapter when fully migrated

## Automated Tools

```python
from farm.core.agent.migration import MigrationAnalyzer

# Analyze your codebase
analyzer = MigrationAnalyzer()
results = analyzer.analyze_directory("./my_project")
report = analyzer.generate_report(results)

print(report)
```

## FAQ

**Q: Will my old code break?**
A: No, use BaseAgentAdapter for backward compatibility.

**Q: How long will the adapter be supported?**
A: At least 2 major versions. Deprecated warnings will guide you.

**Q: Is there a performance difference?**
A: New system is same speed or faster. See benchmarks.

**Q: Can I mix old and new agents?**
A: Yes, both can coexist during migration.

## Support

If you encounter issues during migration, please:
1. Check this guide
2. Run MigrationAnalyzer
3. Check example migrations in tests/
4. Open an issue with migration details
"""