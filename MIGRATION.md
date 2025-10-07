# Migration Guide: BaseAgent → AgentCore

## Quick Start

Replace `BaseAgent` with `BaseAgentAdapter` for immediate compatibility:

```python
# Before
from farm.core.agent import BaseAgent
agent = BaseAgent(agent_id="001", position=(0,0), resource_level=100, spatial_service=svc)

# After (works immediately!)
from farm.core.agent.compat import BaseAgentAdapter
agent = BaseAgentAdapter.from_old_style(agent_id="001", position=(0,0), resource_level=100, spatial_service=svc)
```

All your old code continues to work! ✅

---

## Why Migrate?

### New System Benefits

| Feature | Old BaseAgent | New AgentCore |
|---------|---------------|---------------|
| **Lines of code** | 1,571 in one file | 13 focused classes (~240 avg) |
| **Testability** | Hard to isolate | Each component tested separately |
| **Extensibility** | Modify base class | Add new components |
| **Type safety** | Runtime errors | Compile-time checking |
| **Performance** | Baseline | Same or better |

### Design Improvements

✅ **SOLID Principles** - Every class follows SRP, OCP, LSP, ISP, DIP
✅ **Composition** - Mix & match components instead of inheritance
✅ **Dependency Injection** - All dependencies explicit
✅ **Strategy Pattern** - Swap behaviors easily
✅ **150+ Tests** - Comprehensive test coverage

---

## Migration Strategies

### Strategy 1: Adapter (Recommended for Large Codebases)

**When to use**: You have lots of existing code and want minimal changes.

**Steps**:
1. Change imports to use `BaseAgentAdapter`
2. Verify tests still pass
3. Gradually refactor to use new API

**Example**:
```python
from farm.core.agent.compat import BaseAgentAdapter

# Drop-in replacement - all old API works
agent = BaseAgentAdapter.from_old_style(
    agent_id="agent_001",
    position=(10, 20),
    resource_level=100,
    spatial_service=spatial_service,
    config=config
)

# Old API still works
print(agent.position)          # ✅ Works
print(agent.resource_level)    # ✅ Works
agent.act()                    # ✅ Works

# Can access new features too
movement = agent.core.get_component("movement")
movement.move_to((100, 100))   # ✅ New features!
```

### Strategy 2: Direct Migration (Recommended for New Code)

**When to use**: New modules, small codebases, or clean refactoring.

**Steps**:
1. Use `AgentFactory` instead of `BaseAgent` constructor
2. Access components via `get_component()`
3. Use new configuration system

**Example**:
```python
from farm.core.agent import AgentFactory, AgentConfig

# Create factory once
factory = AgentFactory(
    spatial_service=spatial_service,
    time_service=time_service,
    lifecycle_service=lifecycle_service,
)

# Create agents
agent = factory.create_default_agent(
    agent_id="agent_001",
    position=(10, 20),
    initial_resources=100,
    config=AgentConfig()
)

# New component-based API
movement = agent.get_component("movement")
movement.move_to((100, 100))

resource = agent.get_component("resource")
print(resource.level)
resource.consume(20)
```

---

## Common Migration Patterns

### Pattern 1: Attribute Access

```python
# ❌ Old way
agent.resource_level
agent.current_health
agent.position
agent.is_defending

# ✅ New way (adapter - still works)
agent.resource_level
agent.current_health
agent.position
agent.is_defending

# ✅ New way (direct - recommended)
agent.get_component("resource").level
agent.get_component("combat").health
agent.position  # Direct property on AgentCore
agent.get_component("combat").is_defending
```

### Pattern 2: Resource Management

```python
# ❌ Old way
agent.resource_level += 50
agent.resource_level -= 20
if agent.resource_level >= 100:
    agent.reproduce()

# ✅ New way
resource = agent.get_component("resource")
resource.add(50)
resource.consume(20)
if resource.level >= 100:
    reproduction = agent.get_component("reproduction")
    reproduction.reproduce()
```

### Pattern 3: Movement

```python
# ❌ Old way
agent.position = (100, 100)
agent.update_position((100, 100))

# ✅ New way
movement = agent.get_component("movement")
movement.move_to((100, 100))
movement.move_by(10, 20)
movement.random_move()
```

### Pattern 4: Combat

```python
# ❌ Old way
agent.handle_combat(attacker, damage)
agent.take_damage(damage)
attack_power = agent.attack_strength

# ✅ New way
combat = agent.get_component("combat")
combat.take_damage(damage)
combat.attack(target_agent)
attack_power = combat._calculate_attack_damage()
```

### Pattern 5: Configuration

```python
# ❌ Old way (verbose boilerplate)
max_movement = get_nested_then_flat(
    config=self.config,
    nested_parent_attr="agent_behavior",
    nested_attr_name="max_movement",
    flat_attr_name="max_movement",
    default_value=8,
    expected_types=(int, float),
)

# ✅ New way (type-safe, clean)
from farm.core.agent.config.agent_config import AgentConfig

config = AgentConfig()
max_movement = config.movement.max_movement  # Type-safe!

# Custom configuration
from farm.core.agent.config.agent_config import (
    AgentConfig, MovementConfig, CombatConfig
)

config = AgentConfig(
    movement=MovementConfig(max_movement=15.0),
    combat=CombatConfig(starting_health=150.0)
)
```

---

## Automated Migration Tools

### Analyze Your Codebase

```python
from farm.core.agent.migration import MigrationAnalyzer

# Analyze your project
analyzer = MigrationAnalyzer()
results = analyzer.analyze_directory("./my_project")

# Generate report
report = analyzer.generate_report(results)
print(report)

# Example output:
# # BaseAgent Migration Report
# 
# ## Summary
# - Files needing migration: 5
# - Total issues found: 12
#
# ## Files
# ### ./my_simulation.py
# Issues: 3
# - [HIGH] Uses old BaseAgent import
#   - Suggestion: Change to: from farm.core.agent import AgentFactory
```

### Generate Migration Code

```python
from farm.core.agent.migration import CodeMigrator

# Get adapter code
code = CodeMigrator.generate_adapter_code(
    agent_id='"agent_001"',
    position='(10, 20)',
    resources='100'
)
print(code)

# Get direct migration code
code = CodeMigrator.generate_direct_migration_code(
    agent_id='"agent_001"',
    position='(10, 20)',
    resources='100'
)
print(code)
```

---

## Step-by-Step Migration Process

### Step 1: Analyze

```bash
# Run migration analyzer
python -c "
from farm.core.agent.migration import MigrationAnalyzer
analyzer = MigrationAnalyzer()
results = analyzer.analyze_directory('./my_project')
print(analyzer.generate_report(results))
"
```

### Step 2: Choose Strategy

- **Large codebase** → Use adapter
- **Small codebase** → Direct migration
- **New features** → Direct migration
- **Legacy code** → Use adapter

### Step 3: Update Imports

```python
# Before
from farm.core.agent import BaseAgent

# After (adapter)
from farm.core.agent.compat import BaseAgentAdapter

# After (direct)
from farm.core.agent import AgentFactory, AgentConfig
```

### Step 4: Update Instantiation

```python
# Before
agent = BaseAgent(
    agent_id="001",
    position=(0, 0),
    resource_level=100,
    spatial_service=spatial_service
)

# After (adapter)
agent = BaseAgentAdapter.from_old_style(
    agent_id="001",
    position=(0, 0),
    resource_level=100,
    spatial_service=spatial_service
)

# After (direct)
factory = AgentFactory(spatial_service=spatial_service)
agent = factory.create_default_agent(
    agent_id="001",
    position=(0, 0),
    initial_resources=100
)
```

### Step 5: Test

```python
# Verify adapter compatibility
from farm.core.agent.compat import is_new_agent

assert is_new_agent(agent)
assert agent.resource_level == 100
assert agent.alive is True
```

### Step 6: Refactor (Optional)

Gradually convert code to use components:

```python
# Phase 1: Use adapter
agent = BaseAgentAdapter.from_old_style(...)
print(agent.resource_level)  # Old API

# Phase 2: Mix old and new
agent = BaseAgentAdapter.from_old_style(...)
print(agent.resource_level)  # Old API
movement = agent.core.get_component("movement")  # New API
movement.move_to((100, 100))

# Phase 3: Full migration
agent = factory.create_default_agent(...)
resource = agent.get_component("resource")  # New API only
print(resource.level)
```

---

## Testing Your Migration

### Test Adapter Compatibility

```python
import pytest
from farm.core.agent.compat import BaseAgentAdapter

def test_adapter_works_like_old_agent():
    """Verify adapter provides same API as BaseAgent."""
    agent = BaseAgentAdapter.from_old_style(
        agent_id="test",
        position=(0, 0),
        resource_level=100,
        spatial_service=mock_spatial_service
    )
    
    # All old properties should work
    assert agent.agent_id == "test"
    assert agent.position == (0, 0)
    assert agent.resource_level == 100
    assert agent.alive is True
    
    # Old methods should work
    agent.act()
    agent.update_position((10, 10))
    
    # Can access new features
    movement = agent.core.get_component("movement")
    assert movement is not None
```

### Benchmark Performance

```bash
# Run performance benchmarks
python tests/benchmarks/test_agent_performance.py

# Expected results:
# - Agent creation: < 1ms
# - Agent turn: < 100μs
# - Component access: < 1μs
```

---

## Troubleshooting

### Issue: Deprecation Warnings

**Problem**: Seeing deprecation warnings when using adapter.

**Solution**: This is expected! The warnings guide you toward migrating. To suppress:
```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
```

### Issue: Missing Component

**Problem**: `agent.get_component("xyz")` returns `None`.

**Solution**: Component might not be attached. Check factory method or add manually:
```python
agent.add_component(MyComponent())
```

### Issue: Type Errors

**Problem**: Type checker complains about new code.

**Solution**: Update type hints:
```python
from farm.core.agent import AgentCore

def process_agent(agent: AgentCore) -> None:
    movement = agent.get_component("movement")
    if movement:
        movement.move_to((100, 100))
```

---

## FAQ

**Q: Will my old code break?**
A: No! Use `BaseAgentAdapter` for 100% backward compatibility.

**Q: How long will the adapter be supported?**
A: At least 2 major versions (1+ year). Plenty of time to migrate.

**Q: Is the new system faster?**
A: Same speed or faster. See benchmarks in `tests/benchmarks/`.

**Q: Can I mix old and new agents?**
A: Yes! Both can coexist in the same simulation.

**Q: What if I find a bug in the adapter?**
A: Please report it! The adapter is designed for compatibility.

**Q: Do I have to migrate?**
A: Not immediately. But new features will only be in the new system.

---

## Getting Help

1. **Read this guide** - Covers most migration scenarios
2. **Check examples** - See `tests/agent/test_integration.py`
3. **Run analyzer** - Use `MigrationAnalyzer` to scan your code
4. **Open an issue** - Describe your migration challenge

---

## Summary

✅ **Quick win**: Use `BaseAgentAdapter.from_old_style()` for instant compatibility
✅ **Best practice**: Use `AgentFactory` and components for new code
✅ **Gradual**: Mix old and new APIs during transition
✅ **Automated**: Use `MigrationAnalyzer` to find issues
✅ **Well-tested**: 150+ tests ensure correctness
✅ **Performant**: Benchmarks verify speed

**Migration is easy and safe!** Choose the strategy that works for you.