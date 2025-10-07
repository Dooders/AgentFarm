# Environment Module Redesign - Implementation Checklist

This checklist breaks down the redesign into actionable tasks with clear acceptance criteria.

## Phase 1: Foundation (Week 1)

### Task 1.1: Create Physics Module Structure

- [x] Create `farm/core/physics/` directory
- [x] Create `farm/core/physics/__init__.py` with exports
- [x] Create `farm/core/physics/interface.py` with protocols
- [ ] Add tests for protocol validation

**Acceptance Criteria:**
- Module can be imported without errors
- Protocol type checking works correctly
- Documentation is complete

**Files:**
- `farm/core/physics/__init__.py` ✅
- `farm/core/physics/interface.py` ✅
- `tests/physics/test_interface.py` ⏳

### Task 1.2: Implement Grid2DPhysics Wrapper

- [ ] Create `farm/core/physics/grid_2d.py`
- [ ] Wrap existing Environment 2D grid logic
- [ ] Implement all IPhysicsEngine methods
- [ ] Maintain existing spatial index integration

**Acceptance Criteria:**
- All protocol methods implemented
- Spatial index works correctly
- Performance matches current system
- Tests pass

**Files:**
- `farm/core/physics/grid_2d.py` ⏳
- `tests/physics/test_grid_2d.py` ⏳

**Code Template:**
```python
class Grid2DPhysics:
    def __init__(self, width: int, height: int, ...):
        self.width = width
        self.height = height
        self.spatial_index = SpatialIndex(width, height)
    
    def validate_position(self, position: Tuple[float, float]) -> bool:
        x, y = position
        return 0 <= x <= self.width and 0 <= y <= self.height
    
    # ... implement other methods
```

### Task 1.3: Add Configuration Support

- [ ] Add `PhysicsConfig` to `farm/config/config.py`
- [ ] Update `SimulationConfig` to include physics
- [ ] Add config validation

**Acceptance Criteria:**
- Can create config with physics parameters
- Config serialization/deserialization works
- Validation catches invalid configs

**Files:**
- `farm/config/config.py` (modify)
- `tests/config/test_physics_config.py` ⏳

**Code Template:**
```python
@dataclass
class PhysicsConfig:
    """Configuration for physics engine."""
    engine_type: str = "grid_2d"
    engine_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        if self.engine_type not in ["grid_2d", "static", "continuous"]:
            raise ValueError(f"Unknown physics engine: {self.engine_type}")
```

---

## Phase 2: Refactor Environment (Week 2)

### Task 2.1: Update Environment.__init__

- [ ] Add `physics_engine` parameter
- [ ] Support backward compatibility (width/height params)
- [ ] Auto-create Grid2DPhysics if old API used
- [ ] Store physics engine as `self.physics`

**Acceptance Criteria:**
- Old API still works: `Environment(width=100, height=100, ...)`
- New API works: `Environment(physics_engine=physics, ...)`
- All existing tests pass
- No breaking changes

**Files:**
- `farm/core/environment.py` (modify `__init__`)
- `tests/test_environment.py` (add new tests)

**Code Template:**
```python
def __init__(
    self,
    config: Optional[SimulationConfig] = None,
    physics_engine: Optional[IPhysicsEngine] = None,
    width: Optional[int] = None,  # Deprecated
    height: Optional[int] = None,  # Deprecated
    **kwargs
):
    # Backward compatibility
    if physics_engine is None:
        if width is not None and height is not None:
            from farm.core.physics import Grid2DPhysics
            physics_engine = Grid2DPhysics(width, height, config=config)
        elif config and hasattr(config, 'physics'):
            physics_engine = create_physics_engine(config.physics)
        else:
            physics_engine = Grid2DPhysics(100, 100, config=config)
    
    self.physics = physics_engine
    # ... rest of init
```

### Task 2.2: Delegate Spatial Operations

- [ ] Update `is_valid_position()` to use physics
- [ ] Update `get_nearby_agents()` to use physics
- [ ] Update `get_nearby_resources()` to use physics
- [ ] Update distance calculations

**Acceptance Criteria:**
- All spatial methods delegate to physics
- No direct width/height checks in environment
- Performance unchanged
- Tests pass

**Files:**
- `farm/core/environment.py` (modify methods)

**Code Template:**
```python
def is_valid_position(self, position: Any) -> bool:
    """Validate position using physics engine."""
    return self.physics.validate_position(position)

def get_nearby_agents(self, position: Any, radius: float) -> List[Any]:
    """Get nearby agents using physics engine."""
    return self.physics.get_nearby_entities(position, radius, "agents")
```

### Task 2.3: Update Services

- [ ] Update `EnvironmentValidationService` to use physics
- [ ] Update `SpatialIndexAdapter` if needed
- [ ] Update any other services with spatial dependencies

**Acceptance Criteria:**
- Services delegate to physics
- No direct environment width/height access
- Tests pass

**Files:**
- `farm/core/services/implementations.py` (modify)

### Task 2.4: Comprehensive Testing

- [ ] Test backward compatibility
- [ ] Test new API
- [ ] Test physics delegation
- [ ] Performance regression tests
- [ ] Integration tests

**Acceptance Criteria:**
- All existing tests pass
- New tests cover both APIs
- No performance regression
- 100% coverage on new code

**Files:**
- `tests/test_environment.py` (add tests)
- `tests/test_environment_physics.py` (new)

---

## Phase 3: New Physics Engines (Week 3)

### Task 3.1: Implement StaticPhysics

- [ ] Create `farm/core/physics/static.py`
- [ ] Implement all protocol methods
- [ ] Add position registry
- [ ] Add entity management

**Acceptance Criteria:**
- Protocol fully implemented
- Works with Environment
- Tests pass
- Example works

**Files:**
- `farm/core/physics/static.py` ⏳
- `tests/physics/test_static.py` ⏳
- Update `farm/core/physics/__init__.py` exports

**Code Template:**
```python
class StaticPhysics:
    def __init__(
        self,
        valid_positions: List[Any],
        state_dim: int,
        observation_space_config: Dict[str, Any],
        config: Optional[Any] = None
    ):
        self.valid_positions = valid_positions
        self.state_dim = state_dim
        self.obs_config = observation_space_config
        self.position_map = {pos: idx for idx, pos in enumerate(valid_positions)}
        self.entities = {"agents": {}, "resources": {}, "objects": {}}
    
    # ... implement protocol methods
```

### Task 3.2: Implement ContinuousPhysics

- [ ] Create `farm/core/physics/continuous.py`
- [ ] Implement all protocol methods
- [ ] Add configurable distance metrics
- [ ] Consider spatial indexing for performance

**Acceptance Criteria:**
- Protocol fully implemented
- Works with Environment
- Multiple distance metrics supported
- Tests pass

**Files:**
- `farm/core/physics/continuous.py` ⏳
- `tests/physics/test_continuous.py` ⏳
- Update `farm/core/physics/__init__.py` exports

**Code Template:**
```python
class ContinuousPhysics:
    def __init__(
        self,
        state_dim: int = 2,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        distance_metric: str = "euclidean",
        config: Optional[Any] = None
    ):
        self.state_dim = state_dim
        self.bounds = bounds
        self.distance_metric = distance_metric
        self.entities = {"agents": [], "resources": [], "objects": []}
    
    # ... implement protocol methods
```

### Task 3.3: Create Examples

- [ ] Create `examples/grid_2d_environment.py`
- [ ] Update `examples/static_catapult_example.py` to use real physics
- [ ] Create `examples/continuous_navigation_example.py`
- [ ] Add example documentation

**Acceptance Criteria:**
- All examples run without errors
- Examples demonstrate key features
- Documentation is clear
- Examples can be used as templates

**Files:**
- `examples/grid_2d_environment.py` ⏳
- `examples/static_catapult_example.py` ✅ (update)
- `examples/continuous_navigation_example.py` ⏳

### Task 3.4: Integration Testing

- [ ] Test all physics engines with Environment
- [ ] Test switching between physics engines
- [ ] Test edge cases
- [ ] Performance benchmarks

**Acceptance Criteria:**
- All combinations work correctly
- No memory leaks when switching
- Performance acceptable
- Edge cases handled

**Files:**
- `tests/integration/test_physics_engines.py` ⏳

---

## Phase 4: Polish & Documentation (Week 4)

### Task 4.1: Add Physics Factory

- [ ] Create `farm/core/physics/factory.py`
- [ ] Implement `create_physics_engine()` function
- [ ] Support config-based creation
- [ ] Support custom physics loading

**Acceptance Criteria:**
- Can create any physics engine from config
- Error messages are helpful
- Custom physics can be registered
- Tests pass

**Files:**
- `farm/core/physics/factory.py` ⏳
- `tests/physics/test_factory.py` ⏳

**Code Template:**
```python
def create_physics_engine(config: PhysicsConfig) -> IPhysicsEngine:
    if config.engine_type == "grid_2d":
        return Grid2DPhysics(**config.engine_params)
    elif config.engine_type == "static":
        return StaticPhysics(**config.engine_params)
    elif config.engine_type == "continuous":
        return ContinuousPhysics(**config.engine_params)
    else:
        raise ValueError(f"Unknown physics engine: {config.engine_type}")
```

### Task 4.2: Performance Optimization

- [ ] Profile each physics engine
- [ ] Optimize hot paths
- [ ] Add spatial indexing to ContinuousPhysics if needed
- [ ] Benchmark against old system

**Acceptance Criteria:**
- Grid2D performance matches old system
- Other engines perform acceptably
- No performance regressions
- Benchmarks documented

**Files:**
- `benchmarks/physics/` (new directory)
- `benchmarks/physics/benchmark_physics_engines.py` ⏳

### Task 4.3: Complete Documentation

- [x] Design report (already done)
- [x] Quick summary (already done)
- [x] Comparison guide (already done)
- [ ] API reference
- [ ] Migration guide
- [ ] Tutorial for custom physics

**Acceptance Criteria:**
- All documents complete
- Examples in documentation work
- Clear migration path
- Custom physics guide

**Files:**
- `docs/design/environment_module_design_report.md` ✅
- `docs/design/environment_redesign_summary.md` ✅
- `docs/design/physics_engine_comparison.md` ✅
- `docs/api/physics_engines.md` ⏳
- `docs/guides/custom_physics_engine.md` ⏳
- `docs/guides/migration_to_physics_engines.md` ⏳

### Task 4.4: Update Main Documentation

- [ ] Update README.md
- [ ] Update core_architecture.md
- [ ] Update user guide
- [ ] Update developer guide
- [ ] Add to module overview

**Acceptance Criteria:**
- All references updated
- New features documented
- Examples correct
- Links work

**Files:**
- `README.md` (update)
- `docs/core_architecture.md` (update)
- `docs/user-guide.md` (update)
- `docs/developer-guide.md` (update)
- `docs/module_overview.md` (update)

---

## Testing Checklist

### Unit Tests

- [ ] `test_interface.py` - Protocol validation
- [ ] `test_grid_2d.py` - Grid2D physics
- [ ] `test_static.py` - Static physics
- [ ] `test_continuous.py` - Continuous physics
- [ ] `test_factory.py` - Physics factory

### Integration Tests

- [ ] `test_environment_physics.py` - Environment with different physics
- [ ] `test_physics_switching.py` - Switching physics at runtime
- [ ] `test_backward_compatibility.py` - Old API still works

### Performance Tests

- [ ] Benchmark Grid2D vs old system
- [ ] Benchmark all physics engines
- [ ] Profile memory usage
- [ ] Test with large number of agents

### Regression Tests

- [ ] All existing tests pass
- [ ] No performance regression
- [ ] No memory leaks
- [ ] No API changes (except additions)

---

## Documentation Checklist

### Design Documents ✅

- [x] Full design report
- [x] Quick summary
- [x] Physics engine comparison
- [x] Implementation checklist (this document)

### API Documentation

- [ ] Physics interface reference
- [ ] Grid2D physics API
- [ ] Static physics API
- [ ] Continuous physics API
- [ ] Factory API

### Guides

- [ ] Migration guide (old to new API)
- [ ] Custom physics engine tutorial
- [ ] Choosing the right physics engine
- [ ] Performance tuning guide

### Examples

- [ ] Grid2D example (standard use)
- [ ] Static catapult example
- [ ] Continuous navigation example
- [ ] Custom physics example

---

## Rollout Plan

### Stage 1: Internal Testing (Days 1-3)

- Implement Phase 1 & 2
- Test with existing codebase
- Verify backward compatibility
- Fix any issues

**Gate:** All existing tests pass

### Stage 2: New Features (Days 4-7)

- Implement Phase 3
- Create new examples
- Test new physics engines
- Performance benchmarks

**Gate:** All new tests pass, examples work

### Stage 3: Documentation (Days 8-10)

- Complete all documentation
- Review by team
- Incorporate feedback

**Gate:** Documentation complete and reviewed

### Stage 4: Release (Days 11-14)

- Merge to main branch
- Update CHANGELOG
- Announce to users
- Monitor for issues

**Gate:** No critical issues

---

## Success Criteria

### Functional

- [x] Physics interface defined
- [ ] All three physics engines implemented
- [ ] Backward compatibility maintained
- [ ] All tests pass
- [ ] Examples work

### Non-Functional

- [ ] No performance regression
- [ ] Documentation complete
- [ ] Code coverage > 90%
- [ ] No breaking changes

### User Experience

- [ ] Easy to swap physics engines
- [ ] Clear error messages
- [ ] Good examples
- [ ] Migration guide clear

---

## Risk Mitigation

### Risk: Performance Regression

**Mitigation:**
- Benchmark early and often
- Keep Grid2D optimizations
- Profile before and after

### Risk: Breaking Changes

**Mitigation:**
- Maintain old API
- Comprehensive tests
- Gradual rollout

### Risk: Complexity

**Mitigation:**
- Keep interface minimal
- Clear documentation
- Good examples

### Risk: Adoption

**Mitigation:**
- Show clear benefits
- Easy migration path
- Support users

---

## Definition of Done

A task is done when:
- [x] Code implemented and reviewed
- [x] Tests written and passing
- [x] Documentation updated
- [x] Examples work
- [x] Performance acceptable
- [x] No known bugs

The entire project is done when:
- [ ] All phases complete
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Examples work
- [ ] Performance acceptable
- [ ] User guide updated
- [ ] Released to users

---

## Next Steps

1. **Review** this checklist with team
2. **Prioritize** tasks based on needs
3. **Assign** tasks to developers
4. **Track** progress using checkboxes
5. **Update** regularly as work progresses

---

*Use this checklist to track implementation progress. Check off items as they're completed.*
