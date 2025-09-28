# Hierarchical Configuration Management Implementation Plan

## Current State Analysis

### Existing Configuration System
- **Primary Config**: `SimulationConfig` dataclass in `farm/core/config.py` with 400+ lines
- **Nested Configs**: `VisualizationConfig`, `RedisMemoryConfig` as separate dataclasses
- **Service Layer**: `IConfigService` interface with `EnvConfigService` implementation
- **File I/O**: YAML-based loading/saving with `from_yaml()` and `to_yaml()` methods
- **Validation**: Basic validation exists but no comprehensive hierarchical system
- **Current Structure**: Flat configuration with some nested objects

### Identified Issues
1. **Monolithic Config**: Single large `SimulationConfig` class violates SRP
2. **No Environment Overrides**: No support for environment-specific configurations
3. **Limited Validation**: No runtime validation at startup
4. **No Migration System**: No version compatibility handling
5. **No Hot Reloading**: No dynamic configuration updates
6. **No Hierarchical Lookup**: No inheritance-based configuration resolution

## Implementation Plan

### Phase 1: Core Hierarchical Configuration Framework

#### 1.1 Create Base Configuration Classes
**Files to Create/Modify:**
- `farm/core/config/hierarchical.py` (new)
- `farm/core/config/validation.py` (new)
- `farm/core/config/exceptions.py` (new)

**Implementation:**
```python
@dataclass
class HierarchicalConfig:
    """Hierarchical configuration with inheritance"""
    global_config: Dict[str, Any] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    agent_config: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default=None):
        """Get configuration value with hierarchical lookup"""
        # Check agent-specific config first
        if key in self.agent_config:
            return self.agent_config[key]
        
        # Check environment-specific config
        if key in self.environment_config:
            return self.environment_config[key]
        
        # Fall back to global config
        return self.global_config.get(key, default)
    
    def validate(self):
        """Validate configuration consistency"""
        required_keys = ['simulation_id', 'max_steps', 'environment']
        for key in required_keys:
            if not self.get(key):
                raise ValidationException(key, None, f"Required configuration key '{key}' is missing")
```

#### 1.2 Configuration Validation System
**Implementation:**
```python
class ConfigurationValidator:
    """Runtime configuration validation at startup"""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def validate_config(self, config: HierarchicalConfig) -> ValidationResult:
        """Validate configuration against schema"""
        errors = []
        warnings = []
        
        # Validate required fields
        for field, rules in self.schema.get('required', {}).items():
            if not config.get(field):
                errors.append(f"Required field '{field}' is missing")
        
        # Validate field types and constraints
        for field, rules in self.schema.get('fields', {}).items():
            value = config.get(field)
            if value is not None:
                self._validate_field(field, value, rules, errors, warnings)
        
        return ValidationResult(errors=errors, warnings=warnings)
```

#### 1.3 Configuration Exceptions
**Implementation:**
```python
class ConfigurationError(Exception):
    """Base exception for configuration-related errors"""
    pass

class ValidationException(ConfigurationError):
    """Exception raised when configuration validation fails"""
    def __init__(self, field: str, value: Any, message: str):
        self.field = field
        self.value = value
        super().__init__(f"Validation error for '{field}': {message}")

class ConfigurationMigrationError(ConfigurationError):
    """Exception raised when configuration migration fails"""
    pass
```

### Phase 2: Environment-Specific Configuration System

#### 2.1 Environment Configuration Manager
**Files to Create:**
- `farm/core/config/environment.py` (new)

**Implementation:**
```python
class EnvironmentConfigManager:
    """Manages environment-specific configuration overrides"""
    
    def __init__(self, base_config_path: str, environment: str = None):
        self.base_config_path = base_config_path
        self.environment = environment or os.getenv('FARM_ENVIRONMENT', 'default')
        self.config_hierarchy = self._load_config_hierarchy()
    
    def _load_config_hierarchy(self) -> HierarchicalConfig:
        """Load configuration with environment-specific overrides"""
        # Load base configuration
        base_config = self._load_yaml_config(self.base_config_path)
        
        # Load environment-specific overrides
        env_config_path = self._get_environment_config_path()
        env_config = {}
        if os.path.exists(env_config_path):
            env_config = self._load_yaml_config(env_config_path)
        
        # Load agent-specific overrides (if any)
        agent_config = self._load_agent_specific_config()
        
        return HierarchicalConfig(
            global_config=base_config,
            environment_config=env_config,
            agent_config=agent_config
        )
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get the effective configuration after applying all overrides"""
        effective_config = {}
        
        # Start with global config
        effective_config.update(self.config_hierarchy.global_config)
        
        # Apply environment overrides
        effective_config.update(self.config_hierarchy.environment_config)
        
        # Apply agent-specific overrides
        effective_config.update(self.config_hierarchy.agent_config)
        
        return effective_config
```

#### 2.2 Configuration File Structure
**Directory Structure:**
```
config/
├── base.yaml                 # Base configuration
├── environments/
│   ├── development.yaml      # Development overrides
│   ├── staging.yaml         # Staging overrides
│   ├── production.yaml      # Production overrides
│   └── testing.yaml         # Testing overrides
├── agents/
│   ├── system_agent.yaml    # System agent specific config
│   ├── independent_agent.yaml
│   └── control_agent.yaml
└── migrations/
    ├── v1.0_to_v1.1.yaml    # Migration scripts
    └── v1.1_to_v1.2.yaml
```

### Phase 3: Configuration Migration System

#### 3.1 Migration Framework
**Files to Create:**
- `farm/core/config/migration.py` (new)

**Implementation:**
```python
class ConfigurationMigrator:
    """Handles configuration migration for version compatibility"""
    
    def __init__(self, migrations_dir: str):
        self.migrations_dir = migrations_dir
        self.migrations = self._load_migrations()
    
    def migrate_config(self, config: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Migrate configuration from one version to another"""
        current_version = from_version
        migrated_config = config.copy()
        
        while current_version != to_version:
            migration = self._get_next_migration(current_version)
            if not migration:
                break
            
            migrated_config = migration.apply(migrated_config)
            current_version = migration.to_version
        
        return migrated_config
    
    def _get_next_migration(self, current_version: str) -> Optional[ConfigurationMigration]:
        """Get the next migration to apply"""
        for migration in self.migrations:
            if migration.from_version == current_version:
                return migration
        return None

class ConfigurationMigration:
    """Individual configuration migration"""
    
    def __init__(self, from_version: str, to_version: str, transformations: List[Dict[str, Any]]):
        self.from_version = from_version
        self.to_version = to_version
        self.transformations = transformations
    
    def apply(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply migration transformations to configuration"""
        migrated_config = config.copy()
        
        for transformation in self.transformations:
            migrated_config = self._apply_transformation(migrated_config, transformation)
        
        return migrated_config
```

### Phase 4: Hot-Reloading Configuration System

#### 4.1 Configuration Watcher
**Files to Create:**
- `farm/core/config/hot_reload.py` (new)

**Implementation:**
```python
class ConfigurationWatcher:
    """Watches configuration files for changes and triggers hot reload"""
    
    def __init__(self, config_paths: List[str], callback: Callable[[Dict[str, Any]], None]):
        self.config_paths = config_paths
        self.callback = callback
        self.watcher = None
        self.file_timestamps = {}
    
    def start_watching(self):
        """Start watching configuration files for changes"""
        for path in self.config_paths:
            if os.path.exists(path):
                self.file_timestamps[path] = os.path.getmtime(path)
        
        # Use watchdog library for file system monitoring
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, watcher):
                self.watcher = watcher
            
            def on_modified(self, event):
                if event.src_path in self.watcher.config_paths:
                    self.watcher._handle_config_change(event.src_path)
        
        self.observer = Observer()
        self.observer.schedule(ConfigFileHandler(self), path=os.path.dirname(self.config_paths[0]), recursive=False)
        self.observer.start()
    
    def _handle_config_change(self, file_path: str):
        """Handle configuration file change"""
        try:
            # Reload configuration
            new_config = self._load_config(file_path)
            
            # Validate new configuration
            validator = ConfigurationValidator(self._get_schema())
            validation_result = validator.validate_config(new_config)
            
            if validation_result.is_valid:
                # Apply new configuration
                self.callback(new_config)
                logger.info(f"Configuration hot-reloaded from {file_path}")
            else:
                logger.error(f"Configuration validation failed: {validation_result.errors}")
        
        except Exception as e:
            logger.error(f"Failed to hot-reload configuration: {e}")
```

### Phase 5: Integration and Migration

#### 5.1 Refactor Existing Configuration
**Files to Modify:**
- `farm/core/config.py` (refactor)
- `farm/core/services/implementations.py` (update)

**Steps:**
1. **Extract Configuration Sections**: Break down `SimulationConfig` into logical sections
2. **Create Section-Specific Configs**: 
   - `EnvironmentConfig`
   - `AgentConfig` 
   - `LearningConfig`
   - `VisualizationConfig` (existing)
   - `DatabaseConfig`
   - `RedisConfig` (existing)
3. **Implement Hierarchical Lookup**: Replace direct attribute access with hierarchical lookup
4. **Add Validation**: Integrate validation system into existing config loading

#### 5.2 Update Service Layer
**Implementation:**
```python
class HierarchicalConfigService(IConfigService):
    """Enhanced configuration service with hierarchical support"""
    
    def __init__(self, config_manager: EnvironmentConfigManager):
        self.config_manager = config_manager
        self.validator = ConfigurationValidator(self._get_schema())
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get configuration value with hierarchical lookup"""
        value = self.config_manager.config_hierarchy.get(key, default)
        return str(value) if value is not None else default
    
    def get_typed(self, key: str, expected_type: type, default: Any = None) -> Any:
        """Get configuration value with type conversion"""
        value = self.config_manager.config_hierarchy.get(key, default)
        if value is None:
            return default
        
        try:
            return expected_type(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert '{key}' to {expected_type.__name__}, using default")
            return default
    
    def validate_configuration(self) -> ValidationResult:
        """Validate current configuration"""
        return self.validator.validate_config(self.config_manager.config_hierarchy)
```

### Phase 6: Testing and Documentation

#### 6.1 Comprehensive Testing
**Files to Create:**
- `tests/test_hierarchical_config.py`
- `tests/test_config_validation.py`
- `tests/test_config_migration.py`
- `tests/test_hot_reload.py`

**Test Coverage:**
- Hierarchical configuration lookup
- Environment-specific overrides
- Configuration validation
- Migration system
- Hot-reloading functionality
- Error handling and edge cases

#### 6.2 Documentation Updates
**Files to Update:**
- `docs/configuration_guide.md` (update)
- `docs/hierarchical_configuration.md` (new)

**Documentation Topics:**
- Hierarchical configuration concepts
- Environment-specific configuration setup
- Configuration migration procedures
- Hot-reloading configuration
- Best practices and troubleshooting

## Implementation Timeline

### Week 1-2: Core Framework
- [ ] Implement `HierarchicalConfig` base class
- [ ] Create validation system
- [ ] Add configuration exceptions
- [ ] Write unit tests for core functionality

### Week 3-4: Environment System
- [ ] Implement `EnvironmentConfigManager`
- [ ] Create environment-specific configuration structure
- [ ] Add configuration file loading logic
- [ ] Test environment override functionality

### Week 5-6: Migration System
- [ ] Implement `ConfigurationMigrator`
- [ ] Create migration framework
- [ ] Add version compatibility handling
- [ ] Test migration scenarios

### Week 7-8: Hot Reloading
- [ ] Implement `ConfigurationWatcher`
- [ ] Add file system monitoring
- [ ] Create hot-reload callback system
- [ ] Test dynamic configuration updates

### Week 9-10: Integration
- [ ] Refactor existing `SimulationConfig`
- [ ] Update service layer
- [ ] Migrate existing configuration files
- [ ] Update documentation

### Week 11-12: Testing & Polish
- [ ] Comprehensive integration testing
- [ ] Performance testing
- [ ] Documentation completion
- [ ] Code review and refinement

## Benefits of This Implementation

### 1. **Single Responsibility Principle (SRP)**
- Each configuration class has a single, well-defined purpose
- Separation of concerns between global, environment, and agent-specific configs

### 2. **Open-Closed Principle (OCP)**
- Easy to extend with new configuration sections without modifying existing code
- Migration system allows for backward compatibility

### 3. **Dependency Inversion Principle (DIP)**
- Configuration service depends on abstractions (`IConfigService`)
- Easy to swap different configuration implementations

### 4. **Don't Repeat Yourself (DRY)**
- Centralized configuration loading and validation logic
- Reusable migration and validation components

### 5. **Keep It Simple, Stupid (KISS)**
- Clear hierarchical lookup with simple precedence rules
- Straightforward configuration file structure

### 6. **Composition Over Inheritance**
- Configuration sections are composed together rather than inherited
- Flexible mixing and matching of configuration components

## Risk Mitigation

### 1. **Backward Compatibility**
- Maintain existing `SimulationConfig` interface during transition
- Provide migration tools for existing configurations
- Gradual rollout with feature flags

### 2. **Performance Impact**
- Lazy loading of configuration sections
- Caching of resolved configuration values
- Efficient file watching for hot-reload

### 3. **Configuration Complexity**
- Clear documentation and examples
- Validation with helpful error messages
- Configuration templates for common scenarios

### 4. **Testing Coverage**
- Comprehensive unit tests for all components
- Integration tests for end-to-end scenarios
- Performance benchmarks for configuration loading

This implementation plan provides a robust, scalable, and maintainable hierarchical configuration system that addresses all the requirements while following SOLID principles and best practices.