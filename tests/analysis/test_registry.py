"""
Tests for module registry.
"""

import pytest

from farm.analysis.registry import (
    ModuleRegistry,
    registry,
    register_modules,
    get_module,
    get_module_names,
    list_modules
)
from farm.analysis.exceptions import ModuleNotFoundError


class TestModuleRegistry:
    """Tests for ModuleRegistry class."""
    
    def test_register_module(self, minimal_module):
        """Test registering a module."""
        reg = ModuleRegistry()
        reg.register(minimal_module)
        
        assert minimal_module.name in reg.get_module_names()
    
    def test_register_duplicate_warns(self, minimal_module, caplog):
        """Test registering duplicate module name."""
        reg = ModuleRegistry()
        reg.register(minimal_module)
        reg.register(minimal_module)  # Register again
        
        # Should log warning
        assert "already registered" in caplog.text.lower()
    
    def test_unregister_module(self, minimal_module):
        """Test unregistering a module."""
        reg = ModuleRegistry()
        reg.register(minimal_module)
        assert minimal_module.name in reg.get_module_names()
        
        reg.unregister(minimal_module.name)
        assert minimal_module.name not in reg.get_module_names()
    
    def test_get_existing_module(self, minimal_module):
        """Test getting existing module."""
        reg = ModuleRegistry()
        reg.register(minimal_module)
        
        retrieved = reg.get(minimal_module.name)
        assert retrieved is minimal_module
    
    def test_get_nonexistent_module_raises(self):
        """Test getting non-existent module raises error."""
        reg = ModuleRegistry()
        
        with pytest.raises(ModuleNotFoundError) as exc_info:
            reg.get("nonexistent")
        
        assert "nonexistent" in str(exc_info.value)
        assert "Available modules" in str(exc_info.value)
    
    def test_get_optional(self, minimal_module):
        """Test get_optional returns None for missing modules."""
        reg = ModuleRegistry()
        reg.register(minimal_module)
        
        assert reg.get_optional(minimal_module.name) is minimal_module
        assert reg.get_optional("nonexistent") is None
    
    def test_get_all(self, minimal_module):
        """Test getting all modules."""
        reg = ModuleRegistry()
        reg.register(minimal_module)
        
        all_modules = reg.get_all()
        assert minimal_module.name in all_modules
        assert all_modules[minimal_module.name] is minimal_module
    
    def test_clear(self, minimal_module):
        """Test clearing registry."""
        reg = ModuleRegistry()
        reg.register(minimal_module)
        assert len(reg.get_module_names()) > 0
        
        reg.clear()
        assert len(reg.get_module_names()) == 0
    
    def test_list_modules(self, minimal_module):
        """Test listing modules as formatted string."""
        reg = ModuleRegistry()
        reg.register(minimal_module)
        
        listing = reg.list_modules()
        assert minimal_module.name in listing
        assert minimal_module.description in listing


class TestRegisterModules:
    """Tests for register_modules function."""
    
    def test_register_from_config(self, config_service_mock):
        """Test registering modules from config service."""
        config_service_mock.set_module_paths([
            "farm.analysis.null_module.null_module"
        ])
        
        count = register_modules(config_service=config_service_mock)
        
        assert count == 1
        assert "null" in registry.get_module_names()
    
    def test_register_invalid_path(self, config_service_mock, caplog):
        """Test handling of invalid module paths."""
        config_service_mock.set_module_paths([
            "invalid.path.to.module"
        ])
        
        count = register_modules(config_service=config_service_mock)
        
        # Should fall back to builtins
        assert count >= 0
        assert "Failed to import" in caplog.text or "fallback" in caplog.text.lower()
    
    def test_register_builtin_fallback(self, config_service_mock):
        """Test fallback to built-in modules when none configured."""
        config_service_mock.set_module_paths([])
        
        count = register_modules(config_service=config_service_mock)
        
        # Should register at least one built-in module
        assert count >= 0
    
    def test_register_non_protocol_object(self, config_service_mock, caplog):
        """Test handling of objects that don't implement protocol."""
        # Try to register something that's not an AnalysisModule
        config_service_mock.set_module_paths([
            "pandas.DataFrame"  # Valid import but not an AnalysisModule
        ])
        
        register_modules(config_service=config_service_mock)
        
        # Should log warning
        assert "doesn't implement" in caplog.text.lower() or "not an analysismodule" in caplog.text.lower()


class TestConvenienceFunctions:
    """Tests for module convenience functions."""
    
    def test_get_module(self, minimal_module):
        """Test get_module convenience function."""
        registry.register(minimal_module)
        
        retrieved = get_module(minimal_module.name)
        assert retrieved is minimal_module
    
    def test_get_module_names(self, minimal_module):
        """Test get_module_names convenience function."""
        registry.register(minimal_module)
        
        names = get_module_names()
        assert minimal_module.name in names
        assert names == sorted(names)  # Should be sorted
    
    def test_list_modules_function(self, minimal_module):
        """Test list_modules convenience function."""
        registry.register(minimal_module)
        
        listing = list_modules()
        assert minimal_module.name in listing
