"""
Comprehensive unit tests for AgentComponent base class.

Tests all functionality including initialization, attachment, lifecycle hooks,
service properties, and debug logging.
"""

import pytest
from unittest.mock import Mock, patch

from farm.core.agent.components.base import AgentComponent
from farm.core.agent.services import AgentServices


class TestAgentComponentInitialization:
    """Test component initialization and configuration."""
    
    def test_init_with_explicit_name(self):
        """Test initialization with explicit component name."""
        services = Mock(spec=AgentServices)
        component = AgentComponent(services, "TestComponent")
        
        assert component.services == services
        assert component.name == "TestComponent"
        assert component.core is None
    
    def test_init_with_class_name_fallback(self):
        """Test initialization with class name as fallback."""
        services = Mock(spec=AgentServices)
        component = AgentComponent(services)
        
        assert component.services == services
        assert component.name == "AgentComponent"
        assert component.core is None
    
    def test_init_with_empty_name_uses_class_name(self):
        """Test initialization with empty string name uses class name."""
        services = Mock(spec=AgentServices)
        component = AgentComponent(services, "")
        
        assert component.services == services
        assert component.name == "AgentComponent"
        assert component.core is None


class TestAttachment:
    """Test component attachment to agent core."""
    
    def test_attach_to_core(self):
        """Test attaching component to agent core."""
        services = Mock(spec=AgentServices)
        component = AgentComponent(services)
        
        core = Mock()
        component.attach(core)
        
        assert component.core == core
    
    def test_attach_multiple_times(self):
        """Test attaching component to different cores."""
        services = Mock(spec=AgentServices)
        component = AgentComponent(services)
        
        core1 = Mock()
        core2 = Mock()
        
        component.attach(core1)
        assert component.core == core1
        
        component.attach(core2)
        assert component.core == core2


class TestLifecycleHooks:
    """Test lifecycle hook methods."""
    
    def test_lifecycle_hooks_are_callable(self):
        """Test that all lifecycle hooks are callable without errors."""
        services = Mock(spec=AgentServices)
        component = AgentComponent(services)
        
        # These should not raise exceptions
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()
    
    def test_lifecycle_hooks_with_attached_core(self):
        """Test lifecycle hooks work with attached core."""
        services = Mock(spec=AgentServices)
        component = AgentComponent(services)
        
        core = Mock()
        component.attach(core)
        
        # These should not raise exceptions
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()


class TestServiceProperties:
    """Test service property accessors."""
    
    def test_logging_service_property(self):
        """Test logging service property access."""
        services = Mock(spec=AgentServices)
        services.logging_service = Mock()
        component = AgentComponent(services)
        
        assert component.logging_service == services.logging_service
    
    def test_metrics_service_property(self):
        """Test metrics service property access."""
        services = Mock(spec=AgentServices)
        services.metrics_service = Mock()
        component = AgentComponent(services)
        
        assert component.metrics_service == services.metrics_service
    
    def test_validation_service_property(self):
        """Test validation service property access."""
        services = Mock(spec=AgentServices)
        services.validation_service = Mock()
        component = AgentComponent(services)
        
        assert component.validation_service == services.validation_service
    
    def test_time_service_property(self):
        """Test time service property access."""
        services = Mock(spec=AgentServices)
        services.time_service = Mock()
        component = AgentComponent(services)
        
        assert component.time_service == services.time_service
    
    def test_spatial_service_property(self):
        """Test spatial service property access."""
        services = Mock(spec=AgentServices)
        services.spatial_service = Mock()
        component = AgentComponent(services)
        
        assert component.spatial_service == services.spatial_service
    
    def test_lifecycle_service_property(self):
        """Test lifecycle service property access."""
        services = Mock(spec=AgentServices)
        services.lifecycle_service = Mock()
        component = AgentComponent(services)
        
        assert component.lifecycle_service == services.lifecycle_service
    
    def test_services_can_be_none(self):
        """Test that services can be None without errors."""
        services = Mock(spec=AgentServices)
        services.logging_service = None
        services.metrics_service = None
        services.validation_service = None
        services.time_service = None
        services.spatial_service = None
        services.lifecycle_service = None
        component = AgentComponent(services)
        
        assert component.logging_service is None
        assert component.metrics_service is None
        assert component.validation_service is None
        assert component.time_service is None
        assert component.spatial_service is None
        assert component.lifecycle_service is None


class TestCurrentTime:
    """Test current time property."""
    
    def test_current_time_with_time_service(self):
        """Test current time with available time service."""
        time_service = Mock()
        time_service.current_time.return_value = 42
        services = AgentServices(
            spatial_service=Mock(),
            time_service=time_service
        )
        component = AgentComponent(services)
        
        assert component.current_time == 42
        time_service.current_time.assert_called_once()
    
    def test_current_time_without_time_service(self):
        """Test current time without time service returns 0."""
        services = AgentServices(
            spatial_service=Mock(),
            time_service=None
        )
        component = AgentComponent(services)
        
        assert component.current_time == 0
    
    def test_current_time_service_error(self):
        """Test current time when time service raises exception."""
        time_service = Mock()
        time_service.current_time.side_effect = Exception("Time service error")
        services = AgentServices(
            spatial_service=Mock(),
            time_service=time_service
        )
        component = AgentComponent(services)
        
        # Should raise exception from time service
        with pytest.raises(Exception, match="Time service error"):
            _ = component.current_time


class TestDebugLogging:
    """Test debug logging functionality."""
    
    def test_log_debug_with_logging_service(self):
        """Test debug logging with available logging service."""
        services = Mock(spec=AgentServices)
        logging_service = Mock()
        services.logging_service = logging_service
        component = AgentComponent(services, "TestComponent")
        
        component._log_debug("Test message")
        
        logging_service.log_debug.assert_called_once_with("[TestComponent] Test message")
    
    def test_log_debug_without_logging_service(self):
        """Test debug logging without logging service does not raise error."""
        services = Mock(spec=AgentServices)
        services.logging_service = None
        component = AgentComponent(services)
        
        # Should not raise exception
        component._log_debug("Test message")
    
    def test_log_debug_logging_service_error(self):
        """Test debug logging when logging service raises exception."""
        services = Mock(spec=AgentServices)
        logging_service = Mock()
        logging_service.log_debug.side_effect = Exception("Logging error")
        services.logging_service = logging_service
        component = AgentComponent(services)
        
        # Should not raise exception
        component._log_debug("Test message")
    
    def test_log_debug_message_formatting(self):
        """Test that debug messages are properly formatted with component name."""
        services = Mock(spec=AgentServices)
        logging_service = Mock()
        services.logging_service = logging_service
        component = AgentComponent(services, "CustomComponent")
        
        component._log_debug("Debug info")
        
        logging_service.log_debug.assert_called_once_with("[CustomComponent] Debug info")
    
    def test_log_debug_empty_message(self):
        """Test debug logging with empty message."""
        services = Mock(spec=AgentServices)
        logging_service = Mock()
        services.logging_service = logging_service
        component = AgentComponent(services)
        
        component._log_debug("")
        
        logging_service.log_debug.assert_called_once_with("[AgentComponent] ")


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_component_with_all_none_services(self):
        """Test component with all services set to None."""
        services = AgentServices(
            spatial_service=Mock(),  # Required field
            time_service=None,
            metrics_service=None,
            logging_service=None,
            validation_service=None,
            lifecycle_service=None
        )
        component = AgentComponent(services)
        
        # All properties should return None
        assert component.logging_service is None
        assert component.metrics_service is None
        assert component.validation_service is None
        assert component.time_service is None
        assert component.spatial_service is not None  # Required field
        assert component.lifecycle_service is None
        
        # Current time should return 0
        assert component.current_time == 0
        
        # Debug logging should not raise exception
        component._log_debug("Test")
    
    def test_component_with_mixed_services(self):
        """Test component with some services available and some None."""
        services = Mock(spec=AgentServices)
        services.logging_service = Mock()
        services.metrics_service = None
        services.validation_service = Mock()
        services.time_service = None
        services.spatial_service = Mock()
        services.lifecycle_service = None
        component = AgentComponent(services)
        
        assert component.logging_service is not None
        assert component.metrics_service is None
        assert component.validation_service is not None
        assert component.time_service is None
        assert component.spatial_service is not None
        assert component.lifecycle_service is None
    
    def test_attach_none_core(self):
        """Test attaching None as core."""
        services = Mock(spec=AgentServices)
        component = AgentComponent(services)
        
        component.attach(None)
        assert component.core is None
    
    def test_lifecycle_hooks_with_none_core(self):
        """Test lifecycle hooks work with None core."""
        services = Mock(spec=AgentServices)
        component = AgentComponent(services)
        component.core = None
        
        # These should not raise exceptions
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()


class TestIntegrationScenarios:
    """Test complex integration scenarios."""
    
    def test_complete_component_lifecycle(self):
        """Test a complete component lifecycle workflow."""
        # Setup services
        logging_service = Mock()
        time_service = Mock()
        time_service.current_time.return_value = 100
        services = AgentServices(
            spatial_service=Mock(),
            time_service=time_service,
            metrics_service=Mock(),
            logging_service=logging_service,
            validation_service=Mock(),
            lifecycle_service=Mock()
        )
        
        # Create component
        component = AgentComponent(services, "TestComponent")
        
        # Test initial state
        assert component.name == "TestComponent"
        assert component.core is None
        assert component.current_time == 100
        
        # Attach to core
        core = Mock()
        component.attach(core)
        assert component.core == core
        
        # Test service access
        assert component.logging_service == logging_service
        assert component.time_service == time_service
        
        # Test lifecycle hooks
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()
        
        # Test debug logging
        component._log_debug("Lifecycle test")
        logging_service.log_debug.assert_called_with("[TestComponent] Lifecycle test")
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        # Setup services with errors
        logging_service = Mock()
        logging_service.log_debug.side_effect = Exception("Logging error")
        
        services = AgentServices(
            spatial_service=Mock(),
            time_service=None,  # No time service, so get_current_time returns 0
            metrics_service=None,
            logging_service=logging_service,
            validation_service=None,
            lifecycle_service=None
        )
        
        component = AgentComponent(services)
        
        # Test that component continues to work despite service errors
        assert component.current_time == 0  # Fallback to 0
        component._log_debug("Error test")  # Should not raise exception
        
        # Test lifecycle hooks still work
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()
        
        # Test attachment still works
        core = Mock()
        component.attach(core)
        assert component.core == core
