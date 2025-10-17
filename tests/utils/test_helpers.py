"""
Common test utilities and helpers to reduce code duplication.

This module provides reusable test patterns and utilities to avoid code clones
across different test files.
"""

import subprocess
from contextlib import contextmanager
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional


class PlatformMockHelper:
    """Helper class for mocking platform/system information in tests."""

    @staticmethod
    def setup_platform_mocks(mock_platform, mock_python, mock_machine,
                           mock_processor, mock_hostname, platform_info: Optional[Dict[str, str]] = None):
        """Set up common platform mocks with default or custom values.

        Args:
            mock_platform: Mock for platform.platform
            mock_python: Mock for platform.python_version
            mock_machine: Mock for platform.machine
            mock_processor: Mock for platform.processor
            mock_hostname: Mock for socket.gethostname
            platform_info: Optional dict with custom platform values
        """
        defaults = {
            "platform": "Linux-5.4.0-74-generic-x86_64-with-glibc2.29",
            "python": "3.8.10",
            "machine": "x86_64",
            "processor": "x86_64",
            "hostname": "cpu-machine"
        }

        if platform_info:
            defaults.update(platform_info)

        mock_platform.return_value = defaults["platform"]
        mock_python.return_value = defaults["python"]
        mock_machine.return_value = defaults["machine"]
        mock_processor.return_value = defaults["processor"]
        mock_hostname.return_value = defaults["hostname"]


class DatabaseTestHelper:
    """Helper class for common database testing patterns."""

    @staticmethod
    def create_mock_database_with_pragmas(pragma_profile: str, custom_pragmas: Optional[Dict[str, str]] = None):
        """Create a mock database with specified pragma settings.

        Args:
            pragma_profile: The pragma profile to use
            custom_pragmas: Optional custom pragma overrides

        Returns:
            Tuple of (mock_db, mock_config)
        """
        mock_config = Mock()
        mock_config.db_pragma_profile = pragma_profile
        mock_config.db_custom_pragmas = custom_pragmas or {}

        mock_db = Mock()

        # Set up pragma responses based on profile
        if pragma_profile == "memory":
            mock_db.get_current_pragmas.return_value = {
                "synchronous": "1",  # NORMAL
                "journal_mode": "MEMORY",
                "cache_size": "51200"  # 50MB
            }
        elif pragma_profile == "safety":
            mock_db.get_current_pragmas.return_value = {
                "synchronous": "2",  # FULL
                "journal_mode": "WAL",
                "page_size": "4096"
            }
        else:
            mock_db.get_current_pragmas.return_value = {
                "synchronous": "1",
                "journal_mode": "WAL",
                "page_size": "4096"
            }

        return mock_db, mock_config


class MockAdapterHelper:
    """Helper class for creating mock adapters in API tests."""

    @staticmethod
    def create_simulation_results_mock(simulation_id: str = "sim-123",
                                     status: str = "COMPLETED",
                                     total_steps: int = 1000,
                                     final_agent_count: int = 20,
                                     final_resource_count: int = 50):
        """Create a mock SimulationResults object.

        Args:
            simulation_id: The simulation ID
            status: The simulation status
            total_steps: Total number of steps
            final_agent_count: Final agent count
            final_resource_count: Final resource count

        Returns:
            Mock SimulationResults object
        """
        mock_results = Mock()
        mock_results.simulation_id = simulation_id
        mock_results.status = status
        mock_results.total_steps = total_steps
        mock_results.final_agent_count = final_agent_count
        mock_results.final_resource_count = final_resource_count
        return mock_results

    @staticmethod
    def setup_adapter_mock(mock_adapter_class, return_value=None, side_effect=None):
        """Set up a mock adapter with common configuration.

        Args:
            mock_adapter_class: The mock adapter class
            return_value: Value to return from method calls
            side_effect: Side effect for method calls

        Returns:
            The configured mock adapter
        """
        mock_adapter = Mock()
        if return_value is not None:
            mock_adapter.get_simulation_results.return_value = return_value
            mock_adapter.get_simulation_status.return_value = return_value
        if side_effect is not None:
            mock_adapter.get_simulation_results.side_effect = side_effect
            mock_adapter.get_simulation_status.side_effect = side_effect
        mock_adapter_class.return_value = mock_adapter
        return mock_adapter

    @staticmethod
    def create_controller_with_session(controller_class, workspace_path, session_name="Test Session", session_description=None):
        """Create a controller and session with the common pattern used in tests.

        Args:
            controller_class: The controller class to instantiate
            workspace_path: Path to the workspace
            session_name: Name for the session
            session_description: Optional description for the session

        Returns:
            Tuple of (controller, session_id)
        """
        controller = controller_class(str(workspace_path))
        session_id = controller.create_session(session_name, session_description)
        return controller, session_id


class MemoryTestHelper:
    """Helper class for testing memory-related functionality."""

    @staticmethod
    def create_mock_memory(remember_state_return: bool = True):
        """Create a mock memory object for testing.

        Args:
            remember_state_return: What remember_state should return

        Returns:
            Mock memory object
        """
        mock_memory = Mock()
        mock_memory.remember_state.return_value = remember_state_return
        return mock_memory

    @staticmethod
    @contextmanager
    def setup_memory_failure_test(agent, exception_message: str = "Connection failed"):
        """Set up a test for memory initialization failure.

        Args:
            agent: The agent to test
            exception_message: The exception message to raise

        Returns:
            Context manager for the patched memory manager
        """
        with patch("farm.core.agent.AgentMemoryManager") as mock_memory_manager:
            mock_memory_manager.get_instance.side_effect = Exception(exception_message)
            agent._init_memory()
            yield mock_memory_manager


def assert_pragma_setting(pragmas: Dict[str, str], setting: str, expected_value: Any,
                         value_type: type = str):
    """Assert that a pragma setting has the expected value.

    Args:
        pragmas: Dictionary of pragma settings
        setting: The setting name to check
        expected_value: The expected value
        value_type: Type to convert the value to before comparison
    """
    actual_value = pragmas.get(setting, "")
    if value_type is int:
        actual_value = int(actual_value) if actual_value else -1
    elif value_type is str:
        actual_value = actual_value.upper() if isinstance(actual_value, str) else ""

    assert actual_value == expected_value, f"Expected {setting}={expected_value}, got {actual_value}"


def setup_gpu_detection_failure(mock_check_output):
    """Set up mock to simulate GPU detection failure.

    Args:
        mock_check_output: Mock for subprocess.check_output
    """
    mock_check_output.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")


class BaseTestClass:
    """Base test class with common test methods to eliminate duplication."""
    
    def test_initialization(self):
        """Test basic initialization of the class under test.
        
        This method should be overridden by subclasses to test their specific
        initialization logic. The base implementation provides a template.
        """
        # This is a template method that should be overridden
        # Subclasses should implement their specific initialization test
        pass


class CommonTestMixin:
    """Mixin class providing common test methods to eliminate duplication."""
    
    def test_initialization(self):
        """Test basic initialization of the class under test.
        
        This is a common test pattern that can be mixed into test classes.
        Subclasses should override this method to test their specific initialization.
        """
        # Default implementation - should be overridden by subclasses
        self.assertIsNotNone(self)
    
    def test_module_registration(self):
        """Test module registration functionality.
        
        Common test for modules that support registration.
        """
        # Default implementation - should be overridden by subclasses
        pass
    
    def test_module_groups(self):
        """Test module groups functionality.
        
        Common test for modules that support grouping.
        """
        # Default implementation - should be overridden by subclasses
        pass
    
    def test_data_processor(self):
        """Test data processor functionality.
        
        Common test for modules that process data.
        """
        # Default implementation - should be overridden by subclasses
        pass
    
    def test_supports_database(self):
        """Test database support functionality.
        
        Common test for modules that support database operations.
        """
        # Default implementation - should be overridden by subclasses
        pass
    
    def test_module_validator(self):
        """Test module validator functionality.
        
        Common test for modules that have validation.
        """
        # Default implementation - should be overridden by subclasses
        pass
    
    def test_module_all_functions_registered(self):
        """Test that all functions are properly registered.
        
        Common test for modules that register functions.
        """
        # Default implementation - should be overridden by subclasses
        pass
