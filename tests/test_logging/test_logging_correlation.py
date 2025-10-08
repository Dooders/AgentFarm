"""Comprehensive tests for farm.utils.logging.correlation module.

Tests for:
- Correlation ID generation and management
- Context manager for correlation IDs
- Correlation context binding
"""

import pytest
import uuid

from farm.utils.logging import configure_logging, get_logger
from farm.utils.logging.correlation import (
    add_correlation_id,
    get_correlation_id,
    clear_correlation_id,
    with_correlation_id,
    bind_correlation_context,
)
from farm.utils.logging.test_helpers import capture_logs, assert_log_contains


class TestAddCorrelationId:
    """Test add_correlation_id function."""
    
    def test_generate_correlation_id(self):
        """Test generating a new correlation ID."""
        configure_logging(environment="testing")
        
        corr_id = add_correlation_id()
        
        assert corr_id is not None
        assert isinstance(corr_id, str)
        assert len(corr_id) == 8  # Short UUID
    
    def test_custom_correlation_id(self):
        """Test using a custom correlation ID."""
        configure_logging(environment="testing")
        
        custom_id = "custom_operation_123"
        corr_id = add_correlation_id(custom_id)
        
        assert corr_id == custom_id
    
    def test_correlation_id_in_logs(self):
        """Test that correlation ID appears in logs."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            corr_id = add_correlation_id("test_op")
            logger.info("test_event")
            
            assert_log_contains(logs, "test_event", correlation_id=corr_id)
    
    def test_correlation_id_persists(self):
        """Test that correlation ID persists across multiple logs."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            corr_id = add_correlation_id()
            logger.info("event1")
            logger.info("event2")
            logger.info("event3")
            
            # All events should have the same correlation ID
            for entry in logs.entries:
                assert entry.get("correlation_id") == corr_id
    
    def test_multiple_correlation_ids(self):
        """Test overwriting correlation ID."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            corr_id1 = add_correlation_id("op1")
            logger.info("event1")
            
            corr_id2 = add_correlation_id("op2")
            logger.info("event2")
            
            assert logs.entries[0]["correlation_id"] == corr_id1
            assert logs.entries[1]["correlation_id"] == corr_id2


class TestGetCorrelationId:
    """Test get_correlation_id function."""
    
    def test_get_existing_correlation_id(self):
        """Test getting an existing correlation ID."""
        configure_logging(environment="testing")
        
        corr_id = add_correlation_id("test_op")
        retrieved_id = get_correlation_id()
        
        assert retrieved_id == corr_id
    
    def test_get_correlation_id_when_none_set(self):
        """Test getting correlation ID when none is set."""
        configure_logging(environment="testing")
        clear_correlation_id()
        
        corr_id = get_correlation_id()
        
        assert corr_id is None
    
    def test_get_correlation_id_after_clear(self):
        """Test getting correlation ID after clearing."""
        configure_logging(environment="testing")
        
        add_correlation_id("test_op")
        clear_correlation_id()
        corr_id = get_correlation_id()
        
        assert corr_id is None


class TestClearCorrelationId:
    """Test clear_correlation_id function."""
    
    def test_clear_correlation_id(self):
        """Test clearing correlation ID."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            add_correlation_id("test_op")
            logger.info("event_with_correlation")
            
            clear_correlation_id()
            logger.info("event_without_correlation")
            
            # First event should have correlation ID
            assert "correlation_id" in logs.entries[0]
            
            # Second event should not have correlation ID
            assert "correlation_id" not in logs.entries[1]
    
    def test_clear_when_none_set(self):
        """Test clearing when no correlation ID is set."""
        configure_logging(environment="testing")
        
        # Should not raise error
        clear_correlation_id()


class TestWithCorrelationId:
    """Test with_correlation_id context manager."""
    
    def test_context_manager_with_auto_id(self):
        """Test context manager with auto-generated ID."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            with with_correlation_id() as corr_id:
                logger.info("event_inside_context")
                
                assert corr_id is not None
                assert isinstance(corr_id, str)
            
            logger.info("event_outside_context")
            
            # Event inside context should have correlation ID
            assert logs.entries[0]["correlation_id"] == corr_id
            
            # Event outside context should not have correlation ID
            assert "correlation_id" not in logs.entries[1]
    
    def test_context_manager_with_custom_id(self):
        """Test context manager with custom ID."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            custom_id = "custom_batch_001"
            with with_correlation_id(custom_id) as corr_id:
                logger.info("event_inside_context")
                
                assert corr_id == custom_id
            
            assert_log_contains(logs, "event_inside_context", correlation_id=custom_id)
    
    def test_nested_context_managers(self):
        """Test nested correlation ID context managers."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            with with_correlation_id("outer") as outer_id:
                logger.info("outer_event")
                
                with with_correlation_id("inner") as inner_id:
                    logger.info("inner_event")
                
                logger.info("outer_event_again")
            
            logger.info("no_context_event")
            
            # Check correlation IDs
            assert logs.entries[0]["correlation_id"] == outer_id
            assert logs.entries[1]["correlation_id"] == inner_id
            assert logs.entries[2]["correlation_id"] == outer_id
            assert "correlation_id" not in logs.entries[3]
    
    def test_context_manager_cleanup_on_error(self):
        """Test that correlation ID is cleaned up even on error."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            with pytest.raises(ValueError):
                with with_correlation_id("error_op"):
                    logger.info("before_error")
                    raise ValueError("Test error")
            
            logger.info("after_error")
            
            # Event before error should have correlation ID
            assert "correlation_id" in logs.entries[0]
            
            # Event after error should not have correlation ID
            assert "correlation_id" not in logs.entries[1]
    
    def test_restore_original_correlation_id(self):
        """Test that original correlation ID is restored after context."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            # Set initial correlation ID
            original_id = add_correlation_id("original")
            logger.info("original_event")
            
            # Use temporary correlation ID
            with with_correlation_id("temporary"):
                logger.info("temporary_event")
            
            # Original should be restored
            logger.info("restored_event")
            
            assert logs.entries[0]["correlation_id"] == original_id
            assert logs.entries[1]["correlation_id"] == "temporary"
            assert logs.entries[2]["correlation_id"] == original_id


class TestBindCorrelationContext:
    """Test bind_correlation_context function."""
    
    def test_bind_correlation_with_other_context(self):
        """Test binding correlation ID with other context variables."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            bind_correlation_context(
                correlation_id="batch_001",
                experiment_id="exp_001",
                batch_size=100
            )
            
            logger.info("test_event")
            
            assert_log_contains(
                logs,
                "test_event",
                correlation_id="batch_001",
                experiment_id="exp_001",
                batch_size=100
            )
    
    def test_bind_without_correlation_id(self):
        """Test binding context without correlation ID."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            bind_correlation_context(
                experiment_id="exp_001",
                iteration=5
            )
            
            logger.info("test_event")
            
            assert_log_contains(
                logs,
                "test_event",
                experiment_id="exp_001",
                iteration=5
            )


class TestCorrelationIdFormats:
    """Test different correlation ID formats."""
    
    def test_short_uuid_format(self):
        """Test that auto-generated IDs are short UUIDs."""
        configure_logging(environment="testing")
        
        corr_id = add_correlation_id()
        
        # Should be 8 characters (first segment of UUID)
        assert len(corr_id) == 8
        
        # Should be valid hex
        try:
            int(corr_id, 16)
        except ValueError:
            pytest.fail("Correlation ID is not valid hex")
    
    def test_custom_format_allowed(self):
        """Test that custom formats are allowed."""
        configure_logging(environment="testing")
        
        custom_formats = [
            "simple-id",
            "operation_123",
            "batch-2023-01-01-001",
            "exp_001_run_005",
        ]
        
        for custom_id in custom_formats:
            corr_id = add_correlation_id(custom_id)
            assert corr_id == custom_id


class TestCorrelationIdScenarios:
    """Test real-world correlation ID usage scenarios."""
    
    def test_batch_processing_scenario(self):
        """Test correlation IDs in batch processing scenario."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            # Process multiple batches
            for batch_num in range(3):
                with with_correlation_id(f"batch_{batch_num:03d}") as batch_id:
                    logger.info("batch_started", batch_num=batch_num)
                    
                    # Process items in batch
                    for item_num in range(2):
                        logger.info("item_processed", item_num=item_num)
                    
                    logger.info("batch_completed", batch_num=batch_num)
            
            # Verify all logs in each batch have the same correlation ID
            batch_0_logs = [e for e in logs.entries if e.get("correlation_id") == "batch_000"]
            batch_1_logs = [e for e in logs.entries if e.get("correlation_id") == "batch_001"]
            batch_2_logs = [e for e in logs.entries if e.get("correlation_id") == "batch_002"]
            
            assert len(batch_0_logs) == 4  # batch_started + 2 items + batch_completed
            assert len(batch_1_logs) == 4
            assert len(batch_2_logs) == 4
    
    def test_distributed_operation_scenario(self):
        """Test correlation IDs for distributed operations."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            # Main operation with correlation ID
            operation_id = add_correlation_id("distributed_op_001")
            logger.info("operation_started")
            
            # Simulated sub-operations (would be in different processes/threads)
            logger.info("sub_operation_1", worker_id="worker_1")
            logger.info("sub_operation_2", worker_id="worker_2")
            logger.info("sub_operation_3", worker_id="worker_3")
            
            logger.info("operation_completed")
            
            # All logs should have the same correlation ID
            for entry in logs.entries:
                assert entry.get("correlation_id") == operation_id
    
    def test_experiment_run_scenario(self):
        """Test correlation IDs in experiment scenario."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            experiment_id = "exp_001"
            
            # Multiple runs of the same experiment
            for run_num in range(3):
                run_id = f"{experiment_id}_run_{run_num:03d}"
                with with_correlation_id(run_id):
                    logger.info("run_started", run_num=run_num)
                    logger.info("simulation_step", step=1)
                    logger.info("simulation_step", step=2)
                    logger.info("run_completed", run_num=run_num)
            
            # Each run should have its own correlation ID
            run_0_logs = [e for e in logs.entries if e.get("correlation_id") == f"{experiment_id}_run_000"]
            run_1_logs = [e for e in logs.entries if e.get("correlation_id") == f"{experiment_id}_run_001"]
            run_2_logs = [e for e in logs.entries if e.get("correlation_id") == f"{experiment_id}_run_002"]
            
            assert len(run_0_logs) == 4
            assert len(run_1_logs) == 4
            assert len(run_2_logs) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

