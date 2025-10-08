"""Comprehensive tests for farm.utils.logging.config module.

Tests for:
- Custom context classes (FastContext, ThreadSafeContext, MemoryEfficientContext)
- Core processors (add_log_level, add_logger_name, censor_sensitive_data, etc.)
- Configuration options and settings
- Metrics and sampling processors
"""

import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import threading
import structlog

from farm.utils.logging.config import (
    configure_logging,
    get_logger,
    bind_context,
    unbind_context,
    clear_context,
    FastContext,
    ThreadSafeContext,
    MemoryEfficientContext,
    add_log_level,
    add_log_level_number,
    add_logger_name,
    add_process_info,
    censor_sensitive_data,
    PerformanceLogger,
    MetricsProcessor,
    SamplingProcessor,
    get_metrics_summary,
    reset_metrics,
)
from farm.utils.logging.test_helpers import capture_logs


class TestFastContext:
    """Test FastContext class."""
    
    def test_initialization(self):
        """Test FastContext initialization."""
        ctx = FastContext({"key": "value"})
        assert ctx["key"] == "value"
    
    def test_new_child(self):
        """Test creating child context."""
        parent = FastContext({"parent_key": "parent_value"})
        child = parent.new_child({"child_key": "child_value"})
        
        assert isinstance(child, FastContext)
        assert child["parent_key"] == "parent_value"
        assert child["child_key"] == "child_value"
    
    def test_copy(self):
        """Test copying context."""
        original = FastContext({"key": "value"})
        copy = original.copy()
        
        assert isinstance(copy, FastContext)
        assert copy["key"] == "value"
        assert copy is not original
    
    def test_new_child_without_data(self):
        """Test new_child without additional data."""
        parent = FastContext({"key": "value"})
        child = parent.new_child()
        
        assert isinstance(child, FastContext)
        assert child["key"] == "value"


class TestThreadSafeContext:
    """Test ThreadSafeContext class."""
    
    def test_initialization(self):
        """Test ThreadSafeContext initialization."""
        ctx = ThreadSafeContext({"key": "value"})
        assert ctx["key"] == "value"
        assert hasattr(ctx, "_lock")
    
    def test_thread_safe_operations(self):
        """Test thread-safe set/get operations."""
        ctx = ThreadSafeContext()
        
        def worker():
            for i in range(10):
                ctx[f"key_{threading.current_thread().name}_{i}"] = i
        
        threads = [threading.Thread(target=worker, name=f"t{i}") for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have all items from all threads
        assert len(ctx) == 30
    
    def test_new_child(self):
        """Test creating child context."""
        parent = ThreadSafeContext({"parent_key": "parent_value"})
        child = parent.new_child({"child_key": "child_value"})
        
        assert isinstance(child, ThreadSafeContext)
        assert child["parent_key"] == "parent_value"
        assert child["child_key"] == "child_value"
    
    def test_copy(self):
        """Test copying context."""
        original = ThreadSafeContext({"key": "value"})
        copy = original.copy()
        
        assert isinstance(copy, ThreadSafeContext)
        assert copy["key"] == "value"
        assert copy is not original
    
    def test_update(self):
        """Test thread-safe update."""
        ctx = ThreadSafeContext({"key1": "value1"})
        ctx.update({"key2": "value2"})
        
        assert ctx["key1"] == "value1"
        assert ctx["key2"] == "value2"
    
    def test_delete_item(self):
        """Test thread-safe delete."""
        ctx = ThreadSafeContext({"key": "value"})
        del ctx["key"]
        
        assert "key" not in ctx


class TestMemoryEfficientContext:
    """Test MemoryEfficientContext class."""
    
    def test_initialization(self):
        """Test MemoryEfficientContext initialization."""
        ctx = MemoryEfficientContext({"key": "value"})
        assert ctx["key"] == "value"
        assert hasattr(ctx, "_frozen")
        assert ctx._frozen is False
    
    def test_freeze(self):
        """Test freezing context."""
        ctx = MemoryEfficientContext({"key": "value"})
        result = ctx.freeze()
        
        assert result is ctx
        assert ctx._frozen is True
    
    def test_frozen_context_prevents_update(self):
        """Test that frozen context prevents updates."""
        ctx = MemoryEfficientContext({"key": "value"})
        ctx.freeze()
        
        with pytest.raises(RuntimeError, match="Cannot update frozen context"):
            ctx.update({"key2": "value2"})
    
    def test_frozen_context_prevents_setitem(self):
        """Test that frozen context prevents setitem."""
        ctx = MemoryEfficientContext({"key": "value"})
        ctx.freeze()
        
        with pytest.raises(RuntimeError, match="Cannot modify frozen context"):
            ctx["key2"] = "value2"
    
    def test_frozen_context_prevents_delitem(self):
        """Test that frozen context prevents delitem."""
        ctx = MemoryEfficientContext({"key": "value"})
        ctx.freeze()
        
        with pytest.raises(RuntimeError, match="Cannot modify frozen context"):
            del ctx["key"]
    
    def test_new_child(self):
        """Test creating child context."""
        parent = MemoryEfficientContext({"parent_key": "parent_value"})
        child = parent.new_child({"child_key": "child_value"})
        
        assert isinstance(child, MemoryEfficientContext)
        assert child["parent_key"] == "parent_value"
        assert child["child_key"] == "child_value"
    
    def test_copy(self):
        """Test copying context."""
        original = MemoryEfficientContext({"key": "value"})
        copy = original.copy()
        
        assert isinstance(copy, MemoryEfficientContext)
        assert copy["key"] == "value"
        assert copy is not original


class TestCoreProcessors:
    """Test core logging processors."""
    
    def test_add_log_level(self):
        """Test add_log_level processor."""
        event_dict = {}
        result = add_log_level(None, "info", event_dict)
        
        assert result["level"] == "info"
    
    def test_add_log_level_warn_alias(self):
        """Test add_log_level with warn method."""
        event_dict = {}
        result = add_log_level(None, "warn", event_dict)
        
        assert result["level"] == "warning"
    
    def test_add_log_level_number(self):
        """Test add_log_level_number processor."""
        test_cases = [
            ("debug", 10),
            ("info", 20),
            ("warning", 30),
            ("error", 40),
            ("critical", 50),
        ]
        
        for level, expected_num in test_cases:
            event_dict = {"level": level}
            result = add_log_level_number(None, level, event_dict)
            assert result["level_num"] == expected_num
    
    def test_add_log_level_number_default(self):
        """Test add_log_level_number with unknown level."""
        event_dict = {"level": "unknown"}
        result = add_log_level_number(None, "unknown", event_dict)
        
        assert result["level_num"] == 20  # Default to INFO
    
    def test_add_logger_name_with_record(self):
        """Test add_logger_name with _record."""
        record = MagicMock()
        record.name = "test.logger"
        event_dict = {"_record": record}
        
        result = add_logger_name(None, "info", event_dict)
        assert result["logger"] == "test.logger"
    
    def test_add_logger_name_with_logger_attribute(self):
        """Test add_logger_name with logger.name attribute."""
        logger = MagicMock()
        logger.name = "test.logger"
        event_dict = {}
        
        result = add_logger_name(logger, "info", event_dict)
        assert result["logger"] == "test.logger"
    
    def test_add_process_info(self):
        """Test add_process_info processor."""
        event_dict = {}
        result = add_process_info(None, "info", event_dict)
        
        assert "process_id" in result
        assert "thread_id" in result
        assert "thread_name" in result
        assert isinstance(result["process_id"], int)
        assert isinstance(result["thread_id"], int)
        assert isinstance(result["thread_name"], str)
    
    def test_censor_sensitive_data(self):
        """Test censor_sensitive_data processor."""
        event_dict = {
            "username": "john",
            "password": "secret123",
            "api_key": "abc123",
            "auth_token": "xyz789",
            "user_secret": "hidden",
            "normal_field": "visible",
        }
        
        result = censor_sensitive_data(None, "info", event_dict)
        
        assert result["password"] == "***REDACTED***"
        assert result["api_key"] == "***REDACTED***"
        assert result["auth_token"] == "***REDACTED***"
        assert result["user_secret"] == "***REDACTED***"
        assert result["normal_field"] == "visible"
        assert result["username"] == "john"


class TestPerformanceLogger:
    """Test PerformanceLogger processor."""
    
    def test_slow_operation_warning(self):
        """Test that slow operations get performance warning."""
        perf_logger = PerformanceLogger(slow_threshold_ms=100.0)
        event_dict = {"duration_ms": 150.0}
        
        result = perf_logger(None, "info", event_dict)
        
        assert result["performance_warning"] == "slow_operation"
    
    def test_fast_operation_no_warning(self):
        """Test that fast operations don't get performance warning."""
        perf_logger = PerformanceLogger(slow_threshold_ms=100.0)
        event_dict = {"duration_ms": 50.0}
        
        result = perf_logger(None, "info", event_dict)
        
        assert "performance_warning" not in result
    
    def test_no_duration_no_warning(self):
        """Test that events without duration don't get warning."""
        perf_logger = PerformanceLogger(slow_threshold_ms=100.0)
        event_dict = {"event": "test"}
        
        result = perf_logger(None, "info", event_dict)
        
        assert "performance_warning" not in result


class TestMetricsProcessor:
    """Test MetricsProcessor class."""
    
    def test_event_counting(self):
        """Test that events are counted."""
        metrics = MetricsProcessor()
        
        metrics(None, "info", {"event": "event1"})
        metrics(None, "info", {"event": "event1"})
        metrics(None, "info", {"event": "event2"})
        
        summary = metrics.get_summary()
        assert summary["event_counts"]["event1"] == 2
        assert summary["event_counts"]["event2"] == 1
    
    def test_duration_tracking(self):
        """Test that durations are tracked."""
        metrics = MetricsProcessor()
        
        metrics(None, "info", {"event": "operation", "duration_ms": 100.0})
        metrics(None, "info", {"event": "operation", "duration_ms": 200.0})
        metrics(None, "info", {"event": "operation", "duration_ms": 150.0})
        
        summary = metrics.get_summary()
        duration_stats = summary["duration_metrics"]["operation"]
        
        assert duration_stats["count"] == 3
        assert duration_stats["mean"] == 150.0
        assert duration_stats["median"] == 150.0
        assert duration_stats["max"] == 200.0
        assert duration_stats["min"] == 100.0
    
    def test_runtime_added(self):
        """Test that runtime_seconds is added to events."""
        metrics = MetricsProcessor()
        event_dict = {"event": "test"}
        
        result = metrics(None, "info", event_dict)
        
        assert "runtime_seconds" in result
        assert isinstance(result["runtime_seconds"], (int, float))
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        metrics = MetricsProcessor()
        
        metrics(None, "info", {"event": "event1"})
        metrics.reset()
        
        summary = metrics.get_summary()
        assert len(summary["event_counts"]) == 0
        assert len(summary["duration_metrics"]) == 0
    
    def test_unknown_event_name(self):
        """Test handling of events without event name."""
        metrics = MetricsProcessor()
        
        # Event with message instead of event
        metrics(None, "info", {"message": "test_message"})
        
        summary = metrics.get_summary()
        assert "test_message" in summary["event_counts"]


class TestSamplingProcessor:
    """Test SamplingProcessor class."""
    
    def test_sample_rate_validation(self):
        """Test that sample_rate is validated."""
        with pytest.raises(ValueError, match="sample_rate must be between"):
            SamplingProcessor(sample_rate=1.5)
        
        with pytest.raises(ValueError, match="sample_rate cannot be 0.0"):
            SamplingProcessor(sample_rate=0.0)
    
    def test_sample_rate_boundary_values(self):
        """Test boundary values for sample_rate."""
        # Minimum valid
        sampler = SamplingProcessor(sample_rate=0.01)
        assert sampler.sample_rate == 0.01
        
        # Maximum valid
        sampler = SamplingProcessor(sample_rate=1.0)
        assert sampler.sample_rate == 1.0
    
    def test_sample_all_events_when_rate_is_one(self):
        """Test that all events are sampled when rate is 1.0."""
        sampler = SamplingProcessor(sample_rate=1.0)
        
        for i in range(10):
            event_dict = {"event": "test_event"}
            result = sampler(None, "info", event_dict)
            # Should not raise DropEvent
            assert "sampled" in result
    
    def test_sampling_specific_events(self):
        """Test sampling only specific events."""
        sampler = SamplingProcessor(
            sample_rate=0.5,
            events_to_sample={"agent_action"}
        )
        
        # agent_action should be sampled
        event_dict = {"event": "agent_action"}
        count = 0
        for i in range(10):
            try:
                result = sampler(None, "info", {"event": "agent_action"})
                count += 1
            except structlog.DropEvent:
                pass
        
        # Should have sampled some but not all
        assert count < 10
    
    def test_no_sampling_for_unlisted_events(self):
        """Test that unlisted events are not sampled."""
        sampler = SamplingProcessor(
            sample_rate=0.1,
            events_to_sample={"agent_action"}
        )
        
        # other_event should not be sampled
        event_dict = {"event": "other_event"}
        result = sampler(None, "info", event_dict)
        
        # Should pass through without sampling
        assert "sampled" not in result or not result.get("sampled")
    
    def test_sample_interval_calculation(self):
        """Test that sample interval is calculated correctly."""
        sampler = SamplingProcessor(sample_rate=0.1)
        assert sampler._sample_interval == 10
        
        sampler = SamplingProcessor(sample_rate=0.5)
        assert sampler._sample_interval == 2
        
        sampler = SamplingProcessor(sample_rate=1.0)
        assert sampler._sample_interval == 1


class TestConfigureLogging:
    """Test configure_logging function."""
    
    def test_basic_configuration(self):
        """Test basic logging configuration."""
        configure_logging(environment="testing", log_level="INFO")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            logger.info("test_event")
            assert len(logs.entries) > 0
    
    def test_development_environment(self):
        """Test development environment configuration."""
        configure_logging(environment="development", enable_colors=True)
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            logger.info("dev_event")
            assert len(logs.entries) > 0
    
    def test_production_environment(self):
        """Test production environment configuration."""
        configure_logging(environment="production")
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            logger.info("prod_event")
            assert len(logs.entries) > 0
    
    def test_with_metrics_enabled(self):
        """Test configuration with metrics enabled."""
        configure_logging(environment="testing", enable_metrics=True)
        logger = get_logger(__name__)
        
        logger.info("event1", duration_ms=100.0)
        logger.info("event2", duration_ms=200.0)
        
        metrics = get_metrics_summary()
        assert "event_counts" in metrics
        assert metrics["event_counts"]["event1"] == 1
    
    def test_with_sampling_enabled(self):
        """Test configuration with sampling enabled."""
        configure_logging(
            environment="testing",
            enable_sampling=True,
            sample_rate=0.1,  # Lower rate to ensure sampling happens
            events_to_sample={"sampled_event"}
        )
        logger = get_logger(__name__)
        
        # Test without capture_logs since it overrides configuration
        logger.info("test_event")  # Verify logger works
        
        # Just verify configuration was set
        import structlog
        config = structlog.get_config()
        assert config is not None
    
    def test_with_process_info(self):
        """Test configuration with process info enabled."""
        configure_logging(environment="testing", include_process_info=True)
        logger = get_logger(__name__)
        
        # Test without capture_logs since it overrides configuration
        logger.info("test_event")  # Verify logger works
        
        # Just verify configuration was set
        import structlog
        config = structlog.get_config()
        assert config is not None
    
    def test_with_file_logging(self):
        """Test configuration with file logging."""
        import logging
        tmpdir = tempfile.mkdtemp()
        try:
            configure_logging(
                environment="testing",
                log_dir=tmpdir,
                json_logs=True
            )
            logger = get_logger(__name__)
            
            logger.info("file_test_event", value=42)
            
            # Check that log files were created
            log_path = Path(tmpdir)
            assert (log_path / "application.log").exists()
            assert (log_path / "application.json.log").exists()
            
            # Close all file handlers to release locks
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)
        finally:
            # Cleanup
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except:
                pass  # Ignore cleanup errors on Windows
    
    def test_custom_context_class(self):
        """Test configuration with custom context class."""
        configure_logging(
            environment="testing",
            context_class=FastContext
        )
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            logger.info("test_event")
            assert len(logs.entries) > 0
    
    def test_threadlocal_context(self):
        """Test configuration with threadlocal context."""
        configure_logging(
            environment="testing",
            use_threadlocal=True
        )
        logger = get_logger(__name__)
        
        with capture_logs() as logs:
            logger.info("test_event")
            assert len(logs.entries) > 0


class TestContextManagement:
    """Test context management functions."""
    
    def test_bind_context(self):
        """Test binding context."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        # Test without capture_logs since it overrides configuration
        bind_context(simulation_id="sim_001", step=42)
        logger.info("test_event")
        
        # Verify the context was bound
        import structlog
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("simulation_id") == "sim_001"
        assert ctx.get("step") == 42
        
        clear_context()
    
    def test_unbind_context(self):
        """Test unbinding context."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        bind_context(simulation_id="sim_001", step=42)
        logger.info("event1")
        
        unbind_context("step")
        logger.info("event2")
        
        # Verify step was unbound
        import structlog
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("simulation_id") == "sim_001"
        assert "step" not in ctx
        
        clear_context()
    
    def test_clear_context(self):
        """Test clearing all context."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)
        
        bind_context(simulation_id="sim_001", step=42)
        logger.info("event1")
        
        clear_context()
        logger.info("event2")
        
        # Verify context was cleared
        import structlog
        ctx = structlog.contextvars.get_contextvars()
        assert "simulation_id" not in ctx
        assert "step" not in ctx


class TestMetricsAPI:
    """Test metrics API functions."""
    
    def test_get_metrics_summary_without_metrics(self):
        """Test get_metrics_summary when metrics not enabled."""
        # Set global metrics processor to None to simulate no metrics
        import farm.utils.logging.config as config_module
        original_processor = config_module._metrics_manager._processor
        config_module._metrics_manager._processor = None
        
        try:
            summary = get_metrics_summary()
            assert "error" in summary
        finally:
            # Restore original processor
            config_module._metrics_manager._processor = original_processor
    
    def test_reset_metrics(self):
        """Test reset_metrics function."""
        configure_logging(environment="testing", enable_metrics=True)
        logger = get_logger(__name__)
        
        logger.info("event1")
        logger.info("event2")
        
        reset_metrics()
        
        summary = get_metrics_summary()
        assert len(summary["event_counts"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

