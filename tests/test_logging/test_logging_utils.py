"""Comprehensive tests for farm.utils.logging.utils module.

Tests for:
- Decorators (log_performance, log_errors)
- Context managers (log_context, log_step, log_simulation, log_experiment)
- Specialized loggers (AgentLogger, DatabaseLogger, PerformanceMonitor, LogSampler)
"""

import time
from unittest.mock import patch

import pytest

from farm.utils.logging import configure_logging, get_logger
from farm.utils.logging.test_helpers import assert_log_contains, capture_logs
from farm.utils.logging.utils import (
    AgentLogger,
    DatabaseLogger,
    LogSampler,
    PerformanceMonitor,
    log_context,
    log_errors,
    log_experiment,
    log_performance,
    log_simulation,
    log_step,
)


class TestLogPerformanceDecorator:
    """Test log_performance decorator."""

    def test_basic_performance_logging(self):
        """Test basic performance logging."""
        configure_logging(environment="testing")

        @log_performance()
        def test_function():
            time.sleep(0.01)
            return "result"

        with capture_logs() as logs:
            result = test_function()

            assert result == "result"
            assert len(logs.entries) > 0

            # Find operation_complete log
            perf_logs = [e for e in logs.entries if "operation" in e.get("event", "")]
            assert len(perf_logs) > 0

            perf_log = perf_logs[0]
            assert "duration_ms" in perf_log
            assert perf_log["operation"] == "test_function"
            assert perf_log["status"] == "success"

    def test_custom_operation_name(self):
        """Test performance logging with custom operation name."""
        configure_logging(environment="testing")

        @log_performance(operation_name="custom_operation")
        def test_function():
            return "result"

        with capture_logs() as logs:
            test_function()

            perf_logs = [e for e in logs.entries if "operation" in e.get("event", "")]
            assert any(e.get("operation") == "custom_operation" for e in perf_logs)

    def test_log_args(self):
        """Test logging function arguments."""
        configure_logging(environment="testing")

        @log_performance(log_args=True)
        def test_function(arg1, arg2, kwarg1=None):
            return "result"

        with capture_logs() as logs:
            test_function("value1", "value2", kwarg1="kwvalue")

            perf_logs = [e for e in logs.entries if "operation" in e.get("event", "")]
            assert len(perf_logs) > 0

            perf_log = perf_logs[0]
            assert "args" in perf_log
            assert "kwargs" in perf_log

    def test_log_result(self):
        """Test logging function result."""
        configure_logging(environment="testing")

        @log_performance(log_result=True)
        def test_function():
            return "test_result"

        with capture_logs() as logs:
            test_function()

            perf_logs = [e for e in logs.entries if "operation" in e.get("event", "")]
            perf_log = perf_logs[0]
            assert "result" in perf_log

    def test_slow_operation_warning(self):
        """Test slow operation warning."""
        configure_logging(environment="testing")

        @log_performance(slow_threshold_ms=10.0)
        def slow_function():
            # Simulate work without actual sleep to avoid flaky tests
            return "result"

        with capture_logs() as logs:
            # Mock time.perf_counter to simulate a slow operation (20ms)
            with patch("time.perf_counter", side_effect=[0.0, 0.02]):
                slow_function()

            # Should log warning for slow operation
            slow_logs = [e for e in logs.entries if e.get("event") == "operation_slow"]
            assert len(slow_logs) > 0

    def test_error_handling(self):
        """Test error handling in performance decorator."""
        configure_logging(environment="testing")

        @log_performance()
        def failing_function():
            raise ValueError("Test error")

        with capture_logs() as logs:
            with pytest.raises(ValueError, match="Test error"):
                failing_function()

            # Should log operation_failed
            error_logs = [
                e for e in logs.entries if e.get("event") == "operation_failed"
            ]
            assert len(error_logs) > 0

            error_log = error_logs[0]
            assert error_log["status"] == "error"
            assert error_log["error_type"] == "ValueError"
            assert "duration_ms" in error_log


class TestLogErrorsDecorator:
    """Test log_errors decorator."""

    def test_successful_execution(self):
        """Test that successful execution doesn't log errors."""
        configure_logging(environment="testing")

        @log_errors()
        def success_function():
            return "success"

        with capture_logs() as logs:
            result = success_function()

            assert result == "success"
            # Should not log any errors
            error_logs = [e for e in logs.entries if e.get("level") == "error"]
            assert len(error_logs) == 0

    def test_error_logging(self):
        """Test that errors are logged."""
        configure_logging(environment="testing")

        @log_errors()
        def failing_function():
            raise RuntimeError("Test error")

        with capture_logs() as logs:
            with pytest.raises(RuntimeError, match="Test error"):
                failing_function()

            # Should log unhandled_exception
            error_logs = [
                e for e in logs.entries if e.get("event") == "unhandled_exception"
            ]
            assert len(error_logs) == 1

            error_log = error_logs[0]
            assert error_log["function"] == "failing_function"
            assert error_log["error_type"] == "RuntimeError"
            assert "Test error" in error_log["error_message"]

    def test_custom_logger_name(self):
        """Test with custom logger name."""
        configure_logging(environment="testing")

        @log_errors(logger_name="custom.logger")
        def failing_function():
            raise ValueError("Test")

        with capture_logs() as logs:
            with pytest.raises(ValueError):
                failing_function()

            error_logs = [
                e for e in logs.entries if e.get("event") == "unhandled_exception"
            ]
            assert len(error_logs) == 1


class TestLogContextManager:
    """Test log_context context manager."""

    def test_context_binding(self):
        """Test that context is bound within context manager."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            with log_context(simulation_id="sim_001", step=42):
                logger.info("event_inside_context")

            logger.info("event_outside_context")

            # Event inside context should have bound variables
            assert logs.entries[0]["simulation_id"] == "sim_001"
            assert logs.entries[0]["step"] == 42

            # Event outside context should not have bound variables
            assert "simulation_id" not in logs.entries[1]
            assert "step" not in logs.entries[1]

    def test_context_cleanup_on_exception(self):
        """Test that context is cleaned up even on exception."""
        configure_logging(environment="testing")
        logger = get_logger(__name__)

        with capture_logs() as logs:
            try:
                with log_context(temp_id="temp_001"):
                    logger.info("event_before_error")
                    raise ValueError("Test error")
            except ValueError:
                pass

            logger.info("event_after_exception")

            # Event before error should have context
            assert logs.entries[0]["temp_id"] == "temp_001"

            # Event after exception should not have context
            assert "temp_id" not in logs.entries[1]


class TestLogStepManager:
    """Test log_step context manager."""

    def test_step_logging(self):
        """Test step start and completion logging."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with log_step(step_number=42):
                pass

            # Should have step_started and step_completed
            assert_log_contains(logs, "step_started", step=42)
            assert_log_contains(logs, "step_completed", step=42)

    def test_step_with_extra_context(self):
        """Test step logging with extra context."""
        configure_logging(environment="testing")
        logger = get_logger("farm.simulation")

        with capture_logs() as logs:
            with log_step(step_number=42, simulation_id="sim_001"):
                logger.info("agent_action")

            # All logs should have step and simulation_id
            for entry in logs.entries:
                if entry.get("event") == "agent_action":
                    assert entry["step"] == 42
                    assert entry["simulation_id"] == "sim_001"

    def test_step_error_handling(self):
        """Test step error logging."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with pytest.raises(ValueError):
                with log_step(step_number=42):
                    raise ValueError("Step error")

            # Should have step_failed log
            assert_log_contains(logs, "step_failed", step=42, error_type="ValueError")

    def test_step_duration(self):
        """Test that step duration is logged."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with log_step(step_number=42):
                time.sleep(0.01)

            # step_completed should have duration_ms
            completed_logs = [
                e for e in logs.entries if e.get("event") == "step_completed"
            ]
            assert len(completed_logs) > 0
            assert "duration_ms" in completed_logs[0]


class TestLogSimulationManager:
    """Test log_simulation context manager."""

    def test_simulation_logging(self):
        """Test simulation start and completion logging."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with log_simulation(
                simulation_id="sim_001", num_agents=100, num_steps=1000
            ):
                pass

            # Should have simulation_started and simulation_completed
            assert_log_contains(
                logs, "simulation_started", simulation_id="sim_001", num_agents=100
            )
            assert_log_contains(logs, "simulation_completed", simulation_id="sim_001")

    def test_simulation_duration(self):
        """Test that simulation duration is logged."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with log_simulation(simulation_id="sim_001"):
                time.sleep(0.01)

            # simulation_completed should have duration_seconds
            completed_logs = [
                e for e in logs.entries if e.get("event") == "simulation_completed"
            ]
            assert len(completed_logs) > 0
            assert "duration_seconds" in completed_logs[0]

    def test_simulation_error_handling(self):
        """Test simulation error logging."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with pytest.raises(RuntimeError):
                with log_simulation(simulation_id="sim_001"):
                    raise RuntimeError("Simulation error")

            # Should have simulation_failed log
            assert_log_contains(
                logs,
                "simulation_failed",
                simulation_id="sim_001",
                error_type="RuntimeError",
            )


class TestLogExperimentManager:
    """Test log_experiment context manager."""

    def test_experiment_logging(self):
        """Test experiment start and completion logging."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with log_experiment(
                experiment_id="exp_001",
                experiment_name="test_experiment",
                num_iterations=10,
            ):
                pass

            # Should have experiment_started and experiment_completed
            assert_log_contains(
                logs,
                "experiment_started",
                experiment_id="exp_001",
                experiment_name="test_experiment",
                num_iterations=10,
            )
            assert_log_contains(logs, "experiment_completed", experiment_id="exp_001")

    def test_experiment_duration(self):
        """Test that experiment duration is logged."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with log_experiment(experiment_id="exp_001", experiment_name="test"):
                time.sleep(0.01)

            # experiment_completed should have duration_seconds
            completed_logs = [
                e for e in logs.entries if e.get("event") == "experiment_completed"
            ]
            assert len(completed_logs) > 0
            assert "duration_seconds" in completed_logs[0]

    def test_experiment_error_handling(self):
        """Test experiment error logging."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with pytest.raises(ValueError):
                with log_experiment(experiment_id="exp_001", experiment_name="test"):
                    raise ValueError("Experiment error")

            # Should have experiment_failed log
            assert_log_contains(
                logs,
                "experiment_failed",
                experiment_id="exp_001",
                error_type="ValueError",
            )


class TestLogSampler:
    """Test LogSampler class."""

    def test_full_sampling(self):
        """Test that all events are sampled at rate 1.0."""
        sampler = LogSampler(sample_rate=1.0)

        for i in range(100):
            assert sampler.should_log() is True

    def test_no_sampling(self):
        """Test that some events are not sampled at rate < 1.0."""
        sampler = LogSampler(sample_rate=0.1)

        logged_count = sum(1 for i in range(100) if sampler.should_log())

        # Should log approximately 10% (allow some variance)
        assert 0 < logged_count < 100

    def test_time_based_sampling(self):
        """Test time-based sampling."""
        sampler = LogSampler(sample_rate=1.0, min_interval_ms=50)

        # First call should log
        assert sampler.should_log() is True

        # Immediate second call should not log
        assert sampler.should_log() is False

        # After waiting, should log again
        time.sleep(0.06)  # 60ms
        assert sampler.should_log() is True

    def test_reset(self):
        """Test reset functionality."""
        sampler = LogSampler(sample_rate=1.0)

        for i in range(10):
            sampler.should_log()

        assert sampler.counter == 10

        sampler.reset()
        assert sampler.counter == 0


class TestAgentLogger:
    """Test AgentLogger class."""

    def test_initialization(self):
        """Test AgentLogger initialization."""
        configure_logging(environment="testing")
        agent_logger = AgentLogger(agent_id="agent_001", agent_type="independent")

        assert agent_logger.agent_id == "agent_001"
        assert agent_logger.agent_type == "independent"

    def test_log_action(self):
        """Test logging agent action."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            agent_logger = AgentLogger(agent_id="agent_001")
            agent_logger.log_action(
                action_type="move", success=True, reward=10.0, position=(5, 10)
            )

            assert_log_contains(
                logs,
                "agent_action",
                agent_id="agent_001",
                action_type="move",
                success=True,
                reward=10.0,
            )

    def test_log_state_change(self):
        """Test logging state change."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            agent_logger = AgentLogger(agent_id="agent_001")
            agent_logger.log_state_change(
                state_type="energy", old_value=100.0, new_value=90.0
            )

            assert_log_contains(
                logs,
                "agent_state_change",
                agent_id="agent_001",
                state_type="energy",
                old_value=100.0,
                new_value=90.0,
            )

    def test_log_interaction(self):
        """Test logging agent interaction."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            agent_logger = AgentLogger(agent_id="agent_001")
            agent_logger.log_interaction(
                interaction_type="combat", target_id="agent_002", damage=25.0
            )

            assert_log_contains(
                logs,
                "agent_interaction",
                agent_id="agent_001",
                interaction_type="combat",
                target_id="agent_002",
            )

    def test_log_death(self):
        """Test logging agent death."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            agent_logger = AgentLogger(agent_id="agent_001")
            agent_logger.log_death(cause="starvation", final_energy=0.0)

            assert_log_contains(
                logs, "agent_died", agent_id="agent_001", cause="starvation"
            )

    def test_log_birth(self):
        """Test logging agent birth."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            agent_logger = AgentLogger(agent_id="agent_001")
            agent_logger.log_birth(
                parent_ids=["agent_parent1", "agent_parent2"], initial_energy=100.0
            )

            assert_log_contains(logs, "agent_born", agent_id="agent_001")


class TestDatabaseLogger:
    """Test DatabaseLogger class."""

    def test_initialization(self):
        """Test DatabaseLogger initialization."""
        configure_logging(environment="testing")
        db_logger = DatabaseLogger(db_path="/path/to/db.db", simulation_id="sim_001")

        assert db_logger.db_path == "/path/to/db.db"
        assert db_logger.simulation_id == "sim_001"

    def test_log_query(self):
        """Test logging database query."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            db_logger = DatabaseLogger(db_path="/path/to/db.db")
            db_logger.log_query(
                query_type="select", table="agents", duration_ms=15.5, rows=100
            )

            assert_log_contains(
                logs,
                "database_query",
                query_type="select",
                table="agents",
                duration_ms=15.5,
                rows=100,
            )

    def test_log_transaction(self):
        """Test logging database transaction."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            db_logger = DatabaseLogger(db_path="/path/to/db.db")
            db_logger.log_transaction(status="commit", duration_ms=50.0)

            assert_log_contains(
                logs, "database_transaction", status="commit", duration_ms=50.0
            )


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_basic_monitoring(self):
        """Test basic performance monitoring."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with PerformanceMonitor("test_operation"):
                time.sleep(0.01)

            # Should have started and completed logs
            started_logs = [e for e in logs.entries if "started" in e.get("event", "")]
            completed_logs = [
                e for e in logs.entries if e.get("event") == "operation_completed"
            ]

            assert len(started_logs) > 0
            assert len(completed_logs) > 0

            completed_log = completed_logs[0]
            assert completed_log["operation"] == "test_operation"
            assert "duration_ms" in completed_log

    def test_checkpoints(self):
        """Test performance checkpoints."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with PerformanceMonitor("test_operation") as monitor:
                time.sleep(0.01)
                monitor.checkpoint("checkpoint1")
                time.sleep(0.01)
                monitor.checkpoint("checkpoint2")

            # Should have checkpoint logs
            checkpoint_logs = [
                e for e in logs.entries if e.get("event") == "checkpoint"
            ]
            assert len(checkpoint_logs) == 2

            assert checkpoint_logs[0]["checkpoint"] == "checkpoint1"
            assert checkpoint_logs[1]["checkpoint"] == "checkpoint2"

    def test_error_monitoring(self):
        """Test performance monitoring with errors."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with pytest.raises(ValueError):
                with PerformanceMonitor("failing_operation"):
                    raise ValueError("Test error")

            # Should have operation_failed log
            failed_logs = [
                e for e in logs.entries if e.get("event") == "operation_failed"
            ]
            assert len(failed_logs) > 0

            failed_log = failed_logs[0]
            assert failed_log["operation"] == "failing_operation"
            assert failed_log["error_type"] == "ValueError"
            assert "duration_ms" in failed_log

    def test_custom_logger_name(self):
        """Test with custom logger name."""
        configure_logging(environment="testing")

        with capture_logs() as logs:
            with PerformanceMonitor("test_op", logger_name="custom.logger"):
                pass

            # Logs should exist
            assert len(logs.entries) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
