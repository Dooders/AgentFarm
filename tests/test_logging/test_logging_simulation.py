"""Comprehensive tests for farm.utils.logging.simulation module.

Tests for:
- TypedSimulationLogger class
- SimulationLogger protocol
- get_simulation_logger function
"""

import pytest

from farm.utils.logging import configure_logging
from farm.utils.logging.simulation import (
    TypedSimulationLogger,
    SimulationLogger,
    get_simulation_logger,
)
from farm.utils.logging.test_helpers import (
    capture_logs,
    assert_log_contains,
)
import structlog


class TestSimulationLoggerProtocol:
    """Test SimulationLogger protocol."""

    def test_protocol_compatibility(self):
        """Test that TypedSimulationLogger implements SimulationLogger protocol."""
        configure_logging(environment="testing")
        logger = get_simulation_logger(__name__)

        # Should be compatible with protocol
        assert isinstance(logger, SimulationLogger)


class TestTypedSimulationLogger:
    """Test TypedSimulationLogger class."""

    def test_initialization(self):
        """Test TypedSimulationLogger initialization."""
        configure_logging(environment="testing")
        base_logger = structlog.get_logger(__name__)
        sim_logger = TypedSimulationLogger(base_logger)

        assert sim_logger.logger is base_logger

    def test_log_agent_action(self):
        """Test logging agent action."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_agent_action(
                agent_id="agent_001",
                action="move",
                success=True,
                duration_ms=50.0
            )

            assert_log_contains(
                logs,
                "agent_action",
                agent_id="agent_001",
                action="move",
                success=True,
                duration_ms=50.0
            )

    def test_log_agent_action_without_duration(self):
        """Test logging agent action without duration."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_agent_action(
                agent_id="agent_001",
                action="eat",
                success=True
            )

            assert_log_contains(
                logs,
                "agent_action",
                agent_id="agent_001",
                action="eat",
                success=True
            )

    def test_log_agent_action_with_extra_kwargs(self):
        """Test logging agent action with extra context."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_agent_action(
                agent_id="agent_001",
                action="attack",
                success=True,
                target_id="agent_002",
                damage=25.0
            )

            assert_log_contains(
                logs,
                "agent_action",
                agent_id="agent_001",
                action="attack",
                success=True,
                target_id="agent_002",
                damage=25.0
            )

    def test_log_population_change(self):
        """Test logging population change."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_population_change(
                population=100,
                change=5,
                step=42
            )

            assert_log_contains(
                logs,
                "population_changed",
                population=100,
                change=5,
                step=42
            )

    def test_log_population_change_negative(self):
        """Test logging population decrease."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_population_change(
                population=95,
                change=-5,
                step=43
            )

            assert_log_contains(
                logs,
                "population_changed",
                population=95,
                change=-5,
                step=43
            )

    def test_log_resource_update(self):
        """Test logging resource update."""
        configure_logging(environment="testing", log_level="DEBUG")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_resource_update(
                total_resources=1000.0,
                active_nodes=25,
                step=42
            )

            assert_log_contains(
                logs,
                "resources_updated",
                level="debug",
                total=1000.0,
                active=25,
                step=42
            )

    def test_log_resource_update_with_extra_context(self):
        """Test logging resource update with extra context."""
        configure_logging(environment="testing", log_level="DEBUG")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_resource_update(
                total_resources=500.0,
                active_nodes=10,
                step=42,
                regeneration_rate=5.0
            )

            assert_log_contains(
                logs,
                "resources_updated",
                total=500.0,
                active=10,
                step=42,
                regeneration_rate=5.0
            )

    def test_log_simulation_event(self):
        """Test logging general simulation event."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_simulation_event(
                event="catastrophe",
                step=100,
                severity="high"
            )

            assert_log_contains(
                logs,
                "simulation_event",
                event_name="catastrophe",
                step=100,
                severity="high"
            )

    def test_log_performance_metric(self):
        """Test logging performance metric."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_performance_metric(
                metric_name="step_duration",
                value=125.5,
                unit="ms"
            )

            assert_log_contains(
                logs,
                "performance_metric",
                metric="step_duration",
                value=125.5,
                unit="ms"
            )

    def test_log_performance_metric_default_unit(self):
        """Test logging performance metric with default unit."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_performance_metric(
                metric_name="memory_usage",
                value=512.0
            )

            assert_log_contains(
                logs,
                "performance_metric",
                metric="memory_usage",
                value=512.0,
                unit="ms"  # Default unit
            )

    def test_log_experiment_event(self):
        """Test logging experiment event."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger.log_experiment_event(
                event="parameter_sweep_started",
                experiment_id="exp_001",
                num_runs=10
            )

            assert_log_contains(
                logs,
                "experiment_event",
                event_name="parameter_sweep_started",
                experiment_id="exp_001",
                num_runs=10
            )

    def test_bind_context(self):
        """Test binding context to create new logger."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            bound_logger = sim_logger.bind(simulation_id="sim_001", step=42)

            # Should return a new TypedSimulationLogger
            assert isinstance(bound_logger, TypedSimulationLogger)
            assert bound_logger is not sim_logger

            # Bound context should appear in logs
            bound_logger.log_agent_action("agent_001", "move", True)

            assert_log_contains(
                logs,
                "agent_action",
                simulation_id="sim_001",
                step=42,
                agent_id="agent_001"
            )

    def test_bind_multiple_contexts(self):
        """Test binding multiple contexts."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            logger1 = sim_logger.bind(simulation_id="sim_001")
            logger2 = logger1.bind(step=42)
            logger3 = logger2.bind(agent_type="independent")

            logger3.log_agent_action("agent_001", "move", True)

            assert_log_contains(
                logs,
                "agent_action",
                simulation_id="sim_001",
                step=42,
                agent_type="independent",
                agent_id="agent_001"
            )


class TestGetSimulationLogger:
    """Test get_simulation_logger function."""

    def test_get_simulation_logger_default(self):
        """Test getting simulation logger with default name."""
        configure_logging(environment="testing")
        logger = get_simulation_logger()

        assert isinstance(logger, TypedSimulationLogger)

    def test_get_simulation_logger_with_name(self):
        """Test getting simulation logger with specific name."""
        configure_logging(environment="testing")
        logger = get_simulation_logger(__name__)

        assert isinstance(logger, TypedSimulationLogger)

    def test_multiple_loggers_independent(self):
        """Test that multiple loggers are independent."""
        configure_logging(environment="testing")
        logger1 = get_simulation_logger("logger1")
        logger2 = get_simulation_logger("logger2")

        assert logger1 is not logger2
        assert logger1.logger is not logger2.logger


class TestSimulationLoggerScenarios:
    """Test real-world simulation logging scenarios."""

    def test_complete_simulation_run(self):
        """Test logging a complete simulation run."""
        configure_logging(environment="testing", log_level="DEBUG")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            # Bind simulation context
            sim_logger = sim_logger.bind(simulation_id="sim_001")

            # Simulation start
            sim_logger.log_simulation_event("simulation_started", step=0, num_agents=100)

            # Step 1
            step_logger = sim_logger.bind(step=1)
            step_logger.log_agent_action("agent_001", "move", True, duration_ms=10.0)
            step_logger.log_agent_action("agent_002", "eat", True, duration_ms=5.0)
            step_logger.log_resource_update(total_resources=950.0, active_nodes=25, step=1)
            step_logger.log_population_change(population=100, change=0, step=1)

            # Step 2
            step_logger = sim_logger.bind(step=2)
            step_logger.log_agent_action("agent_003", "reproduce", True, duration_ms=20.0)
            step_logger.log_resource_update(total_resources=900.0, active_nodes=25, step=2)
            step_logger.log_population_change(population=101, change=1, step=2)

            # Simulation end
            sim_logger.log_simulation_event("simulation_completed", step=2)
            sim_logger.log_performance_metric("total_duration", 1250.5, "ms")

            # Verify all logs have simulation_id
            for entry in logs.entries:
                assert entry.get("simulation_id") == "sim_001"

            # Verify event counts
            agent_actions = [e for e in logs.entries if e.get("event") == "agent_action"]
            resource_updates = [e for e in logs.entries if e.get("event") == "resources_updated"]
            pop_changes = [e for e in logs.entries if e.get("event") == "population_changed"]

            assert len(agent_actions) == 3
            assert len(resource_updates) == 2
            assert len(pop_changes) == 2

    def test_multi_agent_interactions(self):
        """Test logging multiple agent interactions."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger = sim_logger.bind(simulation_id="sim_001", step=1)

            # Multiple agents performing actions
            for agent_id in range(10):
                sim_logger.log_agent_action(
                    agent_id=f"agent_{agent_id:03d}",
                    action="move",
                    success=True,
                    position=(agent_id, agent_id * 2)
                )

            # Verify all actions were logged
            agent_actions = [e for e in logs.entries if e.get("event") == "agent_action"]
            assert len(agent_actions) == 10

    def test_experiment_with_multiple_runs(self):
        """Test logging an experiment with multiple runs."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            experiment_logger = sim_logger.bind(experiment_id="exp_001")

            experiment_logger.log_experiment_event("experiment_started", "exp_001", num_runs=3)

            for run_num in range(3):
                run_logger = experiment_logger.bind(run_num=run_num)

                run_logger.log_simulation_event("run_started", step=0)
                run_logger.log_agent_action("agent_001", "move", True)
                run_logger.log_population_change(population=100, change=0, step=1)
                run_logger.log_simulation_event("run_completed", step=1)

            experiment_logger.log_experiment_event("experiment_completed", "exp_001")

            # Verify all logs have experiment_id
            for entry in logs.entries:
                assert entry.get("experiment_id") == "exp_001"

            # Verify run distribution
            run_0_logs = [e for e in logs.entries if e.get("run_num") == 0]
            run_1_logs = [e for e in logs.entries if e.get("run_num") == 1]
            run_2_logs = [e for e in logs.entries if e.get("run_num") == 2]

            assert len(run_0_logs) == 4  # run_started, agent_action, pop_change, run_completed
            assert len(run_1_logs) == 4
            assert len(run_2_logs) == 4

    def test_performance_monitoring(self):
        """Test logging performance metrics throughout simulation."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        with capture_logs() as logs:
            sim_logger = sim_logger.bind(simulation_id="sim_001")

            # Log various performance metrics
            sim_logger.log_performance_metric("step_duration", 100.5, "ms")
            sim_logger.log_performance_metric("memory_usage", 512.0, "MB")
            sim_logger.log_performance_metric("cpu_usage", 45.3, "%")
            sim_logger.log_performance_metric("agent_processing_time", 75.2, "ms")

            # Verify all metrics logged
            perf_metrics = [e for e in logs.entries if e.get("event") == "performance_metric"]
            assert len(perf_metrics) == 4

            # Verify metric names
            metric_names = [e.get("metric") for e in perf_metrics]
            assert "step_duration" in metric_names
            assert "memory_usage" in metric_names
            assert "cpu_usage" in metric_names
            assert "agent_processing_time" in metric_names


class TestTypeSafety:
    """Test type safety features of TypedSimulationLogger."""

    def test_required_parameters(self):
        """Test that required parameters are enforced by type hints."""
        configure_logging(environment="testing")
        sim_logger = get_simulation_logger(__name__)

        # These should work (all required params provided)
        with capture_logs() as logs:
            sim_logger.log_agent_action("agent_001", "move", True)
            sim_logger.log_population_change(100, 5, 42)
            sim_logger.log_resource_update(1000.0, 25, 42)
            sim_logger.log_simulation_event("test_event", 42)
            sim_logger.log_performance_metric("test_metric", 100.0)
            sim_logger.log_experiment_event("test_event", "exp_001")

            assert len(logs.entries) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

