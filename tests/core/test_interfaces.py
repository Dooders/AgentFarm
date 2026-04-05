"""Tests for farm/core/interfaces.py – protocol definitions and structural subtyping."""

import pytest

from farm.core.interfaces import (
    AgentProtocol,
    ChartAnalyzerProtocol,
    DataLoggerProtocol,
    DatabaseFactoryProtocol,
    DatabaseProtocol,
    EnvironmentProtocol,
    MetricsTrackerProtocol,
    RepositoryProtocol,
)


# ---------------------------------------------------------------------------
# Helpers – minimal concrete implementations of each protocol
# ---------------------------------------------------------------------------

class _ConcreteAgent:
    agent_id = "a1"
    position = (0.0, 0.0)
    resource_level = 1.0
    current_health = 100.0
    starting_health = 100.0
    starvation_counter = 0
    is_defending = False
    total_reward = 0.0
    birth_time = 0.0
    alive = True
    genome_id = None
    generation = 0

    def get_state_dict(self):
        return {}


class _ConcreteMetrics:
    def record_birth(self): pass
    def record_death(self): pass
    def record_combat_encounter(self): pass
    def record_successful_attack(self): pass
    def record_resources_shared(self, amount): pass
    def calculate_metrics(self, agent_objects, resources, time, config=None): return {}
    def update_metrics(self, metrics, db=None, time=None, agent_objects=None, resources=None): pass


class _ConcreteEnvironment:
    def get_agents(self): return {}
    def get_resources(self): return []
    def get_current_time(self): return 0


class _ConcreteDB:
    logger = None

    def log_step(self, step_number, agent_states, resource_states, metrics): pass
    def export_data(self, filepath, format="csv", **kwargs): pass
    def get_agent_repository(self): return None
    def get_action_repository(self): return None
    def get_resource_repository(self): return None
    def _execute_in_transaction(self, func): return None
    def close(self): pass
    def get_configuration(self): return {}
    def save_configuration(self, config): pass


class _ConcreteDataLogger:
    def log_agent_action(self, step_number, agent_id, action_type, **kwargs): pass
    def log_step(self, step_number, agent_states, resource_states, metrics): pass
    def log_agent(self, agent_id, birth_time, agent_type, position, initial_resources, starting_health, **kwargs): pass
    def log_health_incident(self, step_number, agent_id, health_before, health_after, cause, details=None): pass
    def flush_all_buffers(self): pass


class _ConcreteRepository:
    def add(self, entity): pass
    def get_by_id(self, entity_id): return None
    def update(self, entity): pass
    def delete(self, entity): pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRuntimeCheckableProtocols:
    """Verify @runtime_checkable protocols work with isinstance()."""

    def test_database_protocol_isinstance(self):
        db = _ConcreteDB()
        assert isinstance(db, DatabaseProtocol)

    def test_data_logger_protocol_isinstance(self):
        logger = _ConcreteDataLogger()
        assert isinstance(logger, DataLoggerProtocol)

    def test_repository_protocol_isinstance(self):
        repo = _ConcreteRepository()
        assert isinstance(repo, RepositoryProtocol)

    def test_plain_object_is_not_database_protocol(self):
        assert not isinstance(object(), DatabaseProtocol)

    def test_plain_object_is_not_data_logger_protocol(self):
        assert not isinstance(object(), DataLoggerProtocol)


class TestProtocolAttributes:
    """Verify protocols expose the expected attribute / method names."""

    def test_agent_protocol_required_attrs(self):
        annotations = AgentProtocol.__protocol_attrs__ if hasattr(AgentProtocol, "__protocol_attrs__") else set()
        # Manually verify key attributes via annotation inspection
        hints = AgentProtocol.__annotations__
        for attr in ("agent_id", "position", "resource_level", "current_health"):
            assert attr in hints, f"Missing attribute '{attr}' in AgentProtocol"

    def test_metrics_tracker_protocol_methods(self):
        for method in ("record_birth", "record_death", "calculate_metrics", "update_metrics"):
            assert hasattr(MetricsTrackerProtocol, method)

    def test_database_protocol_methods(self):
        for method in ("log_step", "export_data", "close", "get_configuration"):
            assert hasattr(DatabaseProtocol, method)

    def test_data_logger_protocol_methods(self):
        for method in ("log_agent_action", "log_step", "log_agent", "log_health_incident", "flush_all_buffers"):
            assert hasattr(DataLoggerProtocol, method)

    def test_repository_protocol_methods(self):
        for method in ("add", "get_by_id", "update", "delete"):
            assert hasattr(RepositoryProtocol, method)

    def test_environment_protocol_methods(self):
        for method in ("get_agents", "get_resources", "get_current_time"):
            assert hasattr(EnvironmentProtocol, method)


class TestConcreteImplementations:
    """Verify concrete helpers actually satisfy protocol contracts."""

    def test_concrete_agent_has_all_attrs(self):
        agent = _ConcreteAgent()
        for attr in ("agent_id", "position", "resource_level", "current_health",
                     "starting_health", "starvation_counter", "is_defending",
                     "total_reward", "birth_time", "alive", "genome_id", "generation"):
            assert hasattr(agent, attr)
        assert agent.get_state_dict() == {}

    def test_concrete_metrics_callable(self):
        m = _ConcreteMetrics()
        m.record_birth()
        m.record_death()
        m.record_combat_encounter()
        m.record_successful_attack()
        m.record_resources_shared(1.0)
        result = m.calculate_metrics({}, [], 0)
        assert isinstance(result, dict)

    def test_concrete_environment_callable(self):
        env = _ConcreteEnvironment()
        assert isinstance(env.get_agents(), dict)
        assert isinstance(env.get_resources(), list)
        assert env.get_current_time() == 0

    def test_concrete_db_callable(self):
        db = _ConcreteDB()
        db.log_step(0, [], [], {})
        db.close()
        assert db.get_configuration() == {}

    def test_concrete_data_logger_callable(self):
        dl = _ConcreteDataLogger()
        dl.log_agent_action(0, "a1", "move")
        dl.log_step(0, [], [], {})
        dl.log_agent("a1", 0, "system", (0, 0), 1.0, 100.0)
        dl.log_health_incident(0, "a1", 100.0, 90.0, "combat")
        dl.flush_all_buffers()

    def test_concrete_repository_callable(self):
        repo = _ConcreteRepository()
        repo.add("entity")
        result = repo.get_by_id(1)
        assert result is None
        repo.update("entity")
        repo.delete("entity")
