"""Tests for hyperparameter chromosome wiring in AgentCore reproduction."""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from farm.core.agent.config.component_configs import AgentComponentConfig
from farm.core.agent.core import AgentCore
from farm.core.decision.config import DecisionConfig
from farm.core.hyperparameter_chromosome import chromosome_from_learning_config


def _build_parent_agent_for_reproduction() -> AgentCore:
    """Create a minimal AgentCore instance with required reproduction attributes."""
    agent = object.__new__(AgentCore)
    agent.agent_id = "parent_1"
    agent.agent_type = "system"
    agent.generation = 4
    agent.services = Mock()
    agent.environment = Mock()
    agent.environment.get_next_agent_id.return_value = "child_1"
    agent.state = Mock()
    agent.state.position = (2.0, 3.0)
    agent.state._state = Mock()
    agent.state._state.model_copy.return_value = Mock()

    config = AgentComponentConfig.default()
    config.decision = DecisionConfig(
        learning_rate=0.01,
        epsilon_decay=0.995,
        memory_size=2000,
    )
    agent.config = config

    resource_component = Mock()
    reproduction_component = Mock()
    reproduction_component.can_reproduce.return_value = True
    reproduction_component.config.offspring_cost = 5.0
    reproduction_component.config.offspring_initial_resources = 7.0
    reproduction_component.offspring_created = 0

    agent._components = {
        "resource": resource_component,
        "reproduction": reproduction_component,
    }
    agent.hyperparameter_chromosome = chromosome_from_learning_config(agent.config.decision)
    return agent


def test_reproduce_mutates_learning_rate_and_passes_child_config():
    parent = _build_parent_agent_for_reproduction()
    offspring = Mock()
    offspring.state = Mock()
    offspring.state._state = Mock()
    offspring.state._state.model_copy.return_value = Mock()

    with patch("farm.core.hyperparameter_chromosome.random.random", return_value=0.0), patch(
        "farm.core.hyperparameter_chromosome.random.gauss", return_value=0.002
    ), patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring

        success = AgentCore.reproduce(parent)

    assert success is True
    parent.get_component("resource").remove.assert_called_once_with(5.0)
    parent.environment.add_agent.assert_called_once_with(offspring, flush_immediately=True)
    assert offspring.generation == 5

    kwargs = factory.create_learning_agent.call_args.kwargs
    child_config = kwargs["config"]
    assert child_config is not parent.config
    assert child_config.decision.learning_rate == 0.012
    # Placeholder genes remain fixed by default registry.
    assert child_config.decision.epsilon_decay == 0.995
    assert child_config.decision.memory_size == 2000
    assert offspring.hyperparameter_chromosome.get_value("learning_rate") == 0.012


def test_reproduce_logs_exception_and_returns_false():
    parent = _build_parent_agent_for_reproduction()

    with patch(
        "farm.core.agent.core.mutate_chromosome",
        side_effect=RuntimeError("boom"),
    ), patch("farm.core.agent.core.logger.exception") as log_exception:
        success = AgentCore.reproduce(parent)

    assert success is False
    parent.get_component("resource").remove.assert_called_once_with(5.0)
    parent.get_component("resource").add.assert_called_once_with(5.0)
    log_exception.assert_called_once()


def test_reproduce_returns_true_when_post_add_bookkeeping_fails():
    """Once offspring insertion succeeds, bookkeeping errors should not flip success to failure."""
    parent = _build_parent_agent_for_reproduction()
    offspring = Mock()
    offspring.state = Mock()
    offspring.state._state = Mock()
    offspring.state._state.model_copy.return_value = Mock()

    class _ReproductionComponentWithFailingCounter:
        def __init__(self):
            self.config = SimpleNamespace(offspring_cost=5.0, offspring_initial_resources=7.0)

        def can_reproduce(self) -> bool:
            return True

        @property
        def offspring_created(self) -> int:
            return 0

        @offspring_created.setter
        def offspring_created(self, _value: int) -> None:
            raise RuntimeError("counter write failed")

    parent._components["reproduction"] = _ReproductionComponentWithFailingCounter()

    with patch("farm.core.hyperparameter_chromosome.random.random", return_value=1.0), patch(
        "farm.core.agent.factory.AgentFactory"
    ) as factory_cls, patch("farm.core.agent.core.logger.exception") as log_exception:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring

        success = AgentCore.reproduce(parent)

    assert success is True
    parent.environment.add_agent.assert_called_once_with(offspring, flush_immediately=True)
    parent.get_component("resource").add.assert_not_called()
    log_exception.assert_called_once()


def test_reproduce_fallback_derives_chromosome_when_missing():
    """Reproduction must recover gracefully when hyperparameter_chromosome is absent."""
    parent = _build_parent_agent_for_reproduction()
    # Simulate an agent that was created without the chromosome attribute
    del parent.hyperparameter_chromosome

    offspring = Mock()
    offspring.state = Mock()
    offspring.state._state = Mock()
    offspring.state._state.model_copy.return_value = Mock()

    with patch("farm.core.hyperparameter_chromosome.random.random", return_value=1.0), patch(
        "farm.core.agent.factory.AgentFactory"
    ) as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring

        success = AgentCore.reproduce(parent)

    assert success is True
    # Chromosome must have been derived and synced back to the parent.
    assert hasattr(parent, "hyperparameter_chromosome")
    assert parent.hyperparameter_chromosome.get_value("learning_rate") == pytest.approx(0.01)


def test_validate_decision_config_rejects_non_positive_learning_rate():
    """_validate_decision_config_for_hyperparameters raises for learning_rate <= 0."""
    agent = object.__new__(AgentCore)
    config = AgentComponentConfig.default()
    # Bypass Pydantic validation to simulate a plain object with bad value
    object.__setattr__(config.decision, "learning_rate", 0.0)
    agent.config = config

    with pytest.raises(ValueError, match="learning_rate.*invalid"):
        agent._validate_decision_config_for_hyperparameters()


def test_validate_decision_config_rejects_non_positive_memory_size():
    """_validate_decision_config_for_hyperparameters raises for memory_size <= 0."""
    agent = object.__new__(AgentCore)
    config = AgentComponentConfig.default()
    # Bypass Pydantic validation to simulate a plain object with bad value
    object.__setattr__(config.decision, "memory_size", -1)
    agent.config = config

    with pytest.raises(ValueError, match="memory_size.*invalid"):
        agent._validate_decision_config_for_hyperparameters()


def test_validate_decision_config_passes_for_valid_values():
    """_validate_decision_config_for_hyperparameters must not raise for valid defaults."""
    agent = object.__new__(AgentCore)
    config = AgentComponentConfig.default()
    agent.config = config

    # Should not raise
    agent._validate_decision_config_for_hyperparameters()


def test_reproduce_rolls_back_partially_added_offspring_before_refund():
    """If add_agent partially inserts offspring and then fails, rollback then refund."""
    parent = _build_parent_agent_for_reproduction()

    class _PartiallyFailingEnvironment:
        def __init__(self):
            self._agent_objects = {}
            self.agents = []
            self.rewards = {}
            self._cumulative_rewards = {}
            self.terminations = {}
            self.truncations = {}
            self.infos = {}
            self.observations = {}
            self.agent_observations = {}
            self.spatial_index = Mock()

        def get_next_agent_id(self) -> str:
            return "child_1"

        def add_agent(self, offspring, flush_immediately: bool = False):
            self._agent_objects[offspring.agent_id] = offspring
            self.agents.append(offspring.agent_id)
            raise RuntimeError("flush failed after insert")

        def remove_agent(self, offspring) -> None:
            self._agent_objects.pop(offspring.agent_id, None)
            if offspring.agent_id in self.agents:
                self.agents.remove(offspring.agent_id)

    parent.environment = _PartiallyFailingEnvironment()
    offspring = SimpleNamespace(
        agent_id="child_1",
        state=Mock(),
        generation=0,
    )
    offspring.state._state = Mock()
    offspring.state._state.model_copy.return_value = Mock()

    with patch("farm.core.hyperparameter_chromosome.random.random", return_value=1.0), patch(
        "farm.core.agent.factory.AgentFactory"
    ) as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is False
    parent.get_component("resource").add.assert_called_once_with(5.0)
    assert "child_1" not in parent.environment._agent_objects
    assert "child_1" not in parent.environment.agents


def test_reproduce_skips_refund_when_partial_offspring_rollback_fails():
    """Avoid refund if offspring may still exist and rollback cannot complete."""
    parent = _build_parent_agent_for_reproduction()

    class _RollbackFailingEnvironment:
        def __init__(self):
            self._agent_objects = {}
            self.agents = []

        def get_next_agent_id(self) -> str:
            return "child_1"

        def add_agent(self, offspring, flush_immediately: bool = False):
            self._agent_objects[offspring.agent_id] = offspring
            self.agents.append(offspring.agent_id)
            raise RuntimeError("flush failed after insert")

        def remove_agent(self, offspring) -> None:
            raise RuntimeError("rollback failed")

    parent.environment = _RollbackFailingEnvironment()
    offspring = SimpleNamespace(
        agent_id="child_1",
        state=Mock(),
        generation=0,
    )
    offspring.state._state = Mock()
    offspring.state._state.model_copy.return_value = Mock()

    with patch("farm.core.hyperparameter_chromosome.random.random", return_value=1.0), patch(
        "farm.core.agent.factory.AgentFactory"
    ) as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is False
    parent.get_component("resource").add.assert_not_called()

