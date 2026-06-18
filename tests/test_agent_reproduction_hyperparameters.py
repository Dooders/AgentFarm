"""Tests for hyperparameter chromosome wiring in AgentCore reproduction."""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from farm.core.agent.config.component_configs import AgentComponentConfig
from farm.core.agent.core import AgentCore
from farm.core.decision.config import DecisionConfig
from farm.core.hyperparameter_chromosome import chromosome_from_learning_config
from farm.core.inheritance_telemetry import InheritanceTelemetry
from farm.core.policy_inheritance import (
    WARMSTART_REASON_INCOMPATIBLE_STATE,
)
from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy


def _build_parent_agent_for_reproduction() -> AgentCore:
    """Create a minimal AgentCore instance with required reproduction attributes.

    The mock environment defaults to ``intrinsic_evolution_policy=None`` so the
    parent's chromosome is inherited unchanged unless a test explicitly opts
    into the in-situ evolution policy.
    """
    agent = object.__new__(AgentCore)
    agent.agent_id = "parent_1"
    agent.agent_type = "system"
    agent.generation = 4
    agent.services = Mock()
    agent.environment = Mock()
    agent.environment.get_next_agent_id.return_value = "child_1"
    agent.environment.intrinsic_evolution_policy = None
    agent.environment.intrinsic_evolution_rng = None
    agent.environment.inheritance_telemetry = InheritanceTelemetry()
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


def test_reproduce_inherits_chromosome_unchanged_without_policy():
    """Without an intrinsic evolution policy, child chromosome equals parent's exactly."""
    parent = _build_parent_agent_for_reproduction()
    parent_lr = parent.hyperparameter_chromosome.get_value("learning_rate")

    offspring = Mock()
    offspring.state = Mock()
    offspring.state._state = Mock()
    offspring.state._state.model_copy.return_value = Mock()

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    assert offspring.generation == 5
    assert offspring.hyperparameter_chromosome.get_value("learning_rate") == parent_lr
    kwargs = factory.create_learning_agent.call_args.kwargs
    child_config = kwargs["config"]
    assert child_config is not parent.config
    # Pure inheritance: every gene matches.
    assert child_config.decision.learning_rate == parent_lr


def test_reproduce_with_policy_mutates_learning_rate_and_passes_child_config():
    """When the policy is enabled, child genes are mutated using the policy's knobs."""
    parent = _build_parent_agent_for_reproduction()
    parent.environment.intrinsic_evolution_policy = IntrinsicEvolutionPolicy(
        mutation_rate=1.0,
        mutation_scale=0.2,
    )
    parent.environment.intrinsic_evolution_rng = None

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
    assert child_config.decision.learning_rate == pytest.approx(0.012, abs=1e-9)
    assert child_config.decision.epsilon_decay == pytest.approx(0.997, abs=1e-9)
    assert child_config.decision.gamma == pytest.approx(0.992, abs=1e-9)
    assert child_config.decision.memory_size == 2000
    assert offspring.hyperparameter_chromosome.get_value("learning_rate") == pytest.approx(0.012, abs=1e-9)


def test_reproduce_lamarckian_policy_applies_warmstart_and_tracks_counter():
    """Lamarckian mode should invoke warm-start and increment applied counter."""
    parent = _build_parent_agent_for_reproduction()
    parent.environment.intrinsic_evolution_policy = IntrinsicEvolutionPolicy(
        mutation_rate=0.0,
        mutation_scale=0.0,
        inheritance_mode="lamarckian",
    )
    parent.behavior = Mock()
    parent.behavior.decision_module = Mock()
    parent.behavior.decision_module.algorithm = Mock()
    parent.behavior.decision_module.algorithm.get_model_state.return_value = {
        "policy_state_dict": {"w": Mock(shape=(2, 2))}
    }

    offspring = Mock()
    offspring.state = Mock()
    offspring.state._state = Mock()
    offspring.state._state.model_copy.return_value = Mock()
    offspring.behavior = Mock()
    offspring.behavior.decision_module = Mock()
    offspring.behavior.decision_module.algorithm = Mock()
    offspring.behavior.decision_module.algorithm.policy = Mock()
    offspring.behavior.decision_module.algorithm.policy.state_dict.return_value = {
        "w": Mock(shape=(2, 2))
    }

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    telemetry = parent.environment.inheritance_telemetry
    assert telemetry.lamarckian_warmstart_applied == 1
    assert telemetry.lamarckian_warmstart_skipped == 0
    assert dict(telemetry.lamarckian_warmstart_skipped_reasons) == {}
    offspring.behavior.decision_module.algorithm.load_model_state.assert_called_once()


def test_step_records_decide_action_failure_in_telemetry():
    """A ``decide_action`` exception during ``step`` must bump the failure counter."""
    parent = _build_parent_agent_for_reproduction()
    parent.alive = True
    parent._invalidate_obs_cache = Mock()
    parent.behavior = Mock()
    parent.behavior.decide_action.side_effect = RuntimeError("predict_proba boom")
    parent._create_observation = Mock(return_value=Mock())
    parent._get_enabled_actions = Mock(return_value=None)
    parent._execute_action = Mock()
    # Resource component on the parent has a starvation counter below the
    # threshold so the early-return path isn't triggered.
    resource_comp = parent._components["resource"]
    resource_comp.starvation_counter = 0
    resource_comp.config = SimpleNamespace(starvation_threshold=100)

    telemetry = parent.environment.inheritance_telemetry

    with patch("farm.core.agent.core.logger.warning") as warn:
        AgentCore.step(parent)

    warn.assert_called_once()
    assert telemetry.decide_action_failures == 1
    assert telemetry.decide_action_failure_reasons["RuntimeError"] == 1
    parent._execute_action.assert_not_called()


def test_reproduce_lamarckian_incompatible_policy_counts_as_skipped():
    """Shape/key mismatch should skip warm-start without failing reproduction."""
    parent = _build_parent_agent_for_reproduction()
    parent.environment.intrinsic_evolution_policy = IntrinsicEvolutionPolicy(
        mutation_rate=0.0,
        mutation_scale=0.0,
        inheritance_mode="lamarckian",
    )
    parent.behavior = Mock()
    parent.behavior.decision_module = Mock()
    parent.behavior.decision_module.algorithm = Mock()
    parent.behavior.decision_module.algorithm.get_model_state.return_value = {
        "policy_state_dict": {"w_parent": Mock(shape=(4, 4))}
    }

    offspring = Mock()
    offspring.state = Mock()
    offspring.state._state = Mock()
    offspring.state._state.model_copy.return_value = Mock()
    offspring.behavior = Mock()
    offspring.behavior.decision_module = Mock()
    offspring.behavior.decision_module.algorithm = Mock()
    offspring.behavior.decision_module.algorithm.policy = Mock()
    offspring.behavior.decision_module.algorithm.policy.state_dict.return_value = {
        "w_child": Mock(shape=(2, 2))
    }

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    telemetry = parent.environment.inheritance_telemetry
    assert telemetry.lamarckian_warmstart_applied == 0
    assert telemetry.lamarckian_warmstart_skipped == 1
    assert telemetry.lamarckian_warmstart_skipped_reasons[
        WARMSTART_REASON_INCOMPATIBLE_STATE
    ] == 1
    offspring.behavior.decision_module.algorithm.load_model_state.assert_not_called()


def test_reproduce_logs_exception_and_returns_false():
    """Errors during chromosome derivation should refund and return False."""
    parent = _build_parent_agent_for_reproduction()

    with patch(
        "farm.core.agent.core.AgentCore._derive_child_chromosome",
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
    # Verify spatial index was fully refreshed (not just marked dirty) so the
    # removed offspring is evicted from KD-tree queries.
    parent.environment.spatial_index.set_references.assert_called_once_with([], [])


def test_reproduce_rolls_back_partial_offspring_without_resource_component():
    """Partial insert cleanup must run even when no resource was deducted (no resource component)."""
    parent = _build_parent_agent_for_reproduction()
    parent._components = {"reproduction": parent._components["reproduction"]}

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
    assert "child_1" not in parent.environment._agent_objects
    assert "child_1" not in parent.environment.agents
    parent.environment.spatial_index.set_references.assert_called_once_with([], [])


def test_reproduce_skips_refund_when_partial_offspring_rollback_fails():
    """Avoid refund if offspring may still exist and rollback cannot complete."""
    parent = _build_parent_agent_for_reproduction()

    class _CleanupFailingDict(dict):
        """dict subclass whose pop() always raises to simulate a cleanup failure."""

        def pop(self, key, *args):
            raise RuntimeError("rollback failed")

    class _RollbackFailingEnvironment:
        def __init__(self):
            # Use the failing dict so direct cleanup (pop) raises during rollback.
            self._agent_objects = _CleanupFailingDict()
            self.agents = []

        def get_next_agent_id(self) -> str:
            return "child_1"

        def add_agent(self, offspring, flush_immediately: bool = False):
            self._agent_objects[offspring.agent_id] = offspring
            self.agents.append(offspring.agent_id)
            raise RuntimeError("flush failed after insert")

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


def test_reproduce_refunds_when_rollback_errors_but_offspring_is_not_present():
    """Refund should proceed when rollback errors but no offspring remains."""
    parent = _build_parent_agent_for_reproduction()

    class _RollbackHookRaisesAfterCleanupEnvironment:
        def __init__(self):
            self._agent_objects = {}
            self.agents = []

        def get_next_agent_id(self) -> str:
            return "child_1"

        def add_agent(self, offspring, flush_immediately: bool = False):
            self._agent_objects[offspring.agent_id] = offspring
            self.agents.append(offspring.agent_id)
            raise RuntimeError("flush failed after insert")

        def rollback_partial_agent_add(self, agent_id: str) -> bool:
            self._agent_objects.pop(agent_id, None)
            if agent_id in self.agents:
                self.agents.remove(agent_id)
            raise RuntimeError("rollback hook threw after cleanup")

    parent.environment = _RollbackHookRaisesAfterCleanupEnvironment()
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


def test_reproduce_uses_environment_partial_add_rollback_hook_when_available():
    """Prefer Environment.rollback_partial_agent_add for rollback-aware cleanup."""
    parent = _build_parent_agent_for_reproduction()

    class _HookedRollbackEnvironment:
        def __init__(self):
            self.rollback_calls = []
            self._agent_objects = {}
            self.agents = []

        def get_next_agent_id(self) -> str:
            return "child_1"

        def add_agent(self, offspring, flush_immediately: bool = False):
            self._agent_objects[offspring.agent_id] = offspring
            self.agents.append(offspring.agent_id)
            raise RuntimeError("flush failed after insert")

        def rollback_partial_agent_add(self, agent_id: str) -> bool:
            self.rollback_calls.append(agent_id)
            self._agent_objects.pop(agent_id, None)
            if agent_id in self.agents:
                self.agents.remove(agent_id)
            return True

    parent.environment = _HookedRollbackEnvironment()
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
    assert parent.environment.rollback_calls == ["child_1"]
    parent.get_component("resource").add.assert_called_once_with(5.0)



def _build_p_mode_parent(inheritance_mode: str) -> AgentCore:
    """Build a parent with the given inheritance_mode and warm-start API mocks."""
    parent = _build_parent_agent_for_reproduction()
    parent.environment.intrinsic_evolution_policy = IntrinsicEvolutionPolicy(
        mutation_rate=0.0,
        mutation_scale=0.0,
        inheritance_mode=inheritance_mode,
    )
    parent.behavior = Mock()
    parent.behavior.decision_module = Mock()
    parent.behavior.decision_module.algorithm = Mock()

    def _get_model_state(
        include_optimizer_state=False,
        include_replay_buffer=False,
        replay_buffer_limit=None,
        include_plasticity_state=False,
    ):
        state = {"policy_state_dict": {"w": Mock(shape=(2, 2))}}
        if include_plasticity_state:
            state["plasticity_state"] = {"learning_rate": 0.01, "epsilon": 0.3}
        if include_optimizer_state:
            state["optimizer_state"] = {"optim": {}}
        if include_replay_buffer:
            state["replay_buffer_state"] = {"transitions": [], "size": 0}
        return state

    parent.behavior.decision_module.algorithm.get_model_state = _get_model_state
    return parent


def _build_offspring_mock() -> Mock:
    offspring = Mock()
    offspring.state = Mock()
    offspring.state._state = Mock()
    offspring.state._state.model_copy.return_value = Mock()
    offspring.behavior = Mock()
    offspring.behavior.decision_module = Mock()
    offspring.behavior.decision_module.algorithm = Mock()
    offspring.behavior.decision_module.algorithm.policy = Mock()
    offspring.behavior.decision_module.algorithm.policy.state_dict.return_value = {
        "w": Mock(shape=(2, 2))
    }
    return offspring


def test_reproduce_p2_mode_applies_warmstart_and_tracks_counter():
    """P2 mode should invoke warm-start and record an applied outcome."""
    parent = _build_p_mode_parent("p2")
    offspring = _build_offspring_mock()

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    telemetry = parent.environment.inheritance_telemetry
    assert telemetry.lamarckian_warmstart_applied == 1
    assert telemetry.lamarckian_warmstart_skipped == 0
    offspring.behavior.decision_module.algorithm.load_model_state.assert_called_once()


def test_reproduce_p3_mode_applies_warmstart_and_tracks_counter():
    """P3 mode should invoke warm-start and record an applied outcome."""
    parent = _build_p_mode_parent("p3")
    offspring = _build_offspring_mock()

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    telemetry = parent.environment.inheritance_telemetry
    assert telemetry.lamarckian_warmstart_applied == 1
    assert telemetry.lamarckian_warmstart_skipped == 0
    offspring.behavior.decision_module.algorithm.load_model_state.assert_called_once()


def test_reproduce_p4_mode_applies_warmstart_when_gate_cleared():
    """P4 mode with sufficient resources should apply warm-start."""
    parent = _build_p_mode_parent("p4")
    # Give the parent enough resources to clear the gate.
    parent.resource_level = 100.0
    offspring = _build_offspring_mock()

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    telemetry = parent.environment.inheritance_telemetry
    assert telemetry.lamarckian_warmstart_applied == 1
    assert telemetry.lamarckian_warmstart_skipped == 0
    offspring.behavior.decision_module.algorithm.load_model_state.assert_called_once()


def test_reproduce_p4_mode_skips_when_gate_not_cleared():
    """P4 mode should skip warm-start when the parent fails the fitness gate."""
    from farm.core.policy_inheritance import WARMSTART_REASON_GATE_NOT_CLEARED

    parent = _build_p_mode_parent("p4")
    parent.resource_level = 0.0  # below P4_FITNESS_GATE_MIN_RESOURCES
    offspring = _build_offspring_mock()

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    telemetry = parent.environment.inheritance_telemetry
    assert telemetry.lamarckian_warmstart_applied == 0
    assert telemetry.lamarckian_warmstart_skipped == 1
    assert telemetry.lamarckian_warmstart_skipped_reasons[WARMSTART_REASON_GATE_NOT_CLEARED] == 1
    offspring.behavior.decision_module.algorithm.load_model_state.assert_not_called()


def test_reproduce_baldwinian_mode_never_calls_warmstart():
    """Baldwinian mode must not touch policy inheritance at all."""
    parent = _build_p_mode_parent("baldwinian")
    offspring = _build_offspring_mock()

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    telemetry = parent.environment.inheritance_telemetry
    assert telemetry.lamarckian_warmstart_applied == 0
    assert telemetry.lamarckian_warmstart_skipped == 0
    offspring.behavior.decision_module.algorithm.load_model_state.assert_not_called()


def test_reproduce_warmstart_failure_is_non_fatal():
    """A raising warm-start must not fail reproduction; it records a skip.

    This guards the contract that policy warm-start is best-effort: a failure
    in the (P2) payload load leaves a cold-start child rather than aborting the
    birth event and refunding the parent's resources.
    """
    from farm.core.policy_inheritance import WARMSTART_REASON_LOAD_FAILED

    parent = _build_p_mode_parent("p2")
    offspring = _build_offspring_mock()
    offspring.behavior.decision_module.algorithm.load_model_state.side_effect = (
        RuntimeError("boom")
    )

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    telemetry = parent.environment.inheritance_telemetry
    assert telemetry.lamarckian_warmstart_applied == 0
    assert telemetry.lamarckian_warmstart_skipped == 1
    assert telemetry.lamarckian_warmstart_skipped_reasons[WARMSTART_REASON_LOAD_FAILED] == 1


def test_reproduce_extended_state_unsupported_is_observable():
    """A P3 run on a kwargless algorithm records EXTENDED_STATE_UNSUPPORTED.

    Rather than silently downgrading to a weights-only (P1) transfer, the
    outcome is attributed so a misconfigured arm shows up as all-skipped.
    """
    from farm.core.policy_inheritance import (
        WARMSTART_REASON_EXTENDED_STATE_UNSUPPORTED,
    )

    parent = _build_p_mode_parent("p3")

    def _kwargless_get_model_state():
        return {"policy_state_dict": {"w": Mock(shape=(2, 2))}}

    parent.behavior.decision_module.algorithm.get_model_state = (
        _kwargless_get_model_state
    )
    offspring = _build_offspring_mock()

    with patch("farm.core.agent.factory.AgentFactory") as factory_cls:
        factory = factory_cls.return_value
        factory.create_learning_agent.return_value = offspring
        success = AgentCore.reproduce(parent)

    assert success is True
    telemetry = parent.environment.inheritance_telemetry
    assert telemetry.lamarckian_warmstart_applied == 0
    assert telemetry.lamarckian_warmstart_skipped == 1
    assert (
        telemetry.lamarckian_warmstart_skipped_reasons[
            WARMSTART_REASON_EXTENDED_STATE_UNSUPPORTED
        ]
        == 1
    )
    offspring.behavior.decision_module.algorithm.load_model_state.assert_not_called()
