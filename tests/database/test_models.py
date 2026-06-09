"""Tests for farm/database/models.py construction and serialization helpers."""

import pytest

from farm.database.models import (
    AgentStateModel,
    ExperimentModel,
    ResourceModel,
    Simulation,
    SimulationStepModel,
)


class TestAgentStateModelInit:
    def test_id_includes_simulation_id_when_present(self):
        state = AgentStateModel(agent_id="a1", step_number=3, simulation_id="sim_1")
        assert state.id == "sim_1:a1-3"

    def test_id_without_simulation_id(self):
        state = AgentStateModel(agent_id="a1", step_number=3)
        assert state.id == "a1-3"

    def test_explicit_id_is_preserved(self):
        state = AgentStateModel(id="custom-id")
        assert state.id == "custom-id"

    def test_missing_identifiers_raise(self):
        with pytest.raises(ValueError, match="agent_id and step_number"):
            AgentStateModel(step_number=3)

    def test_generate_id_matches_init_format(self):
        assert AgentStateModel.generate_id("a1", 3) == "a1-3"


class TestAsDictMethods:
    def test_agent_state_as_dict(self):
        state = AgentStateModel(
            agent_id="a1",
            step_number=2,
            position_x=1.0,
            position_y=2.0,
            resource_level=5.0,
            current_health=0.9,
            is_defending=False,
            total_reward=1.5,
            age=2,
        )
        result = state.as_dict()
        assert result["agent_id"] == "a1"
        assert result["step_number"] == 2
        assert result["total_reward"] == 1.5

    def test_resource_state_as_dict(self):
        resource = ResourceModel(
            resource_id="resource_1", amount=12.5, position_x=4.0, position_y=5.0
        )
        assert resource.as_dict() == {
            "resource_id": "resource_1",
            "amount": 12.5,
            "position": (4.0, 5.0),
        }

    def test_simulation_step_as_dict_extracts_agent_type_counts(self):
        step = SimulationStepModel(
            total_agents=5,
            agent_type_counts={"system": 2, "independent": 3},
        )
        result = step.as_dict()
        assert result["total_agents"] == 5
        assert result["system_agents"] == 2
        assert result["independent_agents"] == 3

    def test_simulation_step_as_dict_handles_missing_counts(self):
        step = SimulationStepModel(total_agents=0, agent_type_counts=None)
        result = step.as_dict()
        assert result["system_agents"] == 0
        assert result["independent_agents"] == 0


class TestReprs:
    def test_experiment_repr(self):
        experiment = ExperimentModel(experiment_id="exp_1", name="trial", status="running")
        assert "exp_1" in repr(experiment)
        assert "trial" in repr(experiment)

    def test_simulation_repr(self):
        sim = Simulation(simulation_id="sim_1", status="done", parameters={}, simulation_db_path="x.db")
        assert "sim_1" in repr(sim)
        assert "done" in repr(sim)
