import os
import time
from datetime import datetime

import sqlalchemy

from farm.database.database import SimulationDatabase
from farm.database.models import AgentModel, AgentStateModel, Simulation


def _make_db(tmp_name: str = None) -> tuple[SimulationDatabase, str]:
    db_path = tmp_name or f"test_state_updates_{time.time()}.db"
    db = SimulationDatabase(db_path, simulation_id="sim_test")
    # Ensure a simulation record exists for FK
    db.add_simulation_record(
        simulation_id="sim_test",
        start_time=datetime.now(),
        status="running",
        parameters={},
    )
    return db, db_path


def test_update_agent_state_persists_simulation_id_and_columns():
    db, path = _make_db()
    try:
        # Seed an agent (required for FK on agent_states.agent_id)
        def _insert_agent(session):
            agent = AgentModel(
                simulation_id="sim_test",
                agent_id="agent_a",
                birth_time=0,
                agent_type="BaseAgent",
                position_x=0.0,
                position_y=0.0,
                initial_resources=10.0,
                starting_health=100.0,
                starvation_counter=0,
                genome_id="g1",
                generation=0,
            )
            session.add(agent)

        db._execute_in_transaction(_insert_agent)

        # Update agent state
        state = {
            "current_health": 95.0,
            "starting_health": 100.0,
            "resource_level": 5.0,
            "position": (1.0, 2.0),
            "is_defending": False,
            "total_reward": 1.5,
            "starvation_counter": 3,
        }
        db.update_agent_state("agent_a", 1, state)

        # Verify persisted row includes simulation_id and new columns
        def _query_state(session):
            return (
                session.query(AgentStateModel)
                .filter(AgentStateModel.agent_id == "agent_a")
                .filter(AgentStateModel.step_number == 1)
                .one()
            )

        row = db._execute_in_transaction(_query_state)
        assert row.simulation_id == "sim_test"
        assert row.current_health == 95.0
        assert row.starting_health == 100.0
        assert row.starvation_counter == 3
        assert row.resource_level == 5.0
        assert row.position_x == 1.0 and row.position_y == 2.0

    finally:
        db.close()
        if os.path.exists(path):
            os.remove(path)


def test_agent_state_as_dict_includes_new_keys():
    # Construct a model instance directly and ensure keys present
    row = AgentStateModel(
        id="a-1",
        simulation_id="sim_test",
        step_number=1,
        agent_id="a",
        position_x=0.0,
        position_y=0.0,
        position_z=0.0,
        resource_level=1.0,
        current_health=50.0,
        starting_health=100.0,
        starvation_counter=2,
        is_defending=False,
        total_reward=0.0,
        age=1,
    )
    d = row.as_dict()
    assert "starting_health" in d
    assert "starvation_counter" in d
    assert d["starting_health"] == 100.0
    assert d["starvation_counter"] == 2

