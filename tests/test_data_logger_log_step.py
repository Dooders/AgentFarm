import os
import time
from datetime import datetime, timezone

from farm.database.database import SimulationDatabase
from farm.database.models import AgentModel, AgentStateModel, SimulationStepModel


def test_log_step_inserts_agent_state_with_new_columns():
    db_path = f"test_log_step_{time.time()}.db"
    db = SimulationDatabase(db_path, simulation_id="simX")
    try:
        # Ensure simulation exists
        db.add_simulation_record(
            simulation_id="simX",
            start_time=datetime.now(timezone.utc),
            status="running",
            parameters={},
        )

        # Seed one agent (FK for agent_states)
        def _insert_agent(session):
            session.add(
                AgentModel(
                    simulation_id="simX",
                    agent_id="a1",
                    birth_time=0,
                    agent_type="BaseAgent",
                    position_x=0.0,
                    position_y=0.0,
                    initial_resources=0.0,
                    starting_health=100.0,
                    genome_id="g",
                    generation=0,
                )
            )

        db._execute_in_transaction(_insert_agent)

        # Prepare agent state tuple per DataLogger.log_step expectations
        agent_states = [
            (
                "a1",      # agent_id
                1.0,        # position_x
                2.0,        # position_y
                5.0,        # resource_level
                80.0,       # current_health
                100.0,      # starting_health
                1,          # starvation_counter
                0,          # is_defending
                0.5,        # total_reward
                1,          # age
            )
        ]

        metrics = {
            "total_agents": 1,
            "system_agents": 0,
            "independent_agents": 1,
            "control_agents": 0,
            "total_resources": 5.0,
            "average_agent_resources": 5.0,
            "births": 0,
            "deaths": 0,
            "current_max_generation": 0,
            "resource_efficiency": 0.0,
            "resource_distribution_entropy": 0.0,
            "average_agent_health": 80.0,
            "average_agent_age": 1,
            "average_reward": 0.5,
            "combat_encounters": 0,
            "successful_attacks": 0,
            "resources_shared": 0.0,
            "resources_shared_this_step": 0.0,
            "combat_encounters_this_step": 0,
            "successful_attacks_this_step": 0,
            "genetic_diversity": 0.0,
            "dominant_genome_ratio": 0.0,
            # resources_consumed default ensured in logger
        }

        db.logger.log_step(
            step_number=1,
            agent_states=agent_states,
            resource_states=[],
            metrics=metrics,
        )

        # Query stored agent state
        def _query(session):
            return (
                session.query(AgentStateModel)
                .filter(AgentStateModel.agent_id == "a1")
                .filter(AgentStateModel.step_number == 1)
                .one()
            )

        st = db._execute_in_transaction(_query)
        assert st.simulation_id == "simX"
        assert st.starting_health == 100.0
        assert st.starvation_counter == 1
        assert st.current_health == 80.0

        # Also ensure a step row exists for the metrics
        def _query_step(session):
            return (
                session.query(SimulationStepModel)
                .filter(SimulationStepModel.step_number == 1)
                .filter(SimulationStepModel.simulation_id == "simX")
                .one()
            )

        step = db._execute_in_transaction(_query_step)
        assert step.total_agents == 1

    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)

