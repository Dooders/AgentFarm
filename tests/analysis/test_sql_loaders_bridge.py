"""Bridge tests: shared sql_loaders used by SimulationAnalyzer and population module."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from farm.analysis.sql_loaders import (
    population_dataframe_from_sqlite,
    run_dataframe_on_sqlite,
    survival_rates_from_session,
)
from farm.core.analysis import SimulationAnalyzer
from farm.database.models import (
    AgentModel,
    AgentStateModel,
    Base,
    Simulation,
    SimulationStepModel,
)


@pytest.mark.unit
def test_population_module_matches_legacy_survival_when_json_matches_states(tmp_path):
    """When agent_type_counts matches live agent states, both code paths agree."""
    db_file = tmp_path / "simulation.db"
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)

    sim_id = "sim-bridge-test"
    with Session(engine) as session:
        session.add(
            Simulation(
                simulation_id=sim_id,
                parameters={},
                simulation_db_path=str(db_file),
            )
        )
        session.add(
            SimulationStepModel(
                step_number=0,
                simulation_id=sim_id,
                total_agents=3,
                agent_type_counts={"system": 2, "independent": 1, "control": 0},
                total_resources=30.0,
                average_agent_resources=10.0,
                resource_efficiency=0.5,
                resources_consumed=0.0,
            )
        )
        session.add_all(
            [
                AgentModel(
                    simulation_id=sim_id,
                    agent_id="a-sys-1",
                    birth_time=0,
                    death_time=None,
                    agent_type="system",
                    position_x=0.0,
                    position_y=0.0,
                    initial_resources=5.0,
                    starting_health=100.0,
                    genome_id="g1",
                    generation=0,
                ),
                AgentModel(
                    simulation_id=sim_id,
                    agent_id="a-sys-2",
                    birth_time=0,
                    death_time=None,
                    agent_type="system",
                    position_x=1.0,
                    position_y=0.0,
                    initial_resources=5.0,
                    starting_health=100.0,
                    genome_id="g2",
                    generation=0,
                ),
                AgentModel(
                    simulation_id=sim_id,
                    agent_id="a-ind-1",
                    birth_time=0,
                    death_time=None,
                    agent_type="independent",
                    position_x=0.0,
                    position_y=1.0,
                    initial_resources=5.0,
                    starting_health=100.0,
                    genome_id="g3",
                    generation=0,
                ),
            ]
        )
        session.add_all(
            [
                AgentStateModel(
                    simulation_id=sim_id,
                    step_number=0,
                    agent_id="a-sys-1",
                    position_x=0.0,
                    position_y=0.0,
                    position_z=0.0,
                    resource_level=5.0,
                    current_health=100.0,
                    starting_health=100.0,
                    starvation_counter=0,
                    is_defending=False,
                    total_reward=0.0,
                    age=0,
                ),
                AgentStateModel(
                    simulation_id=sim_id,
                    step_number=0,
                    agent_id="a-sys-2",
                    position_x=1.0,
                    position_y=0.0,
                    position_z=0.0,
                    resource_level=5.0,
                    current_health=100.0,
                    starting_health=100.0,
                    starvation_counter=0,
                    is_defending=False,
                    total_reward=0.0,
                    age=0,
                ),
                AgentStateModel(
                    simulation_id=sim_id,
                    step_number=0,
                    agent_id="a-ind-1",
                    position_x=0.0,
                    position_y=1.0,
                    position_z=0.0,
                    resource_level=5.0,
                    current_health=100.0,
                    starting_health=100.0,
                    starvation_counter=0,
                    is_defending=False,
                    total_reward=0.0,
                    age=0,
                ),
            ]
        )
        session.commit()

    pop_df = population_dataframe_from_sqlite(str(db_file))
    survival_df = run_dataframe_on_sqlite(str(db_file), survival_rates_from_session)
    analyzer_df = SimulationAnalyzer(db_path=str(db_file), simulation_id=sim_id).calculate_survival_rates()

    assert len(pop_df) == 1
    assert pop_df.iloc[0]["system_agents"] == 2
    assert pop_df.iloc[0]["independent_agents"] == 1

    assert len(survival_df) == 1
    assert survival_df.iloc[0]["system_alive"] == 2
    assert survival_df.iloc[0]["independent_alive"] == 1

    assert analyzer_df.equals(survival_df)
