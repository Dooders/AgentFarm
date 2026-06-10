"""Shared fixtures for farm/research/analysis tests.

Provides a seeded on-disk simulation database mirroring what
`farm.research.analysis.database` expects to read.
"""

from datetime import datetime

import pytest

from farm.database.database import SimulationDatabase
from farm.database.models import ActionModel, AgentModel, SimulationStepModel

SIM_ID = "sim_research"
NUM_STEPS = 5


def seed_simulation_db(db_path: str) -> None:
    """Create a simulation database populated with steps, agents, and actions."""
    db = SimulationDatabase(str(db_path), simulation_id=SIM_ID)
    db.add_simulation_record(
        simulation_id=SIM_ID, start_time=datetime.now(), status="complete", parameters={}
    )
    session = db.Session()
    try:
        for step in range(NUM_STEPS):
            session.add(
                SimulationStepModel(
                    step_number=step,
                    simulation_id=SIM_ID,
                    total_agents=10 + step,
                    agent_type_counts={"system": 4 + step, "independent": 3, "control": 3},
                    average_agent_resources=5.0 - step,
                    resources_consumed=2.0 * step,
                )
            )
        session.add_all(
            [
                AgentModel(
                    simulation_id=SIM_ID,
                    agent_id="agent_sys",
                    agent_type="system",
                    generation=0,
                ),
                AgentModel(
                    simulation_id=SIM_ID,
                    agent_id="agent_ind",
                    agent_type="independent",
                    generation=1,
                ),
            ]
        )
        session.add_all(
            [
                ActionModel(
                    simulation_id=SIM_ID,
                    step_number=1,
                    agent_id="agent_sys",
                    action_type="move",
                    reward=1.0,
                    module_type="dqn",
                ),
                ActionModel(
                    simulation_id=SIM_ID,
                    step_number=2,
                    agent_id="agent_sys",
                    action_type="gather",
                    reward=3.0,
                    module_type="dqn",
                ),
                ActionModel(
                    simulation_id=SIM_ID,
                    step_number=1,
                    agent_id="agent_ind",
                    action_type="move",
                    reward=2.0,
                    module_type="ppo",
                ),
            ]
        )
        session.commit()
    finally:
        session.close()
        db.close()


@pytest.fixture()
def seeded_db_path(tmp_path):
    """Path to a seeded simulation database named simulation.db."""
    db_path = tmp_path / "simulation.db"
    seed_simulation_db(str(db_path))
    return str(db_path)


@pytest.fixture()
def empty_db_path(tmp_path):
    """Path to a simulation database with schema but no rows."""
    db_path = tmp_path / "simulation.db"
    db = SimulationDatabase(str(db_path), simulation_id=SIM_ID)
    db.close()
    return str(db_path)


@pytest.fixture()
def corrupt_db_path(tmp_path):
    """Path to a file that is not a valid SQLite database."""
    db_path = tmp_path / "simulation.db"
    db_path.write_bytes(b"this is not a sqlite database")
    return str(db_path)


def _make_experiment_layout(tmp_path, monkeypatch, experiment_name):
    """Create the hard-coded experiment layout and chdir into its root.

    The analysis module resolves experiments relative to
    ``results/one_of_a_kind_v1/experiments/data``, so tests run from a
    temporary working directory containing that structure.
    """
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "results" / "one_of_a_kind_v1" / "experiments" / "data"
    iteration_dir = data_dir / experiment_name / "iteration_0"
    iteration_dir.mkdir(parents=True)
    return iteration_dir


@pytest.fixture()
def experiment_dir(tmp_path, monkeypatch):
    """Experiment layout with one seeded simulation database."""
    iteration_dir = _make_experiment_layout(tmp_path, monkeypatch, "test_experiment")
    seed_simulation_db(str(iteration_dir / "simulation.db"))
    return "test_experiment"


@pytest.fixture()
def experiment_dir_without_dbs(tmp_path, monkeypatch):
    """Experiment layout whose iteration directory has no databases."""
    _make_experiment_layout(tmp_path, monkeypatch, "empty_experiment")
    return "empty_experiment"


@pytest.fixture()
def experiment_dir_with_corrupt_db(tmp_path, monkeypatch):
    """Experiment layout containing only a corrupt simulation database."""
    iteration_dir = _make_experiment_layout(tmp_path, monkeypatch, "corrupt_experiment")
    (iteration_dir / "simulation.db").write_bytes(b"not a sqlite database")
    return "corrupt_experiment"
