"""Tests for farm.database.scope_utils and AnalysisScope helpers."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.database.enums import AnalysisScope
from farm.database.models import ActionModel, AgentModel, Base
from farm.database.scope_utils import filter_scope


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


def test_analysis_scope_from_string_case_insensitive():
    assert AnalysisScope.from_string("SIMULATION") == AnalysisScope.SIMULATION


def test_analysis_scope_from_string_invalid():
    with pytest.raises(ValueError, match="Invalid scope"):
        AnalysisScope.from_string("not_a_scope")


def _action_query(session):
    return (
        session.query(ActionModel)
        .join(AgentModel, ActionModel.agent_id == AgentModel.agent_id)
        .order_by(ActionModel.step_number, ActionModel.agent_id)
    )


def test_filter_scope_step_requires_step(session):
    q = _action_query(session)
    with pytest.raises(ValueError, match="step is required"):
        filter_scope(q, AnalysisScope.STEP)


def test_filter_scope_step_range_requires_range(session):
    q = _action_query(session)
    with pytest.raises(ValueError, match="step_range is required"):
        filter_scope(q, AnalysisScope.STEP_RANGE)


def test_filter_scope_agent_random_raises_when_empty(session):
    q = _action_query(session)
    with pytest.raises(ValueError, match="No agents found"):
        filter_scope(q, AnalysisScope.AGENT)


def test_filter_scope_agent_random_picks_existing(session):
    session.add(
        AgentModel(
            agent_id="a1",
            birth_time=0,
            agent_type="t",
            position_x=0.0,
            position_y=0.0,
            initial_resources=1.0,
            starting_health=1.0,
        )
    )
    session.add(
        ActionModel(
            step_number=1,
            agent_id="a1",
            action_type="move",
        )
    )
    session.commit()
    q = _action_query(session)
    filtered = filter_scope(q, AnalysisScope.AGENT)
    assert filtered.count() == 1


def test_filter_scope_agent_with_id_filters(session):
    session.add(
        AgentModel(
            agent_id="a1",
            birth_time=0,
            agent_type="t",
            position_x=0.0,
            position_y=0.0,
            initial_resources=1.0,
            starting_health=1.0,
        )
    )
    session.add(
        ActionModel(
            step_number=1,
            agent_id="a1",
            action_type="move",
        )
    )
    session.commit()
    q = _action_query(session)
    filtered = filter_scope(q, AnalysisScope.AGENT, agent_id="a1")
    assert filtered.count() == 1


def test_filter_scope_step_and_range(session):
    session.add(
        AgentModel(
            agent_id="a1",
            birth_time=0,
            agent_type="t",
            position_x=0.0,
            position_y=0.0,
            initial_resources=1.0,
            starting_health=1.0,
        )
    )
    for step in (1, 2, 5):
        session.add(
            ActionModel(
                step_number=step,
                agent_id="a1",
                action_type="move",
            )
        )
    session.commit()
    base = _action_query(session)
    step_q = filter_scope(base, AnalysisScope.STEP, step=5)
    assert step_q.count() == 1

    s2 = _action_query(session)
    range_q = filter_scope(s2, AnalysisScope.STEP_RANGE, step_range=(1, 2))
    assert range_q.count() == 2


def test_filter_scope_simulation_and_episode_unchanged(session):
    session.add(
        AgentModel(
            agent_id="a1",
            birth_time=0,
            agent_type="t",
            position_x=0.0,
            position_y=0.0,
            initial_resources=1.0,
            starting_health=1.0,
        )
    )
    session.add(
        ActionModel(
            step_number=1,
            agent_id="a1",
            action_type="move",
        )
    )
    session.commit()
    q = _action_query(session)
    assert filter_scope(q, AnalysisScope.SIMULATION).count() == 1
    q2 = _action_query(session)
    assert filter_scope(q2, AnalysisScope.EPISODE).count() == 1


def test_filter_scope_step_requires_step_column_on_query(session):
    q = session.query(AgentModel)
    with pytest.raises(ValueError, match="Cannot apply step scope"):
        filter_scope(q, AnalysisScope.STEP, step=1)


def test_filter_scope_string_scope_converted(session):
    session.add(
        AgentModel(
            agent_id="a1",
            birth_time=0,
            agent_type="t",
            position_x=0.0,
            position_y=0.0,
            initial_resources=1.0,
            starting_health=1.0,
        )
    )
    session.add(
        ActionModel(
            step_number=1,
            agent_id="a1",
            action_type="move",
        )
    )
    session.commit()
    q = _action_query(session)
    filtered = filter_scope(q, "agent", agent_id="a1")
    assert filtered.count() == 1
