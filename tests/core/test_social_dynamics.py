"""Tests for farm.core.social_dynamics SQL aggregation helpers."""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.core.social_dynamics import (
    compute_social_dynamics_trends,
    social_dynamics_per_step,
)
from farm.database.models import ActionModel, AgentModel, Base


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


def test_social_dynamics_per_step_rates(session):
    session.add(
        AgentModel(
            agent_id="a1",
            birth_time=0,
            agent_type="system",
            position_x=0.0,
            position_y=0.0,
            initial_resources=1.0,
            starting_health=1.0,
        )
    )
    session.add(
        AgentModel(
            agent_id="a2",
            birth_time=0,
            agent_type="independent",
            position_x=1.0,
            position_y=1.0,
            initial_resources=1.0,
            starting_health=1.0,
        )
    )
    # Step 1: 1 share + 1 attack => total 2, cooperation_rate 0.5, competition 0.5
    session.add(
        ActionModel(
            step_number=1,
            agent_id="a1",
            action_type="share",
            action_target_id="a2",
        )
    )
    session.add(
        ActionModel(
            step_number=1,
            agent_id="a2",
            action_type="attack",
            action_target_id="a1",
        )
    )
    # Step 2: 1 assist only => cooperation_rate 1.0, competition 0
    session.add(
        ActionModel(
            step_number=2,
            agent_id="a1",
            action_type="assist",
            action_target_id="a2",
        )
    )
    session.commit()

    df = social_dynamics_per_step(session, simulation_id=None)
    assert len(df) == 2
    row1 = df.loc[df["step"] == 1].iloc[0]
    assert int(row1["total_social_interactions"]) == 2
    assert float(row1["cooperation_rate"]) == pytest.approx(0.5)
    assert float(row1["competition_intensity"]) == pytest.approx(0.5)

    row2 = df.loc[df["step"] == 2].iloc[0]
    assert int(row2["total_social_interactions"]) == 1
    assert float(row2["cooperation_rate"]) == pytest.approx(1.0)
    assert float(row2["competition_intensity"]) == pytest.approx(0.0)


def test_social_dynamics_filters_simulation_id(session):
    session.add(
        AgentModel(
            agent_id="a1",
            birth_time=0,
            agent_type="system",
            position_x=0.0,
            position_y=0.0,
            initial_resources=1.0,
            starting_health=1.0,
        )
    )
    session.add(
        AgentModel(
            agent_id="a2",
            birth_time=0,
            agent_type="system",
            position_x=0.0,
            position_y=0.0,
            initial_resources=1.0,
            starting_health=1.0,
        )
    )
    session.add(
        ActionModel(
            simulation_id="sim-a",
            step_number=1,
            agent_id="a1",
            action_type="share",
            action_target_id="a2",
        )
    )
    session.add(
        ActionModel(
            simulation_id="sim-b",
            step_number=1,
            agent_id="a1",
            action_type="attack",
            action_target_id="a2",
        )
    )
    session.commit()

    df_a = social_dynamics_per_step(session, simulation_id="sim-a")
    assert len(df_a) == 1
    assert int(df_a.iloc[0]["cooperation_actions"]) == 1
    assert int(df_a.iloc[0]["attack_events"]) == 0


def test_compute_social_dynamics_trends_slope():
    df = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "cooperation_rate": [0.0, 0.5, 1.0],
            "competition_intensity": [0.3, 0.2, 0.1],
        }
    )
    trends = compute_social_dynamics_trends(df)
    assert "cooperation_rate_trend_slope" in trends
    assert trends["cooperation_rate_trend_slope"] > 0
    assert "competition_intensity_trend_slope" in trends
    assert trends["competition_intensity_trend_slope"] < 0
    assert "combat_escalation_mean_delta" in trends
