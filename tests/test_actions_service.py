from unittest.mock import Mock, patch

import pytest

from farm.database.data_types import (
    ActionMetrics,
    AgentActionData,
    BehaviorClustering,
    CausalAnalysis,
    DecisionPatterns,
    ResourceImpact,
    SequencePattern,
    TimePattern,
)
from farm.database.enums import AnalysisScope
from farm.database.repositories.action_repository import ActionRepository
from farm.database.services.actions_service import ActionsService


@pytest.fixture
def mock_action_repository():
    repository = Mock(spec=ActionRepository)

    # Mock sample actions data
    sample_actions = [
        AgentActionData(
            agent_id=1,
            action_type="gather",
            step_number=1,
            action_target_id=None,
            resources_before={"food": 10},
            resources_after={"food": 15},
            state_before_id=1,
            state_after_id=2,
            reward=5,
            details=None,
        ),
        AgentActionData(
            agent_id=1,
            action_type="share",
            step_number=2,
            action_target_id=2,
            resources_before={"food": 15},
            resources_after={"food": 10},
            state_before_id=2,
            state_after_id=3,
            reward=2,
            details=None,
        ),
    ]

    repository.get_actions_by_scope.return_value = sample_actions
    return repository


@pytest.fixture
def actions_service(mock_action_repository):
    return ActionsService(mock_action_repository)


def test_analyze_actions_all_types(actions_service):
    """Test analyzing actions with all analysis types"""

    # Create mock analyzers with Mock objects
    actions_service.stats_analyzer = Mock()
    actions_service.behavior_analyzer = Mock()
    actions_service.causal_analyzer = Mock()
    actions_service.decision_analyzer = Mock()
    actions_service.resource_analyzer = Mock()
    actions_service.sequence_analyzer = Mock()
    actions_service.temporal_analyzer = Mock()

    # Mock the individual analyzer results
    actions_service.stats_analyzer.analyze.return_value = [
        ActionMetrics(
            action_type="gather",
            count=1,
            frequency=0.5,
            avg_reward=5.0,
            min_reward=5.0,
            max_reward=5.0,
            variance_reward=0,
            std_dev_reward=0,
            median_reward=5.0,
            quartiles_reward=[5.0, 5.0],
            confidence_interval=0,
            interaction_rate=0,
            solo_performance=5.0,
            interaction_performance=0,
            temporal_patterns=[],
            resource_impacts=[],
            decision_patterns=[],
        )
    ]

    actions_service.behavior_analyzer.analyze.return_value = BehaviorClustering(
        clusters=[],
        cluster_characteristics={},
        cluster_performance={}
    )

    actions_service.causal_analyzer.analyze.return_value = CausalAnalysis(
        action_type="gather",
        causes=[],
        effects={}
    )

    actions_service.decision_analyzer.analyze.return_value = DecisionPatterns(
        decision_patterns=[]
    )

    actions_service.resource_analyzer.analyze.return_value = [
        ResourceImpact(
            action_type="gather", resource_changes={}, resource_efficiency=1.0
        )
    ]

    actions_service.sequence_analyzer.analyze.return_value = [
        SequencePattern(pattern=[], frequency=0, avg_reward=0)
    ]

    actions_service.temporal_analyzer.analyze.return_value = [
        TimePattern(action_type="gather", temporal_distribution={}, periodicity=0)
    ]

    results = actions_service.analyze_actions(
        scope=AnalysisScope.SIMULATION, agent_id=1
    )

    # Verify all analysis types are present in results
    assert "action_stats" in results
    assert "behavior_clusters" in results
    assert "causal_analysis" in results
    assert "decision_patterns" in results
    assert "resource_impacts" in results
    assert "sequence_patterns" in results
    assert "temporal_patterns" in results


def test_analyze_actions_specific_types(actions_service):
    """Test analyzing actions with specific analysis types"""

    actions_service.stats_analyzer.analyze.return_value = [
        ActionMetrics(
            action_type="gather",
            count=1,
            frequency=0.5,
            avg_reward=5.0,
            min_reward=5.0,
            max_reward=5.0,
            variance_reward=0,
            std_dev_reward=0,
            median_reward=5.0,
            quartiles_reward=[5.0, 5.0],
            confidence_interval=0,
            interaction_rate=0,
            solo_performance=5.0,
            interaction_performance=0,
            temporal_patterns=[],
            resource_impacts=[],
            decision_patterns=[],
        )
    ]

    results = actions_service.analyze_actions(
        scope=AnalysisScope.SIMULATION, agent_id=1, analysis_types=["stats"]
    )

    # Verify only requested analysis type is present
    assert "action_stats" in results
    assert "behavior_clusters" not in results
    assert len(results) == 1




def test_get_action_summary(actions_service):
    """Test getting action summary"""

    # Mock the analyzers
    actions_service.stats_analyzer = Mock()
    actions_service.resource_analyzer = Mock()

    # Mock analyzer results
    actions_service.stats_analyzer.analyze.return_value = [
        ActionMetrics(
            action_type="gather",
            count=2,
            frequency=0.5,
            avg_reward=5.0,
            min_reward=5.0,
            max_reward=5.0,
            variance_reward=0,
            std_dev_reward=0,
            median_reward=5.0,
            quartiles_reward=[5.0, 5.0],
            confidence_interval=0,
            interaction_rate=0,
            solo_performance=5.0,
            interaction_performance=0,
            temporal_patterns=[],
            resource_impacts=[],
            decision_patterns=[],
            rewards=[5.0, 5.0]
        )
    ]

    actions_service.resource_analyzer.analyze.return_value = [
        ResourceImpact(
            action_type="gather",
            resource_efficiency=1.0
        )
    ]

    summary = actions_service.get_action_summary(
        scope=AnalysisScope.SIMULATION, agent_id=1
    )

    assert "gather" in summary
    assert "success_rate" in summary["gather"]
    assert "avg_reward" in summary["gather"]
    assert "frequency" in summary["gather"]
    assert "resource_efficiency" in summary["gather"]

    # Verify the values
    assert summary["gather"]["success_rate"] == 1.0  # Since all rewards > 0
    assert summary["gather"]["avg_reward"] == 5.0
    assert summary["gather"]["frequency"] == 0.5
    assert summary["gather"]["resource_efficiency"] == 1.0


def test_get_unique_action_types(actions_service, mock_action_repository):
    """Test getting unique action types"""

    action_types = actions_service._get_unique_action_types(
        scope=AnalysisScope.SIMULATION, agent_id=1
    )

    assert isinstance(action_types, list)
    assert set(action_types) == {"gather", "share"}
    mock_action_repository.get_actions_by_scope.assert_called_once()
