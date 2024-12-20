import unittest
from unittest.mock import MagicMock
from database.services.actions_service import ActionsService
from database.analyzers.action_stats_analyzer import ActionStatsAnalyzer
from database.analyzers.behavior_clustering_analyzer import BehaviorClusteringAnalyzer
from database.analyzers.causal_analyzer import CausalAnalyzer
from database.analyzers.decision_pattern_analyzer import DecisionPatternAnalyzer
from database.analyzers.resource_impact_analyzer import ResourceImpactAnalyzer
from database.analyzers.sequence_pattern_analyzer import SequencePatternAnalyzer
from database.analyzers.temporal_pattern_analyzer import TemporalPatternAnalyzer
from database.repositories.action_repository import ActionRepository

class TestActionsService(unittest.TestCase):
    def setUp(self):
        # Create mocks for all dependencies
        self.action_repo = MagicMock(spec=ActionRepository)
        self.action_stats = MagicMock(spec=ActionStatsAnalyzer)
        self.behavior_clustering = MagicMock(spec=BehaviorClusteringAnalyzer)
        self.causal = MagicMock(spec=CausalAnalyzer)
        self.decision_pattern = MagicMock(spec=DecisionPatternAnalyzer)
        self.resource_impact = MagicMock(spec=ResourceImpactAnalyzer)
        self.sequence_pattern = MagicMock(spec=SequencePatternAnalyzer)
        self.temporal_pattern = MagicMock(spec=TemporalPatternAnalyzer)

        # Initialize service with mocked dependencies
        self.service = ActionsService(
            action_repo=self.action_repo,
            action_stats_analyzer=self.action_stats,
            behavior_clustering_analyzer=self.behavior_clustering,
            causal_analyzer=self.causal,
            decision_pattern_analyzer=self.decision_pattern,
            resource_impact_analyzer=self.resource_impact,
            sequence_pattern_analyzer=self.sequence_pattern,
            temporal_pattern_analyzer=self.temporal_pattern
        )

    def test_analyze_agent_actions(self):
        """
        Tests the main analyze_agent_actions method to ensure it properly coordinates 
        all analyzer components and processes agent actions correctly.
        """
        # Arrange
        agent_id = "test_agent"
        mock_actions = [
            {"id": 1, "type": "move", "timestamp": "2023-01-01T00:00:00Z"},
            {"id": 2, "type": "gather", "timestamp": "2023-01-01T00:01:00Z"}
        ]
        self.action_repo.get_actions_for_agent.return_value = mock_actions
        
        # Mock return values from analyzers
        expected_stats = {"total_actions": 2}
        expected_clusters = {"clusters": [["move", "gather"]]}
        expected_causal = {"causes": {"move": ["gather"]}}
        expected_decisions = {"patterns": ["resource_gathering"]}
        expected_impact = {"resources": {"wood": 10}}
        expected_sequences = {"common": ["move-gather"]}
        expected_temporal = {"daily": {"morning": ["gather"]}}
        
        self.action_stats.analyze.return_value = expected_stats
        self.behavior_clustering.analyze.return_value = expected_clusters
        self.causal.analyze.return_value = expected_causal
        self.decision_pattern.analyze.return_value = expected_decisions
        self.resource_impact.analyze.return_value = expected_impact
        self.sequence_pattern.analyze.return_value = expected_sequences
        self.temporal_pattern.analyze.return_value = expected_temporal

        # Act
        result = self.service.analyze_agent_actions(agent_id)

        # Assert
        self.assertEqual(result["statistics"], expected_stats)
        self.assertEqual(result["clusters"], expected_clusters)
        self.assertEqual(result["causal"], expected_causal)
        self.assertEqual(result["decisions"], expected_decisions)
        self.assertEqual(result["impact"], expected_impact)
        self.assertEqual(result["sequences"], expected_sequences)
        self.assertEqual(result["temporal"], expected_temporal)
        
        # Verify method calls
        self.action_repo.get_actions_for_agent.assert_called_once_with(agent_id)
        self.action_stats.analyze.assert_called_once_with(mock_actions)
        self.behavior_clustering.analyze.assert_called_once_with(mock_actions)
        self.causal.analyze.assert_called_once_with(mock_actions)
        self.decision_pattern.analyze.assert_called_once_with(mock_actions)
        self.resource_impact.analyze.assert_called_once_with(mock_actions)
        self.sequence_pattern.analyze.assert_called_once_with(mock_actions)
        self.temporal_pattern.analyze.assert_called_once_with(mock_actions)

    def test_analyze_agent_actions_empty(self):
        """
        Tests handling of empty action lists to ensure the service behaves correctly
        when no actions are found for an agent.
        """
        # Arrange
        agent_id = "test_agent"
        self.action_repo.get_actions_for_agent.return_value = []
        
        expected_empty_stats = {"total_actions": 0}
        self.action_stats.analyze.return_value = expected_empty_stats

        # Act
        result = self.service.analyze_agent_actions(agent_id)

        # Assert
        self.assertEqual(result["statistics"], expected_empty_stats)
        self.assertTrue(all(len(value) == 0 for value in result.values() if isinstance(value, (list, dict))))
        self.action_repo.get_actions_for_agent.assert_called_once_with(agent_id)
        self.action_stats.analyze.assert_called_once_with([])

    def test_get_action_statistics(self):
        """
        Tests the retrieval and analysis of action statistics for an agent.
        Verifies that the service correctly fetches actions and returns processed statistics.
        """
        # Arrange
        agent_id = "test_agent"
        mock_actions = [{"id": 1, "type": "move"}]
        mock_stats = {"total_actions": 1, "action_types": {"move": 1}}
        self.action_repo.get_actions_for_agent.return_value = mock_actions
        self.action_stats.analyze.return_value = mock_stats

        # Act
        result = self.service.get_action_statistics(agent_id)

        # Assert
        self.assertEqual(result, mock_stats)
        self.action_repo.get_actions_for_agent.assert_called_once_with(agent_id)
        self.action_stats.analyze.assert_called_once_with(mock_actions)

    def test_get_behavior_clusters(self):
        """
        Tests the behavior clustering functionality to ensure proper grouping of related actions.
        Verifies that the clustering analyzer processes the actions and returns meaningful clusters.
        """
        # Arrange
        agent_id = "test_agent"
        mock_actions = [{"id": 1, "type": "move"}]
        mock_clusters = {"clusters": [["move", "gather"], ["fight", "flee"]]}
        self.action_repo.get_actions_for_agent.return_value = mock_actions
        self.behavior_clustering.analyze.return_value = mock_clusters

        # Act
        result = self.service.get_behavior_clusters(agent_id)

        # Assert
        self.assertEqual(result, mock_clusters)
        self.action_repo.get_actions_for_agent.assert_called_once_with(agent_id)
        self.behavior_clustering.analyze.assert_called_once_with(mock_actions)

    def test_get_causal_relationships(self):
        """
        Tests the identification of cause-and-effect relationships between actions.
        Verifies that the causal analyzer correctly processes action sequences and identifies dependencies.
        """
        # Arrange
        agent_id = "test_agent"
        mock_actions = [{"id": 1, "type": "move"}]
        mock_relationships = {"causes": {"move": ["gather"]}}
        self.action_repo.get_actions_for_agent.return_value = mock_actions
        self.causal.analyze.return_value = mock_relationships

        # Act
        result = self.service.get_causal_relationships(agent_id)

        # Assert
        self.assertEqual(result, mock_relationships)
        self.action_repo.get_actions_for_agent.assert_called_once_with(agent_id)
        self.causal.analyze.assert_called_once_with(mock_actions)

    def test_error_handling(self):
        """
        Tests the service's error handling capabilities when database operations fail.
        Verifies that exceptions are properly propagated and not silently caught.
        """
        # Arrange
        agent_id = "test_agent"
        self.action_repo.get_actions_for_agent.side_effect = Exception("Database error")

        # Act & Assert
        with self.assertRaises(Exception):
            self.service.analyze_agent_actions(agent_id)

    def test_get_decision_patterns(self):
        """
        Tests the analysis of decision-making patterns in agent behavior.
        Verifies that the decision pattern analyzer correctly identifies and returns common decision patterns.
        """
        # Arrange
        agent_id = "test_agent"
        mock_actions = [{"id": 1, "type": "move", "decision_factors": {"threat_level": 0.8}}]
        mock_patterns = {"common_patterns": ["flee_when_threatened", "gather_when_safe"]}
        self.action_repo.get_actions_for_agent.return_value = mock_actions
        self.decision_pattern.analyze.return_value = mock_patterns

        # Act
        result = self.service.get_decision_patterns(agent_id)

        # Assert
        self.assertEqual(result, mock_patterns)
        self.action_repo.get_actions_for_agent.assert_called_once_with(agent_id)
        self.decision_pattern.analyze.assert_called_once_with(mock_actions)

    def test_get_resource_impact(self):
        """
        Tests the analysis of how actions affect resource levels.
        Verifies that the resource impact analyzer correctly calculates and returns resource changes.
        """
        # Arrange
        agent_id = "test_agent"
        mock_actions = [{"id": 1, "type": "gather", "resource_impact": {"wood": 10}}]
        mock_impact = {"resource_changes": {"wood": 100, "food": -50}}
        self.action_repo.get_actions_for_agent.return_value = mock_actions
        self.resource_impact.analyze.return_value = mock_impact

        # Act
        result = self.service.get_resource_impact(agent_id)

        # Assert
        self.assertEqual(result, mock_impact)
        self.action_repo.get_actions_for_agent.assert_called_once_with(agent_id)
        self.resource_impact.analyze.assert_called_once_with(mock_actions)

    def test_get_sequence_patterns(self):
        """
        Tests the identification of common action sequences in agent behavior.
        Verifies that the sequence pattern analyzer correctly identifies and returns recurring action patterns.
        """
        # Arrange
        agent_id = "test_agent"
        mock_actions = [
            {"id": 1, "type": "scout"},
            {"id": 2, "type": "gather"},
            {"id": 3, "type": "return"}
        ]
        mock_sequences = {"common_sequences": [["scout", "gather", "return"]]}
        self.action_repo.get_actions_for_agent.return_value = mock_actions
        self.sequence_pattern.analyze.return_value = mock_sequences

        # Act
        result = self.service.get_sequence_patterns(agent_id)

        # Assert
        self.assertEqual(result, mock_sequences)
        self.action_repo.get_actions_for_agent.assert_called_once_with(agent_id)
        self.sequence_pattern.analyze.assert_called_once_with(mock_actions)

    def test_get_temporal_patterns(self):
        """
        Tests the analysis of time-based patterns in agent actions.
        Verifies that the temporal pattern analyzer correctly identifies patterns based on timestamps.
        """
        # Arrange
        agent_id = "test_agent"
        mock_actions = [
            {"id": 1, "type": "gather", "timestamp": "2023-01-01T08:00:00Z"},
            {"id": 2, "type": "rest", "timestamp": "2023-01-01T20:00:00Z"}
        ]
        mock_patterns = {"daily_patterns": {"morning": ["gather"], "night": ["rest"]}}
        self.action_repo.get_actions_for_agent.return_value = mock_actions
        self.temporal_pattern.analyze.return_value = mock_patterns

        # Act
        result = self.service.get_temporal_patterns(agent_id)

        # Assert
        self.assertEqual(result, mock_patterns)
        self.action_repo.get_actions_for_agent.assert_called_once_with(agent_id)
        self.temporal_pattern.analyze.assert_called_once_with(mock_actions)

    def test_analyze_agent_actions_with_date_range(self):
        """
        Tests the analysis of actions within a specific date range.
        """
        # Arrange
        agent_id = "test_agent"
        start_date = "2023-01-01T00:00:00Z"
        end_date = "2023-01-02T00:00:00Z"
        mock_actions = [{"id": 1, "type": "move", "timestamp": "2023-01-01T12:00:00Z"}]
        self.action_repo.get_actions_for_agent_in_range.return_value = mock_actions

        expected_stats = {"total_actions": 1}
        expected_temporal = {"hourly_distribution": {"12": 1}}
        
        self.action_stats.analyze.return_value = expected_stats
        self.temporal_pattern.analyze.return_value = expected_temporal

        # Act
        result = self.service.analyze_agent_actions(agent_id, start_date, end_date)

        # Assert
        self.assertEqual(result["statistics"], expected_stats)
        self.assertEqual(result["temporal"], expected_temporal)
        self.action_repo.get_actions_for_agent_in_range.assert_called_once_with(
            agent_id, start_date, end_date
        )
        self.action_stats.analyze.assert_called_once_with(mock_actions)

    def test_invalid_agent_id(self):
        """
        Tests input validation for agent IDs.
        Verifies that the service properly handles and rejects invalid agent IDs.
        """
        # Arrange
        agent_id = None

        # Act & Assert
        with self.assertRaises(ValueError):
            self.service.analyze_agent_actions(agent_id)

    def test_invalid_date_range(self):
        """
        Tests validation of date range parameters.
        Verifies that the service properly handles and rejects invalid date ranges.
        """
        # Arrange
        agent_id = "test_agent"
        start_date = "2023-02-01"
        end_date = "2023-01-01"

        # Act & Assert
        with self.assertRaises(ValueError):
            self.service.analyze_agent_actions(agent_id, start_date, end_date)

    def test_multiple_agents_comparison(self):
        """
        Tests the comparison of behaviors between multiple agents.
        """
        # Arrange
        agent_ids = ["agent1", "agent2"]
        mock_actions_1 = [{"id": 1, "type": "move", "agent_id": "agent1"}]
        mock_actions_2 = [{"id": 2, "type": "gather", "agent_id": "agent2"}]
        combined_actions = mock_actions_1 + mock_actions_2
        self.action_repo.get_actions_for_agents.return_value = combined_actions

        expected_comparison = {
            "behavior_clusters": {"cluster1": ["agent1"], "cluster2": ["agent2"]},
            "action_similarities": {"agent1-agent2": 0.5}
        }
        self.behavior_clustering.analyze.return_value = expected_comparison

        # Act
        result = self.service.compare_agent_behaviors(agent_ids)

        # Assert
        self.assertEqual(result, expected_comparison)
        self.action_repo.get_actions_for_agents.assert_called_once_with(agent_ids)
        self.behavior_clustering.analyze.assert_called_once_with(combined_actions)

    def test_action_frequency_analysis(self):
        """
        Tests the calculation of action frequency distributions.
        Verifies that the service correctly counts and categorizes actions by type.
        """
        # Arrange
        agent_id = "test_agent"
        mock_actions = [
            {"id": 1, "type": "move", "timestamp": "2023-01-01T00:00:00Z"},
            {"id": 2, "type": "move", "timestamp": "2023-01-01T00:01:00Z"},
            {"id": 3, "type": "gather", "timestamp": "2023-01-01T00:02:00Z"}
        ]
        expected_frequency = {"move": 2, "gather": 1}
        self.action_repo.get_actions_for_agent.return_value = mock_actions

        # Act
        result = self.service.get_action_frequency(agent_id)

        # Assert
        self.assertEqual(result, expected_frequency)
        self.action_repo.get_actions_for_agent.assert_called_once_with(agent_id)

