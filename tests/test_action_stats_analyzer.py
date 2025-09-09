import unittest
from unittest.mock import Mock, patch

from farm.database.analyzers.action_stats_analyzer import ActionStatsAnalyzer
from farm.database.data_types import AgentActionData
from farm.database.enums import AnalysisScope
from farm.database.repositories.action_repository import ActionRepository


class TestActionStatsAnalyzer(unittest.TestCase):
    def setUp(self):
        """Initialize test fixtures for ActionStatsAnalyzer tests.

        Sets up a mock ActionRepository and creates sample test actions with various
        types (gather, move) and different reward values to test analysis functionality.
        """
        self.repository = Mock(spec=ActionRepository)
        self.analyzer = ActionStatsAnalyzer(self.repository)

        # Create sample test actions
        self.test_actions = [
            AgentActionData(
                agent_id="1",
                action_type="gather",
                step_number=1,
                action_target_id=None,
                resources_before=None,
                resources_after=None,
                state_before_id=None,
                state_after_id=None,
                reward=10.0,
                details=None,
            ),
            AgentActionData(
                agent_id="1",
                action_type="gather",
                step_number=1,
                action_target_id="2",
                resources_before=None,
                resources_after=None,
                state_before_id=None,
                state_after_id=None,
                reward=5.0,
                details=None,
            ),
            AgentActionData(
                agent_id="1",
                action_type="move",
                step_number=2,
                action_target_id=None,
                resources_before=None,
                resources_after=None,
                state_before_id=None,
                state_after_id=None,
                reward=2.0,
                details=None,
            ),
        ]

    def test_analyze_basic_metrics(self):
        """Test the calculation of basic action metrics.

        Verifies that the analyzer correctly calculates:
        - Action counts per type
        - Action frequencies
        - Average rewards

        Uses sample data with 'gather' and 'move' actions to validate metrics.
        """
        # Arrange
        self.repository.get_actions_by_scope.return_value = self.test_actions

        # Act
        results = self.analyzer.analyze(scope=AnalysisScope.SIMULATION)

        # Assert
        self.assertEqual(len(results), 2)  # Should have metrics for 'gather' and 'move'

        # Find gather metrics
        gather_metrics = next(m for m in results if m.action_type == "gather")
        self.assertEqual(gather_metrics.count, 2)
        self.assertAlmostEqual(gather_metrics.frequency, 2 / 3)
        self.assertAlmostEqual(gather_metrics.avg_reward, 7.5)

        # Find move metrics
        move_metrics = next(m for m in results if m.action_type == "move")
        self.assertEqual(move_metrics.count, 1)
        self.assertAlmostEqual(move_metrics.frequency, 1 / 3)
        self.assertEqual(move_metrics.avg_reward, 2.0)

    def test_analyze_interaction_metrics(self):
        """Test the calculation of interaction-related metrics.

        Verifies metrics specific to action interactions:
        - Interaction rate (actions with targets vs. without)
        - Solo performance (rewards for non-interactive actions)
        - Interaction performance (rewards for actions with targets)

        Uses gather actions with and without targets for validation.
        """
        # Arrange
        self.repository.get_actions_by_scope.return_value = self.test_actions

        # Act
        results = self.analyzer.analyze(scope=AnalysisScope.SIMULATION)

        # Assert
        gather_metrics = next(m for m in results if m.action_type == "gather")
        self.assertAlmostEqual(
            gather_metrics.interaction_rate, 0.5
        )  # 1 out of 2 gather actions had target
        self.assertEqual(
            gather_metrics.solo_performance, 10.0
        )  # Reward for non-interactive gather
        self.assertEqual(
            gather_metrics.interaction_performance, 5.0
        )  # Reward for interactive gather

    def test_analyze_with_empty_data(self):
        """Test analyzer behavior when no action data is present.

        Ensures the analyzer handles empty datasets gracefully by returning
        an empty result set without errors.
        """
        # Arrange
        self.repository.get_actions_by_scope.return_value = []

        # Act
        results = self.analyzer.analyze(scope=AnalysisScope.SIMULATION)

        # Assert
        self.assertEqual(len(results), 0)

    def test_analyze_with_scope_filters(self):
        """Test analyzer functionality with different scope filters.

        Verifies that the analyzer correctly applies filtering by:
        - Agent scope (specific agent_id)
        - Step scope (specific step number)

        Ensures proper repository calls are made with correct parameters.
        """
        # Arrange
        self.repository.get_actions_by_scope.return_value = self.test_actions

        # Act
        agent_results = self.analyzer.analyze(scope=AnalysisScope.AGENT, agent_id="1")
        step_results = self.analyzer.analyze(scope=AnalysisScope.STEP, step=1)

        # Assert
        self.repository.get_actions_by_scope.assert_any_call(
            AnalysisScope.AGENT, "1", None, None
        )
        self.repository.get_actions_by_scope.assert_any_call(
            AnalysisScope.STEP, None, 1, None
        )

    def test_analyze_statistical_measures(self):
        """Test the calculation of detailed statistical measures for rewards.

        Verifies computation of statistical metrics including:
        - Minimum and maximum rewards
        - Median reward
        - Variance and standard deviation
        - Reward quartiles

        Uses a sequence of gather actions with varied rewards for validation.
        """
        # Arrange
        actions_with_varied_rewards = [
            AgentActionData(
                agent_id="1",
                action_type="gather",
                step_number=1,
                action_target_id=None,
                resources_before=None,
                resources_after=None,
                state_before_id=None,
                state_after_id=None,
                reward=10.0,
                details=None,
            ),
            AgentActionData(
                agent_id="1",
                action_type="gather",
                step_number=2,
                action_target_id=None,
                resources_before=None,
                resources_after=None,
                state_before_id=None,
                state_after_id=None,
                reward=20.0,
                details=None,
            ),
            AgentActionData(
                agent_id="1",
                action_type="gather",
                step_number=3,
                action_target_id=None,
                resources_before=None,
                resources_after=None,
                state_before_id=None,
                state_after_id=None,
                reward=30.0,
                details=None,
            ),
        ]
        self.repository.get_actions_by_scope.return_value = actions_with_varied_rewards

        # Act
        results = self.analyzer.analyze(scope=AnalysisScope.SIMULATION)

        # Assert
        gather_metrics = next(m for m in results if m.action_type == "gather")
        self.assertEqual(gather_metrics.min_reward, 10.0)
        self.assertEqual(gather_metrics.max_reward, 30.0)
        self.assertEqual(gather_metrics.median_reward, 20.0)
        self.assertIsNotNone(gather_metrics.variance_reward)
        self.assertIsNotNone(gather_metrics.std_dev_reward)
        self.assertIsNotNone(gather_metrics.quartiles_reward)

    @patch(
        "farm.database.analyzers.temporal_pattern_analyzer.TemporalPatternAnalyzer.analyze"
    )
    @patch("farm.database.analyzers.resource_impact_analyzer.ResourceImpactAnalyzer.analyze")
    @patch(
        "farm.database.analyzers.decision_pattern_analyzer.DecisionPatternAnalyzer.analyze"
    )
    def test_analyze_with_step_range(self, mock_decision, mock_resource, mock_temporal):
        """Test analyzer functionality with step range filtering.

        Verifies that the analyzer correctly processes actions within
        a specified step range and makes appropriate repository calls
        with the step range parameter.
        """
        # Arrange
        step_range = (1, 2)
        self.repository.get_actions_by_scope.return_value = self.test_actions

        # Mock the internal analyzers to avoid additional repository calls
        # Create mock objects with action_type attributes
        mock_time_pattern = Mock()
        mock_time_pattern.action_type = "gather"
        mock_temporal.return_value = [mock_time_pattern]

        mock_resource_impact = Mock()
        mock_resource_impact.action_type = "gather"
        mock_resource.return_value = [mock_resource_impact]

        mock_decision_patterns = Mock()
        mock_decision_patterns.decision_patterns = {"gather": Mock()}
        mock_decision.return_value = mock_decision_patterns

        # Act
        results = self.analyzer.analyze(
            scope=AnalysisScope.SIMULATION, step_range=step_range
        )

        # Assert - only the main call should be made
        self.repository.get_actions_by_scope.assert_called_once_with(
            scope=AnalysisScope.SIMULATION, agent_id=None, step=None, step_range=step_range
        )

    @patch(
        "farm.database.analyzers.temporal_pattern_analyzer.TemporalPatternAnalyzer.analyze"
    )
    @patch("farm.database.analyzers.resource_impact_analyzer.ResourceImpactAnalyzer.analyze")
    @patch(
        "farm.database.analyzers.decision_pattern_analyzer.DecisionPatternAnalyzer.analyze"
    )
    def test_pattern_analysis_integration(
        self, mock_decision, mock_resource, mock_temporal
    ):
        """Test integration with various pattern analysis components.

        Verifies proper interaction with:
        - TemporalPatternAnalyzer
        - ResourceImpactAnalyzer
        - DecisionPatternAnalyzer

        Ensures that pattern analysis results are correctly incorporated
        into the final metrics for each action type.
        """
        # Arrange
        # Create mock objects with action_type attributes
        mock_time_pattern = Mock()
        mock_time_pattern.action_type = "gather"
        mock_temporal.return_value = [mock_time_pattern]

        mock_resource_impact = Mock()
        mock_resource_impact.action_type = "gather"
        mock_resource.return_value = [mock_resource_impact]

        mock_decision_patterns = Mock()
        mock_decision_patterns.decision_patterns = {"gather": Mock()}
        mock_decision.return_value = mock_decision_patterns

        self.repository.get_actions_by_scope.return_value = self.test_actions

        # Act
        results = self.analyzer.analyze(scope=AnalysisScope.SIMULATION)

        # Assert
        self.assertTrue(mock_temporal.called)
        self.assertTrue(mock_resource.called)
        self.assertTrue(mock_decision.called)

        gather_metrics = next(m for m in results if m.action_type == "gather")
        self.assertEqual(len(gather_metrics.temporal_patterns), 1)
        self.assertEqual(len(gather_metrics.resource_impacts), 1)
        self.assertEqual(len(gather_metrics.decision_patterns), 1)


if __name__ == "__main__":
    unittest.main()
