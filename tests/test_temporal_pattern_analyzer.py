import unittest
from unittest.mock import Mock

from farm.database.analyzers.temporal_pattern_analyzer import TemporalPatternAnalyzer
from farm.database.data_types import AgentActionData
from farm.database.repositories.action_repository import ActionRepository


class TestTemporalPatternAnalyzer(unittest.TestCase):
    def setUp(self):
        # Setup mock repository and data for testing
        self.repository = Mock(spec=ActionRepository)
        self.analyzer = TemporalPatternAnalyzer(self.repository)

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
                step_number=2,
                action_target_id=None,
                resources_before=None,
                resources_after=None,
                state_before_id=None,
                state_after_id=None,
                reward=15.0,
                details=None,
            ),
            AgentActionData(
                agent_id="1",
                action_type="move",
                step_number=3,
                action_target_id=None,
                resources_before=None,
                resources_after=None,
                state_before_id=None,
                state_after_id=None,
                reward=2.0,
                details=None,
            ),
            AgentActionData(
                agent_id="1",
                action_type="gather",
                step_number=4,
                action_target_id=None,
                resources_before=None,
                resources_after=None,
                state_before_id=None,
                state_after_id=None,
                reward=12.0,
                details=None,
            ),
            AgentActionData(
                agent_id="1",
                action_type="move",
                step_number=5,
                action_target_id=None,
                resources_before=None,
                resources_after=None,
                state_before_id=None,
                state_after_id=None,
                reward=3.0,
                details=None,
            ),
        ]

    def test_analyze_with_rolling_average(self):
        # Test the analyze method with rolling averages
        self.repository.get_actions_by_scope.return_value = self.test_actions

        patterns = self.analyzer.analyze(rolling_window_size=1)
        self.assertEqual(
            len(patterns), 2
        )  # Should have patterns for 'gather' and 'move'

        # Find gather pattern
        gather_pattern = next(p for p in patterns if p.action_type == "gather")
        self.assertIsNotNone(gather_pattern.rolling_average_rewards)
        self.assertIsNotNone(gather_pattern.rolling_average_counts)

        # With time_period_size=100, all actions are in period 0
        # So we should have 1 period
        self.assertEqual(len(gather_pattern.time_distribution), 1)
        self.assertEqual(len(gather_pattern.reward_progression), 1)
        self.assertEqual(len(gather_pattern.rolling_average_rewards), 1)
        self.assertEqual(len(gather_pattern.rolling_average_counts), 1)

        # Check that rewards are averaged correctly for period 0: (10+15+12)/3 = 12.333...
        self.assertAlmostEqual(
            gather_pattern.reward_progression[0], 12.333333, places=5
        )
        self.assertEqual(gather_pattern.time_distribution[0], 3)  # 3 gather actions

    def test_segment_events(self):
        # Test event segmentation
        self.repository.get_actions_by_scope.return_value = self.test_actions

        segments = self.analyzer.segment_events(event_steps=[3, 5])
        self.assertEqual(len(segments), 3)  # Should have 3 segments: 0-3, 3-5, 5+

        # Check first segment (0-3)
        self.assertEqual(segments[0].start_step, 0)
        self.assertEqual(segments[0].end_step, 3)
        self.assertEqual(
            segments[0].action_counts.get("gather", 0), 2
        )  # gather at steps 1,2
        self.assertEqual(
            segments[0].action_counts.get("move", 0), 0
        )  # no move before step 3
        self.assertAlmostEqual(segments[0].average_rewards["gather"], 12.5)  # (10+15)/2

        # Check second segment (3-5)
        self.assertEqual(segments[1].start_step, 3)
        self.assertEqual(segments[1].end_step, 5)
        self.assertEqual(
            segments[1].action_counts.get("gather", 0), 1
        )  # gather at step 4
        self.assertEqual(segments[1].action_counts.get("move", 0), 1)  # move at step 3
        self.assertEqual(segments[1].average_rewards["gather"], 12.0)
        self.assertEqual(segments[1].average_rewards["move"], 2.0)

        # Check third segment (5+)
        self.assertEqual(segments[2].start_step, 5)
        self.assertIsNone(segments[2].end_step)
        self.assertEqual(segments[2].action_counts.get("move", 0), 1)  # move at step 5
        self.assertEqual(segments[2].average_rewards["move"], 3.0)


if __name__ == "__main__":
    unittest.main()
