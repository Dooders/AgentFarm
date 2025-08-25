import unittest

from farm.database.analyzers.temporal_pattern_analyzer import TemporalPatternAnalyzer
from farm.database.repositories.action_repository import ActionRepository
from farm.database.session_manager import SessionManager


class TestTemporalPatternAnalyzer(unittest.TestCase):
    def setUp(self):
        # Setup mock repository and data for testing
        self.repository = ActionRepository(
            session_manager=SessionManager()
        )  # Mock or use a test database
        self.analyzer = TemporalPatternAnalyzer(self.repository)

    def test_analyze_with_rolling_average(self):
        # Test the analyze method with rolling averages
        patterns = self.analyzer.analyze(rolling_window_size=5)
        for pattern in patterns:
            self.assertIsNotNone(pattern.rolling_average_rewards)
            self.assertIsNotNone(pattern.rolling_average_counts)
            # Add more assertions based on expected results

    def test_segment_events(self):
        # Test event segmentation
        segments = self.analyzer.segment_events(event_steps=[100, 200])
        self.assertTrue(len(segments) > 0)
        # Add more assertions based on expected results


if __name__ == "__main__":
    unittest.main()
