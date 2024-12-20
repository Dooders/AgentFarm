import unittest
from database.analyzers.temporal_pattern_analyzer import TemporalPatternAnalyzer
from database.repositories.action_repository import ActionRepository

class TestTemporalPatternAnalyzer(unittest.TestCase):
    def setUp(self):
        # Setup mock repository and data for testing
        self.repository = ActionRepository(session_manager=None)  # Mock or use a test database
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

if __name__ == '__main__':
    unittest.main() 