import unittest

import numpy as np

from farm.core.decision.algorithms.ensemble import RandomForestActionSelector
from farm.core.decision.algorithms.mlp import MLPActionSelector


class TestMLPActionSelector(unittest.TestCase):
    def test_unfitted_returns_uniform(self):
        algo = MLPActionSelector(
            num_actions=4, hidden_layer_sizes=(8,), random_state=42, max_iter=100
        )
        state = np.zeros(6)
        proba = algo.predict_proba(state)
        self.assertEqual(len(proba), 4)
        self.assertAlmostEqual(float(np.sum(proba)), 1.0, places=6)

    def test_train_and_select(self):
        # Simple synthetic dataset mapping quadrant to action
        X = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # class 0
                [0.0, 1.0, 0.0, 0.0],  # class 1
                [0.0, 0.0, 1.0, 0.0],  # class 2
                [0.0, 0.0, 0.0, 1.0],  # class 3
            ]
        )
        y = np.array([0, 1, 2, 3])

        algo = MLPActionSelector(
            num_actions=4, hidden_layer_sizes=(8,), random_state=42, max_iter=200
        )
        algo.train(X, y)

        test_state = np.array([1.0, 0.0, 0.0, 0.0])
        proba = algo.predict_proba(test_state)
        self.assertEqual(len(proba), 4)
        self.assertAlmostEqual(float(np.sum(proba)), 1.0, places=6)
        action = algo.select_action(test_state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)


class TestRandomForestActionSelector(unittest.TestCase):
    def test_unfitted_returns_uniform(self):
        algo = RandomForestActionSelector(
            num_actions=3, n_estimators=10, random_state=0
        )
        state = np.ones(5)
        proba = algo.predict_proba(state)
        self.assertEqual(len(proba), 3)
        self.assertAlmostEqual(float(np.sum(proba)), 1.0, places=6)

    def test_train_and_predict(self):
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        y = np.array([0, 1, 2])
        algo = RandomForestActionSelector(
            num_actions=3, n_estimators=10, random_state=0
        )
        algo.train(X, y)

        proba = algo.predict_proba(np.array([1.0, 0.0]))
        self.assertEqual(len(proba), 3)
        self.assertAlmostEqual(float(np.sum(proba)), 1.0, places=6)
        action = algo.select_action(np.array([1.0, 0.0]))
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 3)


if __name__ == "__main__":
    unittest.main()
