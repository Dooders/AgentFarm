from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from farm.actions.algorithms.base import ActionAlgorithm


class AlgorithmTrainer:
    """Train ML algorithms on collected experience."""

    def train_algorithm(
        self,
        algorithm: ActionAlgorithm,
        training_data: Iterable[Tuple[np.ndarray, int, float]],
    ) -> None:
        states_list: List[np.ndarray] = []
        actions_list: List[int] = []
        rewards_list: List[float] = []

        for state, action, reward in training_data:
            states_list.append(np.asarray(state, dtype=float))
            actions_list.append(int(action))
            rewards_list.append(float(reward))

        if not states_list:
            return

        X = np.vstack(states_list)
        y_actions = np.asarray(actions_list, dtype=int)
        y_rewards = np.asarray(rewards_list, dtype=float)

        algorithm.train(X, y_actions, rewards=y_rewards)

