from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neural_network import MLPClassifier

from .base import ActionAlgorithm


class MLPActionSelector(ActionAlgorithm):
    """MLP-based action selection using supervised learning.

    Trains a classifier on (state -> action) pairs and provides
    probability outputs for exploration.
    """

    def __init__(
        self,
        num_actions: int,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        random_state: Optional[int] = None,
        max_iter: int = 200,
        **_: object,
    ) -> None:
        super().__init__(num_actions=num_actions)
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
        )
        self._fitted: bool = False

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: Optional[np.ndarray] = None,
    ) -> None:
        # Note: MLPClassifier doesn't support sample_weight
        self.model.fit(states, actions)
        self._fitted = True

    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=float)

        X = state.reshape(1, -1)
        proba = self.model.predict_proba(X)[0]

        # Map from observed classes to full action space [0..num_actions-1]
        full = np.zeros(self.num_actions, dtype=float)
        for i, cls in enumerate(self.model.classes_):
            idx = int(cls)
            if 0 <= idx < self.num_actions:
                full[idx] = proba[i]

        s = full.sum()
        if s <= 0:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=float)
        return full / s

    def select_action(self, state: np.ndarray) -> int:
        probs = self.predict_proba(state)
        return int(np.random.choice(self.num_actions, p=probs))
