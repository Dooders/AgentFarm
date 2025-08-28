from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from sklearn.svm import SVC

from .base import ActionAlgorithm


class SVMActionSelector(ActionAlgorithm):
    """SVM-based action classification for discrete action spaces."""

    def __init__(
        self,
        num_actions: int,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        C: float = 1.0,
        gamma: float | Literal["scale", "auto"] = "scale",
        degree: int = 3,
        random_state: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__(num_actions=num_actions)
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            probability=True,
            random_state=random_state,
        )
        self._fitted: bool = False

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: Optional[np.ndarray] = None,
    ) -> None:
        self.model.fit(states, actions)
        self._fitted = True

    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=float)
        X = state.reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        # Align to full action space
        full = np.zeros(self.num_actions, dtype=float)
        for i, cls in enumerate(self.model.classes_):
            idx = int(cls)
            if 0 <= idx < self.num_actions:
                full[idx] = proba[i]
        s = full.sum()
        return (
            full / s
            if s > 0
            else np.full(self.num_actions, 1.0 / self.num_actions, dtype=float)
        )

    def select_action(self, state: np.ndarray) -> int:
        probs = self.predict_proba(state)
        return int(np.random.choice(self.num_actions, p=probs))
