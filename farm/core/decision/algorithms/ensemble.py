from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ActionAlgorithm


def _align_proba(
    num_actions: int, classes: np.ndarray, proba_row: np.ndarray
) -> np.ndarray:
    full = np.zeros(num_actions, dtype=float)
    for i, cls in enumerate(classes):
        idx = int(cls)
        if 0 <= idx < num_actions:
            full[idx] = proba_row[i]
    s = full.sum()
    return full / s if s > 0 else np.full(num_actions, 1.0 / num_actions, dtype=float)


class RandomForestActionSelector(ActionAlgorithm):
    """Ensemble learning approach using decision trees (Random Forest)."""

    def __init__(
        self,
        num_actions: int,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__(num_actions=num_actions)
        from sklearn.ensemble import RandomForestClassifier

        self._random_state = random_state
        self._rng = np.random.RandomState(random_state) if random_state is not None else None
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
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
        return _align_proba(self.num_actions, self.model.classes_, proba)

    def select_action(self, state: np.ndarray) -> int:
        probs = self.predict_proba(state)
        if self._rng is not None:
            return int(self._rng.choice(self.num_actions, p=probs))
        else:
            return int(np.random.choice(self.num_actions, p=probs))


class NaiveBayesActionSelector(ActionAlgorithm):
    """Naive Bayes probabilistic classifier for action selection."""

    def __init__(self, num_actions: int, random_state: Optional[int] = None, **_: object) -> None:
        super().__init__(num_actions=num_actions)
        from sklearn.naive_bayes import GaussianNB

        self._random_state = random_state
        self._rng = np.random.RandomState(random_state) if random_state is not None else None
        self.model = GaussianNB()
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
        return _align_proba(self.num_actions, self.model.classes_, proba)

    def select_action(self, state: np.ndarray) -> int:
        probs = self.predict_proba(state)
        if self._rng is not None:
            return int(self._rng.choice(self.num_actions, p=probs))
        else:
            return int(np.random.choice(self.num_actions, p=probs))


class KNNActionSelector(ActionAlgorithm):
    """K-Nearest Neighbors classifier for action selection."""

    def __init__(self, num_actions: int, n_neighbors: int = 5, random_state: Optional[int] = None, **_: object) -> None:
        super().__init__(num_actions=num_actions)
        from sklearn.neighbors import KNeighborsClassifier

        self._random_state = random_state
        self._rng = np.random.RandomState(random_state) if random_state is not None else None
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
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
        proba_result = self.model.predict_proba(X)
        # Convert sparse matrix to dense if needed
        if hasattr(proba_result, "toarray"):
            proba_result = proba_result.toarray()  # type: ignore
        proba = proba_result[0]  # type: ignore
        return _align_proba(self.num_actions, self.model.classes_, proba)

    def select_action(self, state: np.ndarray) -> int:
        probs = self.predict_proba(state)
        if self._rng is not None:
            return int(self._rng.choice(self.num_actions, p=probs))
        else:
            return int(np.random.choice(self.num_actions, p=probs))


class GradientBoostActionSelector(ActionAlgorithm):
    """Gradient boosting classifier for action selection.

    Tries to use XGBoost if available, otherwise LightGBM. If neither is
    available, raises an ImportError at instantiation time.
    """

    def __init__(
        self,
        num_actions: int,
        random_state: Optional[int] = None,
        **params,
    ) -> None:
        super().__init__(num_actions=num_actions)
        self._random_state = random_state
        self._rng = np.random.RandomState(random_state) if random_state is not None else None
        self._backend: str
        self._fitted: bool = False

        try:
            import xgboost as xgb  # type: ignore

            # XGBoost expects labels as integers 0..num_actions-1
            # Filter out 'num_class' from params to prevent conflicts - we explicitly
            # set it to match num_actions to ensure consistency with our action space
            self.model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=num_actions,
                random_state=random_state,
                **{k: v for k, v in params.items() if k != "num_class"},
            )
            self._backend = "xgboost"
        except Exception:
            try:
                import lightgbm as lgb  # type: ignore

                # Filter out 'num_class' from params to prevent conflicts - we explicitly
                # set it to match num_actions to ensure consistency with our action space
                self.model = lgb.LGBMClassifier(
                    objective="multiclass",
                    num_class=num_actions,
                    random_state=random_state,
                    **{k: v for k, v in params.items() if k != "num_class"},
                )
                self._backend = "lightgbm"
            except Exception as exc:
                raise ImportError(
                    "Neither xgboost nor lightgbm is available. Install one to use GradientBoostActionSelector"
                ) from exc

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
        proba_result = self.model.predict_proba(X)
        # Convert sparse matrix to dense if needed
        if hasattr(proba_result, "toarray"):
            proba_result = proba_result.toarray()  # type: ignore
        proba = proba_result[0]  # type: ignore
        # Some backends may return shape (num_actions,) instead of (num_classes subset,)
        if proba.ndim == 1 and proba.shape[0] == self.num_actions:  # type: ignore
            full = proba.astype(float)  # type: ignore
            s = full.sum()
            return (
                full / s
                if s > 0
                else np.full(self.num_actions, 1.0 / self.num_actions, dtype=float)
            )
        classes = getattr(self.model, "classes_", np.arange(self.num_actions))  # type: ignore
        return _align_proba(self.num_actions, classes, proba)  # type: ignore

    def select_action(self, state: np.ndarray) -> int:
        probs = self.predict_proba(state)
        if self._rng is not None:
            return int(self._rng.choice(self.num_actions, p=probs))
        else:
            return int(np.random.choice(self.num_actions, p=probs))
