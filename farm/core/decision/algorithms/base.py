from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Dict, Optional, Type

import numpy as np


class ActionAlgorithm(ABC):
    """Abstract base class for all action selection algorithms.

    Implementations should support both direct action selection and
    probability prediction for exploration and blending with heuristics.
    """

    def __init__(self, num_actions: int, **_: Any) -> None:
        self.num_actions = int(num_actions)

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Select an action given the current state."""

    @abstractmethod
    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: Optional[np.ndarray] = None,
    ) -> None:
        """Train the algorithm on experience data."""

    @abstractmethod
    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        """Predict action probabilities for exploration."""

    def save_model(self, path: str) -> None:
        """Save the full algorithm instance to disk using joblib."""
        try:
            import joblib

            joblib.dump(self, path)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to save model to {path}: {exc}")

    @classmethod
    def load_model(cls: Type[ActionAlgorithm], path: str) -> ActionAlgorithm:
        """Load an algorithm instance from disk using joblib."""
        try:
            import joblib

            obj = joblib.load(path)
            if not isinstance(obj, ActionAlgorithm):
                raise TypeError(f"Loaded object is not an ActionAlgorithm: {type(obj)}")
            return obj
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load model from {path}: {exc}")


class AlgorithmRegistry:
    """Registry for managing different action selection algorithms.

    Uses lazy imports to avoid importing heavy ML libraries until needed.
    """

    _algorithms: Dict[str, str] = {
        # Traditional ML algorithms
        "mlp": "farm.core.decision.algorithms.mlp:MLPActionSelector",
        "svm": "farm.core.decision.algorithms.svm:SVMActionSelector",
        "random_forest": "farm.core.decision.algorithms.ensemble:RandomForestActionSelector",
        "gradient_boost": "farm.core.decision.algorithms.ensemble:GradientBoostActionSelector",
        "naive_bayes": "farm.core.decision.algorithms.ensemble:NaiveBayesActionSelector",
        "knn": "farm.core.decision.algorithms.ensemble:KNNActionSelector",
        # Reinforcement Learning algorithms (Tianshou-based)
        "ppo": "farm.core.decision.algorithms.tianshou:PPOWrapper",
        "sac": "farm.core.decision.algorithms.tianshou:SACWrapper",
        "a2c": "farm.core.decision.algorithms.tianshou:A2CWrapper",
        "dqn": "farm.core.decision.algorithms.tianshou:DQNWrapper",
        "ddpg": "farm.core.decision.algorithms.tianshou:DDPGWrapper",
    }

    @classmethod
    def register(cls, name: str, dotted_path: str) -> None:
        cls._algorithms[name] = dotted_path

    @classmethod
    def create(cls, name: str, num_actions: int, **params: Any) -> ActionAlgorithm:
        if name not in cls._algorithms:
            valid = ", ".join(sorted(cls._algorithms.keys()))
            raise ValueError(f"Unknown algorithm '{name}'. Valid algorithms: {valid}")

        module_path, _, class_name = cls._algorithms[name].partition(":")
        if not module_path or not class_name:
            raise ValueError(
                f"Invalid registry entry for '{name}': {cls._algorithms[name]}"
            )

        module = import_module(module_path)
        algo_cls = getattr(module, class_name)
        if not isinstance(algo_cls, type):
            raise TypeError(
                f"Registry entry '{name}' does not resolve to a class: {algo_cls}"
            )
        return algo_cls(num_actions=num_actions, **params)
