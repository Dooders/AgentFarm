from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from farm.actions.feature_engineering import FeatureEngineer

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent
    from farm.core.environment import Environment


class ExperienceCollector:
    """Collect training data for ML algorithms.

    Produces a list of (state_features, action, reward) tuples from an episode.
    """

    def __init__(self) -> None:
        self.feature_engineer = FeatureEngineer()

    def collect_episode(
        self, agent: "BaseAgent", environment: "Environment", max_steps: int = 200
    ) -> List[Tuple[np.ndarray, int, float]]:
        data: List[Tuple[np.ndarray, int, float]] = []

        for _ in range(max_steps):
            state = self.feature_engineer.extract_features(agent, environment)
            # Use existing select policy to choose an action
            selection_state = agent.select_module  # for parity with existing loop
            available_actions = agent.actions
            selection_input = agent.select_module  # placeholder to satisfy type checker
            # We rely on the agent's normal decision loop externally
            # Here we just record features; the caller should execute an action and obtain reward.
            data.append((state, 0, 0.0))

        return data

