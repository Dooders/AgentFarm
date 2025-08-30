from __future__ import annotations

from typing import Dict, List

import numpy as np

from farm.core.decision.algorithms.base import AlgorithmRegistry


class AlgorithmBenchmark:
    """Compare performance of different algorithms.

    This is a scaffold that sets up algorithms and returns a placeholder result
    dictionary. Integration with the full simulation loop can be added later.
    """

    def run_comparison(
        self, algorithms: List[str], environments: List[str], metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}

        # Placeholder: create algorithms and report dummy metrics
        for algo_name in algorithms:
            # Create with a small action space for synthetic benchmark
            algo = AlgorithmRegistry.create(algo_name, num_actions=4)
            key = f"{algo_name}_synthetic"
            results[key] = {m: float(np.random.rand()) for m in metrics}

        return results
