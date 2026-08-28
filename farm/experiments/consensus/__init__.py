"""Political consensus experiment: selection paradigms vs. treatment of a fixed bloc.

Compares how different election paradigms (party, individual, score, latent_match)
affect minority-cluster and total welfare when the elected steward allocates a
fixed budget across five public projects. ``constrained_individual`` is an
optional constitutional λ cap, not a voting rule.

Entry point: ``run_experiment.py`` at the repository root.
"""

from farm.experiments.consensus.population import POPULATION_TYPES, PROJECTS

__all__ = ["POPULATION_TYPES", "PROJECTS"]
