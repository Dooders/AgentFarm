"""Political consensus experiment: selection paradigms vs. treatment of non-supporters.

Compares how different election paradigms (party, individual, score, latent_match,
optional constrained_individual) affect the welfare of electoral losers when the
elected steward allocates a fixed budget across five public projects.

Entry point: ``run_experiment.py`` at the repository root.
"""

from farm.experiments.consensus.population import POPULATION_TYPES, PROJECTS

__all__ = ["POPULATION_TYPES", "PROJECTS"]
