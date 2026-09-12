"""The Veil Ceiling: observation collapse in sealed-world agent evaluation.

A pre-registered A/B design measuring whether behaviour under observation
keeps predicting behaviour outside it once agents can detect observation.
See ``docs/research/experiments/veil_ceiling/`` for the pre-registration
document and the recorded results.

Public entry points:

* :func:`farm.experiments.veil_ceiling.simulation.run_simulation` - one seeded run.
* :func:`farm.experiments.veil_ceiling.experiment.run_matrix` - the full cell x seed matrix.
* :func:`farm.experiments.veil_ceiling.analysis.analyze` - metrics, CIs, falsification checks.
"""

from farm.experiments.veil_ceiling.config import (
    CONDITIONS,
    INHERITANCE_MODES,
    Condition,
    LearnerConfig,
    MonitoringConfig,
    RunConfig,
    WorldConfig,
)

__all__ = [
    "CONDITIONS",
    "INHERITANCE_MODES",
    "Condition",
    "LearnerConfig",
    "MonitoringConfig",
    "RunConfig",
    "WorldConfig",
]
