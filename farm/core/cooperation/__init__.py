"""Cooperative units: one configurable model for cooperative institutions.

A cooperative unit forms when agents pool part of their lives and the
arrangement develops goals none of them had alone. This package provides:

* :mod:`~farm.core.cooperation.settings` -- the seven settings (members,
  scope, exit cost, recognition, exclusivity, entry, symmetry) and presets for
  marriage, cohabitation, friendship, business partnership, family and tribe.
* :mod:`~farm.core.cooperation.goals` -- member goals as a mixture of the five
  motives, unit goals, and how a member values a membership.
* :mod:`~farm.core.cooperation.unit` -- the unit, its ledgers and pool economics.
* :mod:`~farm.core.cooperation.registry` -- the rule enforcer: formation,
  exits, exclusivity, nesting, and per-step pool settlement.
* :mod:`~farm.core.cooperation.diagnostics` -- healthy vs. drift, capture,
  hollow, override, trap and fragility.

The modules above are engine-agnostic (participants are string ids). The
spatial AgentFarm world uses them through:

* :mod:`~farm.core.cooperation.config` -- :class:`CooperationConfig`, carried on
  ``SimulationConfig.cooperation`` (off by default).
* :mod:`~farm.core.cooperation.genes` -- heritable settings, motive and
  propensity loci appended to each agent's hyperparameter chromosome.
* :mod:`~farm.core.cooperation.actions` -- the learned ``bond`` and ``leave``
  actions.
* :mod:`~farm.core.cooperation.service` -- :class:`CooperationService`, the
  environment hooks (pooling on gather, settlement, births, deaths, rewards,
  perception, telemetry).

None of these import torch. See ``docs/concepts/cooperation.md``.
"""

from farm.core.cooperation.actions import (
    COOPERATION_ACTION_NAMES,
    CooperationActionType,
    cooperation_service_of,
)
from farm.core.cooperation.config import CooperationConfig
from farm.core.cooperation.diagnostics import (
    FAILURE_MODES,
    DiagnosticThresholds,
    FailureMode,
    UnitDiagnosis,
    diagnose_all,
    diagnose_unit,
    mode_prevalence,
)
from farm.core.cooperation.goals import (
    MOTIVE_SETTING_PREFERENCES,
    MOTIVES,
    MemberGoals,
    MembershipAppraisal,
    Motive,
    UnitGoals,
    ValuationScales,
    motive_fit,
    value_of_membership,
)
from farm.core.cooperation.registry import CooperationEvent, CooperationRegistry
from farm.core.cooperation.service import CooperationService
from farm.core.cooperation.settings import (
    PRESETS,
    SETTING_NAMES,
    CooperationSettings,
    Entry,
    preset,
    rank_weights,
)
from farm.core.cooperation.unit import CooperativeUnit, ExitResult, MemberLedger, UnitEconomics, UnitStepRecord

__all__ = [
    "COOPERATION_ACTION_NAMES",
    "FAILURE_MODES",
    "MOTIVES",
    "MOTIVE_SETTING_PREFERENCES",
    "PRESETS",
    "SETTING_NAMES",
    "CooperationActionType",
    "CooperationConfig",
    "CooperationEvent",
    "CooperationRegistry",
    "CooperationService",
    "CooperationSettings",
    "CooperativeUnit",
    "DiagnosticThresholds",
    "Entry",
    "ExitResult",
    "FailureMode",
    "MemberGoals",
    "MemberLedger",
    "MembershipAppraisal",
    "Motive",
    "UnitDiagnosis",
    "UnitEconomics",
    "UnitGoals",
    "UnitStepRecord",
    "ValuationScales",
    "cooperation_service_of",
    "diagnose_all",
    "diagnose_unit",
    "mode_prevalence",
    "motive_fit",
    "preset",
    "rank_weights",
    "value_of_membership",
]
