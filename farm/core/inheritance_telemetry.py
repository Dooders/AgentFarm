"""Per-run telemetry counters for policy inheritance and decision failures.

The intrinsic-evolution runner attaches one :class:`InheritanceTelemetry`
instance to the live :class:`farm.core.environment.Environment` for the
duration of a run.  Reproduction code paths in :mod:`farm.core.agent.core`
record warm-start outcomes through it, and the agent step records
``decide_action`` failures through :meth:`record_decide_action_failure`.
The runner serialises the resulting counts into the per-run metadata JSON
(``policy_inheritance_metrics``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class InheritanceTelemetry:
    """In-memory counters for Lamarckian warm-start outcomes during a run.

    ``lamarckian_warmstart_skipped_reasons`` is a multiset keyed by the stable
    ``WARMSTART_REASON_*`` strings from
    :mod:`farm.core.policy_inheritance`, so downstream analysis can split
    "skipped because architecture mutated" from genuine missing-API cases.

    ``decide_action_failures`` counts agent-step decision failures so
    experiment harnesses can detect a silent degradation regime (the
    decision module now propagates errors instead of falling back to a
    chromosome-only sampler — see the 2026-05-22 dev-log entry).
    ``decide_action_failure_reasons`` is a multiset keyed by the offending
    exception class name so common failure modes are easy to triage from
    the persisted metadata.
    """

    lamarckian_warmstart_applied: int = 0
    lamarckian_warmstart_skipped: int = 0
    lamarckian_warmstart_skipped_reasons: Counter = field(default_factory=Counter)
    decide_action_failures: int = 0
    decide_action_failure_reasons: Counter = field(default_factory=Counter)

    def record_applied(self) -> None:
        self.lamarckian_warmstart_applied += 1

    def record_skipped(self, reason: str) -> None:
        self.lamarckian_warmstart_skipped += 1
        self.lamarckian_warmstart_skipped_reasons[reason] += 1

    def record_decide_action_failure(self, error_type: str = "Exception") -> None:
        """Record one agent-step decision failure attributed to ``error_type``."""
        self.decide_action_failures += 1
        self.decide_action_failure_reasons[error_type] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Render as a JSON-friendly dict for run metadata."""
        return {
            "lamarckian_warmstart_applied": int(self.lamarckian_warmstart_applied),
            "lamarckian_warmstart_skipped": int(self.lamarckian_warmstart_skipped),
            "lamarckian_warmstart_skipped_reasons": dict(
                self.lamarckian_warmstart_skipped_reasons
            ),
            "decide_action_failures": int(self.decide_action_failures),
            "decide_action_failure_reasons": dict(
                self.decide_action_failure_reasons
            ),
        }
