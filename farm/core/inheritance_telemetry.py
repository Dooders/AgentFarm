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
from typing import Any, Dict, Optional

from farm.core.policy_inheritance import WARMSTART_REASON_GATE_NOT_CLEARED


@dataclass
class InheritanceTelemetry:
    """In-memory counters for policy warm-start outcomes during a run.

    The counters are mode-neutral: a single run uses one ``inheritance_mode``
    (see the inheritance-mode A/B harness), so ``warmstart_applied`` /
    ``warmstart_skipped`` already describe that run's active mode.

    ``warmstart_skipped_reasons`` is a multiset keyed by the stable
    ``WARMSTART_REASON_*`` strings from
    :mod:`farm.core.policy_inheritance`, so downstream analysis can split
    "skipped because architecture mutated" from genuine missing-API cases and
    from the P4 fitness gate.

    ``blend_alpha`` records the P4 blend coefficient actually used for the run
    (``None`` when the active mode does not blend). It is constant per run, so
    :meth:`record_blend_alpha` simply stores the most recently observed value.

    ``decide_action_failures`` counts agent-step decision failures so
    experiment harnesses can detect a silent degradation regime (the
    decision module now propagates errors instead of falling back to a
    chromosome-only sampler — see the 2026-05-22 dev-log entry).
    ``decide_action_failure_reasons`` is a multiset keyed by the offending
    exception class name so common failure modes are easy to triage from
    the persisted metadata.
    """

    warmstart_applied: int = 0
    warmstart_skipped: int = 0
    warmstart_skipped_reasons: Counter = field(default_factory=Counter)
    blend_alpha: Optional[float] = None
    decide_action_failures: int = 0
    decide_action_failure_reasons: Counter = field(default_factory=Counter)

    def record_applied(self) -> None:
        self.warmstart_applied += 1

    def record_skipped(self, reason: str) -> None:
        self.warmstart_skipped += 1
        self.warmstart_skipped_reasons[reason] += 1

    def record_blend_alpha(self, alpha: Optional[float]) -> None:
        """Record the P4 blend coefficient in effect for this run."""
        if alpha is not None:
            self.blend_alpha = float(alpha)

    def record_decide_action_failure(self, error_type: str = "Exception") -> None:
        """Record one agent-step decision failure attributed to ``error_type``."""
        self.decide_action_failures += 1
        self.decide_action_failure_reasons[error_type] += 1

    def _coverage(self) -> Optional[float]:
        """Fraction of warm-start attempts that applied a payload."""
        total = self.warmstart_applied + self.warmstart_skipped
        if total <= 0:
            return None
        return self.warmstart_applied / total

    def _gate_hit_rate(self) -> Optional[float]:
        """Fraction of warm-start attempts that cleared the P4 fitness gate.

        Computed over all attempts as ``(total - gate_not_cleared) / total``.
        ``None`` when there were no attempts. For non-gated modes this is 1.0
        because nothing is attributed to ``gate_not_cleared``.
        """
        total = self.warmstart_applied + self.warmstart_skipped
        if total <= 0:
            return None
        gate_not_cleared = self.warmstart_skipped_reasons.get(
            WARMSTART_REASON_GATE_NOT_CLEARED, 0
        )
        return (total - gate_not_cleared) / total

    def to_dict(self) -> Dict[str, Any]:
        """Render as a JSON-friendly dict for run metadata."""
        return {
            "warmstart_applied": int(self.warmstart_applied),
            "warmstart_skipped": int(self.warmstart_skipped),
            "warmstart_skipped_reasons": dict(self.warmstart_skipped_reasons),
            "warmstart_coverage": self._coverage(),
            "gate_not_cleared": int(
                self.warmstart_skipped_reasons.get(
                    WARMSTART_REASON_GATE_NOT_CLEARED, 0
                )
            ),
            "gate_hit_rate": self._gate_hit_rate(),
            "blend_alpha": self.blend_alpha,
            "decide_action_failures": int(self.decide_action_failures),
            "decide_action_failure_reasons": dict(
                self.decide_action_failure_reasons
            ),
        }
