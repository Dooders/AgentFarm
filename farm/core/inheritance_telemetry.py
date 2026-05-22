"""Per-run counters for offspring policy-inheritance events.

The intrinsic-evolution runner attaches one :class:`InheritanceTelemetry`
instance to the live :class:`farm.core.environment.Environment` for the
duration of a run.  Reproduction code paths in
:mod:`farm.core.agent.core` record warm-start outcomes through it, and the
runner serialises the resulting counts into the per-run metadata JSON
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
    """

    lamarckian_warmstart_applied: int = 0
    lamarckian_warmstart_skipped: int = 0
    lamarckian_warmstart_skipped_reasons: Counter = field(default_factory=Counter)

    def record_applied(self) -> None:
        self.lamarckian_warmstart_applied += 1

    def record_skipped(self, reason: str) -> None:
        self.lamarckian_warmstart_skipped += 1
        self.lamarckian_warmstart_skipped_reasons[reason] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Render as a JSON-friendly dict for run metadata."""
        return {
            "lamarckian_warmstart_applied": int(self.lamarckian_warmstart_applied),
            "lamarckian_warmstart_skipped": int(self.lamarckian_warmstart_skipped),
            "lamarckian_warmstart_skipped_reasons": dict(
                self.lamarckian_warmstart_skipped_reasons
            ),
        }
