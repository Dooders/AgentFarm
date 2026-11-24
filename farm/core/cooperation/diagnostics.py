"""Classify how a cooperative unit is doing: healthy, or which way it fails.

Six failure modes, each with an operational test:

=========== ========================================== =============================================
Mode        What happens                               Operational test (defaults)
=========== ========================================== =============================================
drift       unit goals pull away from member goals     over the last ``window`` steps agent members
                                                       net-lose while the unit retains resources
capture     one member's goals become the unit's       a member's payout share (aid excluded)
                                                       exceeds its contribution share by
                                                       ``capture_gap`` and is at least
                                                       ``capture_share``
hollow      no shared goals ever emerge                pooled inflow is below ``hollow_throughput``
                                                       of members' gross income
override    a higher level overrules the members       a parent vetoed an exit, or an arranged
                                                       member did not consent to the match
trap        exit cost is higher than the unit's value  a member spent ``trap_steps`` wanting out but
                                                       priced in
fragility   exit cost too low to absorb shocks         members' exits dissolved the unit within
                                                       ``fragility_lifetime`` steps of forming
=========== ========================================== =============================================

Only exit-driven dissolutions count as fragility (``fragility_reasons``). A
young unit that ends because a member died is not an institutional failure:
it is *indeterminate* if it lived fewer than ``min_steps`` steps and is
otherwise classified on the remaining modes.

A unit is **healthy** when none of these fire, it created surplus, and no
agent member is a net loser beyond ``loss_tolerance`` per step: it produces
something its members could not produce alone, and its goals stay close
enough to each member's that all of them keep choosing to stay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from farm.core.cooperation.unit import CooperativeUnit


class FailureMode(str, Enum):
    HEALTHY = "healthy"
    DRIFT = "drift"
    CAPTURE = "capture"
    HOLLOW = "hollow"
    OVERRIDE = "override"
    TRAP = "trap"
    FRAGILITY = "fragility"
    UNCLASSIFIED = "unclassified"
    INDETERMINATE = "indeterminate"


#: Order in which co-occurring modes pick the *primary* label.
PRIMARY_ORDER: tuple[FailureMode, ...] = (
    FailureMode.FRAGILITY,
    FailureMode.OVERRIDE,
    FailureMode.CAPTURE,
    FailureMode.TRAP,
    FailureMode.DRIFT,
    FailureMode.HOLLOW,
)

FAILURE_MODES: tuple[FailureMode, ...] = PRIMARY_ORDER


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Cut-offs for :func:`diagnose_unit` (see module table)."""

    min_steps: int = 20
    window: int = 50
    capture_share: float = 0.6
    capture_gap: float = 0.25
    hollow_throughput: float = 0.05
    fragility_lifetime: int = 30
    trap_steps: int = 15
    loss_tolerance: float = 0.02
    fragility_reasons: tuple[str, ...] = ("below_min_members",)

    def __post_init__(self) -> None:
        if self.min_steps < 1 or self.window < 1 or self.fragility_lifetime < 1 or self.trap_steps < 1:
            raise ValueError("step thresholds must be positive")
        for name in ("capture_share", "capture_gap", "hollow_throughput"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.loss_tolerance < 0.0:
            raise ValueError("loss_tolerance must be non-negative")


DEFAULT_THRESHOLDS = DiagnosticThresholds()


@dataclass
class UnitDiagnosis:
    """Result of diagnosing one unit."""

    unit_id: str
    kind: str
    lifetime: int
    active: bool
    modes: list[FailureMode]
    primary: FailureMode
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind,
            "lifetime": self.lifetime,
            "active": self.active,
            "modes": [m.value for m in self.modes],
            "primary": self.primary.value,
            **{f"ev_{k}": v for k, v in self.evidence.items()},
        }


def _agent_ledgers(unit: CooperativeUnit):
    ledgers = {**unit.departed, **unit.members}
    return [ledger for ledger in ledgers.values() if not ledger.is_unit]


def diagnose_unit(
    unit: CooperativeUnit,
    current_step: int,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
    vetoed_unit_ids: Iterable[str] = (),
) -> UnitDiagnosis:
    """Label ``unit`` with every failure mode that fires and a primary mode.

    ``vetoed_unit_ids`` lists units whose exits a parent vetoed (from the
    registry's ``exit_vetoed`` events).
    """
    t = thresholds
    lifetime = unit.lifetime(current_step)
    ledgers = _agent_ledgers(unit)
    evidence: dict[str, Any] = {}
    modes: list[FailureMode] = []

    dissolved_young = (
        (not unit.is_active) and lifetime < t.fragility_lifetime and unit.dissolution_reason in t.fragility_reasons
    )
    if lifetime < t.min_steps and not dissolved_young:
        return UnitDiagnosis(unit.unit_id, unit.kind, lifetime, unit.is_active, [], FailureMode.INDETERMINATE, evidence)
    if dissolved_young:
        modes.append(FailureMode.FRAGILITY)
    evidence["dissolution_reason"] = unit.dissolution_reason

    vetoed = unit.unit_id in set(vetoed_unit_ids)
    unconsented = sum(1 for ledger in ledgers if not ledger.consented)
    evidence["exit_vetoed"] = vetoed
    evidence["unconsented_members"] = unconsented
    if vetoed or unconsented:
        modes.append(FailureMode.OVERRIDE)

    # Capture is about the payout rule favouring one member beyond what it
    # puts in; emergency aid to a needy member is insurance, not capture, so
    # it is excluded here.
    total_received = sum(ledger.received for ledger in ledgers)
    total_contributed = sum(ledger.contributed for ledger in ledgers)
    max_gap = 0.0
    max_share = 0.0
    if len(ledgers) >= 2 and total_received > 0.0 and total_contributed > 0.0:
        for ledger in ledgers:
            r_share = ledger.received / total_received
            c_share = ledger.contributed / total_contributed
            if r_share - c_share > max_gap:
                max_gap, max_share = r_share - c_share, r_share
    evidence["capture_gap"] = max_gap
    evidence["capture_share"] = max_share
    if max_gap >= t.capture_gap and max_share >= t.capture_share:
        modes.append(FailureMode.CAPTURE)

    max_trapped = max((ledger.trapped_steps for ledger in ledgers), default=0)
    evidence["max_trapped_steps"] = max_trapped
    if max_trapped >= t.trap_steps:
        modes.append(FailureMode.TRAP)

    recent = unit.history[-t.window :]
    if recent:
        member_net = sum(r.member_net_sum for r in recent)
        before = len(unit.history) - len(recent) - 1
        pool_before = unit.history[before].pool if before >= 0 else 0.0
        retained = (recent[-1].pool - pool_before) + sum(r.upward for r in recent)
        evidence["window_member_net"] = member_net
        evidence["window_retained"] = retained
        member_steps = sum(r.n_members for r in recent)
        if member_steps and member_net < -t.loss_tolerance * member_steps and retained > 0.0:
            modes.append(FailureMode.DRIFT)

    gross = unit.totals["gross_income"]
    throughput = unit.totals["inflow"] / gross if gross > 0.0 else 0.0
    evidence["throughput"] = throughput
    if throughput < t.hollow_throughput:
        modes.append(FailureMode.HOLLOW)

    worst_rate = 0.0
    for ledger in ledgers:
        if ledger.steps:
            worst_rate = min(worst_rate, ledger.net / ledger.steps)
    evidence["worst_member_net_rate"] = worst_rate
    evidence["surplus"] = unit.totals["surplus"]

    if modes:
        primary = next(m for m in PRIMARY_ORDER if m in modes)
    elif unit.totals["surplus"] > 0.0 and worst_rate >= -t.loss_tolerance:
        primary = FailureMode.HEALTHY
        modes = [FailureMode.HEALTHY]
    else:
        primary = FailureMode.UNCLASSIFIED
    return UnitDiagnosis(unit.unit_id, unit.kind, lifetime, unit.is_active, modes, primary, evidence)


def diagnose_all(
    units: Sequence[CooperativeUnit],
    current_step: int,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
    vetoed_unit_ids: Iterable[str] = (),
) -> list[UnitDiagnosis]:
    vetoed = set(vetoed_unit_ids)
    return [diagnose_unit(u, current_step, thresholds, vetoed) for u in units]


def mode_prevalence(diagnoses: Sequence[UnitDiagnosis], kind: str | None = None) -> dict[str, float]:
    """Share of classifiable units showing each mode (modes can co-occur).

    ``indeterminate`` units are excluded from the denominator. Also reports
    ``primary_<mode>`` shares, which sum to 1 over classified units.
    """
    pool = [d for d in diagnoses if (kind is None or d.kind == kind) and d.primary is not FailureMode.INDETERMINATE]
    out: dict[str, float] = {"n_classified": float(len(pool))}
    labels = list(PRIMARY_ORDER) + [FailureMode.HEALTHY, FailureMode.UNCLASSIFIED]
    for mode in labels:
        if pool:
            out[mode.value] = sum(1 for d in pool if mode in d.modes) / len(pool)
            out[f"primary_{mode.value}"] = sum(1 for d in pool if d.primary is mode) / len(pool)
        else:
            out[mode.value] = float("nan")
            out[f"primary_{mode.value}"] = float("nan")
    return out
