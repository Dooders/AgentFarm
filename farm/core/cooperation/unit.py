"""Cooperative units, their ledgers and the pool economics.

A :class:`CooperativeUnit` is the emergent object: it holds a pool, has its
own goals (:class:`~farm.core.cooperation.goals.UnitGoals`), keeps a ledger
per participant, and can sit inside a higher-level unit. Participants are
either agents (``members``) or child units (``children``); both are ranked by
join order, which is what :func:`~farm.core.cooperation.settings.rank_weights`
uses when ``symmetry < 1``.

Pool dynamics per step (applied by the registry):

1. Members contribute ``scope * income`` (split across their memberships).
2. The unit creates a surplus ``synergy * inflow * (1 - 1/n)`` -- zero for a
   lone participant, the emergent part for two or more.
3. A parent unit takes ``parent.scope`` of the child's inflow + surplus.
4. Needy agent members draw emergency aid from the pool (insurance).
5. The pool above the reserve target is paid out at ``payout_rate``, split by
   rank weights; child units receive their share into their own pool.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from farm.core.cooperation.goals import UnitGoals
from farm.core.cooperation.settings import CooperationSettings, rank_weights


@dataclass(frozen=True)
class UnitEconomics:
    """Environment-level parameters of pooling (not institutional settings).

    Attributes
    ----------
    synergy
        Surplus per unit of pooled inflow for a large unit; scaled by
        ``1 - 1/n`` so a pair gets half of it and a lone participant none.
    payout_rate
        Fraction of the pool above the reserve target paid out each step.
    aid_threshold
        Agent members whose holdings fall below this draw emergency aid.
    aid_amount
        Maximum emergency aid per needy member per step.
    benefit_ema_alpha
        Smoothing for each ledger's running net-benefit estimate.
    """

    synergy: float = 0.2
    payout_rate: float = 0.8
    aid_threshold: float = 2.0
    aid_amount: float = 1.0
    benefit_ema_alpha: float = 0.1

    def __post_init__(self) -> None:
        if self.synergy < 0.0 or not math.isfinite(self.synergy):
            raise ValueError("synergy must be finite and non-negative")
        if not 0.0 <= self.payout_rate <= 1.0:
            raise ValueError("payout_rate must lie in [0, 1]")
        if self.aid_threshold < 0.0 or self.aid_amount < 0.0:
            raise ValueError("aid_threshold and aid_amount must be non-negative")
        if not 0.0 < self.benefit_ema_alpha <= 1.0:
            raise ValueError("benefit_ema_alpha must lie in (0, 1]")

    def surplus(self, inflow: float, n_participants: int) -> float:
        """Emergent surplus created by pooling ``inflow`` among ``n`` participants."""
        if n_participants <= 1 or inflow <= 0.0:
            return 0.0
        return self.synergy * inflow * (1.0 - 1.0 / n_participants)


@dataclass
class MemberLedger:
    """Running account of one participant inside one unit.

    ``consented`` is false when the participant was placed in the unit by an
    arranged match it would not have chosen (the *override* failure mode).
    ``exit_desire_steps`` counts steps on which the participant wanted out;
    ``trapped_steps`` counts those on which the exit price kept it in.
    ``benefit_prior`` (when given at entry) seeds ``benefit_ema`` with what the
    participant expected, so the running estimate starts from expectations
    and is revised by experience instead of being set by the first step.
    """

    participant_id: str
    joined_step: int
    rank_order: int
    is_unit: bool = False
    consented: bool = True
    contributed: float = 0.0
    received: float = 0.0
    aid_received: float = 0.0
    gross_income: float = 0.0
    steps: int = 0
    benefit_ema: float = 0.0
    benefit_prior: float | None = None
    exit_desire_steps: int = 0
    trapped_steps: int = 0
    step_contributed: float = 0.0
    step_received: float = 0.0
    step_aid: float = 0.0

    @property
    def net(self) -> float:
        """Cumulative payouts plus aid minus contributions."""
        return self.received + self.aid_received - self.contributed

    def close_step(self, alpha: float) -> float:
        """Fold this step's flows into the running estimate; return step net."""
        step_net = self.step_received + self.step_aid - self.step_contributed
        if self.steps == 0 and self.benefit_prior is None:
            self.benefit_ema = step_net
        else:
            self.benefit_ema = (1.0 - alpha) * self.benefit_ema + alpha * step_net
        self.steps += 1
        self.step_contributed = 0.0
        self.step_received = 0.0
        self.step_aid = 0.0
        return step_net


@dataclass(frozen=True)
class UnitStepRecord:
    """Per-step aggregates kept on each unit for diagnostics."""

    step: int
    n_participants: int
    n_members: int
    inflow: float
    gross_income: float
    surplus: float
    upward: float
    downward: float
    aid: float
    payout: float
    pool: float
    top_share: float
    unit_utility: float
    member_net_sum: float
    member_net_min: float


class CooperativeUnit:
    """One live cooperative unit.

    Instances are created and mutated by
    :class:`~farm.core.cooperation.registry.CooperationRegistry`; read them
    freely, but route changes through the registry so membership indexes,
    exclusivity and events stay consistent.
    """

    def __init__(
        self,
        unit_id: str,
        settings: CooperationSettings,
        founded_step: int,
        goals: UnitGoals | None = None,
        parent_id: str | None = None,
        arranged: bool = False,
    ) -> None:
        self.unit_id = unit_id
        self.settings = settings
        self.goals = goals if goals is not None else UnitGoals()
        self.founded_step = int(founded_step)
        self.parent_id = parent_id
        self.arranged = bool(arranged)
        self.members: dict[str, MemberLedger] = {}
        self.children: dict[str, MemberLedger] = {}
        self.departed: dict[str, MemberLedger] = {}
        self.pool: float = 0.0
        self.dissolved_step: int | None = None
        self.dissolution_reason: str | None = None
        self.history: list[UnitStepRecord] = []
        self.totals: dict[str, float] = {
            "inflow": 0.0,
            "gross_income": 0.0,
            "surplus": 0.0,
            "upward": 0.0,
            "downward": 0.0,
            "aid": 0.0,
            "payout": 0.0,
            "forfeits": 0.0,
        }
        self._rank_counter = 0
        self._step: dict[str, float] = {}
        self._reset_step()

    # ------------------------------------------------------------------ state
    @property
    def kind(self) -> str:
        return self.settings.kind

    @property
    def is_active(self) -> bool:
        return self.dissolved_step is None

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def n_participants(self) -> int:
        return len(self.members) + len(self.children)

    def lifetime(self, current_step: int) -> int:
        """Steps from founding to dissolution (or to ``current_step``)."""
        end = self.dissolved_step if self.dissolved_step is not None else current_step
        return max(0, int(end) - self.founded_step)

    def participants(self) -> list[MemberLedger]:
        """All participants (agents and child units) in rank order."""
        ledgers = list(self.members.values()) + list(self.children.values())
        return sorted(ledgers, key=lambda ledger: ledger.rank_order)

    def ledger(self, participant_id: str) -> MemberLedger:
        if participant_id in self.members:
            return self.members[participant_id]
        if participant_id in self.children:
            return self.children[participant_id]
        raise KeyError(f"{participant_id!r} is not a participant of {self.unit_id}")

    def weights(self) -> dict[str, float]:
        """Payout shares by participant id (rank-ordered, sums to 1)."""
        ordered = self.participants()
        shares = rank_weights(len(ordered), self.settings.symmetry)
        return {ledger.participant_id: share for ledger, share in zip(ordered, shares)}

    def has_capacity(self) -> bool:
        return self.settings.has_capacity(self.n_participants)

    def utility(self) -> float:
        return self.goals.utility(self.pool, self.n_participants, self.is_active)

    # -------------------------------------------------------------- mutation
    def _add_participant(
        self,
        participant_id: str,
        step: int,
        is_unit: bool,
        consented: bool,
        prior: float | None = None,
    ) -> MemberLedger:
        ledger = MemberLedger(
            participant_id=participant_id,
            joined_step=int(step),
            rank_order=self._rank_counter,
            is_unit=is_unit,
            consented=consented,
            benefit_prior=None if prior is None else float(prior),
            benefit_ema=0.0 if prior is None else float(prior),
        )
        self._rank_counter += 1
        (self.children if is_unit else self.members)[participant_id] = ledger
        return ledger

    def _remove_participant(self, participant_id: str) -> MemberLedger:
        if participant_id in self.members:
            ledger = self.members.pop(participant_id)
        elif participant_id in self.children:
            ledger = self.children.pop(participant_id)
        else:
            raise KeyError(f"{participant_id!r} is not a participant of {self.unit_id}")
        self.departed[participant_id] = ledger
        return ledger

    def _reset_step(self) -> None:
        self._step = {
            "inflow": 0.0,
            "gross_income": 0.0,
            "surplus": 0.0,
            "upward": 0.0,
            "downward": 0.0,
            "aid": 0.0,
            "payout": 0.0,
        }

    def _record_contribution(self, participant_id: str, amount: float, gross: float) -> None:
        ledger = self.ledger(participant_id)
        ledger.contributed += amount
        ledger.step_contributed += amount
        ledger.gross_income += gross
        self.pool += amount
        self._step["inflow"] += amount
        self._step["gross_income"] += gross

    def _close_step(self, step: int, alpha: float, payouts_by_participant: dict[str, float]) -> UnitStepRecord:
        nets = []
        for ledger in self.participants():
            step_net = ledger.close_step(alpha)
            if not ledger.is_unit:
                nets.append(step_net)
        total_payout = sum(payouts_by_participant.values())
        top_share = max(payouts_by_participant.values()) / total_payout if total_payout > 0 else 0.0
        record = UnitStepRecord(
            step=int(step),
            n_participants=self.n_participants,
            n_members=self.n_members,
            inflow=self._step["inflow"],
            gross_income=self._step["gross_income"],
            surplus=self._step["surplus"],
            upward=self._step["upward"],
            downward=self._step["downward"],
            aid=self._step["aid"],
            payout=self._step["payout"],
            pool=self.pool,
            top_share=top_share,
            unit_utility=self.utility(),
            member_net_sum=float(sum(nets)),
            member_net_min=float(min(nets)) if nets else 0.0,
        )
        for key in ("inflow", "gross_income", "surplus", "upward", "downward", "aid", "payout"):
            self.totals[key] += self._step[key]
        self.history.append(record)
        self._reset_step()
        return record

    def to_dict(self, current_step: int | None = None) -> dict[str, Any]:
        """Serialisable summary (no per-step history)."""
        step = current_step if current_step is not None else (self.dissolved_step or self.founded_step)
        return {
            "unit_id": self.unit_id,
            "kind": self.kind,
            "settings": self.settings.to_dict(),
            "founded_step": self.founded_step,
            "dissolved_step": self.dissolved_step,
            "dissolution_reason": self.dissolution_reason,
            "lifetime": self.lifetime(step),
            "parent_id": self.parent_id,
            "arranged": self.arranged,
            "pool": self.pool,
            "members": [ledger.participant_id for ledger in self.members.values()],
            "children": [ledger.participant_id for ledger in self.children.values()],
            "totals": dict(self.totals),
            "ledgers": {
                pid: asdict(ledger) for pid, ledger in {**self.departed, **self.members, **self.children}.items()
            },
        }

    def __repr__(self) -> str:
        state = "active" if self.is_active else f"dissolved@{self.dissolved_step}"
        return (
            f"CooperativeUnit({self.unit_id!r}, kind={self.kind!r}, members={list(self.members)}, "
            f"children={list(self.children)}, pool={self.pool:.2f}, {state})"
        )


@dataclass
class ExitResult:
    """Outcome of :meth:`CooperationRegistry.leave`."""

    left: bool
    paid: float = 0.0
    enforced: bool = False
    vetoed: bool = False
    dissolved: bool = False
    payouts: dict[str, float] = field(default_factory=dict)
