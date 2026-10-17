"""The registry that enforces cooperation settings.

:class:`CooperationRegistry` owns every :class:`CooperativeUnit` in a run and
is the only place membership changes. It turns the seven settings into rules:

* **Members** -- capacity and the dissolution floor (``min_members``).
* **Scope** -- :meth:`CooperationRegistry.contribute` pools that fraction.
* **Exit cost** and **recognition** -- :meth:`CooperationRegistry.leave`
  collects the forfeit only when an enforcement draw succeeds.
* **Exclusivity** and **recognition** -- :meth:`CooperationRegistry.can_join`
  enforces the per-kind membership limit; :meth:`CooperationRegistry.offer_blocked`
  lets outsiders respect (or ignore) an exclusive unit.
* **Entry** -- :meth:`CooperationRegistry.form` with ``arranged=True`` for
  matches a parent makes; :meth:`CooperationRegistry.register_birth` for units
  whose entry is :attr:`~farm.core.cooperation.settings.Entry.BORN`.
* **Symmetry** -- payout weights in :meth:`CooperationRegistry.settle`.

Nesting: a unit can be attached under a parent unit. Each step, surplus and
upward flows are applied leaves-first and payouts roots-first, so emergence
runs both ways -- rules and payouts flow down, contributions flow up. A parent
can veto a member's exit when that exit would dissolve the child unit; the
veto is honoured with probability ``parent.recognition``.

The registry is engine-agnostic: participants are plain string ids, and the
caller supplies incomes and holdings. All randomness comes from the injected
``numpy.random.Generator`` so runs are reproducible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from farm.core.cooperation.goals import UnitGoals
from farm.core.cooperation.settings import CooperationSettings, Entry
from farm.core.cooperation.unit import CooperativeUnit, ExitResult, UnitEconomics

UNIT_ID_PREFIX = "unit-"

EVENT_KINDS: tuple[str, ...] = (
    "formed",
    "joined",
    "born_in",
    "attached",
    "left",
    "exit_vetoed",
    "offer_blocked",
    "removed",
    "dissolved",
)


@dataclass(frozen=True)
class CooperationEvent:
    """One membership change or enforcement decision."""

    step: int
    kind: str
    unit_id: str
    participant_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "kind": self.kind,
            "unit_id": self.unit_id,
            "participant_id": self.participant_id,
            **{f"detail_{k}": v for k, v in self.details.items()},
        }


class CooperationRegistry:
    """Book-keeper and rule enforcer for all cooperative units in a run.

    Parameters
    ----------
    economics
        Pool parameters shared by every unit.
    unit_goals
        Default goals for new units (overridable per :meth:`form` call).
    rng
        Source of the enforcement and veto draws.
    keep_events
        Store individual :class:`CooperationEvent` objects (counts are always
        kept). Disable for very long runs.
    """

    def __init__(
        self,
        economics: UnitEconomics | None = None,
        unit_goals: UnitGoals | None = None,
        rng: np.random.Generator | None = None,
        keep_events: bool = True,
    ) -> None:
        self.economics = economics if economics is not None else UnitEconomics()
        self.unit_goals = unit_goals if unit_goals is not None else UnitGoals()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.keep_events = keep_events
        self.units: dict[str, CooperativeUnit] = {}
        self.events: list[CooperationEvent] = []
        self.event_counts: Counter = Counter()
        self._memberships: dict[str, set[str]] = defaultdict(set)
        self._next_id = 0
        #: Units whose exits a parent has vetoed (kept even when ``keep_events`` is off).
        self.vetoed_unit_ids: set[str] = set()

    # ---------------------------------------------------------------- events
    def _emit(self, step: int, kind: str, unit_id: str, participant_id: str | None = None, **details: Any) -> None:
        self.event_counts[kind] += 1
        if self.keep_events:
            self.events.append(CooperationEvent(int(step), kind, unit_id, participant_id, dict(details)))

    # --------------------------------------------------------------- queries
    def get(self, unit_id: str) -> CooperativeUnit:
        try:
            return self.units[unit_id]
        except KeyError as exc:
            raise KeyError(f"unknown unit {unit_id!r}") from exc

    def active_units(self, kind: str | None = None) -> list[CooperativeUnit]:
        return [u for u in self.units.values() if u.is_active and (kind is None or u.kind == kind)]

    def units_of(self, agent_id: str, kind: str | None = None) -> list[CooperativeUnit]:
        """Active units in which ``agent_id`` is a direct member."""
        out = [self.units[uid] for uid in sorted(self._memberships.get(agent_id, ()))]
        return [u for u in out if kind is None or u.kind == kind]

    def is_member(self, agent_id: str, unit_id: str | None = None) -> bool:
        held = self._memberships.get(agent_id, set())
        return bool(held) if unit_id is None else unit_id in held

    def partners_of(self, agent_id: str, kind: str | None = None) -> set[str]:
        """Other agent members across the agent's active units."""
        out: set[str] = set()
        for unit in self.units_of(agent_id, kind):
            out.update(pid for pid in unit.members if pid != agent_id)
        return out

    def depth(self, unit: CooperativeUnit) -> int:
        d = 0
        seen = {unit.unit_id}
        current = unit
        while current.parent_id is not None:
            parent = self.units.get(current.parent_id)
            if parent is None or parent.unit_id in seen:
                break
            seen.add(parent.unit_id)
            d += 1
            current = parent
        return d

    def root_of(self, unit: CooperativeUnit) -> CooperativeUnit:
        current = unit
        seen = {unit.unit_id}
        while current.parent_id is not None and current.parent_id in self.units:
            current = self.units[current.parent_id]
            if current.unit_id in seen:
                break
            seen.add(current.unit_id)
        return current

    def can_join(self, agent_id: str, target: Any) -> tuple[bool, str]:
        """Whether ``agent_id`` may enter ``target`` (a unit or a settings object).

        Checks duplicate membership, capacity and the per-kind exclusivity
        limit. Returns ``(allowed, reason)``.
        """
        if isinstance(target, CooperativeUnit):
            settings = target.settings
            if not target.is_active:
                return False, "unit_dissolved"
            if agent_id in target.members:
                return False, "already_member"
            if not target.has_capacity():
                return False, "unit_full"
        elif isinstance(target, CooperationSettings):
            settings = target
        else:
            raise TypeError("target must be a CooperativeUnit or CooperationSettings")
        held_same_kind = [u for u in self.units_of(agent_id) if u.kind == settings.kind]
        limit = settings.max_memberships
        if any(u.settings.exclusive for u in held_same_kind):
            limit = 1
        if len(held_same_kind) >= limit:
            return False, "exclusivity_limit"
        return True, "ok"

    # ------------------------------------------------------------- formation
    def _new_id(self) -> str:
        uid = f"{UNIT_ID_PREFIX}{self._next_id}"
        self._next_id += 1
        return uid

    def form(
        self,
        settings: CooperationSettings,
        founders: Sequence[str],
        step: int,
        *,
        children: Sequence[str] = (),
        parent_id: str | None = None,
        arranged: bool = False,
        consents: Mapping[str, bool] | None = None,
        goals: UnitGoals | None = None,
        priors: Mapping[str, float] | None = None,
    ) -> CooperativeUnit:
        """Create a unit from ``founders`` (agents) and optional child units.

        Raises ``ValueError`` when the founders violate capacity, the
        membership floor or exclusivity. ``consents`` records, for arranged
        matches, whether each founder would have chosen the match itself.
        ``priors`` seeds each founder's running benefit estimate with the net
        benefit it expected (see :class:`~farm.core.cooperation.unit.MemberLedger`).
        """
        founders = list(dict.fromkeys(founders))
        children = list(dict.fromkeys(children))
        n = len(founders) + len(children)
        if n < settings.min_members:
            raise ValueError(f"{settings.kind} needs at least {settings.min_members} participants, got {n}")
        if settings.max_members is not None and n > settings.max_members:
            raise ValueError(f"{settings.kind} admits at most {settings.max_members} participants, got {n}")
        for agent_id in founders:
            ok, reason = self.can_join(agent_id, settings)
            if not ok:
                raise ValueError(f"{agent_id!r} cannot found a {settings.kind}: {reason}")
        for child_id in children:
            child = self.get(child_id)
            if not child.is_active or child.parent_id is not None:
                raise ValueError(f"{child_id!r} must be an active root unit to be attached")
        if parent_id is not None:
            parent = self.get(parent_id)
            if not parent.is_active or not parent.has_capacity():
                raise ValueError(f"parent {parent_id!r} cannot take another child unit")

        unit = CooperativeUnit(
            self._new_id(),
            settings,
            step,
            goals=goals if goals is not None else self.unit_goals,
            arranged=arranged,
        )
        self.units[unit.unit_id] = unit
        consents = dict(consents or {})
        priors = dict(priors or {})
        for agent_id in founders:
            unit._add_participant(
                agent_id,
                step,
                is_unit=False,
                consented=bool(consents.get(agent_id, True)),
                prior=priors.get(agent_id),
            )
            self._memberships[agent_id].add(unit.unit_id)
        for child_id in children:
            unit._add_participant(child_id, step, is_unit=True, consented=True)
            self.units[child_id].parent_id = unit.unit_id
        self._emit(
            step,
            "formed",
            unit.unit_id,
            details={
                "kind": settings.kind,
                "founders": tuple(founders),
                "children": tuple(children),
                "arranged": arranged,
            },
        )
        if parent_id is not None:
            self.attach(unit.unit_id, parent_id, step)
        return unit

    def join(
        self,
        unit_id: str,
        agent_id: str,
        step: int,
        *,
        consented: bool = True,
        prior: float | None = None,
    ) -> bool:
        """Add ``agent_id`` to an existing unit. Returns ``False`` if not allowed."""
        unit = self.get(unit_id)
        ok, _ = self.can_join(agent_id, unit)
        if not ok:
            return False
        unit._add_participant(agent_id, step, is_unit=False, consented=consented, prior=prior)
        self._memberships[agent_id].add(unit_id)
        self._emit(step, "joined", unit_id, agent_id)
        return True

    def attach(self, child_id: str, parent_id: str, step: int) -> None:
        """Nest an active root unit under ``parent_id``."""
        child = self.get(child_id)
        parent = self.get(parent_id)
        if child_id == parent_id:
            raise ValueError("a unit cannot be its own parent")
        if not (child.is_active and parent.is_active):
            raise ValueError("both units must be active to attach")
        if child.parent_id is not None:
            raise ValueError(f"{child_id!r} already has a parent")
        if self.root_of(parent).unit_id == child_id:
            raise ValueError("attaching would create a cycle")
        if not parent.has_capacity():
            raise ValueError(f"{parent_id!r} is full")
        parent._add_participant(child_id, step, is_unit=True, consented=True)
        child.parent_id = parent_id
        self._emit(step, "attached", parent_id, child_id)

    def nearest_born_ancestor(self, unit: CooperativeUnit) -> CooperativeUnit | None:
        """Closest active ancestor of ``unit`` whose entry is BORN."""
        seen = {unit.unit_id}
        current = unit
        while current.parent_id is not None and current.parent_id in self.units:
            current = self.units[current.parent_id]
            if current.unit_id in seen:
                return None
            seen.add(current.unit_id)
            if current.is_active and current.settings.entry is Entry.BORN:
                return current
        return None

    def register_birth(self, parent_agent_id: str, child_agent_id: str, step: int) -> list[str]:
        """Enrol a newborn in the BORN-entry unit(s) of its parent's household.

        The household comes first: for each of the parent's chosen or
        arranged units (a marriage, say) the nearest BORN-entry ancestor (the
        family that marriage anchors) takes the child. Only if there is none
        does the child join the parent's own BORN-entry units (the parent's
        birth family). Higher BORN levels (a tribe) include the child through
        nesting, not direct membership. Returns the ids of the units joined;
        full units are skipped.
        """
        household: list[CooperativeUnit] = []
        for unit in self.units_of(parent_agent_id):
            if unit.settings.entry is Entry.BORN:
                continue
            ancestor = self.nearest_born_ancestor(unit)
            if ancestor is not None and ancestor not in household:
                household.append(ancestor)
        targets = household or [u for u in self.units_of(parent_agent_id) if u.settings.entry is Entry.BORN]
        joined: list[str] = []
        for unit in targets:
            ok, _ = self.can_join(child_agent_id, unit)
            if not ok:
                continue
            unit._add_participant(child_agent_id, step, is_unit=False, consented=True)
            self._memberships[child_agent_id].add(unit.unit_id)
            self._emit(step, "born_in", unit.unit_id, child_agent_id, parent=parent_agent_id)
            joined.append(unit.unit_id)
        return joined

    # ----------------------------------------------------------- enforcement
    def offer_blocked(self, agent_id: str, kind: str, step: int) -> bool:
        """Do outsiders respect ``agent_id``'s exclusive unit of ``kind``?

        If the agent holds an exclusive unit of that kind, an outside offer is
        blocked with probability equal to that unit's recognition. Returns
        ``False`` (not blocked) when the agent holds no exclusive unit of the
        kind -- whether it can accept is then a matter for :meth:`can_join`.
        """
        exclusive_units = [u for u in self.units_of(agent_id, kind) if u.settings.exclusive]
        if not exclusive_units:
            return False
        recognition = max(u.settings.recognition for u in exclusive_units)
        blocked = bool(self.rng.random() < recognition)
        if blocked:
            self._emit(step, "offer_blocked", exclusive_units[0].unit_id, agent_id)
        return blocked

    def expected_exit_payment(self, unit_id: str, holdings: float) -> float:
        """Enforcement-weighted forfeit for leaving ``unit_id`` with ``holdings``."""
        return self.get(unit_id).settings.expected_exit_payment(holdings)

    def would_dissolve_without(self, unit: CooperativeUnit, participant_id: str) -> bool:
        remaining = unit.n_participants - (
            1 if participant_id in unit.members or participant_id in unit.children else 0
        )
        return remaining < unit.settings.min_members

    def veto_probability(self, unit_id: str, agent_id: str) -> float:
        """Probability that a parent unit blocks ``agent_id`` leaving ``unit_id``."""
        unit = self.get(unit_id)
        if unit.parent_id is None or unit.parent_id not in self.units:
            return 0.0
        parent = self.units[unit.parent_id]
        if not parent.is_active or not self.would_dissolve_without(unit, agent_id):
            return 0.0
        return parent.settings.recognition

    def record_exit_desire(self, unit_id: str, agent_id: str, *, blocked_by_price: bool, steps: int = 1) -> None:
        """Note that ``agent_id`` wanted out of ``unit_id`` for ``steps`` steps.

        ``blocked_by_price`` marks the trap condition: the agent stayed only
        because the exit payment exceeded what leaving was worth to it.
        Callers that review memberships every ``k`` steps pass ``steps=k`` so
        the counts stay in steps.
        """
        if steps < 1:
            raise ValueError("steps must be at least 1")
        ledger = self.get(unit_id).members[agent_id]
        ledger.exit_desire_steps += steps
        if blocked_by_price:
            ledger.trapped_steps += steps

    def veto_exit(self, unit_id: str, agent_id: str, step: int) -> bool:
        """Draw the parent veto on ``agent_id`` leaving ``unit_id``; record it if it falls."""
        veto_p = self.veto_probability(unit_id, agent_id)
        if veto_p > 0.0 and self.rng.random() < veto_p:
            self._emit(step, "exit_vetoed", unit_id, agent_id, parent=self.get(unit_id).parent_id)
            self.vetoed_unit_ids.add(unit_id)
            return True
        return False

    def leave(
        self, unit_id: str, agent_id: str, step: int, holdings: float = 0.0, *, check_veto: bool = True
    ) -> ExitResult:
        """Exit ``agent_id`` from ``unit_id``, paying the enforced exit cost.

        Order of checks: a parent veto (only when the exit would dissolve the
        child unit, honoured with probability ``parent.recognition``), then an
        enforcement draw with probability ``recognition``; if enforced the
        leaver forfeits ``exit_cost * holdings`` into the unit pool. The
        leaver also forfeits any claim on the pool. ``ExitResult.payouts``
        lists dissolution payouts to remaining members, which the caller must
        credit. Pass ``check_veto=False`` when the veto was already drawn with
        :meth:`veto_exit` (so several exits can be vetoed before any happens).
        """
        unit = self.get(unit_id)
        if agent_id not in unit.members:
            raise KeyError(f"{agent_id!r} is not a member of {unit_id}")
        if check_veto and self.veto_exit(unit_id, agent_id, step):
            return ExitResult(left=False, vetoed=True)
        enforced = bool(unit.settings.exit_cost > 0.0 and self.rng.random() < unit.settings.recognition)
        paid = unit.settings.exit_cost * max(0.0, float(holdings)) if enforced else 0.0
        unit.pool += paid
        unit.totals["forfeits"] += paid
        self._detach_member(unit, agent_id)
        self._emit(step, "left", unit_id, agent_id, paid=paid, enforced=enforced)
        payouts: dict[str, float] = {}
        dissolved = False
        if unit.n_participants < unit.settings.min_members:
            payouts = self.dissolve(unit_id, step, "below_min_members")
            dissolved = True
        return ExitResult(left=True, paid=paid, enforced=enforced, vetoed=False, dissolved=dissolved, payouts=payouts)

    def remove_agent(self, agent_id: str, step: int, reason: str = "death") -> dict[str, float]:
        """Drop ``agent_id`` from every unit (no exit cost). Returns dissolution payouts."""
        payouts: dict[str, float] = defaultdict(float)
        held = self.units_of(agent_id)
        for unit in held:
            self._detach_member(unit, agent_id)
            self._emit(step, "removed", unit.unit_id, agent_id, reason=reason)
        self._memberships.pop(agent_id, None)
        for unit in held:
            if unit.is_active and unit.n_participants < unit.settings.min_members:
                for pid, amount in self.dissolve(unit.unit_id, step, f"member_{reason}").items():
                    payouts[pid] += amount
        return dict(payouts)

    def _detach_member(self, unit: CooperativeUnit, agent_id: str) -> None:
        unit._remove_participant(agent_id)
        held = self._memberships.get(agent_id)
        if held is not None:
            held.discard(unit.unit_id)
            if not held:
                self._memberships.pop(agent_id, None)

    def dissolve(self, unit_id: str, step: int, reason: str) -> dict[str, float]:
        """End a unit: split its pool equally among remaining agent members.

        Child units are released as roots; the parent drops this unit and may
        dissolve in turn. Returns payouts by agent id (including cascades).
        """
        unit = self.get(unit_id)
        if not unit.is_active:
            return {}
        payouts: dict[str, float] = defaultdict(float)
        member_ids = list(unit.members)
        if member_ids and unit.pool > 0.0:
            share = unit.pool / len(member_ids)
            for agent_id in member_ids:
                payouts[agent_id] += share
                unit.members[agent_id].received += share
            unit.totals["payout"] += unit.pool
            unit.pool = 0.0
        for agent_id in member_ids:
            self._detach_member(unit, agent_id)
        for child_id in list(unit.children):
            unit._remove_participant(child_id)
            self.units[child_id].parent_id = None
        unit.dissolved_step = int(step)
        unit.dissolution_reason = reason
        self._emit(step, "dissolved", unit_id, details={"reason": reason, "members": tuple(member_ids)})
        if unit.parent_id is not None and unit.parent_id in self.units:
            parent = self.units[unit.parent_id]
            if unit_id in parent.children:
                parent._remove_participant(unit_id)
                if parent.is_active and parent.n_participants < parent.settings.min_members:
                    for pid, amount in self.dissolve(parent.unit_id, step, "child_dissolved").items():
                        payouts[pid] += amount
        return dict(payouts)

    # ------------------------------------------------------------------ flows
    def contribute(self, agent_id: str, income: float, step: int) -> float:
        """Pool ``scope * income`` into each of the agent's units.

        With several memberships the income is split evenly across them before
        each unit's scope applies. Non-positive income pools nothing. Returns
        the total amount pooled (the caller keeps ``income - pooled``).
        """
        units = self.units_of(agent_id)
        if not units or income <= 0.0:
            for unit in units:
                unit._record_contribution(agent_id, 0.0, 0.0)
            return 0.0
        slice_income = float(income) / len(units)
        pooled = 0.0
        for unit in units:
            amount = unit.settings.scope * slice_income
            unit._record_contribution(agent_id, amount, slice_income)
            pooled += amount
        return pooled

    def settle(self, step: int, holdings: Mapping[str, float] | None = None) -> dict[str, float]:
        """Run one step of pool dynamics for every active unit.

        Leaves-first: surplus, then the parent's scope of (inflow + surplus)
        moves up. Roots-first: emergency aid to needy agent members (when
        ``holdings`` is given), then payout of the pool above the reserve
        target, split by rank weights; child units receive theirs into their
        pool before they pay out. Returns payouts by agent id; the caller
        credits them.
        """
        econ = self.economics
        active = self.active_units()
        depth = {u.unit_id: self.depth(u) for u in active}
        payouts: dict[str, float] = defaultdict(float)

        for unit in sorted(active, key=lambda u: (-depth[u.unit_id], u.unit_id)):
            inflow = unit._step["inflow"]
            surplus = econ.surplus(inflow, unit.n_participants)
            unit.pool += surplus
            unit._step["surplus"] += surplus
            if unit.parent_id is not None and unit.parent_id in self.units:
                parent = self.units[unit.parent_id]
                if parent.is_active and unit.unit_id in parent.children:
                    up = min(unit.pool, parent.settings.scope * (inflow + surplus))
                    if up > 0.0:
                        unit.pool -= up
                        unit._step["upward"] += up
                        parent._record_contribution(unit.unit_id, up, inflow + surplus)

        step_payouts: dict[str, dict[str, float]] = {}
        for unit in sorted(active, key=lambda u: (depth[u.unit_id], u.unit_id)):
            by_participant: dict[str, float] = {}
            if holdings is not None and econ.aid_amount > 0.0:
                for agent_id, ledger in unit.members.items():
                    have = float(holdings.get(agent_id, np.inf))
                    if have >= econ.aid_threshold or unit.pool <= 0.0:
                        continue
                    aid = min(econ.aid_amount, econ.aid_threshold - have, unit.pool)
                    unit.pool -= aid
                    unit._step["aid"] += aid
                    ledger.aid_received += aid
                    ledger.step_aid += aid
                    payouts[agent_id] += aid
            reserve = unit.goals.reserve_target * unit.n_participants
            payable = econ.payout_rate * max(0.0, unit.pool - reserve)
            if payable > 0.0 and unit.n_participants > 0:
                for pid, share in unit.weights().items():
                    amount = payable * share
                    by_participant[pid] = amount
                    ledger = unit.ledger(pid)
                    ledger.received += amount
                    ledger.step_received += amount
                    if ledger.is_unit:
                        child = self.units[pid]
                        child.pool += amount
                        child._step["downward"] += amount
                    else:
                        payouts[pid] += amount
                unit.pool -= payable
                unit._step["payout"] += payable
            step_payouts[unit.unit_id] = by_participant

        for unit in active:
            unit._close_step(step, econ.benefit_ema_alpha, step_payouts.get(unit.unit_id, {}))
        return dict(payouts)

    # -------------------------------------------------------------- summaries
    def membership_counts(self) -> dict[str, int]:
        """Number of agents holding at least one active membership, by kind."""
        out: dict[str, int] = Counter()
        for agent_id in self._memberships:
            for kind in {u.kind for u in self.units_of(agent_id)}:
                out[kind] += 1
        return dict(out)

    def members_in_any_unit(self) -> set[str]:
        return {agent_id for agent_id, held in self._memberships.items() if held}

    def summary(self, current_step: int) -> dict[str, Any]:
        """Run-level counts for logging and metrics."""
        units = list(self.units.values())
        return {
            "units_formed": len(units),
            "units_active": sum(1 for u in units if u.is_active),
            "units_by_kind": dict(Counter(u.kind for u in units)),
            "agents_in_units": len(self.members_in_any_unit()),
            "event_counts": dict(self.event_counts),
            "mean_lifetime": float(np.mean([u.lifetime(current_step) for u in units])) if units else 0.0,
        }

    def iter_units(self, kinds: Iterable[str] | None = None) -> Iterable[CooperativeUnit]:
        wanted = set(kinds) if kinds is not None else None
        for unit in self.units.values():
            if wanted is None or unit.kind in wanted:
                yield unit
