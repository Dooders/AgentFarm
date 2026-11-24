"""The seven settings that configure any cooperative unit.

A cooperative unit forms when agents pool part of their lives and the
arrangement develops goals none of them had alone. Every unit -- a marriage,
a friendship, a business partnership, a family, a tribe -- is the same kind
of object configured differently. :class:`CooperationSettings` holds those
configuration knobs:

* **Members** -- how many agents are in it? (``min_members``, ``max_members``)
* **Scope** -- how much of life is pooled? (``scope``)
* **Exit cost** -- how hard is it to leave? (``exit_cost``)
* **Recognition** -- do outsiders witness and enforce it? (``recognition``)
* **Exclusivity** -- can members join rival units of the same kind? (``exclusive``)
* **Entry** -- how do members get in? (``entry``)
* **Symmetry** -- do members hold equal standing? (``symmetry``)

The settings are deliberately *mechanical*: each one maps to a rule the
:class:`~farm.core.cooperation.registry.CooperationRegistry` enforces, so a
sweep over settings is a sweep over institutions.

* ``scope`` is the fraction of a member's income routed into the unit pool.
* ``exit_cost`` is the fraction of a leaver's holdings forfeited to the unit
  *when the exit is enforced*.
* ``recognition`` is the probability that outsiders enforce the unit's
  rules: an exit cost is collected, an outside offer to a member of an
  exclusive unit is blocked, and a parent unit's veto is honoured.
* ``symmetry`` blends equal shares (``1.0``) with rank-ordered shares
  (``0.0``), where earlier joiners outrank later ones.

See ``docs/concepts/cooperation.md`` for the model as a whole.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any


class Entry(str, Enum):
    """How members get into a unit."""

    CHOSEN = "chosen"
    """Members choose each other (both sides must accept)."""

    ARRANGED = "arranged"
    """A higher-level unit chooses the match on the members' behalf."""

    BORN = "born"
    """Offspring of members join automatically at birth."""


#: Share ratio between consecutive ranks when ``symmetry == 0``. With 0.25 a
#: pair splits 80/20 and a trio roughly 76/19/5 in favour of the head.
RANK_DECAY: float = 0.25

#: Upper bound on simultaneous memberships of one kind for non-exclusive units.
DEFAULT_NONEXCLUSIVE_LIMIT: int = 3

SETTING_NAMES: tuple[str, ...] = (
    "members",
    "scope",
    "exit_cost",
    "recognition",
    "exclusivity",
    "entry",
    "symmetry",
)


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")


@dataclass(frozen=True)
class CooperationSettings:
    """Configuration of one kind of cooperative unit.

    Parameters
    ----------
    kind
        Label for the institution (``"marriage"``, ``"friendship"``...).
        Exclusivity is enforced per kind.
    min_members
        A unit with fewer participants than this dissolves.
    max_members
        Capacity; ``None`` means unbounded.
    scope
        Fraction of each member's income pooled, in ``[0, 1]``.
    exit_cost
        Fraction of a leaver's holdings forfeited to the unit when the exit
        is enforced, in ``[0, 1]``.
    recognition
        Probability that outsiders enforce the unit's rules, in ``[0, 1]``.
    exclusive
        If true, an agent may hold only one unit of this ``kind``.
    entry
        How members get in (see :class:`Entry`).
    symmetry
        ``1.0`` = equal shares, ``0.0`` = shares decay by rank.
    nonexclusive_limit
        Maximum simultaneous memberships of this kind when ``exclusive`` is
        false.
    """

    kind: str = "unit"
    min_members: int = 2
    max_members: int | None = 2
    scope: float = 0.5
    exit_cost: float = 0.0
    recognition: float = 0.0
    exclusive: bool = True
    entry: Entry = Entry.CHOSEN
    symmetry: float = 1.0
    nonexclusive_limit: int = DEFAULT_NONEXCLUSIVE_LIMIT

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("kind must be a non-empty string")
        if self.min_members < 1:
            raise ValueError("min_members must be at least 1")
        if self.max_members is not None and self.max_members < self.min_members:
            raise ValueError("max_members must be None or at least min_members")
        _check_unit_interval("scope", self.scope)
        _check_unit_interval("exit_cost", self.exit_cost)
        _check_unit_interval("recognition", self.recognition)
        _check_unit_interval("symmetry", self.symmetry)
        if self.nonexclusive_limit < 1:
            raise ValueError("nonexclusive_limit must be at least 1")
        if not isinstance(self.entry, Entry):
            object.__setattr__(self, "entry", Entry(self.entry))

    @property
    def max_memberships(self) -> int:
        """How many units of this kind one agent may hold at once."""
        return 1 if self.exclusive else self.nonexclusive_limit

    def has_capacity(self, n_participants: int) -> bool:
        """Whether a unit with ``n_participants`` can admit one more."""
        return self.max_members is None or n_participants < self.max_members

    def expected_exit_payment(self, holdings: float) -> float:
        """Expected forfeiture for leaving with ``holdings`` (enforcement-weighted)."""
        return self.recognition * self.exit_cost * max(0.0, float(holdings))

    def with_changes(self, **changes: Any) -> CooperationSettings:
        """Return a copy with ``changes`` applied (validated)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entry"] = self.entry.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CooperationSettings:
        payload = dict(data)
        if "entry" in payload:
            payload["entry"] = Entry(payload["entry"])
        return cls(**payload)


def rank_weights(n: int, symmetry: float) -> tuple[float, ...]:
    """Payout shares for ``n`` participants ordered by rank (0 = head).

    ``symmetry`` blends an equal split with a geometric rank split whose ratio
    is :data:`RANK_DECAY`. The result sums to 1.
    """
    if n <= 0:
        return ()
    _check_unit_interval("symmetry", symmetry)
    equal = 1.0 / n
    raw = [RANK_DECAY**rank for rank in range(n)]
    total = sum(raw)
    return tuple(symmetry * equal + (1.0 - symmetry) * (r / total) for r in raw)


#: Reference institutions, positioned on the seven settings.
#: The numbers are illustrative calibrations of the qualitative table, not
#: empirical estimates; experiments should sweep around them.
PRESETS: dict[str, CooperationSettings] = {
    "marriage": CooperationSettings(
        kind="marriage",
        min_members=2,
        max_members=2,
        scope=0.7,
        exit_cost=0.5,
        recognition=0.9,
        exclusive=True,
        entry=Entry.CHOSEN,
        symmetry=1.0,
    ),
    "cohabitation": CooperationSettings(
        kind="cohabitation",
        min_members=2,
        max_members=2,
        scope=0.6,
        exit_cost=0.15,
        recognition=0.1,
        exclusive=True,
        entry=Entry.CHOSEN,
        symmetry=1.0,
    ),
    "friendship": CooperationSettings(
        kind="friendship",
        min_members=2,
        max_members=6,
        scope=0.15,
        exit_cost=0.0,
        recognition=0.0,
        exclusive=False,
        entry=Entry.CHOSEN,
        symmetry=1.0,
    ),
    "business_partnership": CooperationSettings(
        kind="business_partnership",
        min_members=2,
        max_members=4,
        scope=0.3,
        exit_cost=0.25,
        recognition=0.8,
        exclusive=False,
        entry=Entry.CHOSEN,
        symmetry=1.0,
    ),
    "family": CooperationSettings(
        kind="family",
        min_members=1,
        max_members=None,
        scope=0.4,
        exit_cost=0.8,
        recognition=0.8,
        exclusive=True,
        entry=Entry.BORN,
        symmetry=0.5,
    ),
    "tribe": CooperationSettings(
        kind="tribe",
        min_members=1,
        max_members=None,
        scope=0.15,
        exit_cost=0.9,
        recognition=0.7,
        exclusive=True,
        entry=Entry.BORN,
        symmetry=0.3,
    ),
}


def preset(name: str, **overrides: Any) -> CooperationSettings:
    """Return a named preset, optionally with ``overrides`` applied."""
    try:
        base = PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown cooperation preset {name!r}; choose from {sorted(PRESETS)}") from exc
    return base.with_changes(**overrides) if overrides else base
