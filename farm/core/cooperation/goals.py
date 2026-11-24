"""Member goals, unit goals, and how members value a unit.

Every cooperative unit carries two layers of goals:

* **Member goals** -- what each agent wants for itself. Agents carry a
  mixture over five *motives* (political, economic, emotional, hedonist,
  religious). A motive is not a kind of institution; it is a reason to want
  particular settings from the same kind of unit.
* **Unit goals** -- what the arrangement itself comes to need: persistence,
  a reserve, growth. The unit pursues them through its payout policy, so
  they can pull away from what members want (the *drift* failure mode).

Which settings each motive wants:

=========== ===================================
Motive      Settings it cares about most
=========== ===================================
political   recognition (+), exclusivity (+)
economic    scope (+), exit cost (+)
emotional   scope (+), exclusivity (+)
hedonist    symmetry (-), exit cost (-)
religious   recognition (+), exit cost (+)
=========== ===================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from farm.core.cooperation.settings import CooperationSettings


class Motive(str, Enum):
    """Reasons to enter a cooperative unit."""

    POLITICAL = "political"
    ECONOMIC = "economic"
    EMOTIONAL = "emotional"
    HEDONIST = "hedonist"
    RELIGIOUS = "religious"


MOTIVES: tuple[Motive, ...] = tuple(Motive)

#: Which settings each motive cares about, and in which direction (+1 wants
#: the setting high, -1 wants it low).
MOTIVE_SETTING_PREFERENCES: dict[Motive, dict[str, int]] = {
    Motive.POLITICAL: {"recognition": +1, "exclusivity": +1},
    Motive.ECONOMIC: {"scope": +1, "exit_cost": +1},
    Motive.EMOTIONAL: {"scope": +1, "exclusivity": +1},
    Motive.HEDONIST: {"symmetry": -1, "exit_cost": -1},
    Motive.RELIGIOUS: {"recognition": +1, "exit_cost": +1},
}

#: Value components each motive emphasises when an agent appraises a unit.
#: ``wealth`` = expected net transfers, ``security`` = insurance against bad
#: draws, ``affinity`` = compatibility with partners, ``belonging`` = gains
#: that accrue to the higher-level unit the agent belongs to.
VALUE_COMPONENTS: tuple[str, ...] = ("wealth", "security", "affinity", "belonging")

MOTIVE_COMPONENTS: dict[Motive, dict[str, float]] = {
    Motive.POLITICAL: {"wealth": 0.3, "security": 0.0, "affinity": 0.0, "belonging": 1.0},
    Motive.ECONOMIC: {"wealth": 0.6, "security": 1.0, "affinity": 0.0, "belonging": 0.0},
    Motive.EMOTIONAL: {"wealth": 0.0, "security": 0.3, "affinity": 1.0, "belonging": 0.0},
    Motive.HEDONIST: {"wealth": 1.0, "security": 0.0, "affinity": 0.0, "belonging": 0.0},
    Motive.RELIGIOUS: {"wealth": 0.0, "security": 0.5, "affinity": 0.0, "belonging": 0.5},
}


def _setting_level(settings: CooperationSettings, name: str) -> float:
    """Map a setting to ``[0, 1]`` so preferences can score it."""
    if name == "exclusivity":
        return 1.0 if settings.exclusive else 0.0
    if name in ("scope", "exit_cost", "recognition", "symmetry"):
        return float(getattr(settings, name))
    raise KeyError(f"no scalar level for setting {name!r}")


def motive_fit(motive: Motive, settings: CooperationSettings) -> float:
    """How well ``settings`` suit one motive, in ``[-1, 1]``."""
    prefs = MOTIVE_SETTING_PREFERENCES[Motive(motive)]
    scores = [direction * (2.0 * _setting_level(settings, name) - 1.0) for name, direction in prefs.items()]
    return sum(scores) / len(scores)


@dataclass(frozen=True)
class MemberGoals:
    """An agent's private goals: a mixture over the five motives.

    ``motives`` maps each motive to a non-negative weight; weights are
    normalised to sum to 1 on construction. Use :meth:`single` for a pure
    motive and :meth:`mixture` for a blend.
    """

    motives: Mapping[Motive, float] = field(default_factory=lambda: {m: 1.0 / len(MOTIVES) for m in MOTIVES})

    def __post_init__(self) -> None:
        cleaned: dict[Motive, float] = {}
        for key, value in dict(self.motives).items():
            weight = float(value)
            if weight < 0.0 or not math.isfinite(weight):
                raise ValueError(f"motive weight for {key!r} must be finite and non-negative")
            cleaned[Motive(key)] = cleaned.get(Motive(key), 0.0) + weight
        total = sum(cleaned.values())
        if total <= 0.0:
            raise ValueError("at least one motive weight must be positive")
        normalized = {m: cleaned.get(m, 0.0) / total for m in MOTIVES}
        object.__setattr__(self, "motives", normalized)

    @classmethod
    def single(cls, motive: Motive) -> MemberGoals:
        return cls({Motive(motive): 1.0})

    @classmethod
    def mixture(cls, weights: Mapping[Motive, float]) -> MemberGoals:
        return cls(dict(weights))

    def weight(self, motive: Motive) -> float:
        return float(self.motives.get(Motive(motive), 0.0))

    @property
    def dominant(self) -> Motive:
        """The highest-weight motive (ties resolved in :data:`MOTIVES` order)."""
        return max(MOTIVES, key=lambda m: (self.motives[m], -MOTIVES.index(m)))

    def component_weights(self) -> dict[str, float]:
        """Weights on the value components implied by the motive mixture."""
        out = {name: 0.0 for name in VALUE_COMPONENTS}
        for motive, pi in self.motives.items():
            for name, w in MOTIVE_COMPONENTS[motive].items():
                out[name] += pi * w
        return out

    def settings_fit(self, settings: CooperationSettings) -> float:
        """Motive-weighted fit of ``settings``, in ``[-1, 1]``."""
        return sum(pi * motive_fit(m, settings) for m, pi in self.motives.items())


@dataclass(frozen=True)
class MembershipAppraisal:
    """Inputs an agent uses to value being in (or joining) a unit.

    All monetary terms are *per step* in resource units.

    Attributes
    ----------
    net_transfer
        Expected payout minus expected contribution.
    insurance
        Expected value of being carried through bad draws (aid from the
        unit reserve, risk pooling).
    compatibility
        Affinity with the other members, in ``[-1, 1]``.
    parent_gain
        Per-step gain this membership brings to the agent's higher-level unit
        (zero when there is none).
    """

    net_transfer: float = 0.0
    insurance: float = 0.0
    compatibility: float = 0.0
    parent_gain: float = 0.0


@dataclass(frozen=True)
class ValuationScales:
    """Converts appraisal terms into one utility.

    ``horizon`` turns per-step flows into a stock comparable with a one-off
    exit payment. ``fit_scale`` and ``affinity_scale`` price the
    non-monetary terms in resource units.
    """

    horizon: float = 20.0
    fit_scale: float = 2.0
    affinity_scale: float = 4.0


DEFAULT_SCALES = ValuationScales()


def value_of_membership(
    goals: MemberGoals,
    settings: CooperationSettings,
    appraisal: MembershipAppraisal,
    scales: ValuationScales = DEFAULT_SCALES,
) -> float:
    """Utility (in resource units) of holding a membership versus going solo.

    ``V = H * (w_wealth * net + w_security * insurance + w_belonging * parent_gain)
    + fit_scale * fit(settings) + affinity_scale * w_affinity * compatibility``

    A positive value means the agent prefers being in the unit. Compare a
    current membership against ``-expected_exit_payment`` to decide whether
    leaving is worth the price; a negative value that is still above that
    price is the *trap* condition.
    """
    w = goals.component_weights()
    flows = (
        w["wealth"] * appraisal.net_transfer
        + w["security"] * appraisal.insurance
        + w["belonging"] * appraisal.parent_gain
    )
    return (
        scales.horizon * flows
        + scales.fit_scale * goals.settings_fit(settings)
        + scales.affinity_scale * w["affinity"] * appraisal.compatibility
    )


@dataclass(frozen=True)
class UnitGoals:
    """What the unit itself pursues.

    The unit scores itself on staying alive (``persistence``), holding a
    reserve of ``reserve_target`` per participant (``reserve``) and size
    (``growth``). The reserve target is what the payout policy protects, so
    a heavily weighted reserve is the lever by which unit goals can pull
    away from member goals.
    """

    persistence: float = 1.0
    reserve: float = 0.5
    growth: float = 0.25
    reserve_target: float = 2.0

    def __post_init__(self) -> None:
        for name in ("persistence", "reserve", "growth", "reserve_target"):
            value = float(getattr(self, name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")

    def utility(self, pool: float, n_participants: int, active: bool = True) -> float:
        """Unit-level utility for a pool level and size."""
        if not active or n_participants <= 0:
            return 0.0
        target = self.reserve_target * n_participants
        reserve_score = 1.0 if target <= 0 else min(1.0, max(0.0, pool) / target)
        return self.persistence + self.reserve * reserve_score + self.growth * math.log1p(n_participants)
