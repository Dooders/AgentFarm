"""Configuration for cooperation in the spatial AgentFarm world.

:class:`CooperationConfig` is carried on
:class:`~farm.config.config.SimulationConfig` as ``cooperation``. It is off by
default; with ``enabled=False`` the environment, the agents' action space and
their chromosomes are exactly what they were before cooperation existed.

Two modes:

* ``"active"`` -- agents can ``bond`` and ``leave``; units pool gathered
  income, pay out, insure and enforce their settings.
* ``"placebo"`` -- the same two actions and the same heritable loci exist,
  but ``bond`` never forms a unit. This is the control for an experiment:
  it has the action space, the chromosome and the mutation load of the
  active world, so any difference from it is due to the institution, and
  the cooperation loci drift without selection.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping

from farm.core.cooperation.genes import COOPERATION_GENE_NAMES
from farm.core.cooperation.goals import ValuationScales
from farm.core.cooperation.settings import CooperationSettings
from farm.core.cooperation.unit import UnitEconomics

COOPERATION_MODES: tuple[str, ...] = ("active", "placebo")
TERMS_RULES: tuple[str, ...] = ("mean", "initiator")


@dataclass(frozen=True)
class CooperationConfig:
    """Switches and parameters for brain-learned cooperation.

    Formation
        ``bond`` marks the agent as open to a unit for ``offer_window`` steps
        and looks for an open agent within ``bond_range``. Two open agents
        near each other form a unit (or the newcomer joins the other's unit
        if it has room). ``terms`` decides the unit's settings: ``"mean"``
        averages the founders' settings loci, ``"initiator"`` takes the terms
        of the agent whose offer was standing. With ``allow_switching`` an
        open agent already in an exclusive unit may leave it (paying the
        enforced exit cost) to form a new one.
    Institutions
        Settings come from the founders' heritable loci unless
        ``fixed_settings`` is given, in which case every unit uses those.
        With ``evolve_genes=False`` the loci are fixed at their starting
        values (no mutation).
    Starting genes
        ``randomize_founder_genes`` draws each founder's cooperation loci
        uniformly within bounds; otherwise founders start from the defaults
        overridden by ``founder_gene_values``.
    Rewards
        Payouts, aid and dissolution shares reach an agent between its
        actions, so they are credited to its next reward at its own
        ``reward_resource_weight`` (``credit_payouts``). Members also get a
        per-step membership reward from their motives:
        ``membership_reward_scale * (fit_scale * fit + affinity_scale *
        w_affinity * like_mindedness) / horizon``.
    Perception
        ``partners_as_allies`` makes unit partners (and only partners) the
        agent's allies in its observation; ``membership_marker`` marks the
        agent's own cell in the allies channel while it is in a unit.
    Reproduction
        ``partner_coparent`` makes a member's partner its co-parent when the
        evolution policy uses crossover. ``family_on_birth`` founds a family
        unit (``family_settings``) around a member's unit at its first birth;
        later children of the household are born into it.
    """

    enabled: bool = False
    mode: str = "active"
    kind: str = "bond"

    bond_range: float = 6.0
    offer_window: int = 10
    max_candidates: int = 5
    terms: str = "mean"
    allow_switching: bool = True

    fixed_settings: Mapping[str, Any] | None = None
    evolve_genes: bool = True
    randomize_founder_genes: bool = True
    founder_gene_values: Mapping[str, float] = field(default_factory=dict)

    synergy: float = 0.2
    payout_rate: float = 0.8
    aid_threshold: float = 2.0
    aid_amount: float = 1.0
    benefit_ema_alpha: float = 0.1
    reserve_target: float = 2.0

    credit_payouts: bool = True
    membership_reward_scale: float = 1.0
    horizon: float = 20.0
    fit_scale: float = 2.0
    affinity_scale: float = 4.0

    partners_as_allies: bool = True
    membership_marker: bool = True

    partner_coparent: bool = True
    family_on_birth: bool = False
    family_settings: Mapping[str, Any] | None = None

    review_interval: int = 10
    record_interval: int = 1
    keep_events: bool = False
    seed: int | None = None
    output_dir: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in COOPERATION_MODES:
            raise ValueError(f"mode must be one of {COOPERATION_MODES}, got {self.mode!r}")
        if self.terms not in TERMS_RULES:
            raise ValueError(f"terms must be one of {TERMS_RULES}, got {self.terms!r}")
        if self.bond_range < 0.0:
            raise ValueError("bond_range must be non-negative")
        for name in ("offer_window", "max_candidates", "review_interval", "record_interval"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("membership_reward_scale", "horizon", "fit_scale", "affinity_scale", "reserve_target"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.horizon <= 0.0:
            raise ValueError("horizon must be positive")
        unknown = set(self.founder_gene_values) - set(COOPERATION_GENE_NAMES)
        if unknown:
            raise KeyError(f"unknown cooperation gene(s) in founder_gene_values: {sorted(unknown)}")
        # Validate eagerly so a bad config fails at construction, not mid-run.
        self.economics()
        if self.fixed_settings is not None:
            self.unit_settings()
        self.family_unit_settings()

    # ------------------------------------------------------------ derived
    @property
    def active(self) -> bool:
        """Whether units actually form (enabled and not a placebo)."""
        return bool(self.enabled and self.mode == "active")

    def economics(self) -> UnitEconomics:
        return UnitEconomics(
            synergy=self.synergy,
            payout_rate=self.payout_rate,
            aid_threshold=self.aid_threshold,
            aid_amount=self.aid_amount,
            benefit_ema_alpha=self.benefit_ema_alpha,
        )

    def scales(self) -> ValuationScales:
        return ValuationScales(horizon=self.horizon, fit_scale=self.fit_scale, affinity_scale=self.affinity_scale)

    def unit_settings(self) -> CooperationSettings | None:
        """Fixed settings for every unit, or ``None`` when genes decide."""
        if self.fixed_settings is None:
            return None
        payload = {"kind": self.kind, **dict(self.fixed_settings)}
        return CooperationSettings.from_dict(payload)

    def family_unit_settings(self) -> CooperationSettings:
        from farm.core.cooperation.settings import preset

        if self.family_settings is None:
            return preset("family")
        payload = {"kind": "family", **dict(self.family_settings)}
        return CooperationSettings.from_dict(payload)

    # ------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        for key in ("fixed_settings", "family_settings"):
            if raw[key] is not None:
                raw[key] = {k: getattr(v, "value", v) for k, v in dict(raw[key]).items()}
        raw["founder_gene_values"] = dict(raw["founder_gene_values"])
        return raw

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> CooperationConfig:
        if data is None:
            return cls()
        if isinstance(data, CooperationConfig):
            return data
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"unknown cooperation config field(s): {sorted(unknown)}")
        return cls(**dict(data))


__all__ = ["COOPERATION_MODES", "TERMS_RULES", "CooperationConfig"]
