"""Shared learning-positive regime helpers (#904 / transferable-signal gate).

Centralizes population and ecology shaping so the precondition gate
(``measure_transferable_signal.py``) and the inheritance A/B harness
(``run_stable_profile_seed_sweep.py``) apply identical regime definitions.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from farm.config import SimulationConfig
from farm.runners.intrinsic_evolution_experiment import STABLE_SUB_PROFILES

# Mapping from STABLE_SUB_PROFILES override keys to SimulationConfig fields.
_PROFILE_FIELD_MAP = {
    "initial_agent_resource_level": ("agent_behavior", "initial_resource_level"),
    "initial_resource_count": ("resources", "initial_resources"),
    "resource_regen_rate": ("resources", "resource_regen_rate"),
    "resource_regen_amount": ("resources", "resource_regen_amount"),
}

# Defaults used by the #904 learning-positive inheritance A/B (mirrors gate).
DEFAULT_LEARNING_POSITIVE_POPULATION = 8
# Saturated-ecology variant: 4× the starting population so the colony fills
# quickly and runs as a crowded, high-churn ecology for most of the horizon.
DEFAULT_LEARNING_POSITIVE_MAX_POPULATION = 32
# Low-churn variant: cap equals the starting population so reproduction can
# only replace dead agents — it cannot expand the colony.  Keeps ecology
# density comparable to the gate regime (fixed-8, no-repro).
DEFAULT_LEARNING_POSITIVE_LOW_CHURN_MAX_POPULATION = DEFAULT_LEARNING_POSITIVE_POPULATION


def apply_independent_population(
    config: SimulationConfig,
    population: int,
    max_population: int,
) -> None:
    """Small independent-only population (learning-positive regime).

    Zeros out system/control/order/chaos agents so every slot is an
    independent learning agent — the same shape used by the transferable-signal
    gate and the inheritance-ladder A/B.
    """
    pop = config.population
    pop.system_agents = 0
    pop.independent_agents = int(population)
    pop.control_agents = 0
    for attr in ("order_agents", "chaos_agents"):
        if hasattr(pop, attr):
            setattr(pop, attr, 0)
    pop.max_population = int(max_population)


def apply_stable_profile_ecology(
    config: SimulationConfig,
    profile: str,
) -> Dict[str, Any]:
    """Apply STABLE_SUB_PROFILES[profile] ecology to a SimulationConfig."""
    overrides = STABLE_SUB_PROFILES[profile]
    for key, value in overrides.items():
        mapping = _PROFILE_FIELD_MAP.get(key)
        if mapping is None:
            print(
                f"apply_stable_profile_ecology: ignoring unknown profile key {key!r}",
                file=sys.stderr,
            )
            continue
        section_name, field_name = mapping
        section = getattr(config, section_name, None)
        if section is not None and hasattr(section, field_name):
            setattr(section, field_name, value)
    return dict(overrides)


def use_disk_database(config: SimulationConfig) -> None:
    """Force disk-backed, persisted SQLite (never in-memory)."""
    config.database.use_in_memory_db = False
    if hasattr(config.database, "persist_db_on_completion"):
        config.database.persist_db_on_completion = True


def maybe_apply_learning_positive_population(
    config: SimulationConfig,
    population: Optional[int],
    max_population: Optional[int],
) -> None:
    """Apply population override when ``population`` is set; no-op otherwise."""
    if population is None:
        return
    cap = max_population if max_population is not None else population * 4
    apply_independent_population(config, population, cap)


def build_learning_positive_regime_config(
    profile: str,
    *,
    environment: str = "development",
    population: int = DEFAULT_LEARNING_POSITIVE_POPULATION,
    max_population: int = DEFAULT_LEARNING_POSITIVE_MAX_POPULATION,
    force_disk_database: bool = True,
) -> SimulationConfig:
    """Learning-positive training regime: small population + stable ecology."""
    config = SimulationConfig.from_centralized_config(environment=environment)
    apply_independent_population(config, population, max_population)
    if force_disk_database:
        use_disk_database(config)
    apply_stable_profile_ecology(config, profile)
    return config
