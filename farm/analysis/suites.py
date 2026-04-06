"""
Built-in analysis suite definitions.

A suite is a named list of module names to run together via AnalysisService.run_suite().
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from farm.analysis.exceptions import ConfigurationError
from farm.analysis.registry import get_module_names

#: Maps suite name -> module names to run (excluding dynamic ``full``).
BUILTIN_SUITE_MODULES: Dict[str, Tuple[str, ...]] = {
    "system_dynamics": ("population", "resources", "temporal"),
    "agent_behavior": ("actions", "agents", "spatial", "learning"),
    "social": ("social_behavior", "combat", "dominance"),
}


def list_builtin_suite_names() -> List[str]:
    """Return built-in suite names, including ``full``."""
    return sorted(BUILTIN_SUITE_MODULES.keys()) + ["full"]


def resolve_suite_module_names(
    *,
    suite: Optional[str] = None,
    modules: Optional[Sequence[str]] = None,
) -> List[str]:
    """Resolve which module names to run.

    If ``modules`` is provided and non-empty, it is used as-is (preserving order,
    deduplicating while keeping first occurrence). Otherwise ``suite`` must be a
    built-in suite name.

    Args:
        suite: Built-in suite name (e.g. ``system_dynamics``, ``full``).
        modules: Explicit list of module names.

    Returns:
        Ordered list of module names to execute.

    Raises:
        ConfigurationError: If arguments are invalid or the suite is unknown.
    """
    if modules:
        seen: set[str] = set()
        out: List[str] = []
        for name in modules:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    if not suite:
        raise ConfigurationError(
            "run_suite requires either `suite` (built-in name) or a non-empty `modules` list."
        )

    if suite == "full":
        return list(get_module_names())

    if suite not in BUILTIN_SUITE_MODULES:
        known = ", ".join(list_builtin_suite_names())
        raise ConfigurationError(f"Unknown analysis suite '{suite}'. Known suites: {known}")

    return list(BUILTIN_SUITE_MODULES[suite])
