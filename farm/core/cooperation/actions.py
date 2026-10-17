"""The two cooperation actions, ``bond`` and ``leave``.

They are registered in :mod:`farm.core.action` alongside the built-in actions
but only enter an agent's action list (and the environment's action space)
when the environment carries a cooperation service; see
:func:`filter_cooperation_actions`. Their indices continue the built-in
:class:`~farm.core.action.ActionType` numbering so a cooperation-enabled
action space is the historical one with two actions appended.

Both functions are thin: the rules live in
:class:`~farm.core.cooperation.service.CooperationService`.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class CooperationActionType(IntEnum):
    """Action-space indices of the cooperation actions (after ``COMMUNICATE = 7``)."""

    BOND = 8
    LEAVE = 9


#: Registry names of the cooperation actions, in index order.
COOPERATION_ACTION_NAMES: tuple[str, ...] = ("bond", "leave")

#: ``name -> index`` for the cooperation actions.
COOPERATION_ACTION_INDEX: dict[str, int] = {
    "bond": CooperationActionType.BOND.value,
    "leave": CooperationActionType.LEAVE.value,
}

#: Registry weights used before an agent's propensity genes are applied.
DEFAULT_ACTION_WEIGHTS: dict[str, float] = {"bond": 0.1, "leave": 0.05}


_SERVICE_CLASS: Any = None


def cooperation_service_of(environment: Any) -> Any:
    """The environment's :class:`CooperationService`, or ``None``.

    Strict about the type so that test doubles (``Mock`` environments, whose
    every attribute exists) never switch cooperation on by accident.
    """
    global _SERVICE_CLASS
    if environment is None:
        return None
    if _SERVICE_CLASS is None:
        from farm.core.cooperation.service import CooperationService

        _SERVICE_CLASS = CooperationService
    service = getattr(environment, "cooperation", None)
    return service if isinstance(service, _SERVICE_CLASS) else None


def filter_cooperation_actions(actions: list, environment: Any) -> list:
    """Drop ``bond``/``leave`` from ``actions`` unless cooperation is enabled."""
    if cooperation_service_of(environment) is not None:
        return list(actions)
    return [a for a in actions if getattr(a, "name", None) not in COOPERATION_ACTION_NAMES]


def _unavailable(name: str) -> dict:
    return {
        "success": False,
        "error": f"{name} requires cooperation to be enabled",
        "details": {},
    }


def bond_action(agent: Any) -> dict:
    """Offer to form or join a cooperative unit with a nearby open agent."""
    service = cooperation_service_of(getattr(agent, "environment", None))
    if service is None:
        return _unavailable("bond")
    return service.bond(agent)


def leave_action(agent: Any) -> dict:
    """Leave the agent's least rewarding unit, paying any enforced exit cost."""
    service = cooperation_service_of(getattr(agent, "environment", None))
    if service is None:
        return _unavailable("leave")
    return service.leave(agent)


__all__ = [
    "COOPERATION_ACTION_INDEX",
    "COOPERATION_ACTION_NAMES",
    "DEFAULT_ACTION_WEIGHTS",
    "CooperationActionType",
    "bond_action",
    "cooperation_service_of",
    "filter_cooperation_actions",
    "leave_action",
]
