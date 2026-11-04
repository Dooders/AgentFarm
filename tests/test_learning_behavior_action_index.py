"""The learning behaviour credits the policy output it actually sampled.

``LearningAgentBehavior.decide_action`` samples an index into ``core.actions``
(registry order: attack, move, gather, reproduce, share, defend, pass,
communicate), so the experience stored for that step must use the same
index. Using the ``ActionType`` value instead (defend=0, attack=1, ...) sends
the update to the wrong output for five of the eight actions.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from farm.core.action import Action, ActionType, action_registry  # noqa: E402
from farm.core.agent.behaviors.learning import LearningAgentBehavior  # noqa: E402

pytestmark = pytest.mark.unit


def _core():
    core = Mock()
    core.actions = [Action(a.name, a.weight, a.function) for a in action_registry.get_all(normalized=False)]
    core.actions = [a for a in core.actions if a.name not in ("bond", "leave")]
    core.config = None
    core.agent_id = "a1"
    return core


def test_update_credits_the_sampled_policy_index():
    core = _core()
    module = Mock()
    module.num_actions = len(core.actions)
    module.decide_action.return_value = core.actions.index(next(a for a in core.actions if a.name == "gather"))
    behavior = LearningAgentBehavior(module)
    state = torch.zeros(4)

    chosen = behavior.decide_action(core, state)
    assert chosen.name == "gather"
    behavior.update(state=state, action=chosen, reward=1.0, next_state=state, done=False)

    stored_index = module.update.call_args.kwargs["action"]
    assert stored_index == core.actions.index(chosen)
    # ...which is not the ActionType value for most actions.
    assert [a.name for a in core.actions] != [t.name.lower() for t in ActionType]


@pytest.mark.parametrize("name", ["attack", "move", "reproduce", "share", "defend"])
def test_every_registry_action_maps_to_its_own_position(name):
    core = _core()
    module = Mock()
    module.num_actions = len(core.actions)
    behavior = LearningAgentBehavior(module)
    module.decide_action.return_value = 0
    behavior.decide_action(core, torch.zeros(4))
    action = next(a for a in core.actions if a.name == name)
    assert behavior.policy_index(action) == core.actions.index(action)


def test_policy_index_falls_back_to_action_type_before_any_decision():
    behavior = LearningAgentBehavior(Mock(num_actions=8))
    assert behavior.policy_index(Action("move", 1.0, lambda a: None)) == ActionType.MOVE.value
