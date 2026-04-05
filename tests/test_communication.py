"""Tests for the agent-to-agent communication system.

Covers:
- CommunicationComponent lifecycle and message API
- communicate_action delivery to nearby agents
- Edge cases: no recipients, missing component, inbox capacity
"""

import pytest
from unittest.mock import Mock

from farm.core.agent import (
    AgentCore,
    AgentFactory,
    AgentServices,
    AgentComponentConfig,
    CommunicationComponent,
    Message,
    MessageType,
)
from farm.core.agent.config import CommunicationConfig
from farm.core.action import communicate_action, ActionType, get_action_space


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_services():
    """Return a minimal AgentServices mock."""
    spatial = Mock()
    spatial.get_nearby = Mock(return_value={"agents": [], "resources": []})
    return AgentServices(
        spatial_service=spatial,
        time_service=Mock(current_time=Mock(return_value=0)),
        metrics_service=Mock(),
        logging_service=Mock(),
        validation_service=Mock(is_valid_position=Mock(return_value=True)),
        lifecycle_service=Mock(),
    )


@pytest.fixture
def factory(mock_services):
    return AgentFactory(mock_services)


@pytest.fixture
def agent(factory):
    """An agent with all default components (including CommunicationComponent)."""
    return factory.create_default_agent("agent_001", (50.0, 50.0), initial_resources=100.0)


# ---------------------------------------------------------------------------
# CommunicationConfig
# ---------------------------------------------------------------------------


class TestCommunicationConfig:
    def test_default_values(self):
        cfg = CommunicationConfig()
        assert cfg.communication_range == 50.0
        assert cfg.inbox_capacity == 20
        assert cfg.outbox_capacity == 20
        assert cfg.broadcast_cost == 0.0
        assert cfg.reward_per_message == 0.01

    def test_custom_values(self):
        cfg = CommunicationConfig(communication_range=100.0, inbox_capacity=5)
        assert cfg.communication_range == 100.0
        assert cfg.inbox_capacity == 5


# ---------------------------------------------------------------------------
# CommunicationComponent unit tests
# ---------------------------------------------------------------------------


class TestCommunicationComponent:
    def test_attached_to_agent(self, agent):
        comp = agent.get_component("communication")
        assert comp is not None
        assert isinstance(comp, CommunicationComponent)

    def test_initial_state(self, agent):
        comp = agent.get_component("communication")
        assert comp.inbox_size == 0
        assert comp.outbox_size == 0
        assert comp.messages_sent == 0
        assert comp.messages_received == 0

    def test_send_creates_message_in_outbox(self, agent):
        comp = agent.get_component("communication")
        msg = comp.send(MessageType.INFO, {"key": "value"})
        assert isinstance(msg, Message)
        assert comp.outbox_size == 1
        assert msg.message_type == MessageType.INFO
        assert msg.sender_id == agent.agent_id

    def test_receive_adds_to_inbox(self, agent):
        comp = agent.get_component("communication")
        msg = Message(
            sender_id="sender_001",
            message_type=MessageType.THREAT_ALERT,
            content={"threat": "attacker nearby"},
            step=1,
        )
        comp.receive(msg)
        assert comp.inbox_size == 1
        assert comp.messages_received == 1

    def test_get_messages_no_filter(self, agent):
        comp = agent.get_component("communication")
        msg1 = Message("s1", MessageType.INFO, {}, step=1)
        msg2 = Message("s2", MessageType.RESOURCE_REQUEST, {}, step=1)
        comp.receive(msg1)
        comp.receive(msg2)
        msgs = comp.get_messages()
        assert len(msgs) == 2

    def test_get_messages_filtered_by_type(self, agent):
        comp = agent.get_component("communication")
        comp.receive(Message("s1", MessageType.INFO, {}, step=1))
        comp.receive(Message("s2", MessageType.THREAT_ALERT, {}, step=1))
        assert len(comp.get_messages(MessageType.INFO)) == 1
        assert len(comp.get_messages(MessageType.THREAT_ALERT)) == 1
        assert len(comp.get_messages(MessageType.RESOURCE_OFFER)) == 0

    def test_clear_inbox(self, agent):
        comp = agent.get_component("communication")
        comp.receive(Message("s1", MessageType.INFO, {}, step=1))
        comp.clear_inbox()
        assert comp.inbox_size == 0

    def test_flush_outbox_returns_and_clears(self, agent):
        comp = agent.get_component("communication")
        comp.send(MessageType.INFO, {})
        comp.send(MessageType.RESOURCE_OFFER, {"amount": 5})
        msgs = comp.flush_outbox()
        assert len(msgs) == 2
        assert comp.outbox_size == 0
        assert comp.messages_sent == 2

    def test_on_step_start_preserves_outbox(self, agent):
        comp = agent.get_component("communication")
        comp.send(MessageType.INFO, {})
        assert comp.outbox_size == 1
        comp.on_step_start()
        assert comp.outbox_size == 1

    def test_on_terminate_clears_inbox_and_outbox(self, agent):
        comp = agent.get_component("communication")
        comp.receive(Message("s1", MessageType.INFO, {}, step=1))
        comp.send(MessageType.INFO, {})
        comp.on_terminate()
        assert comp.inbox_size == 0
        assert comp.outbox_size == 0

    def test_inbox_capacity_bounded(self):
        """Messages beyond inbox_capacity should drop oldest (deque with maxlen)."""
        cfg = CommunicationConfig(inbox_capacity=3)
        services = AgentServices(spatial_service=Mock())
        comp = CommunicationComponent(services, cfg)

        for i in range(5):
            comp.receive(Message(f"s{i}", MessageType.INFO, {"i": i}, step=i))

        assert comp.inbox_size == 3
        # Newest three should be retained
        msgs = comp.get_messages()
        steps = [m.step for m in msgs]
        assert steps == [2, 3, 4]

    def test_outbox_capacity_bounded(self):
        """Queued sends beyond outbox_capacity should drop oldest."""
        cfg = CommunicationConfig(outbox_capacity=3)
        services = AgentServices(spatial_service=Mock())
        comp = CommunicationComponent(services, cfg)

        for i in range(5):
            comp.send(MessageType.INFO, {"i": i})

        assert comp.outbox_size == 3
        flushed = comp.flush_outbox()
        assert len(flushed) == 3
        assert [m.content["i"] for m in flushed] == [2, 3, 4]

    def test_broadcast_flag(self, agent):
        comp = agent.get_component("communication")
        msg = comp.send(MessageType.INFO, {}, recipient_id=None)
        assert msg.is_broadcast() is True

    def test_direct_message_flag(self, agent):
        comp = agent.get_component("communication")
        msg = comp.send(MessageType.INFO, {}, recipient_id="agent_002")
        assert msg.is_broadcast() is False
        assert msg.recipient_id == "agent_002"

    def test_communication_range_property(self, agent):
        comp = agent.get_component("communication")
        assert comp.communication_range == CommunicationConfig().communication_range


# ---------------------------------------------------------------------------
# Message dataclass tests
# ---------------------------------------------------------------------------


class TestMessage:
    def test_message_defaults(self):
        msg = Message("sender", MessageType.INFO, {"key": "val"}, step=5)
        assert msg.priority == 0.0
        assert msg.recipient_id is None

    def test_message_type_values(self):
        assert MessageType.RESOURCE_REQUEST == "resource_request"
        assert MessageType.RESOURCE_OFFER == "resource_offer"
        assert MessageType.THREAT_ALERT == "threat_alert"
        assert MessageType.INFO == "info"
        assert MessageType.CUSTOM == "custom"


# ---------------------------------------------------------------------------
# communicate_action tests
# ---------------------------------------------------------------------------


class TestCommunicateAction:
    def _make_nearby_agent(self, mock_services, agent_id="nearby_001", position=(55.0, 55.0)):
        factory = AgentFactory(mock_services)
        return factory.create_default_agent(agent_id, position, initial_resources=50.0)

    def test_communicate_no_nearby_agents(self, agent):
        """Action fails gracefully when no agents are nearby."""
        result = communicate_action(agent)
        assert result["success"] is False
        assert "No nearby agents" in result["error"]

    def test_communicate_delivers_to_nearby_agent(self, mock_services, factory):
        """Broadcast message is delivered to a nearby agent's inbox."""
        sender = factory.create_default_agent("sender", (50.0, 50.0), initial_resources=100.0)
        recipient = factory.create_default_agent("recipient", (60.0, 60.0), initial_resources=50.0)

        # Patch spatial service to return the recipient
        mock_services.spatial_service.get_nearby.return_value = {"agents": [recipient]}

        result = communicate_action(sender)

        assert result["success"] is True
        assert result["details"]["messages_delivered"] == 1
        assert result["details"]["messages_flushed"] == 1
        assert result["details"]["messages_by_type"] == {"info": 1}

        recipient_comp = recipient.get_component("communication")
        assert recipient_comp.inbox_size == 1
        msg = recipient_comp.get_messages()[0]
        assert msg.sender_id == sender.agent_id
        assert msg.message_type == MessageType.INFO

    def test_communicate_excludes_self(self, mock_services, factory):
        """The sender is excluded from recipients even if it appears in nearby list."""
        sender = factory.create_default_agent("sender", (50.0, 50.0), initial_resources=100.0)
        mock_services.spatial_service.get_nearby.return_value = {"agents": [sender]}

        result = communicate_action(sender)
        assert result["success"] is False  # Self filtered out -> no recipients

    def test_communicate_reward_earned(self, mock_services, factory):
        """Sending agent earns a small reward for each message delivered."""
        sender = factory.create_default_agent("sender", (50.0, 50.0), initial_resources=100.0)
        recipient = factory.create_default_agent("recipient", (55.0, 55.0), initial_resources=50.0)
        mock_services.spatial_service.get_nearby.return_value = {"agents": [recipient]}

        reward_before = sender.total_reward
        communicate_action(sender)
        assert sender.total_reward > reward_before

    def test_communicate_no_comm_component(self, mock_services, factory):
        """Action fails gracefully when sender has no CommunicationComponent."""
        # Create minimal agent (no communication component)
        agent = factory.create_minimal_agent("minimal_001", (50.0, 50.0))
        mock_services.spatial_service.get_nearby.return_value = {"agents": []}

        result = communicate_action(agent)
        assert result["success"] is False
        assert "CommunicationComponent" in result["error"]

    def test_communicate_recipient_without_component(self, mock_services, factory):
        """Action handles recipient without CommunicationComponent gracefully."""
        sender = factory.create_default_agent("sender", (50.0, 50.0), initial_resources=100.0)
        recipient = factory.create_minimal_agent("minimal_recipient", (55.0, 55.0))
        mock_services.spatial_service.get_nearby.return_value = {"agents": [recipient]}

        result = communicate_action(sender)
        # Broadcast succeeded (nearby agents exist), but delivered=0 as recipient lacks component
        assert result["success"] is True
        assert result["details"]["messages_delivered"] == 0
        assert result["details"]["note"] is not None

    def test_communicate_dead_agent_excluded(self, mock_services, factory):
        """Dead agents are not eligible recipients."""
        sender = factory.create_default_agent("sender", (50.0, 50.0), initial_resources=100.0)
        dead_agent = factory.create_default_agent("dead_001", (55.0, 55.0), initial_resources=50.0)
        dead_agent.alive = False
        mock_services.spatial_service.get_nearby.return_value = {"agents": [dead_agent]}

        result = communicate_action(sender)
        assert result["success"] is False

    def test_communicate_broadcast_cost_deducted(self, mock_services, factory):
        """Non-zero broadcast_cost is deducted when communication succeeds."""
        sender = factory.create_default_agent("sender", (50.0, 50.0), initial_resources=100.0)
        sender.config = AgentComponentConfig(communication=CommunicationConfig(broadcast_cost=5.0))
        recipient = factory.create_default_agent("recipient", (55.0, 55.0), initial_resources=50.0)
        mock_services.spatial_service.get_nearby.return_value = {"agents": [recipient]}

        before = sender.resource_level
        result = communicate_action(sender)

        assert result["success"] is True
        assert sender.resource_level == before - 5.0

    def test_communicate_insufficient_resources_for_cost(self, mock_services, factory):
        """Action fails when broadcast_cost exceeds available resources."""
        sender = factory.create_default_agent("sender", (50.0, 50.0), initial_resources=3.0)
        sender.config = AgentComponentConfig(communication=CommunicationConfig(broadcast_cost=5.0))
        recipient = factory.create_default_agent("recipient", (55.0, 55.0), initial_resources=50.0)
        mock_services.spatial_service.get_nearby.return_value = {"agents": [recipient]}

        before = sender.resource_level
        result = communicate_action(sender)

        assert result["success"] is False
        assert "Insufficient resources" in result["error"]
        assert sender.resource_level == before

    def test_communicate_unicast_delivers_only_to_target(self, mock_services, factory):
        """A message with recipient_id is routed only to the matching agent."""
        sender = factory.create_default_agent("sender", (50.0, 50.0), initial_resources=100.0)
        target = factory.create_default_agent("target_001", (55.0, 55.0), initial_resources=50.0)
        bystander = factory.create_default_agent("bystander_001", (57.0, 57.0), initial_resources=50.0)

        # Pre-queue a unicast message addressed to target_001 only
        sender_comm = sender.get_component("communication")
        sender_comm.send(
            MessageType.THREAT_ALERT,
            content={"alert": "test"},
            recipient_id="target_001",
        )

        # Both agents are within communication range
        mock_services.spatial_service.get_nearby.return_value = {"agents": [target, bystander]}

        result = communicate_action(sender)
        assert result["success"] is True

        target_comm = target.get_component("communication")
        bystander_comm = bystander.get_component("communication")

        # Target receives: unicast THREAT_ALERT + broadcast INFO = 2 messages
        # Bystander receives: broadcast INFO only = 1 message (unicast not delivered)
        assert target_comm.inbox_size == 2
        assert bystander_comm.inbox_size == 1

        # Confirm the unicast message reached only the intended target
        threat_msgs = target_comm.get_messages(MessageType.THREAT_ALERT)
        assert len(threat_msgs) == 1
        assert threat_msgs[0].recipient_id == "target_001"
        assert len(bystander_comm.get_messages(MessageType.THREAT_ALERT)) == 0

    def test_broadcast_info_payload_independent_per_recipient(self, mock_services, factory):
        """Broadcast delivery uses shallow-copied content dicts per neighbour."""
        sender = factory.create_default_agent("sender", (50.0, 50.0), initial_resources=100.0)
        a = factory.create_default_agent("a", (55.0, 55.0), initial_resources=50.0)
        b = factory.create_default_agent("b", (56.0, 56.0), initial_resources=50.0)
        mock_services.spatial_service.get_nearby.return_value = {"agents": [a, b]}

        communicate_action(sender)
        ma = a.get_component("communication").get_messages()[0]
        mb = b.get_component("communication").get_messages()[0]
        assert ma is not mb
        ma.content["mutated"] = True
        assert "mutated" not in mb.content

    def test_prequeued_send_survives_step_start_until_communicate(self, mock_services, factory):
        """Outbox persists across on_step_start; communicate flushes queued + INFO."""
        sender = factory.create_default_agent("sender", (50.0, 50.0), initial_resources=100.0)
        recipient = factory.create_default_agent("recipient", (55.0, 55.0), initial_resources=50.0)
        mock_services.spatial_service.get_nearby.return_value = {"agents": [recipient]}

        sender_comm = sender.get_component("communication")
        sender_comm.send(MessageType.CUSTOM, {"queued": True})
        sender_comm.on_step_start()

        result = communicate_action(sender)
        assert result["success"] is True
        assert result["details"]["messages_by_type"] == {"custom": 1, "info": 1}

        inbox = recipient.get_component("communication").get_messages()
        types = {m.message_type for m in inbox}
        assert MessageType.CUSTOM in types and MessageType.INFO in types


# ---------------------------------------------------------------------------
# ActionType and action space
# ---------------------------------------------------------------------------


class TestActionTypeAndSpace:
    def test_communicate_in_action_type(self):
        assert ActionType.COMMUNICATE == 7

    def test_communicate_in_action_space(self):
        space = get_action_space()
        assert "communicate" in space
        assert space["communicate"] == ActionType.COMMUNICATE.value

    def test_communicate_registered_in_registry(self):
        from farm.core.action import action_registry
        action = action_registry.get("communicate")
        assert action is not None
        assert action.name == "communicate"
        assert action.function is communicate_action
