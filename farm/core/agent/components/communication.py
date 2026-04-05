"""
Communication component.

Implements agent-to-agent (A2A) message-passing for explicit information exchange
between nearby agents.  Messages follow an **asynchronous inbox/outbox** pattern
inspired by the FIPA ACL (Foundation for Intelligent Physical Agents Agent
Communication Language) standard and classic publish/subscribe paradigms:

- **Asynchronous delivery** — senders push to an outbox; recipients read from an
  inbox.  Queued messages persist across steps until :meth:`CommunicationComponent.flush_outbox`
  runs (typically when the agent executes the ``communicate`` action).
- **Proximity-limited broadcast** — by default, broadcast messages only reach
  agents within ``communication_range`` distance.
- **Typed messages** — each message carries a :class:`MessageType` that recipients
  can filter on, enabling simple protocol-level semantics.

Supported message types
-----------------------
``RESOURCE_REQUEST``   Request resources from nearby agents.
``RESOURCE_OFFER``     Proactively offer resources to nearby agents.
``THREAT_ALERT``       Warn nearby agents about a danger (attacker, scarcity…).
``INFO``               General-purpose informational payload.
``CUSTOM``             Arbitrary user-defined payload.

Design notes
------------
The component is **optional** — agents without it simply cannot send or receive
messages.  The outbox is **not** cleared at step boundaries; call :meth:`flush_outbox`
from the ``communicate`` action to deliver pending messages.  Recipients observe
them in a later step depending on simulation turn order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Deque, Dict, List, Optional

from farm.core.agent.config.component_configs import CommunicationConfig
from farm.core.agent.services import AgentServices
from farm.utils.logging import get_logger

from .base import AgentComponent

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore

logger = get_logger(__name__)


class MessageType(str, Enum):
    """Enumeration of supported A2A message types."""

    RESOURCE_REQUEST = "resource_request"
    """Sender is asking nearby agents for resources."""

    RESOURCE_OFFER = "resource_offer"
    """Sender is proactively offering to share resources."""

    THREAT_ALERT = "threat_alert"
    """Sender is warning nearby agents of a threat (e.g. an attacker)."""

    INFO = "info"
    """General-purpose informational exchange."""

    CUSTOM = "custom"
    """Arbitrary user-defined message; interpret ``content`` freely."""


@dataclass
class Message:
    """A single agent-to-agent message.

    Attributes:
        sender_id:      Agent ID of the sender.
        message_type:   Semantic type of the message.
        content:        Arbitrary payload (must be JSON-serialisable).
        step:           Simulation step when the message was created.
        priority:       Reserved for future prioritisation (not used for ordering yet).
        recipient_id:   Target agent ID, or ``None`` for a broadcast.
    """

    sender_id: str
    message_type: MessageType
    content: Dict
    step: int
    priority: float = 0.0
    recipient_id: Optional[str] = None

    def is_broadcast(self) -> bool:
        """Return ``True`` if this message has no specific recipient."""
        return self.recipient_id is None


class CommunicationComponent(AgentComponent):
    """Component that gives an agent the ability to send and receive messages.

    Responsibilities:
    - Maintain a bounded inbox queue of received :class:`Message` objects.
    - Accumulate outbound messages in a bounded outbox until :meth:`flush_outbox`.
    - Expose helpers for sending and reading messages.
    - Clear inbox explicitly via :meth:`clear_inbox`; outbox drains on flush or terminate.
    """

    def __init__(self, services: AgentServices, config: CommunicationConfig):
        """Initialise the communication component.

        Args:
            services: Agent services container.
            config:   Communication-specific configuration.
        """
        super().__init__(services, "CommunicationComponent")
        self.config = config
        self._inbox: Deque[Message] = deque(maxlen=config.inbox_capacity)
        self._outbox: Deque[Message] = deque(maxlen=config.outbox_capacity)
        self._messages_sent: int = 0
        self._messages_received: int = 0

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_step_start(self) -> None:
        """Outbox is left intact so :meth:`send` queues can span step boundaries."""
        pass

    def on_step_end(self) -> None:
        """No-op; outbox flushing is done by the communicate action."""
        pass

    def on_terminate(self) -> None:
        """Clear all pending messages when agent dies."""
        self._inbox.clear()
        self._outbox.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        message_type: MessageType,
        content: Dict,
        recipient_id: Optional[str] = None,
        priority: float = 0.0,
    ) -> Message:
        """Queue a message for delivery during :meth:`flush_outbox`.

        Args:
            message_type:  Semantic type of the message.
            content:       Payload dict.
            recipient_id:  Target agent ID, or ``None`` to broadcast.
            priority:      Higher value = higher priority.

        Returns:
            The created :class:`Message`.
        """
        current_step = self.current_time
        if self.core is None:
            logger.warning("CommunicationComponent.send() called before component was attached to an agent core")
        msg = Message(
            sender_id=self.core.agent_id if self.core else "unknown",
            message_type=message_type,
            content=content,
            step=current_step,
            priority=priority,
            recipient_id=recipient_id,
        )
        self._outbox.append(msg)
        return msg

    def receive(self, message: Message) -> None:
        """Deliver a message to this agent's inbox.

        The inbox is bounded by ``config.inbox_capacity``; the oldest message
        is dropped when the inbox is full (deque with maxlen).

        Args:
            message: The message to deliver.
        """
        self._inbox.append(message)
        self._messages_received += 1

    def get_messages(
        self,
        message_type: Optional[MessageType] = None,
    ) -> List[Message]:
        """Return messages from the inbox, optionally filtered by type.

        Messages are returned in FIFO order and remain in the inbox until
        :meth:`clear_inbox` is called.

        Args:
            message_type: If provided, only messages of this type are returned.

        Returns:
            List of matching messages (may be empty).
        """
        if message_type is None:
            return list(self._inbox)
        return [m for m in self._inbox if m.message_type == message_type]

    def clear_inbox(self) -> None:
        """Remove all messages from the inbox."""
        self._inbox.clear()

    def flush_outbox(self) -> List[Message]:
        """Return pending outbox messages and clear the outbox.

        Called by the communicate action to retrieve messages for delivery
        to nearby agents.

        Returns:
            List of queued :class:`Message` objects (may be empty).
        """
        messages = list(self._outbox)
        self._outbox.clear()
        self._messages_sent += len(messages)
        return messages

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def inbox_size(self) -> int:
        """Number of messages currently in the inbox."""
        return len(self._inbox)

    @property
    def outbox_size(self) -> int:
        """Number of messages queued for delivery."""
        return len(self._outbox)

    @property
    def messages_sent(self) -> int:
        """Total messages sent since component was created."""
        return self._messages_sent

    @property
    def messages_received(self) -> int:
        """Total messages received since component was created."""
        return self._messages_received

    @property
    def communication_range(self) -> float:
        """Maximum range for proximity-based delivery."""
        return self.config.communication_range
