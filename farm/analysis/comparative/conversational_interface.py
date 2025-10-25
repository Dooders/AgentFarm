"""
Conversational interface for analysis interaction.

This module provides a chat-based interface for interacting with the analysis system
using natural language queries and responses.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import asyncio
from enum import Enum

from farm.utils.logging import get_logger
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult
from farm.analysis.comparative.ai_assistant import AIAnalysisAssistant, QueryType, ResponseType
from farm.analysis.comparative.automated_insights import Insight, InsightType, InsightSeverity
from farm.analysis.comparative.smart_recommendations import Recommendation, RecommendationType, RecommendationPriority

logger = get_logger(__name__)


class ConversationState(Enum):
    """States of the conversation."""
    IDLE = "idle"
    WAITING_FOR_INPUT = "waiting_for_input"
    PROCESSING = "processing"
    PROVIDING_RESPONSE = "providing_response"
    ERROR = "error"


class MessageType(Enum):
    """Types of messages in the conversation."""
    USER_QUERY = "user_query"
    ASSISTANT_RESPONSE = "assistant_response"
    SYSTEM_MESSAGE = "system_message"
    ERROR_MESSAGE = "error_message"
    ANALYSIS_UPDATE = "analysis_update"
    INSIGHT_NOTIFICATION = "insight_notification"
    RECOMMENDATION_ALERT = "recommendation_alert"


@dataclass
class ConversationMessage:
    """A message in the conversation."""
    
    id: str
    type: MessageType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversationSession:
    """A conversation session."""
    
    id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    messages: List[ConversationMessage] = field(default_factory=list)
    state: ConversationState = ConversationState.IDLE
    context: Dict[str, Any] = field(default_factory=dict)
    active_analysis: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationConfig:
    """Configuration for the conversational interface."""
    
    # Session management
    max_session_duration: int = 3600  # 1 hour
    max_messages_per_session: int = 100
    session_cleanup_interval: int = 300  # 5 minutes
    
    # Response settings
    max_response_length: int = 2000
    include_visualizations: bool = True
    include_recommendations: bool = True
    include_insights: bool = True
    
    # AI assistant settings
    enable_ai_assistant: bool = True
    ai_confidence_threshold: float = 0.6
    
    # Notification settings
    enable_notifications: bool = True
    notification_cooldown: int = 30  # seconds
    
    # Learning settings
    enable_learning: bool = True
    learning_rate: float = 0.1


class ConversationalInterface:
    """Conversational interface for analysis interaction."""
    
    def __init__(self, config: Optional[ConversationConfig] = None):
        """Initialize the conversational interface."""
        self.config = config or ConversationConfig()
        self.sessions: Dict[str, ConversationSession] = {}
        self.ai_assistant: Optional[AIAnalysisAssistant] = None
        
        # Initialize AI assistant if enabled
        if self.config.enable_ai_assistant:
            self.ai_assistant = AIAnalysisAssistant()
        
        # Start session cleanup task
        self._start_session_cleanup()
        
        logger.info("ConversationalInterface initialized")
    
    def _start_session_cleanup(self):
        """Start the session cleanup task."""
        try:
            async def cleanup_sessions():
                while True:
                    await asyncio.sleep(self.config.session_cleanup_interval)
                    self._cleanup_expired_sessions()
            
            asyncio.create_task(cleanup_sessions())
        except RuntimeError:
            # No event loop running, skip cleanup task creation
            # This can happen during testing or when not in an async context
            pass
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if (current_time - session.last_activity).seconds > self.config.max_session_duration:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    async def start_session(self, user_id: str, preferences: Optional[Dict[str, Any]] = None) -> str:
        """Start a new conversation session."""
        session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = ConversationSession(
            id=session_id,
            user_id=user_id,
            preferences=preferences or {}
        )
        
        self.sessions[session_id] = session
        
        # Send welcome message
        welcome_message = await self._create_welcome_message()
        await self._add_message(session, MessageType.SYSTEM_MESSAGE, welcome_message)
        
        logger.info(f"Started new session: {session_id}")
        return session_id
    
    async def _create_welcome_message(self) -> str:
        """Create a welcome message for new sessions."""
        return """Welcome to the Simulation Analysis Assistant! 🤖

I can help you with:
• **Analysis Requests**: "Analyze my simulation results" or "Compare these simulations"
• **Result Interpretation**: "What does this mean?" or "Explain these results"
• **Recommendations**: "What should I do?" or "Give me recommendations"
• **Trend Analysis**: "Show me trends" or "How is performance changing?"
• **Anomaly Detection**: "Are there any anomalies?" or "Find outliers"
• **Performance Questions**: "How is the performance?" or "Why is it slow?"

Just ask me in natural language and I'll do my best to help!"""
    
    async def process_message(self, session_id: str, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user message and return a response."""
        if session_id not in self.sessions:
            return {"error": "Session not found", "session_id": session_id}
        
        session = self.sessions[session_id]
        session.last_activity = datetime.now()
        
        # Add user message to session
        await self._add_message(session, MessageType.USER_QUERY, message)
        
        # Update session state
        session.state = ConversationState.PROCESSING
        
        try:
            # Process the message
            response = await self._process_user_message(session, message, context)
            
            # Add assistant response to session
            await self._add_message(session, MessageType.ASSISTANT_RESPONSE, response["content"], response.get("metadata", {}))
            
            # Update session state
            session.state = ConversationState.IDLE
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = f"I encountered an error processing your message: {str(e)}"
            await self._add_message(session, MessageType.ERROR_MESSAGE, error_response)
            session.state = ConversationState.ERROR
            
            return {
                "content": error_response,
                "type": "error",
                "metadata": {"error": str(e)}
            }
    
    async def _process_user_message(self, session: ConversationSession, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user message and generate a response."""
        # Check for special commands
        if message.startswith("/"):
            return await self._handle_command(session, message, context)
        
        # Use AI assistant if available
        if self.ai_assistant and self.config.enable_ai_assistant:
            return await self._handle_ai_assistant_query(session, message, context)
        
        # Fallback to simple response
        return await self._handle_simple_query(session, message, context)
    
    async def _handle_command(self, session: ConversationSession, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle special commands."""
        command = message[1:].lower().strip()
        
        if command == "help":
            return {
                "content": self._get_help_message(),
                "type": "help",
                "metadata": {"command": "help"}
            }
        
        elif command == "status":
            return {
                "content": self._get_session_status(session),
                "type": "status",
                "metadata": {"command": "status"}
            }
        
        elif command == "clear":
            session.messages = []
            return {
                "content": "Conversation history cleared.",
                "type": "system",
                "metadata": {"command": "clear"}
            }
        
        elif command.startswith("set "):
            return await self._handle_set_command(session, command[4:])
        
        elif command == "insights":
            return await self._handle_insights_command(session)
        
        elif command == "recommendations":
            return await self._handle_recommendations_command(session)
        
        else:
            return {
                "content": f"Unknown command: {command}. Type /help for available commands.",
                "type": "error",
                "metadata": {"command": command}
            }
    
    def _get_help_message(self) -> str:
        """Get help message with available commands."""
        return """Available Commands:

**Basic Commands:**
• `/help` - Show this help message
• `/status` - Show session status
• `/clear` - Clear conversation history

**Analysis Commands:**
• `/insights` - Show current analysis insights
• `/recommendations` - Show current recommendations

**Settings Commands:**
• `/set <key> <value>` - Set a preference
• `/set notifications on/off` - Enable/disable notifications
• `/set visualizations on/off` - Enable/disable visualizations

**Natural Language:**
You can also just ask me questions in natural language like:
• "What does this analysis mean?"
• "Show me the performance trends"
• "Are there any anomalies in my data?"
• "Give me recommendations for optimization" """
    
    def _get_session_status(self, session: ConversationSession) -> str:
        """Get session status information."""
        status = f"""**Session Status:**
• **Session ID**: {session.id}
• **User ID**: {session.user_id}
• **Created**: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}
• **Last Activity**: {session.last_activity.strftime('%Y-%m-%d %H:%M:%S')}
• **Messages**: {len(session.messages)}
• **State**: {session.state.value}
• **Active Analysis**: {session.active_analysis or 'None'}

**Preferences:**
"""
        for key, value in session.preferences.items():
            status += f"• **{key}**: {value}\n"
        
        return status
    
    async def _handle_set_command(self, session: ConversationSession, command: str) -> Dict[str, Any]:
        """Handle set preference command."""
        parts = command.split(" ", 1)
        if len(parts) != 2:
            return {
                "content": "Usage: /set <key> <value>",
                "type": "error",
                "metadata": {"command": "set"}
            }
        
        key, value = parts
        session.preferences[key] = value
        
        return {
            "content": f"Set {key} to {value}",
            "type": "system",
            "metadata": {"command": "set", "key": key, "value": value}
        }
    
    async def _handle_insights_command(self, session: ConversationSession) -> Dict[str, Any]:
        """Handle insights command."""
        if not session.active_analysis:
            return {
                "content": "No active analysis. Please run an analysis first.",
                "type": "error",
                "metadata": {"command": "insights"}
            }
        
        # This would fetch insights from the active analysis
        # For now, return a placeholder
        return {
            "content": "Insights feature coming soon. This will show analysis insights when available.",
            "type": "insights",
            "metadata": {"command": "insights", "analysis_id": session.active_analysis}
        }
    
    async def _handle_recommendations_command(self, session: ConversationSession) -> Dict[str, Any]:
        """Handle recommendations command."""
        if not session.active_analysis:
            return {
                "content": "No active analysis. Please run an analysis first.",
                "type": "error",
                "metadata": {"command": "recommendations"}
            }
        
        # This would fetch recommendations from the active analysis
        # For now, return a placeholder
        return {
            "content": "Recommendations feature coming soon. This will show analysis recommendations when available.",
            "type": "recommendations",
            "metadata": {"command": "recommendations", "analysis_id": session.active_analysis}
        }
    
    async def _handle_ai_assistant_query(self, session: ConversationSession, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle query using AI assistant."""
        if not self.ai_assistant:
            return await self._handle_simple_query(session, message, context)
        
        try:
            # Set analysis context if available
            if session.active_analysis:
                # This would set the current analysis context
                # For now, we'll pass None
                self.ai_assistant.set_analysis_context(None)
            
            # Process query with AI assistant
            response = await self.ai_assistant.process_query(message, context)
            
            # Convert AI response to conversation response
            return {
                "content": response.content,
                "type": response.response_type.value,
                "metadata": {
                    "confidence": response.confidence,
                    "data": response.data,
                    "recommendations": response.recommendations,
                    "visualizations": response.visualizations
                }
            }
        
        except Exception as e:
            logger.error(f"Error in AI assistant: {e}")
            return await self._handle_simple_query(session, message, context)
    
    async def _handle_simple_query(self, session: ConversationSession, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle query with simple response logic."""
        message_lower = message.lower()
        
        # Simple keyword-based responses
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return {
                "content": "Hello! I'm here to help you with simulation analysis. What would you like to know?",
                "type": "greeting",
                "metadata": {}
            }
        
        elif any(word in message_lower for word in ["help", "what can you do"]):
            return {
                "content": self._get_help_message(),
                "type": "help",
                "metadata": {}
            }
        
        elif any(word in message_lower for word in ["analyze", "analysis", "compare"]):
            return {
                "content": "I can help you analyze simulation results. To get started, please provide your analysis data or run an analysis first.",
                "type": "analysis_request",
                "metadata": {}
            }
        
        elif any(word in message_lower for word in ["performance", "speed", "time"]):
            return {
                "content": "I can help you analyze performance metrics. Please provide performance data or run a performance analysis.",
                "type": "performance_query",
                "metadata": {}
            }
        
        elif any(word in message_lower for word in ["anomaly", "outlier", "unusual"]):
            return {
                "content": "I can help you detect anomalies in your data. Please provide your data or run an anomaly detection analysis.",
                "type": "anomaly_query",
                "metadata": {}
            }
        
        else:
            return {
                "content": "I'm not sure I understand your question. Could you rephrase it or ask for help?",
                "type": "unclear",
                "metadata": {}
            }
    
    async def _add_message(self, session: ConversationSession, message_type: MessageType, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the session."""
        message = ConversationMessage(
            id=f"msg_{len(session.messages) + 1}",
            type=message_type,
            content=content,
            metadata=metadata or {}
        )
        
        session.messages.append(message)
        
        # Limit messages per session
        if len(session.messages) > self.config.max_messages_per_session:
            session.messages = session.messages[-self.config.max_messages_per_session:]
    
    async def set_analysis_context(self, session_id: str, analysis_result: OrchestrationResult, insights: List[Insight] = None, recommendations: List[Recommendation] = None):
        """Set analysis context for a session."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        session.active_analysis = analysis_result.id if hasattr(analysis_result, 'id') else "analysis_1"
        session.context["analysis_result"] = analysis_result
        session.context["insights"] = insights or []
        session.context["recommendations"] = recommendations or []
        
        # Set AI assistant context if available
        if self.ai_assistant:
            self.ai_assistant.set_analysis_context(analysis_result)
        
        # Send analysis update message
        update_message = f"Analysis context updated. Analysis ID: {session.active_analysis}"
        await self._add_message(session, MessageType.ANALYSIS_UPDATE, update_message)
        
        logger.info(f"Analysis context set for session {session_id}")
        return {"success": True, "analysis_id": session.active_analysis}
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        return [
            {
                "id": msg.id,
                "type": msg.type.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in session.messages
        ]
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions."""
        return [
            {
                "session_id": session.id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "message_count": len(session.messages),
                "state": session.state.value,
                "active_analysis": session.active_analysis
            }
            for session in self.sessions.values()
        ]
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a conversation session."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        
        # Send goodbye message
        goodbye_message = "Thank you for using the Simulation Analysis Assistant. Goodbye! 👋"
        await self._add_message(session, MessageType.SYSTEM_MESSAGE, goodbye_message)
        
        # Remove session
        del self.sessions[session_id]
        
        logger.info(f"Ended session: {session_id}")
        return {"success": True, "message": "Session ended"}
    
    async def send_notification(self, session_id: str, notification_type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Send a notification to a session."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        
        # Check notification cooldown
        if self.config.notification_cooldown > 0:
            last_notification = session.context.get("last_notification_time")
            if last_notification and (datetime.now() - last_notification).seconds < self.config.notification_cooldown:
                return {"error": "Notification cooldown active"}
        
        # Send notification
        message_type = MessageType.INSIGHT_NOTIFICATION if notification_type == "insight" else MessageType.RECOMMENDATION_ALERT
        await self._add_message(session, message_type, content, metadata)
        
        # Update last notification time
        session.context["last_notification_time"] = datetime.now()
        
        logger.info(f"Sent notification to session {session_id}: {notification_type}")
        return {"success": True}
    
    def get_interface_stats(self) -> Dict[str, Any]:
        """Get interface statistics."""
        return {
            "active_sessions": len(self.sessions),
            "total_messages": sum(len(session.messages) for session in self.sessions.values()),
            "ai_assistant_enabled": self.ai_assistant is not None,
            "config": {
                "max_session_duration": self.config.max_session_duration,
                "max_messages_per_session": self.config.max_messages_per_session,
                "enable_ai_assistant": self.config.enable_ai_assistant,
                "enable_notifications": self.config.enable_notifications
            }
        }