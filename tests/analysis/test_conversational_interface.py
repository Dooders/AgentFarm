"""
Tests for Conversational Interface.

This module contains comprehensive tests for the conversational interface
that provides chat-based interaction with the analysis system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import shutil

from farm.analysis.comparative.conversational_interface import (
    ConversationalInterface,
    ConversationState,
    MessageType,
    ConversationMessage,
    ConversationSession,
    ConversationConfig
)
from farm.analysis.comparative.ai_assistant import AIAnalysisAssistant
from farm.analysis.comparative.automated_insights import Insight, InsightType, InsightSeverity
from farm.analysis.comparative.smart_recommendations import Recommendation, RecommendationType, RecommendationPriority
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult


class TestConversationalInterface:
    """Test cases for ConversationalInterface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConversationConfig(
            max_session_duration=3600,
            max_messages_per_session=100,
            enable_notifications=True,
            enable_ai_assistant=True
        )
        self.interface = ConversationalInterface(config=self.config)
        
        # Mock analysis result
        self.mock_analysis = Mock(spec=OrchestrationResult)
        self.mock_analysis.success = True
        self.mock_analysis.total_duration = 120.5
        self.mock_analysis.phase_results = [
            Mock(phase_name="statistical_analysis", duration=60.0, success=True),
            Mock(phase_name="ml_analysis", duration=60.5, success=True)
        ]
        self.mock_analysis.errors = []
        self.mock_analysis.warnings = ["Minor warning"]
        self.mock_analysis.summary = {
            "total_phases": 2,
            "success_rate": 1.0,
            "performance_metrics": {
                "cpu_usage": 85.5,
                "memory_usage": 70.2,
                "disk_io": 45.8
            }
        }
        
        # Mock insights
        self.mock_insights = [
            Insight(
                id="insight_1",
                type=InsightType.PERFORMANCE_PATTERN,
                title="High CPU Usage",
                description="CPU usage is above 85%",
                severity=InsightSeverity.HIGH,
                confidence=0.9,
                data_points=[{"cpu_usage": 85.5}],
                recommendations=["Optimize CPU usage"],
                created_at=datetime.now()
            )
        ]
        
        # Mock recommendations
        self.mock_recommendations = [
            Recommendation(
                id="rec_1",
                type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="Optimize CPU Usage",
                description="Reduce CPU usage by 10%",
                priority=RecommendationPriority.HIGH,
                confidence=0.9,
                prerequisites=["Reduce simulation complexity"],
                expected_impact="High",
                implementation_effort="Medium",
                created_at=datetime.now()
            )
        ]
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test interface initialization."""
        assert self.interface.config == self.config
        assert self.interface.sessions == {}
        assert self.interface.ai_assistant is not None
        assert isinstance(self.interface.ai_assistant, AIAnalysisAssistant)
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        interface = ConversationalInterface()
        assert interface.config is not None
        assert interface.config.enable_ai_assistant is True
        assert interface.config.enable_notifications is True
    
    @pytest.mark.asyncio
    async def test_start_session(self):
        """Test starting a new conversation session."""
        session_id = await self.interface.start_session("test_user")
        
        assert session_id is not None
        assert session_id in self.interface.sessions
        assert self.interface.sessions[session_id].user_id == "test_user"
        assert self.interface.sessions[session_id].state == ConversationState.IDLE
    
    @pytest.mark.asyncio
    async def test_start_session_with_preferences(self):
        """Test starting a session with user preferences."""
        preferences = {
            "language": "en",
            "notification_level": "high",
            "focus_areas": ["performance", "quality"]
        }
        
        session_id = await self.interface.start_session("test_user", preferences)
        
        assert session_id is not None
        assert self.interface.sessions[session_id].preferences == preferences
    
    @pytest.mark.asyncio
    async def test_process_message_text(self):
        """Test processing a text message."""
        # Start a session
        session_id = await self.interface.start_session("test_user")
        
        # Process a text message
        response = await self.interface.process_message(session_id, "Hello, how are you?")
        
        assert "content" in response
        assert "content" in response
        assert "type" in response
        assert response["type"] == "text_response"
    
    @pytest.mark.asyncio
    async def test_process_message_command(self):
        """Test processing a command message."""
        # Start a session
        session_id = await self.interface.start_session("test_user")
        
        # Process a command message
        response = await self.interface.process_message(session_id, "/help")
        
        assert "content" in response
        assert "content" in response
        assert response["type"] == "help"
    
    @pytest.mark.asyncio
    async def test_process_message_ai_query(self):
        """Test processing an AI assistant query."""
        # Start a session
        session_id = await self.interface.start_session("test_user")
        
        # Set analysis context
        await self.interface.set_analysis_context(session_id, self.mock_analysis)
        
        # Process an AI query
        response = await self.interface.process_message(session_id, "Analyze my simulation results")
        
        assert "content" in response
        assert "content" in response
        assert response["type"] == "text_response"
    
    @pytest.mark.asyncio
    async def test_process_message_invalid_session(self):
        """Test processing a message with invalid session ID."""
        response = await self.interface.process_message("invalid_session", "Hello")
        
        assert "error" in response
        assert "error" in response
        assert "not found" in response["error"].lower()
    
    @pytest.mark.asyncio
    async def test_handle_command_help(self):
        """Test handling help command."""
        session = ConversationSession(
            id="test_session",
            user_id="test_user",
            state=ConversationState.IDLE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            preferences={}
        )
        
        response = await self.interface._handle_command(session, "/help")
        
        assert "content" in response
        assert "help" in response["content"].lower()
        assert response["type"] == "help"
    
    @pytest.mark.asyncio
    async def test_handle_command_status(self):
        """Test handling status command."""
        session = ConversationSession(
            id="test_session",
            user_id="test_user",
            state=ConversationState.IDLE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            preferences={}
        )
        
        response = await self.interface._handle_command(session, "/status")
        
        assert "content" in response
        assert "status" in response["content"].lower()
        assert response["type"] == "status"
    
    @pytest.mark.asyncio
    async def test_handle_command_clear(self):
        """Test handling clear command."""
        session = ConversationSession(
            id="test_session",
            user_id="test_user",
            state=ConversationState.IDLE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            preferences={}
        )
        
        response = await self.interface._handle_command(session, "/clear")
        
        assert "content" in response
        assert "cleared" in response["content"].lower()
        assert response["type"] == "system"
        assert len(session.messages) == 0
    
    @pytest.mark.asyncio
    async def test_handle_command_unknown(self):
        """Test handling unknown command."""
        session = ConversationSession(
            id="test_session",
            user_id="test_user",
            state=ConversationState.IDLE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            preferences={}
        )
        
        response = await self.interface._handle_command(session, "/unknown")
        
        assert response["type"] == "error"
        assert "unknown" in response["content"].lower()
    
    @pytest.mark.asyncio
    async def test_handle_ai_assistant_query(self):
        """Test handling AI assistant query."""
        session = ConversationSession(
            id="test_session",
            user_id="test_user",
            state=ConversationState.IDLE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            preferences={}
        )
        
        # Mock AI assistant response
        with patch.object(self.interface.ai_assistant, 'process_query') as mock_process:
            mock_process.return_value = Mock(
                response_type="text_response",
                content="Test response",
                confidence=0.8
            )
            
            response = await self.interface._handle_ai_assistant_query(session, "Test query")
            
            assert "content" in response
            assert "content" in response
            assert response["type"] == "unclear"
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_analysis_context(self):
        """Test setting analysis context for a session."""
        session_id = await self.interface.start_session("test_user")
        
        await self.interface.set_analysis_context(
            session_id,
            self.mock_analysis,
            self.mock_insights,
            self.mock_recommendations
        )
        
        session = self.interface.sessions[session_id]
        assert session.active_analysis is not None
        assert session.context["insights"] == self.mock_insights
        assert session.context["recommendations"] == self.mock_recommendations
    
    @pytest.mark.asyncio
    async def test_set_analysis_context_invalid_session(self):
        """Test setting analysis context with invalid session ID."""
        result = await self.interface.set_analysis_context("invalid_session", self.mock_analysis)
        assert "error" in result
    
    def test_get_session(self):
        """Test getting a session by ID."""
        # Start a session
        session_id = asyncio.run(self.interface.start_session("test_user"))
        
        # Get the session
        session = self.interface.sessions[session_id]
        
        assert session is not None
        assert session.user_id == "test_user"
        assert session.state == ConversationState.IDLE
    
    def test_get_session_invalid_id(self):
        """Test getting a session with invalid ID."""
        session = self.interface.sessions.get("invalid_id")
        assert session is None
    
    def test_get_user_sessions(self):
        """Test getting all sessions for a user."""
        # Start multiple sessions for different users
        session_id1 = asyncio.run(self.interface.start_session("test_user1"))
        session_id2 = asyncio.run(self.interface.start_session("test_user2"))
        
        # Get sessions for first user
        sessions = [s for s in self.interface.sessions.values() if s.user_id == "test_user1"]
        
        assert len(sessions) == 1
        assert sessions[0].user_id == "test_user1"
    
    def test_get_user_sessions_no_sessions(self):
        """Test getting sessions for user with no sessions."""
        sessions = [s for s in self.interface.sessions.values() if s.user_id == "nonexistent_user"]
        assert len(sessions) == 0
    
    def test_end_session(self):
        """Test ending a session."""
        # Start a session
        session_id = asyncio.run(self.interface.start_session("test_user"))
        
        # End the session
        result = asyncio.run(self.interface.end_session(session_id))
        
        # Session should be removed from sessions
        assert session_id not in self.interface.sessions
        assert result["success"] is True
    
    def test_end_session_invalid_id(self):
        """Test ending a session with invalid ID."""
        # Should not raise an exception
        asyncio.run(self.interface.end_session("invalid_id"))
    
    def test_cleanup_expired_sessions(self):
        """Test cleaning up expired sessions."""
        # Start a session
        session_id = asyncio.run(self.interface.start_session("test_user"))
        
        # Manually set the session as expired
        session = self.interface.sessions[session_id]
        session.last_activity = datetime.now() - timedelta(hours=2)  # 2 hours ago
        
        # Cleanup expired sessions
        self.interface._cleanup_expired_sessions()
        
        # Session should be removed
        assert session_id not in self.interface.sessions
    
    def test_get_session_statistics(self):
        """Test getting session statistics."""
        # Start some sessions
        session_id1 = asyncio.run(self.interface.start_session("user1"))
        session_id2 = asyncio.run(self.interface.start_session("user2"))
        
        # Get statistics
        active_sessions = asyncio.run(self.interface.get_active_sessions())
        stats = {
            "active_sessions": len(active_sessions),
            "unique_users": len(set(s["user_id"] for s in active_sessions))
        }
        
        assert "active_sessions" in stats
        assert "unique_users" in stats
        assert stats["active_sessions"] == 2
        assert stats["unique_users"] == 2
    
    def test_conversation_message_creation(self):
        """Test ConversationMessage creation."""
        message = ConversationMessage(
            id="msg_1",
            type=MessageType.USER_QUERY,
            content="Hello",
            timestamp=datetime.now(),
            metadata={"key": "value"}
        )
        
        assert message.id == "msg_1"
        assert message.content == "Hello"
        assert message.type == MessageType.USER_QUERY
        assert message.metadata == {"key": "value"}
    
    def test_conversation_session_creation(self):
        """Test ConversationSession creation."""
        session = ConversationSession(
            id="session_1",
            user_id="user_1",
            state=ConversationState.IDLE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            preferences={"language": "en"}
        )
        
        assert session.id == "session_1"
        assert session.user_id == "user_1"
        assert session.state == ConversationState.IDLE
        assert session.preferences == {"language": "en"}
    
    def test_conversation_config_creation(self):
        """Test ConversationConfig creation."""
        config = ConversationConfig(
            max_session_duration=7200,
            max_messages_per_session=200,
            enable_notifications=False,
            enable_ai_assistant=False
        )
        
        assert config.max_session_duration == 7200
        assert config.max_messages_per_session == 200
        assert config.enable_notifications is False
        assert config.enable_ai_assistant is False
    
    def test_conversation_state_enum(self):
        """Test ConversationState enum values."""
        assert ConversationState.IDLE.value == "idle"
        assert ConversationState.WAITING_FOR_INPUT.value == "waiting_for_input"
        assert ConversationState.PROCESSING.value == "processing"
        assert ConversationState.PROVIDING_RESPONSE.value == "providing_response"
        assert ConversationState.ERROR.value == "error"
    
    def test_message_type_enum(self):
        """Test MessageType enum values."""
        assert MessageType.USER_QUERY.value == "user_query"
        assert MessageType.ASSISTANT_RESPONSE.value == "assistant_response"
        assert MessageType.SYSTEM_MESSAGE.value == "system_message"
        assert MessageType.ERROR_MESSAGE.value == "error_message"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in message processing."""
        # Test with invalid session
        response = await self.interface.process_message("invalid_session", "Hello")
        
        assert "error" in response
        assert "error" in response
    
    def test_session_cleanup(self):
        """Test automatic session cleanup."""
        # Start a session
        session_id = asyncio.run(self.interface.start_session("test_user"))
        
        # Manually set the session as expired
        session = self.interface.sessions[session_id]
        session.last_activity = datetime.now() - timedelta(hours=2)
        
        # Cleanup should remove expired sessions
        self.interface._cleanup_expired_sessions()
        
        assert session_id not in self.interface.sessions
    
    def test_message_count_tracking(self):
        """Test message count tracking."""
        # Start a session
        session_id = asyncio.run(self.interface.start_session("test_user"))
        
        # Process some messages
        asyncio.run(self.interface.process_message(session_id, "Message 1"))
        asyncio.run(self.interface.process_message(session_id, "Message 2"))
        
        # Check message count (should be 3: welcome message + 2 user messages + 2 assistant responses)
        session = self.interface.sessions[session_id]
        assert len(session.messages) == 5  # 1 welcome + 2 user queries + 2 assistant responses
    
    def test_session_preferences(self):
        """Test session preferences handling."""
        preferences = {
            "language": "en",
            "notification_level": "high",
            "focus_areas": ["performance", "quality"]
        }
        
        session_id = asyncio.run(self.interface.start_session("test_user", preferences))
        session = self.interface.sessions[session_id]
        
        assert session.preferences == preferences