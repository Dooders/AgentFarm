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
                recommendation_type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="Optimize CPU Usage",
                description="Reduce CPU usage by 10%",
                priority=RecommendationPriority.HIGH,
                confidence=0.9,
                action_items=["Reduce simulation complexity"],
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
        assert self.interface.sessions[session_id].state == ConversationState.ACTIVE
    
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
        
        assert response["success"] is True
        assert "message" in response
        assert "timestamp" in response
        assert response["message_type"] == MessageType.TEXT_RESPONSE
    
    @pytest.mark.asyncio
    async def test_process_message_command(self):
        """Test processing a command message."""
        # Start a session
        session_id = await self.interface.start_session("test_user")
        
        # Process a command message
        response = await self.interface.process_message(session_id, "/help")
        
        assert response["success"] is True
        assert "message" in response
        assert response["message_type"] == MessageType.COMMAND_RESPONSE
    
    @pytest.mark.asyncio
    async def test_process_message_ai_query(self):
        """Test processing an AI assistant query."""
        # Start a session
        session_id = await self.interface.start_session("test_user")
        
        # Set analysis context
        await self.interface.set_analysis_context(session_id, self.mock_analysis)
        
        # Process an AI query
        response = await self.interface.process_message(session_id, "Analyze my simulation results")
        
        assert response["success"] is True
        assert "message" in response
        assert response["message_type"] == MessageType.AI_RESPONSE
    
    @pytest.mark.asyncio
    async def test_process_message_invalid_session(self):
        """Test processing a message with invalid session ID."""
        response = await self.interface.process_message("invalid_session", "Hello")
        
        assert response["success"] is False
        assert "error" in response
        assert "not found" in response["error"].lower()
    
    @pytest.mark.asyncio
    async def test_handle_command_help(self):
        """Test handling help command."""
        session = ConversationSession(
            session_id="test_session",
            user_id="test_user",
            state=ConversationState.ACTIVE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            message_count=0,
            preferences={}
        )
        
        response = await self.interface._handle_command(session, "/help")
        
        assert response["success"] is True
        assert "help" in response["message"].lower()
        assert response["message_type"] == MessageType.COMMAND_RESPONSE
    
    @pytest.mark.asyncio
    async def test_handle_command_status(self):
        """Test handling status command."""
        session = ConversationSession(
            session_id="test_session",
            user_id="test_user",
            state=ConversationState.ACTIVE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            message_count=0,
            preferences={}
        )
        
        response = await self.interface._handle_command(session, "/status")
        
        assert response["success"] is True
        assert "status" in response["message"].lower()
        assert response["message_type"] == MessageType.COMMAND_RESPONSE
    
    @pytest.mark.asyncio
    async def test_handle_command_clear(self):
        """Test handling clear command."""
        session = ConversationSession(
            session_id="test_session",
            user_id="test_user",
            state=ConversationState.ACTIVE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            message_count=5,
            preferences={}
        )
        
        response = await self.interface._handle_command(session, "/clear")
        
        assert response["success"] is True
        assert "cleared" in response["message"].lower()
        assert response["message_type"] == MessageType.COMMAND_RESPONSE
        assert session.message_count == 0
    
    @pytest.mark.asyncio
    async def test_handle_command_unknown(self):
        """Test handling unknown command."""
        session = ConversationSession(
            session_id="test_session",
            user_id="test_user",
            state=ConversationState.ACTIVE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            message_count=0,
            preferences={}
        )
        
        response = await self.interface._handle_command(session, "/unknown")
        
        assert response["success"] is False
        assert "unknown" in response["error"].lower()
        assert response["message_type"] == MessageType.ERROR_RESPONSE
    
    @pytest.mark.asyncio
    async def test_handle_ai_assistant_query(self):
        """Test handling AI assistant query."""
        session = ConversationSession(
            session_id="test_session",
            user_id="test_user",
            state=ConversationState.ACTIVE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            message_count=0,
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
            
            assert response["success"] is True
            assert "message" in response
            assert response["message_type"] == MessageType.AI_RESPONSE
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
        assert session.analysis_context is not None
        assert session.insights == self.mock_insights
        assert session.recommendations == self.mock_recommendations
    
    @pytest.mark.asyncio
    async def test_set_analysis_context_invalid_session(self):
        """Test setting analysis context with invalid session ID."""
        with pytest.raises(ValueError, match="Session not found"):
            await self.interface.set_analysis_context("invalid_session", self.mock_analysis)
    
    def test_get_session(self):
        """Test getting a session by ID."""
        # Start a session
        session_id = asyncio.run(self.interface.start_session("test_user"))
        
        # Get the session
        session = self.interface.get_session(session_id)
        
        assert session is not None
        assert session.user_id == "test_user"
        assert session.state == ConversationState.ACTIVE
    
    def test_get_session_invalid_id(self):
        """Test getting a session with invalid ID."""
        session = self.interface.get_session("invalid_id")
        assert session is None
    
    def test_get_user_sessions(self):
        """Test getting all sessions for a user."""
        # Start multiple sessions for the same user
        session_id1 = asyncio.run(self.interface.start_session("test_user"))
        session_id2 = asyncio.run(self.interface.start_session("test_user"))
        
        # Get user sessions
        sessions = self.interface.get_user_sessions("test_user")
        
        assert len(sessions) == 2
        assert all(session.user_id == "test_user" for session in sessions)
    
    def test_get_user_sessions_no_sessions(self):
        """Test getting sessions for user with no sessions."""
        sessions = self.interface.get_user_sessions("nonexistent_user")
        assert len(sessions) == 0
    
    def test_end_session(self):
        """Test ending a session."""
        # Start a session
        session_id = asyncio.run(self.interface.start_session("test_user"))
        
        # End the session
        self.interface.end_session(session_id)
        
        session = self.interface.sessions[session_id]
        assert session.state == ConversationState.ENDED
    
    def test_end_session_invalid_id(self):
        """Test ending a session with invalid ID."""
        # Should not raise an exception
        self.interface.end_session("invalid_id")
    
    def test_cleanup_expired_sessions(self):
        """Test cleaning up expired sessions."""
        # Start a session
        session_id = asyncio.run(self.interface.start_session("test_user"))
        
        # Manually set the session as expired
        session = self.interface.sessions[session_id]
        session.created_at = datetime.now() - timedelta(hours=2)  # 2 hours ago
        
        # Cleanup expired sessions
        self.interface.cleanup_expired_sessions()
        
        # Session should be removed
        assert session_id not in self.interface.sessions
    
    def test_get_session_statistics(self):
        """Test getting session statistics."""
        # Start some sessions
        session_id1 = asyncio.run(self.interface.start_session("user1"))
        session_id2 = asyncio.run(self.interface.start_session("user2"))
        
        # Get statistics
        stats = self.interface.get_session_statistics()
        
        assert "total_sessions" in stats
        assert "active_sessions" in stats
        assert "unique_users" in stats
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 2
        assert stats["unique_users"] == 2
    
    def test_conversation_message_creation(self):
        """Test ConversationMessage creation."""
        message = ConversationMessage(
            message_id="msg_1",
            session_id="session_1",
            user_id="user_1",
            content="Hello",
            message_type=MessageType.USER_MESSAGE,
            timestamp=datetime.now(),
            metadata={"key": "value"}
        )
        
        assert message.message_id == "msg_1"
        assert message.session_id == "session_1"
        assert message.user_id == "user_1"
        assert message.content == "Hello"
        assert message.message_type == MessageType.USER_MESSAGE
        assert message.metadata == {"key": "value"}
    
    def test_conversation_session_creation(self):
        """Test ConversationSession creation."""
        session = ConversationSession(
            session_id="session_1",
            user_id="user_1",
            state=ConversationState.ACTIVE,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            message_count=5,
            preferences={"language": "en"},
            analysis_context=self.mock_analysis,
            insights=self.mock_insights,
            recommendations=self.mock_recommendations
        )
        
        assert session.session_id == "session_1"
        assert session.user_id == "user_1"
        assert session.state == ConversationState.ACTIVE
        assert session.message_count == 5
        assert session.preferences == {"language": "en"}
        assert session.analysis_context == self.mock_analysis
        assert session.insights == self.mock_insights
        assert session.recommendations == self.mock_recommendations
    
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
        assert ConversationState.ACTIVE == "active"
        assert ConversationState.PAUSED == "paused"
        assert ConversationState.ENDED == "ended"
        assert ConversationState.ERROR == "error"
    
    def test_message_type_enum(self):
        """Test MessageType enum values."""
        assert MessageType.USER_MESSAGE == "user_message"
        assert MessageType.SYSTEM_MESSAGE == "system_message"
        assert MessageType.AI_RESPONSE == "ai_response"
        assert MessageType.COMMAND_RESPONSE == "command_response"
        assert MessageType.ERROR_RESPONSE == "error_response"
        assert MessageType.TEXT_RESPONSE == "text_response"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in message processing."""
        # Test with invalid session
        response = await self.interface.process_message("invalid_session", "Hello")
        
        assert response["success"] is False
        assert "error" in response
    
    def test_session_cleanup(self):
        """Test automatic session cleanup."""
        # Start a session
        session_id = asyncio.run(self.interface.start_session("test_user"))
        
        # Manually set the session as expired
        session = self.interface.sessions[session_id]
        session.created_at = datetime.now() - timedelta(hours=2)
        
        # Cleanup should remove expired sessions
        self.interface.cleanup_expired_sessions()
        
        assert session_id not in self.interface.sessions
    
    def test_message_count_tracking(self):
        """Test message count tracking."""
        # Start a session
        session_id = asyncio.run(self.interface.start_session("test_user"))
        
        # Process some messages
        asyncio.run(self.interface.process_message(session_id, "Message 1"))
        asyncio.run(self.interface.process_message(session_id, "Message 2"))
        
        # Check message count
        session = self.interface.sessions[session_id]
        assert session.message_count == 2
    
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