"""
Tests for AI Analysis Assistant.

This module contains comprehensive tests for the AI-powered analysis assistant
with natural language processing capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from farm.analysis.comparative.ai_assistant import (
    AIAnalysisAssistant,
    QueryType,
    ResponseType,
    QueryContext,
    AssistantResponse,
    KnowledgeEntry
)
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult


class TestAIAnalysisAssistant:
    """Test cases for AIAnalysisAssistant."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.assistant = AIAnalysisAssistant(knowledge_base_path=self.temp_dir)
        
        # Mock analysis result
        self.mock_analysis = Mock(spec=OrchestrationResult)
        self.mock_analysis.success = True
        self.mock_analysis.total_duration = 120.5
        self.mock_analysis.phase_results = [
            Mock(phase_name="statistical_analysis", duration=60.0),
            Mock(phase_name="ml_analysis", duration=60.5)
        ]
        self.mock_analysis.errors = []
        self.mock_analysis.warnings = ["Minor warning"]
        self.mock_analysis.summary = {"total_phases": 2, "success_rate": 1.0}
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test assistant initialization."""
        assert self.assistant.knowledge_base_path == Path(self.temp_dir)
        assert len(self.assistant.knowledge_base) > 0  # Should have default entries
        assert self.assistant.conversation_history == []
        assert self.assistant.current_analysis is None
    
    def test_initialization_with_openai_key(self):
        """Test initialization with OpenAI API key."""
        with patch('farm.analysis.comparative.ai_assistant.OPENAI_AVAILABLE', True):
            assistant = AIAnalysisAssistant(openai_api_key="test-key")
            assert assistant.openai_api_key == "test-key"
            assert assistant.openai_available is True
    
    def test_initialization_without_openai(self):
        """Test initialization without OpenAI."""
        with patch('farm.analysis.comparative.ai_assistant.OPENAI_AVAILABLE', False):
            assistant = AIAnalysisAssistant()
            assert assistant.openai_available is False
    
    def test_initialization_without_transformers(self):
        """Test initialization without transformers."""
        with patch('farm.analysis.comparative.ai_assistant.TRANSFORMERS_AVAILABLE', False):
            assistant = AIAnalysisAssistant()
            assert assistant.transformers_available is False
    
    def test_initialization_without_spacy(self):
        """Test initialization without spaCy."""
        with patch('farm.analysis.comparative.ai_assistant.SPACY_AVAILABLE', False):
            assistant = AIAnalysisAssistant()
            assert assistant.spacy_available is False
    
    @pytest.mark.asyncio
    async def test_process_query_analysis_request(self):
        """Test processing analysis request queries."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis)
        
        # Test analysis request
        response = await self.assistant.process_query("Analyze my simulation results")
        
        assert response.response_type == ResponseType.DATA_ANALYSIS
        assert "analysis" in response.content.lower()
        assert response.confidence > 0.0
        assert response.data is not None
    
    @pytest.mark.asyncio
    async def test_process_query_result_interpretation(self):
        """Test processing result interpretation queries."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis)
        
        # Test result interpretation
        response = await self.assistant.process_query("What does this analysis mean?")
        
        assert response.response_type == ResponseType.TEXT_RESPONSE
        assert "analysis" in response.content.lower()
        assert response.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_process_query_recommendation_request(self):
        """Test processing recommendation request queries."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis)
        
        # Test recommendation request
        response = await self.assistant.process_query("What should I do next?")
        
        assert response.response_type == ResponseType.RECOMMENDATION
        assert "recommendation" in response.content.lower()
        assert response.confidence > 0.0
        assert len(response.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_process_query_trend_analysis(self):
        """Test processing trend analysis queries."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis)
        
        # Test trend analysis
        response = await self.assistant.process_query("Show me the trends")
        
        assert response.response_type == ResponseType.DATA_ANALYSIS
        assert "trend" in response.content.lower()
        assert response.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_process_query_anomaly_inquiry(self):
        """Test processing anomaly inquiry queries."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis)
        
        # Test anomaly inquiry
        response = await self.assistant.process_query("Are there any anomalies?")
        
        assert response.response_type == ResponseType.DATA_ANALYSIS
        assert "anomaly" in response.content.lower()
        assert response.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_process_query_performance_question(self):
        """Test processing performance questions."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis)
        
        # Test performance question
        response = await self.assistant.process_query("How is the performance?")
        
        assert response.response_type == ResponseType.DATA_ANALYSIS
        assert "performance" in response.content.lower()
        assert response.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_process_query_general_help(self):
        """Test processing general help queries."""
        # Test general help
        response = await self.assistant.process_query("Help me understand what you can do")
        
        assert response.response_type == ResponseType.TEXT_RESPONSE
        assert "help" in response.content.lower()
        assert response.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_process_query_without_analysis_context(self):
        """Test processing queries without analysis context."""
        # Test without analysis context
        response = await self.assistant.process_query("Analyze my results")
        
        assert response.response_type == ResponseType.TEXT_RESPONSE
        assert "analysis" in response.content.lower()
        assert "don't have" in response.content.lower() or "no" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_parse_query(self):
        """Test query parsing."""
        # Test analysis request
        context = await self.assistant._parse_query("Analyze my simulation data")
        assert context.query_type == QueryType.ANALYSIS_REQUEST
        assert context.confidence > 0.0
        
        # Test result interpretation
        context = await self.assistant._parse_query("What does this mean?")
        assert context.query_type == QueryType.RESULT_INTERPRETATION
        assert context.confidence > 0.0
        
        # Test recommendation request
        context = await self.assistant._parse_query("Give me recommendations")
        assert context.query_type == QueryType.RECOMMENDATION_REQUEST
        assert context.confidence > 0.0
        
        # Test trend analysis
        context = await self.assistant._parse_query("Show me trends over time")
        assert context.query_type == QueryType.TREND_ANALYSIS
        assert context.confidence > 0.0
        
        # Test anomaly inquiry
        context = await self.assistant._parse_query("Find anomalies in my data")
        assert context.query_type == QueryType.ANOMALY_INQUIRY
        assert context.confidence > 0.0
        
        # Test performance question
        context = await self.assistant._parse_query("How is the performance?")
        assert context.query_type == QueryType.PERFORMANCE_QUESTION
        assert context.confidence > 0.0
        
        # Test general help
        context = await self.assistant._parse_query("Help me")
        assert context.query_type == QueryType.GENERAL_HELP
        assert context.confidence > 0.0
        
        # Test unknown query
        context = await self.assistant._parse_query("Random text")
        assert context.query_type == QueryType.UNKNOWN
        assert context.confidence > 0.0
    
    def test_extract_parameters(self):
        """Test parameter extraction from queries."""
        # Test number extraction
        params = self.assistant._extract_parameters("Run analysis for 5 simulations")
        assert "numbers" in params
        assert 5.0 in params["numbers"]
        
        # Test time references
        params = self.assistant._extract_parameters("Analysis took 2 hours")
        assert "hours" in params
        assert 2 in params["hours"]
        
        # Test comparison terms
        params = self.assistant._extract_parameters("This is better than that")
        assert "comparison_terms" in params
        assert "better" in params["comparison_terms"]
    
    def test_find_relevant_knowledge(self):
        """Test finding relevant knowledge entries."""
        # Test with simulation query
        relevant = self.assistant._find_relevant_knowledge("simulation analysis")
        assert len(relevant) > 0
        assert any("simulation" in entry.tags for entry in relevant)
        
        # Test with performance query
        relevant = self.assistant._find_relevant_knowledge("performance optimization")
        assert len(relevant) > 0
        assert any("performance" in entry.tags for entry in relevant)
        
        # Test with empty query
        relevant = self.assistant._find_relevant_knowledge("")
        assert len(relevant) == 0
    
    def test_generate_analysis_summary(self):
        """Test analysis summary generation."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis)
        
        # Generate summary
        summary = self.assistant._generate_analysis_summary()
        
        assert "Analysis Summary" in summary
        assert "Success" in summary
        assert "120.5" in summary  # Duration
        assert "2" in summary  # Phase count
    
    def test_extract_analysis_data(self):
        """Test analysis data extraction."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis)
        
        # Extract data
        data = self.assistant._extract_analysis_data()
        
        assert data["success"] is True
        assert data["duration"] == 120.5
        assert data["phases"] == 2
        assert data["errors"] == 0
        assert data["warnings"] == 1
    
    def test_generate_result_interpretation(self):
        """Test result interpretation generation."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis)
        
        # Create query context
        query_context = QueryContext(
            query_text="What does this mean?",
            query_type=QueryType.RESULT_INTERPRETATION
        )
        
        # Generate interpretation
        interpretation = self.assistant._generate_result_interpretation(query_context, [])
        
        assert "analysis" in interpretation.lower()
        assert "successfully" in interpretation.lower()
        assert "120.5" in interpretation
    
    def test_generate_smart_recommendations(self):
        """Test smart recommendations generation."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis)
        
        # Create query context
        query_context = QueryContext(
            query_text="Give me recommendations",
            query_type=QueryType.RECOMMENDATION_REQUEST
        )
        
        # Generate recommendations
        recommendations = self.assistant._generate_smart_recommendations(query_context)
        
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
    
    def test_set_analysis_context(self):
        """Test setting analysis context."""
        # Set analysis context
        self.assistant.set_analysis_context(self.mock_analysis, {"test": "value"})
        
        assert self.assistant.current_analysis == self.mock_analysis
        assert self.assistant.analysis_context == {"test": "value"}
    
    def test_add_knowledge_entry(self):
        """Test adding knowledge entries."""
        # Add knowledge entry
        entry_id = self.assistant.add_knowledge_entry(
            title="Test Entry",
            content="Test content",
            category="test",
            tags=["test", "example"]
        )
        
        assert entry_id is not None
        assert len(self.assistant.knowledge_base) > 0
        
        # Find the added entry
        entry = next((e for e in self.assistant.knowledge_base if e.id == entry_id), None)
        assert entry is not None
        assert entry.title == "Test Entry"
        assert entry.content == "Test content"
        assert entry.category == "test"
        assert "test" in entry.tags
    
    def test_get_conversation_history(self):
        """Test getting conversation history."""
        # Add some messages
        self.assistant.conversation_history = [
            {"query": "Test query", "response": "Test response", "timestamp": "2023-01-01T00:00:00"}
        ]
        
        history = self.assistant.get_conversation_history()
        assert len(history) == 1
        assert history[0]["query"] == "Test query"
        assert history[0]["response"] == "Test response"
    
    def test_clear_conversation_history(self):
        """Test clearing conversation history."""
        # Add some messages
        self.assistant.conversation_history = [
            {"query": "Test query", "response": "Test response", "timestamp": "2023-01-01T00:00:00"}
        ]
        
        # Clear history
        self.assistant.clear_conversation_history()
        assert len(self.assistant.conversation_history) == 0
    
    def test_get_knowledge_base_stats(self):
        """Test getting knowledge base statistics."""
        stats = self.assistant.get_knowledge_base_stats()
        
        assert "total_entries" in stats
        assert "categories" in stats
        assert "total_tags" in stats
        assert stats["total_entries"] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in query processing."""
        # Test with invalid query
        response = await self.assistant.process_query("")
        
        assert response.response_type == ResponseType.TEXT_RESPONSE
        assert "not sure" in response.content.lower() or "understand" in response.content.lower()
    
    def test_query_context_creation(self):
        """Test QueryContext creation."""
        context = QueryContext(
            query_text="Test query",
            query_type=QueryType.ANALYSIS_REQUEST,
            entities=["entity1", "entity2"],
            intent="test_intent",
            confidence=0.8,
            parameters={"param1": "value1"},
            conversation_history=[{"query": "prev", "response": "prev_resp"}]
        )
        
        assert context.query_text == "Test query"
        assert context.query_type == QueryType.ANALYSIS_REQUEST
        assert context.entities == ["entity1", "entity2"]
        assert context.intent == "test_intent"
        assert context.confidence == 0.8
        assert context.parameters == {"param1": "value1"}
        assert len(context.conversation_history) == 1
    
    def test_assistant_response_creation(self):
        """Test AssistantResponse creation."""
        response = AssistantResponse(
            response_type=ResponseType.TEXT_RESPONSE,
            content="Test response",
            data={"key": "value"},
            visualizations=[{"type": "chart", "data": []}],
            recommendations=["rec1", "rec2"],
            actions=[{"action": "test", "params": {}}],
            confidence=0.9,
            metadata={"meta": "data"}
        )
        
        assert response.response_type == ResponseType.TEXT_RESPONSE
        assert response.content == "Test response"
        assert response.data == {"key": "value"}
        assert len(response.visualizations) == 1
        assert len(response.recommendations) == 2
        assert len(response.actions) == 1
        assert response.confidence == 0.9
        assert response.metadata == {"meta": "data"}
    
    def test_knowledge_entry_creation(self):
        """Test KnowledgeEntry creation."""
        entry = KnowledgeEntry(
            id="test_id",
            title="Test Title",
            content="Test content",
            category="test",
            tags=["tag1", "tag2"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            usage_count=5,
            relevance_score=0.8
        )
        
        assert entry.id == "test_id"
        assert entry.title == "Test Title"
        assert entry.content == "Test content"
        assert entry.category == "test"
        assert entry.tags == ["tag1", "tag2"]
        assert entry.usage_count == 5
        assert entry.relevance_score == 0.8