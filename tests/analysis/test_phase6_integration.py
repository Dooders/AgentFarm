"""
Integration tests for Phase 6 components.

This module contains comprehensive integration tests for all Phase 6 components
working together in the AI-Powered Intelligence and Automation system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import shutil

from farm.analysis.comparative.ai_assistant import AIAnalysisAssistant
from farm.analysis.comparative.automated_insights import AutomatedInsightGenerator, InsightType, InsightSeverity
from farm.analysis.comparative.smart_recommendations import SmartRecommendationEngine, RecommendationType, RecommendationPriority
from farm.analysis.comparative.conversational_interface import ConversationalInterface
from farm.analysis.comparative.knowledge_base import KnowledgeBase, KnowledgeType, LearningLevel
from farm.analysis.comparative.predictive_analytics import PredictiveAnalytics, PredictionType
from farm.analysis.comparative.adaptive_learning import AdaptiveLearningSystem, LearningEventType, LearningPriority
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult


class TestPhase6Integration:
    """Integration tests for Phase 6 components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize all Phase 6 components
        self.knowledge_base = KnowledgeBase()
        self.ai_assistant = AIAnalysisAssistant(knowledge_base_path=self.temp_dir)
        self.insight_generator = AutomatedInsightGenerator()
        self.recommendation_engine = SmartRecommendationEngine()
        self.conversational_interface = ConversationalInterface()
        self.predictive_analytics = PredictiveAnalytics()
        self.adaptive_learning = AdaptiveLearningSystem(knowledge_base=self.knowledge_base)
        
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
        
        # Mock simulation data
        self.mock_simulation_data = {
            "performance_metrics": {
                "cpu_usage": [85.5, 87.2, 89.1, 86.8, 88.5],
                "memory_usage": [70.2, 72.1, 74.5, 71.8, 73.2],
                "disk_io": [45.8, 47.2, 48.9, 46.5, 47.8]
            },
            "simulation_results": {
                "total_agents": 1000,
                "simulation_time": 3600,
                "success_rate": 0.95
            }
        }
        
        # Mock user context
        self.mock_user_context = {
            "user_id": "test_user",
            "experience_level": "intermediate",
            "preferences": {
                "focus_areas": ["performance", "quality"],
                "notification_level": "medium"
            },
            "recent_actions": ["ran_analysis", "viewed_results"],
            "system_state": {
                "available_resources": "high",
                "time_constraints": "medium"
            }
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis_workflow(self):
        """Test complete end-to-end analysis workflow."""
        # 1. Generate insights from analysis
        insights = await self.insight_generator.generate_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        assert len(insights) > 0
        assert all(insight.insight_type in [InsightType.PERFORMANCE, InsightType.ANOMALY, 
                                          InsightType.TREND, InsightType.CORRELATION, 
                                          InsightType.OPTIMIZATION, InsightType.ML_INSIGHT] 
                  for insight in insights)
        
        # 2. Generate recommendations based on insights
        recommendations = await self.recommendation_engine.generate_recommendations(
            self.mock_analysis,
            insights,
            self.mock_user_context
        )
        
        assert len(recommendations) > 0
        assert all(rec.recommendation_type in [RecommendationType.PERFORMANCE_OPTIMIZATION,
                                             RecommendationType.QUALITY_IMPROVEMENT,
                                             RecommendationType.ERROR_RESOLUTION,
                                             RecommendationType.OPTIMIZATION,
                                             RecommendationType.BEST_PRACTICES]
                  for rec in recommendations)
        
        # 3. Start conversational session
        session_id = await self.conversational_interface.start_session(
            "test_user",
            self.mock_user_context["preferences"]
        )
        
        assert session_id is not None
        
        # 4. Set analysis context in conversational interface
        await self.conversational_interface.set_analysis_context(
            session_id,
            self.mock_analysis,
            insights,
            recommendations
        )
        
        # 5. Process user query through conversational interface
        response = await self.conversational_interface.process_message(
            session_id,
            "Analyze my simulation results and give me recommendations"
        )
        
        assert response["success"] is True
        assert "message" in response
        
        # 6. Record learning events
        await self.adaptive_learning.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {
                "analysis_id": "test_analysis",
                "success": True,
                "duration": 120.5,
                "insights_generated": len(insights),
                "recommendations_generated": len(recommendations)
            },
            LearningPriority.HIGH
        )
        
        # 7. Learn from the analysis
        await self.knowledge_base.learn_from_analysis(
            self.mock_analysis,
            insights,
            recommendations,
            {"user_rating": 5, "feedback": "Great analysis"}
        )
        
        # Verify learning occurred
        learning_events = self.adaptive_learning.get_learning_events()
        assert len(learning_events) > 0
        
        knowledge_stats = self.knowledge_base.get_knowledge_base_stats()
        assert knowledge_stats["total_entries"] > 0
    
    @pytest.mark.asyncio
    async def test_ai_assistant_integration(self):
        """Test AI assistant integration with other components."""
        # Set analysis context in AI assistant
        self.ai_assistant.set_analysis_context(self.mock_analysis)
        
        # Process query through AI assistant
        response = await self.ai_assistant.process_query(
            "What are the key performance issues in my simulation?"
        )
        
        assert response.response_type in ["data_analysis", "text_response"]
        assert response.confidence > 0.0
        assert response.data is not None
        
        # Test with insights and recommendations
        insights = await self.insight_generator.generate_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        recommendations = await self.recommendation_engine.generate_recommendations(
            self.mock_analysis,
            insights,
            self.mock_user_context
        )
        
        # AI assistant should be able to reference insights and recommendations
        response = await self.ai_assistant.process_query(
            "What insights were generated and what should I do about them?"
        )
        
        assert response.response_type in ["data_analysis", "text_response", "recommendation"]
        assert response.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_conversational_interface_integration(self):
        """Test conversational interface integration with AI assistant."""
        # Start session
        session_id = await self.conversational_interface.start_session("test_user")
        
        # Set analysis context
        await self.conversational_interface.set_analysis_context(
            session_id,
            self.mock_analysis
        )
        
        # Test various types of queries
        queries = [
            "Help me understand my analysis results",
            "What are the main performance issues?",
            "Give me recommendations for improvement",
            "Show me trends in my data",
            "Are there any anomalies I should be concerned about?"
        ]
        
        for query in queries:
            response = await self.conversational_interface.process_message(session_id, query)
            
            assert response["success"] is True
            assert "message" in response
            assert response["message_type"] in ["ai_response", "text_response", "data_analysis"]
    
    @pytest.mark.asyncio
    async def test_predictive_analytics_integration(self):
        """Test predictive analytics integration with other components."""
        # Create historical data
        historical_data = []
        for i in range(30):
            historical_data.append({
                "timestamp": datetime.now() - timedelta(days=30-i),
                "cpu_usage": 70.0 + i * 0.5,
                "memory_usage": 60.0 + i * 0.3,
                "disk_io": 40.0 + i * 0.2,
                "simulation_time": 1000 + i * 10,
                "success_rate": 0.95 - i * 0.001
            })
        
        # Test performance trend prediction
        predictions = await self.predictive_analytics.predict_performance_trends(
            historical_data,
            prediction_horizon=7
        )
        
        assert len(predictions) > 0
        assert all(pred.prediction_type == PredictionType.PERFORMANCE_TREND for pred in predictions)
        
        # Test anomaly prediction
        current_data = {
            "cpu_usage": 95.0,
            "memory_usage": 85.0,
            "disk_io": 65.0,
            "simulation_time": 1500,
            "success_rate": 0.85
        }
        
        anomaly_predictions = await self.predictive_analytics.predict_anomalies(
            current_data,
            historical_data
        )
        
        assert len(anomaly_predictions) > 0
        assert all(pred.prediction_type == PredictionType.ANOMALY for pred in anomaly_predictions)
        
        # Test analysis success prediction
        analysis_config = {
            "simulation_type": "agent_based",
            "agent_count": 1000,
            "simulation_duration": 3600,
            "complexity_level": "high"
        }
        
        success_prediction = await self.predictive_analytics.predict_analysis_success(
            analysis_config,
            historical_data
        )
        
        assert success_prediction.prediction_type == PredictionType.ANALYSIS_SUCCESS
        assert success_prediction.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_integration(self):
        """Test adaptive learning integration with other components."""
        # Record various learning events
        await self.adaptive_learning.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {
                "analysis_id": "analysis_1",
                "success": True,
                "duration": 100.0,
                "phases": 2
            },
            LearningPriority.HIGH
        )
        
        await self.adaptive_learning.record_learning_event(
            LearningEventType.USER_FEEDBACK,
            {
                "user_id": "user_1",
                "feedback_type": "positive",
                "rating": 5,
                "comment": "Great analysis"
            },
            LearningPriority.MEDIUM
        )
        
        await self.adaptive_learning.record_learning_event(
            LearningEventType.INSIGHT_GENERATION,
            {
                "insight_count": 5,
                "insight_types": ["performance", "anomaly"],
                "confidence_scores": [0.9, 0.8, 0.7, 0.6, 0.5]
            },
            LearningPriority.HIGH
        )
        
        # Test pattern recognition
        await self.adaptive_learning._pattern_recognition_task()
        
        # Verify learning patterns were generated
        patterns = self.adaptive_learning.get_learning_patterns()
        assert len(patterns) > 0
        
        # Test performance monitoring
        self.adaptive_learning.performance_metrics = {
            "cpu_usage": [80.0, 85.0, 90.0],
            "memory_usage": [70.0, 75.0, 80.0],
            "analysis_duration": [100.0, 110.0, 120.0]
        }
        
        await self.adaptive_learning._monitor_performance()
        
        # Test model updating
        await self.adaptive_learning._update_learning_models()
        
        # Verify learning statistics
        stats = self.adaptive_learning.get_learning_statistics()
        assert stats["total_events"] == 3
        assert "events_by_type" in stats
        assert "learning_patterns" in stats
    
    @pytest.mark.asyncio
    async def test_knowledge_base_integration(self):
        """Test knowledge base integration with other components."""
        # Add knowledge entries
        entry_id1 = await self.knowledge_base.add_entry(
            title="Performance Optimization Pattern",
            content="When CPU usage exceeds 90%, consider reducing simulation complexity",
            category="performance",
            knowledge_type=KnowledgeType.ANALYSIS_PATTERN,
            tags=["performance", "optimization", "cpu"],
            learning_level=LearningLevel.INTERMEDIATE
        )
        
        entry_id2 = await self.knowledge_base.add_entry(
            title="Anomaly Detection Best Practice",
            content="Use statistical methods for anomaly detection in time series data",
            category="anomaly_detection",
            knowledge_type=KnowledgeType.BEST_PRACTICE,
            tags=["anomaly", "detection", "statistics"],
            learning_level=LearningLevel.ADVANCED
        )
        
        assert entry_id1 is not None
        assert entry_id2 is not None
        
        # Test knowledge search
        search_results = await self.knowledge_base.search_entries(
            query="performance optimization",
            max_results=5
        )
        
        assert len(search_results) > 0
        assert any("performance" in entry.title.lower() for entry in search_results)
        
        # Test learning from analysis
        insights = await self.insight_generator.generate_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        
        recommendations = await self.recommendation_engine.generate_recommendations(
            self.mock_analysis,
            insights,
            self.mock_user_context
        )
        
        await self.knowledge_base.learn_from_analysis(
            self.mock_analysis,
            insights,
            recommendations,
            {"user_rating": 4, "feedback": "Good analysis"}
        )
        
        # Verify learning occurred
        learning_patterns = self.knowledge_base.get_learning_patterns()
        assert len(learning_patterns) > 0
        
        # Test knowledge base statistics
        stats = self.knowledge_base.get_knowledge_base_stats()
        assert stats["total_entries"] >= 2
        assert "categories" in stats
        assert "knowledge_types" in stats
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test with invalid analysis result
        invalid_analysis = Mock(spec=OrchestrationResult)
        invalid_analysis.success = False
        invalid_analysis.errors = ["Test error"]
        invalid_analysis.phase_results = []
        invalid_analysis.summary = {}
        
        # Should still work with invalid analysis
        insights = await self.insight_generator.generate_insights(invalid_analysis)
        assert len(insights) >= 0
        
        recommendations = await self.recommendation_engine.generate_recommendations(
            invalid_analysis,
            insights
        )
        assert len(recommendations) >= 0
        
        # Test with empty simulation data
        empty_data = {}
        insights = await self.insight_generator.generate_insights(
            self.mock_analysis,
            empty_data
        )
        assert len(insights) >= 0
        
        # Test with invalid user context
        invalid_context = {}
        recommendations = await self.recommendation_engine.generate_recommendations(
            self.mock_analysis,
            insights,
            invalid_context
        )
        assert len(recommendations) >= 0
    
    @pytest.mark.asyncio
    async def test_performance_integration(self):
        """Test performance characteristics of integrated components."""
        import time
        
        # Test insight generation performance
        start_time = time.time()
        insights = await self.insight_generator.generate_insights(
            self.mock_analysis,
            self.mock_simulation_data
        )
        insight_time = time.time() - start_time
        
        assert insight_time < 5.0  # Should complete within 5 seconds
        assert len(insights) > 0
        
        # Test recommendation generation performance
        start_time = time.time()
        recommendations = await self.recommendation_engine.generate_recommendations(
            self.mock_analysis,
            insights,
            self.mock_user_context
        )
        recommendation_time = time.time() - start_time
        
        assert recommendation_time < 3.0  # Should complete within 3 seconds
        assert len(recommendations) > 0
        
        # Test conversational interface performance
        session_id = await self.conversational_interface.start_session("test_user")
        await self.conversational_interface.set_analysis_context(
            session_id,
            self.mock_analysis,
            insights,
            recommendations
        )
        
        start_time = time.time()
        response = await self.conversational_interface.process_message(
            session_id,
            "Analyze my results"
        )
        conversation_time = time.time() - start_time
        
        assert conversation_time < 2.0  # Should complete within 2 seconds
        assert response["success"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations across components."""
        # Test concurrent insight generation
        tasks = []
        for i in range(5):
            task = self.insight_generator.generate_insights(
                self.mock_analysis,
                self.mock_simulation_data
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(len(insights) > 0 for insights in results)
        
        # Test concurrent recommendation generation
        insights = results[0]  # Use first set of insights
        tasks = []
        for i in range(3):
            task = self.recommendation_engine.generate_recommendations(
                self.mock_analysis,
                insights,
                self.mock_user_context
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(len(recommendations) > 0 for recommendations in results)
        
        # Test concurrent learning events
        tasks = []
        for i in range(10):
            task = self.adaptive_learning.record_learning_event(
                LearningEventType.ANALYSIS_COMPLETION,
                {"analysis_id": f"analysis_{i}", "success": True},
                LearningPriority.MEDIUM
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(event_id is not None for event_id in results)
        
        # Verify all events were recorded
        events = self.adaptive_learning.get_learning_events()
        assert len(events) == 10
    
    def test_component_initialization(self):
        """Test that all components initialize correctly."""
        # Test AI assistant initialization
        assert self.ai_assistant is not None
        assert self.ai_assistant.knowledge_base_path is not None
        
        # Test insight generator initialization
        assert self.insight_generator is not None
        assert self.insight_generator.config is not None
        
        # Test recommendation engine initialization
        assert self.recommendation_engine is not None
        assert self.recommendation_engine.config is not None
        
        # Test conversational interface initialization
        assert self.conversational_interface is not None
        assert self.conversational_interface.ai_assistant is not None
        
        # Test knowledge base initialization
        assert self.knowledge_base is not None
        assert len(self.knowledge_base.knowledge_entries) > 0
        
        # Test predictive analytics initialization
        assert self.predictive_analytics is not None
        assert self.predictive_analytics.config is not None
        
        # Test adaptive learning initialization
        assert self.adaptive_learning is not None
        assert self.adaptive_learning.knowledge_base == self.knowledge_base
    
    def test_component_configuration(self):
        """Test component configuration and customization."""
        # Test AI assistant configuration
        assert self.ai_assistant.model_name == "gpt-3.5-turbo"
        assert self.ai_assistant.openai_available is not None
        
        # Test insight generator configuration
        assert self.insight_generator.config.enable_performance_insights is True
        assert self.insight_generator.config.enable_anomaly_detection is True
        
        # Test recommendation engine configuration
        assert self.recommendation_engine.config.enable_performance_recommendations is True
        assert self.recommendation_engine.config.enable_learning is True
        
        # Test conversational interface configuration
        assert self.conversational_interface.config.enable_ai_assistant is True
        assert self.conversational_interface.config.enable_command_processing is True
        
        # Test knowledge base configuration
        assert self.knowledge_base.config.enable_semantic_search is True
        assert self.knowledge_base.config.enable_learning is True
        
        # Test predictive analytics configuration
        assert self.predictive_analytics.config.enable_ml_predictions is True
        assert self.predictive_analytics.config.enable_statistical_predictions is True
        
        # Test adaptive learning configuration
        assert self.adaptive_learning.config.enable_learning is True
        assert self.adaptive_learning.config.enable_pattern_recognition is True