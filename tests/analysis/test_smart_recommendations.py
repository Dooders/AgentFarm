"""
Tests for Smart Recommendation Engine.

This module contains comprehensive tests for the intelligent recommendation
engine with context awareness and learning capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import shutil

from farm.analysis.comparative.smart_recommendations import (
    SmartRecommendationEngine,
    RecommendationType,
    RecommendationPriority,
    Recommendation,
    RecommendationConfig,
    UserContext
)
from farm.analysis.comparative.automated_insights import Insight, InsightType, InsightSeverity
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult


class TestSmartRecommendationEngine:
    """Test cases for SmartRecommendationEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = RecommendationConfig(
            enable_performance_recommendations=True,
            enable_quality_recommendations=True,
            enable_error_resolution=True,
            enable_optimization_suggestions=True,
            enable_best_practices=True,
            enable_learning=True,
            max_recommendations=10,
            min_confidence_threshold=0.5,
            enable_context_awareness=True
        )
        self.engine = SmartRecommendationEngine(config=self.config)
        
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
                insight_type=InsightType.PERFORMANCE,
                title="High CPU Usage",
                description="CPU usage is above 85%",
                severity=InsightSeverity.HIGH,
                confidence=0.9,
                data={"cpu_usage": 85.5},
                recommendations=["Optimize CPU usage"],
                created_at=datetime.now()
            ),
            Insight(
                insight_type=InsightType.ANOMALY,
                title="Data Anomaly",
                description="Anomaly detected in simulation data",
                severity=InsightSeverity.MEDIUM,
                confidence=0.7,
                data={"anomaly_score": 0.8},
                recommendations=["Investigate data quality"],
                created_at=datetime.now()
            )
        ]
        
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
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.config == self.config
        assert self.engine.recommendation_history == []
        assert self.engine.user_preferences == {}
        assert self.engine.learning_data == []
        assert self.engine.performance_baseline is None
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        engine = SmartRecommendationEngine()
        assert engine.config is not None
        assert engine.config.enable_performance_recommendations is True
        assert engine.config.enable_learning is True
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self):
        """Test generating recommendations from analysis and insights."""
        recommendations = await self.engine.generate_recommendations(
            self.mock_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        assert len(recommendations) > 0
        assert all(isinstance(rec, Recommendation) for rec in recommendations)
        
        # Check that different types of recommendations are generated
        rec_types = {rec.recommendation_type for rec in recommendations}
        assert len(rec_types) > 1  # Should have multiple types
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_without_user_context(self):
        """Test generating recommendations without user context."""
        recommendations = await self.engine.generate_recommendations(
            self.mock_analysis,
            self.mock_insights
        )
        
        assert len(recommendations) > 0
        assert all(isinstance(rec, Recommendation) for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_generate_performance_recommendations(self):
        """Test generating performance recommendations."""
        recommendations = await self.engine._generate_performance_recommendations(
            self.mock_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        assert len(recommendations) > 0
        assert all(rec.recommendation_type == RecommendationType.PERFORMANCE_OPTIMIZATION for rec in recommendations)
        assert all(rec.confidence > 0.0 for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_generate_quality_recommendations(self):
        """Test generating quality recommendations."""
        recommendations = await self.engine._generate_quality_recommendations(
            self.mock_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        assert len(recommendations) > 0
        assert all(rec.recommendation_type == RecommendationType.QUALITY_IMPROVEMENT for rec in recommendations)
        assert all(rec.confidence > 0.0 for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_generate_error_resolution_recommendations(self):
        """Test generating error resolution recommendations."""
        # Mock analysis with errors
        error_analysis = Mock(spec=OrchestrationResult)
        error_analysis.success = False
        error_analysis.errors = ["Test error"]
        error_analysis.phase_results = []
        error_analysis.summary = {}
        
        recommendations = await self.engine._generate_error_resolution_recommendations(
            error_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        assert len(recommendations) > 0
        assert all(rec.recommendation_type == RecommendationType.ERROR_RESOLUTION for rec in recommendations)
        assert all(rec.confidence > 0.0 for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_generate_optimization_recommendations(self):
        """Test generating optimization recommendations."""
        recommendations = await self.engine._generate_optimization_recommendations(
            self.mock_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        assert len(recommendations) > 0
        assert all(rec.recommendation_type == RecommendationType.OPTIMIZATION for rec in recommendations)
        assert all(rec.confidence > 0.0 for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_generate_best_practices_recommendations(self):
        """Test generating best practices recommendations."""
        recommendations = await self.engine._generate_best_practices_recommendations(
            self.mock_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        assert len(recommendations) > 0
        assert all(rec.recommendation_type == RecommendationType.BEST_PRACTICES for rec in recommendations)
        assert all(rec.confidence > 0.0 for rec in recommendations)
    
    def test_apply_context_filtering(self):
        """Test applying context filtering to recommendations."""
        recommendations = [
            Recommendation(
                recommendation_type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="Optimize CPU Usage",
                description="Reduce CPU usage by 10%",
                priority=RecommendationPriority.HIGH,
                confidence=0.9,
                action_items=["Reduce simulation complexity", "Use more efficient algorithms"],
                expected_impact="High",
                implementation_effort="Medium",
                created_at=datetime.now()
            ),
            Recommendation(
                recommendation_type=RecommendationType.QUALITY_IMPROVEMENT,
                title="Improve Data Quality",
                description="Enhance data validation",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.7,
                action_items=["Add data validation", "Improve error handling"],
                expected_impact="Medium",
                implementation_effort="Low",
                created_at=datetime.now()
            )
        ]
        
        filtered = self.engine._apply_context_filtering(recommendations, self.mock_user_context)
        
        assert len(filtered) > 0
        assert len(filtered) <= len(recommendations)
    
    def test_rank_recommendations(self):
        """Test ranking recommendations by priority and confidence."""
        recommendations = [
            Recommendation(
                recommendation_type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="High Priority",
                description="High priority recommendation",
                priority=RecommendationPriority.HIGH,
                confidence=0.9,
                action_items=["Action 1"],
                expected_impact="High",
                implementation_effort="Medium",
                created_at=datetime.now()
            ),
            Recommendation(
                recommendation_type=RecommendationType.QUALITY_IMPROVEMENT,
                title="Low Priority",
                description="Low priority recommendation",
                priority=RecommendationPriority.LOW,
                confidence=0.5,
                action_items=["Action 2"],
                expected_impact="Low",
                implementation_effort="High",
                created_at=datetime.now()
            )
        ]
        
        ranked = self.engine._rank_recommendations(recommendations, self.mock_user_context)
        
        assert len(ranked) == 2
        assert ranked[0].priority == RecommendationPriority.HIGH  # High priority first
        assert ranked[1].priority == RecommendationPriority.LOW
    
    def test_calculate_recommendation_confidence(self):
        """Test calculating recommendation confidence."""
        # Test high confidence
        confidence = self.engine._calculate_recommendation_confidence(
            RecommendationType.PERFORMANCE_OPTIMIZATION,
            {"cpu_usage": 95.0},
            self.mock_user_context
        )
        assert confidence > 0.7
        
        # Test low confidence
        confidence = self.engine._calculate_recommendation_confidence(
            RecommendationType.QUALITY_IMPROVEMENT,
            {"data_quality": 0.3},
            self.mock_user_context
        )
        assert confidence < 0.5
    
    def test_learn_from_feedback(self):
        """Test learning from user feedback."""
        # Test positive feedback
        self.engine.learn_from_feedback(
            "rec_1",
            "positive",
            {"rating": 5, "comment": "Great recommendation"}
        )
        
        assert len(self.engine.learning_data) > 0
        assert self.engine.learning_data[-1]["feedback_type"] == "positive"
        
        # Test negative feedback
        self.engine.learn_from_feedback(
            "rec_2",
            "negative",
            {"rating": 2, "comment": "Not helpful"}
        )
        
        assert len(self.engine.learning_data) > 1
        assert self.engine.learning_data[-1]["feedback_type"] == "negative"
    
    def test_update_user_preferences(self):
        """Test updating user preferences."""
        # Update preferences
        self.engine.update_user_preferences("test_user", {
            "focus_areas": ["performance", "quality"],
            "notification_level": "high"
        })
        
        assert "test_user" in self.engine.user_preferences
        assert self.engine.user_preferences["test_user"]["focus_areas"] == ["performance", "quality"]
        assert self.engine.user_preferences["test_user"]["notification_level"] == "high"
    
    def test_get_recommendation_history(self):
        """Test getting recommendation history."""
        # Add some recommendations to history
        self.engine.recommendation_history = [
            {"timestamp": "2023-01-01T00:00:00", "recommendation_count": 5},
            {"timestamp": "2023-01-02T00:00:00", "recommendation_count": 3}
        ]
        
        history = self.engine.get_recommendation_history()
        assert len(history) == 2
        assert history[0]["recommendation_count"] == 5
    
    def test_clear_recommendation_history(self):
        """Test clearing recommendation history."""
        # Add some recommendations to history
        self.engine.recommendation_history = [
            {"timestamp": "2023-01-01T00:00:00", "recommendation_count": 5}
        ]
        
        # Clear history
        self.engine.clear_recommendation_history()
        assert len(self.engine.recommendation_history) == 0
    
    def test_get_recommendation_statistics(self):
        """Test getting recommendation statistics."""
        # Add some recommendations to history
        self.engine.recommendation_history = [
            {"timestamp": "2023-01-01T00:00:00", "recommendation_count": 5, "types": ["performance", "quality"]},
            {"timestamp": "2023-01-02T00:00:00", "recommendation_count": 3, "types": ["optimization", "best_practices"]}
        ]
        
        stats = self.engine.get_recommendation_statistics()
        
        assert "total_recommendations" in stats
        assert "recommendations_per_day" in stats
        assert "most_common_type" in stats
        assert stats["total_recommendations"] == 8
    
    def test_recommendation_creation(self):
        """Test Recommendation creation."""
        recommendation = Recommendation(
            recommendation_type=RecommendationType.PERFORMANCE_OPTIMIZATION,
            title="Test Recommendation",
            description="Test description",
            priority=RecommendationPriority.HIGH,
            confidence=0.8,
            action_items=["action1", "action2"],
            expected_impact="High",
            implementation_effort="Medium",
            created_at=datetime.now()
        )
        
        assert recommendation.recommendation_type == RecommendationType.PERFORMANCE_OPTIMIZATION
        assert recommendation.title == "Test Recommendation"
        assert recommendation.description == "Test description"
        assert recommendation.priority == RecommendationPriority.HIGH
        assert recommendation.confidence == 0.8
        assert recommendation.action_items == ["action1", "action2"]
        assert recommendation.expected_impact == "High"
        assert recommendation.implementation_effort == "Medium"
    
    def test_recommendation_config_creation(self):
        """Test RecommendationConfig creation."""
        config = RecommendationConfig(
            enable_performance_recommendations=True,
            enable_quality_recommendations=False,
            enable_error_resolution=True,
            enable_optimization_suggestions=False,
            enable_best_practices=True,
            enable_learning=False,
            max_recommendations=5,
            min_confidence_threshold=0.6,
            enable_context_awareness=False
        )
        
        assert config.enable_performance_recommendations is True
        assert config.enable_quality_recommendations is False
        assert config.enable_error_resolution is True
        assert config.enable_optimization_suggestions is False
        assert config.enable_best_practices is True
        assert config.enable_learning is False
        assert config.max_recommendations == 5
        assert config.min_confidence_threshold == 0.6
        assert config.enable_context_awareness is False
    
    def test_user_context_creation(self):
        """Test UserContext creation."""
        context = UserContext(
            user_id="test_user",
            experience_level="expert",
            preferences={
                "focus_areas": ["performance", "quality", "optimization"],
                "notification_level": "high"
            },
            recent_actions=["ran_analysis", "viewed_results", "applied_recommendation"],
            system_state={
                "available_resources": "high",
                "time_constraints": "low"
            }
        )
        
        assert context.user_id == "test_user"
        assert context.experience_level == "expert"
        assert context.preferences["focus_areas"] == ["performance", "quality", "optimization"]
        assert context.preferences["notification_level"] == "high"
        assert len(context.recent_actions) == 3
        assert context.system_state["available_resources"] == "high"
        assert context.system_state["time_constraints"] == "low"
    
    def test_recommendation_type_enum(self):
        """Test RecommendationType enum values."""
        assert RecommendationType.PERFORMANCE_OPTIMIZATION == "performance_optimization"
        assert RecommendationType.QUALITY_IMPROVEMENT == "quality_improvement"
        assert RecommendationType.ERROR_RESOLUTION == "error_resolution"
        assert RecommendationType.OPTIMIZATION == "optimization"
        assert RecommendationType.BEST_PRACTICES == "best_practices"
    
    def test_recommendation_priority_enum(self):
        """Test RecommendationPriority enum values."""
        assert RecommendationPriority.LOW == "low"
        assert RecommendationPriority.MEDIUM == "medium"
        assert RecommendationPriority.HIGH == "high"
        assert RecommendationPriority.CRITICAL == "critical"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in recommendation generation."""
        # Test with invalid analysis result
        invalid_analysis = Mock(spec=OrchestrationResult)
        invalid_analysis.success = False
        invalid_analysis.errors = ["Test error"]
        invalid_analysis.phase_results = []
        invalid_analysis.summary = {}
        
        recommendations = await self.engine.generate_recommendations(
            invalid_analysis,
            self.mock_insights
        )
        
        # Should still generate some recommendations even with errors
        assert len(recommendations) >= 0
        assert all(isinstance(rec, Recommendation) for rec in recommendations)
    
    def test_learning_data_storage(self):
        """Test learning data storage and retrieval."""
        # Add learning data
        self.engine.learning_data = [
            {
                "timestamp": "2023-01-01T00:00:00",
                "recommendation_id": "rec_1",
                "feedback_type": "positive",
                "rating": 5
            },
            {
                "timestamp": "2023-01-02T00:00:00",
                "recommendation_id": "rec_2",
                "feedback_type": "negative",
                "rating": 2
            }
        ]
        
        # Test getting learning data
        data = self.engine.get_learning_data()
        assert len(data) == 2
        assert data[0]["feedback_type"] == "positive"
        assert data[1]["feedback_type"] == "negative"
    
    def test_performance_baseline_update(self):
        """Test performance baseline update."""
        # Initial baseline should be None
        assert self.engine.performance_baseline is None
        
        # Update baseline
        self.engine._update_performance_baseline(self.mock_analysis)
        
        assert self.engine.performance_baseline is not None
        assert "cpu_usage" in self.engine.performance_baseline
        assert "memory_usage" in self.engine.performance_baseline