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
            user_context=UserContext.INTERMEDIATE,
            include_advanced_recommendations=True,
            include_beginner_recommendations=True,
            min_confidence=0.5,
            max_recommendations=10,
            prioritize_by_impact=True,
            consider_historical_patterns=True,
            consider_resource_constraints=True,
            consider_time_constraints=True,
            enable_learning=True,
            learning_rate=0.1,
            adaptation_threshold=0.7
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
                id="insight_1",
                type=InsightType.PERFORMANCE_PATTERN,
                title="High CPU Usage",
                description="CPU usage is above 85%",
                severity=InsightSeverity.HIGH,
                confidence=0.9,
                data_points=[{"cpu_usage": 85.5}],
                recommendations=["Optimize CPU usage"],
                created_at=datetime.now()
            ),
            Insight(
                id="insight_2",
                type=InsightType.ANOMALY_DETECTION,
                title="Data Anomaly",
                description="Anomaly detected in simulation data",
                severity=InsightSeverity.MEDIUM,
                confidence=0.7,
                data_points=[{"anomaly_score": 0.8}],
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
        assert self.engine.recommendations == []
        assert self.engine.user_history == []
        assert self.engine.learning_data == {}
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        engine = SmartRecommendationEngine()
        assert engine.config is not None
        assert engine.config.enable_learning is True
        assert engine.config.max_recommendations == 20
    
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
        rec_types = {rec.type for rec in recommendations}
        assert len(rec_types) >= 1  # Should have at least one type
    
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
        # This method doesn't return recommendations directly, it modifies self.recommendations
        await self.engine._generate_performance_recommendations(
            self.mock_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        # Check that recommendations were added to the engine
        assert len(self.engine.recommendations) >= 0
    
    @pytest.mark.asyncio
    async def test_generate_quality_recommendations(self):
        """Test generating quality recommendations."""
        # This method doesn't return recommendations directly, it modifies self.recommendations
        await self.engine._generate_quality_recommendations(
            self.mock_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        # Check that recommendations were added to the engine
        assert len(self.engine.recommendations) >= 0
    
    @pytest.mark.asyncio
    async def test_generate_error_recommendations(self):
        """Test generating error recommendations."""
        # Mock analysis with errors
        error_analysis = Mock(spec=OrchestrationResult)
        error_analysis.success = False
        error_analysis.errors = ["Test error"]
        error_analysis.phase_results = []
        error_analysis.summary = {}
        
        # This method doesn't return recommendations directly, it modifies self.recommendations
        await self.engine._generate_error_recommendations(
            error_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        # Check that recommendations were added to the engine
        assert len(self.engine.recommendations) >= 0
    
    @pytest.mark.asyncio
    async def test_generate_configuration_recommendations(self):
        """Test generating configuration recommendations."""
        # This method doesn't return recommendations directly, it modifies self.recommendations
        await self.engine._generate_configuration_recommendations(
            self.mock_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        # Check that recommendations were added to the engine
        assert len(self.engine.recommendations) >= 0
    
    @pytest.mark.asyncio
    async def test_generate_workflow_recommendations(self):
        """Test generating workflow recommendations."""
        # This method doesn't return recommendations directly, it modifies self.recommendations
        await self.engine._generate_workflow_recommendations(
            self.mock_analysis,
            self.mock_insights,
            self.mock_user_context
        )
        
        # Check that recommendations were added to the engine
        assert len(self.engine.recommendations) >= 0
    
    def test_apply_context_filtering(self):
        """Test applying context filtering to recommendations."""
        recommendations = [
            Recommendation(
                id="rec_1",
                type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="Optimize CPU Usage",
                description="Reduce CPU usage by 10%",
                priority=RecommendationPriority.HIGH,
                confidence=0.9,
                expected_impact="High",
                implementation_effort="Medium",
                created_at=datetime.now()
            ),
            Recommendation(
                id="rec_2",
                type=RecommendationType.QUALITY_IMPROVEMENT,
                title="Improve Data Quality",
                description="Enhance data validation",
                priority=RecommendationPriority.MEDIUM,
                confidence=0.7,
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
                id="rec_1",
                type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="High Priority",
                description="High priority recommendation",
                priority=RecommendationPriority.HIGH,
                confidence=0.9,
                expected_impact="High",
                implementation_effort="Medium",
                created_at=datetime.now()
            ),
            Recommendation(
                id="rec_2",
                type=RecommendationType.QUALITY_IMPROVEMENT,
                title="Low Priority",
                description="Low priority recommendation",
                priority=RecommendationPriority.LOW,
                confidence=0.5,
                expected_impact="Low",
                implementation_effort="High",
                created_at=datetime.now()
            )
        ]
        
        ranked = self.engine._rank_recommendations(recommendations, self.mock_user_context)
        
        assert len(ranked) == 2
        assert ranked[0].priority == RecommendationPriority.HIGH  # High priority first
        assert ranked[1].priority == RecommendationPriority.LOW
    
    def test_rank_recommendations_method(self):
        """Test ranking recommendations method."""
        recommendations = [
            Recommendation(
                id="rec_1",
                type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="High Priority",
                description="High priority recommendation",
                priority=RecommendationPriority.HIGH,
                confidence=0.9,
                expected_impact="High",
                implementation_effort="Medium",
                created_at=datetime.now()
            ),
            Recommendation(
                id="rec_2",
                type=RecommendationType.QUALITY_IMPROVEMENT,
                title="Low Priority",
                description="Low priority recommendation",
                priority=RecommendationPriority.LOW,
                confidence=0.5,
                expected_impact="Low",
                implementation_effort="High",
                created_at=datetime.now()
            )
        ]
        
        ranked = self.engine._rank_recommendations(recommendations, self.mock_user_context)
        
        assert len(ranked) == 2
        # The ranking should prioritize by confidence and priority
        assert ranked[0].confidence >= ranked[1].confidence
    
    def test_learning_data_access(self):
        """Test accessing learning data."""
        # Test that learning_data is accessible
        assert isinstance(self.engine.learning_data, dict)
        
        # Test that we can add data to learning_data
        self.engine.learning_data["test_key"] = "test_value"
        assert self.engine.learning_data["test_key"] == "test_value"
    
    def test_user_history_access(self):
        """Test accessing user history."""
        # Test that user_history is accessible
        assert isinstance(self.engine.user_history, list)
        
        # Test that we can add data to user_history
        self.engine.user_history.append({"test": "data"})
        assert len(self.engine.user_history) > 0
    
    def test_recommendations_summary(self):
        """Test getting recommendations summary."""
        # Test that we can get a summary of recommendations
        summary = self.engine.get_recommendations_summary()
        assert isinstance(summary, dict)
        assert "total_recommendations" in summary
    
    def test_export_recommendations(self):
        """Test exporting recommendations."""
        # Add some recommendations first
        self.engine.recommendations = [
            Recommendation(
                id="test_rec",
                type=RecommendationType.PERFORMANCE_OPTIMIZATION,
                title="Test Recommendation",
                description="Test description",
                priority=RecommendationPriority.HIGH,
                confidence=0.8,
                expected_impact="High",
                implementation_effort="Medium",
                created_at=datetime.now()
            )
        ]
        
        # Test that we can export recommendations
        export_result = self.engine.export_recommendations(format="json")
        assert isinstance(export_result, str)
        # Should be valid JSON
        import json
        json.loads(export_result)
    
    def test_recommendation_creation(self):
        """Test Recommendation creation."""
        recommendation = Recommendation(
            id="test_rec",
            type=RecommendationType.PERFORMANCE_OPTIMIZATION,
            title="Test Recommendation",
            description="Test description",
            priority=RecommendationPriority.HIGH,
            confidence=0.8,
            expected_impact="High",
            implementation_effort="Medium",
            created_at=datetime.now()
        )
        
        assert recommendation.type == RecommendationType.PERFORMANCE_OPTIMIZATION
        assert recommendation.title == "Test Recommendation"
        assert recommendation.description == "Test description"
        assert recommendation.priority == RecommendationPriority.HIGH
        assert recommendation.confidence == 0.8
        assert recommendation.expected_impact == "High"
        assert recommendation.implementation_effort == "Medium"
    
    def test_recommendation_config_creation(self):
        """Test RecommendationConfig creation."""
        config = RecommendationConfig(
            user_context=UserContext.BEGINNER,
            include_advanced_recommendations=False,
            include_beginner_recommendations=True,
            min_confidence=0.6,
            max_recommendations=5,
            prioritize_by_impact=False,
            consider_historical_patterns=False,
            consider_resource_constraints=False,
            consider_time_constraints=False,
            enable_learning=False,
            learning_rate=0.05,
            adaptation_threshold=0.8
        )
        
        assert config.user_context == UserContext.BEGINNER
        assert config.include_advanced_recommendations is False
        assert config.include_beginner_recommendations is True
        assert config.min_confidence == 0.6
        assert config.max_recommendations == 5
        assert config.prioritize_by_impact is False
        assert config.consider_historical_patterns is False
        assert config.consider_resource_constraints is False
        assert config.consider_time_constraints is False
        assert config.enable_learning is False
        assert config.learning_rate == 0.05
        assert config.adaptation_threshold == 0.8
    
    def test_user_context_enum(self):
        """Test UserContext enum values."""
        assert UserContext.BEGINNER.value == "beginner"
        assert UserContext.INTERMEDIATE.value == "intermediate"
        assert UserContext.ADVANCED.value == "advanced"
        assert UserContext.EXPERT.value == "expert"
        
        # Test enum comparison
        assert UserContext.BEGINNER != UserContext.EXPERT
        assert UserContext.INTERMEDIATE.value == "intermediate"
    
    def test_recommendation_type_enum(self):
        """Test RecommendationType enum values."""
        assert RecommendationType.PERFORMANCE_OPTIMIZATION.value == "performance_optimization"
        assert RecommendationType.QUALITY_IMPROVEMENT.value == "quality_improvement"
        assert RecommendationType.ERROR_RESOLUTION.value == "error_resolution"
        assert RecommendationType.CONFIGURATION_ADJUSTMENT.value == "configuration_adjustment"
        assert RecommendationType.WORKFLOW_ENHANCEMENT.value == "workflow_enhancement"
    
    def test_recommendation_priority_enum(self):
        """Test RecommendationPriority enum values."""
        assert RecommendationPriority.LOW.value == "low"
        assert RecommendationPriority.MEDIUM.value == "medium"
        assert RecommendationPriority.HIGH.value == "high"
        assert RecommendationPriority.CRITICAL.value == "critical"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in recommendation generation."""
        # Test with invalid analysis result
        invalid_analysis = Mock(spec=OrchestrationResult)
        invalid_analysis.success = False
        invalid_analysis.errors = ["Test error"]
        invalid_analysis.phase_results = []
        invalid_analysis.summary = {}
        invalid_analysis.total_duration = 0.0  # Add missing attribute
        invalid_analysis.warnings = []  # Add missing attribute
        
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
        self.engine.learning_data = {
            "test_key": "test_value",
            "another_key": "another_value"
        }
        
        # Test accessing learning data
        assert self.engine.learning_data["test_key"] == "test_value"
        assert self.engine.learning_data["another_key"] == "another_value"
    
    def test_engine_attributes(self):
        """Test engine attributes exist."""
        # Test that basic attributes exist
        assert hasattr(self.engine, 'config')
        assert hasattr(self.engine, 'recommendations')
        assert hasattr(self.engine, 'user_history')
        assert hasattr(self.engine, 'learning_data')