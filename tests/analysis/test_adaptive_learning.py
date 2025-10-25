"""
Tests for Adaptive Learning System.

This module contains comprehensive tests for the adaptive learning system
that continuously improves the analysis system based on learning events.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import shutil

from farm.analysis.comparative.adaptive_learning import (
    AdaptiveLearningSystem,
    LearningEventType,
    LearningPriority,
    LearningEvent,
    LearningPattern,
    AdaptiveLearningConfig
)
from farm.analysis.comparative.knowledge_base import KnowledgeBase
from farm.analysis.comparative.automated_insights import Insight, InsightType, InsightSeverity
from farm.analysis.comparative.smart_recommendations import Recommendation, RecommendationType, RecommendationPriority
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult


class TestAdaptiveLearningSystem:
    """Test cases for AdaptiveLearningSystem."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.knowledge_base = KnowledgeBase()
        self.config = AdaptiveLearningConfig(
            enable_learning=True,
            learning_rate=0.1,
            min_events_for_learning=10,
            enable_pattern_recognition=True,
            enable_performance_monitoring=True,
            enable_feedback_integration=True,
            enable_model_retraining=True,
            learning_threshold=0.7
        )
        self.learning_system = AdaptiveLearningSystem(
            knowledge_base=self.knowledge_base,
            config=self.config
        )
        
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
        """Test learning system initialization."""
        assert self.learning_system.knowledge_base == self.knowledge_base
        assert self.learning_system.config == self.config
        assert self.learning_system.learning_events == []
        assert self.learning_system.learning_patterns == []
        assert self.learning_system.performance_metrics == {}
        assert self.learning_system.learning_models == {}
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        learning_system = AdaptiveLearningSystem()
        assert learning_system.config is not None
        assert learning_system.config.enable_learning is True
        assert learning_system.config.enable_pattern_recognition is True
    
    def test_initialization_without_sklearn(self):
        """Test initialization without sklearn."""
        with patch('farm.analysis.comparative.adaptive_learning.SKLEARN_AVAILABLE', False):
            learning_system = AdaptiveLearningSystem()
            assert learning_system.sklearn_available is False
            assert learning_system.learning_models == {}
    
    @pytest.mark.asyncio
    async def test_record_learning_event(self):
        """Test recording a learning event."""
        event_id = await self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {
                "analysis_id": "test_analysis",
                "success": True,
                "duration": 120.5,
                "phases": 2
            },
            LearningPriority.HIGH
        )
        
        assert event_id is not None
        assert len(self.learning_system.learning_events) > 0
        
        # Check the recorded event
        event = next((e for e in self.learning_system.learning_events if e.event_id == event_id), None)
        assert event is not None
        assert event.event_type == LearningEventType.ANALYSIS_COMPLETION
        assert event.priority == LearningPriority.HIGH
        assert event.data["analysis_id"] == "test_analysis"
    
    @pytest.mark.asyncio
    async def test_record_learning_event_without_sklearn(self):
        """Test recording a learning event without sklearn."""
        with patch('farm.analysis.comparative.adaptive_learning.SKLEARN_AVAILABLE', False):
            learning_system = AdaptiveLearningSystem()
            event_id = await learning_system.record_learning_event(
                LearningEventType.ANALYSIS_COMPLETION,
                {"analysis_id": "test_analysis"},
                LearningPriority.HIGH
            )
            
            assert event_id is not None
            assert len(learning_system.learning_events) > 0
    
    @pytest.mark.asyncio
    async def test_process_learning_event(self):
        """Test processing a learning event."""
        event = LearningEvent(
            event_id="test_event",
            event_type=LearningEventType.ANALYSIS_COMPLETION,
            data={"analysis_id": "test_analysis", "success": True},
            priority=LearningPriority.HIGH,
            timestamp=datetime.now()
        )
        
        await self.learning_system._process_learning_event(event)
        
        # Should have processed the event
        assert len(self.learning_system.learning_events) > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_analysis_completion(self):
        """Test learning from analysis completion."""
        await self.learning_system._learn_from_analysis_completion({
            "analysis_id": "test_analysis",
            "success": True,
            "duration": 120.5,
            "phases": 2,
            "performance_metrics": {
                "cpu_usage": 85.5,
                "memory_usage": 70.2
            }
        })
        
        # Should have learned patterns
        assert len(self.learning_system.learning_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_user_feedback(self):
        """Test learning from user feedback."""
        await self.learning_system._learn_from_user_feedback({
            "user_id": "test_user",
            "feedback_type": "positive",
            "rating": 5,
            "comment": "Great analysis",
            "context": "performance_analysis"
        })
        
        # Should have learned patterns
        assert len(self.learning_system.learning_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_insight_generation(self):
        """Test learning from insight generation."""
        await self.learning_system._learn_from_insight_generation({
            "insight_count": 5,
            "insight_types": ["performance", "anomaly"],
            "confidence_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
            "user_rating": 4
        })
        
        # Should have learned patterns
        assert len(self.learning_system.learning_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_recommendation_application(self):
        """Test learning from recommendation application."""
        await self.learning_system._learn_from_recommendation_application({
            "recommendation_id": "rec_1",
            "recommendation_type": "performance_optimization",
            "applied": True,
            "effectiveness": 0.8,
            "user_satisfaction": 4
        })
        
        # Should have learned patterns
        assert len(self.learning_system.learning_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_prediction_accuracy(self):
        """Test learning from prediction accuracy."""
        await self.learning_system._learn_from_prediction_accuracy({
            "prediction_type": "performance_trend",
            "predicted_value": 95.0,
            "actual_value": 92.0,
            "accuracy": 0.97,
            "confidence": 0.8
        })
        
        # Should have learned patterns
        assert len(self.learning_system.learning_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_error(self):
        """Test learning from errors."""
        await self.learning_system._learn_from_error({
            "error_type": "timeout",
            "error_message": "Analysis timed out",
            "context": "large_simulation",
            "severity": "high",
            "resolution": "reduced_complexity"
        })
        
        # Should have learned patterns
        assert len(self.learning_system.learning_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_performance_change(self):
        """Test learning from performance changes."""
        await self.learning_system._learn_from_performance_change({
            "metric": "cpu_usage",
            "old_value": 90.0,
            "new_value": 80.0,
            "change_type": "improvement",
            "context": "optimization_applied"
        })
        
        # Should have learned patterns
        assert len(self.learning_system.learning_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_pattern_recognition_task(self):
        """Test pattern recognition task."""
        # Add some learning events
        await self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {"analysis_id": "test_1", "success": True},
            LearningPriority.HIGH
        )
        await self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {"analysis_id": "test_2", "success": True},
            LearningPriority.HIGH
        )
        
        # Run pattern recognition
        await self.learning_system._pattern_recognition_task()
        
        # Should have identified patterns
        assert len(self.learning_system.learning_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_pattern_recognition_task_without_sklearn(self):
        """Test pattern recognition task without sklearn."""
        with patch('farm.analysis.comparative.adaptive_learning.SKLEARN_AVAILABLE', False):
            learning_system = AdaptiveLearningSystem()
            
            # Add some learning events
            await learning_system.record_learning_event(
                LearningEventType.ANALYSIS_COMPLETION,
                {"analysis_id": "test_1", "success": True},
                LearningPriority.HIGH
            )
            
            # Run pattern recognition
            await learning_system._pattern_recognition_task()
            
            # Should still work without sklearn
            assert len(learning_system.learning_patterns) >= 0
    
    @pytest.mark.asyncio
    async def test_monitor_performance(self):
        """Test performance monitoring."""
        # Add some performance data
        self.learning_system.performance_metrics = {
            "cpu_usage": [80.0, 85.0, 90.0],
            "memory_usage": [70.0, 75.0, 80.0],
            "analysis_duration": [100.0, 110.0, 120.0]
        }
        
        await self.learning_system._monitor_performance()
        
        # Should have monitored performance
        assert len(self.learning_system.learning_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_update_learning_models(self):
        """Test updating learning models."""
        # Add some learning events
        await self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {"analysis_id": "test_1", "success": True, "duration": 100.0},
            LearningPriority.HIGH
        )
        await self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {"analysis_id": "test_2", "success": False, "duration": 200.0},
            LearningPriority.HIGH
        )
        
        # Update learning models
        await self.learning_system._update_learning_models()
        
        # Should have updated models
        assert len(self.learning_system.learning_models) > 0
    
    @pytest.mark.asyncio
    async def test_update_learning_models_without_sklearn(self):
        """Test updating learning models without sklearn."""
        with patch('farm.analysis.comparative.adaptive_learning.SKLEARN_AVAILABLE', False):
            learning_system = AdaptiveLearningSystem()
            
            # Add some learning events
            await learning_system.record_learning_event(
                LearningEventType.ANALYSIS_COMPLETION,
                {"analysis_id": "test_1", "success": True},
                LearningPriority.HIGH
            )
            
            # Update learning models
            await learning_system._update_learning_models()
            
            # Should still work without sklearn
            assert len(learning_system.learning_models) == 0
    
    def test_get_learning_events(self):
        """Test getting learning events."""
        # Add some learning events
        asyncio.run(self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {"analysis_id": "test_1"},
            LearningPriority.HIGH
        ))
        asyncio.run(self.learning_system.record_learning_event(
            LearningEventType.USER_FEEDBACK,
            {"user_id": "user_1"},
            LearningPriority.MEDIUM
        ))
        
        # Get learning events
        events = self.learning_system.get_learning_events()
        
        assert len(events) == 2
        assert all(isinstance(event, LearningEvent) for event in events)
    
    def test_get_learning_events_by_type(self):
        """Test getting learning events by type."""
        # Add some learning events
        asyncio.run(self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {"analysis_id": "test_1"},
            LearningPriority.HIGH
        ))
        asyncio.run(self.learning_system.record_learning_event(
            LearningEventType.USER_FEEDBACK,
            {"user_id": "user_1"},
            LearningPriority.MEDIUM
        ))
        
        # Get events by type
        events = self.learning_system.get_learning_events_by_type(LearningEventType.ANALYSIS_COMPLETION)
        
        assert len(events) == 1
        assert events[0].event_type == LearningEventType.ANALYSIS_COMPLETION
    
    def test_get_learning_patterns(self):
        """Test getting learning patterns."""
        # Add some learning events to generate patterns
        asyncio.run(self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {"analysis_id": "test_1", "success": True},
            LearningPriority.HIGH
        ))
        
        # Run pattern recognition
        asyncio.run(self.learning_system._pattern_recognition_task())
        
        # Get learning patterns
        patterns = self.learning_system.get_learning_patterns()
        
        assert len(patterns) > 0
        assert all(isinstance(pattern, LearningPattern) for pattern in patterns)
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Add some performance data
        self.learning_system.performance_metrics = {
            "cpu_usage": [80.0, 85.0, 90.0],
            "memory_usage": [70.0, 75.0, 80.0]
        }
        
        # Get performance metrics
        metrics = self.learning_system.get_performance_metrics()
        
        assert len(metrics) == 2
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
    
    def test_get_learning_statistics(self):
        """Test getting learning statistics."""
        # Add some learning events
        asyncio.run(self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {"analysis_id": "test_1"},
            LearningPriority.HIGH
        ))
        asyncio.run(self.learning_system.record_learning_event(
            LearningEventType.USER_FEEDBACK,
            {"user_id": "user_1"},
            LearningPriority.MEDIUM
        ))
        
        # Get learning statistics
        stats = self.learning_system.get_learning_statistics()
        
        assert "total_events" in stats
        assert "events_by_type" in stats
        assert "events_by_priority" in stats
        assert "learning_patterns" in stats
        assert stats["total_events"] == 2
    
    def test_clear_learning_data(self):
        """Test clearing learning data."""
        # Add some learning events
        asyncio.run(self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {"analysis_id": "test_1"},
            LearningPriority.HIGH
        ))
        
        # Clear learning data
        self.learning_system.clear_learning_data()
        
        assert len(self.learning_system.learning_events) == 0
        assert len(self.learning_system.learning_patterns) == 0
        assert len(self.learning_system.performance_metrics) == 0
    
    def test_learning_event_creation(self):
        """Test LearningEvent creation."""
        event = LearningEvent(
            event_id="test_event",
            event_type=LearningEventType.ANALYSIS_COMPLETION,
            data={"analysis_id": "test_analysis", "success": True},
            priority=LearningPriority.HIGH,
            timestamp=datetime.now()
        )
        
        assert event.event_id == "test_event"
        assert event.event_type == LearningEventType.ANALYSIS_COMPLETION
        assert event.data == {"analysis_id": "test_analysis", "success": True}
        assert event.priority == LearningPriority.HIGH
    
    def test_learning_pattern_creation(self):
        """Test LearningPattern creation."""
        pattern = LearningPattern(
            pattern_id="pattern_1",
            pattern_type="analysis_success",
            description="Successful analysis pattern",
            conditions={"success_rate": 0.9, "duration": 120},
            actions=["optimize_performance", "cache_results"],
            confidence=0.8,
            frequency=10,
            last_seen=datetime.now(),
            created_at=datetime.now()
        )
        
        assert pattern.pattern_id == "pattern_1"
        assert pattern.pattern_type == "analysis_success"
        assert pattern.description == "Successful analysis pattern"
        assert pattern.conditions == {"success_rate": 0.9, "duration": 120}
        assert pattern.actions == ["optimize_performance", "cache_results"]
        assert pattern.confidence == 0.8
        assert pattern.frequency == 10
    
    def test_adaptive_learning_config_creation(self):
        """Test AdaptiveLearningConfig creation."""
        config = AdaptiveLearningConfig(
            enable_learning=False,
            learning_rate=0.2,
            min_events_for_learning=20,
            enable_pattern_recognition=False,
            enable_performance_monitoring=False,
            enable_feedback_integration=False,
            enable_model_retraining=False,
            learning_threshold=0.8
        )
        
        assert config.enable_learning is False
        assert config.learning_rate == 0.2
        assert config.min_events_for_learning == 20
        assert config.enable_pattern_recognition is False
        assert config.enable_performance_monitoring is False
        assert config.enable_feedback_integration is False
        assert config.enable_model_retraining is False
        assert config.learning_threshold == 0.8
    
    def test_learning_event_type_enum(self):
        """Test LearningEventType enum values."""
        assert LearningEventType.ANALYSIS_COMPLETION == "analysis_completion"
        assert LearningEventType.USER_FEEDBACK == "user_feedback"
        assert LearningEventType.INSIGHT_GENERATION == "insight_generation"
        assert LearningEventType.RECOMMENDATION_APPLICATION == "recommendation_application"
        assert LearningEventType.PREDICTION_ACCURACY == "prediction_accuracy"
        assert LearningEventType.ERROR == "error"
        assert LearningEventType.PERFORMANCE_CHANGE == "performance_change"
    
    def test_learning_priority_enum(self):
        """Test LearningPriority enum values."""
        assert LearningPriority.LOW == "low"
        assert LearningPriority.MEDIUM == "medium"
        assert LearningPriority.HIGH == "high"
        assert LearningPriority.CRITICAL == "critical"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in learning operations."""
        # Test with invalid event data
        event_id = await self.learning_system.record_learning_event(
            LearningEventType.ANALYSIS_COMPLETION,
            {},  # Empty data should not cause errors
            LearningPriority.HIGH
        )
        
        assert event_id is not None
        assert len(self.learning_system.learning_events) > 0
    
    def test_learning_threshold_check(self):
        """Test learning threshold checking."""
        # Test below threshold
        self.learning_system.learning_events = [Mock() for _ in range(5)]  # Below min_events_for_learning
        assert self.learning_system._should_learn() is False
        
        # Test above threshold
        self.learning_system.learning_events = [Mock() for _ in range(15)]  # Above min_events_for_learning
        assert self.learning_system._should_learn() is True
    
    def test_pattern_confidence_calculation(self):
        """Test pattern confidence calculation."""
        # Test high confidence pattern
        confidence = self.learning_system._calculate_pattern_confidence(
            "analysis_success",
            {"success_rate": 0.95, "frequency": 20}
        )
        assert confidence > 0.8
        
        # Test low confidence pattern
        confidence = self.learning_system._calculate_pattern_confidence(
            "analysis_success",
            {"success_rate": 0.5, "frequency": 2}
        )
        assert confidence < 0.5
    
    def test_learning_data_validation(self):
        """Test learning data validation."""
        # Test valid data
        valid_data = {"analysis_id": "test", "success": True}
        assert self.learning_system._validate_learning_data(valid_data) is True
        
        # Test invalid data (empty)
        assert self.learning_system._validate_learning_data({}) is False
        
        # Test invalid data (None)
        assert self.learning_system._validate_learning_data(None) is False