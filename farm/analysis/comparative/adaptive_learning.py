"""
Adaptive learning system for continuous improvement.

This module provides adaptive learning capabilities that continuously improve
the analysis system based on user feedback, performance patterns, and outcomes.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from enum import Enum
import statistics
import math
import hashlib

# Optional imports for advanced learning
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from farm.utils.logging import get_logger
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult
from farm.analysis.comparative.automated_insights import Insight, InsightType, InsightSeverity
from farm.analysis.comparative.smart_recommendations import Recommendation, RecommendationType, RecommendationPriority
from farm.analysis.comparative.knowledge_base import KnowledgeBase, KnowledgeEntry, KnowledgeType
from farm.analysis.comparative.predictive_analytics import Prediction, PredictionType

logger = get_logger(__name__)


class LearningEventType(Enum):
    """Types of learning events."""
    ANALYSIS_COMPLETED = "analysis_completed"
    USER_FEEDBACK = "user_feedback"
    INSIGHT_GENERATED = "insight_generated"
    RECOMMENDATION_APPLIED = "recommendation_applied"
    PREDICTION_ACCURATE = "prediction_accurate"
    PREDICTION_INACCURATE = "prediction_inaccurate"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_DEGRADED = "performance_degraded"
    PERFORMANCE_IMPROVED = "performance_improved"
    CONFIGURATION_CHANGED = "configuration_changed"


class LearningPriority(Enum):
    """Priority levels for learning events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LearningEvent:
    """A learning event for adaptive learning."""
    
    id: str
    type: LearningEventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    priority: LearningPriority = LearningPriority.MEDIUM
    processed: bool = False
    learning_impact: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """A learned pattern from system behavior."""
    
    id: str
    pattern_type: str
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    confidence: float
    frequency: int
    success_rate: float
    last_updated: datetime = field(default_factory=datetime.now)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveLearningConfig:
    """Configuration for adaptive learning."""
    
    # Learning settings
    enable_learning: bool = True
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.7
    min_events_for_learning: int = 10
    
    # Pattern recognition
    enable_pattern_recognition: bool = True
    pattern_min_frequency: int = 3
    pattern_confidence_threshold: float = 0.6
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_window_size: int = 100
    performance_degradation_threshold: float = 0.1
    
    # Feedback integration
    enable_feedback_learning: bool = True
    feedback_weight: float = 0.3
    feedback_decay_rate: float = 0.95
    
    # Model updates
    enable_model_updates: bool = True
    model_update_frequency: int = 24  # hours
    model_validation_threshold: float = 0.8


class AdaptiveLearningSystem:
    """Adaptive learning system for continuous improvement."""
    
    def __init__(self, 
                 knowledge_base: Optional[KnowledgeBase] = None,
                 config: Optional[AdaptiveLearningConfig] = None):
        """Initialize the adaptive learning system."""
        self.config = config or AdaptiveLearningConfig()
        self.knowledge_base = knowledge_base or KnowledgeBase()
        
        # Learning state
        self.learning_events: List[LearningEvent] = []
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.feedback_history: List[Dict[str, Any]] = []
        
        # Learning models
        self.performance_model = None
        self.success_prediction_model = None
        self.recommendation_effectiveness_model = None
        
        # Initialize learning components
        self._initialize_learning_models()
        
        # Start learning tasks
        if self.config.enable_learning:
            self._start_learning_tasks()
        
        logger.info("AdaptiveLearningSystem initialized")
    
    def _initialize_learning_models(self):
        """Initialize learning models."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available - using basic learning methods")
            return
        
        try:
            # Performance prediction model
            self.performance_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Success prediction model
            self.success_prediction_model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            )
            
            # Recommendation effectiveness model
            self.recommendation_effectiveness_model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            
            logger.info("Learning models initialized")
        except Exception as e:
            logger.error(f"Error initializing learning models: {e}")
    
    def _start_learning_tasks(self):
        """Start background learning tasks."""
        # Pattern recognition task
        asyncio.create_task(self._pattern_recognition_task())
        
        # Performance monitoring task
        if self.config.enable_performance_monitoring:
            asyncio.create_task(self._performance_monitoring_task())
        
        # Model update task
        if self.config.enable_model_updates:
            asyncio.create_task(self._model_update_task())
    
    async def record_learning_event(self, 
                                  event_type: LearningEventType,
                                  data: Dict[str, Any],
                                  priority: LearningPriority = LearningPriority.MEDIUM) -> str:
        """Record a learning event."""
        event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"
        
        event = LearningEvent(
            id=event_id,
            type=event_type,
            data=data,
            priority=priority
        )
        
        self.learning_events.append(event)
        
        # Process high-priority events immediately
        if priority in [LearningPriority.HIGH, LearningPriority.CRITICAL]:
            await self._process_learning_event(event)
        
        logger.info(f"Recorded learning event: {event_type.value}")
        return event_id
    
    async def _process_learning_event(self, event: LearningEvent):
        """Process a learning event."""
        try:
            if event.type == LearningEventType.ANALYSIS_COMPLETED:
                await self._learn_from_analysis_completion(event)
            elif event.type == LearningEventType.USER_FEEDBACK:
                await self._learn_from_user_feedback(event)
            elif event.type == LearningEventType.INSIGHT_GENERATED:
                await self._learn_from_insight_generation(event)
            elif event.type == LearningEventType.RECOMMENDATION_APPLIED:
                await self._learn_from_recommendation_application(event)
            elif event.type == LearningEventType.PREDICTION_ACCURATE:
                await self._learn_from_accurate_prediction(event)
            elif event.type == LearningEventType.PREDICTION_INACCURATE:
                await self._learn_from_inaccurate_prediction(event)
            elif event.type == LearningEventType.ERROR_OCCURRED:
                await self._learn_from_error(event)
            elif event.type == LearningEventType.PERFORMANCE_DEGRADED:
                await self._learn_from_performance_degradation(event)
            elif event.type == LearningEventType.PERFORMANCE_IMPROVED:
                await self._learn_from_performance_improvement(event)
            elif event.type == LearningEventType.CONFIGURATION_CHANGED:
                await self._learn_from_configuration_change(event)
            
            event.processed = True
            event.learning_impact = self._calculate_learning_impact(event)
            
        except Exception as e:
            logger.error(f"Error processing learning event {event.id}: {e}")
    
    async def _learn_from_analysis_completion(self, event: LearningEvent):
        """Learn from analysis completion events."""
        data = event.data
        analysis_result = data.get("analysis_result")
        
        if not analysis_result:
            return
        
        # Extract performance metrics
        performance_metrics = {
            "duration": analysis_result.total_duration,
            "success": analysis_result.success,
            "error_count": len(analysis_result.errors),
            "warning_count": len(analysis_result.warnings),
            "phase_count": len(analysis_result.phase_results)
        }
        
        # Record performance history
        self.performance_history.append({
            "timestamp": event.timestamp,
            "metrics": performance_metrics,
            "analysis_id": getattr(analysis_result, 'id', 'unknown')
        })
        
        # Learn patterns
        await self._learn_analysis_patterns(analysis_result, performance_metrics)
        
        # Update knowledge base
        if self.knowledge_base:
            await self.knowledge_base.learn_from_analysis(
                analysis_result,
                data.get("insights", []),
                data.get("recommendations", [])
            )
    
    async def _learn_from_user_feedback(self, event: LearningEvent):
        """Learn from user feedback events."""
        data = event.data
        feedback = data.get("feedback", {})
        
        # Record feedback
        self.feedback_history.append({
            "timestamp": event.timestamp,
            "feedback": feedback,
            "analysis_id": data.get("analysis_id"),
            "user_id": data.get("user_id")
        })
        
        # Learn from feedback patterns
        await self._learn_feedback_patterns(feedback)
        
        # Update recommendation effectiveness
        if "recommendation_id" in data:
            await self._update_recommendation_effectiveness(
                data["recommendation_id"],
                feedback.get("rating", 0),
                feedback.get("helpful", False)
            )
    
    async def _learn_from_insight_generation(self, event: LearningEvent):
        """Learn from insight generation events."""
        data = event.data
        insights = data.get("insights", [])
        
        # Learn insight patterns
        for insight in insights:
            await self._learn_insight_patterns(insight)
    
    async def _learn_from_recommendation_application(self, event: LearningEvent):
        """Learn from recommendation application events."""
        data = event.data
        recommendation = data.get("recommendation")
        outcome = data.get("outcome", {})
        
        if recommendation:
            await self._update_recommendation_effectiveness(
                recommendation.get("id", "unknown"),
                outcome.get("rating", 0),
                outcome.get("successful", False)
            )
    
    async def _learn_from_accurate_prediction(self, event: LearningEvent):
        """Learn from accurate predictions."""
        data = event.data
        prediction = data.get("prediction")
        
        if prediction:
            # Strengthen prediction patterns
            await self._strengthen_prediction_patterns(prediction, True)
    
    async def _learn_from_inaccurate_prediction(self, event: LearningEvent):
        """Learn from inaccurate predictions."""
        data = event.data
        prediction = data.get("prediction")
        
        if prediction:
            # Weaken prediction patterns
            await self._weaken_prediction_patterns(prediction, False)
    
    async def _learn_from_error(self, event: LearningEvent):
        """Learn from error events."""
        data = event.data
        error = data.get("error", {})
        
        # Learn error patterns
        await self._learn_error_patterns(error)
        
        # Update knowledge base with error information
        if self.knowledge_base:
            await self.knowledge_base.add_entry(
                title=f"Error Pattern: {error.get('type', 'Unknown')}",
                content=error.get("message", "No message"),
                category="error_patterns",
                knowledge_type=KnowledgeType.ERROR_PATTERN,
                tags=["error", error.get("type", "unknown")],
                source="system"
            )
    
    async def _learn_from_performance_degradation(self, event: LearningEvent):
        """Learn from performance degradation events."""
        data = event.data
        degradation_metrics = data.get("metrics", {})
        
        # Learn performance degradation patterns
        await self._learn_performance_degradation_patterns(degradation_metrics)
    
    async def _learn_from_performance_improvement(self, event: LearningEvent):
        """Learn from performance improvement events."""
        data = event.data
        improvement_metrics = data.get("metrics", {})
        
        # Learn performance improvement patterns
        await self._learn_performance_improvement_patterns(improvement_metrics)
    
    async def _learn_from_configuration_change(self, event: LearningEvent):
        """Learn from configuration change events."""
        data = event.data
        old_config = data.get("old_config", {})
        new_config = data.get("new_config", {})
        
        # Learn configuration impact patterns
        await self._learn_configuration_impact_patterns(old_config, new_config)
    
    async def _learn_analysis_patterns(self, analysis_result: OrchestrationResult, metrics: Dict[str, Any]):
        """Learn patterns from analysis results."""
        pattern_id = f"analysis_pattern_{getattr(analysis_result, 'id', 'unknown')}"
        
        # Extract conditions
        conditions = {
            "phase_count": len(analysis_result.phase_results),
            "parallel_enabled": getattr(analysis_result, 'parallel_enabled', False),
            "memory_limit": getattr(analysis_result, 'memory_limit', 0),
            "timeout": getattr(analysis_result, 'timeout', 0)
        }
        
        # Extract outcomes
        outcomes = {
            "success": analysis_result.success,
            "duration": analysis_result.total_duration,
            "error_count": len(analysis_result.errors),
            "warning_count": len(analysis_result.warnings)
        }
        
        # Update or create pattern
        if pattern_id in self.learning_patterns:
            pattern = self.learning_patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_updated = datetime.now()
            pattern.examples.append({
                "conditions": conditions,
                "outcomes": outcomes,
                "timestamp": datetime.now().isoformat()
            })
        else:
            pattern = LearningPattern(
                id=pattern_id,
                pattern_type="analysis",
                conditions=conditions,
                outcomes=outcomes,
                confidence=0.8,
                frequency=1,
                success_rate=1.0 if analysis_result.success else 0.0,
                examples=[{
                    "conditions": conditions,
                    "outcomes": outcomes,
                    "timestamp": datetime.now().isoformat()
                }]
            )
            self.learning_patterns[pattern_id] = pattern
    
    async def _learn_feedback_patterns(self, feedback: Dict[str, Any]):
        """Learn patterns from user feedback."""
        # Extract feedback patterns
        feedback_type = feedback.get("type", "general")
        rating = feedback.get("rating", 0)
        comments = feedback.get("comments", "")
        
        # Learn from positive feedback
        if rating > 3:
            await self._learn_positive_feedback_patterns(feedback)
        else:
            await self._learn_negative_feedback_patterns(feedback)
    
    async def _learn_positive_feedback_patterns(self, feedback: Dict[str, Any]):
        """Learn from positive feedback patterns."""
        # This would identify what led to positive feedback
        # and strengthen those patterns
        pass
    
    async def _learn_negative_feedback_patterns(self, feedback: Dict[str, Any]):
        """Learn from negative feedback patterns."""
        # This would identify what led to negative feedback
        # and weaken those patterns
        pass
    
    async def _learn_insight_patterns(self, insight: Insight):
        """Learn patterns from insight generation."""
        pattern_id = f"insight_pattern_{insight.type.value}"
        
        conditions = {
            "type": insight.type.value,
            "severity": insight.severity.value,
            "confidence": insight.confidence,
            "tags": insight.tags
        }
        
        outcomes = {
            "title": insight.title,
            "description": insight.description,
            "recommendations": insight.recommendations
        }
        
        if pattern_id in self.learning_patterns:
            pattern = self.learning_patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_updated = datetime.now()
        else:
            pattern = LearningPattern(
                id=pattern_id,
                pattern_type="insight",
                conditions=conditions,
                outcomes=outcomes,
                confidence=insight.confidence,
                frequency=1,
                success_rate=0.8  # Default success rate
            )
            self.learning_patterns[pattern_id] = pattern
    
    async def _learn_error_patterns(self, error: Dict[str, Any]):
        """Learn patterns from errors."""
        error_type = error.get("type", "unknown")
        pattern_id = f"error_pattern_{error_type}"
        
        conditions = {
            "error_type": error_type,
            "error_message": error.get("message", ""),
            "context": error.get("context", {})
        }
        
        outcomes = {
            "resolved": error.get("resolved", False),
            "resolution_time": error.get("resolution_time", 0),
            "resolution_method": error.get("resolution_method", "")
        }
        
        if pattern_id in self.learning_patterns:
            pattern = self.learning_patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_updated = datetime.now()
        else:
            pattern = LearningPattern(
                id=pattern_id,
                pattern_type="error",
                conditions=conditions,
                outcomes=outcomes,
                confidence=0.7,
                frequency=1,
                success_rate=0.5  # Default success rate
            )
            self.learning_patterns[pattern_id] = pattern
    
    async def _learn_performance_degradation_patterns(self, metrics: Dict[str, Any]):
        """Learn patterns from performance degradation."""
        pattern_id = "performance_degradation"
        
        conditions = {
            "degradation_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        outcomes = {
            "degradation_severity": metrics.get("severity", "medium"),
            "affected_components": metrics.get("components", [])
        }
        
        if pattern_id in self.learning_patterns:
            pattern = self.learning_patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_updated = datetime.now()
        else:
            pattern = LearningPattern(
                id=pattern_id,
                pattern_type="performance_degradation",
                conditions=conditions,
                outcomes=outcomes,
                confidence=0.8,
                frequency=1,
                success_rate=0.3  # Low success rate for degradation
            )
            self.learning_patterns[pattern_id] = pattern
    
    async def _learn_performance_improvement_patterns(self, metrics: Dict[str, Any]):
        """Learn patterns from performance improvement."""
        pattern_id = "performance_improvement"
        
        conditions = {
            "improvement_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        outcomes = {
            "improvement_factor": metrics.get("improvement_factor", 1.0),
            "improved_components": metrics.get("components", [])
        }
        
        if pattern_id in self.learning_patterns:
            pattern = self.learning_patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_updated = datetime.now()
        else:
            pattern = LearningPattern(
                id=pattern_id,
                pattern_type="performance_improvement",
                conditions=conditions,
                outcomes=outcomes,
                confidence=0.8,
                frequency=1,
                success_rate=0.9  # High success rate for improvement
            )
            self.learning_patterns[pattern_id] = pattern
    
    async def _learn_configuration_impact_patterns(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Learn patterns from configuration changes."""
        pattern_id = "configuration_impact"
        
        conditions = {
            "old_config": old_config,
            "new_config": new_config,
            "timestamp": datetime.now().isoformat()
        }
        
        outcomes = {
            "changes": self._identify_config_changes(old_config, new_config),
            "impact_expected": True
        }
        
        if pattern_id in self.learning_patterns:
            pattern = self.learning_patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_updated = datetime.now()
        else:
            pattern = LearningPattern(
                id=pattern_id,
                pattern_type="configuration_impact",
                conditions=conditions,
                outcomes=outcomes,
                confidence=0.7,
                frequency=1,
                success_rate=0.6
            )
            self.learning_patterns[pattern_id] = pattern
    
    def _identify_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[str]:
        """Identify configuration changes."""
        changes = []
        
        for key in set(old_config.keys()) | set(new_config.keys()):
            old_value = old_config.get(key)
            new_value = new_config.get(key)
            
            if old_value != new_value:
                changes.append(f"{key}: {old_value} -> {new_value}")
        
        return changes
    
    async def _update_recommendation_effectiveness(self, recommendation_id: str, rating: float, successful: bool):
        """Update recommendation effectiveness based on feedback."""
        # This would update the effectiveness of recommendations
        # based on user feedback and outcomes
        pass
    
    async def _strengthen_prediction_patterns(self, prediction: Prediction, accurate: bool):
        """Strengthen prediction patterns for accurate predictions."""
        # This would strengthen patterns that led to accurate predictions
        pass
    
    async def _weaken_prediction_patterns(self, prediction: Prediction, accurate: bool):
        """Weaken prediction patterns for inaccurate predictions."""
        # This would weaken patterns that led to inaccurate predictions
        pass
    
    async def _pattern_recognition_task(self):
        """Background task for pattern recognition."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._recognize_patterns()
            except Exception as e:
                logger.error(f"Error in pattern recognition task: {e}")
    
    async def _recognize_patterns(self):
        """Recognize patterns from learning events."""
        if len(self.learning_events) < self.config.min_events_for_learning:
            return
        
        # Process unprocessed events
        unprocessed_events = [e for e in self.learning_events if not e.processed]
        
        for event in unprocessed_events:
            await self._process_learning_event(event)
        
        # Clean up old events
        cutoff_time = datetime.now() - timedelta(days=30)
        self.learning_events = [e for e in self.learning_events if e.timestamp > cutoff_time]
    
    async def _performance_monitoring_task(self):
        """Background task for performance monitoring."""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                await self._monitor_performance()
            except Exception as e:
                logger.error(f"Error in performance monitoring task: {e}")
    
    async def _monitor_performance(self):
        """Monitor system performance and detect degradation."""
        if len(self.performance_history) < self.config.performance_window_size:
            return
        
        # Get recent performance data
        recent_performance = self.performance_history[-self.config.performance_window_size:]
        
        # Calculate performance trends
        durations = [p["metrics"]["duration"] for p in recent_performance]
        success_rates = [1.0 if p["metrics"]["success"] else 0.0 for p in recent_performance]
        
        # Check for performance degradation
        if len(durations) > 10:
            recent_avg_duration = statistics.mean(durations[-10:])
            older_avg_duration = statistics.mean(durations[:-10])
            
            if recent_avg_duration > older_avg_duration * (1 + self.config.performance_degradation_threshold):
                await self.record_learning_event(
                    LearningEventType.PERFORMANCE_DEGRADED,
                    {
                        "metrics": {
                            "duration_increase": (recent_avg_duration - older_avg_duration) / older_avg_duration,
                            "recent_avg_duration": recent_avg_duration,
                            "older_avg_duration": older_avg_duration
                        }
                    },
                    LearningPriority.HIGH
                )
        
        # Check for performance improvement
        if len(success_rates) > 10:
            recent_success_rate = statistics.mean(success_rates[-10:])
            older_success_rate = statistics.mean(success_rates[:-10])
            
            if recent_success_rate > older_success_rate + 0.1:  # 10% improvement
                await self.record_learning_event(
                    LearningEventType.PERFORMANCE_IMPROVED,
                    {
                        "metrics": {
                            "success_rate_improvement": recent_success_rate - older_success_rate,
                            "recent_success_rate": recent_success_rate,
                            "older_success_rate": older_success_rate
                        }
                    },
                    LearningPriority.MEDIUM
                )
    
    async def _model_update_task(self):
        """Background task for model updates."""
        while True:
            try:
                await asyncio.sleep(self.config.model_update_frequency * 3600)  # Run every N hours
                await self._update_learning_models()
            except Exception as e:
                logger.error(f"Error in model update task: {e}")
    
    async def _update_learning_models(self):
        """Update learning models based on new data."""
        if not SKLEARN_AVAILABLE:
            return
        
        # Update performance model
        if self.performance_model and len(self.performance_history) > 50:
            await self._update_performance_model()
        
        # Update success prediction model
        if self.success_prediction_model and len(self.learning_events) > 50:
            await self._update_success_prediction_model()
        
        # Update recommendation effectiveness model
        if self.recommendation_effectiveness_model and len(self.feedback_history) > 20:
            await self._update_recommendation_effectiveness_model()
    
    async def _update_performance_model(self):
        """Update the performance prediction model."""
        # This would retrain the performance model with new data
        pass
    
    async def _update_success_prediction_model(self):
        """Update the success prediction model."""
        # This would retrain the success prediction model with new data
        pass
    
    async def _update_recommendation_effectiveness_model(self):
        """Update the recommendation effectiveness model."""
        # This would retrain the recommendation effectiveness model with new data
        pass
    
    def _calculate_learning_impact(self, event: LearningEvent) -> float:
        """Calculate the learning impact of an event."""
        # Simple impact calculation based on event type and priority
        impact_weights = {
            LearningEventType.ANALYSIS_COMPLETED: 0.3,
            LearningEventType.USER_FEEDBACK: 0.5,
            LearningEventType.INSIGHT_GENERATED: 0.2,
            LearningEventType.RECOMMENDATION_APPLIED: 0.4,
            LearningEventType.PREDICTION_ACCURATE: 0.3,
            LearningEventType.PREDICTION_INACCURATE: 0.4,
            LearningEventType.ERROR_OCCURRED: 0.6,
            LearningEventType.PERFORMANCE_DEGRADED: 0.7,
            LearningEventType.PERFORMANCE_IMPROVED: 0.5,
            LearningEventType.CONFIGURATION_CHANGED: 0.3
        }
        
        priority_weights = {
            LearningPriority.LOW: 0.1,
            LearningPriority.MEDIUM: 0.3,
            LearningPriority.HIGH: 0.6,
            LearningPriority.CRITICAL: 1.0
        }
        
        base_impact = impact_weights.get(event.type, 0.1)
        priority_impact = priority_weights.get(event.priority, 0.1)
        
        return base_impact * priority_impact
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning system."""
        # Analyze learning events
        event_counts = {}
        for event in self.learning_events:
            event_counts[event.type.value] = event_counts.get(event.type.value, 0) + 1
        
        # Analyze patterns
        pattern_counts = {}
        for pattern in self.learning_patterns.values():
            pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
        
        # Calculate learning metrics
        total_events = len(self.learning_events)
        processed_events = sum(1 for e in self.learning_events if e.processed)
        avg_learning_impact = statistics.mean([e.learning_impact for e in self.learning_events]) if self.learning_events else 0
        
        return {
            "total_events": total_events,
            "processed_events": processed_events,
            "event_types": event_counts,
            "pattern_types": pattern_counts,
            "total_patterns": len(self.learning_patterns),
            "average_learning_impact": avg_learning_impact,
            "performance_history_size": len(self.performance_history),
            "feedback_history_size": len(self.feedback_history),
            "learning_enabled": self.config.enable_learning,
            "models_available": SKLEARN_AVAILABLE
        }
    
    def export_learning_data(self, format: str = "json", file_path: Optional[Union[str, Path]] = None) -> Union[str, Path]:
        """Export learning data to various formats."""
        if format == "json":
            data = {
                "learning_events": [event.__dict__ for event in self.learning_events],
                "learning_patterns": [pattern.__dict__ for pattern in self.learning_patterns.values()],
                "performance_history": self.performance_history,
                "feedback_history": self.feedback_history,
                "exported_at": datetime.now().isoformat()
            }
            json_str = json.dumps(data, indent=2, default=str)
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(json_str)
                return Path(file_path)
            else:
                return json_str
        
        elif format == "markdown":
            md_content = "# Learning Data Export\n\n"
            md_content += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Learning events summary
            md_content += f"## Learning Events ({len(self.learning_events)} total)\n\n"
            event_counts = {}
            for event in self.learning_events:
                event_counts[event.type.value] = event_counts.get(event.type.value, 0) + 1
            
            for event_type, count in event_counts.items():
                md_content += f"- **{event_type}**: {count}\n"
            md_content += "\n"
            
            # Learning patterns summary
            md_content += f"## Learning Patterns ({len(self.learning_patterns)} total)\n\n"
            for pattern in list(self.learning_patterns.values())[:10]:  # Show first 10
                md_content += f"### {pattern.pattern_type.title()}\n"
                md_content += f"**Frequency**: {pattern.frequency}\n"
                md_content += f"**Confidence**: {pattern.confidence:.2f}\n"
                md_content += f"**Success Rate**: {pattern.success_rate:.2f}\n"
                md_content += f"**Last Updated**: {pattern.last_updated.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(md_content)
                return Path(file_path)
            else:
                return md_content
        
        else:
            raise ValueError(f"Unsupported format: {format}")