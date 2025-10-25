"""
Tests for Knowledge Base and Learning System.

This module contains comprehensive tests for the knowledge base system
that stores, retrieves, and learns from analysis patterns and user interactions.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import shutil
import json

from farm.analysis.comparative.knowledge_base import (
    KnowledgeBase,
    KnowledgeType,
    LearningLevel,
    KnowledgeEntry,
    LearningPattern,
    KnowledgeBaseConfig
)
from farm.analysis.comparative.automated_insights import Insight, InsightType, InsightSeverity
from farm.analysis.comparative.smart_recommendations import Recommendation, RecommendationType, RecommendationPriority
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult


class TestKnowledgeBase:
    """Test cases for KnowledgeBase."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = KnowledgeBaseConfig(
            max_entries=1000,
            enable_semantic_search=True,
            enable_learning=True,
            learning_threshold=0.7,
            enable_auto_categorization=True,
            enable_usage_tracking=True
        )
        self.kb = KnowledgeBase(config=self.config)
        
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
        """Test knowledge base initialization."""
        assert self.kb.config == self.config
        assert len(self.kb.knowledge_entries) > 0  # Should have default entries
        assert self.kb.learning_patterns == []
        assert self.kb.usage_stats == {}
        assert self.kb.vectorizer is not None
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        kb = KnowledgeBase()
        assert kb.config is not None
        assert kb.config.enable_semantic_search is True
        assert kb.config.enable_learning is True
    
    def test_initialization_without_sklearn(self):
        """Test initialization without sklearn."""
        with patch('farm.analysis.comparative.knowledge_base.SKLEARN_AVAILABLE', False):
            kb = KnowledgeBase()
            assert kb.vectorizer is None
            assert kb.sklearn_available is False
    
    @pytest.mark.asyncio
    async def test_add_entry(self):
        """Test adding a knowledge entry."""
        entry_id = await self.kb.add_entry(
            title="Test Entry",
            content="Test content for knowledge base",
            category="test",
            knowledge_type=KnowledgeType.ANALYSIS_PATTERN,
            tags=["test", "example"],
            learning_level=LearningLevel.INTERMEDIATE,
            source="test_source"
        )
        
        assert entry_id is not None
        assert len(self.kb.knowledge_entries) > 0
        
        # Find the added entry
        entry = next((e for e in self.kb.knowledge_entries if e.id == entry_id), None)
        assert entry is not None
        assert entry.title == "Test Entry"
        assert entry.content == "Test content for knowledge base"
        assert entry.category == "test"
        assert entry.knowledge_type == KnowledgeType.ANALYSIS_PATTERN
        assert entry.tags == ["test", "example"]
        assert entry.learning_level == LearningLevel.INTERMEDIATE
        assert entry.source == "test_source"
    
    @pytest.mark.asyncio
    async def test_add_entry_without_sklearn(self):
        """Test adding entry without sklearn."""
        with patch('farm.analysis.comparative.knowledge_base.SKLEARN_AVAILABLE', False):
            kb = KnowledgeBase()
            entry_id = await kb.add_entry(
                title="Test Entry",
                content="Test content",
                category="test"
            )
            
            assert entry_id is not None
            assert len(kb.knowledge_entries) > 0
    
    @pytest.mark.asyncio
    async def test_search_entries_keyword(self):
        """Test searching entries by keyword."""
        # Add a test entry
        await self.kb.add_entry(
            title="Performance Optimization",
            content="How to optimize simulation performance",
            category="performance",
            tags=["optimization", "performance"]
        )
        
        # Search for entries
        results = await self.kb.search_entries(
            query="performance optimization",
            max_results=5
        )
        
        assert len(results) > 0
        assert any("performance" in entry.title.lower() or "performance" in entry.content.lower() for entry in results)
    
    @pytest.mark.asyncio
    async def test_search_entries_semantic(self):
        """Test searching entries with semantic search."""
        # Add a test entry
        await self.kb.add_entry(
            title="CPU Usage Analysis",
            content="Analyzing CPU usage patterns in simulations",
            category="analysis",
            tags=["cpu", "usage", "analysis"]
        )
        
        # Search with semantic search
        results = await self.kb.search_entries(
            query="computer processor utilization",
            max_results=5
        )
        
        assert len(results) > 0
        # Should find the CPU-related entry even with different terminology
    
    @pytest.mark.asyncio
    async def test_search_entries_without_sklearn(self):
        """Test searching entries without sklearn."""
        with patch('farm.analysis.comparative.knowledge_base.SKLEARN_AVAILABLE', False):
            kb = KnowledgeBase()
            
            # Add a test entry
            await kb.add_entry(
                title="Test Entry",
                content="Test content",
                category="test"
            )
            
            # Search for entries
            results = await kb.search_entries(query="test")
            
            assert len(results) > 0
            assert all("test" in entry.title.lower() or "test" in entry.content.lower() for entry in results)
    
    @pytest.mark.asyncio
    async def test_search_entries_by_category(self):
        """Test searching entries by category."""
        # Add entries with different categories
        await self.kb.add_entry(
            title="Performance Entry",
            content="Performance related content",
            category="performance"
        )
        await self.kb.add_entry(
            title="Quality Entry",
            content="Quality related content",
            category="quality"
        )
        
        # Search by category
        results = await self.kb.search_entries(
            query="test",
            category="performance"
        )
        
        assert len(results) > 0
        assert all(entry.category == "performance" for entry in results)
    
    @pytest.mark.asyncio
    async def test_search_entries_by_knowledge_type(self):
        """Test searching entries by knowledge type."""
        # Add entries with different knowledge types
        await self.kb.add_entry(
            title="Analysis Pattern",
            content="Pattern for analysis",
            category="patterns",
            knowledge_type=KnowledgeType.ANALYSIS_PATTERN
        )
        await self.kb.add_entry(
            title="Best Practice",
            content="Best practice content",
            category="practices",
            knowledge_type=KnowledgeType.BEST_PRACTICE
        )
        
        # Search by knowledge type
        results = await self.kb.search_entries(
            query="test",
            knowledge_type=KnowledgeType.ANALYSIS_PATTERN
        )
        
        assert len(results) > 0
        assert all(entry.knowledge_type == KnowledgeType.ANALYSIS_PATTERN for entry in results)
    
    @pytest.mark.asyncio
    async def test_search_entries_by_learning_level(self):
        """Test searching entries by learning level."""
        # Add entries with different learning levels
        await self.kb.add_entry(
            title="Beginner Entry",
            content="Beginner content",
            category="beginner",
            learning_level=LearningLevel.BEGINNER
        )
        await self.kb.add_entry(
            title="Expert Entry",
            content="Expert content",
            category="expert",
            learning_level=LearningLevel.EXPERT
        )
        
        # Search by learning level
        results = await self.kb.search_entries(
            query="test",
            learning_level=LearningLevel.BEGINNER
        )
        
        assert len(results) > 0
        assert all(entry.learning_level == LearningLevel.BEGINNER for entry in results)
    
    @pytest.mark.asyncio
    async def test_learn_from_analysis(self):
        """Test learning from analysis results."""
        await self.kb.learn_from_analysis(
            self.mock_analysis,
            self.mock_insights,
            self.mock_recommendations,
            {"user_rating": 5, "feedback": "Great analysis"}
        )
        
        # Should have learned patterns
        assert len(self.kb.learning_patterns) > 0
        
        # Should have updated usage stats
        assert len(self.kb.usage_stats) > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_analysis_without_sklearn(self):
        """Test learning from analysis without sklearn."""
        with patch('farm.analysis.comparative.knowledge_base.SKLEARN_AVAILABLE', False):
            kb = KnowledgeBase()
            
            await kb.learn_from_analysis(
                self.mock_analysis,
                self.mock_insights,
                self.mock_recommendations
            )
            
            # Should still learn patterns
            assert len(kb.learning_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_learn_from_user_interaction(self):
        """Test learning from user interactions."""
        await self.kb.learn_from_user_interaction(
            "user_1",
            "search",
            {"query": "performance optimization", "results_clicked": 3}
        )
        
        # Should have learned patterns
        assert len(self.kb.learning_patterns) > 0
        
        # Should have updated usage stats
        assert "user_1" in self.kb.usage_stats
    
    @pytest.mark.asyncio
    async def test_learn_from_feedback(self):
        """Test learning from user feedback."""
        await self.kb.learn_from_feedback(
            "entry_1",
            "positive",
            {"rating": 5, "comment": "Very helpful"}
        )
        
        # Should have learned patterns
        assert len(self.kb.learning_patterns) > 0
        
        # Should have updated usage stats
        assert "entry_1" in self.kb.usage_stats
    
    @pytest.mark.asyncio
    async def test_learn_from_error(self):
        """Test learning from errors."""
        await self.kb.learn_from_error(
            "analysis_error",
            {"error_type": "timeout", "context": "large_simulation"}
        )
        
        # Should have learned patterns
        assert len(self.kb.learning_patterns) > 0
    
    def test_get_entry_by_id(self):
        """Test getting an entry by ID."""
        # Add a test entry
        entry_id = asyncio.run(self.kb.add_entry(
            title="Test Entry",
            content="Test content",
            category="test"
        ))
        
        # Get the entry
        entry = self.kb.get_entry_by_id(entry_id)
        
        assert entry is not None
        assert entry.id == entry_id
        assert entry.title == "Test Entry"
    
    def test_get_entry_by_id_not_found(self):
        """Test getting an entry with non-existent ID."""
        entry = self.kb.get_entry_by_id("non_existent_id")
        assert entry is None
    
    def test_update_entry(self):
        """Test updating an entry."""
        # Add a test entry
        entry_id = asyncio.run(self.kb.add_entry(
            title="Original Title",
            content="Original content",
            category="test"
        ))
        
        # Update the entry
        success = self.kb.update_entry(
            entry_id,
            title="Updated Title",
            content="Updated content",
            tags=["updated", "test"]
        )
        
        assert success is True
        
        # Check the updated entry
        entry = self.kb.get_entry_by_id(entry_id)
        assert entry.title == "Updated Title"
        assert entry.content == "Updated content"
        assert entry.tags == ["updated", "test"]
    
    def test_update_entry_not_found(self):
        """Test updating a non-existent entry."""
        success = self.kb.update_entry(
            "non_existent_id",
            title="Updated Title"
        )
        
        assert success is False
    
    def test_delete_entry(self):
        """Test deleting an entry."""
        # Add a test entry
        entry_id = asyncio.run(self.kb.add_entry(
            title="Test Entry",
            content="Test content",
            category="test"
        ))
        
        # Delete the entry
        success = self.kb.delete_entry(entry_id)
        
        assert success is True
        
        # Check that the entry is deleted
        entry = self.kb.get_entry_by_id(entry_id)
        assert entry is None
    
    def test_delete_entry_not_found(self):
        """Test deleting a non-existent entry."""
        success = self.kb.delete_entry("non_existent_id")
        assert success is False
    
    def test_get_entries_by_category(self):
        """Test getting entries by category."""
        # Add entries with different categories
        asyncio.run(self.kb.add_entry(
            title="Performance Entry 1",
            content="Performance content 1",
            category="performance"
        ))
        asyncio.run(self.kb.add_entry(
            title="Performance Entry 2",
            content="Performance content 2",
            category="performance"
        ))
        asyncio.run(self.kb.add_entry(
            title="Quality Entry",
            content="Quality content",
            category="quality"
        ))
        
        # Get entries by category
        entries = self.kb.get_entries_by_category("performance")
        
        assert len(entries) >= 2
        assert all(entry.category == "performance" for entry in entries)
    
    def test_get_entries_by_knowledge_type(self):
        """Test getting entries by knowledge type."""
        # Add entries with different knowledge types
        asyncio.run(self.kb.add_entry(
            title="Pattern Entry",
            content="Pattern content",
            category="patterns",
            knowledge_type=KnowledgeType.ANALYSIS_PATTERN
        ))
        asyncio.run(self.kb.add_entry(
            title="Practice Entry",
            content="Practice content",
            category="practices",
            knowledge_type=KnowledgeType.BEST_PRACTICE
        ))
        
        # Get entries by knowledge type
        entries = self.kb.get_entries_by_knowledge_type(KnowledgeType.ANALYSIS_PATTERN)
        
        assert len(entries) >= 1
        assert all(entry.knowledge_type == KnowledgeType.ANALYSIS_PATTERN for entry in entries)
    
    def test_get_entries_by_learning_level(self):
        """Test getting entries by learning level."""
        # Add entries with different learning levels
        asyncio.run(self.kb.add_entry(
            title="Beginner Entry",
            content="Beginner content",
            category="beginner",
            learning_level=LearningLevel.BEGINNER
        ))
        asyncio.run(self.kb.add_entry(
            title="Expert Entry",
            content="Expert content",
            category="expert",
            learning_level=LearningLevel.EXPERT
        ))
        
        # Get entries by learning level
        entries = self.kb.get_entries_by_learning_level(LearningLevel.BEGINNER)
        
        assert len(entries) >= 1
        assert all(entry.learning_level == LearningLevel.BEGINNER for entry in entries)
    
    def test_get_most_used_entries(self):
        """Test getting most used entries."""
        # Add entries with different usage counts
        entry_id1 = asyncio.run(self.kb.add_entry(
            title="Popular Entry",
            content="Popular content",
            category="popular"
        ))
        entry_id2 = asyncio.run(self.kb.add_entry(
            title="Unpopular Entry",
            content="Unpopular content",
            category="unpopular"
        ))
        
        # Update usage counts
        entry1 = self.kb.get_entry_by_id(entry_id1)
        entry1.usage_count = 10
        entry2 = self.kb.get_entry_by_id(entry_id2)
        entry2.usage_count = 1
        
        # Get most used entries
        entries = self.kb.get_most_used_entries(limit=5)
        
        assert len(entries) > 0
        assert entries[0].usage_count >= entries[1].usage_count if len(entries) > 1 else True
    
    def test_get_recent_entries(self):
        """Test getting recent entries."""
        # Add entries at different times
        entry_id1 = asyncio.run(self.kb.add_entry(
            title="Recent Entry",
            content="Recent content",
            category="recent"
        ))
        entry_id2 = asyncio.run(self.kb.add_entry(
            title="Older Entry",
            content="Older content",
            category="older"
        ))
        
        # Get recent entries
        entries = self.kb.get_recent_entries(limit=5)
        
        assert len(entries) > 0
        assert entries[0].created_at >= entries[1].created_at if len(entries) > 1 else True
    
    def test_get_knowledge_base_stats(self):
        """Test getting knowledge base statistics."""
        # Add some entries
        asyncio.run(self.kb.add_entry(
            title="Test Entry 1",
            content="Test content 1",
            category="test1"
        ))
        asyncio.run(self.kb.add_entry(
            title="Test Entry 2",
            content="Test content 2",
            category="test2"
        ))
        
        # Get statistics
        stats = self.kb.get_knowledge_base_stats()
        
        assert "total_entries" in stats
        assert "categories" in stats
        assert "knowledge_types" in stats
        assert "learning_levels" in stats
        assert "total_tags" in stats
        assert stats["total_entries"] > 0
    
    def test_get_learning_patterns(self):
        """Test getting learning patterns."""
        # Add some learning patterns
        await self.kb.learn_from_analysis(
            self.mock_analysis,
            self.mock_insights,
            self.mock_recommendations
        )
        
        # Get learning patterns
        patterns = self.kb.get_learning_patterns()
        
        assert len(patterns) > 0
        assert all(isinstance(pattern, LearningPattern) for pattern in patterns)
    
    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        # Add some usage data
        self.kb.usage_stats = {
            "entry_1": {"views": 10, "clicks": 5},
            "entry_2": {"views": 5, "clicks": 2}
        }
        
        # Get usage stats
        stats = self.kb.get_usage_stats()
        
        assert len(stats) > 0
        assert "entry_1" in stats
        assert "entry_2" in stats
    
    def test_export_knowledge_base(self):
        """Test exporting knowledge base."""
        # Add some entries
        asyncio.run(self.kb.add_entry(
            title="Test Entry",
            content="Test content",
            category="test"
        ))
        
        # Export knowledge base
        export_data = self.kb.export_knowledge_base()
        
        assert "entries" in export_data
        assert "patterns" in export_data
        assert "usage_stats" in export_data
        assert len(export_data["entries"]) > 0
    
    def test_import_knowledge_base(self):
        """Test importing knowledge base."""
        # Create export data
        export_data = {
            "entries": [
                {
                    "id": "imported_1",
                    "title": "Imported Entry",
                    "content": "Imported content",
                    "category": "imported",
                    "knowledge_type": "analysis_pattern",
                    "tags": ["imported"],
                    "learning_level": "intermediate",
                    "source": "import",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "usage_count": 0,
                    "relevance_score": 0.5
                }
            ],
            "patterns": [],
            "usage_stats": {}
        }
        
        # Import knowledge base
        success = self.kb.import_knowledge_base(export_data)
        
        assert success is True
        
        # Check that the entry was imported
        entry = self.kb.get_entry_by_id("imported_1")
        assert entry is not None
        assert entry.title == "Imported Entry"
    
    def test_knowledge_entry_creation(self):
        """Test KnowledgeEntry creation."""
        entry = KnowledgeEntry(
            id="test_id",
            title="Test Title",
            content="Test content",
            category="test",
            knowledge_type=KnowledgeType.ANALYSIS_PATTERN,
            tags=["test", "example"],
            learning_level=LearningLevel.INTERMEDIATE,
            source="test_source",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            usage_count=5,
            relevance_score=0.8
        )
        
        assert entry.id == "test_id"
        assert entry.title == "Test Title"
        assert entry.content == "Test content"
        assert entry.category == "test"
        assert entry.knowledge_type == KnowledgeType.ANALYSIS_PATTERN
        assert entry.tags == ["test", "example"]
        assert entry.learning_level == LearningLevel.INTERMEDIATE
        assert entry.source == "test_source"
        assert entry.usage_count == 5
        assert entry.relevance_score == 0.8
    
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
    
    def test_knowledge_base_config_creation(self):
        """Test KnowledgeBaseConfig creation."""
        config = KnowledgeBaseConfig(
            max_entries=500,
            enable_semantic_search=False,
            enable_learning=False,
            learning_threshold=0.8,
            enable_auto_categorization=False,
            enable_usage_tracking=False
        )
        
        assert config.max_entries == 500
        assert config.enable_semantic_search is False
        assert config.enable_learning is False
        assert config.learning_threshold == 0.8
        assert config.enable_auto_categorization is False
        assert config.enable_usage_tracking is False
    
    def test_knowledge_type_enum(self):
        """Test KnowledgeType enum values."""
        assert KnowledgeType.ANALYSIS_PATTERN == "analysis_pattern"
        assert KnowledgeType.BEST_PRACTICE == "best_practice"
        assert KnowledgeType.TROUBLESHOOTING == "troubleshooting"
        assert KnowledgeType.OPTIMIZATION == "optimization"
        assert KnowledgeType.USER_INTERACTION == "user_interaction"
    
    def test_learning_level_enum(self):
        """Test LearningLevel enum values."""
        assert LearningLevel.BEGINNER == "beginner"
        assert LearningLevel.INTERMEDIATE == "intermediate"
        assert LearningLevel.ADVANCED == "advanced"
        assert LearningLevel.EXPERT == "expert"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in knowledge base operations."""
        # Test with invalid entry data
        with pytest.raises(ValueError):
            await self.kb.add_entry(
                title="",  # Empty title should raise error
                content="Test content",
                category="test"
            )
    
    def test_entry_validation(self):
        """Test entry validation."""
        # Test valid entry
        entry = KnowledgeEntry(
            id="test_id",
            title="Test Title",
            content="Test content",
            category="test",
            knowledge_type=KnowledgeType.ANALYSIS_PATTERN,
            tags=["test"],
            learning_level=LearningLevel.INTERMEDIATE,
            source="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            usage_count=0,
            relevance_score=0.5
        )
        
        assert self.kb._validate_entry(entry) is True
        
        # Test invalid entry (empty title)
        entry.title = ""
        assert self.kb._validate_entry(entry) is False