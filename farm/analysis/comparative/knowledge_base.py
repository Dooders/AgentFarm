"""
Knowledge base and learning system.

This module provides a comprehensive knowledge base for storing, retrieving,
and learning from analysis patterns, insights, and user interactions.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from enum import Enum
import hashlib
import statistics
import math

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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from farm.utils.logging import get_logger
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult
from farm.analysis.comparative.automated_insights import Insight, InsightType, InsightSeverity
from farm.analysis.comparative.smart_recommendations import Recommendation, RecommendationType, RecommendationPriority

logger = get_logger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge entries."""
    ANALYSIS_PATTERN = "analysis_pattern"
    INSIGHT_PATTERN = "insight_pattern"
    RECOMMENDATION_PATTERN = "recommendation_pattern"
    USER_INTERACTION = "user_interaction"
    ERROR_PATTERN = "error_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    BEST_PRACTICE = "best_practice"
    LESSON_LEARNED = "lesson_learned"
    FAQ = "faq"
    TUTORIAL = "tutorial"


class LearningLevel(Enum):
    """Learning levels for knowledge entries."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class KnowledgeEntry:
    """A knowledge base entry."""
    
    id: str
    type: KnowledgeType
    title: str
    content: str
    category: str
    tags: List[str] = field(default_factory=list)
    learning_level: LearningLevel = LearningLevel.INTERMEDIATE
    confidence: float = 1.0
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_entries: List[str] = field(default_factory=list)
    source: str = "system"


@dataclass
class LearningPattern:
    """A learned pattern from analysis data."""
    
    id: str
    pattern_type: str
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    confidence: float
    frequency: int
    last_seen: datetime = field(default_factory=datetime.now)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeBaseConfig:
    """Configuration for the knowledge base."""
    
    # Storage settings
    storage_path: Union[str, Path] = "knowledge_base"
    max_entries: int = 10000
    auto_cleanup: bool = True
    cleanup_interval: int = 3600  # 1 hour
    
    # Learning settings
    enable_learning: bool = True
    learning_rate: float = 0.1
    min_confidence_threshold: float = 0.6
    pattern_min_frequency: int = 3
    
    # Search settings
    enable_semantic_search: bool = True
    max_search_results: int = 20
    similarity_threshold: float = 0.7
    
    # Caching settings
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour


class KnowledgeBase:
    """Knowledge base and learning system."""
    
    def __init__(self, config: Optional[KnowledgeBaseConfig] = None):
        """Initialize the knowledge base."""
        self.config = config or KnowledgeBaseConfig()
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Knowledge storage
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.patterns: Dict[str, LearningPattern] = {}
        self.cache: Dict[str, Any] = {}
        
        # Learning components
        self.vectorizer = None
        self.entry_vectors = None
        
        # Load existing knowledge
        self._load_knowledge()
        self._initialize_learning()
        
        # Start cleanup task
        if self.config.auto_cleanup:
            self._start_cleanup_task()
        
        logger.info("KnowledgeBase initialized")
    
    def _load_knowledge(self):
        """Load knowledge from storage."""
        # Load entries
        entries_file = self.storage_path / "entries.json"
        if entries_file.exists():
            try:
                with open(entries_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = KnowledgeEntry(**entry_data)
                        self.entries[entry.id] = entry
                logger.info(f"Loaded {len(self.entries)} knowledge entries")
            except Exception as e:
                logger.error(f"Error loading knowledge entries: {e}")
        
        # Load patterns
        patterns_file = self.storage_path / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for pattern_data in data:
                        pattern = LearningPattern(**pattern_data)
                        self.patterns[pattern.id] = pattern
                logger.info(f"Loaded {len(self.patterns)} learning patterns")
            except Exception as e:
                logger.error(f"Error loading learning patterns: {e}")
    
    def _save_knowledge(self):
        """Save knowledge to storage."""
        # Save entries
        entries_file = self.storage_path / "entries.json"
        try:
            data = [entry.__dict__ for entry in self.entries.values()]
            with open(entries_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving knowledge entries: {e}")
        
        # Save patterns
        patterns_file = self.storage_path / "patterns.json"
        try:
            data = [pattern.__dict__ for pattern in self.patterns.values()]
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving learning patterns: {e}")
    
    def _initialize_learning(self):
        """Initialize learning components."""
        if not SKLEARN_AVAILABLE or not self.config.enable_learning:
            return
        
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Build initial vectors if we have entries
            if self.entries:
                self._update_vectors()
            
            logger.info("Learning components initialized")
        except Exception as e:
            logger.error(f"Error initializing learning components: {e}")
    
    def _update_vectors(self):
        """Update vector representations of entries."""
        if not self.vectorizer or not self.entries:
            return
        
        try:
            texts = [f"{entry.title} {entry.content}" for entry in self.entries.values()]
            self.entry_vectors = self.vectorizer.fit_transform(texts)
            logger.info("Updated entry vectors")
        except Exception as e:
            logger.error(f"Error updating vectors: {e}")
    
    def _start_cleanup_task(self):
        """Start the cleanup task."""
        async def cleanup():
            while True:
                await asyncio.sleep(self.config.cleanup_interval)
                self._cleanup_expired_entries()
                self._cleanup_cache()
        
        asyncio.create_task(cleanup())
    
    def _cleanup_expired_entries(self):
        """Clean up expired entries."""
        current_time = datetime.now()
        expired_entries = []
        
        for entry_id, entry in self.entries.items():
            if entry.expires_at and current_time > entry.expires_at:
                expired_entries.append(entry_id)
        
        for entry_id in expired_entries:
            del self.entries[entry_id]
        
        if expired_entries:
            logger.info(f"Cleaned up {len(expired_entries)} expired entries")
            self._save_knowledge()
    
    def _cleanup_cache(self):
        """Clean up cache."""
        if len(self.cache) > self.config.cache_size:
            # Remove oldest entries
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1].get('timestamp', 0))
            items_to_remove = len(self.cache) - self.config.cache_size
            for key, _ in sorted_items[:items_to_remove]:
                del self.cache[key]
    
    async def add_entry(self, 
                       title: str, 
                       content: str, 
                       category: str,
                       knowledge_type: KnowledgeType = KnowledgeType.ANALYSIS_PATTERN,
                       tags: List[str] = None,
                       learning_level: LearningLevel = LearningLevel.INTERMEDIATE,
                       source: str = "user") -> str:
        """Add a new knowledge entry."""
        entry_id = self._generate_entry_id(title, content)
        
        entry = KnowledgeEntry(
            id=entry_id,
            type=knowledge_type,
            title=title,
            content=content,
            category=category,
            tags=tags or [],
            learning_level=learning_level,
            source=source
        )
        
        self.entries[entry_id] = entry
        
        # Update vectors if learning is enabled
        if self.config.enable_learning:
            self._update_vectors()
        
        # Save knowledge
        self._save_knowledge()
        
        logger.info(f"Added knowledge entry: {entry_id}")
        return entry_id
    
    def _generate_entry_id(self, title: str, content: str) -> str:
        """Generate a unique entry ID."""
        text = f"{title}_{content}"
        hash_obj = hashlib.md5(text.encode())
        return f"entry_{hash_obj.hexdigest()[:12]}"
    
    async def search_entries(self, 
                           query: str, 
                           knowledge_type: Optional[KnowledgeType] = None,
                           category: Optional[str] = None,
                           learning_level: Optional[LearningLevel] = None,
                           max_results: Optional[int] = None) -> List[KnowledgeEntry]:
        """Search knowledge entries."""
        # Check cache first
        cache_key = f"search_{hashlib.md5(query.encode()).hexdigest()}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_result['timestamp'] < self.config.cache_ttl:
                return cached_result['results']
        
        # Filter entries by criteria
        filtered_entries = list(self.entries.values())
        
        if knowledge_type:
            filtered_entries = [e for e in filtered_entries if e.type == knowledge_type]
        
        if category:
            filtered_entries = [e for e in filtered_entries if e.category == category]
        
        if learning_level:
            filtered_entries = [e for e in filtered_entries if e.learning_level == learning_level]
        
        # Perform search
        if self.config.enable_semantic_search and self.vectorizer and self.entry_vectors is not None:
            results = await self._semantic_search(query, filtered_entries)
        else:
            results = await self._keyword_search(query, filtered_entries)
        
        # Limit results
        max_results = max_results or self.config.max_search_results
        results = results[:max_results]
        
        # Cache results
        if self.config.enable_caching:
            self.cache[cache_key] = {
                'results': results,
                'timestamp': datetime.now().timestamp()
            }
        
        return results
    
    async def _semantic_search(self, query: str, entries: List[KnowledgeEntry]) -> List[KnowledgeEntry]:
        """Perform semantic search using vector similarity."""
        if not self.vectorizer or not self.entry_vectors is not None:
            return await self._keyword_search(query, entries)
        
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.entry_vectors).flatten()
            
            # Create entry-similarity pairs
            entry_similarities = []
            for i, entry in enumerate(entries):
                if i < len(similarities):
                    entry_similarities.append((entry, similarities[i]))
            
            # Filter by similarity threshold and sort
            filtered_pairs = [
                (entry, sim) for entry, sim in entry_similarities
                if sim >= self.config.similarity_threshold
            ]
            filtered_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return [entry for entry, _ in filtered_pairs]
        
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return await self._keyword_search(query, entries)
    
    async def _keyword_search(self, query: str, entries: List[KnowledgeEntry]) -> List[KnowledgeEntry]:
        """Perform keyword-based search."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_entries = []
        
        for entry in entries:
            score = 0
            
            # Title matching
            title_words = set(entry.title.lower().split())
            title_matches = len(query_words.intersection(title_words))
            score += title_matches * 3
            
            # Content matching
            content_words = set(entry.content.lower().split())
            content_matches = len(query_words.intersection(content_words))
            score += content_matches * 1
            
            # Tag matching
            tag_words = set(' '.join(entry.tags).lower().split())
            tag_matches = len(query_words.intersection(tag_words))
            score += tag_matches * 2
            
            if score > 0:
                scored_entries.append((entry, score))
        
        # Sort by score and return
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored_entries]
    
    async def learn_from_analysis(self, 
                                analysis_result: OrchestrationResult,
                                insights: List[Insight],
                                recommendations: List[Recommendation],
                                user_feedback: Optional[Dict[str, Any]] = None):
        """Learn from analysis results and patterns."""
        if not self.config.enable_learning:
            return
        
        # Learn analysis patterns
        await self._learn_analysis_patterns(analysis_result)
        
        # Learn insight patterns
        await self._learn_insight_patterns(insights)
        
        # Learn recommendation patterns
        await self._learn_recommendation_patterns(recommendations)
        
        # Learn from user feedback
        if user_feedback:
            await self._learn_from_feedback(user_feedback)
        
        # Save learned patterns
        self._save_knowledge()
    
    async def _learn_analysis_patterns(self, analysis_result: OrchestrationResult):
        """Learn patterns from analysis results."""
        pattern_id = f"analysis_{analysis_result.id if hasattr(analysis_result, 'id') else 'unknown'}"
        
        # Extract pattern conditions
        conditions = {
            "duration": analysis_result.total_duration,
            "phase_count": len(analysis_result.phase_results),
            "error_count": len(analysis_result.errors),
            "warning_count": len(analysis_result.warnings),
            "success": analysis_result.success
        }
        
        # Extract pattern outcomes
        outcomes = {
            "phases": [phase.phase_name for phase in analysis_result.phase_results],
            "errors": analysis_result.errors,
            "warnings": analysis_result.warnings
        }
        
        # Create or update pattern
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_seen = datetime.now()
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
                examples=[{
                    "conditions": conditions,
                    "outcomes": outcomes,
                    "timestamp": datetime.now().isoformat()
                }]
            )
            self.patterns[pattern_id] = pattern
    
    async def _learn_insight_patterns(self, insights: List[Insight]):
        """Learn patterns from insights."""
        for insight in insights:
            pattern_id = f"insight_{insight.type.value}_{insight.id}"
            
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
            
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
            else:
                pattern = LearningPattern(
                    id=pattern_id,
                    pattern_type="insight",
                    conditions=conditions,
                    outcomes=outcomes,
                    confidence=insight.confidence,
                    frequency=1
                )
                self.patterns[pattern_id] = pattern
    
    async def _learn_recommendation_patterns(self, recommendations: List[Recommendation]):
        """Learn patterns from recommendations."""
        for rec in recommendations:
            pattern_id = f"recommendation_{rec.type.value}_{rec.id}"
            
            conditions = {
                "type": rec.type.value,
                "priority": rec.priority.value,
                "confidence": rec.confidence,
                "context": rec.context
            }
            
            outcomes = {
                "title": rec.title,
                "description": rec.description,
                "recommendations": rec.recommendations
            }
            
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
            else:
                pattern = LearningPattern(
                    id=pattern_id,
                    pattern_type="recommendation",
                    conditions=conditions,
                    outcomes=outcomes,
                    confidence=rec.confidence,
                    frequency=1
                )
                self.patterns[pattern_id] = pattern
    
    async def _learn_from_feedback(self, feedback: Dict[str, Any]):
        """Learn from user feedback."""
        # Extract feedback patterns
        feedback_type = feedback.get("type", "general")
        rating = feedback.get("rating", 0)
        comments = feedback.get("comments", "")
        
        # Create feedback entry
        entry_id = await self.add_entry(
            title=f"User Feedback: {feedback_type}",
            content=comments,
            category="feedback",
            knowledge_type=KnowledgeType.USER_INTERACTION,
            tags=["feedback", feedback_type],
            source="user"
        )
        
        # Update related patterns based on feedback
        if "analysis_id" in feedback:
            analysis_id = feedback["analysis_id"]
            pattern_id = f"analysis_{analysis_id}"
            
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                # Update success rate based on rating
                if rating > 0:
                    pattern.metadata["user_rating"] = rating
                    pattern.metadata["feedback_count"] = pattern.metadata.get("feedback_count", 0) + 1
    
    async def get_recommendations(self, context: Dict[str, Any]) -> List[KnowledgeEntry]:
        """Get knowledge-based recommendations for a given context."""
        # Search for relevant entries
        query_parts = []
        
        if "analysis_type" in context:
            query_parts.append(context["analysis_type"])
        
        if "errors" in context and context["errors"]:
            query_parts.append("error resolution")
        
        if "performance" in context:
            query_parts.append("performance optimization")
        
        if "insights" in context:
            query_parts.append("insights analysis")
        
        query = " ".join(query_parts) if query_parts else "analysis help"
        
        # Search for relevant entries
        relevant_entries = await self.search_entries(
            query,
            max_results=10
        )
        
        # Filter by context relevance
        filtered_entries = []
        for entry in relevant_entries:
            if self._is_entry_relevant_to_context(entry, context):
                filtered_entries.append(entry)
        
        return filtered_entries
    
    def _is_entry_relevant_to_context(self, entry: KnowledgeEntry, context: Dict[str, Any]) -> bool:
        """Check if an entry is relevant to the given context."""
        # Simple relevance check based on tags and content
        entry_text = f"{entry.title} {entry.content} {' '.join(entry.tags)}".lower()
        
        # Check for context keywords
        context_keywords = []
        for key, value in context.items():
            if isinstance(value, str):
                context_keywords.append(value.lower())
            elif isinstance(value, list):
                context_keywords.extend([str(v).lower() for v in value])
        
        # Check if any context keyword appears in entry
        for keyword in context_keywords:
            if keyword in entry_text:
                return True
        
        return False
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning system."""
        if not self.patterns:
            return {"message": "No learning data available"}
        
        # Analyze patterns
        pattern_types = {}
        pattern_frequencies = []
        pattern_confidences = []
        
        for pattern in self.patterns.values():
            pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
            pattern_frequencies.append(pattern.frequency)
            pattern_confidences.append(pattern.confidence)
        
        # Calculate statistics
        avg_frequency = statistics.mean(pattern_frequencies) if pattern_frequencies else 0
        avg_confidence = statistics.mean(pattern_confidences) if pattern_confidences else 0
        
        # Find most common patterns
        most_common = sorted(self.patterns.values(), key=lambda x: x.frequency, reverse=True)[:5]
        
        return {
            "total_patterns": len(self.patterns),
            "pattern_types": pattern_types,
            "average_frequency": avg_frequency,
            "average_confidence": avg_confidence,
            "most_common_patterns": [
                {
                    "id": p.id,
                    "type": p.pattern_type,
                    "frequency": p.frequency,
                    "confidence": p.confidence
                }
                for p in most_common
            ]
        }
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        # Count entries by type
        type_counts = {}
        category_counts = {}
        level_counts = {}
        
        for entry in self.entries.values():
            type_counts[entry.type.value] = type_counts.get(entry.type.value, 0) + 1
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
            level_counts[entry.learning_level.value] = level_counts.get(entry.learning_level.value, 0) + 1
        
        return {
            "total_entries": len(self.entries),
            "total_patterns": len(self.patterns),
            "entries_by_type": type_counts,
            "entries_by_category": category_counts,
            "entries_by_level": level_counts,
            "cache_size": len(self.cache),
            "learning_enabled": self.config.enable_learning,
            "semantic_search_enabled": self.config.enable_semantic_search
        }
    
    async def export_knowledge(self, format: str = "json", file_path: Optional[Union[str, Path]] = None) -> Union[str, Path]:
        """Export knowledge base to various formats."""
        if format == "json":
            data = {
                "entries": [entry.__dict__ for entry in self.entries.values()],
                "patterns": [pattern.__dict__ for pattern in self.patterns.values()],
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
            md_content = "# Knowledge Base Export\n\n"
            md_content += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md_content += f"Total entries: {len(self.entries)}\n"
            md_content += f"Total patterns: {len(self.patterns)}\n\n"
            
            # Export entries by category
            categories = set(entry.category for entry in self.entries.values())
            for category in sorted(categories):
                category_entries = [e for e in self.entries.values() if e.category == category]
                md_content += f"## {category.title()} ({len(category_entries)} entries)\n\n"
                
                for entry in category_entries:
                    md_content += f"### {entry.title}\n"
                    md_content += f"**Type**: {entry.type.value}\n"
                    md_content += f"**Level**: {entry.learning_level.value}\n"
                    md_content += f"**Tags**: {', '.join(entry.tags)}\n\n"
                    md_content += f"{entry.content}\n\n"
                    md_content += "---\n\n"
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(md_content)
                return Path(file_path)
            else:
                return md_content
        
        else:
            raise ValueError(f"Unsupported format: {format}")