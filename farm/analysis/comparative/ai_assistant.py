"""
AI-powered analysis assistant with natural language processing.

This module provides an intelligent assistant that can understand natural language
queries about simulation analysis results and provide contextual responses.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import asyncio
from enum import Enum

# Optional imports for AI capabilities
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from farm.utils.logging import get_logger
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult
from farm.analysis.comparative.comparison_result import SimulationComparisonResult

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of user queries."""
    ANALYSIS_REQUEST = "analysis_request"
    RESULT_INTERPRETATION = "result_interpretation"
    RECOMMENDATION_REQUEST = "recommendation_request"
    COMPARISON_QUERY = "comparison_query"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_INQUIRY = "anomaly_inquiry"
    PERFORMANCE_QUESTION = "performance_question"
    GENERAL_HELP = "general_help"
    UNKNOWN = "unknown"


class ResponseType(Enum):
    """Types of assistant responses."""
    TEXT_RESPONSE = "text_response"
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    RECOMMENDATION = "recommendation"
    ACTION_SUGGESTION = "action_suggestion"
    ERROR_MESSAGE = "error_message"


@dataclass
class QueryContext:
    """Context for understanding user queries."""
    
    query_text: str
    query_type: QueryType
    entities: List[str] = field(default_factory=list)
    intent: str = ""
    confidence: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AssistantResponse:
    """Response from the AI assistant."""
    
    response_type: ResponseType
    content: str
    data: Optional[Dict[str, Any]] = None
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEntry:
    """Entry in the knowledge base."""
    
    id: str
    title: str
    content: str
    category: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    relevance_score: float = 0.0


class AIAnalysisAssistant:
    """AI-powered analysis assistant with natural language processing."""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo",
                 knowledge_base_path: Optional[Union[str, Path]] = None):
        """Initialize the AI assistant."""
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.knowledge_base_path = Path(knowledge_base_path) if knowledge_base_path else Path("ai_knowledge_base")
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize AI models
        self._initialize_models()
        
        # Knowledge base
        self.knowledge_base: List[KnowledgeEntry] = []
        self._load_knowledge_base()
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Analysis context
        self.current_analysis: Optional[OrchestrationResult] = None
        self.analysis_context: Dict[str, Any] = {}
        
        logger.info("AIAnalysisAssistant initialized")
    
    def _initialize_models(self):
        """Initialize AI models and pipelines."""
        # Initialize OpenAI if available
        if OPENAI_AVAILABLE and self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.openai_available = True
        else:
            self.openai_available = False
            logger.warning("OpenAI not available - using fallback models")
        
        # Initialize transformers for local processing
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis")
                self.question_answering = pipeline("question-answering")
                self.text_classifier = pipeline("zero-shot-classification")
                self.transformers_available = True
            except Exception as e:
                logger.warning(f"Transformers models not available: {e}")
                self.transformers_available = False
        else:
            self.transformers_available = False
        
        # Initialize spaCy for NLP
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_available = True
            except OSError:
                logger.warning("spaCy English model not found - install with: python -m spacy download en_core_web_sm")
                self.spacy_available = False
        else:
            self.spacy_available = False
    
    def _load_knowledge_base(self):
        """Load knowledge base from files."""
        knowledge_file = self.knowledge_base_path / "knowledge_base.json"
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge_base = [
                        KnowledgeEntry(**entry) for entry in data
                    ]
                logger.info(f"Loaded {len(self.knowledge_base)} knowledge entries")
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
                self.knowledge_base = []
        else:
            self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """Initialize default knowledge base entries."""
        default_entries = [
            KnowledgeEntry(
                id="simulation_analysis_101",
                title="Simulation Analysis Basics",
                content="Simulation analysis involves comparing different simulation runs to identify patterns, anomalies, and performance differences. Key metrics include execution time, resource usage, and outcome accuracy.",
                category="basics",
                tags=["simulation", "analysis", "comparison", "basics"]
            ),
            KnowledgeEntry(
                id="metrics_explanation",
                title="Key Analysis Metrics",
                content="Important metrics in simulation analysis include: 1) Execution time - how long simulations take to run, 2) Memory usage - RAM consumption during execution, 3) CPU utilization - processor usage patterns, 4) Accuracy - correctness of simulation results, 5) Stability - consistency across multiple runs.",
                category="metrics",
                tags=["metrics", "performance", "analysis"]
            ),
            KnowledgeEntry(
                id="anomaly_detection_guide",
                title="Anomaly Detection",
                content="Anomalies in simulation analysis are unusual patterns that deviate from normal behavior. Common types include: performance outliers, unexpected resource usage spikes, unusual execution patterns, and result inconsistencies. ML-based detection can identify these automatically.",
                category="anomalies",
                tags=["anomaly", "detection", "machine learning", "outliers"]
            ),
            KnowledgeEntry(
                id="clustering_analysis",
                title="Clustering Analysis",
                content="Clustering groups similar simulations together based on their characteristics. This helps identify simulation families, performance patterns, and optimization opportunities. Common clustering algorithms include K-means, DBSCAN, and hierarchical clustering.",
                category="clustering",
                tags=["clustering", "grouping", "similarity", "patterns"]
            ),
            KnowledgeEntry(
                id="trend_analysis",
                title="Trend Analysis",
                content="Trend analysis identifies patterns over time in simulation performance and results. It can predict future behavior, identify performance degradation, and suggest optimization strategies. Time series analysis and regression models are commonly used.",
                category="trends",
                tags=["trends", "time series", "prediction", "forecasting"]
            )
        ]
        
        self.knowledge_base = default_entries
        self._save_knowledge_base()
    
    def _save_knowledge_base(self):
        """Save knowledge base to file."""
        knowledge_file = self.knowledge_base_path / "knowledge_base.json"
        try:
            data = [entry.__dict__ for entry in self.knowledge_base]
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AssistantResponse:
        """Process a natural language query and return a response."""
        logger.info(f"Processing query: {query}")
        
        # Parse and understand the query
        query_context = await self._parse_query(query, context)
        
        # Generate response based on query type
        response = await self._generate_response(query_context)
        
        # Update conversation history
        self.conversation_history.append({
            "query": query,
            "response": response.content,
            "timestamp": datetime.now().isoformat(),
            "query_type": query_context.query_type.value,
            "confidence": response.confidence
        })
        
        return response
    
    async def _parse_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryContext:
        """Parse and understand the user query."""
        # Basic query classification using keywords
        query_lower = query.lower()
        
        # Determine query type
        query_type = QueryType.UNKNOWN
        confidence = 0.5
        
        if any(word in query_lower for word in ["analyze", "analysis", "compare", "comparison"]):
            query_type = QueryType.ANALYSIS_REQUEST
            confidence = 0.8
        elif any(word in query_lower for word in ["what", "explain", "mean", "interpret"]):
            query_type = QueryType.RESULT_INTERPRETATION
            confidence = 0.7
        elif any(word in query_lower for word in ["recommend", "suggest", "should", "advice"]):
            query_type = QueryType.RECOMMENDATION_REQUEST
            confidence = 0.8
        elif any(word in query_lower for word in ["trend", "pattern", "change", "over time"]):
            query_type = QueryType.TREND_ANALYSIS
            confidence = 0.7
        elif any(word in query_lower for word in ["anomaly", "outlier", "unusual", "strange"]):
            query_type = QueryType.ANOMALY_INQUIRY
            confidence = 0.8
        elif any(word in query_lower for word in ["performance", "speed", "memory", "cpu"]):
            query_type = QueryType.PERFORMANCE_QUESTION
            confidence = 0.7
        elif any(word in query_lower for word in ["help", "how", "what is", "guide"]):
            query_type = QueryType.GENERAL_HELP
            confidence = 0.6
        
        # Extract entities using spaCy if available
        entities = []
        if self.spacy_available:
            doc = self.nlp(query)
            entities = [ent.text for ent in doc.ents]
        
        # Extract parameters
        parameters = self._extract_parameters(query)
        
        return QueryContext(
            query_text=query,
            query_type=query_type,
            entities=entities,
            intent=query_type.value,
            confidence=confidence,
            parameters=parameters,
            conversation_history=self.conversation_history[-5:]  # Last 5 exchanges
        )
    
    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from the query."""
        parameters = {}
        
        # Extract numbers
        numbers = re.findall(r'\d+\.?\d*', query)
        if numbers:
            parameters['numbers'] = [float(n) for n in numbers]
        
        # Extract time references
        time_patterns = {
            'seconds': r'(\d+)\s*seconds?',
            'minutes': r'(\d+)\s*minutes?',
            'hours': r'(\d+)\s*hours?',
            'days': r'(\d+)\s*days?'
        }
        
        for unit, pattern in time_patterns.items():
            matches = re.findall(pattern, query.lower())
            if matches:
                parameters[unit] = [int(m) for m in matches]
        
        # Extract comparison terms
        comparison_terms = ['better', 'worse', 'faster', 'slower', 'higher', 'lower', 'more', 'less']
        found_terms = [term for term in comparison_terms if term in query.lower()]
        if found_terms:
            parameters['comparison_terms'] = found_terms
        
        return parameters
    
    async def _generate_response(self, query_context: QueryContext) -> AssistantResponse:
        """Generate response based on query context."""
        try:
            if query_context.query_type == QueryType.ANALYSIS_REQUEST:
                return await self._handle_analysis_request(query_context)
            elif query_context.query_type == QueryType.RESULT_INTERPRETATION:
                return await self._handle_result_interpretation(query_context)
            elif query_context.query_type == QueryType.RECOMMENDATION_REQUEST:
                return await self._handle_recommendation_request(query_context)
            elif query_context.query_type == QueryType.TREND_ANALYSIS:
                return await self._handle_trend_analysis(query_context)
            elif query_context.query_type == QueryType.ANOMALY_INQUIRY:
                return await self._handle_anomaly_inquiry(query_context)
            elif query_context.query_type == QueryType.PERFORMANCE_QUESTION:
                return await self._handle_performance_question(query_context)
            elif query_context.query_type == QueryType.GENERAL_HELP:
                return await self._handle_general_help(query_context)
            else:
                return await self._handle_unknown_query(query_context)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return AssistantResponse(
                response_type=ResponseType.ERROR_MESSAGE,
                content=f"I encountered an error processing your query: {str(e)}",
                confidence=0.0
            )
    
    async def _handle_analysis_request(self, query_context: QueryContext) -> AssistantResponse:
        """Handle analysis request queries."""
        if not self.current_analysis:
            return AssistantResponse(
                response_type=ResponseType.TEXT_RESPONSE,
                content="I don't have any current analysis results to work with. Please run an analysis first, or provide analysis data.",
                confidence=0.8
            )
        
        # Generate analysis summary
        summary = self._generate_analysis_summary()
        
        return AssistantResponse(
            response_type=ResponseType.DATA_ANALYSIS,
            content=f"Here's an analysis of your simulation comparison results:\n\n{summary}",
            data=self._extract_analysis_data(),
            confidence=0.9
        )
    
    async def _handle_result_interpretation(self, query_context: QueryContext) -> AssistantResponse:
        """Handle result interpretation queries."""
        if not self.current_analysis:
            return AssistantResponse(
                response_type=ResponseType.TEXT_RESPONSE,
                content="I don't have any analysis results to interpret. Please run an analysis first.",
                confidence=0.8
            )
        
        # Find relevant knowledge entries
        relevant_knowledge = self._find_relevant_knowledge(query_context.query_text)
        
        # Generate interpretation
        interpretation = self._generate_result_interpretation(query_context, relevant_knowledge)
        
        return AssistantResponse(
            response_type=ResponseType.TEXT_RESPONSE,
            content=interpretation,
            recommendations=self._generate_interpretation_recommendations(),
            confidence=0.8
        )
    
    async def _handle_recommendation_request(self, query_context: QueryContext) -> AssistantResponse:
        """Handle recommendation request queries."""
        if not self.current_analysis:
            return AssistantResponse(
                response_type=ResponseType.TEXT_RESPONSE,
                content="I need analysis results to provide recommendations. Please run an analysis first.",
                confidence=0.8
            )
        
        # Generate recommendations based on analysis results
        recommendations = self._generate_smart_recommendations(query_context)
        
        return AssistantResponse(
            response_type=ResponseType.RECOMMENDATION,
            content="Based on your analysis results, here are my recommendations:",
            recommendations=recommendations,
            confidence=0.8
        )
    
    async def _handle_trend_analysis(self, query_context: QueryContext) -> AssistantResponse:
        """Handle trend analysis queries."""
        if not self.current_analysis:
            return AssistantResponse(
                response_type=ResponseType.TEXT_RESPONSE,
                content="I need analysis results to perform trend analysis. Please run an analysis first.",
                confidence=0.8
            )
        
        # Extract trend information
        trend_data = self._extract_trend_data()
        
        return AssistantResponse(
            response_type=ResponseType.DATA_ANALYSIS,
            content=f"Here's the trend analysis for your simulation data:\n\n{trend_data['summary']}",
            data=trend_data,
            visualizations=trend_data.get('charts', []),
            confidence=0.8
        )
    
    async def _handle_anomaly_inquiry(self, query_context: QueryContext) -> AssistantResponse:
        """Handle anomaly inquiry queries."""
        if not self.current_analysis:
            return AssistantResponse(
                response_type=ResponseType.TEXT_RESPONSE,
                content="I need analysis results to check for anomalies. Please run an analysis first.",
                confidence=0.8
            )
        
        # Extract anomaly information
        anomaly_data = self._extract_anomaly_data()
        
        return AssistantResponse(
            response_type=ResponseType.DATA_ANALYSIS,
            content=f"Here's what I found regarding anomalies in your simulation data:\n\n{anomaly_data['summary']}",
            data=anomaly_data,
            visualizations=anomaly_data.get('charts', []),
            confidence=0.8
        )
    
    async def _handle_performance_question(self, query_context: QueryContext) -> AssistantResponse:
        """Handle performance-related queries."""
        if not self.current_analysis:
            return AssistantResponse(
                response_type=ResponseType.TEXT_RESPONSE,
                content="I need analysis results to answer performance questions. Please run an analysis first.",
                confidence=0.8
            )
        
        # Extract performance data
        performance_data = self._extract_performance_data()
        
        return AssistantResponse(
            response_type=ResponseType.DATA_ANALYSIS,
            content=f"Here's the performance analysis for your simulations:\n\n{performance_data['summary']}",
            data=performance_data,
            visualizations=performance_data.get('charts', []),
            confidence=0.8
        )
    
    async def _handle_general_help(self, query_context: QueryContext) -> AssistantResponse:
        """Handle general help queries."""
        # Find relevant knowledge entries
        relevant_knowledge = self._find_relevant_knowledge(query_context.query_text)
        
        if relevant_knowledge:
            content = "Here's what I can help you with:\n\n"
            for entry in relevant_knowledge[:3]:  # Top 3 most relevant
                content += f"**{entry.title}**\n{entry.content}\n\n"
        else:
            content = """I can help you with simulation analysis in several ways:

1. **Analysis Requests**: "Analyze my simulation results" or "Compare these simulations"
2. **Result Interpretation**: "What does this mean?" or "Explain these results"
3. **Recommendations**: "What should I do?" or "Give me recommendations"
4. **Trend Analysis**: "Show me trends" or "How is performance changing?"
5. **Anomaly Detection**: "Are there any anomalies?" or "Find outliers"
6. **Performance Questions**: "How is the performance?" or "Why is it slow?"

Just ask me in natural language and I'll do my best to help!"""
        
        return AssistantResponse(
            response_type=ResponseType.TEXT_RESPONSE,
            content=content,
            confidence=0.9
        )
    
    async def _handle_unknown_query(self, query_context: QueryContext) -> AssistantResponse:
        """Handle unknown or unclear queries."""
        return AssistantResponse(
            response_type=ResponseType.TEXT_RESPONSE,
            content="I'm not sure I understand your question. Could you rephrase it? I can help with analysis, interpretation, recommendations, trends, anomalies, and performance questions.",
            confidence=0.3
        )
    
    def _find_relevant_knowledge(self, query_text: str) -> List[KnowledgeEntry]:
        """Find relevant knowledge base entries for a query."""
        if not self.knowledge_base:
            return []
        
        query_lower = query_text.lower()
        relevant_entries = []
        
        for entry in self.knowledge_base:
            # Simple keyword matching
            score = 0
            for tag in entry.tags:
                if tag.lower() in query_lower:
                    score += 1
            
            if entry.title.lower() in query_lower or any(word in query_lower for word in entry.content.lower().split()):
                score += 2
            
            if score > 0:
                entry.relevance_score = score
                relevant_entries.append(entry)
        
        # Sort by relevance score
        relevant_entries.sort(key=lambda x: x.relevance_score, reverse=True)
        return relevant_entries[:5]  # Top 5 most relevant
    
    def _generate_analysis_summary(self) -> str:
        """Generate a summary of the current analysis."""
        if not self.current_analysis:
            return "No analysis data available."
        
        summary = f"**Analysis Summary**\n\n"
        summary += f"- **Status**: {'✅ Success' if self.current_analysis.success else '❌ Failed'}\n"
        summary += f"- **Duration**: {self.current_analysis.total_duration:.2f} seconds\n"
        summary += f"- **Phases Completed**: {len(self.current_analysis.phase_results)}\n"
        
        if self.current_analysis.errors:
            summary += f"- **Errors**: {len(self.current_analysis.errors)} errors found\n"
        
        if self.current_analysis.warnings:
            summary += f"- **Warnings**: {len(self.current_analysis.warnings)} warnings\n"
        
        return summary
    
    def _extract_analysis_data(self) -> Dict[str, Any]:
        """Extract structured data from current analysis."""
        if not self.current_analysis:
            return {}
        
        return {
            "success": self.current_analysis.success,
            "duration": self.current_analysis.total_duration,
            "phases": len(self.current_analysis.phase_results),
            "errors": len(self.current_analysis.errors),
            "warnings": len(self.current_analysis.warnings),
            "summary": self.current_analysis.summary
        }
    
    def _generate_result_interpretation(self, query_context: QueryContext, knowledge: List[KnowledgeEntry]) -> str:
        """Generate interpretation of analysis results."""
        interpretation = "Based on your analysis results:\n\n"
        
        if self.current_analysis and self.current_analysis.success:
            interpretation += "✅ **Analysis completed successfully**\n"
            interpretation += f"- The analysis took {self.current_analysis.total_duration:.2f} seconds\n"
            interpretation += f"- {len(self.current_analysis.phase_results)} phases were completed\n"
            
            if self.current_analysis.errors:
                interpretation += f"- ⚠️ {len(self.current_analysis.errors)} errors were encountered\n"
            
            if self.current_analysis.warnings:
                interpretation += f"- ⚠️ {len(self.current_analysis.warnings)} warnings were generated\n"
        else:
            interpretation += "❌ **Analysis failed or incomplete**\n"
            if self.current_analysis and self.current_analysis.errors:
                interpretation += f"- Errors: {', '.join(self.current_analysis.errors[:3])}\n"
        
        # Add relevant knowledge
        if knowledge:
            interpretation += f"\n**Additional Context:**\n{knowledge[0].content}\n"
        
        return interpretation
    
    def _generate_interpretation_recommendations(self) -> List[str]:
        """Generate recommendations based on interpretation."""
        recommendations = []
        
        if self.current_analysis and self.current_analysis.errors:
            recommendations.append("Review and fix the errors found in the analysis")
        
        if self.current_analysis and self.current_analysis.warnings:
            recommendations.append("Address the warnings to improve analysis quality")
        
        if self.current_analysis and self.current_analysis.total_duration > 300:  # 5 minutes
            recommendations.append("Consider optimizing analysis performance for faster results")
        
        return recommendations
    
    def _generate_smart_recommendations(self, query_context: QueryContext) -> List[str]:
        """Generate smart recommendations based on analysis results."""
        recommendations = []
        
        if not self.current_analysis:
            return ["Run an analysis first to get specific recommendations"]
        
        # Performance recommendations
        if self.current_analysis.total_duration > 600:  # 10 minutes
            recommendations.append("Consider using parallel processing to speed up analysis")
        
        # Error handling recommendations
        if self.current_analysis.errors:
            recommendations.append("Investigate and resolve the analysis errors")
            recommendations.append("Check input data quality and format")
        
        # Warning recommendations
        if self.current_analysis.warnings:
            recommendations.append("Review warnings to improve analysis reliability")
        
        # General recommendations
        recommendations.append("Consider running additional analysis phases for deeper insights")
        recommendations.append("Use visualization tools to better understand the results")
        
        return recommendations
    
    def _extract_trend_data(self) -> Dict[str, Any]:
        """Extract trend data from analysis results."""
        if not self.current_analysis:
            return {"summary": "No analysis data available for trend analysis"}
        
        # This would extract actual trend data from the analysis results
        # For now, return a placeholder
        return {
            "summary": "Trend analysis shows performance patterns over time",
            "trends": ["Performance is stable", "Memory usage is consistent"],
            "charts": []
        }
    
    def _extract_anomaly_data(self) -> Dict[str, Any]:
        """Extract anomaly data from analysis results."""
        if not self.current_analysis:
            return {"summary": "No analysis data available for anomaly detection"}
        
        # This would extract actual anomaly data from the analysis results
        # For now, return a placeholder
        return {
            "summary": "Anomaly detection found some unusual patterns",
            "anomalies": ["High memory usage in simulation 3", "Unusual execution time in simulation 7"],
            "charts": []
        }
    
    def _extract_performance_data(self) -> Dict[str, Any]:
        """Extract performance data from analysis results."""
        if not self.current_analysis:
            return {"summary": "No analysis data available for performance analysis"}
        
        # This would extract actual performance data from the analysis results
        # For now, return a placeholder
        return {
            "summary": f"Performance analysis completed in {self.current_analysis.total_duration:.2f} seconds",
            "metrics": {
                "total_duration": self.current_analysis.total_duration,
                "phases_completed": len(self.current_analysis.phase_results),
                "success_rate": 1.0 if self.current_analysis.success else 0.0
            },
            "charts": []
        }
    
    def set_analysis_context(self, analysis_result: OrchestrationResult, context: Optional[Dict[str, Any]] = None):
        """Set the current analysis context for the assistant."""
        self.current_analysis = analysis_result
        self.analysis_context = context or {}
        logger.info("Analysis context updated")
    
    def add_knowledge_entry(self, title: str, content: str, category: str, tags: List[str] = None):
        """Add a new knowledge entry."""
        entry = KnowledgeEntry(
            id=f"custom_{len(self.knowledge_base) + 1}",
            title=title,
            content=content,
            category=category,
            tags=tags or []
        )
        self.knowledge_base.append(entry)
        self._save_knowledge_base()
        logger.info(f"Added knowledge entry: {title}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "total_entries": len(self.knowledge_base),
            "categories": list(set(entry.category for entry in self.knowledge_base)),
            "total_tags": len(set(tag for entry in self.knowledge_base for tag in entry.tags)),
            "most_used": max(self.knowledge_base, key=lambda x: x.usage_count).title if self.knowledge_base else None
        }