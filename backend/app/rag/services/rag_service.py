"""
RAG Service for Trading Agent Intelligence
Provides retrieval-augmented generation capabilities for trading agents
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import asyncpg

from app.core.cache import TTLCache
from app.rag.services.local_embedding import LocalEmbeddingService
from app.rag.services.memory_manager import AgentMemoryManager

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of RAG queries"""

    PATTERN_SIMILARITY = "pattern_similarity"
    DECISION_HISTORY = "decision_history"
    MARKET_KNOWLEDGE = "market_knowledge"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    MULTI_MODAL = "multi_modal"


@dataclass
class RAGQuery:
    """RAG query parameters"""

    agent_id: str
    query_text: str
    query_type: QueryType
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    similarity_threshold: float = 0.7
    max_results: int = 10
    context_window_hours: int = 24
    include_metadata: bool = True


@dataclass
class RAGResult:
    """RAG query result"""

    query_id: str
    agent_id: str
    results: List[Dict[str, Any]]
    total_found: int
    avg_confidence: float
    query_duration_ms: int
    context_summary: str
    recommendations: List[str]
    timestamp: datetime


@dataclass
class PatternMatch:
    """Pattern matching result"""

    pattern_id: int
    pattern_name: str
    similarity_score: float
    success_rate: float
    occurrence_count: int
    last_seen: datetime
    market_context: Dict[str, Any]
    confidence_score: float


class AgentRAGService:
    """
    RAG Service for Trading Agent Intelligence
    Provides sophisticated retrieval-augmented generation capabilities
    """

    def __init__(
        self,
        db_connection_string: str,
        memory_manager: Optional[AgentMemoryManager] = None,
        embedding_service: Optional[LocalEmbeddingService] = None,
        cache_size: int = 1000,
        cache_ttl: int = 300,
    ):
        self.db_connection_string = db_connection_string
        self.memory_manager = memory_manager or AgentMemoryManager(db_connection_string)
        self.embedding_service = embedding_service or LocalEmbeddingService()

        # Query result cache
        self.query_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)

        # Performance metrics
        self._stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "avg_query_time_ms": 0.0,
            "pattern_matches_found": 0,
            "recommendations_generated": 0,
        }

        # Query type handlers
        self._query_handlers = {
            QueryType.PATTERN_SIMILARITY: self._handle_pattern_similarity,
            QueryType.DECISION_HISTORY: self._handle_decision_history,
            QueryType.MARKET_KNOWLEDGE: self._handle_market_knowledge,
            QueryType.PERFORMANCE_ANALYSIS: self._handle_performance_analysis,
            QueryType.MULTI_MODAL: self._handle_multi_modal,
        }

        logger.info("AgentRAGService initialized")

    async def initialize(self):
        """Initialize the RAG service and dependencies"""
        try:
            await self.memory_manager.initialize()
            await self.embedding_service.initialize()
            logger.info("AgentRAGService initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize AgentRAGService: {e}")
            raise

    async def query(self, rag_query: RAGQuery) -> RAGResult:
        """
        Execute a RAG query and return augmented results

        Args:
            rag_query: RAG query parameters

        Returns:
            RAGResult with retrieved patterns and recommendations
        """
        start_time = datetime.now()
        query_id = f"{rag_query.agent_id}_{int(start_time.timestamp())}"

        try:
            # Check cache first
            cache_key = self._generate_cache_key(rag_query)
            if cache_key in self.query_cache:
                self._stats["cache_hits"] += 1
                logger.info(f"RAG query cache hit for {query_id}")
                return self.query_cache[cache_key]

            # Execute query based on type
            handler = self._query_handlers.get(rag_query.query_type)
            if not handler:
                raise ValueError(f"Unsupported query type: {rag_query.query_type}")

            results = await handler(rag_query)

            # Generate contextual recommendations
            recommendations = await self._generate_recommendations(rag_query, results)

            # Create result summary
            context_summary = await self._create_context_summary(results)

            # Calculate metrics
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            avg_confidence = self._calculate_avg_confidence(results)

            # Create RAG result
            rag_result = RAGResult(
                query_id=query_id,
                agent_id=rag_query.agent_id,
                results=results,
                total_found=len(results),
                avg_confidence=avg_confidence,
                query_duration_ms=duration_ms,
                context_summary=context_summary,
                recommendations=recommendations,
                timestamp=start_time,
            )

            # Cache result
            self.query_cache[cache_key] = rag_result

            # Update stats
            self._update_stats(duration_ms, len(results), len(recommendations))

            logger.info(
                f"RAG query {query_id} completed: {len(results)} results, "
                f"{duration_ms}ms, confidence: {avg_confidence:.3f}"
            )

            return rag_result

        except Exception as e:
            logger.error(f"RAG query {query_id} failed: {e}")
            raise

    async def _handle_pattern_similarity(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """Handle pattern similarity queries"""
        try:
            # Generate embedding for query
            embedding_result = await self.embedding_service.get_embedding(
                query.query_text
            )
            query_embedding = embedding_result.embedding

            # Execute similarity search using database function
            async with self._get_db_connection() as conn:
                results = await conn.fetch(
                    """
                    SELECT * FROM find_similar_patterns(
                        $1, $2::vector, $3, $4
                    )
                """,
                    query.agent_id,
                    query_embedding.tolist(),
                    query.similarity_threshold,
                    query.max_results,
                )

                # Enrich results with additional context
                enriched_results = []
                for row in results:
                    # Get pattern metadata
                    pattern_meta = await conn.fetchrow(
                        """
                        SELECT pattern_metadata, market_data, last_seen, created_at
                        FROM agent_patterns
                        WHERE id = $1
                    """,
                        row["id"],
                    )

                    enriched_results.append(
                        {
                            "type": "pattern_match",
                            "pattern_id": row["id"],
                            "pattern_name": row["pattern_name"],
                            "similarity_score": float(row["similarity"]),
                            "success_rate": float(row["success_rate"]),
                            "occurrence_count": row["occurrence_count"],
                            "pattern_metadata": (
                                pattern_meta["pattern_metadata"] if pattern_meta else {}
                            ),
                            "market_data": (
                                pattern_meta["market_data"] if pattern_meta else {}
                            ),
                            "last_seen": (
                                pattern_meta["last_seen"] if pattern_meta else None
                            ),
                            "relevance_score": self._calculate_relevance_score(
                                float(row["similarity"]),
                                float(row["success_rate"]),
                                row["occurrence_count"],
                            ),
                        }
                    )

                return enriched_results

        except Exception as e:
            logger.error(f"Pattern similarity query failed: {e}")
            return []

    async def _handle_decision_history(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """Handle decision history queries"""
        try:
            async with self._get_db_connection() as conn:
                # Build time window filter
                time_filter = datetime.now() - timedelta(
                    hours=query.context_window_hours
                )

                # Query decision history
                sql = """
                    SELECT
                        id, symbol, decision, confidence, context_embedding,
                        market_context, reasoning, outcome, outcome_pnl,
                        created_at, resolved_at
                    FROM agent_decisions
                    WHERE agent_type = $1
                    AND created_at >= $2
                """

                params = [query.agent_id, time_filter]

                # Add symbol filter if specified
                if query.symbol:
                    sql += " AND symbol = $3"
                    params.append(query.symbol)

                sql += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
                params.append(query.max_results)

                results = await conn.fetch(sql, *params)

                # Process results
                decision_results = []
                for row in results:
                    decision_results.append(
                        {
                            "type": "decision_history",
                            "decision_id": row["id"],
                            "symbol": row["symbol"],
                            "decision": row["decision"],
                            "confidence": float(row["confidence"]),
                            "market_context": row["market_context"],
                            "reasoning": row["reasoning"],
                            "outcome": row["outcome"],
                            "outcome_pnl": (
                                float(row["outcome_pnl"])
                                if row["outcome_pnl"]
                                else None
                            ),
                            "created_at": row["created_at"],
                            "resolved_at": row["resolved_at"],
                            "relevance_score": self._calculate_decision_relevance(
                                row, query
                            ),
                        }
                    )

                return decision_results

        except Exception as e:
            logger.error(f"Decision history query failed: {e}")
            return []

    async def _handle_market_knowledge(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """Handle market knowledge queries"""
        try:
            # Generate embedding for query
            embedding_result = await self.embedding_service.get_embedding(
                query.query_text
            )
            query_embedding = embedding_result.embedding

            async with self._get_db_connection() as conn:
                # Vector similarity search on market knowledge
                results = await conn.fetch(
                    """
                    SELECT
                        id, source_type, content_text, symbols, metadata,
                        confidence_score, relevance_score, created_at,
                        1 - (embedding <-> $1::vector) as similarity
                    FROM market_knowledge
                    WHERE 1 - (embedding <-> $1::vector) >= $2
                    AND is_validated = true
                    ORDER BY embedding <-> $1::vector
                    LIMIT $3
                """,
                    query_embedding.tolist(),
                    query.similarity_threshold,
                    query.max_results,
                )

                # Filter by symbol if specified
                knowledge_results = []
                for row in results:
                    # Check symbol filter
                    if query.symbol and query.symbol not in (row["symbols"] or []):
                        continue

                    knowledge_results.append(
                        {
                            "type": "market_knowledge",
                            "knowledge_id": row["id"],
                            "source_type": row["source_type"],
                            "content_text": row["content_text"],
                            "symbols": row["symbols"],
                            "metadata": row["metadata"],
                            "confidence_score": float(row["confidence_score"]),
                            "relevance_score": float(row["relevance_score"]),
                            "similarity_score": float(row["similarity"]),
                            "created_at": row["created_at"],
                        }
                    )

                return knowledge_results

        except Exception as e:
            logger.error(f"Market knowledge query failed: {e}")
            return []

    async def _handle_performance_analysis(
        self, query: RAGQuery
    ) -> List[Dict[str, Any]]:
        """Handle performance analysis queries"""
        try:
            async with self._get_db_connection() as conn:
                # Query agent performance metrics
                results = await conn.fetch(
                    """
                    SELECT
                        measurement_period, total_decisions, correct_predictions,
                        accuracy_rate, avg_confidence, total_pnl, sharpe_ratio,
                        max_drawdown, learning_progress, measured_at
                    FROM agent_performance
                    WHERE agent_type = $1
                    AND measured_at >= $2
                    ORDER BY measured_at DESC
                    LIMIT $3
                """,
                    query.agent_id,
                    datetime.now() - timedelta(hours=query.context_window_hours),
                    query.max_results,
                )

                # Process performance data
                performance_results = []
                for row in results:
                    performance_results.append(
                        {
                            "type": "performance_analysis",
                            "period": row["measurement_period"],
                            "total_decisions": row["total_decisions"],
                            "correct_predictions": row["correct_predictions"],
                            "accuracy_rate": float(row["accuracy_rate"]),
                            "avg_confidence": float(row["avg_confidence"]),
                            "total_pnl": float(row["total_pnl"]),
                            "sharpe_ratio": (
                                float(row["sharpe_ratio"])
                                if row["sharpe_ratio"]
                                else None
                            ),
                            "max_drawdown": float(row["max_drawdown"]),
                            "learning_progress": float(row["learning_progress"]),
                            "measured_at": row["measured_at"],
                            "performance_score": self._calculate_performance_score(row),
                        }
                    )

                return performance_results

        except Exception as e:
            logger.error(f"Performance analysis query failed: {e}")
            return []

    async def _handle_multi_modal(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """Handle multi-modal queries combining multiple data types"""
        try:
            # Execute multiple query types in parallel
            tasks = [
                self._handle_pattern_similarity(query),
                self._handle_decision_history(query),
                self._handle_market_knowledge(query),
            ]

            pattern_results, decision_results, knowledge_results = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            # Combine and rank results
            combined_results = []

            # Add pattern results
            if isinstance(pattern_results, list):
                combined_results.extend(pattern_results[: query.max_results // 3])

            # Add decision results
            if isinstance(decision_results, list):
                combined_results.extend(decision_results[: query.max_results // 3])

            # Add knowledge results
            if isinstance(knowledge_results, list):
                combined_results.extend(knowledge_results[: query.max_results // 3])

            # Sort by relevance score
            combined_results.sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )

            return combined_results[: query.max_results]

        except Exception as e:
            logger.error(f"Multi-modal query failed: {e}")
            return []

    async def _generate_recommendations(
        self, query: RAGQuery, results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate contextual recommendations based on query results"""
        recommendations = []

        try:
            if not results:
                return [
                    "No relevant patterns found - consider expanding search criteria"
                ]

            # Analyze result patterns
            high_confidence_results = [
                r for r in results if r.get("confidence_score", 0) > 0.8
            ]
            recent_results = [r for r in results if self._is_recent_result(r)]

            # Pattern-specific recommendations
            if query.query_type == QueryType.PATTERN_SIMILARITY:
                successful_patterns = [
                    r for r in results if r.get("success_rate", 0) > 0.7
                ]
                if successful_patterns:
                    recommendations.append(
                        f"Found {len(successful_patterns)} high-success patterns - "
                        f"consider applying similar strategies"
                    )

                frequent_patterns = [
                    r for r in results if r.get("occurrence_count", 0) > 10
                ]
                if frequent_patterns:
                    recommendations.append(
                        f"{len(frequent_patterns)} frequently occurring patterns detected - "
                        f"reliable for current market conditions"
                    )

            # Decision history recommendations
            if query.query_type == QueryType.DECISION_HISTORY:
                profitable_decisions = [
                    r for r in results if r.get("outcome_pnl", 0) > 0
                ]
                if profitable_decisions:
                    avg_profit = sum(
                        r.get("outcome_pnl", 0) for r in profitable_decisions
                    ) / len(profitable_decisions)
                    recommendations.append(
                        f"Similar past decisions showed {len(profitable_decisions)} profitable outcomes "
                        f"with avg profit: ${avg_profit:.2f}"
                    )

            # General recommendations
            if high_confidence_results:
                recommendations.append(
                    f"{len(high_confidence_results)} high-confidence matches found - "
                    f"strong signal for current analysis"
                )

            if recent_results:
                recommendations.append(
                    f"{len(recent_results)} recent relevant patterns - "
                    f"consider current market regime alignment"
                )

            return recommendations[:5]  # Limit to top 5 recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Error generating recommendations - using raw results"]

    async def _create_context_summary(self, results: List[Dict[str, Any]]) -> str:
        """Create a context summary from query results"""
        try:
            if not results:
                return "No relevant context found"

            result_types = {}
            for result in results:
                result_type = result.get("type", "unknown")
                result_types[result_type] = result_types.get(result_type, 0) + 1

            summary_parts = []
            for result_type, count in result_types.items():
                summary_parts.append(f"{count} {result_type.replace('_', ' ')} entries")

            avg_confidence = self._calculate_avg_confidence(results)

            return (
                f"Retrieved {len(results)} relevant entries: "
                f"{', '.join(summary_parts)}. "
                f"Average confidence: {avg_confidence:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to create context summary: {e}")
            return "Context summary unavailable"

    def _calculate_relevance_score(
        self, similarity: float, success_rate: float, occurrence_count: int
    ) -> float:
        """Calculate relevance score for pattern matches"""
        # Weighted scoring: similarity (40%), success_rate (40%), frequency (20%)
        frequency_score = min(occurrence_count / 50.0, 1.0)  # Normalize frequency
        return (0.4 * similarity) + (0.4 * success_rate) + (0.2 * frequency_score)

    def _calculate_decision_relevance(
        self, decision: Dict[str, Any], query: RAGQuery
    ) -> float:
        """Calculate relevance score for decision history"""
        base_score = 0.5

        # Boost for same symbol
        if query.symbol and decision.get("symbol") == query.symbol:
            base_score += 0.3

        # Boost for recent decisions
        if decision.get("created_at"):
            days_old = (datetime.now() - decision["created_at"]).days
            recency_boost = max(0, 0.2 * (30 - days_old) / 30)  # Higher for recent
            base_score += recency_boost

        # Boost for profitable outcomes
        if decision.get("outcome_pnl", 0) > 0:
            base_score += 0.2

        return min(base_score, 1.0)

    def _calculate_performance_score(self, performance: Dict[str, Any]) -> float:
        """Calculate performance score"""
        accuracy = performance.get("accuracy_rate", 0)
        pnl_normalized = max(
            -1, min(1, performance.get("total_pnl", 0) / 10000)
        )  # Normalize PnL
        sharpe = min(
            1, max(0, (performance.get("sharpe_ratio", 0) + 2) / 4)
        )  # Normalize Sharpe

        return (0.4 * accuracy) + (0.3 * (pnl_normalized + 1) / 2) + (0.3 * sharpe)

    def _calculate_avg_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average confidence score from results"""
        if not results:
            return 0.0

        confidence_scores = []
        for result in results:
            # Try different confidence score keys
            for key in [
                "confidence_score",
                "similarity_score",
                "relevance_score",
                "confidence",
            ]:
                if key in result and result[key] is not None:
                    confidence_scores.append(float(result[key]))
                    break

        return (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

    def _is_recent_result(self, result: Dict[str, Any]) -> bool:
        """Check if result is from recent timeframe"""
        try:
            timestamp_keys = ["created_at", "measured_at", "last_seen"]
            for key in timestamp_keys:
                if key in result and result[key]:
                    age = datetime.now() - result[key]
                    return age.days <= 7  # Recent if within 7 days
            return False
        except Exception:
            return False

    def _generate_cache_key(self, query: RAGQuery) -> str:
        """Generate cache key for query"""
        key_parts = [
            query.agent_id,
            query.query_type.value,
            hash(query.query_text),
            query.symbol or "all",
            str(query.similarity_threshold),
            str(query.max_results),
        ]
        return "|".join(str(part) for part in key_parts)

    def _update_stats(
        self, duration_ms: int, results_count: int, recommendations_count: int
    ):
        """Update service statistics"""
        self._stats["queries_executed"] += 1
        self._stats["pattern_matches_found"] += results_count
        self._stats["recommendations_generated"] += recommendations_count

        # Update rolling average
        current_avg = self._stats["avg_query_time_ms"]
        query_count = self._stats["queries_executed"]
        self._stats["avg_query_time_ms"] = (
            current_avg * (query_count - 1) + duration_ms
        ) / query_count

    @asynccontextmanager
    async def _get_db_connection(self):
        """Get database connection using same pattern as memory manager"""
        conn = await asyncpg.connect(self.db_connection_string)
        try:
            yield conn
        finally:
            await conn.close()

    async def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics"""
        return {
            **self._stats,
            "cache_size": len(self.query_cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] / max(1, self._stats["queries_executed"])
            ),
            "service_status": "operational",
        }

    async def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
        logger.info("RAG service cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test database connection
            async with self._get_db_connection() as conn:
                await conn.fetchval("SELECT 1")

            # Test memory manager
            await self.memory_manager.health_check()

            # Test embedding service
            test_embedding = await self.embedding_service.get_embedding("health check")

            return {
                "status": "healthy",
                "database_connection": "ok",
                "memory_manager": "ok",
                "embedding_service": "ok",
                "embedding_dimension": len(test_embedding.embedding),
                "stats": await self.get_rag_stats(),
            }

        except Exception as e:
            logger.error(f"RAG service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": await self.get_rag_stats(),
            }


# Export main classes
__all__ = ["AgentRAGService", "RAGQuery", "RAGResult", "QueryType", "PatternMatch"]
