"""
Base Trading Agent Framework with LangChain Integration
Foundation for all specialized trading strategy agents
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import numpy as np
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, BaseMessage, HumanMessage

from app.rag.services.embedding_factory import get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Standard trading signal format across all agents"""

    agent_type: str
    strategy_name: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reasoning: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PatternMatch:
    """Represents a similar pattern found in historical data"""

    pattern_id: int
    pattern_name: str
    similarity: float
    success_rate: float
    occurrence_count: int
    avg_pnl: Optional[float] = None


@dataclass
class LearningOutcome:
    """Result of a completed trade for agent learning"""

    original_signal: TradingSignal
    actual_outcome: str  # WIN, LOSS, BREAKEVEN
    pnl: float
    accuracy_score: float  # How accurate was the confidence prediction
    market_conditions: Dict[str, Any]


class BaseTradingAgent(ABC):
    """
    Base class for all trading strategy agents
    Provides common functionality for pattern matching, learning, and decision making
    """

    def __init__(
        self,
        agent_name: str,
        strategy_type: str,
        db_connection_string: Optional[str] = None,
        min_confidence_threshold: float = 0.6,
        learning_enabled: bool = True,
    ):
        self.agent_name = agent_name
        self.strategy_type = strategy_type
        self.db_connection_string = (
            db_connection_string or "postgresql://localhost/swaggy_rag"
        )
        self.min_confidence_threshold = min_confidence_threshold
        self.learning_enabled = learning_enabled

        # Agent state
        self.is_initialized = False
        self.embedding_service = None

        # LangChain memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000,  # Conservative for M1 memory
        )

        # Performance tracking
        self.performance_stats = {
            "total_signals": 0,
            "correct_predictions": 0,
            "total_pnl": 0.0,
            "patterns_learned": 0,
            "avg_confidence": 0.0,
        }

        # Tools for LangChain integration
        self.tools = []

        logger.info(f"Initialized {agent_name} agent with strategy: {strategy_type}")

    async def initialize(self) -> None:
        """Initialize the agent asynchronously"""
        if self.is_initialized:
            return

        try:
            # Initialize embedding service
            self.embedding_service = await get_embedding_service()

            # Create agent-specific tools
            self.tools = await self._create_tools()

            # Load existing patterns from database
            await self._load_existing_patterns()

            self.is_initialized = True
            logger.info(f"✅ {self.agent_name} agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize {self.agent_name} agent: {e}")
            raise

    @abstractmethod
    async def _create_tools(self) -> List[Tool]:
        """Create LangChain tools specific to this agent's strategy"""
        pass

    @abstractmethod
    async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
        """
        Analyze market data and generate a trading signal
        This is the main method each agent must implement
        """
        pass

    @abstractmethod
    def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract strategy-specific features from market data"""
        pass

    async def find_similar_patterns(
        self,
        current_features: Dict[str, Any],
        similarity_threshold: float = 0.8,
        limit: int = 10,
    ) -> List[PatternMatch]:
        """Find similar historical patterns using vector similarity"""
        try:
            if not self.embedding_service:
                await self.initialize()

            # Convert features to text for embedding
            feature_text = self._features_to_text(current_features)

            # Generate embedding
            embedding_result = await self.embedding_service.embed_text(feature_text)
            pattern_embedding = embedding_result.embedding

            # Search database for similar patterns
            async with self._get_db_connection() as conn:
                similar_patterns = await conn.fetch(
                    """
                    SELECT * FROM find_similar_patterns($1, $2, $3, $4)
                """,
                    self.agent_name,
                    pattern_embedding.tolist(),
                    similarity_threshold,
                    limit,
                )

                return [
                    PatternMatch(
                        pattern_id=row["id"],
                        pattern_name=row["pattern_name"],
                        similarity=row["similarity"],
                        success_rate=row["success_rate"],
                        occurrence_count=row["occurrence_count"],
                    )
                    for row in similar_patterns
                ]

        except Exception as e:
            logger.error(f"Error finding similar patterns for {self.agent_name}: {e}")
            return []

    async def learn_from_outcome(self, outcome: LearningOutcome) -> None:
        """Learn from a completed trade outcome"""
        if not self.learning_enabled:
            return

        try:
            # Extract pattern from the original signal
            pattern_features = self._extract_pattern_features(outcome)
            pattern_text = self._features_to_text(pattern_features)

            # Generate embedding for the pattern
            embedding_result = await self.embedding_service.embed_text(pattern_text)

            # Store or update pattern in database
            await self._store_learned_pattern(
                pattern_features, embedding_result.embedding, outcome
            )

            # Update performance statistics
            await self._update_performance_stats(outcome)

            logger.info(
                f"✅ {self.agent_name} learned from {outcome.actual_outcome} trade"
            )

        except Exception as e:
            logger.error(f"Error in learning for {self.agent_name}: {e}")

    async def get_pattern_context(self, current_features: Dict[str, Any]) -> str:
        """Get contextual information about similar patterns for LLM reasoning"""
        similar_patterns = await self.find_similar_patterns(current_features)

        if not similar_patterns:
            return "No similar patterns found in historical data."

        context_parts = [f"Found {len(similar_patterns)} similar historical patterns:"]

        for i, pattern in enumerate(similar_patterns[:5], 1):  # Top 5 patterns
            context_parts.append(
                f"{i}. Pattern '{pattern.pattern_name}' "
                f"(similarity: {pattern.similarity:.2f}, "
                f"success rate: {pattern.success_rate:.1%}, "
                f"seen {pattern.occurrence_count} times)"
            )

        return "\n".join(context_parts)

    def _features_to_text(self, features: Dict[str, Any]) -> str:
        """Convert feature dictionary to text for embedding"""
        text_parts = [f"{self.agent_name} analysis:"]

        for key, value in features.items():
            if isinstance(value, (int, float)):
                text_parts.append(f"{key}: {value:.4f}")
            elif isinstance(value, str):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                nested_text = ", ".join(f"{k}={v}" for k, v in value.items())
                text_parts.append(f"{key}: {nested_text}")
            elif isinstance(value, list):
                list_text = ", ".join(str(v) for v in value)
                text_parts.append(f"{key}: [{list_text}]")

        return " | ".join(text_parts)

    def _extract_pattern_features(self, outcome: LearningOutcome) -> Dict[str, Any]:
        """Extract features from a learning outcome for pattern storage"""
        signal = outcome.original_signal

        return {
            "strategy": self.strategy_type,
            "action": signal.action,
            "confidence": signal.confidence,
            "outcome": outcome.actual_outcome,
            "pnl": outcome.pnl,
            "accuracy": outcome.accuracy_score,
            "market_conditions": outcome.market_conditions,
            "metadata": signal.metadata,
        }

    @asynccontextmanager
    async def _get_db_connection(self):
        """Get database connection context manager"""
        conn = await asyncpg.connect(self.db_connection_string)
        try:
            yield conn
        finally:
            await conn.close()

    async def _load_existing_patterns(self) -> None:
        """Load existing patterns for this agent from database"""
        try:
            async with self._get_db_connection() as conn:
                patterns = await conn.fetch(
                    """
                    SELECT COUNT(*) as pattern_count, AVG(success_rate) as avg_success_rate
                    FROM agent_patterns 
                    WHERE agent_type = $1 AND is_active = TRUE
                """,
                    self.agent_name,
                )

                if patterns:
                    pattern_info = patterns[0]
                    self.performance_stats["patterns_learned"] = pattern_info[
                        "pattern_count"
                    ]
                    logger.info(
                        f"Loaded {pattern_info['pattern_count']} existing patterns "
                        f"for {self.agent_name} (avg success rate: {pattern_info['avg_success_rate']:.2%})"
                    )

        except Exception as e:
            logger.warning(
                f"Could not load existing patterns for {self.agent_name}: {e}"
            )

    async def _store_learned_pattern(
        self, features: Dict[str, Any], embedding: np.ndarray, outcome: LearningOutcome
    ) -> None:
        """Store a learned pattern in the database"""
        try:
            pattern_name = f"{self.strategy_type}_{outcome.actual_outcome}_{datetime.now().strftime('%Y%m%d')}"

            async with self._get_db_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent_patterns (
                        agent_type, strategy_name, pattern_name, pattern_embedding,
                        pattern_metadata, market_data, success_rate, occurrence_count,
                        total_profit_loss, is_active
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    self.agent_name,
                    self.strategy_type,
                    pattern_name,
                    embedding.tolist(),
                    json.dumps(features),
                    json.dumps(outcome.market_conditions),
                    1.0 if outcome.actual_outcome == "WIN" else 0.0,
                    1,
                    outcome.pnl,
                    True,
                )

        except Exception as e:
            logger.error(f"Error storing pattern for {self.agent_name}: {e}")

    async def _update_performance_stats(self, outcome: LearningOutcome) -> None:
        """Update agent performance statistics"""
        self.performance_stats["total_signals"] += 1
        if outcome.actual_outcome == "WIN":
            self.performance_stats["correct_predictions"] += 1

        self.performance_stats["total_pnl"] += outcome.pnl

        # Update average confidence
        total_signals = self.performance_stats["total_signals"]
        current_avg = self.performance_stats["avg_confidence"]
        new_confidence = outcome.original_signal.confidence
        self.performance_stats["avg_confidence"] = (
            current_avg * (total_signals - 1) + new_confidence
        ) / total_signals

        # Store in database
        try:
            async with self._get_db_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent_decisions (
                        agent_type, strategy_name, symbol, decision, confidence,
                        market_context, reasoning, outcome, outcome_pnl, outcome_accuracy
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    self.agent_name,
                    self.strategy_type,
                    outcome.original_signal.symbol,
                    outcome.original_signal.action,
                    outcome.original_signal.confidence,
                    json.dumps(outcome.market_conditions),
                    outcome.original_signal.reasoning,
                    outcome.actual_outcome,
                    outcome.pnl,
                    outcome.accuracy_score,
                )

        except Exception as e:
            logger.error(f"Error updating performance stats for {self.agent_name}: {e}")

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this agent"""
        try:
            async with self._get_db_connection() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT * FROM agent_performance_summary 
                    WHERE agent_type = $1 AND strategy_name = $2
                """,
                    self.agent_name,
                    self.strategy_type,
                )

                if stats:
                    return dict(stats)
                else:
                    return self.performance_stats.copy()

        except Exception as e:
            logger.error(
                f"Error getting performance summary for {self.agent_name}: {e}"
            )
            return self.performance_stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Check agent health and readiness"""
        health_info = {
            "agent_name": self.agent_name,
            "strategy_type": self.strategy_type,
            "is_initialized": self.is_initialized,
            "learning_enabled": self.learning_enabled,
            "tools_count": len(self.tools),
            "performance": self.performance_stats.copy(),
        }

        # Test database connection
        try:
            async with self._get_db_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                health_info["database_connected"] = result == 1
        except:
            health_info["database_connected"] = False

        # Test embedding service
        try:
            if self.embedding_service:
                embed_health = await self.embedding_service.health_check()
                health_info["embedding_service"] = embed_health["status"]
            else:
                health_info["embedding_service"] = "not_initialized"
        except:
            health_info["embedding_service"] = "error"

        return health_info


# Factory function for creating agents
async def create_trading_agent(agent_type: str, **kwargs) -> BaseTradingAgent:
    """Factory function to create specific trading agents"""

    # Import consolidated strategy agent
    from app.rag.agents.consolidated_strategy_agent import ConsolidatedStrategyAgent

    # Create agent with specified strategy
    valid_strategies = ["markov", "elliott_wave", "fibonacci", "wyckoff"]
    if agent_type not in valid_strategies:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Valid types: {valid_strategies}"
        )

    agent = ConsolidatedStrategyAgent(strategies=[agent_type], **kwargs)

    await agent.initialize()
    return agent
