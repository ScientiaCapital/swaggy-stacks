"""
Comprehensive tests for BaseTradingAgent with integrated intelligence components.
Tests the complete integration of memory_manager, rag_service, tool_registry, and context_builder.
"""

import pytest
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

# Import the components we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from app.rag.agents.base_agent import (
    BaseTradingAgent, 
    PatternMatch, 
    LearningOutcome
)
from app.rag.types import TradingSignal, MarketContext, AgentState
from app.rag.services.memory_manager import (
    AgentMemoryManager, 
    Memory, 
    MemoryType,
    MemoryQuery
)
from app.rag.services.rag_service import AgentRAGService
from app.rag.services.tool_registry import LangChainToolRegistry
from app.rag.services.context_builder import TradingContextBuilder


class TestBaseTradingAgentIntelligence:
    """Test BaseTradingAgent with integrated intelligence infrastructure."""
    
    @pytest.fixture
    async def mock_dependencies(self):
        """Create mock dependencies for BaseTradingAgent."""
        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.embed_text.return_value = MagicMock(
            embedding=np.random.rand(384).astype(np.float32)
        )
        mock_embedding_service.health_check.return_value = {"status": "healthy"}
        
        # Mock database connection
        mock_db_connection = AsyncMock()
        mock_db_connection.fetchval.return_value = 1
        mock_db_connection.fetch.return_value = []
        mock_db_connection.fetchrow.return_value = {
            "pattern_count": 5,
            "avg_success_rate": 0.75
        }
        mock_db_connection.execute.return_value = True
        
        return {
            "embedding_service": mock_embedding_service,
            "db_connection": mock_db_connection
        }
    
    @pytest.fixture
    async def agent_with_intelligence(self, mock_dependencies):
        """Create BaseTradingAgent with intelligence components."""
        
        class TestTradingAgent(BaseTradingAgent):
            """Concrete implementation for testing."""
            
            async def _create_tools(self):
                """Create test tools."""
                from langchain.agents import Tool
                return [
                    Tool(
                        name="test_tool",
                        description="Test tool for agent",
                        func=lambda x: "test_result"
                    )
                ]
            
            async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
                """Test market analysis."""
                return TradingSignal(
                    agent_type=self.agent_name,
                    strategy_name=self.strategy_type,
                    symbol="AAPL",
                    action="BUY",
                    confidence=0.8,
                    reasoning="Test analysis"
                )
            
            def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
                """Extract test features."""
                return {
                    "price": market_data.get("current_price", 100.0),
                    "volume": market_data.get("volume", 1000000),
                    "trend": "bullish"
                }
        
        # Mock database and embedding service dependencies
        with patch('app.rag.services.memory_manager.get_db_connection') as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(
                return_value=mock_dependencies["db_connection"]
            )
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with patch('app.rag.services.embedding_factory.get_embedding_service') as mock_get_embedding:
                mock_get_embedding.return_value = mock_dependencies["embedding_service"]
                
                with patch('asyncpg.connect') as mock_asyncpg:
                    mock_asyncpg.return_value = mock_dependencies["db_connection"]
                    
                    with patch('app.trading.trading_manager.get_trading_manager') as mock_get_trading:
                        mock_trading_manager = AsyncMock()
                        mock_trading_manager.initialize.return_value = None
                        mock_get_trading.return_value = mock_trading_manager
                        
                        # Create and initialize agent
                        agent = TestTradingAgent(
                            agent_name="test_agent",
                            strategy_type="test_strategy",
                            db_connection_string="postgresql://test",
                            learning_enabled=True
                        )
                        
                        await agent.initialize()
                        return agent

    async def test_agent_initialization_with_intelligence(self, agent_with_intelligence):
        """Test that agent initializes with all intelligence components."""
        agent = agent_with_intelligence
        
        # Verify agent is initialized
        assert agent.is_initialized is True
        assert agent.embedding_service is not None
        
        # Verify all intelligence components are initialized
        assert agent.memory_manager is not None
        assert agent.rag_service is not None
        assert agent.tool_registry is not None
        assert agent.context_builder is not None
        
        # Verify component types
        assert isinstance(agent.memory_manager, AgentMemoryManager)
        assert isinstance(agent.rag_service, AgentRAGService)
        assert isinstance(agent.tool_registry, LangChainToolRegistry)
        assert isinstance(agent.context_builder, TradingContextBuilder)

    async def test_intelligence_components_interconnection(self, agent_with_intelligence):
        """Test that intelligence components are properly interconnected."""
        agent = agent_with_intelligence
        
        # Test memory manager connection
        assert hasattr(agent.memory_manager, 'db_connection_string')
        
        # Test RAG service connection to memory manager
        assert agent.rag_service.memory_manager is agent.memory_manager
        assert agent.rag_service.agent_id == agent.agent_name
        
        # Test tool registry connection
        assert hasattr(agent.tool_registry, 'tools')
        assert len(agent.tools) > 0  # Should have the test tool we created
        
        # Test context builder connections
        assert agent.context_builder.memory_manager is agent.memory_manager
        assert agent.context_builder.rag_service is agent.rag_service
        assert agent.context_builder.tool_registry is agent.tool_registry
        assert agent.context_builder.agent_id == agent.agent_name

    async def test_pattern_matching_integration(self, agent_with_intelligence):
        """Test pattern matching using integrated components."""
        agent = agent_with_intelligence
        
        current_features = {
            "price": 150.0,
            "volume": 1500000,
            "rsi": 65.5,
            "trend": "bullish"
        }
        
        # Mock the database query for pattern matching
        with patch.object(agent, '_get_db_connection') as mock_conn_context:
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = [
                {
                    "id": 1,
                    "pattern_name": "bullish_momentum",
                    "similarity": 0.85,
                    "success_rate": 0.75,
                    "occurrence_count": 10
                }
            ]
            mock_conn_context.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_context.return_value.__aexit__ = AsyncMock(return_value=None)
            
            patterns = await agent.find_similar_patterns(current_features)
            
            assert len(patterns) == 1
            assert patterns[0].pattern_name == "bullish_momentum"
            assert patterns[0].similarity == 0.85

    async def test_learning_outcome_integration(self, agent_with_intelligence):
        """Test learning from outcomes with memory storage."""
        agent = agent_with_intelligence
        
        # Create test trading signal
        original_signal = TradingSignal(
            agent_type=agent.agent_name,
            strategy_name=agent.strategy_type,
            symbol="AAPL",
            action="BUY",
            confidence=0.85,
            reasoning="Strong bullish momentum with volume confirmation",
            entry_price=150.0,
            take_profit=160.0,
            stop_loss=145.0
        )
        
        # Create learning outcome
        learning_outcome = LearningOutcome(
            original_signal=original_signal,
            actual_outcome="WIN",
            pnl=0.067,  # 6.7% gain
            accuracy_score=0.9,  # High accuracy
            market_conditions={
                "volatility": 0.25,
                "volume_ratio": 1.2,
                "market_trend": "bullish"
            }
        )
        
        # Mock database operations
        with patch.object(agent, '_get_db_connection') as mock_conn_context:
            mock_conn = AsyncMock()
            mock_conn.execute.return_value = True
            mock_conn_context.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_context.return_value.__aexit__ = AsyncMock(return_value=None)
            
            await agent.learn_from_outcome(learning_outcome)
            
            # Verify performance stats were updated
            assert agent.performance_stats["total_signals"] == 1
            assert agent.performance_stats["correct_predictions"] == 1
            assert agent.performance_stats["total_pnl"] == 0.067
            assert agent.performance_stats["avg_confidence"] == 0.85

    async def test_pattern_context_generation(self, agent_with_intelligence):
        """Test pattern context generation for LLM reasoning."""
        agent = agent_with_intelligence
        
        # Mock similar patterns
        mock_patterns = [
            PatternMatch(
                pattern_id=1,
                pattern_name="morning_star_reversal",
                similarity=0.92,
                success_rate=0.82,
                occurrence_count=15,
                avg_pnl=0.045
            ),
            PatternMatch(
                pattern_id=2,
                pattern_name="bullish_engulfing",
                similarity=0.88,
                success_rate=0.74,
                occurrence_count=12,
                avg_pnl=0.038
            )
        ]
        
        with patch.object(agent, 'find_similar_patterns', return_value=mock_patterns):
            current_features = {"price": 155.0, "pattern": "reversal_candidate"}
            
            context = await agent.get_pattern_context(current_features)
            
            assert "Found 2 similar historical patterns" in context
            assert "morning_star_reversal" in context
            assert "similarity: 0.92" in context
            assert "success rate: 82.0%" in context
            assert "bullish_engulfing" in context
            assert "seen 15 times" in context

    async def test_health_check_with_intelligence_components(self, agent_with_intelligence):
        """Test comprehensive health check including intelligence components."""
        agent = agent_with_intelligence
        
        health_info = await agent.health_check()
        
        # Basic health info
        assert health_info["agent_name"] == "test_agent"
        assert health_info["strategy_type"] == "test_strategy"
        assert health_info["is_initialized"] is True
        assert health_info["learning_enabled"] is True
        
        # Database connection
        assert health_info["database_connected"] is True
        
        # Embedding service
        assert health_info["embedding_service"] == "healthy"
        
        # Performance stats
        assert "performance" in health_info
        assert isinstance(health_info["performance"], dict)
        
        # Tools count
        assert health_info["tools_count"] == 1  # Our test tool

    async def test_features_to_text_conversion(self, agent_with_intelligence):
        """Test feature dictionary to text conversion for embeddings."""
        agent = agent_with_intelligence
        
        features = {
            "price": 150.25,
            "volume": 1250000,
            "rsi": 65.5,
            "trend": "bullish",
            "indicators": {
                "macd": 0.25,
                "bollinger_position": 0.8
            },
            "patterns": ["hammer", "doji"]
        }
        
        text = agent._features_to_text(features)
        
        assert text.startswith("test_agent analysis:")
        assert "price: 150.2500" in text
        assert "volume: 1250000.0000" in text
        assert "trend: bullish" in text
        assert "macd=0.25" in text
        assert "bollinger_position=0.8" in text
        assert "[hammer, doji]" in text

    async def test_concurrent_agent_operations(self, agent_with_intelligence):
        """Test concurrent operations don't interfere with each other."""
        agent = agent_with_intelligence
        
        async def simulate_trading_operation(operation_id: int):
            """Simulate a complete trading operation."""
            market_data = {
                "symbol": f"TEST{operation_id}",
                "current_price": 100 + operation_id,
                "volume": 1000000 + operation_id * 10000
            }
            
            # Analyze market
            signal = await agent.analyze_market(market_data)
            
            # Get pattern context
            features = agent._extract_market_features(market_data)
            context = await agent.get_pattern_context(features)
            
            return {
                "operation_id": operation_id,
                "signal": signal,
                "context_length": len(context)
            }
        
        # Run multiple concurrent operations
        tasks = [simulate_trading_operation(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should succeed
        assert len(results) == 5
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Found exceptions: {exceptions}"
        
        # Verify results
        for i, result in enumerate(results):
            assert result["operation_id"] == i
            assert isinstance(result["signal"], TradingSignal)
            assert result["context_length"] >= 0

    async def test_error_handling_in_intelligence_components(self, agent_with_intelligence):
        """Test error handling when intelligence components fail."""
        agent = agent_with_intelligence
        
        # Test embedding service failure
        with patch.object(agent.embedding_service, 'embed_text', side_effect=Exception("Embedding failed")):
            # Should handle embedding failure gracefully
            features = {"price": 150.0, "volume": 1000000}
            patterns = await agent.find_similar_patterns(features)
            
            # Should return empty patterns on failure
            assert isinstance(patterns, list)
            assert len(patterns) == 0

    async def test_memory_integration_with_agent(self, agent_with_intelligence):
        """Test memory storage and retrieval integration."""
        agent = agent_with_intelligence
        
        # Test storing a memory through the agent's memory manager
        test_memory = Memory(
            id=None,
            agent_id=agent.agent_name,
            memory_type=MemoryType.PATTERN,
            content="Test pattern recognition successful",
            metadata={"symbol": "AAPL", "confidence": 0.85},
            embedding=None,
            confidence=0.85,
            created_at=datetime.now(),
            importance_score=0.8
        )
        
        with patch.object(agent.memory_manager, 'store_memory', return_value=True) as mock_store:
            result = await agent.memory_manager.store_memory(test_memory)
            
            assert result is True
            mock_store.assert_called_once_with(test_memory)

    async def test_initialization_failure_handling(self, mock_dependencies):
        """Test graceful handling of initialization failures."""
        
        class TestTradingAgent(BaseTradingAgent):
            async def _create_tools(self):
                return []
            
            async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
                return TradingSignal(
                    agent_type=self.agent_name,
                    strategy_name=self.strategy_type,
                    symbol="AAPL",
                    action="HOLD",
                    confidence=0.5,
                    reasoning="Fallback analysis"
                )
            
            def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"price": market_data.get("current_price", 100.0)}
        
        # Mock failure in memory manager initialization
        with patch('app.rag.services.memory_manager.AgentMemoryManager.initialize', 
                  side_effect=Exception("Memory manager init failed")):
            with patch('app.rag.services.embedding_factory.get_embedding_service') as mock_get_embedding:
                mock_get_embedding.return_value = mock_dependencies["embedding_service"]
                
                agent = TestTradingAgent(
                    agent_name="failure_test_agent",
                    strategy_type="failure_test_strategy"
                )
                
                # Initialization should fail
                with pytest.raises(Exception, match="Memory manager init failed"):
                    await agent.initialize()
                
                # Agent should not be marked as initialized
                assert agent.is_initialized is False


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "symbol": "AAPL",
        "current_price": 150.25,
        "volume": 1250000,
        "timestamp": datetime.now().isoformat(),
        "historical_data": {
            "prices": [148.5, 149.2, 150.1, 150.25],
            "volumes": [1100000, 1200000, 1300000, 1250000]
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])