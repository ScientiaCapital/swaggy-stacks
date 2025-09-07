"""
Simple focused test for BaseTradingAgent integration with intelligence components.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from app.rag.agents.base_agent import BaseTradingAgent
from app.rag.types import TradingSignal


class TestBaseTradingAgentSimple:
    """Simple test for BaseTradingAgent intelligence integration."""
    
    async def test_basic_agent_initialization(self):
        """Test that agent can be initialized with intelligence components."""
        
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
        
        # Create agent without initializing intelligence components (to test basic setup)
        agent = TestTradingAgent(
            agent_name="test_agent", 
            strategy_type="test_strategy",
            learning_enabled=False  # Disable learning to simplify test
        )
        
        # Test basic agent properties
        assert agent.agent_name == "test_agent"
        assert agent.strategy_type == "test_strategy"
        assert agent.is_initialized is False
        assert agent.learning_enabled is False
        
        # Test basic market analysis without full initialization
        market_data = {"current_price": 150.0, "volume": 1000000}
        signal = await agent.analyze_market(market_data)
        
        assert isinstance(signal, TradingSignal)
        assert signal.agent_type == "test_agent"
        assert signal.symbol == "AAPL"
        assert signal.action == "BUY"
        assert signal.confidence == 0.8

    async def test_features_to_text_conversion(self):
        """Test feature dictionary to text conversion."""
        
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
                    reasoning="Test"
                )
            
            def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"price": market_data.get("current_price", 100.0)}
        
        agent = TestTradingAgent(
            agent_name="test_agent",
            strategy_type="test_strategy"
        )
        
        features = {
            "price": 150.25,
            "volume": 1250000,
            "rsi": 65.5,
            "trend": "bullish"
        }
        
        text = agent._features_to_text(features)
        
        assert text.startswith("test_agent analysis:")
        assert "price: 150.2500" in text
        assert "volume: 1250000.0000" in text
        assert "trend: bullish" in text

    async def test_pattern_context_with_no_patterns(self):
        """Test pattern context when no patterns are found."""
        
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
                    reasoning="Test"
                )
            
            def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"price": market_data.get("current_price", 100.0)}
        
        agent = TestTradingAgent(
            agent_name="test_agent",
            strategy_type="test_strategy"
        )
        
        with patch.object(agent, 'find_similar_patterns', return_value=[]):
            current_features = {"price": 155.0, "pattern": "unknown"}
            context = await agent.get_pattern_context(current_features)
            
            assert context == "No similar patterns found in historical data."

    async def test_agent_initialization_with_mocked_intelligence(self):
        """Test agent initialization with properly mocked intelligence components."""
        
        class TestTradingAgent(BaseTradingAgent):
            async def _create_tools(self):
                from langchain.agents import Tool
                return [
                    Tool(
                        name="test_tool", 
                        description="Test tool", 
                        func=lambda x: "test"
                    )
                ]
            
            async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
                return TradingSignal(
                    agent_type=self.agent_name,
                    strategy_name=self.strategy_type,
                    symbol="AAPL",
                    action="HOLD",
                    confidence=0.5,
                    reasoning="Test"
                )
            
            def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"price": market_data.get("current_price", 100.0)}
        
        # Mock all the external dependencies
        with patch('app.rag.services.embedding_factory.get_embedding_service') as mock_embedding:
            mock_embedding_service = AsyncMock()
            mock_embedding_service.embed_text.return_value = MagicMock(
                embedding=np.random.rand(384).astype(np.float32)
            )
            mock_embedding_service.health_check.return_value = {"status": "healthy"}
            mock_embedding.return_value = mock_embedding_service
            
            with patch('asyncpg.connect') as mock_connect:
                mock_conn = AsyncMock()
                mock_conn.fetchval.return_value = 1
                mock_conn.fetch.return_value = []
                mock_conn.fetchrow.return_value = {"pattern_count": 0, "avg_success_rate": 0.0}
                mock_connect.return_value = mock_conn
                
                with patch('app.trading.trading_manager.get_trading_manager') as mock_trading:
                    mock_trading_manager = AsyncMock()
                    mock_trading_manager.initialize.return_value = None
                    mock_trading.return_value = mock_trading_manager
                    
                    # Create and initialize agent
                    agent = TestTradingAgent(
                        agent_name="test_agent",
                        strategy_type="test_strategy",
                        db_connection_string="postgresql://test",
                        learning_enabled=True
                    )
                    
                    # This should now work without errors
                    await agent.initialize()
                    
                    # Verify agent is initialized
                    assert agent.is_initialized is True
                    assert agent.embedding_service is not None
                    
                    # Verify intelligence components exist
                    assert agent.memory_manager is not None
                    assert agent.rag_service is not None
                    assert agent.tool_registry is not None
                    assert agent.context_builder is not None
                    
                    # Test health check
                    health = await agent.health_check()
                    assert health["agent_name"] == "test_agent"
                    assert health["is_initialized"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])