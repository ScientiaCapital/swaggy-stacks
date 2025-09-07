"""Integration tests for the complete Agent Intelligence Infrastructure."""

import pytest
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

# Import all components for integration testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from app.rag.services.memory_manager import AgentMemoryManager, Memory, MemoryType
from app.rag.services.rag_service import TradingRAGService, RAGQuery, DocumentType
from app.rag.services.tool_registry import TradingToolRegistry, ToolDefinition, ToolCategory
from app.rag.services.context_builder import TradingContextBuilder, ContextComponent
from app.rag.agents.langgraph.trading_workflow import TradingWorkflowEngine
from tests.rag.fixtures.market_data_fixtures import (
    sample_market_data,
    sample_agent_memory,
    mock_embedding_vectors,
    mock_db_connection,
    mock_redis_client
)


@pytest.mark.asyncio
class TestAgentIntelligenceIntegration:
    """Integration tests for the complete agent intelligence system."""
    
    @pytest.fixture
    async def integrated_system(self, mock_db_connection, mock_redis_client):
        """Create a fully integrated agent intelligence system."""
        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.embed_text.return_value = MagicMock(
            embedding=np.random.rand(384).astype(np.float32)
        )
        mock_embedding_service.embed_batch.return_value = [
            MagicMock(embedding=np.random.rand(384).astype(np.float32))
            for _ in range(3)
        ]
        
        # Mock database and Redis connections
        with patch('app.rag.services.memory_manager.get_db_connection') as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db_connection)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with patch('app.rag.services.memory_manager.get_redis_client') as mock_get_redis:
                mock_get_redis.return_value = mock_redis_client
                
                with patch('app.rag.services.rag_service.get_db_connection') as mock_rag_db:
                    mock_rag_db.return_value.__aenter__ = AsyncMock(return_value=mock_db_connection)
                    mock_rag_db.return_value.__aexit__ = AsyncMock(return_value=None)
                    
                    with patch('app.rag.services.tool_registry.get_db_connection') as mock_tool_db:
                        mock_tool_db.return_value.__aenter__ = AsyncMock(return_value=mock_db_connection)
                        mock_tool_db.return_value.__aexit__ = AsyncMock(return_value=None)
                        
                        # Initialize all components
                        memory_manager = AgentMemoryManager(
                            embedding_service=mock_embedding_service,
                            vector_dimension=384
                        )
                        await memory_manager.initialize()
                        
                        rag_service = TradingRAGService(
                            embedding_service=mock_embedding_service,
                            memory_manager=memory_manager
                        )
                        await rag_service.initialize()
                        
                        tool_registry = TradingToolRegistry()
                        await tool_registry.initialize()
                        
                        context_builder = TradingContextBuilder(
                            memory_manager=memory_manager,
                            rag_service=rag_service,
                            tool_registry=tool_registry
                        )
                        await context_builder.initialize()
                        
                        workflow_engine = TradingWorkflowEngine()
                        await workflow_engine.initialize()
                        
                        return {
                            "memory_manager": memory_manager,
                            "rag_service": rag_service,
                            "tool_registry": tool_registry,
                            "context_builder": context_builder,
                            "workflow_engine": workflow_engine,
                            "embedding_service": mock_embedding_service,
                            "db_connection": mock_db_connection,
                            "redis_client": mock_redis_client
                        }
    
    async def test_complete_trading_decision_workflow(self, integrated_system):
        """Test a complete trading decision workflow through all components."""
        system = integrated_system
        
        # Step 1: Store historical trading patterns in memory
        historical_memory = Memory(
            memory_id="pattern_001",
            agent_id="strategy_agent",
            content="AAPL bullish momentum pattern at $148 with high volume led to 5% gain",
            memory_type=MemoryType.PATTERN,
            metadata={
                "symbol": "AAPL",
                "entry_price": 148.00,
                "exit_price": 155.40,
                "gain_percent": 5.0,
                "volume_confirmation": True
            },
            importance=0.9
        )
        
        await system["memory_manager"].store_memory(historical_memory)
        
        # Step 2: Add technical analysis to RAG knowledge base
        technical_document = """
        Apple Inc. (AAPL) Technical Analysis - January 15, 2024
        
        Current Price: $150.25
        Support Levels: $148.00, $145.50
        Resistance Levels: $152.00, $155.00
        
        Technical Indicators:
        - RSI: 65.5 (approaching overbought but still in bullish range)
        - MACD: Bullish crossover confirmed with histogram expanding
        - Volume: 45M shares (20% above 20-day average)
        - Bollinger Bands: Price near upper band with expansion
        
        Pattern Analysis:
        Strong bullish momentum pattern with volume confirmation.
        Similar pattern in October 2023 led to 8% rally over 2 weeks.
        """
        
        await system["rag_service"].add_document(
            content=technical_document,
            document_type=DocumentType.TECHNICAL_ANALYSIS,
            metadata={
                "symbol": "AAPL",
                "analyst": "TradingBot",
                "confidence": 0.85
            }
        )
        
        # Step 3: Register trading tools
        def mock_rsi_calculator(prices: List[float], period: int = 14) -> Dict[str, Any]:
            # Simple RSI calculation mock
            return {
                "rsi": 65.5,
                "signal": "neutral_to_bullish",
                "overbought_threshold": 70,
                "oversold_threshold": 30
            }
        
        def mock_market_data_tool(symbol: str) -> Dict[str, Any]:
            return {
                "symbol": symbol,
                "current_price": 150.25,
                "volume": 45000000,
                "change": 0.75,
                "change_percent": 0.50
            }
        
        rsi_tool = ToolDefinition(
            name="rsi_calculator",
            description="Calculate RSI indicator",
            category=ToolCategory.TECHNICAL_INDICATORS,
            parameters={
                "prices": {"type": "array", "items": {"type": "number"}, "required": True},
                "period": {"type": "integer", "default": 14}
            },
            implementation=mock_rsi_calculator
        )
        
        market_data_tool = ToolDefinition(
            name="get_market_data",
            description="Get current market data",
            category=ToolCategory.MARKET_DATA,
            parameters={
                "symbol": {"type": "string", "required": True}
            },
            implementation=mock_market_data_tool
        )
        
        await system["tool_registry"].register_tool(rsi_tool)
        await system["tool_registry"].register_tool(market_data_tool)
        
        # Step 4: Build comprehensive context for decision making
        current_market_data = {
            "symbol": "AAPL",
            "current_price": 150.25,
            "volume": 45000000,
            "trend": "bullish",
            "timestamp": datetime.now().isoformat()
        }
        
        # Mock memory search results
        from app.rag.services.memory_manager import MemorySearchResult
        mock_memory_results = [
            MemorySearchResult(
                memory=historical_memory,
                similarity_score=0.92
            )
        ]
        system["memory_manager"].search_memories.return_value = mock_memory_results
        
        # Mock RAG search results
        from app.rag.services.rag_service import RAGResult
        mock_rag_results = [
            RAGResult(
                content="Technical indicators show bullish momentum with RSI at 65.5",
                relevance_score=0.88,
                document_type=DocumentType.TECHNICAL_ANALYSIS,
                source_id="tech_001"
            )
        ]
        system["rag_service"].search.return_value = mock_rag_results
        
        # Execute tools to get current analysis
        from app.rag.services.tool_registry import ToolExecutionContext
        context = ToolExecutionContext(agent_id="strategy_agent", session_id="integration_test")
        
        market_data_result = await system["tool_registry"].execute_tool(
            tool_name="get_market_data",
            parameters={"symbol": "AAPL"},
            context=context
        )
        
        rsi_result = await system["tool_registry"].execute_tool(
            tool_name="rsi_calculator",
            parameters={"prices": [148, 149, 150, 151, 150.25], "period": 14},
            context=context
        )
        
        # Build context with all components
        decision_context = await system["context_builder"].build_context(
            agent_id="strategy_agent",
            session_id="integration_test",
            decision_type="trade_execution",
            market_data=current_market_data,
            tool_results=[market_data_result, rsi_result]
        )
        
        # Verify integration worked
        assert decision_context.agent_id == "strategy_agent"
        assert decision_context.decision_type == "trade_execution"
        assert len(decision_context.memories) > 0
        assert len(decision_context.rag_results) > 0
        assert len(decision_context.tool_results) == 2
        
        # Verify context contains relevant information
        context_summary = decision_context.generate_summary()
        assert "AAPL" in context_summary
        assert "bullish" in context_summary.lower()
        assert "150.25" in context_summary
    
    async def test_langgraph_workflow_integration(self, integrated_system):
        """Test LangGraph workflow integration with all components."""
        system = integrated_system
        
        # Mock workflow execution
        initial_state = {
            "session_id": "workflow_integration_test",
            "symbol": "AAPL",
            "user_preferences": {
                "risk_tolerance": "medium",
                "strategy_focus": ["momentum"],
                "analysis_depth": "detailed"
            }
        }
        
        # Mock workflow engine to use our integrated components
        with patch.object(system["workflow_engine"], '_get_market_context') as mock_market:
            mock_market.return_value = {
                "price": 150.25,
                "volume": 45000000,
                "trend": "bullish"
            }
            
            with patch.object(system["workflow_engine"], '_generate_signals') as mock_signals:
                mock_signals.return_value = [
                    {
                        "strategy": "momentum",
                        "signal": "BUY",
                        "confidence": 0.85,
                        "reasoning": "Strong bullish momentum with volume confirmation"
                    }
                ]
                
                final_state = await system["workflow_engine"].execute_workflow(initial_state)
                
                # Verify workflow completed successfully
                assert final_state["session_id"] == "workflow_integration_test"
                assert "market_context" in final_state
                assert "strategy_signals" in final_state
    
    async def test_memory_rag_interaction(self, integrated_system):
        """Test interaction between memory manager and RAG service."""
        system = integrated_system
        
        # Store a trading pattern in memory
        pattern_memory = Memory(
            memory_id="interaction_test_001",
            agent_id="test_agent",
            content="Tesla breakout pattern above $200 with volume surge",
            memory_type=MemoryType.PATTERN,
            metadata={"symbol": "TSLA", "breakout_level": 200},
            importance=0.8
        )
        
        await system["memory_manager"].store_memory(pattern_memory)
        
        # Add related document to RAG
        await system["rag_service"].add_document(
            content="Tesla technical analysis shows breakout potential above $200 resistance",
            document_type=DocumentType.PATTERN_ANALYSIS,
            metadata={"symbol": "TSLA", "analysis_type": "breakout"}
        )
        
        # Search for related information across both systems
        from app.rag.services.memory_manager import MemoryQuery
        memory_query = MemoryQuery(
            agent_id="test_agent",
            query_text="Tesla breakout patterns",
            memory_types=[MemoryType.PATTERN],
            limit=5
        )
        
        from app.rag.services.rag_service import RAGQuery
        rag_query = RAGQuery(
            query_text="Tesla breakout patterns",
            document_types=[DocumentType.PATTERN_ANALYSIS],
            max_results=5
        )
        
        # Mock search results
        from app.rag.services.memory_manager import MemorySearchResult
        mock_memory_search = [
            MemorySearchResult(memory=pattern_memory, similarity_score=0.9)
        ]
        system["memory_manager"].search_memories.return_value = mock_memory_search
        
        from app.rag.services.rag_service import RAGResult
        mock_rag_search = [
            RAGResult(
                content="Tesla breakout analysis shows strong potential",
                relevance_score=0.85,
                document_type=DocumentType.PATTERN_ANALYSIS,
                source_id="tesla_001"
            )
        ]
        system["rag_service"].search.return_value = mock_rag_search
        
        # Execute searches
        memory_results = await system["memory_manager"].search_memories(memory_query)
        rag_results = await system["rag_service"].search(rag_query)
        
        # Verify both systems found relevant information
        assert len(memory_results) > 0
        assert len(rag_results) > 0
        assert memory_results[0].memory.metadata["symbol"] == "TSLA"
        assert "Tesla" in rag_results[0].content
    
    async def test_tool_context_integration(self, integrated_system):
        """Test integration between tool registry and context builder."""
        system = integrated_system
        
        # Register a complex tool that needs context
        def context_aware_risk_tool(symbol: str, position_size: int, portfolio_data: Dict) -> Dict:
            return {
                "risk_score": 0.25,
                "position_risk": position_size * 0.02,
                "portfolio_impact": 0.15,
                "recommendations": [
                    "Position size within acceptable risk limits",
                    "Monitor correlation with existing holdings"
                ]
            }
        
        risk_tool = ToolDefinition(
            name="portfolio_risk_analyzer",
            description="Analyze portfolio risk for new position",
            category=ToolCategory.RISK_MANAGEMENT,
            parameters={
                "symbol": {"type": "string", "required": True},
                "position_size": {"type": "integer", "required": True},
                "portfolio_data": {"type": "object", "required": True}
            },
            implementation=context_aware_risk_tool
        )
        
        await system["tool_registry"].register_tool(risk_tool)
        
        # Build context that includes portfolio information
        portfolio_context = await system["context_builder"].build_context(
            agent_id="risk_agent",
            session_id="risk_test",
            decision_type="risk_assessment",
            market_data={
                "symbol": "AAPL",
                "current_price": 150.25,
                "portfolio": {
                    "total_value": 100000,
                    "cash": 20000,
                    "positions": {"MSFT": 10000, "GOOGL": 15000}
                }
            }
        )
        
        # Execute risk tool with context
        from app.rag.services.tool_registry import ToolExecutionContext
        tool_context = ToolExecutionContext(
            agent_id="risk_agent",
            session_id="risk_test"
        )
        
        risk_result = await system["tool_registry"].execute_tool(
            tool_name="portfolio_risk_analyzer",
            parameters={
                "symbol": "AAPL",
                "position_size": 1000,
                "portfolio_data": portfolio_context.market_data.get("portfolio", {})
            },
            context=tool_context
        )
        
        # Verify risk analysis completed
        assert risk_result.success is True
        assert "risk_score" in risk_result.data
        assert "recommendations" in risk_result.data
    
    async def test_cross_component_learning(self, integrated_system):
        """Test learning and improvement across all components."""
        system = integrated_system
        
        # Simulate a trading decision and outcome
        decision_memory = Memory(
            memory_id="learning_test_001",
            agent_id="learning_agent",
            content="Bought AAPL at $150.25 based on momentum signals",
            memory_type=MemoryType.DECISION,
            metadata={
                "symbol": "AAPL",
                "action": "BUY",
                "entry_price": 150.25,
                "reasoning": "RSI 65.5, bullish MACD, high volume",
                "confidence": 0.85
            },
            importance=0.7
        )
        
        await system["memory_manager"].store_memory(decision_memory)
        
        # Simulate outcome after some time (positive outcome)
        outcome_memory = Memory(
            memory_id="learning_test_002",
            agent_id="learning_agent",
            content="AAPL position closed at $157.50 for 4.8% gain",
            memory_type=MemoryType.OUTCOME,
            metadata={
                "symbol": "AAPL",
                "action": "SELL",
                "exit_price": 157.50,
                "gain_percent": 4.8,
                "holding_period_days": 5,
                "related_decision": "learning_test_001"
            },
            importance=0.9  # Successful outcomes are important
        )
        
        await system["memory_manager"].store_memory(outcome_memory)
        
        # Update decision memory importance based on outcome
        await system["memory_manager"].update_memory_importance(
            memory_id="learning_test_001",
            agent_id="learning_agent",
            new_importance=0.95,  # Increased due to successful outcome
            outcome_feedback="Successful trade with 4.8% gain in 5 days"
        )
        
        # Search for similar patterns to reinforce learning
        from app.rag.services.memory_manager import MemoryQuery
        learning_query = MemoryQuery(
            agent_id="learning_agent",
            query_text="successful AAPL momentum trades",
            memory_types=[MemoryType.DECISION, MemoryType.OUTCOME],
            min_importance=0.8
        )
        
        # Mock the search to return our learning memories
        from app.rag.services.memory_manager import MemorySearchResult
        mock_learning_results = [
            MemorySearchResult(memory=decision_memory, similarity_score=0.95),
            MemorySearchResult(memory=outcome_memory, similarity_score=0.92)
        ]
        system["memory_manager"].search_memories.return_value = mock_learning_results
        
        learning_results = await system["memory_manager"].search_memories(learning_query)
        
        # Verify learning system captured the successful pattern
        assert len(learning_results) >= 2
        successful_decision = next((r for r in learning_results 
                                  if r.memory.memory_type == MemoryType.DECISION), None)
        assert successful_decision is not None
        assert successful_decision.memory.importance == 0.95  # Should be updated
    
    async def test_system_performance_under_load(self, integrated_system):
        """Test system performance with concurrent operations."""
        system = integrated_system
        
        async def simulate_agent_operation(agent_id: str, operation_id: int):
            """Simulate a complete agent operation."""
            # Store memory
            memory = Memory(
                memory_id=f"perf_test_{operation_id}",
                agent_id=agent_id,
                content=f"Performance test operation {operation_id}",
                memory_type=MemoryType.DECISION,
                importance=0.5
            )
            await system["memory_manager"].store_memory(memory)
            
            # Build context
            context = await system["context_builder"].build_context(
                agent_id=agent_id,
                session_id=f"perf_session_{operation_id}",
                decision_type="performance_test",
                market_data={"symbol": "TEST", "price": 100.0}
            )
            
            return context.context_id
        
        # Run concurrent operations
        tasks = [
            simulate_agent_operation(f"agent_{i % 3}", i)  # 3 different agents
            for i in range(20)  # 20 concurrent operations
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete successfully
        assert len(results) == 20
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Found exceptions: {exceptions}"
    
    async def test_error_recovery_and_resilience(self, integrated_system):
        """Test error recovery and system resilience."""
        system = integrated_system
        
        # Test memory manager failure recovery
        with patch.object(system["memory_manager"], 'store_memory', side_effect=Exception("DB Error")):
            # System should handle memory storage failure gracefully
            try:
                context = await system["context_builder"].build_context(
                    agent_id="resilience_test",
                    session_id="error_test",
                    decision_type="error_recovery",
                    market_data={"symbol": "ERROR_TEST"}
                )
                # Should still build context even if memory fails
                assert context is not None
            except Exception as e:
                # If it does fail, it should be handled gracefully
                assert "graceful" in str(e).lower() or context is not None
        
        # Test tool execution failure handling
        def failing_tool():
            raise RuntimeError("Tool execution failed")
        
        failing_tool_def = ToolDefinition(
            name="failing_test_tool",
            description="Tool that always fails",
            category=ToolCategory.ANALYTICS,
            implementation=failing_tool
        )
        
        await system["tool_registry"].register_tool(failing_tool_def)
        
        from app.rag.services.tool_registry import ToolExecutionContext
        tool_context = ToolExecutionContext(agent_id="resilience_test")
        
        result = await system["tool_registry"].execute_tool(
            tool_name="failing_test_tool",
            parameters={},
            context=tool_context
        )
        
        # Should return failure result, not crash
        assert result.success is False
        assert "failed" in result.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])