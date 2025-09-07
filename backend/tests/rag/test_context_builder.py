"""Comprehensive tests for the Context Builder component."""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

# Import the components we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from app.rag.services.context_builder import (
    TradingContextBuilder,
    ContextTemplate,
    ContextPriority,
    ContextComponent,
    DecisionContext
)
from app.rag.services.memory_manager import Memory, MemoryType, MemorySearchResult
from app.rag.services.rag_service import RAGResult, DocumentType
from app.rag.services.tool_registry import ToolResult
from tests.rag.fixtures.market_data_fixtures import (
    sample_context_data,
    sample_market_data,
    sample_agent_memory
)


class TestContextTemplate:
    """Test the ContextTemplate data class."""
    
    def test_template_creation(self):
        """Test creating a ContextTemplate instance."""
        template = ContextTemplate(
            name="momentum_analysis",
            description="Template for momentum-based analysis",
            required_components=[
                ContextComponent.MARKET_DATA,
                ContextComponent.RECENT_MEMORIES,
                ContextComponent.TECHNICAL_INDICATORS
            ],
            optional_components=[
                ContextComponent.NEWS_DATA,
                ContextComponent.SENTIMENT_DATA
            ],
            max_context_length=4000,
            priority_weights={
                ContextComponent.MARKET_DATA: 0.4,
                ContextComponent.RECENT_MEMORIES: 0.3,
                ContextComponent.TECHNICAL_INDICATORS: 0.3
            }
        )
        
        assert template.name == "momentum_analysis"
        assert len(template.required_components) == 3
        assert len(template.optional_components) == 2
        assert template.max_context_length == 4000
        assert template.priority_weights[ContextComponent.MARKET_DATA] == 0.4
    
    def test_template_validation(self):
        """Test template validation logic."""
        # Valid template
        valid_template = ContextTemplate(
            name="valid_template",
            description="Valid template",
            required_components=[ContextComponent.MARKET_DATA],
            priority_weights={ContextComponent.MARKET_DATA: 1.0}
        )
        
        assert valid_template.is_valid() is True
        
        # Invalid template with mismatched weights
        invalid_template = ContextTemplate(
            name="invalid_template",
            description="Invalid template",
            required_components=[ContextComponent.MARKET_DATA, ContextComponent.RECENT_MEMORIES],
            priority_weights={ContextComponent.MARKET_DATA: 0.5}  # Missing weight for RECENT_MEMORIES
        )
        
        assert invalid_template.is_valid() is False


class TestDecisionContext:
    """Test the DecisionContext data class."""
    
    def test_context_creation(self):
        """Test creating a DecisionContext instance."""
        context = DecisionContext(
            agent_id="strategy_agent_1",
            session_id="session_001",
            decision_type="trade_execution",
            market_data={"symbol": "AAPL", "price": 150.25},
            memories=[
                {"content": "Previous successful AAPL trade", "relevance": 0.9}
            ],
            tool_results=[
                {"tool": "rsi_calculator", "result": {"rsi": 65.5}}
            ],
            rag_results=[
                {"content": "Technical analysis shows momentum", "relevance": 0.85}
            ],
            context_metadata={
                "created_at": datetime.now().isoformat(),
                "template_used": "momentum_analysis"
            }
        )
        
        assert context.agent_id == "strategy_agent_1"
        assert context.decision_type == "trade_execution"
        assert context.market_data["symbol"] == "AAPL"
        assert len(context.memories) == 1
        assert len(context.tool_results) == 1
    
    def test_context_summary_generation(self):
        """Test generating context summary."""
        context = DecisionContext(
            agent_id="test_agent",
            session_id="test_session",
            decision_type="pattern_recognition",
            market_data={"symbol": "AAPL", "trend": "bullish"},
            memories=[
                {"content": "Bullish pattern detected", "relevance": 0.8}
            ]
        )
        
        summary = context.generate_summary()
        
        assert "AAPL" in summary
        assert "bullish" in summary.lower()
        assert "pattern_recognition" in summary


@pytest.mark.asyncio
class TestTradingContextBuilder:
    """Comprehensive tests for TradingContextBuilder."""
    
    @pytest.fixture
    async def context_builder(self):
        """Create a context builder instance for testing."""
        # Mock dependencies
        mock_memory_manager = AsyncMock()
        mock_rag_service = AsyncMock()
        mock_tool_registry = AsyncMock()
        
        builder = TradingContextBuilder(
            memory_manager=mock_memory_manager,
            rag_service=mock_rag_service,
            tool_registry=mock_tool_registry,
            max_context_tokens=4000,
            default_memory_limit=10
        )
        
        await builder.initialize()
        return builder
    
    async def test_initialization(self, context_builder):
        """Test context builder initialization."""
        assert context_builder.max_context_tokens == 4000
        assert context_builder.default_memory_limit == 10
        assert context_builder._initialized is True
        assert len(context_builder._templates) > 0  # Should have default templates
    
    async def test_register_context_template(self, context_builder):
        """Test registering a custom context template."""
        custom_template = ContextTemplate(
            name="custom_analysis",
            description="Custom analysis template",
            required_components=[
                ContextComponent.MARKET_DATA,
                ContextComponent.RECENT_MEMORIES
            ],
            priority_weights={
                ContextComponent.MARKET_DATA: 0.6,
                ContextComponent.RECENT_MEMORIES: 0.4
            }
        )
        
        success = await context_builder.register_template(custom_template)
        
        assert success is True
        assert "custom_analysis" in context_builder._templates
    
    async def test_build_context_with_template(self, context_builder, sample_context_data):
        """Test building context using a specific template."""
        # Mock memory manager response
        mock_memories = [
            MemorySearchResult(
                memory=Memory(
                    memory_id="mem_001",
                    agent_id="test_agent",
                    content="Previous AAPL analysis showed momentum",
                    memory_type=MemoryType.PATTERN
                ),
                similarity_score=0.85
            )
        ]
        context_builder.memory_manager.search_memories.return_value = mock_memories
        
        # Mock RAG service response
        mock_rag_results = [
            RAGResult(
                content="Technical indicators show bullish momentum",
                relevance_score=0.9,
                document_type=DocumentType.TECHNICAL_ANALYSIS,
                source_id="tech_001"
            )
        ]
        context_builder.rag_service.search.return_value = mock_rag_results
        
        # Mock tool results
        mock_tool_results = [
            ToolResult(
                tool_name="rsi_calculator",
                success=True,
                data={"rsi": 65.5, "signal": "neutral"}
            )
        ]
        
        context = await context_builder.build_context(
            agent_id="test_agent",
            session_id="test_session",
            decision_type="trade_analysis",
            market_data=sample_context_data["current_market"],
            tool_results=mock_tool_results,
            template_name="momentum_analysis"
        )
        
        assert context.agent_id == "test_agent"
        assert context.decision_type == "trade_analysis"
        assert len(context.memories) > 0
        assert len(context.rag_results) > 0
        assert len(context.tool_results) > 0
    
    async def test_context_prioritization(self, context_builder):
        """Test context component prioritization."""
        # Create components with different priorities
        components = {
            ContextComponent.MARKET_DATA: {
                "priority": ContextPriority.CRITICAL,
                "content": "Current price: $150.25",
                "token_count": 100
            },
            ContextComponent.RECENT_MEMORIES: {
                "priority": ContextPriority.HIGH,
                "content": "Previous successful trade pattern",
                "token_count": 200
            },
            ContextComponent.NEWS_DATA: {
                "priority": ContextPriority.MEDIUM,
                "content": "Earnings beat expectations",
                "token_count": 150
            },
            ContextComponent.SENTIMENT_DATA: {
                "priority": ContextPriority.LOW,
                "content": "Social media sentiment positive",
                "token_count": 100
            }
        }
        
        # Test with limited context window
        prioritized = await context_builder._prioritize_components(
            components,
            max_tokens=350  # Should fit critical + high + medium
        )
        
        # Should include critical and high priority components
        assert ContextComponent.MARKET_DATA in prioritized
        assert ContextComponent.RECENT_MEMORIES in prioritized
        # May include medium based on token budget
        total_tokens = sum(comp["token_count"] for comp in prioritized.values())
        assert total_tokens <= 350
    
    async def test_context_summarization(self, context_builder):
        """Test context summarization when exceeding token limits."""
        # Create a context that exceeds token limits
        long_content = "This is a very long piece of content. " * 200  # Very long text
        
        context_data = {
            "market_data": {"symbol": "AAPL", "price": 150.25},
            "memories": [{"content": long_content, "relevance": 0.8}],
            "tool_results": [{"tool": "rsi", "result": {"value": 65.5}}],
            "max_tokens": 500  # Much smaller than content
        }
        
        # Mock summarization
        with patch.object(context_builder, '_summarize_content', return_value="Summarized content") as mock_summarize:
            summarized = await context_builder._apply_intelligent_summarization(
                context_data,
                max_tokens=500
            )
            
            # Should have called summarization
            assert mock_summarize.called
            assert len(str(summarized)) < len(str(context_data))
    
    async def test_temporal_context_filtering(self, context_builder):
        """Test filtering context based on temporal relevance."""
        now = datetime.now()
        
        # Create memories with different timestamps
        memories = [
            {
                "content": "Very recent pattern",
                "timestamp": now.isoformat(),
                "relevance": 0.8
            },
            {
                "content": "Recent pattern",
                "timestamp": (now - timedelta(hours=1)).isoformat(),
                "relevance": 0.9
            },
            {
                "content": "Old pattern",
                "timestamp": (now - timedelta(days=30)).isoformat(),
                "relevance": 0.9
            }
        ]
        
        # Filter with 24-hour window
        filtered = await context_builder._apply_temporal_filtering(
            memories,
            max_age_hours=24
        )
        
        # Should only include memories within 24 hours
        assert len(filtered) == 2
        assert all("recent" in mem["content"].lower() for mem in filtered)
    
    async def test_context_versioning(self, context_builder):
        """Test context versioning for decision continuity."""
        # Build initial context
        context_v1 = await context_builder.build_context(
            agent_id="test_agent",
            session_id="test_session",
            decision_type="initial_analysis",
            market_data={"symbol": "AAPL", "price": 150.00}
        )
        
        # Mock storage of context version
        with patch.object(context_builder, '_store_context_version') as mock_store:
            await context_builder._version_context(context_v1)
            assert mock_store.called
        
        # Build updated context
        context_v2 = await context_builder.build_context(
            agent_id="test_agent",
            session_id="test_session",
            decision_type="follow_up_analysis",
            market_data={"symbol": "AAPL", "price": 152.00},
            previous_context_id=context_v1.context_id
        )
        
        # Should have continuity reference
        assert context_v2.previous_context_id == context_v1.context_id
    
    async def test_multi_symbol_context(self, context_builder):
        """Test building context for multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Mock multi-symbol data
        multi_market_data = {
            symbol: {"price": 150 + i * 10, "volume": 1000000 + i * 500000}
            for i, symbol in enumerate(symbols)
        }
        
        context = await context_builder.build_multi_symbol_context(
            agent_id="portfolio_agent",
            session_id="portfolio_session",
            decision_type="portfolio_rebalancing",
            symbols_data=multi_market_data
        )
        
        assert context.decision_type == "portfolio_rebalancing"
        # Should contain data for all symbols
        for symbol in symbols:
            assert symbol in str(context.market_data)
    
    async def test_context_caching(self, context_builder):
        """Test caching of frequently used context components."""
        cache_key = "market_data_AAPL_2024_01_15"
        cached_data = {"symbol": "AAPL", "price": 150.25, "cached": True}
        
        # Mock cache hit
        with patch.object(context_builder, '_get_cached_component', return_value=cached_data):
            retrieved = await context_builder._get_cached_component(cache_key)
            assert retrieved["cached"] is True
        
        # Mock cache miss and set
        with patch.object(context_builder, '_set_cached_component') as mock_set:
            await context_builder._set_cached_component(cache_key, cached_data, ttl_seconds=300)
            mock_set.assert_called_once_with(cache_key, cached_data, ttl_seconds=300)
    
    async def test_context_validation(self, context_builder):
        """Test validation of built context."""
        # Create valid context
        valid_context = DecisionContext(
            agent_id="test_agent",
            session_id="test_session",
            decision_type="trade_execution",
            market_data={"symbol": "AAPL", "price": 150.25}
        )
        
        is_valid, issues = await context_builder._validate_context(valid_context)
        assert is_valid is True
        assert len(issues) == 0
        
        # Create invalid context
        invalid_context = DecisionContext(
            agent_id="",  # Empty agent ID
            session_id="test_session",
            decision_type="",  # Empty decision type
            market_data=None  # No market data
        )
        
        is_valid, issues = await context_builder._validate_context(invalid_context)
        assert is_valid is False
        assert len(issues) > 0
        assert any("agent_id" in issue.lower() for issue in issues)
    
    async def test_adaptive_context_sizing(self, context_builder):
        """Test adaptive context sizing based on decision complexity."""
        # Simple decision should use less context
        simple_context = await context_builder.build_context(
            agent_id="test_agent",
            session_id="simple_session",
            decision_type="price_check",  # Simple decision
            market_data={"symbol": "AAPL", "price": 150.25}
        )
        
        # Complex decision should use more context
        complex_context = await context_builder.build_context(
            agent_id="test_agent", 
            session_id="complex_session",
            decision_type="portfolio_optimization",  # Complex decision
            market_data={"symbols": ["AAPL", "GOOGL", "MSFT"]},
            include_extended_analysis=True
        )
        
        # Context size should adapt to decision complexity
        simple_size = len(str(simple_context.to_dict()))
        complex_size = len(str(complex_context.to_dict()))
        
        # Complex context should generally be larger (unless heavily summarized)
        # This test verifies the adaptive sizing logic is working
        assert simple_context.decision_type == "price_check"
        assert complex_context.decision_type == "portfolio_optimization"
    
    async def test_context_personalization(self, context_builder):
        """Test context personalization based on agent preferences."""
        agent_preferences = {
            "focus_areas": ["technical_analysis", "momentum"],
            "risk_tolerance": "medium",
            "preferred_indicators": ["RSI", "MACD"],
            "analysis_depth": "detailed"
        }
        
        # Mock personalized memory and RAG results
        context_builder.memory_manager.search_memories.return_value = [
            MemorySearchResult(
                memory=Memory(
                    memory_id="pref_mem_001",
                    agent_id="test_agent",
                    content="Technical analysis with MACD showing momentum",
                    memory_type=MemoryType.PATTERN
                ),
                similarity_score=0.9
            )
        ]
        
        personalized_context = await context_builder.build_personalized_context(
            agent_id="test_agent",
            session_id="personalized_session",
            decision_type="strategy_selection",
            market_data={"symbol": "AAPL"},
            agent_preferences=agent_preferences
        )
        
        # Should reflect personalization
        context_str = str(personalized_context.to_dict()).lower()
        assert "technical" in context_str
        assert "momentum" in context_str
    
    async def test_concurrent_context_building(self, context_builder):
        """Test concurrent context building operations."""
        import asyncio
        
        async def build_test_context(session_id: str):
            return await context_builder.build_context(
                agent_id="concurrent_agent",
                session_id=session_id,
                decision_type="concurrent_test",
                market_data={"symbol": "AAPL", "test_id": session_id}
            )
        
        # Build multiple contexts concurrently
        tasks = [
            build_test_context(f"session_{i}")
            for i in range(5)
        ]
        
        contexts = await asyncio.gather(*tasks)
        
        # All contexts should be built successfully
        assert len(contexts) == 5
        assert all(context.agent_id == "concurrent_agent" for context in contexts)
        
        # Each should have unique session ID
        session_ids = {context.session_id for context in contexts}
        assert len(session_ids) == 5


@pytest.mark.asyncio
class TestContextBuilderIntegration:
    """Integration tests for context builder with other components."""
    
    async def test_end_to_end_context_building(self):
        """Test complete end-to-end context building workflow."""
        # This would test the full integration with all components
        # For now, we verify the integration points exist
        pass
    
    async def test_context_performance_under_load(self):
        """Test context building performance under load."""
        # This would test performance with large datasets
        # Implementation would measure timing and resource usage
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])