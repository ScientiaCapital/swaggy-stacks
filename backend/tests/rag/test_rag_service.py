"""Comprehensive tests for the RAG Service component."""

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

from app.rag.services.rag_service import (
    TradingRAGService,
    RAGQuery,
    RAGResult,
    ChunkingStrategy,
    DocumentType
)
from app.rag.services.memory_manager import Memory, MemoryType
from tests.rag.fixtures.market_data_fixtures import (
    sample_market_data,
    sample_agent_memory,
    mock_embedding_vectors,
    mock_db_connection
)


class TestRAGQuery:
    """Test the RAGQuery data class."""
    
    def test_query_creation(self):
        """Test creating a RAGQuery instance."""
        query = RAGQuery(
            query_text="Find bullish momentum patterns",
            context={
                "symbol": "AAPL",
                "timeframe": "1d"
            },
            max_results=10,
            min_relevance_score=0.7,
            document_types=[DocumentType.MARKET_ANALYSIS],
            include_metadata=True
        )
        
        assert query.query_text == "Find bullish momentum patterns"
        assert query.context["symbol"] == "AAPL"
        assert query.max_results == 10
        assert query.min_relevance_score == 0.7
        assert DocumentType.MARKET_ANALYSIS in query.document_types
        assert query.include_metadata is True
    
    def test_query_with_temporal_filter(self):
        """Test query with temporal filtering."""
        query = RAGQuery(
            query_text="Recent trading patterns",
            time_range_days=7,
            max_results=5
        )
        
        assert query.time_range_days == 7
        assert query.max_results == 5


class TestRAGResult:
    """Test the RAGResult data class."""
    
    def test_result_creation(self):
        """Test creating a RAGResult instance."""
        result = RAGResult(
            content="AAPL shows strong momentum with volume confirmation",
            relevance_score=0.85,
            document_type=DocumentType.PATTERN_ANALYSIS,
            metadata={
                "symbol": "AAPL",
                "timestamp": "2024-01-15T10:30:00",
                "confidence": 0.8
            },
            source_id="pattern_001"
        )
        
        assert result.content == "AAPL shows strong momentum with volume confirmation"
        assert result.relevance_score == 0.85
        assert result.document_type == DocumentType.PATTERN_ANALYSIS
        assert result.metadata["symbol"] == "AAPL"
        assert result.source_id == "pattern_001"


@pytest.mark.asyncio
class TestTradingRAGService:
    """Comprehensive tests for TradingRAGService."""
    
    @pytest.fixture
    async def rag_service(self, mock_db_connection):
        """Create a RAG service instance for testing."""
        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.embed_text.return_value = MagicMock(
            embedding=np.random.rand(384).astype(np.float32)
        )
        mock_embedding_service.embed_batch.return_value = [
            MagicMock(embedding=np.random.rand(384).astype(np.float32))
            for _ in range(3)
        ]
        
        # Mock memory manager
        mock_memory_manager = AsyncMock()
        
        with patch('app.rag.services.rag_service.get_db_connection') as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db_connection)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=None)
            
            service = TradingRAGService(
                embedding_service=mock_embedding_service,
                memory_manager=mock_memory_manager,
                chunk_size=500,
                chunk_overlap=50
            )
            await service.initialize()
            return service
    
    async def test_initialization(self, rag_service):
        """Test RAG service initialization."""
        assert rag_service.chunk_size == 500
        assert rag_service.chunk_overlap == 50
        assert rag_service.embedding_service is not None
        assert rag_service.memory_manager is not None
        assert rag_service._initialized is True
    
    async def test_add_document(self, rag_service, mock_db_connection):
        """Test adding a document to the RAG knowledge base."""
        document_content = """
        Apple Inc. (AAPL) Technical Analysis Report
        
        Current Price: $150.25
        Trend: Bullish momentum with strong volume confirmation
        Key Indicators:
        - RSI: 65.5 (neutral to slightly overbought)
        - MACD: Bullish crossover confirmed
        - Volume: 45M shares (above average)
        
        Pattern Analysis:
        The stock is showing a classic volume breakout pattern with strong institutional support.
        Previous resistance at $148 has now become support.
        """
        
        metadata = {
            "symbol": "AAPL",
            "report_date": "2024-01-15",
            "analyst": "TradingBot",
            "confidence": 0.85
        }
        
        # Mock successful database insertion
        mock_db_connection.return_value = "doc_001"
        
        doc_id = await rag_service.add_document(
            content=document_content,
            document_type=DocumentType.TECHNICAL_ANALYSIS,
            metadata=metadata,
            source_id="tech_analysis_001"
        )
        
        assert doc_id == "doc_001"
        # Verify embedding service was called for chunks
        assert rag_service.embedding_service.embed_batch.called
        # Verify database insertion was attempted
        assert len(mock_db_connection.queries_executed) > 0
    
    async def test_semantic_search(self, rag_service, mock_db_connection):
        """Test semantic search functionality."""
        # Mock search results from database
        mock_chunks = [
            {
                "chunk_id": "chunk_001",
                "content": "AAPL bullish momentum pattern with volume confirmation",
                "document_type": "technical_analysis",
                "metadata": json.dumps({"symbol": "AAPL", "confidence": 0.85}),
                "source_id": "tech_001",
                "relevance_score": 0.92
            },
            {
                "chunk_id": "chunk_002", 
                "content": "Strong support level at $148 with institutional backing",
                "document_type": "pattern_analysis",
                "metadata": json.dumps({"symbol": "AAPL", "support_level": 148}),
                "source_id": "pattern_001",
                "relevance_score": 0.87
            }
        ]
        
        with patch.object(mock_db_connection, 'fetch', return_value=mock_chunks):
            query = RAGQuery(
                query_text="Find AAPL bullish patterns",
                context={"symbol": "AAPL"},
                max_results=5,
                min_relevance_score=0.8
            )
            
            results = await rag_service.search(query)
            
            assert len(results) == 2
            assert results[0].relevance_score == 0.92
            assert results[0].metadata["symbol"] == "AAPL"
            assert results[1].relevance_score == 0.87
    
    async def test_chunking_strategies(self, rag_service):
        """Test different document chunking strategies."""
        long_document = "This is a test document. " * 100  # Create a long document
        
        # Test semantic chunking
        semantic_chunks = await rag_service._chunk_document(
            content=long_document,
            strategy=ChunkingStrategy.SEMANTIC
        )
        
        # Test fixed size chunking
        fixed_chunks = await rag_service._chunk_document(
            content=long_document,
            strategy=ChunkingStrategy.FIXED_SIZE
        )
        
        assert len(semantic_chunks) > 0
        assert len(fixed_chunks) > 0
        # Semantic chunking might produce different chunk count than fixed
        assert all(len(chunk) <= rag_service.chunk_size + rag_service.chunk_overlap for chunk in fixed_chunks)
    
    async def test_market_context_enhancement(self, rag_service):
        """Test market context enhancement for queries."""
        query = RAGQuery(
            query_text="bullish patterns",
            context={
                "symbol": "AAPL",
                "current_price": 150.25,
                "market_regime": "bull_market",
                "sector": "technology"
            }
        )
        
        enhanced_query = await rag_service._enhance_query_with_market_context(query)
        
        assert "AAPL" in enhanced_query
        assert "bull_market" in enhanced_query or "bullish" in enhanced_query
        assert "technology" in enhanced_query
    
    async def test_temporal_relevance_scoring(self, rag_service):
        """Test temporal relevance scoring for search results."""
        current_time = datetime.now()
        
        # Create test results with different timestamps
        results = [
            RAGResult(
                content="Recent pattern analysis",
                relevance_score=0.8,
                document_type=DocumentType.PATTERN_ANALYSIS,
                metadata={"timestamp": current_time.isoformat()},
                source_id="recent_001"
            ),
            RAGResult(
                content="Old pattern analysis", 
                relevance_score=0.8,
                document_type=DocumentType.PATTERN_ANALYSIS,
                metadata={"timestamp": (current_time - timedelta(days=30)).isoformat()},
                source_id="old_001"
            )
        ]
        
        # Apply temporal scoring
        scored_results = await rag_service._apply_temporal_scoring(results, decay_days=14)
        
        # Recent result should have higher score after temporal adjustment
        recent_result = next(r for r in scored_results if r.source_id == "recent_001")
        old_result = next(r for r in scored_results if r.source_id == "old_001")
        
        assert recent_result.relevance_score >= old_result.relevance_score
    
    async def test_multi_modal_search(self, rag_service, mock_db_connection):
        """Test search across multiple document types."""
        # Mock diverse search results
        mock_results = [
            {
                "chunk_id": "tech_001",
                "content": "Technical analysis shows bullish momentum",
                "document_type": "technical_analysis",
                "metadata": json.dumps({"symbol": "AAPL"}),
                "relevance_score": 0.9
            },
            {
                "chunk_id": "news_001",
                "content": "AAPL earnings beat expectations",
                "document_type": "market_news",
                "metadata": json.dumps({"symbol": "AAPL"}),
                "relevance_score": 0.85
            },
            {
                "chunk_id": "sentiment_001",
                "content": "Bullish sentiment on social media",
                "document_type": "sentiment_analysis",
                "metadata": json.dumps({"symbol": "AAPL"}),
                "relevance_score": 0.8
            }
        ]
        
        with patch.object(mock_db_connection, 'fetch', return_value=mock_results):
            query = RAGQuery(
                query_text="AAPL bullish signals",
                document_types=[
                    DocumentType.TECHNICAL_ANALYSIS,
                    DocumentType.MARKET_NEWS,
                    DocumentType.SENTIMENT_ANALYSIS
                ],
                max_results=10
            )
            
            results = await rag_service.search(query)
            
            assert len(results) == 3
            # Results should be from different document types
            doc_types = {r.document_type for r in results}
            assert len(doc_types) == 3
    
    async def test_query_expansion(self, rag_service):
        """Test automatic query expansion with financial synonyms."""
        original_query = "bullish momentum"
        
        expanded = await rag_service._expand_query(original_query)
        
        # Should contain original terms plus synonyms
        expanded_lower = expanded.lower()
        assert "bullish" in expanded_lower
        assert "momentum" in expanded_lower
        # Should include financial synonyms
        assert any(term in expanded_lower for term in ["uptrend", "positive", "rising", "strength"])
    
    async def test_result_deduplication(self, rag_service):
        """Test deduplication of similar search results."""
        # Create similar results that should be deduplicated
        results = [
            RAGResult(
                content="AAPL shows bullish momentum with high volume",
                relevance_score=0.9,
                document_type=DocumentType.TECHNICAL_ANALYSIS,
                source_id="tech_001"
            ),
            RAGResult(
                content="AAPL demonstrates bullish momentum with volume confirmation",
                relevance_score=0.85,
                document_type=DocumentType.TECHNICAL_ANALYSIS,
                source_id="tech_002"
            ),
            RAGResult(
                content="Tesla shows different pattern",
                relevance_score=0.8,
                document_type=DocumentType.TECHNICAL_ANALYSIS,
                source_id="tech_003"
            )
        ]
        
        deduplicated = await rag_service._deduplicate_results(results, similarity_threshold=0.8)
        
        # Should keep the highest scored result from similar group and the different one
        assert len(deduplicated) == 2
        assert deduplicated[0].relevance_score == 0.9  # Highest score kept
        assert "Tesla" in deduplicated[1].content  # Different content kept
    
    async def test_batch_document_processing(self, rag_service):
        """Test batch processing of multiple documents."""
        documents = [
            {
                "content": f"Technical analysis for symbol {i}",
                "document_type": DocumentType.TECHNICAL_ANALYSIS,
                "metadata": {"symbol": f"STOCK{i}"},
                "source_id": f"batch_{i}"
            }
            for i in range(5)
        ]
        
        with patch.object(rag_service, 'add_document', return_value="mock_id") as mock_add:
            results = await rag_service.batch_add_documents(documents)
            
            assert len(results) == 5
            assert mock_add.call_count == 5
            assert all(doc_id == "mock_id" for doc_id in results)
    
    async def test_financial_entity_extraction(self, rag_service):
        """Test extraction of financial entities from content."""
        content = """
        AAPL is showing strong momentum at $150.25 with a P/E ratio of 25.5.
        The stock broke through resistance at $148 with volume of 45M shares.
        RSI is at 65.5 indicating neutral to slightly overbought conditions.
        """
        
        entities = await rag_service._extract_financial_entities(content)
        
        # Should extract stock symbol, prices, ratios, indicators
        assert "AAPL" in [e["value"] for e in entities if e["type"] == "SYMBOL"]
        assert any(e["type"] == "PRICE" and float(e["value"]) == 150.25 for e in entities)
        assert any(e["type"] == "INDICATOR" and e["value"] == "RSI" for e in entities)
        assert any(e["type"] == "VOLUME" for e in entities)
    
    async def test_contextual_reranking(self, rag_service):
        """Test contextual reranking of search results."""
        results = [
            RAGResult(
                content="AAPL technical analysis shows momentum",
                relevance_score=0.8,
                document_type=DocumentType.TECHNICAL_ANALYSIS,
                metadata={"symbol": "AAPL", "type": "momentum"}
            ),
            RAGResult(
                content="AAPL earnings report positive",
                relevance_score=0.85,
                document_type=DocumentType.MARKET_NEWS,
                metadata={"symbol": "AAPL", "type": "earnings"}
            )
        ]
        
        context = {
            "user_focus": "technical_analysis",
            "preferred_types": ["momentum", "pattern"]
        }
        
        reranked = await rag_service._rerank_with_context(results, context)
        
        # Technical analysis result should be ranked higher due to context
        assert reranked[0].document_type == DocumentType.TECHNICAL_ANALYSIS
    
    async def test_error_handling_invalid_document(self, rag_service):
        """Test error handling for invalid document content."""
        with pytest.raises(ValueError):
            await rag_service.add_document(
                content="",  # Empty content should raise error
                document_type=DocumentType.TECHNICAL_ANALYSIS
            )
    
    async def test_search_performance_optimization(self, rag_service, mock_db_connection):
        """Test search performance with large result sets."""
        # Mock large result set
        large_result_set = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"Content {i} with AAPL analysis",
                "document_type": "technical_analysis",
                "metadata": json.dumps({"symbol": "AAPL"}),
                "relevance_score": 0.9 - (i * 0.01)  # Decreasing relevance
            }
            for i in range(100)
        ]
        
        with patch.object(mock_db_connection, 'fetch', return_value=large_result_set):
            query = RAGQuery(
                query_text="AAPL analysis",
                max_results=10  # Should limit results
            )
            
            results = await rag_service.search(query)
            
            # Should respect max_results limit
            assert len(results) <= 10
            # Should return highest relevance scores first
            assert all(results[i].relevance_score >= results[i+1].relevance_score 
                      for i in range(len(results)-1))
    
    async def test_real_time_document_updates(self, rag_service, mock_db_connection):
        """Test updating existing documents in the knowledge base."""
        # Mock existing document
        original_content = "AAPL at $150 showing neutral signals"
        updated_content = "AAPL at $155 showing strong bullish signals"
        
        mock_db_connection.return_value = True
        
        # Update document
        success = await rag_service.update_document(
            document_id="doc_001",
            new_content=updated_content,
            metadata={"last_updated": datetime.now().isoformat()}
        )
        
        assert success is True
        # Verify embedding service was called for re-embedding
        assert rag_service.embedding_service.embed_batch.called
    
    async def test_concurrent_search_operations(self, rag_service):
        """Test thread safety of concurrent search operations."""
        async def perform_search(query_text: str):
            query = RAGQuery(
                query_text=query_text,
                max_results=5
            )
            return await rag_service.search(query)
        
        # Run multiple concurrent searches
        tasks = [
            perform_search(f"Query {i} AAPL analysis")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All searches should succeed without exceptions
        assert len(results) == 10
        assert not any(isinstance(result, Exception) for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])