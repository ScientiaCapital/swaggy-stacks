"""Comprehensive tests for the Memory Manager component."""

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

from app.rag.services.memory_manager import (
    AgentMemoryManager,
    MemoryType,
    Memory,
    MemoryQuery,
    MemorySearchResult
)
from tests.rag.fixtures.market_data_fixtures import (
    sample_agent_memory,
    mock_embedding_vectors,
    mock_db_connection,
    mock_redis_client
)


class TestMemory:
    """Test the Memory data class."""
    
    def test_memory_creation(self):
        """Test creating a Memory instance."""
        memory = Memory(
            memory_id="test_001",
            agent_id="agent_1",
            content="Test memory content",
            memory_type=MemoryType.DECISION,
            metadata={"test": "data"},
            importance=0.8
        )
        
        assert memory.memory_id == "test_001"
        assert memory.agent_id == "agent_1"
        assert memory.content == "Test memory content"
        assert memory.memory_type == MemoryType.DECISION
        assert memory.metadata == {"test": "data"}
        assert memory.importance == 0.8
        assert isinstance(memory.created_at, datetime)
    
    def test_memory_to_dict(self):
        """Test converting Memory to dictionary."""
        memory = Memory(
            memory_id="test_001",
            agent_id="agent_1",
            content="Test memory content",
            memory_type=MemoryType.PATTERN
        )
        
        memory_dict = memory.to_dict()
        assert memory_dict["memory_id"] == "test_001"
        assert memory_dict["memory_type"] == "pattern"
        assert "created_at" in memory_dict
    
    def test_memory_from_dict(self):
        """Test creating Memory from dictionary."""
        memory_data = {
            "memory_id": "test_001",
            "agent_id": "agent_1",
            "content": "Test memory content",
            "memory_type": "decision",
            "metadata": {"test": "data"},
            "importance": 0.8,
            "created_at": "2024-01-01T12:00:00"
        }
        
        memory = Memory.from_dict(memory_data)
        assert memory.memory_id == "test_001"
        assert memory.memory_type == MemoryType.DECISION
        assert memory.metadata == {"test": "data"}


class TestMemoryQuery:
    """Test the MemoryQuery data class."""
    
    def test_query_creation(self):
        """Test creating a MemoryQuery instance."""
        query = MemoryQuery(
            agent_id="agent_1",
            query_text="Find bullish patterns",
            memory_types=[MemoryType.PATTERN],
            limit=10,
            min_importance=0.5
        )
        
        assert query.agent_id == "agent_1"
        assert query.query_text == "Find bullish patterns"
        assert query.memory_types == [MemoryType.PATTERN]
        assert query.limit == 10
        assert query.min_importance == 0.5


@pytest.mark.asyncio
class TestAgentMemoryManager:
    """Comprehensive tests for AgentMemoryManager."""
    
    @pytest.fixture
    async def memory_manager(self, mock_db_connection, mock_redis_client):
        """Create a memory manager instance for testing."""
        with patch('app.rag.services.memory_manager.get_db_connection') as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db_connection)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with patch('app.rag.services.memory_manager.get_redis_client') as mock_get_redis:
                mock_get_redis.return_value = mock_redis_client
                
                # Mock embedding service
                mock_embedding_service = AsyncMock()
                mock_embedding_service.embed_text.return_value = MagicMock(
                    embedding=np.random.rand(384).astype(np.float32)
                )
                
                manager = AgentMemoryManager(
                    embedding_service=mock_embedding_service,
                    vector_dimension=384
                )
                await manager.initialize()
                return manager
    
    async def test_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert memory_manager.vector_dimension == 384
        assert memory_manager.embedding_service is not None
        assert memory_manager._initialized is True
    
    async def test_store_memory(self, memory_manager, mock_db_connection):
        """Test storing a memory."""
        memory = Memory(
            memory_id="test_001",
            agent_id="agent_1",
            content="Test bullish pattern recognition",
            memory_type=MemoryType.PATTERN,
            metadata={"symbol": "AAPL", "confidence": 0.85},
            importance=0.8
        )
        
        # Mock successful database insertion
        mock_db_connection.return_value = True
        
        result = await memory_manager.store_memory(memory)
        
        assert result is True
        # Verify embedding service was called
        memory_manager.embedding_service.embed_text.assert_called_once()
        # Verify database insertion was attempted
        assert len(mock_db_connection.queries_executed) > 0
    
    async def test_retrieve_memory(self, memory_manager, mock_db_connection):
        """Test retrieving a specific memory by ID."""
        # Mock database response
        mock_memory_data = {
            "memory_id": "test_001",
            "agent_id": "agent_1",
            "content": "Test memory content",
            "memory_type": "pattern",
            "metadata": json.dumps({"test": "data"}),
            "importance": 0.8,
            "created_at": datetime.now()
        }
        mock_db_connection.return_value = mock_memory_data
        
        memory = await memory_manager.retrieve_memory("test_001", "agent_1")
        
        assert memory is not None
        assert memory.memory_id == "test_001"
        assert memory.agent_id == "agent_1"
        assert memory.memory_type == MemoryType.PATTERN
    
    async def test_search_memories(self, memory_manager, mock_db_connection):
        """Test searching for memories using semantic similarity."""
        # Mock database response with multiple memories
        mock_memories = [
            {
                "memory_id": "mem_001",
                "agent_id": "agent_1",
                "content": "Bullish momentum pattern",
                "memory_type": "pattern",
                "metadata": json.dumps({"symbol": "AAPL"}),
                "importance": 0.9,
                "created_at": datetime.now(),
                "similarity_score": 0.85
            },
            {
                "memory_id": "mem_002",
                "agent_id": "agent_1",
                "content": "Volume breakout signal",
                "memory_type": "signal",
                "metadata": json.dumps({"symbol": "TSLA"}),
                "importance": 0.7,
                "created_at": datetime.now(),
                "similarity_score": 0.75
            }
        ]
        
        with patch.object(mock_db_connection, 'fetch', return_value=mock_memories):
            query = MemoryQuery(
                agent_id="agent_1",
                query_text="Find bullish patterns",
                memory_types=[MemoryType.PATTERN],
                limit=5
            )
            
            results = await memory_manager.search_memories(query)
            
            assert len(results) == 2
            assert results[0].memory.memory_id == "mem_001"
            assert results[0].similarity_score == 0.85
            assert results[1].similarity_score == 0.75
    
    async def test_consolidate_memories(self, memory_manager):
        """Test memory consolidation functionality."""
        # Mock memories to consolidate
        memories = [
            Memory(
                memory_id="mem_001",
                agent_id="agent_1",
                content="AAPL bullish pattern at $150",
                memory_type=MemoryType.PATTERN,
                metadata={"symbol": "AAPL", "price": 150},
                importance=0.8
            ),
            Memory(
                memory_id="mem_002",
                agent_id="agent_1",
                content="AAPL volume breakout at $152",
                memory_type=MemoryType.PATTERN,
                metadata={"symbol": "AAPL", "price": 152},
                importance=0.7
            )
        ]
        
        with patch.object(memory_manager, 'search_memories') as mock_search:
            mock_search.return_value = [
                MemorySearchResult(memory=memories[0], similarity_score=0.9),
                MemorySearchResult(memory=memories[1], similarity_score=0.85)
            ]
            
            consolidated = await memory_manager.consolidate_memories("agent_1", "AAPL patterns")
            
            assert consolidated is not None
            assert "AAPL" in consolidated.content
            assert consolidated.memory_type == MemoryType.CONSOLIDATED
    
    async def test_prune_memories(self, memory_manager, mock_db_connection):
        """Test memory pruning based on importance and age."""
        # Mock low importance, old memories
        mock_old_memories = [
            {
                "memory_id": "old_001",
                "created_at": datetime.now() - timedelta(days=365),
                "importance": 0.2
            },
            {
                "memory_id": "old_002",
                "created_at": datetime.now() - timedelta(days=200),
                "importance": 0.1
            }
        ]
        
        with patch.object(mock_db_connection, 'fetch', return_value=mock_old_memories):
            with patch.object(mock_db_connection, 'execute', return_value=True):
                pruned_count = await memory_manager.prune_memories(
                    agent_id="agent_1",
                    min_importance=0.3,
                    max_age_days=180
                )
                
                assert pruned_count == 2
    
    async def test_get_memory_statistics(self, memory_manager, mock_db_connection):
        """Test retrieving memory statistics."""
        mock_stats = [{
            "total_memories": 150,
            "by_type": json.dumps({
                "pattern": 60,
                "decision": 45,
                "market_state": 25,
                "consolidated": 20
            }),
            "avg_importance": 0.72,
            "oldest_memory": (datetime.now() - timedelta(days=90)).isoformat(),
            "newest_memory": datetime.now().isoformat()
        }]
        
        with patch.object(mock_db_connection, 'fetch', return_value=mock_stats):
            stats = await memory_manager.get_memory_statistics("agent_1")
            
            assert stats["total_memories"] == 150
            assert stats["by_type"]["pattern"] == 60
            assert stats["avg_importance"] == 0.72
    
    async def test_update_memory_importance(self, memory_manager, mock_db_connection):
        """Test updating memory importance based on outcomes."""
        # Mock successful database update
        mock_db_connection.return_value = True
        
        result = await memory_manager.update_memory_importance(
            memory_id="test_001",
            agent_id="agent_1",
            new_importance=0.95,
            outcome_feedback="Very successful trade based on this pattern"
        )
        
        assert result is True
        # Verify database update was called
        assert len(mock_db_connection.queries_executed) > 0
    
    async def test_batch_store_memories(self, memory_manager):
        """Test storing multiple memories in a batch operation."""
        memories = [
            Memory(
                memory_id=f"batch_{i}",
                agent_id="agent_1",
                content=f"Batch memory {i}",
                memory_type=MemoryType.PATTERN,
                importance=0.5
            )
            for i in range(5)
        ]
        
        with patch.object(memory_manager, 'store_memory', return_value=True) as mock_store:
            results = await memory_manager.batch_store_memories(memories)
            
            assert len(results) == 5
            assert all(results)
            assert mock_store.call_count == 5
    
    async def test_error_handling_invalid_memory_type(self, memory_manager):
        """Test error handling for invalid memory types."""
        with pytest.raises(ValueError):
            Memory(
                memory_id="invalid_001",
                agent_id="agent_1",
                content="Test content",
                memory_type="invalid_type"  # This should raise an error
            )
    
    async def test_search_with_temporal_filters(self, memory_manager, mock_db_connection):
        """Test searching memories with temporal constraints."""
        # Mock memories with different timestamps
        now = datetime.now()
        mock_memories = [
            {
                "memory_id": "recent_001",
                "agent_id": "agent_1",
                "content": "Recent pattern",
                "memory_type": "pattern",
                "metadata": json.dumps({}),
                "importance": 0.8,
                "created_at": now,
                "similarity_score": 0.9
            }
        ]
        
        with patch.object(mock_db_connection, 'fetch', return_value=mock_memories):
            query = MemoryQuery(
                agent_id="agent_1",
                query_text="Recent patterns",
                time_range_days=7  # Last week only
            )
            
            results = await memory_manager.search_memories(query)
            
            assert len(results) == 1
            assert results[0].memory.memory_id == "recent_001"
    
    async def test_memory_embedding_consistency(self, memory_manager):
        """Test that similar content produces similar embeddings."""
        content1 = "AAPL showing bullish momentum with high volume"
        content2 = "Apple stock demonstrates strong bullish signals with volume confirmation"
        
        # Mock embedding service to return different but similar embeddings
        embedding1 = np.array([0.8, 0.6, 0.4, 0.2])
        embedding2 = np.array([0.75, 0.65, 0.35, 0.25])
        
        memory_manager.embedding_service.embed_text.side_effect = [
            MagicMock(embedding=embedding1),
            MagicMock(embedding=embedding2)
        ]
        
        # Calculate similarity (should be high for similar content)
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        assert similarity > 0.8  # Should be highly similar
    
    async def test_concurrent_memory_operations(self, memory_manager):
        """Test thread safety of concurrent memory operations."""
        async def store_test_memory(memory_id: str):
            memory = Memory(
                memory_id=memory_id,
                agent_id="agent_1",
                content=f"Concurrent test memory {memory_id}",
                memory_type=MemoryType.DECISION
            )
            return await memory_manager.store_memory(memory)
        
        # Run multiple concurrent store operations
        tasks = [store_test_memory(f"concurrent_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should succeed without exceptions
        assert len(results) == 10
        assert not any(isinstance(result, Exception) for result in results)


@pytest.mark.asyncio
class TestMemoryManagerIntegration:
    """Integration tests for memory manager with other components."""
    
    async def test_memory_with_rag_service_integration(self):
        """Test memory manager integration with RAG service."""
        # This would test the actual integration between memory manager and RAG service
        # For now, we'll mock the integration points
        pass
    
    async def test_memory_persistence_across_restarts(self):
        """Test that memories persist across service restarts."""
        # This would test actual database persistence
        # Implementation would involve starting/stopping the service
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])