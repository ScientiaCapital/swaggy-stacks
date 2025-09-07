"""
Agent Memory Manager for persistent memory operations.

Provides memory storage and retrieval for trading agents using pgvector
and existing PostgreSQL infrastructure.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import json
import numpy as np
import asyncpg

from app.core.logging import get_logger
from app.core.cache import get_embedding_cache
from app.rag.services.embedding_factory import get_embedding_service

logger = get_logger(__name__)


class MemoryType(str, Enum):
    """Types of agent memories"""
    EPISODIC = "episodic"      # Complete trading sessions/episodes
    PATTERN = "pattern"        # Market patterns and behaviors
    DECISION = "decision"      # Individual decisions and outcomes
    KNOWLEDGE = "knowledge"    # Market facts and research


@dataclass
class Memory:
    """Represents a stored agent memory"""
    id: Optional[int]
    agent_id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray]
    confidence: float
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    importance_score: float = 0.0


@dataclass
class MemoryQuery:
    """Query parameters for memory retrieval"""
    query_text: str
    memory_types: List[MemoryType]
    agent_id: Optional[str] = None
    limit: int = 10
    similarity_threshold: float = 0.7
    time_window: Optional[timedelta] = None
    include_metadata: bool = True


class AgentMemoryManager:
    """
    Manages persistent memory operations for trading agents.
    
    Provides storage, retrieval, and consolidation of agent memories
    using pgvector for similarity search and PostgreSQL for persistence.
    """
    
    def __init__(self, db_connection_string: str):
        """Initialize memory manager"""
        self.db_connection_string = db_connection_string
        self.embedding_service = None
        self.memory_cache = get_embedding_cache()  # Reuse existing cache pattern
        self.consolidation_threshold = 0.9  # High similarity for consolidation
        
        # Performance tracking
        self._stats = {
            "memories_stored": 0,
            "memories_retrieved": 0,
            "cache_hits": 0,
            "consolidations_performed": 0
        }
    
    async def initialize(self):
        """Initialize embedding service and validate database schema"""
        try:
            self.embedding_service = await get_embedding_service()
            
            # Test database connectivity
            async with self._get_db_connection() as conn:
                # Verify required tables exist
                tables = await conn.fetch("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('agent_patterns', 'agent_decisions', 'market_knowledge')
                """)
                
                if len(tables) < 3:
                    logger.warning("Some required tables are missing from database schema")
                else:
                    logger.info("AgentMemoryManager initialized successfully")
                    
        except Exception as e:
            logger.error(f"Failed to initialize AgentMemoryManager: {e}")
            raise
    
    @asynccontextmanager
    async def _get_db_connection(self):
        """Get database connection context manager - reuses base_agent pattern"""
        conn = await asyncpg.connect(self.db_connection_string)
        try:
            yield conn
        finally:
            await conn.close()
    
    async def store_memory(
        self,
        agent_id: str,
        memory_type: MemoryType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        importance_score: float = 0.5
    ) -> int:
        """
        Store a new memory for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory (episodic, pattern, decision, knowledge)  
            content: Text content of the memory
            metadata: Additional metadata dictionary
            confidence: Confidence score (0.0-1.0)
            importance_score: Importance for retention (0.0-1.0)
            
        Returns:
            int: ID of the stored memory
        """
        if not self.embedding_service:
            raise RuntimeError("Memory manager not initialized")
        
        metadata = metadata or {}
        
        try:
            # Generate embedding for content
            embedding_result = await self.embedding_service.embed_text(content)
            embedding = embedding_result.embedding
            
            # Store based on memory type using existing tables
            async with self._get_db_connection() as conn:
                async with conn.transaction():
                    if memory_type == MemoryType.PATTERN:
                        memory_id = await self._store_pattern_memory(
                            conn, agent_id, content, embedding, metadata, confidence
                        )
                    elif memory_type == MemoryType.DECISION:
                        memory_id = await self._store_decision_memory(
                            conn, agent_id, content, embedding, metadata, confidence
                        )
                    else:  # EPISODIC or KNOWLEDGE
                        memory_id = await self._store_knowledge_memory(
                            conn, agent_id, content, embedding, metadata, confidence, memory_type
                        )
            
            self._stats["memories_stored"] += 1
            logger.debug(f"Stored {memory_type} memory for agent {agent_id}: ID {memory_id}")
            
            # Cache the memory
            cache_key = f"memory:{memory_id}"
            memory_obj = Memory(
                id=memory_id,
                agent_id=agent_id,
                memory_type=memory_type,
                content=content,
                metadata=metadata,
                embedding=embedding,
                confidence=confidence,
                created_at=datetime.now(),
                importance_score=importance_score
            )
            await self.memory_cache.set(cache_key, memory_obj)
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory for agent {agent_id}: {e}")
            raise
    
    async def retrieve_memories(
        self,
        query: MemoryQuery
    ) -> List[Memory]:
        """
        Retrieve memories based on similarity search.
        
        Args:
            query: MemoryQuery with search parameters
            
        Returns:
            List[Memory]: Retrieved memories sorted by relevance
        """
        if not self.embedding_service:
            raise RuntimeError("Memory manager not initialized")
        
        try:
            # Generate query embedding
            embedding_result = await self.embedding_service.embed_text(query.query_text)
            query_embedding = embedding_result.embedding
            
            memories = []
            
            # Search in each memory type
            for memory_type in query.memory_types:
                type_memories = await self._search_memories_by_type(
                    memory_type, query, query_embedding
                )
                memories.extend(type_memories)
            
            # Sort by similarity and apply limit
            memories.sort(key=lambda m: m.confidence, reverse=True)
            memories = memories[:query.limit]
            
            self._stats["memories_retrieved"] += len(memories)
            logger.debug(f"Retrieved {len(memories)} memories for query")
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise
    
    async def _search_memories_by_type(
        self,
        memory_type: MemoryType,
        query: MemoryQuery,
        query_embedding: np.ndarray
    ) -> List[Memory]:
        """Search memories in specific table based on type"""
        
        async with self._get_db_connection() as conn:
            if memory_type == MemoryType.PATTERN:
                return await self._search_pattern_memories(conn, query, query_embedding)
            elif memory_type == MemoryType.DECISION:
                return await self._search_decision_memories(conn, query, query_embedding)
            else:  # EPISODIC or KNOWLEDGE
                return await self._search_knowledge_memories(conn, query, query_embedding, memory_type)
    
    async def _store_pattern_memory(
        self,
        conn: asyncpg.Connection,
        agent_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        confidence: float
    ) -> int:
        """Store pattern memory in agent_patterns table"""
        
        strategy_name = metadata.get('strategy_name', 'unknown')
        pattern_name = metadata.get('pattern_name', 'custom_pattern')
        market_data = metadata.get('market_data', {})
        
        result = await conn.fetchrow("""
            INSERT INTO agent_patterns (
                agent_type, strategy_name, pattern_name, pattern_embedding,
                pattern_metadata, market_data, avg_confidence, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """, agent_id, strategy_name, pattern_name, embedding.tolist(),
            json.dumps(metadata), json.dumps(market_data), confidence, datetime.now())
        
        return result['id']
    
    async def _store_decision_memory(
        self,
        conn: asyncpg.Connection,
        agent_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        confidence: float
    ) -> int:
        """Store decision memory in agent_decisions table"""
        
        symbol = metadata.get('symbol', 'UNKNOWN')
        decision = metadata.get('decision', 'HOLD')
        strategy_name = metadata.get('strategy_name', 'unknown')
        market_context = metadata.get('market_context', {})
        reasoning = content
        
        result = await conn.fetchrow("""
            INSERT INTO agent_decisions (
                agent_type, strategy_name, symbol, decision, confidence,
                context_embedding, market_context, reasoning, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
        """, agent_id, strategy_name, symbol, decision, confidence,
            embedding.tolist(), json.dumps(market_context), reasoning, datetime.now())
        
        return result['id']
    
    async def _store_knowledge_memory(
        self,
        conn: asyncpg.Connection,
        agent_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        confidence: float,
        memory_type: MemoryType
    ) -> int:
        """Store episodic/knowledge memory in market_knowledge table"""
        
        source_type = memory_type.value
        symbols = metadata.get('symbols', [])
        source_timestamp = metadata.get('timestamp', datetime.now())
        
        result = await conn.fetchrow("""
            INSERT INTO market_knowledge (
                source_type, content_text, embedding, symbols,
                metadata, confidence_score, created_at, source_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """, source_type, content, embedding.tolist(), symbols,
            json.dumps(metadata), confidence, datetime.now(), source_timestamp)
        
        return result['id']
    
    async def _search_pattern_memories(
        self,
        conn: asyncpg.Connection,
        query: MemoryQuery,
        query_embedding: np.ndarray
    ) -> List[Memory]:
        """Search pattern memories using vector similarity"""
        
        # Use existing find_similar_patterns procedure if available, otherwise direct query
        sql = """
            SELECT id, agent_type, strategy_name, pattern_name, pattern_embedding,
                   pattern_metadata, avg_confidence, created_at,
                   1 - (pattern_embedding <-> $1) as similarity
            FROM agent_patterns
            WHERE ($2::text IS NULL OR agent_type = $2)
            AND is_active = true
            AND 1 - (pattern_embedding <-> $1) >= $3
            ORDER BY pattern_embedding <-> $1
            LIMIT $4
        """
        
        rows = await conn.fetch(
            sql,
            query_embedding.tolist(),
            query.agent_id,
            query.similarity_threshold,
            query.limit
        )
        
        memories = []
        for row in rows:
            metadata = json.loads(row['pattern_metadata']) if row['pattern_metadata'] else {}
            
            memory = Memory(
                id=row['id'],
                agent_id=row['agent_type'],
                memory_type=MemoryType.PATTERN,
                content=f"Pattern: {row['pattern_name']} (Strategy: {row['strategy_name']})",
                metadata=metadata,
                embedding=np.array(row['pattern_embedding']),
                confidence=float(row['similarity']),
                created_at=row['created_at']
            )
            memories.append(memory)
        
        return memories
    
    async def _search_decision_memories(
        self,
        conn: asyncpg.Connection,
        query: MemoryQuery,
        query_embedding: np.ndarray
    ) -> List[Memory]:
        """Search decision memories using vector similarity"""
        
        sql = """
            SELECT id, agent_type, strategy_name, symbol, decision, confidence,
                   context_embedding, market_context, reasoning, created_at,
                   1 - (context_embedding <-> $1) as similarity
            FROM agent_decisions
            WHERE ($2::text IS NULL OR agent_type = $2)
            AND context_embedding IS NOT NULL
            AND 1 - (context_embedding <-> $1) >= $3
            ORDER BY context_embedding <-> $1
            LIMIT $4
        """
        
        rows = await conn.fetch(
            sql,
            query_embedding.tolist(),
            query.agent_id,
            query.similarity_threshold,
            query.limit
        )
        
        memories = []
        for row in rows:
            market_context = json.loads(row['market_context']) if row['market_context'] else {}
            metadata = {
                'symbol': row['symbol'],
                'decision': row['decision'],
                'strategy_name': row['strategy_name'],
                'market_context': market_context
            }
            
            memory = Memory(
                id=row['id'],
                agent_id=row['agent_type'],
                memory_type=MemoryType.DECISION,
                content=row['reasoning'] or f"Decision: {row['decision']} on {row['symbol']}",
                metadata=metadata,
                embedding=np.array(row['context_embedding']),
                confidence=float(row['similarity']),
                created_at=row['created_at']
            )
            memories.append(memory)
        
        return memories
    
    async def _search_knowledge_memories(
        self,
        conn: asyncpg.Connection,
        query: MemoryQuery,
        query_embedding: np.ndarray,
        memory_type: MemoryType
    ) -> List[Memory]:
        """Search knowledge/episodic memories using vector similarity"""
        
        sql = """
            SELECT id, source_type, content_text, embedding, symbols,
                   metadata, confidence_score, created_at,
                   1 - (embedding <-> $1) as similarity
            FROM market_knowledge
            WHERE source_type = $2
            AND 1 - (embedding <-> $1) >= $3
            ORDER BY embedding <-> $1
            LIMIT $4
        """
        
        rows = await conn.fetch(
            sql,
            query_embedding.tolist(),
            memory_type.value,
            query.similarity_threshold,
            query.limit
        )
        
        memories = []
        for row in rows:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            metadata['symbols'] = row['symbols']
            
            memory = Memory(
                id=row['id'],
                agent_id="system",  # Knowledge memories are system-wide
                memory_type=memory_type,
                content=row['content_text'],
                metadata=metadata,
                embedding=np.array(row['embedding']),
                confidence=float(row['similarity']),
                created_at=row['created_at']
            )
            memories.append(memory)
        
        return memories
    
    async def consolidate_memories(self, agent_id: str) -> int:
        """
        Consolidate similar memories to prevent redundancy.
        
        Args:
            agent_id: Agent to consolidate memories for
            
        Returns:
            int: Number of memories consolidated
        """
        consolidated_count = 0
        
        try:
            # Consolidate patterns
            consolidated_count += await self._consolidate_pattern_memories(agent_id)
            
            # Update stats
            self._stats["consolidations_performed"] += consolidated_count
            
            if consolidated_count > 0:
                logger.info(f"Consolidated {consolidated_count} memories for agent {agent_id}")
            
            return consolidated_count
            
        except Exception as e:
            logger.error(f"Failed to consolidate memories for agent {agent_id}: {e}")
            raise
    
    async def _consolidate_pattern_memories(self, agent_id: str) -> int:
        """Consolidate similar pattern memories"""
        
        async with self._get_db_connection() as conn:
            # Find patterns with high similarity
            sql = """
                SELECT p1.id as id1, p2.id as id2, 
                       1 - (p1.pattern_embedding <-> p2.pattern_embedding) as similarity
                FROM agent_patterns p1
                JOIN agent_patterns p2 ON p1.id < p2.id
                WHERE p1.agent_type = $1 AND p2.agent_type = $1
                AND p1.is_active = true AND p2.is_active = true
                AND 1 - (p1.pattern_embedding <-> p2.pattern_embedding) >= $2
                ORDER BY similarity DESC
            """
            
            similar_pairs = await conn.fetch(sql, agent_id, self.consolidation_threshold)
            
            consolidated_count = 0
            
            for pair in similar_pairs:
                # Merge the patterns - keep the one with better performance
                await conn.execute("""
                    UPDATE agent_patterns 
                    SET occurrence_count = occurrence_count + (
                        SELECT occurrence_count FROM agent_patterns WHERE id = $2
                    ),
                    total_profit_loss = total_profit_loss + (
                        SELECT total_profit_loss FROM agent_patterns WHERE id = $2  
                    ),
                    last_seen = NOW()
                    WHERE id = $1
                """, pair['id1'], pair['id2'])
                
                # Deactivate the merged pattern
                await conn.execute("""
                    UPDATE agent_patterns SET is_active = false WHERE id = $1
                """, pair['id2'])
                
                consolidated_count += 1
        
        return consolidated_count
    
    async def get_memory_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics for an agent or system-wide"""
        
        async with self._get_db_connection() as conn:
            agent_filter = f"WHERE agent_type = '{agent_id}'" if agent_id else ""
            
            # Pattern stats
            pattern_stats = await conn.fetchrow(f"""
                SELECT COUNT(*) as total, AVG(success_rate) as avg_success_rate
                FROM agent_patterns {agent_filter}
            """)
            
            # Decision stats
            decision_stats = await conn.fetchrow(f"""
                SELECT COUNT(*) as total, AVG(confidence) as avg_confidence
                FROM agent_decisions {agent_filter}
            """)
            
            # Knowledge stats (system-wide only)
            knowledge_stats = await conn.fetchrow("""
                SELECT COUNT(*) as total, AVG(confidence_score) as avg_confidence
                FROM market_knowledge
            """)
        
        return {
            "agent_id": agent_id or "system",
            "pattern_memories": pattern_stats['total'],
            "decision_memories": decision_stats['total'], 
            "knowledge_memories": knowledge_stats['total'],
            "avg_pattern_success_rate": float(pattern_stats['avg_success_rate'] or 0),
            "avg_decision_confidence": float(decision_stats['avg_confidence'] or 0),
            "avg_knowledge_confidence": float(knowledge_stats['avg_confidence'] or 0),
            "manager_stats": self._stats.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on memory manager"""
        
        status = "healthy"
        issues = []
        
        try:
            # Test database connectivity
            async with self._get_db_connection() as conn:
                await conn.fetchval("SELECT 1")
            
            # Test embedding service
            if not self.embedding_service:
                status = "degraded"
                issues.append("embedding_service_not_initialized")
            else:
                test_result = await self.embedding_service.embed_text("test")
                if test_result.embedding.shape[0] != 1536:
                    status = "degraded"
                    issues.append("embedding_dimension_mismatch")
            
            # Test cache
            await self.memory_cache.set("health_check", "test")
            cached_value = await self.memory_cache.get("health_check")
            if cached_value != "test":
                status = "degraded"
                issues.append("cache_not_working")
                
        except Exception as e:
            status = "critical"
            issues.append(f"health_check_error: {str(e)}")
        
        return {
            "status": status,
            "issues": issues,
            "stats": self._stats.copy(),
            "cache_size": len(self.memory_cache)
        }