"""
Mock Embedding Service for testing and ML-disabled environments
No ML dependencies - can be used when torch is not available
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from app.rag.services.embedding_base import EmbeddingResult, EmbeddingServiceInterface

logger = logging.getLogger(__name__)


class MockEmbeddingService(EmbeddingServiceInterface):
    """
    Mock embedding service that returns deterministic fake embeddings
    Useful for testing and when ML features are disabled
    """

    def __init__(
        self,
        model_name: str = "mock-embedding-service",
        target_dim: int = 1536,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
    ):
        self.model_name = model_name
        self.target_dim = target_dim
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.model_version = "mock-v1.0"

        # Simple in-memory cache
        self.cache = {}

        # Performance stats
        self.stats = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "avg_processing_time": 0.001,  # Very fast mock processing
            "memory_usage_mb": 1,  # Minimal memory usage
        }

        logger.info(f"MockEmbeddingService initialized with target_dim: {target_dim}")

    async def initialize(self) -> None:
        """Initialize the mock service (no-op)"""
        logger.info("MockEmbeddingService ready - no initialization required")

    async def embed_text(self, text: str) -> Optional[EmbeddingResult]:
        """Generate a single mock embedding"""
        results = await self.embed_texts([text])
        return results[0] if results else None

    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate mock embeddings for multiple texts"""
        if not texts:
            return []

        start_time = datetime.now()
        results = []

        for text in texts:
            cache_key = self._get_cache_key(text)

            # Check cache
            if cache_key in self.cache:
                embedding = self.cache[cache_key]
                cache_hit = True
                self.stats["cache_hits"] += 1
            else:
                # Generate deterministic mock embedding
                embedding = self._generate_mock_embedding(text)
                self.cache[cache_key] = embedding
                cache_hit = False

                # Clean cache if too large
                if len(self.cache) > self.cache_size:
                    # Remove oldest entries (simple FIFO)
                    keys_to_remove = list(self.cache.keys())[
                        : len(self.cache) - self.cache_size + 1
                    ]
                    for k in keys_to_remove:
                        del self.cache[k]

            # Calculate mock confidence based on text length
            confidence = min(1.0, len(text) / 100.0)

            result = EmbeddingResult(
                text=text,
                embedding=embedding,
                model_version=self.model_version,
                processing_time=0.001,  # Mock fast processing
                confidence=confidence,
                cache_hit=cache_hit,
                dimension=self.target_dim,
            )
            results.append(result)

        # Update stats
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats["total_embeddings"] += len(texts)

        logger.debug(
            f"Generated {len(texts)} mock embeddings in {processing_time:.3f}s "
            f"({self.stats['cache_hits']}/{len(texts)} cache hits)"
        )

        return results

    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding based on text content"""
        # Use hash of text as seed for reproducible results
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)

        # Generate deterministic random embedding
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, self.target_dim).astype(np.float32)

        # Normalize to unit vector (common for embeddings)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{text}:{self.model_version}".encode()).hexdigest()

    async def health_check(self) -> Dict[str, Any]:
        """Return mock health status"""
        return {
            "status": "healthy",
            "service_type": "mock",
            "model_name": self.model_name,
            "model_loaded": True,  # Always "loaded" for mock
            "device": "cpu",  # Mock always uses CPU
            "target_dimensions": self.target_dim,
            "actual_dimensions": self.target_dim,
            "cache_size": len(self.cache),
            "stats": self.stats.copy(),
            "memory_usage_mb": self.stats["memory_usage_mb"],
        }

    def clear_cache(self) -> int:
        """Clear mock cache"""
        cleared_count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {cleared_count} items from mock embedding cache")
        return cleared_count

    def get_stats(self) -> Dict[str, Any]:
        """Get mock performance statistics"""
        cache_hit_rate = 0.0
        if self.stats["total_embeddings"] > 0:
            cache_hit_rate = (
                self.stats["cache_hits"] / self.stats["total_embeddings"]
            ) * 100

        return {
            "service_type": "mock",
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "total_embeddings": self.stats["total_embeddings"],
            "cache_hits": self.stats["cache_hits"],
            "avg_processing_time": f"{self.stats['avg_processing_time']:.3f}s",
            "memory_usage_mb": self.stats["memory_usage_mb"],
            "cache_size": len(self.cache),
            "device": "cpu",
            "model_name": self.model_name,
        }
