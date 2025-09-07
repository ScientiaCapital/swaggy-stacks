"""
Abstract base interface for embedding services
No ML dependencies - can be imported anywhere
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EmbeddingResult:
    """Standard embedding result format"""
    text: str
    embedding: np.ndarray
    model_version: str
    processing_time: float
    confidence: float
    cache_hit: bool = False
    dimension: int = 1536


class EmbeddingServiceInterface(ABC):
    """Abstract interface for all embedding services"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding service"""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> Optional[EmbeddingResult]:
        """Generate embedding for a single text"""
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health and performance"""
        pass

    @abstractmethod
    def clear_cache(self) -> int:
        """Clear embedding cache and return number of items cleared"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        pass