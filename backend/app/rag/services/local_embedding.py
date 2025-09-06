"""
Local Embedding Service for M1 MacBook with SuperBPE compatibility
Optimized for 8GB RAM with 1536-dimensional embeddings
"""
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from datetime import datetime
import logging
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from cachetools import TTLCache

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    text: str
    embedding: np.ndarray
    model_version: str
    processing_time: float
    confidence: float
    cache_hit: bool = False
    dimension: int = 1536

class LocalEmbeddingService:
    """
    M1-optimized local embedding service with SuperBPE compatibility
    Pads smaller embeddings to 1536 dimensions for consistency
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        use_mps: bool = True,
        batch_size: int = 8  # Small batch size for 8GB RAM
    ):
        self.model_name = model_name
        self.target_dim = 1536  # SuperBPE compatibility
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size
        
        # Detect M1 GPU availability
        self.device = self._get_optimal_device(use_mps)
        
        # Initialize model
        self.model = None
        self.actual_dim = None
        self.model_version = "local-superbpe-v1.0"
        
        # Caching for performance
        self.embedding_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        
        # Thread pool for CPU fallback
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance stats
        self.stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'mps_embeddings': 0,
            'cpu_embeddings': 0,
            'avg_processing_time': 0.0,
            'memory_usage_mb': 0
        }
        
        logger.info(f"LocalEmbeddingService initialized with device: {self.device}")
    
    def _get_optimal_device(self, use_mps: bool) -> str:
        """Determine the best device for M1 Mac"""
        if use_mps and torch.backends.mps.is_available():
            return "mps"  # M1 GPU
        elif torch.cuda.is_available():
            return "cuda"  # Nvidia GPU
        else:
            return "cpu"
    
    async def initialize(self) -> None:
        """Initialize the embedding model asynchronously"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load model in thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                self._load_model
            )
            
            # Test embedding to get actual dimensions
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)[0]
            self.actual_dim = test_embedding.shape[0]
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Actual dimensions: {self.actual_dim}")
            logger.info(f"   Target dimensions: {self.target_dim}")
            logger.info(f"   Padding required: {self.actual_dim < self.target_dim}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model"""
        model = SentenceTransformer(self.model_name, device=self.device)
        
        # Configure for M1 optimization
        if self.device == "mps":
            # Enable MPS optimizations
            torch.backends.mps.enable_fallback(True)
        
        return model
    
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text"""
        results = await self.embed_texts([text])
        return results[0] if results else None
    
    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts with caching and batching
        """
        if not texts:
            return []
        
        if self.model is None:
            await self.initialize()
        
        start_time = datetime.now()
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                cached_embedding = self.embedding_cache[cache_key]
                results.append(EmbeddingResult(
                    text=text,
                    embedding=cached_embedding,
                    model_version=self.model_version,
                    processing_time=0.0,
                    confidence=1.0,  # Cached results have high confidence
                    cache_hit=True,
                    dimension=self.target_dim
                ))
                self.stats['cache_hits'] += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append(None)  # Placeholder
        
        # Process uncached texts in batches
        if uncached_texts:
            uncached_results = await self._process_uncached_texts(uncached_texts)
            
            # Fill in the results and cache them
            for i, result in enumerate(uncached_results):
                cache_key = self._get_cache_key(result.text)
                self.embedding_cache[cache_key] = result.embedding
                results[uncached_indices[i]] = result
        
        # Update stats
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats['total_embeddings'] += len(texts)
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (self.stats['total_embeddings'] - len(texts)) + processing_time) 
            / self.stats['total_embeddings']
        )
        
        logger.debug(f"Processed {len(texts)} embeddings in {processing_time:.3f}s "
                    f"({self.stats['cache_hits']}/{len(texts)} cache hits)")
        
        return [r for r in results if r is not None]
    
    async def _process_uncached_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Process texts that aren't in cache"""
        all_results = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_start = datetime.now()
            
            try:
                # Use thread pool for CPU-bound embedding generation
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    self.executor,
                    self._generate_batch_embeddings,
                    batch
                )
                
                batch_time = (datetime.now() - batch_start).total_seconds()
                
                # Create results with padding to 1536 dimensions
                for j, embedding in enumerate(embeddings):
                    padded_embedding = self._pad_to_superbpe_dims(embedding)
                    confidence = self._calculate_confidence(embedding)
                    
                    result = EmbeddingResult(
                        text=batch[j],
                        embedding=padded_embedding,
                        model_version=self.model_version,
                        processing_time=batch_time / len(batch),
                        confidence=confidence,
                        cache_hit=False,
                        dimension=self.target_dim
                    )
                    all_results.append(result)
                
                # Update device stats
                if self.device == "mps":
                    self.stats['mps_embeddings'] += len(batch)
                else:
                    self.stats['cpu_embeddings'] += len(batch)
                    
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Create error results
                for text in batch:
                    error_embedding = np.zeros(self.target_dim, dtype=np.float32)
                    all_results.append(EmbeddingResult(
                        text=text,
                        embedding=error_embedding,
                        model_version=f"{self.model_version}-error",
                        processing_time=0.0,
                        confidence=0.0,
                        cache_hit=False,
                        dimension=self.target_dim
                    ))
        
        return all_results
    
    def _generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=len(texts)  # Process entire batch at once
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.actual_dim or 384), dtype=np.float32)
    
    def _pad_to_superbpe_dims(self, embedding: np.ndarray) -> np.ndarray:
        """
        Pad embedding to 1536 dimensions for SuperBPE compatibility
        Uses intelligent padding strategies
        """
        if embedding.shape[0] >= self.target_dim:
            return embedding[:self.target_dim]
        
        padded = np.zeros(self.target_dim, dtype=np.float32)
        
        # Copy original embedding
        padded[:embedding.shape[0]] = embedding
        
        # Use learned padding (repeat pattern for better representation)
        if embedding.shape[0] > 0:
            remaining_dims = self.target_dim - embedding.shape[0]
            repeat_count = remaining_dims // embedding.shape[0]
            remainder = remaining_dims % embedding.shape[0]
            
            current_pos = embedding.shape[0]
            
            # Repeat the embedding with diminishing weights
            for i in range(repeat_count):
                weight = 0.1 / (i + 1)  # Diminishing importance
                end_pos = current_pos + embedding.shape[0]
                padded[current_pos:end_pos] = embedding * weight
                current_pos = end_pos
            
            # Handle remainder
            if remainder > 0:
                weight = 0.05
                padded[current_pos:current_pos + remainder] = embedding[:remainder] * weight
        
        return padded
    
    def _calculate_confidence(self, embedding: np.ndarray) -> float:
        """Calculate confidence based on embedding properties"""
        if len(embedding) == 0:
            return 0.0
        
        # Use L2 norm as a confidence measure
        norm = np.linalg.norm(embedding)
        
        # Normalize to 0-1 range
        confidence = min(1.0, norm / 2.0)
        
        return float(confidence)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{text}:{self.model_version}".encode()).hexdigest()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health and performance"""
        try:
            # Test embedding generation
            test_result = await self.embed_text("Health check test")
            
            # Get memory usage (approximate)
            memory_usage = 0
            if self.model is not None:
                try:
                    # Estimate model memory usage
                    if hasattr(self.model, 'get_max_seq_length'):
                        memory_usage = 500  # Approximate MB for sentence transformers
                except:
                    memory_usage = 0
            
            self.stats['memory_usage_mb'] = memory_usage
            
            return {
                'status': 'healthy',
                'model_name': self.model_name,
                'model_loaded': self.model is not None,
                'device': self.device,
                'target_dimensions': self.target_dim,
                'actual_dimensions': self.actual_dim,
                'cache_size': len(self.embedding_cache),
                'stats': self.stats.copy(),
                'test_embedding_shape': test_result.embedding.shape if test_result else None,
                'test_confidence': test_result.confidence if test_result else 0.0,
                'memory_usage_mb': memory_usage
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': self.model is not None,
                'device': self.device
            }
    
    def clear_cache(self) -> int:
        """Clear embedding cache and return number of items cleared"""
        cleared_count = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"Cleared {cleared_count} items from embedding cache")
        return cleared_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        cache_hit_rate = 0.0
        if self.stats['total_embeddings'] > 0:
            cache_hit_rate = (self.stats['cache_hits'] / self.stats['total_embeddings']) * 100
        
        return {
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'total_embeddings': self.stats['total_embeddings'],
            'cache_hits': self.stats['cache_hits'],
            'mps_embeddings': self.stats['mps_embeddings'],
            'cpu_embeddings': self.stats['cpu_embeddings'],
            'avg_processing_time': f"{self.stats['avg_processing_time']:.3f}s",
            'memory_usage_mb': self.stats['memory_usage_mb'],
            'cache_size': len(self.embedding_cache),
            'device': self.device
        }

# Singleton instance for global use
_embedding_service: Optional[LocalEmbeddingService] = None

async def get_embedding_service() -> LocalEmbeddingService:
    """Get singleton embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        use_mps = os.getenv('USE_MPS_DEVICE', 'true').lower() == 'true'
        
        _embedding_service = LocalEmbeddingService(
            model_name=model_name,
            use_mps=use_mps
        )
        await _embedding_service.initialize()
    
    return _embedding_service

# Test function
async def test_embedding_service():
    """Test the embedding service"""
    print("🧪 Testing Local Embedding Service...")
    
    service = await get_embedding_service()
    
    # Test single embedding
    result = await service.embed_text("This is a test for market analysis")
    print(f"✅ Single embedding: {result.embedding.shape}, confidence: {result.confidence:.3f}")
    
    # Test batch embeddings
    texts = [
        "Bullish market trend detected",
        "Bearish signal from technical indicators",
        "Market consolidation phase",
        "High volume breakout pattern"
    ]
    
    batch_results = await service.embed_texts(texts)
    print(f"✅ Batch embeddings: {len(batch_results)} results")
    
    for i, result in enumerate(batch_results):
        print(f"   {i+1}. Shape: {result.embedding.shape}, Cache hit: {result.cache_hit}")
    
    # Test cache effectiveness
    cached_results = await service.embed_texts(texts[:2])  # Repeat some texts
    cache_hits = sum(1 for r in cached_results if r.cache_hit)
    print(f"✅ Cache test: {cache_hits}/{len(cached_results)} cache hits")
    
    # Health check
    health = await service.health_check()
    print(f"✅ Health check: {health['status']}")
    print(f"   Device: {health['device']}")
    print(f"   Dimensions: {health['actual_dimensions']} → {health['target_dimensions']}")
    
    # Stats
    stats = service.get_stats()
    print(f"✅ Performance stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_embedding_service())