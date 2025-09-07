"""
Enhanced caching infrastructure with Redis backend support.
Extends existing TTLCache patterns with two-tier caching (L1: memory, L2: Redis).
"""

import asyncio
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import asynccontextmanager

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from cachetools import TTLCache
import structlog

from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class CacheMetrics:
    """Track cache performance metrics"""
    
    def __init__(self):
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        self.l1_sets = 0
        self.l2_sets = 0
        self.errors = 0
        self.total_requests = 0
        self.avg_l1_time = 0.0
        self.avg_l2_time = 0.0
        self.start_time = datetime.now()

    def record_l1_hit(self, response_time: float):
        self.l1_hits += 1
        self.total_requests += 1
        self._update_avg_time('l1', response_time)
    
    def record_l2_hit(self, response_time: float):
        self.l2_hits += 1
        self.total_requests += 1
        self._update_avg_time('l2', response_time)
    
    def record_miss(self):
        self.misses += 1
        self.total_requests += 1
    
    def record_l1_set(self):
        self.l1_sets += 1
    
    def record_l2_set(self):
        self.l2_sets += 1
    
    def record_error(self):
        self.errors += 1
    
    def _update_avg_time(self, cache_type: str, response_time: float):
        if cache_type == 'l1':
            self.avg_l1_time = (self.avg_l1_time * (self.l1_hits - 1) + response_time) / self.l1_hits
        else:
            self.avg_l2_time = (self.avg_l2_time * (self.l2_hits - 1) + response_time) / self.l2_hits
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.l1_hits + self.l2_hits) / self.total_requests * 100
    
    @property
    def l1_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.l1_hits / self.total_requests * 100
    
    @property
    def l2_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.l2_hits / self.total_requests * 100


class EnhancedTTLCache:
    """
    Two-tier cache system extending TTLCache functionality with Redis backend.
    L1: In-memory TTLCache for hot data
    L2: Redis for persistent and shared caching
    """
    
    def __init__(
        self,
        l1_maxsize: int = 10000,
        l1_ttl: int = 3600,
        l2_ttl: int = 7200,
        redis_url: Optional[str] = None,
        key_prefix: str = "cache",
        enable_warming: bool = True,
        warming_threshold: int = 100
    ):
        self.l1_maxsize = l1_maxsize
        self.l1_ttl = l1_ttl
        self.l2_ttl = l2_ttl
        self.key_prefix = key_prefix
        self.enable_warming = enable_warming
        self.warming_threshold = warming_threshold
        
        # L1 Cache (in-memory TTL)
        self.l1_cache = TTLCache(maxsize=l1_maxsize, ttl=l1_ttl)
        
        # L2 Cache (Redis) - only if available
        self.redis_available = REDIS_AVAILABLE
        self.redis_client = None
        
        if REDIS_AVAILABLE:
            try:
                redis_url = redis_url or getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
            except Exception as e:
                logger.warning("Failed to initialize Redis client, falling back to L1 only", error=str(e))
                self.redis_available = False
        else:
            logger.warning("Redis not available, using L1 cache only")
        
        # Cache metrics
        self.metrics = CacheMetrics()
        
        # Warming task tracking
        self._warming_keys: set = set()
        self._warming_lock = asyncio.Lock()
        
        logger.info(
            "EnhancedTTLCache initialized",
            l1_maxsize=l1_maxsize,
            l1_ttl=l1_ttl,
            l2_ttl=l2_ttl,
            key_prefix=key_prefix,
            warming_enabled=enable_warming
        )
    
    def _make_key(self, key: str) -> str:
        """Create Redis key with namespace prefix"""
        return f"{self.key_prefix}:{key}"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with L1 -> L2 fallback"""
        start_time = datetime.now()
        
        try:
            # Check L1 cache first (fastest)
            if key in self.l1_cache:
                response_time = (datetime.now() - start_time).total_seconds()
                self.metrics.record_l1_hit(response_time)
                value = self.l1_cache[key]
                logger.debug("Cache L1 hit", key=key, response_time=response_time)
                
                # Trigger warming if needed
                if self.enable_warming:
                    asyncio.create_task(self._maybe_warm_key(key))
                
                return value
            
            # Check L2 cache (Redis) if available
            if self.redis_available and self.redis_client:
                redis_key = self._make_key(key)
                cached_data = await self.redis_client.get(redis_key)
            else:
                cached_data = None
            
            if cached_data is not None:
                # Deserialize from Redis
                try:
                    value = pickle.loads(cached_data)
                    response_time = (datetime.now() - start_time).total_seconds()
                    self.metrics.record_l2_hit(response_time)
                    
                    # Promote to L1 cache
                    self.l1_cache[key] = value
                    self.metrics.record_l1_set()
                    
                    logger.debug("Cache L2 hit and promoted", key=key, response_time=response_time)
                    return value
                
                except (pickle.UnpicklingError, Exception) as e:
                    logger.warning("Failed to deserialize cached value", key=key, error=str(e))
                    # Remove corrupted key
                    await self.redis_client.delete(redis_key)
            
            # Cache miss
            self.metrics.record_miss()
            logger.debug("Cache miss", key=key)
            return default
            
        except Exception as e:
            self.metrics.record_error()
            logger.error("Cache get error", key=key, error=str(e))
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_override: Optional[int] = None,
        l1_only: bool = False
    ) -> bool:
        """Set value in both L1 and L2 caches"""
        try:
            # Set in L1 cache
            self.l1_cache[key] = value
            self.metrics.record_l1_set()
            
            if not l1_only and self.redis_available and self.redis_client:
                # Set in L2 cache (Redis)
                redis_key = self._make_key(key)
                serialized = pickle.dumps(value)
                ttl = ttl_override or self.l2_ttl
                
                await self.redis_client.setex(redis_key, ttl, serialized)
                self.metrics.record_l2_set()
            
            logger.debug("Cache set", key=key, l1_only=l1_only)
            return True
            
        except Exception as e:
            self.metrics.record_error()
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from both caches"""
        try:
            success = True
            
            # Delete from L1
            if key in self.l1_cache:
                del self.l1_cache[key]
            
            # Delete from L2 if available
            deleted = 0
            if self.redis_available and self.redis_client:
                redis_key = self._make_key(key)
                deleted = await self.redis_client.delete(redis_key)
            
            logger.debug("Cache delete", key=key, redis_deleted=deleted > 0)
            return success
            
        except Exception as e:
            self.metrics.record_error()
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries"""
        try:
            cleared_count = 0
            
            # Clear L1
            l1_count = len(self.l1_cache)
            self.l1_cache.clear()
            cleared_count += l1_count
            
            # Clear L2 with pattern if available
            if self.redis_available and self.redis_client:
                if pattern:
                    redis_pattern = self._make_key(pattern)
                    keys = await self.redis_client.keys(redis_pattern)
                    if keys:
                        deleted = await self.redis_client.delete(*keys)
                        cleared_count += deleted
                else:
                    # Clear all keys with our prefix
                    redis_pattern = f"{self.key_prefix}:*"
                    keys = await self.redis_client.keys(redis_pattern)
                    if keys:
                        deleted = await self.redis_client.delete(*keys)
                        cleared_count += deleted
            
            logger.info("Cache cleared", pattern=pattern, cleared_count=cleared_count)
            return cleared_count
            
        except Exception as e:
            self.metrics.record_error()
            logger.error("Cache clear error", pattern=pattern, error=str(e))
            return 0
    
    async def _maybe_warm_key(self, key: str):
        """Potentially warm cache key if it's popular"""
        async with self._warming_lock:
            if key not in self._warming_keys and self.metrics.l1_hits > self.warming_threshold:
                self._warming_keys.add(key)
                logger.debug("Added key to warming set", key=key)
    
    async def warm_cache(self, warming_func: Callable, keys: List[str]) -> int:
        """Warm cache with provided keys using warming function"""
        if not self.enable_warming:
            return 0
        
        warmed_count = 0
        
        try:
            for key in keys:
                if key not in self.l1_cache:
                    try:
                        value = await warming_func(key)
                        if value is not None:
                            await self.set(key, value)
                            warmed_count += 1
                    except Exception as e:
                        logger.warning("Failed to warm key", key=key, error=str(e))
            
            logger.info("Cache warming completed", warmed_count=warmed_count, total_keys=len(keys))
            
        except Exception as e:
            logger.error("Cache warming error", error=str(e))
        
        return warmed_count
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache health and return metrics"""
        try:
            # Test L1 cache
            test_key = f"health_check_{datetime.now().timestamp()}"
            self.l1_cache[test_key] = "test"
            l1_working = test_key in self.l1_cache
            
            # Test L2 cache (Redis) if available
            l2_working = False
            redis_info = {}
            
            if self.redis_available and self.redis_client:
                try:
                    redis_test_key = self._make_key("health_test")
                    await self.redis_client.set(redis_test_key, "test", ex=10)
                    l2_working = await self.redis_client.get(redis_test_key) == b"test"
                    await self.redis_client.delete(redis_test_key)
                    
                    # Get Redis info
                    redis_info = await self.redis_client.info()
                except Exception as e:
                    logger.warning("Redis health check failed", error=str(e))
                    l2_working = False
            
            # Status is healthy if L1 is working, degraded if only L1, unhealthy if neither
            status = "unhealthy"
            if l1_working:
                if l2_working or not self.redis_available:
                    status = "healthy"
                else:
                    status = "degraded"
                    
            return {
                "status": status,
                "l1_cache": {
                    "working": l1_working,
                    "size": len(self.l1_cache),
                    "maxsize": self.l1_maxsize,
                    "ttl": self.l1_ttl
                },
                "l2_cache": {
                    "available": self.redis_available,
                    "working": l2_working,
                    "connected": redis_info.get('connected_clients', 0) > 0 if redis_info else False,
                    "memory_usage": redis_info.get('used_memory_human', 'unavailable') if redis_info else 'unavailable',
                    "ttl": self.l2_ttl
                },
                "metrics": {
                    "hit_rate": f"{self.metrics.hit_rate:.1f}%",
                    "l1_hit_rate": f"{self.metrics.l1_hit_rate:.1f}%",
                    "l2_hit_rate": f"{self.metrics.l2_hit_rate:.1f}%",
                    "total_requests": self.metrics.total_requests,
                    "errors": self.metrics.errors,
                    "avg_l1_time": f"{self.metrics.avg_l1_time:.3f}s",
                    "avg_l2_time": f"{self.metrics.avg_l2_time:.3f}s"
                }
            }
            
        except Exception as e:
            logger.error("Cache health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "l1_cache": {"working": False},
                "l2_cache": {"working": False}
            }
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.redis_available and self.redis_client:
                await self.redis_client.close()
            logger.info("Cache connections closed")
        except Exception as e:
            logger.error("Error closing cache", error=str(e))


# Global cache instances
_embedding_cache: Optional[EnhancedTTLCache] = None
_market_cache: Optional[EnhancedTTLCache] = None


def get_embedding_cache() -> EnhancedTTLCache:
    """Get or create embedding cache instance"""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EnhancedTTLCache(
            l1_maxsize=10000,
            l1_ttl=3600,
            l2_ttl=7200,
            key_prefix="embedding",
            enable_warming=True,
            warming_threshold=100
        )
    return _embedding_cache


def get_market_cache() -> EnhancedTTLCache:
    """Get or create market research cache instance"""
    global _market_cache
    if _market_cache is None:
        _market_cache = EnhancedTTLCache(
            l1_maxsize=5000,
            l1_ttl=900,  # 15 minutes
            l2_ttl=1800,  # 30 minutes
            key_prefix="market",
            enable_warming=True,
            warming_threshold=50
        )
    return _market_cache


@asynccontextmanager
async def cache_context():
    """Context manager for cache lifecycle"""
    try:
        yield
    finally:
        # Clean up cache connections
        if _embedding_cache:
            await _embedding_cache.close()
        if _market_cache:
            await _market_cache.close()