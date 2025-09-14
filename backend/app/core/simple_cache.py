"""
Simple, testable cache implementation
"""

import asyncio
import time
from typing import Any, Dict, Optional
from cachetools import TTLCache as MemoryCache

from app.core.cache_metrics import CacheMetrics
from app.core.redis_client import RedisClient


class SimpleCache:
    """Simple two-level cache (memory + Redis)"""

    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        ttl: int = 3600,
        redis_url: Optional[str] = None
    ):
        self.name = name
        self.max_size = max_size
        self.ttl = ttl

        # L1 Cache (memory)
        self._memory_cache = MemoryCache(maxsize=max_size, ttl=ttl)

        # L2 Cache (Redis)
        self._redis_client = RedisClient(redis_url) if redis_url else None
        self._redis_connected = False

        # Metrics
        self.metrics = CacheMetrics()

    async def initialize(self) -> bool:
        """Initialize Redis connection if configured"""
        if self._redis_client:
            self._redis_connected = await self._redis_client.connect()
        return True

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 then L2)"""
        start_time = time.time()

        # Try L1 cache first
        if key in self._memory_cache:
            value = self._memory_cache[key]
            response_time = time.time() - start_time
            self.metrics.record_l1_hit(response_time)
            return value

        # Try L2 cache (Redis)
        if self._redis_connected and self._redis_client:
            redis_key = f"{self.name}:{key}"
            value = await self._redis_client.get(redis_key)
            if value is not None:
                # Store in L1 cache
                self._memory_cache[key] = value
                self.metrics.record_l1_set()

                response_time = time.time() - start_time
                self.metrics.record_l2_hit(response_time)
                return value

        # Cache miss
        self.metrics.record_miss()
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in both cache levels"""
        effective_ttl = ttl or self.ttl

        try:
            # Set in L1 cache
            self._memory_cache[key] = value
            self.metrics.record_l1_set()

            # Set in L2 cache (Redis) if available
            if self._redis_connected and self._redis_client:
                redis_key = f"{self.name}:{key}"
                await self._redis_client.set(redis_key, value, effective_ttl)
                self.metrics.record_l2_set()

            return True

        except Exception as e:
            self.metrics.record_error()
            return False

    async def delete(self, key: str) -> bool:
        """Delete from both cache levels"""
        try:
            # Remove from L1
            self._memory_cache.pop(key, None)

            # Remove from L2
            if self._redis_connected and self._redis_client:
                redis_key = f"{self.name}:{key}"
                await self._redis_client.delete(redis_key)

            return True

        except Exception as e:
            self.metrics.record_error()
            return False

    async def clear(self) -> bool:
        """Clear all cached data"""
        try:
            self._memory_cache.clear()
            return True
        except Exception as e:
            self.metrics.record_error()
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "name": self.name,
            "l1_size": len(self._memory_cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "redis_connected": self._redis_connected,
            **self.metrics.get_summary()
        }

    async def close(self):
        """Close cache and connections"""
        if self._redis_client:
            await self._redis_client.disconnect()


class CacheFactory:
    """Factory for creating cache instances"""

    _instances: Dict[str, SimpleCache] = {}

    @classmethod
    def create_cache(
        self,
        name: str,
        max_size: int = 1000,
        ttl: int = 3600,
        redis_url: Optional[str] = None
    ) -> SimpleCache:
        """Create or get existing cache instance"""
        if name not in self._instances:
            self._instances[name] = SimpleCache(
                name=name,
                max_size=max_size,
                ttl=ttl,
                redis_url=redis_url
            )
        return self._instances[name]

    @classmethod
    def get_cache(self, name: str) -> Optional[SimpleCache]:
        """Get existing cache instance"""
        return self._instances.get(name)

    @classmethod
    async def initialize_all(self) -> bool:
        """Initialize all cache instances"""
        for cache in self._instances.values():
            await cache.initialize()
        return True

    @classmethod
    async def close_all(self):
        """Close all cache instances"""
        for cache in self._instances.values():
            await cache.close()
        self._instances.clear()