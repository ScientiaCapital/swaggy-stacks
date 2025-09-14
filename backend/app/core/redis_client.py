"""
Simple Redis client abstraction
"""

import pickle
from typing import Any, Optional

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

import structlog

logger = structlog.get_logger(__name__)


class RedisClient:
    """Simple Redis client wrapper"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._client: Optional[Any] = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - operations will be no-ops")
            return False

        try:
            self._client = redis.from_url(self.redis_url, decode_responses=False)
            await self._client.ping()
            self._connected = True
            logger.info("Connected to Redis", redis_url=self.redis_url)
            return True
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e), redis_url=self.redis_url)
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Redis"""
        if self._client and self._connected:
            try:
                await self._client.close()
                self._connected = False
                logger.info("Disconnected from Redis")
            except Exception as e:
                logger.error("Error disconnecting from Redis", error=str(e))

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        if not self._connected or not self._client:
            return None

        try:
            data = await self._client.get(key)
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.error("Redis get error", error=str(e), key=key)
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis"""
        if not self._connected or not self._client:
            return False

        try:
            data = pickle.dumps(value)
            if ttl:
                await self._client.setex(key, ttl, data)
            else:
                await self._client.set(key, data)
            return True
        except Exception as e:
            logger.error("Redis set error", error=str(e), key=key)
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self._connected or not self._client:
            return False

        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.error("Redis delete error", error=str(e), key=key)
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self._connected or not self._client:
            return False

        try:
            result = await self._client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error("Redis exists error", error=str(e), key=key)
            return False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self._connected