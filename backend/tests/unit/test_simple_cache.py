"""
Test simple cache implementation
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from app.core.simple_cache import SimpleCache, CacheFactory


class TestSimpleCache:
    """Test SimpleCache functionality"""

    @pytest.fixture
    def cache(self):
        """Create a simple cache instance for testing"""
        return SimpleCache(name="test_cache", max_size=100, ttl=300)

    def test_cache_initialization(self, cache):
        """Test cache initialization"""
        assert cache.name == "test_cache"
        assert cache.max_size == 100
        assert cache.ttl == 300
        assert cache._redis_client is None
        assert not cache._redis_connected
        assert cache.metrics is not None

    def test_cache_with_redis_url(self):
        """Test cache initialization with Redis URL"""
        cache = SimpleCache(
            name="redis_cache",
            max_size=200,
            ttl=600,
            redis_url="redis://localhost:6379"
        )

        assert cache.name == "redis_cache"
        assert cache._redis_client is not None

    @pytest.mark.asyncio
    async def test_initialize_without_redis(self, cache):
        """Test initialization without Redis"""
        result = await cache.initialize()
        assert result is True
        assert not cache._redis_connected

    @pytest.mark.asyncio
    async def test_initialize_with_redis_success(self):
        """Test successful Redis initialization"""
        cache = SimpleCache("test", redis_url="redis://localhost:6379")

        # Mock successful Redis connection
        with patch.object(cache._redis_client, 'connect', return_value=True) as mock_connect:
            result = await cache.initialize()

            assert result is True
            assert cache._redis_connected is True
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_memory_cache_hit(self, cache):
        """Test L1 (memory) cache hit"""
        # Pre-populate memory cache
        cache._memory_cache["test_key"] = "test_value"

        result = await cache.get("test_key")

        assert result == "test_value"
        assert cache.metrics.l1_hits == 1
        assert cache.metrics.total_requests == 1

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache):
        """Test cache miss"""
        result = await cache.get("nonexistent_key")

        assert result is None
        assert cache.metrics.misses == 1
        assert cache.metrics.total_requests == 1

    @pytest.mark.asyncio
    async def test_get_redis_cache_hit(self):
        """Test L2 (Redis) cache hit"""
        cache = SimpleCache("test", redis_url="redis://localhost:6379")
        cache._redis_connected = True

        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "redis_value"
        cache._redis_client = mock_redis

        result = await cache.get("test_key")

        assert result == "redis_value"
        assert cache.metrics.l2_hits == 1
        assert cache.metrics.l1_sets == 1  # Should set in L1 cache too
        mock_redis.get.assert_called_once_with("test:test_key")

    @pytest.mark.asyncio
    async def test_set_memory_only(self, cache):
        """Test setting value in memory cache only"""
        result = await cache.set("key1", "value1")

        assert result is True
        assert cache._memory_cache["key1"] == "value1"
        assert cache.metrics.l1_sets == 1

    @pytest.mark.asyncio
    async def test_set_with_redis(self):
        """Test setting value in both caches"""
        cache = SimpleCache("test", redis_url="redis://localhost:6379")
        cache._redis_connected = True

        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        cache._redis_client = mock_redis

        result = await cache.set("key1", "value1", ttl=600)

        assert result is True
        assert cache._memory_cache["key1"] == "value1"
        assert cache.metrics.l1_sets == 1
        assert cache.metrics.l2_sets == 1
        mock_redis.set.assert_called_once_with("test:key1", "value1", 600)

    @pytest.mark.asyncio
    async def test_delete_memory_only(self, cache):
        """Test deleting from memory cache only"""
        # Pre-populate
        cache._memory_cache["key1"] = "value1"

        result = await cache.delete("key1")

        assert result is True
        assert "key1" not in cache._memory_cache

    @pytest.mark.asyncio
    async def test_delete_with_redis(self):
        """Test deleting from both caches"""
        cache = SimpleCache("test", redis_url="redis://localhost:6379")
        cache._redis_connected = True

        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.delete.return_value = True
        cache._redis_client = mock_redis

        # Pre-populate memory cache
        cache._memory_cache["key1"] = "value1"

        result = await cache.delete("key1")

        assert result is True
        assert "key1" not in cache._memory_cache
        mock_redis.delete.assert_called_once_with("test:key1")

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clearing cache"""
        # Pre-populate
        cache._memory_cache["key1"] = "value1"
        cache._memory_cache["key2"] = "value2"

        result = await cache.clear()

        assert result is True
        assert len(cache._memory_cache) == 0

    def test_get_stats(self, cache):
        """Test getting cache statistics"""
        # Add some metrics
        cache.metrics.record_l1_hit(0.05)
        cache.metrics.record_miss()

        stats = cache.get_stats()

        assert isinstance(stats, dict)
        assert stats["name"] == "test_cache"
        assert stats["l1_size"] == 0
        assert stats["max_size"] == 100
        assert stats["ttl"] == 300
        assert not stats["redis_connected"]
        assert stats["total_requests"] == 2
        assert stats["l1_hits"] == 1
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_close_without_redis(self, cache):
        """Test closing cache without Redis"""
        await cache.close()  # Should not raise exception

    @pytest.mark.asyncio
    async def test_close_with_redis(self):
        """Test closing cache with Redis"""
        cache = SimpleCache("test", redis_url="redis://localhost:6379")

        # Mock Redis client
        mock_redis = AsyncMock()
        cache._redis_client = mock_redis

        await cache.close()

        mock_redis.disconnect.assert_called_once()


class TestCacheFactory:
    """Test CacheFactory functionality"""

    def test_create_cache(self):
        """Test creating cache through factory"""
        # Clear any existing instances
        CacheFactory._instances.clear()

        cache = CacheFactory.create_cache(
            name="factory_cache",
            max_size=500,
            ttl=1200
        )

        assert isinstance(cache, SimpleCache)
        assert cache.name == "factory_cache"
        assert cache.max_size == 500
        assert cache.ttl == 1200

    def test_create_cache_singleton(self):
        """Test factory returns same instance for same name"""
        # Clear any existing instances
        CacheFactory._instances.clear()

        cache1 = CacheFactory.create_cache("singleton_cache")
        cache2 = CacheFactory.create_cache("singleton_cache")

        assert cache1 is cache2

    def test_get_cache_existing(self):
        """Test getting existing cache"""
        # Clear and create
        CacheFactory._instances.clear()
        original = CacheFactory.create_cache("existing_cache")

        retrieved = CacheFactory.get_cache("existing_cache")

        assert retrieved is original

    def test_get_cache_nonexistent(self):
        """Test getting non-existent cache"""
        result = CacheFactory.get_cache("nonexistent_cache")
        assert result is None

    @pytest.mark.asyncio
    async def test_initialize_all(self):
        """Test initializing all cache instances"""
        # Clear and create test caches
        CacheFactory._instances.clear()
        cache1 = CacheFactory.create_cache("cache1")
        cache2 = CacheFactory.create_cache("cache2")

        # Mock initialize methods
        cache1.initialize = AsyncMock(return_value=True)
        cache2.initialize = AsyncMock(return_value=True)

        result = await CacheFactory.initialize_all()

        assert result is True
        cache1.initialize.assert_called_once()
        cache2.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all cache instances"""
        # Clear and create test caches
        CacheFactory._instances.clear()
        cache1 = CacheFactory.create_cache("cache1")
        cache2 = CacheFactory.create_cache("cache2")

        # Mock close methods
        cache1.close = AsyncMock()
        cache2.close = AsyncMock()

        await CacheFactory.close_all()

        cache1.close.assert_called_once()
        cache2.close.assert_called_once()
        assert len(CacheFactory._instances) == 0