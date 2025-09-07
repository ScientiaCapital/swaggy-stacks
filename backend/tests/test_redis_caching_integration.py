"""
Tests for Redis caching integration with pattern recognition tools
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta

from app.rag.services.tools.pattern_recognition_tool import PatternRecognitionTool
from app.rag.services.tools.technical_indicator_tool import TechnicalIndicatorTool
from app.rag.services.tools.analyzers.cached_analyzer import CachedAnalyzer
from app.rag.services.tools.analyzers.fibonacci_analyzer import FibonacciAnalyzer


class TestRedisCachingIntegration:
    """Test Redis caching integration across trading tools"""

    @pytest.fixture
    def sample_price_data(self):
        """Generate sample OHLCV data for testing"""
        base_price = 100.0
        data = []
        
        for i in range(50):
            # Generate realistic price movement
            price_change = (i % 10 - 5) * 0.5  # Oscillating pattern
            close_price = base_price + price_change + (i * 0.1)  # Slight uptrend
            
            data.append({
                "timestamp": (datetime.now() - timedelta(days=49-i)).isoformat(),
                "open": close_price - 0.2,
                "high": close_price + 0.3,
                "low": close_price - 0.5,
                "close": close_price,
                "volume": 1000000 + (i * 10000)
            })
        
        return data

    @pytest.fixture
    def mock_cache(self):
        """Mock cache for testing"""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)  # Cache miss by default
        cache.set = AsyncMock(return_value=True)
        cache.clear = AsyncMock(return_value=5)
        cache.health_check = AsyncMock(return_value={
            "status": "healthy",
            "l1_cache": {"working": True},
            "l2_cache": {"working": True}
        })
        return cache

    @pytest.mark.asyncio
    async def test_cached_analyzer_cache_miss(self, sample_price_data, mock_cache):
        """Test cached analyzer behavior on cache miss"""
        # Create base analyzer
        base_analyzer = FibonacciAnalyzer()
        
        # Create cached analyzer with mock cache
        cached_analyzer = CachedAnalyzer(base_analyzer, cache_name="test_fib", ttl_seconds=1800)
        cached_analyzer.cache = mock_cache
        
        # Execute analysis
        result = await cached_analyzer.analyze(
            data=sample_price_data,
            lookback_period=20,
            min_strength=0.5,
            symbol="TEST",
            timeframe="1d"
        )
        
        # Verify cache was checked
        mock_cache.get.assert_called_once()
        
        # Verify result was cached (if successful)
        if result and not result.get("error"):
            mock_cache.set.assert_called_once()
        
        # Verify cache info is included
        assert "cache_info" in result
        assert result["cache_info"]["cached"] is False
        assert "cache_key" in result["cache_info"]

    @pytest.mark.asyncio
    async def test_cached_analyzer_cache_hit(self, sample_price_data, mock_cache):
        """Test cached analyzer behavior on cache hit"""
        # Create cached result
        cached_result = {
            "patterns": [{"name": "Test Pattern", "strength": 0.8}],
            "cache_info": {"cached": True, "cache_key": "test_key"}
        }
        mock_cache.get.return_value = cached_result
        
        # Create cached analyzer
        base_analyzer = FibonacciAnalyzer()
        cached_analyzer = CachedAnalyzer(base_analyzer, cache_name="test_fib")
        cached_analyzer.cache = mock_cache
        
        # Execute analysis
        result = await cached_analyzer.analyze(
            data=sample_price_data,
            lookback_period=20,
            min_strength=0.5
        )
        
        # Verify cache was checked and hit
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_not_called()  # Should not cache again
        
        # Verify cached result returned
        assert result == cached_result
        assert result["cache_info"]["cached"] is True

    @pytest.mark.asyncio
    async def test_pattern_recognition_tool_caching(self, sample_price_data, mock_cache):
        """Test PatternRecognitionTool Redis caching integration"""
        tool = PatternRecognitionTool()
        
        # Mock the cache for all analyzers
        for analyzer in tool.analyzers.values():
            analyzer.cache = mock_cache
        
        # Test Fibonacci analysis
        parameters = {
            "pattern_type": "fibonacci",
            "data": sample_price_data,
            "lookback_period": 20,
            "min_strength": 0.5,
            "symbol": "AAPL",
            "timeframe": "1d"
        }
        
        result = await tool.execute(parameters)
        
        # Verify cache was used
        mock_cache.get.assert_called()
        
        # Verify successful result
        assert result.success is True
        assert "data" in result.__dict__

    @pytest.mark.asyncio
    async def test_technical_indicator_tool_caching(self, sample_price_data, mock_cache):
        """Test TechnicalIndicatorTool Redis caching integration"""
        tool = TechnicalIndicatorTool()
        tool.cache = mock_cache
        
        # Test RSI calculation with caching
        parameters = {
            "indicator": "rsi",
            "data": sample_price_data,
            "period": 14
        }
        
        result = await tool.execute(parameters)
        
        # Verify cache was checked
        mock_cache.get.assert_called_once()
        
        # Verify result structure
        assert result.success is True
        if result.metadata:
            assert "cache_hit" in result.metadata

    @pytest.mark.asyncio
    async def test_cache_key_generation_consistency(self, sample_price_data):
        """Test that cache keys are generated consistently"""
        tool = TechnicalIndicatorTool()
        
        parameters = {
            "indicator": "rsi",
            "data": sample_price_data,
            "period": 14
        }
        
        # Generate cache key twice with same parameters
        key1 = tool._generate_cache_key("rsi", sample_price_data, parameters)
        key2 = tool._generate_cache_key("rsi", sample_price_data, parameters)
        
        # Should be identical
        assert key1 == key2
        
        # Should contain indicator name
        assert "rsi" in key1.lower()

    @pytest.mark.asyncio
    async def test_cache_key_generation_different_params(self, sample_price_data):
        """Test that different parameters generate different cache keys"""
        tool = TechnicalIndicatorTool()
        
        params1 = {"indicator": "rsi", "data": sample_price_data, "period": 14}
        params2 = {"indicator": "rsi", "data": sample_price_data, "period": 21}
        
        key1 = tool._generate_cache_key("rsi", sample_price_data, params1)
        key2 = tool._generate_cache_key("rsi", sample_price_data, params2)
        
        # Should be different
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_pattern_tool_cache_management(self, mock_cache):
        """Test cache management methods in PatternRecognitionTool"""
        tool = PatternRecognitionTool()
        
        # Mock cache for all analyzers
        for analyzer in tool.analyzers.values():
            analyzer.cache = mock_cache
            analyzer.clear_cache = AsyncMock(return_value=10)
            analyzer.get_cache_stats = AsyncMock(return_value={
                "analyzer": "TestAnalyzer",
                "cache_health": {"status": "healthy"}
            })
        
        # Test clearing specific analyzer cache
        result = await tool.clear_cache("fibonacci")
        assert "fibonacci" in result
        assert result["fibonacci"] == 10
        
        # Test clearing all caches
        result = await tool.clear_cache()
        assert len(result) == len(tool.analyzers)
        
        # Test getting cache stats
        stats = await tool.get_cache_stats()
        assert "tool" in stats
        assert "analyzer_stats" in stats
        assert "cache_health_summary" in stats

    @pytest.mark.asyncio
    async def test_technical_tool_cache_management(self, mock_cache):
        """Test cache management methods in TechnicalIndicatorTool"""
        tool = TechnicalIndicatorTool()
        tool.cache = mock_cache
        
        # Test clearing cache
        result = await tool.clear_cache("rsi")
        assert result == 5  # Mock return value
        mock_cache.clear.assert_called_with("indicator_rsi_*")
        
        # Test getting cache stats
        stats = await tool.get_cache_stats()
        assert "tool" in stats
        assert "cache_health" in stats
        assert "supported_indicators" in stats

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, sample_price_data):
        """Test graceful handling of cache errors"""
        # Create failing cache mock
        failing_cache = MagicMock()
        failing_cache.get = AsyncMock(side_effect=Exception("Cache connection failed"))
        failing_cache.set = AsyncMock(side_effect=Exception("Cache write failed"))
        
        # Test cached analyzer with failing cache
        base_analyzer = FibonacciAnalyzer()
        cached_analyzer = CachedAnalyzer(base_analyzer, cache_name="test_fib")
        cached_analyzer.cache = failing_cache
        
        # Should fall back to direct analysis
        result = await cached_analyzer.analyze(
            data=sample_price_data,
            lookback_period=20,
            min_strength=0.5
        )
        
        # Should still get result despite cache failure
        assert result is not None
        assert "cache_info" in result
        if "cache_error" not in result["cache_info"]:
            # Fallback analysis should work
            assert "patterns" in result

    def test_cache_configuration_values(self):
        """Test that cache TTL values are appropriate for different tools"""
        tool = PatternRecognitionTool()
        
        # Fibonacci and Elliott Wave should have longer cache (1 hour)
        assert tool.analyzers['fibonacci'].ttl_seconds == 3600
        assert tool.analyzers['elliott_wave'].ttl_seconds == 3600
        
        # Wyckoff and Confluence should have shorter cache (30 minutes)
        assert tool.analyzers['wyckoff'].ttl_seconds == 1800
        assert tool.analyzers['confluence'].ttl_seconds == 1800
        
        # Technical indicators should have shortest cache (15 minutes)
        indicator_tool = TechnicalIndicatorTool()
        # TTL is set in the execute method to 900 seconds (15 minutes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])