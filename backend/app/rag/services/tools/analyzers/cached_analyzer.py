"""
Cached analyzer wrapper for pattern recognition with Redis backend
"""

import hashlib
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from app.core.cache import get_embedding_cache, get_market_cache
from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


class CachedAnalyzer:
    """Wrapper class to add Redis caching to any BaseAnalyzer"""
    
    def __init__(self, analyzer: BaseAnalyzer, cache_name: str = "pattern", ttl_seconds: int = 1800):
        """
        Initialize cached analyzer
        
        Args:
            analyzer: The BaseAnalyzer to wrap
            cache_name: Name for cache namespace (pattern, fibonacci, etc.)
            ttl_seconds: Time to live for cache entries (30 minutes default)
        """
        self.analyzer = analyzer
        self.cache_name = cache_name
        self.ttl_seconds = ttl_seconds
        
        # Use market cache for pattern analysis (shorter TTL for market data)
        self.cache = get_market_cache()
        
    def _generate_cache_key(self, data: List[Dict], lookback_period: int, 
                          min_strength: float, **kwargs) -> str:
        """Generate deterministic cache key from analysis parameters"""
        try:
            # Create a data fingerprint (last 20 points + data length for efficiency)
            data_sample = data[-20:] if len(data) > 20 else data
            data_fingerprint = hashlib.md5(
                json.dumps(data_sample, sort_keys=True).encode()
            ).hexdigest()[:8]
            
            # Include key parameters in cache key
            key_parts = [
                self.cache_name,
                self.analyzer.__class__.__name__.lower(),
                f"len_{len(data)}",
                f"data_{data_fingerprint}",
                f"lookback_{lookback_period}",
                f"strength_{min_strength}",
            ]
            
            # Add kwargs to cache key
            if kwargs:
                sorted_kwargs = sorted(kwargs.items())
                kwargs_str = hashlib.md5(
                    json.dumps(sorted_kwargs, sort_keys=True).encode()
                ).hexdigest()[:8]
                key_parts.append(f"kwargs_{kwargs_str}")
            
            cache_key = "_".join(key_parts)
            return cache_key
            
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            # Fallback to simple key
            return f"{self.cache_name}_{self.analyzer.__class__.__name__}_{datetime.now().timestamp()}"
    
    async def analyze(self, data: List[Dict], lookback_period: int, 
                     min_strength: float, **kwargs) -> Dict[str, Any]:
        """
        Cached analysis with Redis backend
        
        Returns cached result if available, otherwise runs analysis and caches result
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(data, lookback_period, min_strength, **kwargs)
            
            # Try to get from cache first
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache HIT for {self.analyzer.__class__.__name__}: {cache_key}")
                
                # Add cache metadata
                cached_result["cache_info"] = {
                    "cached": True,
                    "cache_key": cache_key,
                    "retrieved_at": datetime.now().isoformat()
                }
                return cached_result
            
            # Cache miss - run analysis
            logger.debug(f"Cache MISS for {self.analyzer.__class__.__name__}: {cache_key}")
            result = await self.analyzer.analyze(data, lookback_period, min_strength, **kwargs)
            
            # Cache successful results only
            if result and not result.get("error"):
                # Add cache metadata before caching
                result["cache_info"] = {
                    "cached": False,
                    "cache_key": cache_key,
                    "analyzed_at": datetime.now().isoformat(),
                    "ttl_seconds": self.ttl_seconds
                }
                
                # Cache with custom TTL
                await self.cache.set(cache_key, result, ttl_override=self.ttl_seconds)
                logger.debug(f"Cached result for {self.analyzer.__class__.__name__}: {cache_key}")
            else:
                # Don't cache errors, but add cache info
                result["cache_info"] = {
                    "cached": False,
                    "cache_key": cache_key,
                    "analyzed_at": datetime.now().isoformat(),
                    "error": "Result not cached due to errors"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Cached analysis error for {self.analyzer.__class__.__name__}: {e}")
            
            # Fallback to direct analysis without caching
            try:
                result = await self.analyzer.analyze(data, lookback_period, min_strength, **kwargs)
                result["cache_info"] = {
                    "cached": False,
                    "cache_key": "fallback",
                    "analyzed_at": datetime.now().isoformat(),
                    "error": f"Cache error: {str(e)}"
                }
                return result
            except Exception as fallback_error:
                return {
                    "patterns": [],
                    "error": f"Analysis failed: {str(fallback_error)}",
                    "cache_info": {
                        "cached": False,
                        "cache_error": str(e),
                        "analysis_error": str(fallback_error)
                    }
                }
    
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cached results for this analyzer"""
        try:
            cache_pattern = pattern or f"{self.cache_name}_{self.analyzer.__class__.__name__.lower()}*"
            cleared_count = await self.cache.clear(cache_pattern)
            logger.info(f"Cleared {cleared_count} cache entries for {self.analyzer.__class__.__name__}")
            return cleared_count
        except Exception as e:
            logger.error(f"Failed to clear cache for {self.analyzer.__class__.__name__}: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            health_info = await self.cache.health_check()
            return {
                "analyzer": self.analyzer.__class__.__name__,
                "cache_name": self.cache_name,
                "ttl_seconds": self.ttl_seconds,
                "cache_health": health_info
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}


def cached_analyzer(cache_name: str = "pattern", ttl_seconds: int = 1800):
    """
    Decorator to wrap any BaseAnalyzer with Redis caching
    
    Usage:
        @cached_analyzer("fibonacci", ttl_seconds=3600)
        class FibonacciAnalyzer(BaseAnalyzer):
            # ... analyzer implementation
    """
    def decorator(analyzer_class):
        class CachedAnalyzerClass(analyzer_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._cached_wrapper = CachedAnalyzer(self, cache_name, ttl_seconds)
            
            async def analyze(self, data: List[Dict], lookback_period: int, 
                           min_strength: float, **kwargs) -> Dict[str, Any]:
                return await self._cached_wrapper.analyze(data, lookback_period, min_strength, **kwargs)
            
            async def clear_cache(self, pattern: Optional[str] = None) -> int:
                return await self._cached_wrapper.clear_cache(pattern)
            
            async def get_cache_stats(self) -> Dict[str, Any]:
                return await self._cached_wrapper.get_cache_stats()
        
        return CachedAnalyzerClass
    return decorator