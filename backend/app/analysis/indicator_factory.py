"""
Indicator Factory for unified indicator management
Provides factory pattern to manage traditional and modern indicators with caching
"""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, Literal

import numpy as np
import pandas as pd
import structlog

from app.analysis.technical_indicators import TechnicalIndicators
from app.analysis.modern_indicators import ModernIndicators
from app.core.database import get_redis
from app.core.exceptions import TradingError

logger = structlog.get_logger()

IndicatorType = Literal["traditional", "modern", "both"]


class IndicatorFactory:
    """
    Factory class for unified indicator management with caching and performance optimization

    Provides a single interface to access both traditional and modern indicators
    with Redis caching for improved performance.
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize IndicatorFactory

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.traditional_indicators = TechnicalIndicators()
        self.modern_indicators = ModernIndicators()
        self.cache_ttl = cache_ttl
        self.redis_client = None
        self._init_redis()

        logger.info("IndicatorFactory initialized", cache_ttl=cache_ttl)

    def _init_redis(self):
        """Initialize Redis client for caching"""
        try:
            self.redis_client = get_redis()
            # Test connection
            self.redis_client.ping()
            logger.info("Redis client connected for indicator caching")
        except Exception as e:
            logger.warning("Redis not available, caching disabled", error=str(e))
            self.redis_client = None

    def _generate_cache_key(self, data: pd.DataFrame, indicator_type: str,
                          params: Optional[Dict] = None) -> str:
        """
        Generate unique cache key for indicator data

        Args:
            data: OHLCV DataFrame
            indicator_type: Type of indicators to calculate
            params: Additional parameters for indicator calculation

        Returns:
            Unique cache key string
        """
        # Create hash from data characteristics
        data_hash = hashlib.md5()

        # Hash data shape and key statistics
        data_hash.update(str(len(data)).encode())
        data_hash.update(str(data.columns.tolist()).encode())

        # Hash first, last, and some middle values for uniqueness
        if len(data) > 0:
            first_row = data.iloc[0].to_dict()
            last_row = data.iloc[-1].to_dict()
            data_hash.update(json.dumps(first_row, sort_keys=True, default=str).encode())
            data_hash.update(json.dumps(last_row, sort_keys=True, default=str).encode())

            # Add middle point for longer datasets
            if len(data) > 10:
                middle_row = data.iloc[len(data)//2].to_dict()
                data_hash.update(json.dumps(middle_row, sort_keys=True, default=str).encode())

        # Add parameters to hash
        if params:
            data_hash.update(json.dumps(params, sort_keys=True, default=str).encode())

        # Create final cache key
        cache_key = f"indicators:{indicator_type}:{data_hash.hexdigest()}"
        return cache_key

    def _get_cached_indicators(self, cache_key: str) -> Optional[Dict]:
        """
        Retrieve cached indicators from Redis

        Args:
            cache_key: Cache key to lookup

        Returns:
            Cached indicators dict or None if not found
        """
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                # Deserialize with pickle for numpy array support
                indicators = pickle.loads(cached_data)
                logger.debug("Cache hit for indicators", cache_key=cache_key)
                return indicators
        except Exception as e:
            logger.warning("Failed to retrieve cached indicators",
                         cache_key=cache_key, error=str(e))

        return None

    def _cache_indicators(self, cache_key: str, indicators: Dict):
        """
        Cache indicators in Redis

        Args:
            cache_key: Cache key for storage
            indicators: Indicators dict to cache
        """
        if not self.redis_client:
            return

        try:
            # Serialize with pickle for numpy array support
            serialized_data = pickle.dumps(indicators)
            self.redis_client.setex(cache_key, self.cache_ttl, serialized_data)
            logger.debug("Cached indicators", cache_key=cache_key, ttl=self.cache_ttl)
        except Exception as e:
            logger.warning("Failed to cache indicators",
                         cache_key=cache_key, error=str(e))

    def get_indicator(self, name: str, indicator_type: IndicatorType = "both") -> Any:
        """
        Get a specific indicator class instance

        Args:
            name: Indicator name (not used in current implementation)
            indicator_type: Type of indicators to return

        Returns:
            Indicator instance(s)
        """
        if indicator_type == "traditional":
            return self.traditional_indicators
        elif indicator_type == "modern":
            return self.modern_indicators
        elif indicator_type == "both":
            return {
                "traditional": self.traditional_indicators,
                "modern": self.modern_indicators
            }
        else:
            raise TradingError(f"Unknown indicator type: {indicator_type}")

    def calculate_all(self, data: pd.DataFrame,
                     indicator_type: IndicatorType = "both",
                     use_cache: bool = True,
                     force_refresh: bool = False) -> Dict[str, Any]:
        """
        Calculate all indicators of specified type(s) with caching support

        Args:
            data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            indicator_type: Type of indicators to calculate
            use_cache: Whether to use Redis caching
            force_refresh: Force recalculation ignoring cache

        Returns:
            Dict containing all calculated indicators
        """
        try:
            # Input validation
            if data.empty or len(data) < 50:
                raise TradingError(
                    f"Insufficient data: need at least 50 periods, got {len(data)}"
                )

            # Required columns validation
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise TradingError(f"Missing required columns: {missing_columns}")

            # Generate cache key
            cache_key = self._generate_cache_key(data, indicator_type)

            # Check cache if enabled and not forcing refresh
            if use_cache and not force_refresh:
                cached_result = self._get_cached_indicators(cache_key)
                if cached_result:
                    return cached_result

            # Calculate indicators based on type
            result = {}

            if indicator_type in ["traditional", "both"]:
                logger.info("Calculating traditional indicators", data_length=len(data))
                traditional_result = self.traditional_indicators.calculate_all_indicators(data)
                result["traditional"] = traditional_result

            if indicator_type in ["modern", "both"]:
                logger.info("Calculating modern indicators", data_length=len(data))
                modern_result = self.modern_indicators.calculate_all_indicators(data)
                result["modern"] = modern_result

            # For unified interface when requesting both
            if indicator_type == "both":
                # Merge all indicators into a single namespace with prefixes
                unified_indicators = {}

                # Add traditional indicators with 'trad_' prefix
                for key, value in result["traditional"].items():
                    unified_indicators[f"trad_{key}"] = value

                # Add modern indicators with 'mod_' prefix
                for key, value in result["modern"].items():
                    unified_indicators[f"mod_{key}"] = value

                # Add combined analysis
                unified_indicators["combined_signals"] = self._generate_combined_signals(
                    result["traditional"], result["modern"]
                )

                result["unified"] = unified_indicators

            # Cache result if caching is enabled
            if use_cache:
                self._cache_indicators(cache_key, result)

            logger.info("Indicator calculation completed",
                       indicator_type=indicator_type,
                       result_keys=list(result.keys()),
                       cached=use_cache)

            return result

        except Exception as e:
            logger.error("Failed to calculate indicators",
                        indicator_type=indicator_type,
                        error=str(e))
            raise TradingError(f"Indicator calculation failed: {str(e)}")

    def _generate_combined_signals(self, traditional: Dict, modern: Dict) -> Dict:
        """
        Generate combined signals from both traditional and modern indicators

        Args:
            traditional: Traditional indicators results
            modern: Modern indicators results

        Returns:
            Combined signals dict
        """
        try:
            combined_signals = {}

            # Get individual composite signals
            trad_composite = traditional.get("composite_signals", {})
            mod_composite = modern.get("modern_composite_signals", {})

            # Combined trend analysis
            trad_trend = trad_composite.get("trend", "NEUTRAL")
            kama_trend = mod_composite.get("kama_trend", "NEUTRAL")

            if trad_trend == "BULLISH" and kama_trend == "BULLISH":
                combined_signals["trend_consensus"] = "STRONG_BULLISH"
            elif trad_trend == "BEARISH" and kama_trend == "BEARISH":
                combined_signals["trend_consensus"] = "STRONG_BEARISH"
            elif trad_trend in ["BULLISH", "BEARISH"] or kama_trend in ["BULLISH", "BEARISH"]:
                if trad_trend == "BULLISH" or kama_trend == "BULLISH":
                    combined_signals["trend_consensus"] = "WEAK_BULLISH"
                else:
                    combined_signals["trend_consensus"] = "WEAK_BEARISH"
            else:
                combined_signals["trend_consensus"] = "NEUTRAL"

            # Combined momentum analysis
            trad_momentum = trad_composite.get("momentum", "NEUTRAL")
            pfe_momentum = mod_composite.get("pfe_momentum", "NEUTRAL")
            rvi_momentum = mod_composite.get("rvi_momentum", "NEUTRAL")

            momentum_score = 0
            if trad_momentum == "OVERSOLD":
                momentum_score += 1
            elif trad_momentum == "OVERBOUGHT":
                momentum_score -= 1

            if pfe_momentum in ["STRONG_BULLISH", "BULLISH"]:
                momentum_score += 1
            elif pfe_momentum in ["STRONG_BEARISH", "BEARISH"]:
                momentum_score -= 1

            if rvi_momentum == "BULLISH":
                momentum_score += 1
            elif rvi_momentum == "BEARISH":
                momentum_score -= 1

            if momentum_score >= 2:
                combined_signals["momentum_consensus"] = "STRONG_BULLISH"
            elif momentum_score == 1:
                combined_signals["momentum_consensus"] = "BULLISH"
            elif momentum_score == -1:
                combined_signals["momentum_consensus"] = "BEARISH"
            elif momentum_score <= -2:
                combined_signals["momentum_consensus"] = "STRONG_BEARISH"
            else:
                combined_signals["momentum_consensus"] = "NEUTRAL"

            # Final combined signal
            trad_signal = trad_composite.get("composite", "HOLD")
            mod_signal = mod_composite.get("modern_composite", "HOLD")

            if trad_signal == "BUY" and mod_signal == "BUY":
                combined_signals["final_signal"] = "STRONG_BUY"
            elif trad_signal == "SELL" and mod_signal == "SELL":
                combined_signals["final_signal"] = "STRONG_SELL"
            elif trad_signal == "BUY" or mod_signal == "BUY":
                combined_signals["final_signal"] = "BUY"
            elif trad_signal == "SELL" or mod_signal == "SELL":
                combined_signals["final_signal"] = "SELL"
            else:
                combined_signals["final_signal"] = "HOLD"

            # Signal strength (average of both systems)
            trad_strength = trad_composite.get("signal_strength", 0.0)
            mod_strength = mod_composite.get("modern_signal_strength", 0.0)
            combined_signals["signal_strength"] = (trad_strength + mod_strength) / 2

            return combined_signals

        except Exception as e:
            logger.error("Failed to generate combined signals", error=str(e))
            return {
                "trend_consensus": "NEUTRAL",
                "momentum_consensus": "NEUTRAL",
                "final_signal": "HOLD",
                "signal_strength": 0.0
            }

    def invalidate_cache(self, pattern: Optional[str] = None):
        """
        Invalidate cached indicators

        Args:
            pattern: Cache key pattern to invalidate (default: all indicators)
        """
        if not self.redis_client:
            return

        try:
            if pattern:
                cache_pattern = f"indicators:{pattern}:*"
            else:
                cache_pattern = "indicators:*"

            keys = self.redis_client.keys(cache_pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info("Cache invalidated", pattern=cache_pattern, deleted=deleted)
            else:
                logger.info("No cache entries found to invalidate", pattern=cache_pattern)

        except Exception as e:
            logger.error("Failed to invalidate cache", pattern=pattern, error=str(e))

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with cache statistics
        """
        if not self.redis_client:
            return {"cache_enabled": False}

        try:
            # Count indicator cache keys
            indicator_keys = self.redis_client.keys("indicators:*")

            # Get Redis info
            redis_info = self.redis_client.info()

            return {
                "cache_enabled": True,
                "indicator_cache_entries": len(indicator_keys),
                "redis_connected_clients": redis_info.get("connected_clients", 0),
                "redis_used_memory": redis_info.get("used_memory_human", "unknown"),
                "cache_ttl_seconds": self.cache_ttl
            }

        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {"cache_enabled": False, "error": str(e)}