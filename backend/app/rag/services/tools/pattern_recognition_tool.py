"""
Pattern Recognition Tool for identifying trading patterns
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .analyzers import (
    ConfluenceAnalyzer,
    ElliottWaveAnalyzer,
    FibonacciAnalyzer,
    WyckoffAnalyzer,
)
from .analyzers.cached_analyzer import CachedAnalyzer
from .base_tool import AgentTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class PatternRecognitionTool(AgentTool):
    """Tool for recognizing trading patterns using modular analysis methods"""

    def __init__(self):
        super().__init__(
            name="pattern_recognition",
            description="Identify trading patterns including Fibonacci Golden Zone, Elliott Wave, Wyckoff Method, and confluence analysis",
        )
        self.category = "pattern_analysis"

        # Initialize cached analyzers for improved performance
        self.analyzers = {
            "fibonacci": CachedAnalyzer(
                FibonacciAnalyzer(), cache_name="fibonacci", ttl_seconds=3600
            ),
            "elliott_wave": CachedAnalyzer(
                ElliottWaveAnalyzer(), cache_name="elliott_wave", ttl_seconds=3600
            ),
            "wyckoff": CachedAnalyzer(
                WyckoffAnalyzer(), cache_name="wyckoff", ttl_seconds=1800
            ),
            "confluence": CachedAnalyzer(
                ConfluenceAnalyzer(), cache_name="confluence", ttl_seconds=1800
            ),
        }

    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        return [
            ToolParameter(
                name="pattern_type",
                type="str",
                description="Type of pattern: 'fibonacci', 'elliott_wave', 'wyckoff', 'confluence'",
                required=True,
            ),
            ToolParameter(
                name="data",
                type="list",
                description="Price data with OHLC and volume values",
                required=True,
            ),
            ToolParameter(
                name="lookback_period",
                type="int",
                description="Number of periods to look back for pattern recognition",
                required=False,
                default=20,
            ),
            ToolParameter(
                name="symbol",
                type="str",
                description="Stock symbol for pattern context",
                required=False,
                default="UNKNOWN",
            ),
            ToolParameter(
                name="min_strength",
                type="float",
                description="Minimum pattern strength threshold (0.0 to 1.0)",
                required=False,
                default=0.5,
            ),
            ToolParameter(
                name="timeframe",
                type="str",
                description="Chart timeframe (1m, 5m, 15m, 1h, 1d)",
                required=False,
                default="1d",
            ),
        ]

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute pattern recognition analysis"""
        try:
            pattern_type = parameters["pattern_type"].lower()
            data = parameters["data"]
            lookback_period = parameters.get("lookback_period", 20)
            symbol = parameters.get("symbol", "UNKNOWN")
            min_strength = parameters.get("min_strength", 0.5)
            timeframe = parameters.get("timeframe", "1d")

            # Validate input data
            if not data or len(data) < 10:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Insufficient data for pattern recognition. Need at least 10 data points.",
                )

            # Validate pattern type
            if pattern_type not in self.analyzers:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown pattern type: {pattern_type}. Supported types: {list(self.analyzers.keys())}",
                )

            # Execute analysis using the appropriate analyzer
            analyzer = self.analyzers[pattern_type]
            analysis_result = await analyzer.analyze(
                data=data,
                lookback_period=lookback_period,
                min_strength=min_strength,
                symbol=symbol,
                timeframe=timeframe,
            )

            # Add metadata
            metadata = {
                "pattern_type": pattern_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "data_points_analyzed": len(data),
                "lookback_period": lookback_period,
                "min_strength": min_strength,
                "analysis_timestamp": datetime.now().isoformat(),
                "analyzer_used": analyzer.__class__.__name__,
            }

            # Check for analysis errors
            if "error" in analysis_result:
                return ToolResult(
                    success=False,
                    data=analysis_result,
                    error=analysis_result["error"],
                    metadata=metadata,
                )

            # Enhance results with additional context
            enhanced_result = self._enhance_analysis_result(
                analysis_result, pattern_type, symbol
            )

            return ToolResult(success=True, data=enhanced_result, metadata=metadata)

        except Exception as e:
            logger.error(f"Pattern recognition error: {e}")
            return ToolResult(
                success=False, data=None, error=f"Pattern recognition failed: {str(e)}"
            )

    def _enhance_analysis_result(
        self, result: Dict[str, Any], pattern_type: str, symbol: str
    ) -> Dict[str, Any]:
        """Enhance analysis result with additional context and insights"""
        enhanced = result.copy()

        # Add pattern-specific insights
        if pattern_type == "fibonacci":
            enhanced["insights"] = self._generate_fibonacci_insights(result)
        elif pattern_type == "elliott_wave":
            enhanced["insights"] = self._generate_elliott_wave_insights(result)
        elif pattern_type == "wyckoff":
            enhanced["insights"] = self._generate_wyckoff_insights(result)
        elif pattern_type == "confluence":
            enhanced["insights"] = self._generate_confluence_insights(result)

        # Add general trading context
        enhanced["trading_context"] = {
            "symbol": symbol,
            "analysis_type": pattern_type,
            "key_recommendation": self._get_key_recommendation(result, pattern_type),
            "risk_level": self._assess_risk_level(result),
            "confidence_level": self._assess_confidence_level(result),
        }

        return enhanced

    def _generate_fibonacci_insights(self, result: Dict[str, Any]) -> List[str]:
        """Generate insights for Fibonacci analysis"""
        insights = []

        if result.get("in_golden_zone"):
            insights.append(
                "Price is in the Golden Zone (61.8% retracement) - high probability reversal area"
            )

        patterns = result.get("patterns", [])
        golden_zone_patterns = [p for p in patterns if p.get("is_golden_zone")]

        if golden_zone_patterns:
            insights.append("Golden Zone confluence detected - ideal entry opportunity")

        if result.get("trend_direction") == "bullish":
            insights.append(
                "Look for bullish reversal signals at Fibonacci support levels"
            )
        elif result.get("trend_direction") == "bearish":
            insights.append(
                "Look for bearish reversal signals at Fibonacci resistance levels"
            )

        return insights

    def _generate_elliott_wave_insights(self, result: Dict[str, Any]) -> List[str]:
        """Generate insights for Elliott Wave analysis"""
        insights = []

        wave_position = result.get("probable_wave_position", "")
        if "Wave 5" in wave_position:
            insights.append("Possible Wave 5 completion - watch for trend reversal")
        elif "Wave 3" in wave_position:
            insights.append("Potential Wave 3 development - strongest trending wave")
        elif "corrective" in wave_position.lower():
            insights.append("Corrective wave phase - expect choppy price action")

        patterns = result.get("patterns", [])
        impulse_patterns = [p for p in patterns if p.get("type") == "impulse"]

        if impulse_patterns:
            insights.append(
                "Impulse wave structure identified - trend continuation likely"
            )

        return insights

    def _generate_wyckoff_insights(self, result: Dict[str, Any]) -> List[str]:
        """Generate insights for Wyckoff analysis"""
        insights = []

        current_phase = result.get("current_phase", "")

        if current_phase == "accumulation":
            insights.append(
                "Accumulation phase detected - smart money building positions"
            )
        elif current_phase == "distribution":
            insights.append(
                "Distribution phase detected - smart money selling into strength"
            )
        elif current_phase == "markup":
            insights.append("Markup phase - trending market with strong momentum")
        elif current_phase == "markdown":
            insights.append("Markdown phase - declining market with selling pressure")

        patterns = result.get("patterns", [])
        springs = [p for p in patterns if p.get("type") == "spring"]
        upthrusts = [p for p in patterns if p.get("type") == "upthrust"]

        if springs:
            insights.append(
                "Spring pattern detected - false breakdown followed by reversal"
            )
        if upthrusts:
            insights.append(
                "Upthrust pattern detected - false breakout followed by reversal"
            )

        return insights

    def _generate_confluence_insights(self, result: Dict[str, Any]) -> List[str]:
        """Generate insights for confluence analysis"""
        insights = []

        confluence_summary = result.get("confluence_summary", {})
        confluence_summary.get("setups_found", 0)
        high_quality_setups = confluence_summary.get("high_quality_setups", 0)

        if high_quality_setups > 0:
            insights.append(
                f"Found {high_quality_setups} high-quality confluence setup(s)"
            )

        market_sentiment = result.get("market_sentiment", {})
        sentiment = market_sentiment.get("sentiment", "")
        confidence = market_sentiment.get("confidence", 0)

        if sentiment and confidence > 0.7:
            insights.append(
                f"Strong {sentiment} sentiment with {confidence:.0%} confidence"
            )

        recommendations = result.get("trading_recommendations", [])
        if recommendations:
            best_rec = max(recommendations, key=lambda x: x.get("confidence", 0))
            insights.append(
                f"Best setup: {best_rec.get('setup', 'Unknown')} with {best_rec.get('confidence', 0):.0%} confidence"
            )

        return insights

    def _get_key_recommendation(self, result: Dict[str, Any], pattern_type: str) -> str:
        """Get the key trading recommendation"""
        patterns = result.get("patterns", [])

        if not patterns:
            return "No clear pattern signals"

        # Get the strongest pattern
        strongest_pattern = max(patterns, key=lambda x: x.get("strength", 0))
        signal = strongest_pattern.get("signal", "no_signal")

        # Convert technical signals to trading recommendations
        if "buy" in signal.lower() or "bullish" in signal.lower():
            return "Consider long positions"
        elif "sell" in signal.lower() or "bearish" in signal.lower():
            return "Consider short positions"
        elif "reversal" in signal.lower():
            return "Watch for trend reversal"
        elif "continuation" in signal.lower():
            return "Trend likely to continue"
        else:
            return "Monitor for clearer signals"

    def _assess_risk_level(self, result: Dict[str, Any]) -> str:
        """Assess the risk level based on pattern analysis"""
        patterns = result.get("patterns", [])

        if not patterns:
            return "unknown"

        avg_strength = sum(p.get("strength", 0) for p in patterns) / len(patterns)

        if avg_strength >= 0.8:
            return "low"  # High confidence patterns have lower risk
        elif avg_strength >= 0.6:
            return "medium"
        else:
            return "high"

    def _assess_confidence_level(self, result: Dict[str, Any]) -> str:
        """Assess confidence level in the analysis"""
        patterns = result.get("patterns", [])

        if not patterns:
            return "low"

        # For confluence analysis, use specific confidence metrics
        if "confluence_summary" in result:
            avg_confidence = result["confluence_summary"].get("average_confidence", 0)
            if avg_confidence >= 0.8:
                return "high"
            elif avg_confidence >= 0.6:
                return "medium"
            else:
                return "low"

        # For other analyses, use pattern strength
        avg_strength = sum(p.get("strength", 0) for p in patterns) / len(patterns)

        if avg_strength >= 0.8:
            return "high"
        elif avg_strength >= 0.6:
            return "medium"
        else:
            return "low"

    async def clear_cache(self, pattern_type: Optional[str] = None) -> Dict[str, int]:
        """Clear cached results for specific or all analyzers"""
        results = {}

        if pattern_type and pattern_type.lower() in self.analyzers:
            # Clear specific analyzer cache
            analyzer = self.analyzers[pattern_type.lower()]
            cleared = await analyzer.clear_cache()
            results[pattern_type] = cleared
            logger.info(f"Cleared {cleared} cache entries for {pattern_type}")
        else:
            # Clear all analyzer caches
            for name, analyzer in self.analyzers.items():
                cleared = await analyzer.clear_cache()
                results[name] = cleared
                logger.info(f"Cleared {cleared} cache entries for {name}")

        return results

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics for all analyzers"""
        stats = {}

        for name, analyzer in self.analyzers.items():
            try:
                analyzer_stats = await analyzer.get_cache_stats()
                stats[name] = analyzer_stats
            except Exception as e:
                logger.warning(f"Failed to get cache stats for {name}: {e}")
                stats[name] = {"error": str(e)}

        return {
            "tool": "PatternRecognitionTool",
            "total_analyzers": len(self.analyzers),
            "analyzer_stats": stats,
            "cache_health_summary": self._summarize_cache_health(stats),
        }

    def _summarize_cache_health(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize overall cache health across all analyzers"""
        healthy_analyzers = 0
        total_analyzers = len(self.analyzers)

        for name, analyzer_stats in stats.items():
            if isinstance(analyzer_stats, dict) and not analyzer_stats.get("error"):
                cache_health = analyzer_stats.get("cache_health", {})
                if cache_health.get("status") == "healthy":
                    healthy_analyzers += 1

        health_percentage = (
            (healthy_analyzers / total_analyzers * 100) if total_analyzers > 0 else 0
        )

        return {
            "healthy_analyzers": healthy_analyzers,
            "total_analyzers": total_analyzers,
            "health_percentage": round(health_percentage, 1),
            "overall_status": (
                "healthy"
                if health_percentage >= 75
                else "degraded" if health_percentage >= 50 else "unhealthy"
            ),
        }
