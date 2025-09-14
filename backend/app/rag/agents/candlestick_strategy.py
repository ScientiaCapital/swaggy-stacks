"""
Candlestick Strategy Plugin
Integrates candlestick pattern recognition from PatternTool with StrategyPlugin interface
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from langchain.agents import Tool

from app.monitoring.metrics import PrometheusMetrics

logger = logging.getLogger(__name__)


class CandlestickStrategy:
    """Candlestick pattern-based trading strategy plugin"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = "candlestick"
        self.pattern_confidence_threshold = config.get("pattern_confidence_threshold", 0.6)
        self.volume_confirmation_weight = config.get("volume_confirmation_weight", 0.2)
        self.trend_confirmation_weight = config.get("trend_confirmation_weight", 0.3)

        # Initialize PatternTool integration
        self.pattern_tool = None
        self._initialize_pattern_tool()

    def _initialize_pattern_tool(self):
        """Initialize PatternTool for candlestick pattern detection"""
        try:
            from app.rag.tools.pattern_tool import PatternTool
            self.pattern_tool = PatternTool()
            logger.info("✅ PatternTool initialized for candlestick strategy")
        except Exception as e:
            logger.warning(f"⚠️ PatternTool initialization failed: {e}")

    def get_tools(self) -> List[Tool]:
        """Return LangChain tools for candlestick strategy"""
        return [
            Tool(
                name="detect_candlestick_patterns",
                func=self._detect_patterns_tool,
                description="Detect candlestick patterns (Pin Bar, Engulfing, Doji, Morning/Evening Star, Harami, Tweezers)",
            ),
            Tool(
                name="analyze_pattern_strength",
                func=self._analyze_pattern_strength_tool,
                description="Analyze strength and reliability of detected candlestick patterns",
            ),
            Tool(
                name="get_pattern_trade_setup",
                func=self._get_trade_setup_tool,
                description="Get specific trade setup recommendations based on candlestick patterns",
            ),
        ]

    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform candlestick pattern analysis"""
        start_time = datetime.now()
        metrics = PrometheusMetrics()
        symbol = market_data.get("symbol", "UNKNOWN")

        try:
            prices = market_data.get("prices", [])
            volumes = market_data.get("volumes", [])
            highs = market_data.get("highs", [])
            lows = market_data.get("lows", [])

            if len(prices) < 4:  # Minimum for pattern detection
                return {"error": "insufficient_data", "message": "Need at least 4 periods for pattern detection"}

            patterns = []
            pattern_strength = 0.0
            dominant_signal = "neutral"

            # Use PatternTool for pattern detection if available
            if self.pattern_tool:
                try:
                    # Convert OHLC data for pattern detection
                    ohlc_data = self._prepare_ohlc_data(prices, highs, lows, volumes)
                    detected_patterns = self.pattern_tool._detect_candlestick_patterns(prices, volumes)

                    if detected_patterns:
                        patterns = detected_patterns
                        pattern_strength, dominant_signal = self._calculate_aggregate_signal(patterns)

                except Exception as e:
                    logger.warning(f"PatternTool detection failed, using fallback: {e}")
                    patterns = self._fallback_pattern_detection(prices, volumes)
                    pattern_strength, dominant_signal = self._calculate_aggregate_signal(patterns)
            else:
                # Fallback pattern detection
                patterns = self._fallback_pattern_detection(prices, volumes)
                pattern_strength, dominant_signal = self._calculate_aggregate_signal(patterns)

            # Add trend and volume confirmation
            trend_context = self._calculate_trend_context(prices)
            volume_confirmation = self._calculate_volume_confirmation(volumes, patterns) if volumes else 0.5

            # Adjust pattern strength based on confirmations
            confirmed_strength = pattern_strength * (1 +
                (self.trend_confirmation_weight * self._trend_signal_alignment(trend_context, dominant_signal)) +
                (self.volume_confirmation_weight * (volume_confirmation - 0.5))
            )

            analysis_time = (datetime.now() - start_time).total_seconds()
            metrics.record_strategy_analysis_latency(analysis_time, "candlestick", symbol)

            return {
                "patterns": patterns,
                "pattern_count": len(patterns),
                "dominant_signal": dominant_signal,
                "pattern_strength": min(max(confirmed_strength, 0.0), 1.0),
                "trend_context": trend_context,
                "volume_confirmation": volume_confirmation,
                "confidence": min(max(confirmed_strength, 0.0), 1.0),
            }

        except Exception as e:
            analysis_time = (datetime.now() - start_time).total_seconds()
            metrics.record_strategy_analysis_latency(analysis_time, "candlestick", symbol)
            logger.error(f"Candlestick analysis failed for {symbol}: {str(e)}")
            return {"error": "analysis_failed", "message": str(e)}

    def generate_signal(self, analysis: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate candlestick pattern-based trading signal"""
        if "error" in analysis:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": analysis.get("message", "Pattern analysis failed"),
            }

        dominant_signal = analysis.get("dominant_signal", "neutral")
        confidence = analysis.get("confidence", 0.0)
        patterns = analysis.get("patterns", [])
        pattern_count = analysis.get("pattern_count", 0)

        if confidence < self.pattern_confidence_threshold:
            return {
                "action": "HOLD",
                "confidence": confidence,
                "reasoning": f"Pattern confidence too low: {confidence:.1%} (threshold: {self.pattern_confidence_threshold:.1%})",
            }

        # Convert dominant signal to trading action
        action_mapping = {
            "bullish": "BUY",
            "bearish": "SELL",
            "neutral": "HOLD"
        }

        action = action_mapping.get(dominant_signal, "HOLD")

        # Generate detailed reasoning
        pattern_names = [p.get("name", "unknown") for p in patterns]
        reasoning = f"Detected {pattern_count} candlestick pattern(s): {', '.join(pattern_names[:3])}{'...' if len(pattern_names) > 3 else ''}"

        if analysis.get("trend_context") != "neutral":
            reasoning += f" with {analysis.get('trend_context', 'unknown')} trend context"

        if analysis.get("volume_confirmation", 0.5) > 0.6:
            reasoning += " and volume confirmation"

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "patterns_detected": pattern_names,
        }

    def _prepare_ohlc_data(self, prices: List[float], highs: List[float] = None,
                          lows: List[float] = None, volumes: List[float] = None) -> Dict[str, List[float]]:
        """Prepare OHLC data structure for pattern detection"""
        # If highs/lows not provided, estimate from prices
        if not highs:
            highs = prices.copy()
        if not lows:
            lows = prices.copy()

        return {
            "opens": prices[:-1] if len(prices) > 1 else prices,
            "highs": highs,
            "lows": lows,
            "closes": prices,
            "volumes": volumes or [1000] * len(prices)  # Default volume if not provided
        }

    def _fallback_pattern_detection(self, prices: List[float], volumes: List[float] = None) -> List[Dict[str, Any]]:
        """Fallback pattern detection when PatternTool is not available"""
        patterns = []

        if len(prices) < 4:
            return patterns

        # Simple pattern detection logic
        recent_prices = prices[-4:]  # Last 4 candles

        # Detect basic patterns
        if self._is_doji_pattern(recent_prices[-1:]):
            patterns.append({
                "name": "Doji",
                "direction": "neutral",
                "strength": 0.6,
                "description": "Indecision pattern"
            })

        if self._is_engulfing_pattern(recent_prices[-2:]):
            direction = "bullish" if recent_prices[-1] > recent_prices[-2] else "bearish"
            patterns.append({
                "name": "Engulfing",
                "direction": direction,
                "strength": 0.7,
                "description": f"{direction.capitalize()} engulfing pattern"
            })

        return patterns

    def _is_doji_pattern(self, candles: List[float]) -> bool:
        """Simple Doji detection based on small body"""
        if not candles:
            return False
        # Simplified: assume Doji if price doesn't move much
        return True  # Placeholder implementation

    def _is_engulfing_pattern(self, candles: List[float]) -> bool:
        """Simple engulfing pattern detection"""
        if len(candles) < 2:
            return False
        return abs(candles[-1] - candles[-2]) / candles[-2] > 0.02  # 2% move

    def _calculate_aggregate_signal(self, patterns: List[Dict[str, Any]]) -> Tuple[float, str]:
        """Calculate aggregate signal strength and direction from multiple patterns"""
        if not patterns:
            return 0.0, "neutral"

        bullish_strength = 0.0
        bearish_strength = 0.0
        neutral_strength = 0.0

        for pattern in patterns:
            strength = pattern.get("strength", 0.0)
            direction = pattern.get("direction", "neutral")

            if direction == "bullish":
                bullish_strength += strength
            elif direction == "bearish":
                bearish_strength += strength
            else:
                neutral_strength += strength

        # Determine dominant signal
        max_strength = max(bullish_strength, bearish_strength, neutral_strength)

        if max_strength == 0:
            return 0.0, "neutral"

        if bullish_strength == max_strength:
            return bullish_strength / len(patterns), "bullish"
        elif bearish_strength == max_strength:
            return bearish_strength / len(patterns), "bearish"
        else:
            return neutral_strength / len(patterns), "neutral"

    def _calculate_trend_context(self, prices: List[float]) -> str:
        """Calculate trend context for pattern confirmation"""
        if len(prices) < 20:
            return "neutral"

        # Simple moving average trend
        recent_prices = prices[-10:]
        older_prices = prices[-20:-10]

        recent_avg = np.mean(recent_prices)
        older_avg = np.mean(older_prices)

        trend_strength = (recent_avg - older_avg) / older_avg

        if trend_strength > 0.02:  # 2% uptrend
            return "bullish"
        elif trend_strength < -0.02:  # 2% downtrend
            return "bearish"
        else:
            return "neutral"

    def _calculate_volume_confirmation(self, volumes: List[float], patterns: List[Dict[str, Any]]) -> float:
        """Calculate volume confirmation for pattern strength"""
        if not volumes or len(volumes) < 4:
            return 0.5  # Neutral if no volume data

        recent_volume = np.mean(volumes[-2:])  # Last 2 periods
        average_volume = np.mean(volumes[-20:] if len(volumes) >= 20 else volumes)

        volume_ratio = recent_volume / average_volume if average_volume > 0 else 1.0

        # Higher volume confirmation for significant patterns
        return min(0.5 + (volume_ratio - 1.0) * 0.3, 1.0)

    def _trend_signal_alignment(self, trend_context: str, signal: str) -> float:
        """Calculate how well the signal aligns with trend context"""
        alignment_matrix = {
            ("bullish", "bullish"): 1.0,
            ("bearish", "bearish"): 1.0,
            ("bullish", "bearish"): -0.5,
            ("bearish", "bullish"): -0.5,
            ("neutral", "bullish"): 0.2,
            ("neutral", "bearish"): 0.2,
            ("bullish", "neutral"): 0.0,
            ("bearish", "neutral"): 0.0,
            ("neutral", "neutral"): 0.0,
        }

        return alignment_matrix.get((trend_context, signal), 0.0)

    # Tool implementations
    def _detect_patterns_tool(self, price_data: str) -> str:
        """Tool implementation for pattern detection"""
        try:
            prices = [float(x.strip()) for x in price_data.split(",")]
            analysis = self.analyze_market({"prices": prices})
            patterns = analysis.get("patterns", [])

            if patterns:
                pattern_summary = []
                for pattern in patterns[:5]:  # Show top 5 patterns
                    pattern_summary.append(
                        f"{pattern.get('name', 'Unknown')}: {pattern.get('direction', 'neutral')} "
                        f"(strength: {pattern.get('strength', 0.0):.2f})"
                    )
                return f"Detected patterns: {', '.join(pattern_summary)}"
            else:
                return "No significant candlestick patterns detected"

        except Exception as e:
            return f"Pattern detection error: {str(e)}"

    def _analyze_pattern_strength_tool(self, pattern_name: str) -> str:
        """Tool implementation for pattern strength analysis"""
        pattern_characteristics = {
            "doji": "Indecision pattern, reversal potential at key levels",
            "engulfing": "Strong reversal pattern, high reliability with volume",
            "pin_bar": "Rejection pattern, strong with long wicks",
            "morning_star": "Bullish reversal, three-candle pattern",
            "evening_star": "Bearish reversal, three-candle pattern",
            "harami": "Inside bar pattern, continuation or reversal",
            "tweezers": "Double top/bottom pattern, reversal signal"
        }

        pattern_key = pattern_name.lower().replace(" ", "_")
        characteristics = pattern_characteristics.get(pattern_key, "Pattern analysis not available")

        return f"Pattern: {pattern_name} - {characteristics}"

    def _get_trade_setup_tool(self, pattern_data: str) -> str:
        """Tool implementation for trade setup recommendations"""
        try:
            # Parse pattern data (simplified)
            parts = pattern_data.split(",")
            if len(parts) >= 2:
                pattern_name = parts[0].strip()
                direction = parts[1].strip()

                if direction.lower() == "bullish":
                    return f"Trade Setup for {pattern_name}: Entry on break above pattern high, Stop below pattern low, Target 2:1 R/R"
                elif direction.lower() == "bearish":
                    return f"Trade Setup for {pattern_name}: Entry on break below pattern low, Stop above pattern high, Target 2:1 R/R"
                else:
                    return f"Trade Setup for {pattern_name}: Wait for directional confirmation before entry"
            else:
                return "Invalid pattern data format. Use: 'pattern_name, direction'"

        except Exception as e:
            return f"Trade setup error: {str(e)}"

    def extract_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract candlestick pattern-specific features for pattern matching"""
        analysis = self.analyze_market(market_data)

        return {
            "strategy": self.name,
            "timestamp": datetime.now().isoformat(),
            "pattern_count": analysis.get("pattern_count", 0),
            "dominant_signal": analysis.get("dominant_signal", "neutral"),
            "pattern_strength": analysis.get("pattern_strength", 0.0),
            "trend_context": analysis.get("trend_context", "neutral"),
            "volume_confirmation": analysis.get("volume_confirmation", 0.5),
        }