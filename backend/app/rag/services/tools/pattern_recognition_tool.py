"""
Pattern Recognition Tool for identifying trading patterns
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging

from .base_tool import AgentTool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class PatternRecognitionTool(AgentTool):
    """Tool for recognizing trading patterns using various analysis methods"""
    
    def __init__(self):
        super().__init__(
            name="pattern_recognition",
            description="Identify trading patterns including candlestick patterns, trend patterns, and Markov states"
        )
        self.category = "pattern_analysis"
    
    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        return [
            ToolParameter(
                name="pattern_type",
                type="str",
                description="Type of pattern: 'candlestick', 'trend', 'support_resistance', 'markov', 'fibonacci', 'elliott_wave', 'wyckoff', 'confluence'",
                required=True
            ),
            ToolParameter(
                name="data",
                type="list",
                description="Price data with OHLC values",
                required=True
            ),
            ToolParameter(
                name="lookback_period",
                type="int",
                description="Number of periods to look back for pattern recognition",
                required=False,
                default=20
            ),
            ToolParameter(
                name="symbol",
                type="str",
                description="Stock symbol for pattern context",
                required=False
            ),
            ToolParameter(
                name="min_strength",
                type="float",
                description="Minimum pattern strength threshold (0.0 to 1.0)",
                required=False,
                default=0.6
            )
        ]
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute pattern recognition"""
        try:
            pattern_type = parameters["pattern_type"].lower()
            data = parameters["data"]
            lookback_period = parameters.get("lookback_period", 20)
            symbol = parameters.get("symbol", "UNKNOWN")
            min_strength = parameters.get("min_strength", 0.6)
            
            # Validate and prepare data
            if not data or len(data) < 3:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Insufficient data for pattern recognition (minimum 3 data points required)"
                )
            
            if pattern_type == "candlestick":
                return await self._recognize_candlestick_patterns(data, lookback_period, min_strength)
            elif pattern_type == "trend":
                return await self._recognize_trend_patterns(data, lookback_period, min_strength)
            elif pattern_type == "support_resistance":
                return await self._recognize_support_resistance(data, lookback_period, min_strength)
            elif pattern_type == "markov":
                return await self._recognize_markov_states(data, lookback_period, symbol)
            elif pattern_type == "fibonacci":
                return await self._recognize_fibonacci_patterns(data, lookback_period, min_strength)
            elif pattern_type == "elliott_wave":
                return await self._recognize_elliott_wave_patterns(data, lookback_period, min_strength)
            elif pattern_type == "wyckoff":
                return await self._recognize_wyckoff_patterns(data, lookback_period, min_strength)
            elif pattern_type == "confluence":
                return await self._analyze_confluence_patterns(data, lookback_period, min_strength, symbol)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown pattern type: {pattern_type}. Supported: 'candlestick', 'trend', 'support_resistance', 'markov', 'fibonacci', 'elliott_wave', 'wyckoff', 'confluence'"
                )
                
        except Exception as e:
            logger.error(f"Pattern recognition error: {e}")
            return ToolResult(success=False, data=None, error=f"Pattern recognition failed: {str(e)}")
    
    async def _recognize_candlestick_patterns(self, data: List[Dict], lookback_period: int, min_strength: float) -> ToolResult:
        """Recognize candlestick patterns"""
        try:
            patterns_found = []
            recent_data = data[-lookback_period:] if len(data) > lookback_period else data
            
            for i in range(1, len(recent_data)):
                current = recent_data[i]
                previous = recent_data[i-1] if i > 0 else None
                
                # Ensure required fields exist
                if not all(key in current for key in ['open', 'high', 'low', 'close']):
                    continue
                
                o, h, l, c = float(current['open']), float(current['high']), float(current['low']), float(current['close'])
                
                # Single candlestick patterns
                patterns = self._identify_single_candlestick_patterns(o, h, l, c, previous)
                
                # Multi-candlestick patterns
                if i >= 2:
                    multi_patterns = self._identify_multi_candlestick_patterns(recent_data[i-2:i+1])
                    patterns.extend(multi_patterns)
                
                # Filter by strength
                for pattern in patterns:
                    if pattern['strength'] >= min_strength:
                        pattern['index'] = i
                        pattern['timestamp'] = current.get('timestamp', datetime.now().isoformat())
                        patterns_found.append(pattern)
            
            # Sort by strength (strongest first)
            patterns_found.sort(key=lambda x: x['strength'], reverse=True)
            
            result_data = {
                "patterns_found": patterns_found,
                "pattern_count": len(patterns_found),
                "data_points_analyzed": len(recent_data),
                "strongest_pattern": patterns_found[0] if patterns_found else None
            }
            
            metadata = {
                "pattern_type": "candlestick",
                "lookback_period": lookback_period,
                "min_strength": min_strength,
                "analyzed_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Candlestick pattern recognition failed: {str(e)}")
    
    def _identify_single_candlestick_patterns(self, o: float, h: float, l: float, c: float, previous: Optional[Dict]) -> List[Dict]:
        """Identify single candlestick patterns"""
        patterns = []
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l
        
        if total_range == 0:
            return patterns
        
        # Doji - small body relative to range
        if body / total_range < 0.1:
            strength = 0.8 if upper_shadow / total_range > 0.3 and lower_shadow / total_range > 0.3 else 0.6
            patterns.append({
                "name": "Doji",
                "type": "reversal",
                "strength": strength,
                "signal": "neutral/reversal"
            })
        
        # Hammer - small body at top, long lower shadow
        if (body / total_range < 0.3 and 
            lower_shadow / total_range > 0.6 and 
            upper_shadow / total_range < 0.1):
            patterns.append({
                "name": "Hammer",
                "type": "bullish_reversal",
                "strength": 0.75,
                "signal": "bullish"
            })
        
        # Shooting Star - small body at bottom, long upper shadow
        if (body / total_range < 0.3 and 
            upper_shadow / total_range > 0.6 and 
            lower_shadow / total_range < 0.1):
            patterns.append({
                "name": "Shooting Star",
                "type": "bearish_reversal",
                "strength": 0.75,
                "signal": "bearish"
            })
        
        # Marubozu - no shadows, all body
        if upper_shadow / total_range < 0.05 and lower_shadow / total_range < 0.05:
            signal = "bullish" if c > o else "bearish"
            patterns.append({
                "name": "Marubozu",
                "type": "continuation",
                "strength": 0.7,
                "signal": signal
            })
        
        return patterns
    
    def _identify_multi_candlestick_patterns(self, candles: List[Dict]) -> List[Dict]:
        """Identify multi-candlestick patterns"""
        patterns = []
        
        if len(candles) < 2:
            return patterns
        
        # Engulfing patterns
        if len(candles) >= 2:
            prev_candle = candles[-2]
            curr_candle = candles[-1]
            
            prev_o, prev_c = float(prev_candle['open']), float(prev_candle['close'])
            curr_o, curr_c = float(curr_candle['open']), float(curr_candle['close'])
            
            # Bullish Engulfing
            if (prev_c < prev_o and curr_c > curr_o and  # prev bearish, curr bullish
                curr_o < prev_c and curr_c > prev_o):    # current engulfs previous
                patterns.append({
                    "name": "Bullish Engulfing",
                    "type": "bullish_reversal",
                    "strength": 0.8,
                    "signal": "bullish"
                })
            
            # Bearish Engulfing
            if (prev_c > prev_o and curr_c < curr_o and  # prev bullish, curr bearish
                curr_o > prev_c and curr_c < prev_o):    # current engulfs previous
                patterns.append({
                    "name": "Bearish Engulfing",
                    "type": "bearish_reversal",
                    "strength": 0.8,
                    "signal": "bearish"
                })
        
        # Morning/Evening Star patterns (3 candles)
        if len(candles) >= 3:
            first, second, third = candles[-3], candles[-2], candles[-1]
            
            first_o, first_c = float(first['open']), float(first['close'])
            second_h, second_l = float(second['high']), float(second['low'])
            third_o, third_c = float(third['open']), float(third['close'])
            
            # Morning Star (bullish reversal)
            if (first_c < first_o and  # First candle bearish
                third_c > third_o and  # Third candle bullish
                second_h < min(first_c, third_o) and  # Second candle gaps below
                third_c > (first_o + first_c) / 2):  # Third closes above midpoint of first
                patterns.append({
                    "name": "Morning Star",
                    "type": "bullish_reversal",
                    "strength": 0.85,
                    "signal": "bullish"
                })
            
            # Evening Star (bearish reversal)
            if (first_c > first_o and  # First candle bullish
                third_c < third_o and  # Third candle bearish
                second_l > max(first_c, third_o) and  # Second candle gaps above
                third_c < (first_o + first_c) / 2):  # Third closes below midpoint of first
                patterns.append({
                    "name": "Evening Star",
                    "type": "bearish_reversal",
                    "strength": 0.85,
                    "signal": "bearish"
                })
        
        return patterns
    
    async def _recognize_trend_patterns(self, data: List[Dict], lookback_period: int, min_strength: float) -> ToolResult:
        """Recognize trend patterns"""
        try:
            recent_data = data[-lookback_period:] if len(data) > lookback_period else data
            closes = [float(d['close']) for d in recent_data]
            
            if len(closes) < 5:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Insufficient data for trend analysis (minimum 5 points required)"
                )
            
            patterns_found = []
            
            # Trend direction analysis
            short_ma = sum(closes[-5:]) / 5  # 5-period MA
            medium_ma = sum(closes[-10:]) / 10 if len(closes) >= 10 else short_ma  # 10-period MA
            
            current_price = closes[-1]
            price_change = (current_price - closes[0]) / closes[0]
            
            # Identify trend strength
            trend_strength = min(abs(price_change) * 10, 1.0)  # Scale to 0-1
            
            if trend_strength >= min_strength:
                if price_change > 0.02:  # 2% increase
                    patterns_found.append({
                        "name": "Uptrend",
                        "type": "bullish_trend",
                        "strength": trend_strength,
                        "signal": "bullish",
                        "price_change": price_change * 100
                    })
                elif price_change < -0.02:  # 2% decrease
                    patterns_found.append({
                        "name": "Downtrend",
                        "type": "bearish_trend",
                        "strength": trend_strength,
                        "signal": "bearish",
                        "price_change": price_change * 100
                    })
            
            # Moving average crossover
            if current_price > short_ma > medium_ma:
                patterns_found.append({
                    "name": "Bullish MA Alignment",
                    "type": "bullish_momentum",
                    "strength": 0.7,
                    "signal": "bullish"
                })
            elif current_price < short_ma < medium_ma:
                patterns_found.append({
                    "name": "Bearish MA Alignment",
                    "type": "bearish_momentum",
                    "strength": 0.7,
                    "signal": "bearish"
                })
            
            result_data = {
                "patterns_found": patterns_found,
                "pattern_count": len(patterns_found),
                "current_price": current_price,
                "short_ma": short_ma,
                "medium_ma": medium_ma,
                "overall_trend": "bullish" if price_change > 0 else "bearish"
            }
            
            metadata = {
                "pattern_type": "trend",
                "lookback_period": lookback_period,
                "analyzed_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Trend pattern recognition failed: {str(e)}")
    
    async def _recognize_support_resistance(self, data: List[Dict], lookback_period: int, min_strength: float) -> ToolResult:
        """Recognize support and resistance levels"""
        try:
            recent_data = data[-lookback_period:] if len(data) > lookback_period else data
            highs = [float(d['high']) for d in recent_data]
            lows = [float(d['low']) for d in recent_data]
            closes = [float(d['close']) for d in recent_data]
            
            current_price = closes[-1]
            patterns_found = []
            
            # Find significant highs and lows
            resistance_levels = self._find_resistance_levels(highs, current_price, min_strength)
            support_levels = self._find_support_levels(lows, current_price, min_strength)
            
            patterns_found.extend(resistance_levels)
            patterns_found.extend(support_levels)
            
            result_data = {
                "patterns_found": patterns_found,
                "pattern_count": len(patterns_found),
                "current_price": current_price,
                "nearest_resistance": min([p['level'] for p in resistance_levels if p['level'] > current_price], default=None),
                "nearest_support": max([p['level'] for p in support_levels if p['level'] < current_price], default=None)
            }
            
            metadata = {
                "pattern_type": "support_resistance",
                "lookback_period": lookback_period,
                "analyzed_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Support/resistance recognition failed: {str(e)}")
    
    def _find_resistance_levels(self, highs: List[float], current_price: float, min_strength: float) -> List[Dict]:
        """Find resistance levels from highs"""
        levels = []
        max_high = max(highs)
        recent_high = max(highs[-5:])  # Recent high
        
        # Major resistance (all-time high in period)
        if max_high > current_price:
            strength = min(0.9, (max_high - current_price) / current_price + 0.5)
            if strength >= min_strength:
                levels.append({
                    "name": "Major Resistance",
                    "type": "resistance",
                    "level": max_high,
                    "strength": strength,
                    "signal": "bearish"
                })
        
        # Recent resistance
        if recent_high != max_high and recent_high > current_price:
            strength = min(0.8, (recent_high - current_price) / current_price + 0.4)
            if strength >= min_strength:
                levels.append({
                    "name": "Recent Resistance",
                    "type": "resistance",
                    "level": recent_high,
                    "strength": strength,
                    "signal": "bearish"
                })
        
        return levels
    
    def _find_support_levels(self, lows: List[float], current_price: float, min_strength: float) -> List[Dict]:
        """Find support levels from lows"""
        levels = []
        min_low = min(lows)
        recent_low = min(lows[-5:])  # Recent low
        
        # Major support (all-time low in period)
        if min_low < current_price:
            strength = min(0.9, (current_price - min_low) / current_price + 0.5)
            if strength >= min_strength:
                levels.append({
                    "name": "Major Support",
                    "type": "support",
                    "level": min_low,
                    "strength": strength,
                    "signal": "bullish"
                })
        
        # Recent support
        if recent_low != min_low and recent_low < current_price:
            strength = min(0.8, (current_price - recent_low) / current_price + 0.4)
            if strength >= min_strength:
                levels.append({
                    "name": "Recent Support",
                    "type": "support",
                    "level": recent_low,
                    "strength": strength,
                    "signal": "bullish"
                })
        
        return levels
    
    async def _recognize_markov_states(self, data: List[Dict], lookback_period: int, symbol: str) -> ToolResult:
        """Recognize Markov chain states (simplified implementation)"""
        try:
            recent_data = data[-lookback_period:] if len(data) > lookback_period else data
            closes = [float(d['close']) for d in recent_data]
            volumes = [float(d.get('volume', 0)) for d in recent_data]
            
            if len(closes) < 3:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Insufficient data for Markov analysis (minimum 3 points required)"
                )
            
            # Calculate returns and volatility
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            volatility = sum(abs(r) for r in returns) / len(returns) if returns else 0
            
            # Calculate volume trend
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            recent_volume = volumes[-1] if volumes else 0
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Determine Markov state based on price action and volume
            current_return = returns[-1] if returns else 0
            state_patterns = []
            
            # High volatility, high volume - trending state
            if volatility > 0.02 and volume_ratio > 1.2:
                if current_return > 0:
                    state_patterns.append({
                        "name": "Bullish Trending",
                        "type": "markov_state",
                        "state": "bull_trend",
                        "strength": 0.8,
                        "signal": "bullish",
                        "volatility": volatility,
                        "volume_ratio": volume_ratio
                    })
                else:
                    state_patterns.append({
                        "name": "Bearish Trending",
                        "type": "markov_state",
                        "state": "bear_trend",
                        "strength": 0.8,
                        "signal": "bearish",
                        "volatility": volatility,
                        "volume_ratio": volume_ratio
                    })
            
            # Low volatility, normal volume - consolidation state
            elif volatility < 0.01:
                state_patterns.append({
                    "name": "Consolidation",
                    "type": "markov_state",
                    "state": "consolidation",
                    "strength": 0.7,
                    "signal": "neutral",
                    "volatility": volatility,
                    "volume_ratio": volume_ratio
                })
            
            # Medium volatility - transition state
            else:
                state_patterns.append({
                    "name": "Transition",
                    "type": "markov_state",
                    "state": "transition",
                    "strength": 0.6,
                    "signal": "uncertain",
                    "volatility": volatility,
                    "volume_ratio": volume_ratio
                })
            
            result_data = {
                "patterns_found": state_patterns,
                "current_state": state_patterns[0]["state"] if state_patterns else "unknown",
                "volatility": volatility,
                "volume_ratio": volume_ratio,
                "current_return": current_return,
                "symbol": symbol
            }
            
            metadata = {
                "pattern_type": "markov",
                "lookback_period": lookback_period,
                "analyzed_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Markov state recognition failed: {str(e)}")
    
    async def _recognize_fibonacci_patterns(self, data: List[Dict], lookback_period: int, min_strength: float) -> ToolResult:
        """Recognize Fibonacci retracement and extension patterns with Golden Zone emphasis"""
        try:
            recent_data = data[-lookback_period:] if len(data) > lookback_period else data
            closes = [float(d['close']) for d in recent_data]
            highs = [float(d['high']) for d in recent_data]
            lows = [float(d['low']) for d in recent_data]
            
            if len(closes) < 10:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Insufficient data for Fibonacci analysis (minimum 10 points required)"
                )
            
            patterns_found = []
            current_price = closes[-1]
            
            # Find significant swing high and low for Fibonacci retracement
            swing_high = max(highs)
            swing_low = min(lows)
            swing_range = swing_high - swing_low
            
            if swing_range == 0:
                return ToolResult(success=False, data=None, error="No price range for Fibonacci analysis")
            
            # Standard Fibonacci retracement levels
            fib_levels = {
                0.236: "23.6% Retracement",
                0.382: "38.2% Retracement", 
                0.500: "50% Retracement",
                0.618: "Golden Ratio (61.8%)",  # Golden Zone
                0.786: "78.6% Retracement"
            }
            
            # Fibonacci extension levels
            fib_extensions = {
                1.272: "127.2% Extension",
                1.414: "141.4% Extension", 
                1.618: "Golden Ratio Extension (161.8%)",
                2.000: "200% Extension"
            }
            
            # Calculate retracement levels (assuming recent move from low to high)
            trend_direction = "bullish" if closes[-1] > closes[0] else "bearish"
            
            for ratio, name in fib_levels.items():
                if trend_direction == "bullish":
                    # Bullish trend - retracement from high
                    fib_level = swing_high - (swing_range * ratio)
                    distance_from_current = abs(current_price - fib_level) / current_price
                else:
                    # Bearish trend - retracement from low  
                    fib_level = swing_low + (swing_range * ratio)
                    distance_from_current = abs(current_price - fib_level) / current_price
                
                # Check if current price is near this level (within 2%)
                if distance_from_current < 0.02:
                    strength = 0.9 if ratio == 0.618 else 0.7  # Golden Zone gets higher strength
                    if strength >= min_strength:
                        patterns_found.append({
                            "name": name,
                            "type": "fibonacci_retracement",
                            "level": fib_level,
                            "ratio": ratio,
                            "strength": strength,
                            "signal": "reversal_zone",
                            "is_golden_zone": ratio == 0.618,
                            "distance_percent": distance_from_current * 100
                        })
            
            # Calculate extension levels for continuation patterns
            if trend_direction == "bullish":
                base_move = swing_high - swing_low
                for ratio, name in fib_extensions.items():
                    extension_level = swing_high + (base_move * (ratio - 1))
                    distance_from_current = abs(current_price - extension_level) / current_price
                    
                    if distance_from_current < 0.03:  # Within 3% for extensions
                        strength = 0.8 if ratio == 1.618 else 0.6
                        if strength >= min_strength:
                            patterns_found.append({
                                "name": name,
                                "type": "fibonacci_extension",
                                "level": extension_level,
                                "ratio": ratio,
                                "strength": strength,
                                "signal": "continuation_target",
                                "is_golden_extension": ratio == 1.618
                            })
            
            # Golden Zone confluence analysis
            golden_zone_fib = swing_high - (swing_range * 0.618) if trend_direction == "bullish" else swing_low + (swing_range * 0.618)
            golden_zone_distance = abs(current_price - golden_zone_fib) / current_price
            
            if golden_zone_distance < 0.015:  # Within 1.5% of Golden Zone
                patterns_found.append({
                    "name": "Fibonacci Golden Zone",
                    "type": "high_probability_reversal",
                    "level": golden_zone_fib,
                    "ratio": 0.618,
                    "strength": 0.95,
                    "signal": "strong_reversal_zone",
                    "is_golden_zone": True,
                    "confluence_note": "High probability reversal zone - ideal for entry"
                })
            
            result_data = {
                "patterns_found": patterns_found,
                "swing_high": swing_high,
                "swing_low": swing_low,
                "current_price": current_price,
                "trend_direction": trend_direction,
                "golden_zone_level": golden_zone_fib,
                "in_golden_zone": golden_zone_distance < 0.015,
                "fibonacci_summary": {
                    "total_levels": len(patterns_found),
                    "golden_zone_proximity": golden_zone_distance * 100,
                    "strongest_level": max(patterns_found, key=lambda x: x["strength"]) if patterns_found else None
                }
            }
            
            metadata = {
                "pattern_type": "fibonacci",
                "analysis_type": "retracements_and_extensions",
                "swing_analysis": f"{swing_low:.2f} - {swing_high:.2f}",
                "analyzed_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Fibonacci pattern recognition failed: {str(e)}")
    
    async def _recognize_elliott_wave_patterns(self, data: List[Dict], lookback_period: int, min_strength: float) -> ToolResult:
        """Recognize Elliott Wave patterns and wave counts"""
        try:
            recent_data = data[-lookback_period:] if len(data) > lookback_period else data
            closes = [float(d['close']) for d in recent_data]
            highs = [float(d['high']) for d in recent_data]
            lows = [float(d['low']) for d in recent_data]
            
            if len(closes) < 15:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Insufficient data for Elliott Wave analysis (minimum 15 points required)"
                )
            
            patterns_found = []
            
            # Find swing points (simplified pivot detection)
            swing_points = self._identify_swing_points(highs, lows, closes)
            
            if len(swing_points) < 5:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Insufficient swing points for Elliott Wave analysis"
                )
            
            # Analyze for 5-wave impulse pattern
            impulse_waves = self._analyze_impulse_waves(swing_points)
            if impulse_waves and impulse_waves["confidence"] >= min_strength:
                patterns_found.append(impulse_waves)
            
            # Analyze for 3-wave corrective pattern  
            corrective_waves = self._analyze_corrective_waves(swing_points)
            if corrective_waves and corrective_waves["confidence"] >= min_strength:
                patterns_found.append(corrective_waves)
            
            # Check for wave completion signals
            current_price = closes[-1]
            wave_signals = self._identify_wave_completion_signals(swing_points, current_price)
            patterns_found.extend(wave_signals)
            
            result_data = {
                "patterns_found": patterns_found,
                "swing_points": swing_points[-10:],  # Last 10 swing points
                "current_price": current_price,
                "wave_count_summary": {
                    "impulse_waves_detected": len([p for p in patterns_found if p.get("wave_type") == "impulse"]),
                    "corrective_waves_detected": len([p for p in patterns_found if p.get("wave_type") == "corrective"]),
                    "completion_signals": len([p for p in patterns_found if "completion" in p.get("name", "").lower()])
                },
                "elliott_analysis": self._get_elliott_wave_forecast(swing_points, current_price)
            }
            
            metadata = {
                "pattern_type": "elliott_wave",
                "swing_points_analyzed": len(swing_points),
                "analyzed_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Elliott Wave pattern recognition failed: {str(e)}")
    
    def _identify_swing_points(self, highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
        """Identify swing highs and lows"""
        swing_points = []
        lookback = 3  # Look 3 periods back and forward
        
        for i in range(lookback, len(highs) - lookback):
            # Check for swing high
            if all(highs[i] >= highs[j] for j in range(i-lookback, i+lookback+1) if j != i):
                swing_points.append({
                    "index": i,
                    "price": highs[i],
                    "type": "high",
                    "timestamp": i
                })
            
            # Check for swing low
            elif all(lows[i] <= lows[j] for j in range(i-lookback, i+lookback+1) if j != i):
                swing_points.append({
                    "index": i,
                    "price": lows[i],
                    "type": "low", 
                    "timestamp": i
                })
        
        return sorted(swing_points, key=lambda x: x["index"])
    
    def _analyze_impulse_waves(self, swing_points: List[Dict]) -> Optional[Dict]:
        """Analyze for 5-wave impulse pattern"""
        if len(swing_points) < 6:
            return None
        
        # Take last 6 swing points to form 5 waves
        recent_swings = swing_points[-6:]
        
        # Check Elliott Wave rules for impulse waves
        # Wave 2 never retraces more than 100% of Wave 1
        # Wave 4 never retraces into Wave 1 territory  
        # Wave 3 is never the shortest wave
        
        try:
            wave_1 = abs(recent_swings[1]["price"] - recent_swings[0]["price"])
            wave_2 = abs(recent_swings[2]["price"] - recent_swings[1]["price"])
            wave_3 = abs(recent_swings[3]["price"] - recent_swings[2]["price"])
            wave_4 = abs(recent_swings[4]["price"] - recent_swings[3]["price"])
            wave_5 = abs(recent_swings[5]["price"] - recent_swings[4]["price"])
            
            # Elliott Wave validation
            rule_violations = 0
            
            # Rule 1: Wave 2 retracement check (should be < 100%)
            if wave_2 >= wave_1:
                rule_violations += 1
            
            # Rule 2: Wave 3 should not be shortest
            if wave_3 < wave_1 and wave_3 < wave_5:
                rule_violations += 1
            
            # Rule 3: Wave 4 territory check (simplified)
            if wave_4 >= wave_3 * 0.8:  # Wave 4 shouldn't retrace too much
                rule_violations += 1
            
            confidence = max(0.0, 1.0 - (rule_violations * 0.3))
            
            if confidence > 0.4:
                return {
                    "name": "Elliott Impulse Wave Pattern",
                    "wave_type": "impulse",
                    "strength": confidence,
                    "confidence": confidence,
                    "signal": "trend_continuation" if confidence > 0.7 else "weak_trend",
                    "wave_structure": f"1-2-3-4-5 Wave Pattern",
                    "rule_violations": rule_violations,
                    "wave_measurements": {
                        "wave_1": wave_1,
                        "wave_2": wave_2,
                        "wave_3": wave_3,
                        "wave_4": wave_4,
                        "wave_5": wave_5
                    }
                }
        except (IndexError, ZeroDivisionError):
            pass
        
        return None
    
    def _analyze_corrective_waves(self, swing_points: List[Dict]) -> Optional[Dict]:
        """Analyze for 3-wave corrective pattern (A-B-C)"""
        if len(swing_points) < 4:
            return None
        
        recent_swings = swing_points[-4:]
        
        try:
            wave_a = abs(recent_swings[1]["price"] - recent_swings[0]["price"])
            wave_b = abs(recent_swings[2]["price"] - recent_swings[1]["price"])
            wave_c = abs(recent_swings[3]["price"] - recent_swings[2]["price"])
            
            # Corrective wave characteristics
            # Wave B typically retraces 50-78.6% of Wave A
            # Wave C often equals Wave A or extends to 1.618 * Wave A
            
            b_retracement_ratio = wave_b / wave_a if wave_a > 0 else 0
            c_to_a_ratio = wave_c / wave_a if wave_a > 0 else 0
            
            confidence = 0.5
            
            # Good B wave retracement (50-78.6% of A)
            if 0.5 <= b_retracement_ratio <= 0.786:
                confidence += 0.2
            
            # C wave equals A or is 1.618 extension
            if abs(c_to_a_ratio - 1.0) < 0.1 or abs(c_to_a_ratio - 1.618) < 0.1:
                confidence += 0.3
            
            if confidence >= 0.6:
                return {
                    "name": "Elliott Corrective Wave Pattern",
                    "wave_type": "corrective", 
                    "strength": confidence,
                    "confidence": confidence,
                    "signal": "correction_completion" if confidence > 0.8 else "correction_in_progress",
                    "wave_structure": "A-B-C Correction",
                    "retracement_ratio": b_retracement_ratio,
                    "c_wave_extension": c_to_a_ratio,
                    "wave_measurements": {
                        "wave_a": wave_a,
                        "wave_b": wave_b,
                        "wave_c": wave_c
                    }
                }
        except (IndexError, ZeroDivisionError):
            pass
        
        return None
    
    def _identify_wave_completion_signals(self, swing_points: List[Dict], current_price: float) -> List[Dict]:
        """Identify potential wave completion signals"""
        signals = []
        
        if len(swing_points) < 3:
            return signals
        
        # Check if current price suggests wave completion
        last_swing = swing_points[-1]
        previous_swing = swing_points[-2]
        
        # Distance from last swing point
        distance_from_swing = abs(current_price - last_swing["price"]) / last_swing["price"]
        
        # If price has moved significantly from last swing, might be starting new wave
        if distance_from_swing > 0.03:  # 3% move
            wave_direction = "up" if current_price > last_swing["price"] else "down"
            
            signals.append({
                "name": f"Potential Wave {wave_direction.title()} Initiation",
                "type": "wave_signal",
                "strength": min(0.8, distance_from_swing * 10),
                "signal": f"new_wave_{wave_direction}",
                "distance_from_last_swing": distance_from_swing * 100,
                "current_price": current_price,
                "last_swing_price": last_swing["price"]
            })
        
        return signals
    
    def _get_elliott_wave_forecast(self, swing_points: List[Dict], current_price: float) -> Dict:
        """Get Elliott Wave forecast based on current structure"""
        if len(swing_points) < 3:
            return {"forecast": "insufficient_data"}
        
        recent_trend = "up" if swing_points[-1]["price"] > swing_points[-3]["price"] else "down"
        wave_count_estimate = len(swing_points) % 8  # Elliott waves in 8-wave cycles (5 impulse + 3 corrective)
        
        return {
            "current_trend": recent_trend,
            "estimated_wave_position": wave_count_estimate,
            "cycle_progress": f"{wave_count_estimate}/8",
            "forecast": f"Wave {wave_count_estimate} of cycle",
            "next_expected": "corrective_phase" if wave_count_estimate >= 5 else "impulse_continuation"
        }
    
    async def _recognize_wyckoff_patterns(self, data: List[Dict], lookback_period: int, min_strength: float) -> ToolResult:
        """Recognize Wyckoff method patterns - accumulation, distribution, and markup/markdown"""
        try:
            recent_data = data[-lookback_period:] if len(data) > lookback_period else data
            closes = [float(d['close']) for d in recent_data]
            highs = [float(d['high']) for d in recent_data]
            lows = [float(d['low']) for d in recent_data]
            volumes = [float(d.get('volume', 0)) for d in recent_data if d.get('volume')]
            
            if len(closes) < 20:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Insufficient data for Wyckoff analysis (minimum 20 points required)"
                )
            
            patterns_found = []
            current_price = closes[-1]
            
            # Volume analysis (if available)
            volume_analysis = None
            if volumes and len(volumes) >= len(closes):
                volume_analysis = self._analyze_volume_price_relationship(closes, volumes)
            
            # Identify Wyckoff phases
            wyckoff_phases = self._identify_wyckoff_phases(closes, highs, lows, volume_analysis)
            patterns_found.extend(wyckoff_phases)
            
            # Look for specific Wyckoff events
            wyckoff_events = self._identify_wyckoff_events(closes, highs, lows, volume_analysis)
            patterns_found.extend(wyckoff_events)
            
            # Determine current Wyckoff cycle position
            cycle_analysis = self._analyze_wyckoff_cycle(closes, volume_analysis)
            
            result_data = {
                "patterns_found": patterns_found,
                "current_price": current_price,
                "volume_analysis": volume_analysis,
                "wyckoff_cycle": cycle_analysis,
                "phase_summary": {
                    "accumulation_signals": len([p for p in patterns_found if "accumulation" in p.get("name", "").lower()]),
                    "distribution_signals": len([p for p in patterns_found if "distribution" in p.get("name", "").lower()]),
                    "markup_signals": len([p for p in patterns_found if "markup" in p.get("name", "").lower()]),
                    "markdown_signals": len([p for p in patterns_found if "markdown" in p.get("name", "").lower()])
                }
            }
            
            metadata = {
                "pattern_type": "wyckoff",
                "has_volume_data": volume_analysis is not None,
                "analyzed_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Wyckoff pattern recognition failed: {str(e)}")
    
    def _analyze_volume_price_relationship(self, closes: List[float], volumes: List[float]) -> Dict:
        """Analyze volume-price relationship for Wyckoff analysis"""
        if len(closes) != len(volumes) or len(closes) < 10:
            return {"status": "insufficient_data"}
        
        # Calculate price changes and volume ratios
        price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        volume_ratios = []
        avg_volume = sum(volumes) / len(volumes)
        
        for i in range(1, len(volumes)):
            volume_ratios.append(volumes[i] / avg_volume)
        
        # Analyze volume-price divergences
        volume_price_correlation = []
        for i in range(min(len(price_changes), len(volume_ratios))):
            price_direction = 1 if price_changes[i] > 0 else -1
            volume_strength = 1 if volume_ratios[i] > 1.2 else -1 if volume_ratios[i] < 0.8 else 0
            
            # Wyckoff principle: Price up + Volume up = Healthy, Price up + Volume down = Weakness
            correlation_score = price_direction * volume_strength
            volume_price_correlation.append(correlation_score)
        
        return {
            "status": "analyzed",
            "avg_volume": avg_volume,
            "current_volume_ratio": volume_ratios[-1] if volume_ratios else 1.0,
            "volume_price_correlation": volume_price_correlation[-10:],  # Last 10 periods
            "recent_volume_trend": "increasing" if volume_ratios[-3:] and all(v > 1.0 for v in volume_ratios[-3:]) else "decreasing"
        }
    
    def _identify_wyckoff_phases(self, closes: List[float], highs: List[float], lows: List[float], volume_analysis: Optional[Dict]) -> List[Dict]:
        """Identify Wyckoff accumulation, distribution, markup, and markdown phases"""
        phases = []
        
        # Calculate price volatility and trend strength
        price_range = max(highs) - min(lows)
        recent_range = max(highs[-10:]) - min(lows[-10:])
        volatility_ratio = recent_range / price_range if price_range > 0 else 0
        
        # Trend analysis
        short_trend = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        long_trend = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
        
        # Accumulation Phase Detection
        # Characteristics: Sideways price action, increasing volume on up days
        if abs(short_trend) < 0.03 and volatility_ratio < 0.5:  # Low price movement
            volume_confirmation = True
            if volume_analysis and volume_analysis.get("status") == "analyzed":
                # Check for volume expansion on up moves
                correlations = volume_analysis.get("volume_price_correlation", [])
                positive_correlations = [c for c in correlations[-5:] if c > 0]
                volume_confirmation = len(positive_correlations) >= 3
            
            if volume_confirmation:
                phases.append({
                    "name": "Wyckoff Accumulation Phase",
                    "type": "accumulation",
                    "strength": 0.8,
                    "signal": "bullish_accumulation",
                    "characteristics": ["sideways_price_action", "volume_expansion_on_rallies"],
                    "trend_strength": abs(short_trend)
                })
        
        # Distribution Phase Detection
        # Characteristics: Sideways price action at highs, volume expansion on down days
        elif abs(short_trend) < 0.03 and closes[-1] > closes[-20] * 0.95:  # Near recent highs
            volume_confirmation = True
            if volume_analysis and volume_analysis.get("status") == "analyzed":
                correlations = volume_analysis.get("volume_price_correlation", [])
                negative_correlations = [c for c in correlations[-5:] if c < 0]
                volume_confirmation = len(negative_correlations) >= 3
            
            if volume_confirmation:
                phases.append({
                    "name": "Wyckoff Distribution Phase",
                    "type": "distribution",
                    "strength": 0.8,
                    "signal": "bearish_distribution",
                    "characteristics": ["sideways_price_action_at_highs", "volume_expansion_on_declines"]
                })
        
        # Markup Phase Detection
        # Characteristics: Strong uptrend, volume confirmation
        elif short_trend > 0.05:  # Strong upward movement
            phases.append({
                "name": "Wyckoff Markup Phase",
                "type": "markup", 
                "strength": min(0.9, short_trend * 10),
                "signal": "strong_uptrend",
                "trend_strength": short_trend
            })
        
        # Markdown Phase Detection  
        # Characteristics: Strong downtrend
        elif short_trend < -0.05:  # Strong downward movement
            phases.append({
                "name": "Wyckoff Markdown Phase",
                "type": "markdown",
                "strength": min(0.9, abs(short_trend) * 10), 
                "signal": "strong_downtrend",
                "trend_strength": short_trend
            })
        
        return phases
    
    def _identify_wyckoff_events(self, closes: List[float], highs: List[float], lows: List[float], volume_analysis: Optional[Dict]) -> List[Dict]:
        """Identify specific Wyckoff events like Springs, Upthrusts, etc."""
        events = []
        
        if len(closes) < 10:
            return events
        
        recent_high = max(highs[-10:])
        recent_low = min(lows[-10:])
        current_price = closes[-1]
        
        # Spring Event (False breakdown below support)
        support_level = min(lows[-20:-5]) if len(lows) >= 20 else recent_low
        if current_price < support_level and closes[-5] > support_level:
            # Price broke below support but recovered
            events.append({
                "name": "Wyckoff Spring (False Breakdown)",
                "type": "spring_event",
                "strength": 0.85,
                "signal": "bullish_reversal",
                "support_level": support_level,
                "current_price": current_price,
                "event_description": "False breakdown below support followed by recovery"
            })
        
        # Upthrust Event (False breakout above resistance)
        resistance_level = max(highs[-20:-5]) if len(highs) >= 20 else recent_high
        if current_price > resistance_level and closes[-5] < resistance_level:
            # Price broke above resistance but failed
            events.append({
                "name": "Wyckoff Upthrust (False Breakout)", 
                "type": "upthrust_event",
                "strength": 0.85,
                "signal": "bearish_reversal",
                "resistance_level": resistance_level,
                "current_price": current_price,
                "event_description": "False breakout above resistance followed by failure"
            })
        
        return events
    
    def _analyze_wyckoff_cycle(self, closes: List[float], volume_analysis: Optional[Dict]) -> Dict:
        """Analyze current position in Wyckoff cycle"""
        if len(closes) < 20:
            return {"cycle_position": "insufficient_data"}
        
        # Calculate relative position in recent range
        recent_high = max(closes[-20:])
        recent_low = min(closes[-20:])
        current_price = closes[-1]
        range_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
        
        # Determine cycle phase based on price position and trend
        long_trend = (closes[-1] - closes[-20]) / closes[-20]
        
        if range_position < 0.3 and long_trend < -0.1:
            cycle_phase = "markdown_completion"
            next_phase = "accumulation"
        elif range_position < 0.3 and abs(long_trend) < 0.05:
            cycle_phase = "accumulation"
            next_phase = "markup"
        elif range_position > 0.7 and long_trend > 0.1:
            cycle_phase = "markup"
            next_phase = "distribution"
        elif range_position > 0.7 and abs(long_trend) < 0.05:
            cycle_phase = "distribution"
            next_phase = "markdown"
        else:
            cycle_phase = "transition"
            next_phase = "uncertain"
        
        return {
            "cycle_position": cycle_phase,
            "next_expected_phase": next_phase,
            "range_position_percent": range_position * 100,
            "trend_strength": long_trend,
            "confidence": 0.7 if cycle_phase != "transition" else 0.4
        }
    
    async def _analyze_confluence_patterns(self, data: List[Dict], lookback_period: int, min_strength: float, symbol: str) -> ToolResult:
        """Analyze confluence of Fibonacci, Elliott Wave, and Wyckoff patterns for high-probability setups"""
        try:
            # Get individual analyses
            fib_result = await self._recognize_fibonacci_patterns(data, lookback_period, min_strength)
            elliott_result = await self._recognize_elliott_wave_patterns(data, lookback_period, min_strength)
            wyckoff_result = await self._recognize_wyckoff_patterns(data, lookback_period, min_strength)
            
            if not all([fib_result.success, elliott_result.success, wyckoff_result.success]):
                return ToolResult(
                    success=False,
                    data=None,
                    error="Failed to complete one or more confluence analyses"
                )
            
            confluence_signals = []
            current_price = float(data[-1]['close'])
            
            # Extract key levels and signals
            fib_data = fib_result.data
            elliott_data = elliott_result.data  
            wyckoff_data = wyckoff_result.data
            
            # Golden Zone Confluence Analysis
            if fib_data.get("in_golden_zone"):
                golden_zone_level = fib_data["golden_zone_level"]
                confluence_factors = ["fibonacci_golden_zone"]
                confidence_score = 0.9  # Start with high Fibonacci confidence
                
                # Check Elliott Wave confirmation
                elliott_patterns = elliott_data.get("patterns_found", [])
                corrective_waves = [p for p in elliott_patterns if p.get("wave_type") == "corrective"]
                if corrective_waves:
                    confluence_factors.append("elliott_wave_correction")
                    confidence_score += 0.1
                
                # Check Wyckoff confirmation
                wyckoff_patterns = wyckoff_data.get("patterns_found", [])
                accumulation_signals = [p for p in wyckoff_patterns if "accumulation" in p.get("name", "").lower()]
                if accumulation_signals:
                    confluence_factors.append("wyckoff_accumulation")
                    confidence_score += 0.1
                
                # Add volume confirmation if available
                if wyckoff_data.get("volume_analysis", {}).get("status") == "analyzed":
                    confluence_factors.append("volume_confirmation")
                    confidence_score += 0.05
                
                confluence_signals.append({
                    "name": "Golden Zone Confluence Setup",
                    "type": "high_probability_reversal",
                    "level": golden_zone_level,
                    "current_price": current_price,
                    "strength": min(1.0, confidence_score),
                    "confidence": min(1.0, confidence_score),
                    "signal": "strong_reversal_probability",
                    "confluence_factors": confluence_factors,
                    "factor_count": len(confluence_factors),
                    "setup_quality": "excellent" if len(confluence_factors) >= 4 else "good" if len(confluence_factors) >= 3 else "fair",
                    "entry_recommendation": {
                        "zone": f"{golden_zone_level * 0.995:.2f} - {golden_zone_level * 1.005:.2f}",
                        "stop_loss": golden_zone_level * 0.97 if current_price > golden_zone_level else golden_zone_level * 1.03,
                        "targets": [
                            golden_zone_level * 1.05 if current_price < golden_zone_level else golden_zone_level * 0.95,
                            golden_zone_level * 1.10 if current_price < golden_zone_level else golden_zone_level * 0.90
                        ]
                    }
                })
            
            # Multi-Method Trend Confluence
            trend_signals = self._analyze_trend_confluence(fib_data, elliott_data, wyckoff_data, current_price)
            confluence_signals.extend(trend_signals)
            
            # Support/Resistance Confluence
            sr_confluence = self._analyze_support_resistance_confluence(fib_data, elliott_data, wyckoff_data, current_price)
            confluence_signals.extend(sr_confluence)
            
            # Overall market structure assessment
            market_structure = self._assess_market_structure(fib_data, elliott_data, wyckoff_data)
            
            result_data = {
                "confluence_signals": confluence_signals,
                "signal_count": len(confluence_signals),
                "highest_confidence_signal": max(confluence_signals, key=lambda x: x["confidence"]) if confluence_signals else None,
                "market_structure": market_structure,
                "individual_analyses": {
                    "fibonacci": {
                        "in_golden_zone": fib_data.get("in_golden_zone", False),
                        "patterns_found": len(fib_data.get("patterns_found", []))
                    },
                    "elliott_wave": {
                        "patterns_found": len(elliott_data.get("patterns_found", [])),
                        "cycle_progress": elliott_data.get("elliott_analysis", {}).get("cycle_progress")
                    },
                    "wyckoff": {
                        "current_phase": wyckoff_data.get("wyckoff_cycle", {}).get("cycle_position"),
                        "patterns_found": len(wyckoff_data.get("patterns_found", []))
                    }
                },
                "trading_recommendation": self._generate_confluence_trading_recommendation(confluence_signals, market_structure)
            }
            
            metadata = {
                "pattern_type": "confluence",
                "methods_analyzed": ["fibonacci", "elliott_wave", "wyckoff"],
                "confluence_count": len(confluence_signals),
                "analyzed_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Confluence pattern analysis failed: {str(e)}")
    
    def _analyze_trend_confluence(self, fib_data: Dict, elliott_data: Dict, wyckoff_data: Dict, current_price: float) -> List[Dict]:
        """Analyze trend confluence between methods"""
        trend_signals = []
        
        # Get trend signals from each method
        fib_trend = fib_data.get("trend_direction", "neutral")
        
        elliott_patterns = elliott_data.get("patterns_found", [])
        elliott_trend = "bullish" if any("bullish" in p.get("signal", "") for p in elliott_patterns) else "bearish" if any("bearish" in p.get("signal", "") for p in elliott_patterns) else "neutral"
        
        wyckoff_cycle = wyckoff_data.get("wyckoff_cycle", {})
        wyckoff_phase = wyckoff_cycle.get("cycle_position", "neutral")
        wyckoff_trend = "bullish" if wyckoff_phase in ["accumulation", "markup"] else "bearish" if wyckoff_phase in ["distribution", "markdown"] else "neutral"
        
        # Check for trend confluence
        trends = [fib_trend, elliott_trend, wyckoff_trend]
        bullish_count = trends.count("bullish")
        bearish_count = trends.count("bearish")
        
        if bullish_count >= 2:
            confidence = 0.7 + (bullish_count - 2) * 0.1
            trend_signals.append({
                "name": "Multi-Method Bullish Confluence",
                "type": "trend_confluence",
                "strength": confidence,
                "confidence": confidence,
                "signal": "bullish_trend",
                "supporting_methods": [method for method, trend in zip(["fibonacci", "elliott_wave", "wyckoff"], trends) if trend == "bullish"],
                "agreement_level": f"{bullish_count}/3 methods agree"
            })
        
        elif bearish_count >= 2:
            confidence = 0.7 + (bearish_count - 2) * 0.1
            trend_signals.append({
                "name": "Multi-Method Bearish Confluence",
                "type": "trend_confluence", 
                "strength": confidence,
                "confidence": confidence,
                "signal": "bearish_trend",
                "supporting_methods": [method for method, trend in zip(["fibonacci", "elliott_wave", "wyckoff"], trends) if trend == "bearish"],
                "agreement_level": f"{bearish_count}/3 methods agree"
            })
        
        return trend_signals
    
    def _analyze_support_resistance_confluence(self, fib_data: Dict, elliott_data: Dict, wyckoff_data: Dict, current_price: float) -> List[Dict]:
        """Analyze support/resistance confluence"""
        sr_signals = []
        
        # Extract key levels from each method
        fib_levels = [p["level"] for p in fib_data.get("patterns_found", []) if "level" in p]
        
        # Elliott Wave swing points
        elliott_swings = elliott_data.get("swing_points", [])
        elliott_levels = [s["price"] for s in elliott_swings[-5:]]  # Recent swing levels
        
        # Wyckoff events and phase levels
        wyckoff_events = wyckoff_data.get("patterns_found", [])
        wyckoff_levels = [p.get("support_level") or p.get("resistance_level") for p in wyckoff_events if p.get("support_level") or p.get("resistance_level")]
        
        # Find confluence levels (levels within 1% of each other)
        all_levels = fib_levels + elliott_levels + wyckoff_levels
        confluence_tolerance = 0.01  # 1%
        
        for level in all_levels:
            if not level:
                continue
                
            # Find other levels within tolerance
            nearby_levels = [l for l in all_levels if l and abs(l - level) / level < confluence_tolerance and l != level]
            
            if len(nearby_levels) >= 1:  # At least one other level nearby
                avg_level = (level + sum(nearby_levels)) / (len(nearby_levels) + 1)
                confidence = min(0.9, 0.6 + len(nearby_levels) * 0.1)
                
                level_type = "resistance" if avg_level > current_price else "support"
                
                sr_signals.append({
                    "name": f"Multi-Method {level_type.title()} Confluence",
                    "type": f"{level_type}_confluence",
                    "level": avg_level,
                    "strength": confidence,
                    "confidence": confidence,
                    "signal": f"strong_{level_type}",
                    "level_count": len(nearby_levels) + 1,
                    "distance_from_price": abs(avg_level - current_price) / current_price * 100
                })
        
        # Remove duplicates (levels too close to each other)
        unique_sr_signals = []
        for signal in sr_signals:
            is_duplicate = False
            for existing in unique_sr_signals:
                if abs(signal["level"] - existing["level"]) / signal["level"] < confluence_tolerance:
                    # Keep the one with higher confidence
                    if signal["confidence"] > existing["confidence"]:
                        unique_sr_signals.remove(existing)
                        unique_sr_signals.append(signal)
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_sr_signals.append(signal)
        
        return unique_sr_signals
    
    def _assess_market_structure(self, fib_data: Dict, elliott_data: Dict, wyckoff_data: Dict) -> Dict:
        """Assess overall market structure using all methods"""
        structure = {
            "primary_trend": "neutral",
            "trend_strength": "weak", 
            "phase": "uncertain",
            "structure_quality": "poor"
        }
        
        # Fibonacci structure assessment
        fib_trend = fib_data.get("trend_direction", "neutral")
        in_golden_zone = fib_data.get("in_golden_zone", False)
        
        # Elliott Wave structure  
        elliott_forecast = elliott_data.get("elliott_analysis", {})
        wave_position = elliott_forecast.get("estimated_wave_position", 0)
        
        # Wyckoff structure
        wyckoff_cycle = wyckoff_data.get("wyckoff_cycle", {})
        wyckoff_phase = wyckoff_cycle.get("cycle_position", "uncertain")
        
        # Determine primary trend
        trend_votes = []
        if fib_trend != "neutral":
            trend_votes.append(fib_trend)
        if elliott_forecast.get("current_trend"):
            trend_votes.append(elliott_forecast["current_trend"])
        if wyckoff_phase in ["markup", "accumulation"]:
            trend_votes.append("bullish")
        elif wyckoff_phase in ["markdown", "distribution"]:
            trend_votes.append("bearish")
        
        if trend_votes:
            bullish_votes = trend_votes.count("bullish") + trend_votes.count("up")  
            bearish_votes = trend_votes.count("bearish") + trend_votes.count("down")
            
            if bullish_votes > bearish_votes:
                structure["primary_trend"] = "bullish"
                structure["trend_strength"] = "strong" if bullish_votes >= 2 else "moderate"
            elif bearish_votes > bullish_votes:
                structure["primary_trend"] = "bearish"
                structure["trend_strength"] = "strong" if bearish_votes >= 2 else "moderate"
        
        # Determine market phase
        if in_golden_zone:
            structure["phase"] = "reversal_zone"
        elif wyckoff_phase in ["accumulation", "distribution"]:
            structure["phase"] = wyckoff_phase
        elif wave_position in [1, 3, 5]:  # Elliott impulse waves
            structure["phase"] = "impulse"
        elif wave_position in [2, 4]:  # Elliott corrective waves
            structure["phase"] = "correction"
        
        # Assess overall structure quality
        agreement_count = len(set(trend_votes))
        if agreement_count == 1 and len(trend_votes) >= 2:  # High agreement
            structure["structure_quality"] = "excellent"
        elif agreement_count <= 2:
            structure["structure_quality"] = "good"
        
        return structure
    
    def _generate_confluence_trading_recommendation(self, confluence_signals: List[Dict], market_structure: Dict) -> Dict:
        """Generate trading recommendation based on confluence analysis"""
        if not confluence_signals:
            return {
                "recommendation": "no_clear_setup",
                "confidence": "low",
                "rationale": "No significant confluence patterns detected"
            }
        
        highest_confidence = max(confluence_signals, key=lambda x: x["confidence"])
        
        # Golden Zone setups get priority
        golden_zone_setups = [s for s in confluence_signals if "golden_zone" in s.get("name", "").lower()]
        
        if golden_zone_setups:
            setup = golden_zone_setups[0]
            return {
                "recommendation": "high_probability_reversal_trade",
                "confidence": "high" if setup["confidence"] > 0.8 else "medium",
                "setup_type": "fibonacci_golden_zone",
                "entry_zone": setup.get("entry_recommendation", {}).get("zone"),
                "stop_loss": setup.get("entry_recommendation", {}).get("stop_loss"),
                "targets": setup.get("entry_recommendation", {}).get("targets"),
                "rationale": f"Golden Zone confluence with {setup.get('factor_count', 0)} confirming factors",
                "risk_reward": "favorable",
                "timeframe": "swing_trade"
            }
        
        # Trend confluence setups
        elif market_structure.get("trend_strength") in ["strong", "moderate"]:
            trend_direction = market_structure["primary_trend"]
            return {
                "recommendation": f"trend_following_{trend_direction}",
                "confidence": "medium" if market_structure["trend_strength"] == "strong" else "low",
                "setup_type": "trend_confluence",
                "rationale": f"Multi-method {trend_direction} trend confluence",
                "risk_reward": "moderate",
                "timeframe": "medium_term"
            }
        
        else:
            return {
                "recommendation": "wait_for_clearer_setup",
                "confidence": "low",
                "rationale": "Conflicting signals between methods - wait for better confluence",
                "suggested_action": "monitor_for_confluence_development"
            }