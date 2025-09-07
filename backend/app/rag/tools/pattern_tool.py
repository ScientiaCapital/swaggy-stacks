"""
Pattern Recognition Tools for LangChain Integration

Provides chart pattern and candlestick pattern detection as LangChain Tools
"""

import json
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from langchain.agents import Tool
from scipy import stats

logger = logging.getLogger(__name__)


class PatternTool:
    """Pattern recognition tools for technical analysis"""
    
    def get_tools(self) -> List[Tool]:
        """Get all pattern recognition tools"""
        return [
            Tool(
                name="find_chart_patterns",
                description="Find chart patterns like triangles, head and shoulders (symbol required)",
                func=self._find_chart_patterns
            ),
            Tool(
                name="detect_candlestick_patterns",
                description="Detect candlestick patterns like doji, hammer, engulfing",
                func=self._detect_candlestick_patterns
            ),
            Tool(
                name="identify_support_resistance",
                description="Identify key support and resistance levels",
                func=self._identify_support_resistance
            ),
            Tool(
                name="analyze_volume_patterns",
                description="Analyze volume patterns and anomalies",
                func=self._analyze_volume_patterns
            ),
            Tool(
                name="detect_trend_patterns",
                description="Detect trend patterns and potential reversals",
                func=self._detect_trend_patterns
            ),
            Tool(
                name="find_breakout_patterns",
                description="Find potential breakout patterns and levels",
                func=self._find_breakout_patterns
            )
        ]
    
    def _find_chart_patterns(self, symbol: str) -> str:
        """Find chart patterns like triangles, head and shoulders"""
        try:
            symbol = symbol.strip().upper()
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if hist.empty or len(hist) < 50:
                return f"Insufficient data for pattern analysis for {symbol}"
            
            highs = hist['High'].values
            lows = hist['Low'].values
            closes = hist['Close'].values
            
            patterns_found = []
            
            # Triangle pattern detection (simplified)
            patterns_found.extend(self._detect_triangle_patterns(highs, lows))
            
            # Head and shoulders pattern (simplified)
            patterns_found.extend(self._detect_head_shoulders(highs, lows))
            
            # Double top/bottom patterns
            patterns_found.extend(self._detect_double_patterns(highs, lows))
            
            return json.dumps({
                "symbol": symbol,
                "analysis_period": "6 months",
                "patterns_found": patterns_found,
                "pattern_count": len(patterns_found),
                "confidence_levels": {
                    pattern["name"]: pattern["confidence"] 
                    for pattern in patterns_found
                },
                "trading_implications": self._get_pattern_implications(patterns_found)
            })
            
        except Exception as e:
            logger.error(f"Error finding chart patterns: {e}")
            return f"Error finding chart patterns: {str(e)}"
    
    def _detect_candlestick_patterns(self, symbol: str) -> str:
        """Detect candlestick patterns"""
        try:
            symbol = symbol.strip().upper()
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if hist.empty or len(hist) < 10:
                return f"Insufficient data for candlestick analysis for {symbol}"
            
            opens = hist['Open'].values
            highs = hist['High'].values
            lows = hist['Low'].values
            closes = hist['Close'].values
            
            patterns = []
            
            # Analyze last 20 days for patterns
            lookback = min(20, len(hist))
            
            for i in range(lookback):
                idx = -(i + 1)  # Look backward from current
                
                # Current candle data
                o, h, l, c = opens[idx], highs[idx], lows[idx], closes[idx]
                body_size = abs(c - o)
                total_range = h - l
                upper_shadow = h - max(o, c)
                lower_shadow = min(o, c) - l
                
                # Pattern detection
                if body_size / total_range < 0.1 and total_range > 0:  # Doji
                    patterns.append({
                        "pattern": "doji",
                        "date": hist.index[idx].strftime("%Y-%m-%d"),
                        "significance": "indecision",
                        "confidence": 0.8
                    })
                
                elif (lower_shadow > body_size * 2 and 
                      upper_shadow < body_size and 
                      c > o):  # Hammer
                    patterns.append({
                        "pattern": "hammer",
                        "date": hist.index[idx].strftime("%Y-%m-%d"),
                        "significance": "bullish_reversal",
                        "confidence": 0.7
                    })
                
                elif (upper_shadow > body_size * 2 and 
                      lower_shadow < body_size and 
                      o > c):  # Shooting Star
                    patterns.append({
                        "pattern": "shooting_star",
                        "date": hist.index[idx].strftime("%Y-%m-%d"),
                        "significance": "bearish_reversal",
                        "confidence": 0.7
                    })
                
                # Engulfing patterns (requires previous candle)
                if i < lookback - 1:
                    prev_idx = idx - 1
                    prev_o, prev_c = opens[prev_idx], closes[prev_idx]
                    
                    if (c > o and prev_c < prev_o and  # Current bullish, prev bearish
                        c > prev_o and o < prev_c):    # Current engulfs previous
                        patterns.append({
                            "pattern": "bullish_engulfing",
                            "date": hist.index[idx].strftime("%Y-%m-%d"),
                            "significance": "bullish_reversal",
                            "confidence": 0.8
                        })
                    
                    elif (o > c and prev_o < prev_c and  # Current bearish, prev bullish
                          o > prev_c and c < prev_o):    # Current engulfs previous
                        patterns.append({
                            "pattern": "bearish_engulfing",
                            "date": hist.index[idx].strftime("%Y-%m-%d"),
                            "significance": "bearish_reversal",
                            "confidence": 0.8
                        })
            
            return json.dumps({
                "symbol": symbol,
                "candlestick_patterns": patterns[-10:],  # Most recent 10
                "pattern_summary": {
                    "total_patterns": len(patterns),
                    "bullish_signals": len([p for p in patterns if "bullish" in p["significance"]]),
                    "bearish_signals": len([p for p in patterns if "bearish" in p["significance"]]),
                    "indecision_signals": len([p for p in patterns if "indecision" in p["significance"]])
                },
                "recent_signal": patterns[0] if patterns else None
            })
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
            return f"Error detecting candlestick patterns: {str(e)}"
    
    def _identify_support_resistance(self, symbol: str) -> str:
        """Identify support and resistance levels"""
        try:
            symbol = symbol.strip().upper()
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if hist.empty:
                return f"No data available for support/resistance analysis for {symbol}"
            
            highs = hist['High']
            lows = hist['Low']
            closes = hist['Close']
            current_price = float(closes.iloc[-1])
            
            # Find local maxima and minima
            resistance_levels = self._find_local_extrema(highs, "max")
            support_levels = self._find_local_extrema(lows, "min")
            
            # Filter and rank levels
            resistance_levels = [r for r in resistance_levels if r > current_price][:3]
            support_levels = [s for s in support_levels if s < current_price][:3]
            
            # Calculate distances
            nearest_resistance = min(resistance_levels) if resistance_levels else None
            nearest_support = max(support_levels) if support_levels else None
            
            return json.dumps({
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "resistance_levels": [round(r, 2) for r in resistance_levels],
                "support_levels": [round(s, 2) for s in support_levels],
                "key_levels": {
                    "nearest_resistance": round(nearest_resistance, 2) if nearest_resistance else None,
                    "nearest_support": round(nearest_support, 2) if nearest_support else None,
                    "resistance_distance": round(((nearest_resistance - current_price) / current_price) * 100, 2) if nearest_resistance else None,
                    "support_distance": round(((current_price - nearest_support) / current_price) * 100, 2) if nearest_support else None
                },
                "trading_zone": {
                    "in_support_zone": any(abs(current_price - s) / current_price < 0.02 for s in support_levels),
                    "in_resistance_zone": any(abs(current_price - r) / current_price < 0.02 for r in resistance_levels),
                    "trend_bias": "bullish" if nearest_support and (current_price - nearest_support) / current_price < 0.05 else "bearish" if nearest_resistance and (nearest_resistance - current_price) / current_price < 0.05 else "neutral"
                }
            })
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {e}")
            return f"Error identifying support/resistance: {str(e)}"
    
    def _analyze_volume_patterns(self, symbol: str) -> str:
        """Analyze volume patterns"""
        try:
            symbol = symbol.strip().upper()
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if hist.empty:
                return f"No volume data available for {symbol}"
            
            volumes = hist['Volume']
            closes = hist['Close']
            
            # Volume analysis
            avg_volume = volumes.mean()
            recent_volume = volumes.iloc[-1]
            volume_trend = volumes.rolling(20).mean().iloc[-1] / volumes.rolling(50).mean().iloc[-1] if len(volumes) >= 50 else 1
            
            # Price-volume relationship
            price_changes = closes.pct_change()
            volume_price_corr = volumes.corr(price_changes.abs()) if len(volumes) > 10 else 0
            
            # Volume spikes
            volume_threshold = avg_volume * 2
            recent_spikes = volumes.tail(10) > volume_threshold
            spike_dates = recent_spikes[recent_spikes].index.strftime("%Y-%m-%d").tolist()
            
            return json.dumps({
                "symbol": symbol,
                "volume_analysis": {
                    "average_volume": int(avg_volume),
                    "recent_volume": int(recent_volume),
                    "volume_ratio": round(recent_volume / avg_volume, 2),
                    "volume_trend": "increasing" if volume_trend > 1.1 else "decreasing" if volume_trend < 0.9 else "stable"
                },
                "volume_patterns": {
                    "price_volume_correlation": round(volume_price_corr, 3),
                    "volume_spikes_recent": len(spike_dates),
                    "spike_dates": spike_dates,
                    "volume_confirmation": volume_trend > 1 and volume_price_corr > 0.3
                },
                "interpretation": {
                    "trend_strength": "strong" if recent_volume > avg_volume * 1.5 else "moderate" if recent_volume > avg_volume else "weak",
                    "institutional_activity": "high" if recent_volume > avg_volume * 2 else "normal",
                    "breakout_potential": recent_volume > avg_volume * 2 and volume_price_corr > 0.5
                }
            })
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {e}")
            return f"Error analyzing volume patterns: {str(e)}"
    
    def _detect_trend_patterns(self, symbol: str) -> str:
        """Detect trend patterns and potential reversals"""
        try:
            symbol = symbol.strip().upper()
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if hist.empty or len(hist) < 50:
                return f"Insufficient data for trend analysis for {symbol}"
            
            closes = hist['Close']
            
            # Multiple timeframe trend analysis
            short_trend = self._calculate_trend(closes.tail(20))
            medium_trend = self._calculate_trend(closes.tail(50))
            long_trend = self._calculate_trend(closes.tail(100)) if len(closes) >= 100 else medium_trend
            
            # Trend strength
            trend_strength = abs(short_trend["slope_percent"])
            
            # Divergence detection
            price_momentum = closes.pct_change(10).iloc[-1]
            volume_momentum = hist['Volume'].pct_change(10).iloc[-1] if 'Volume' in hist else 0
            
            return json.dumps({
                "symbol": symbol,
                "trend_analysis": {
                    "short_term_trend": short_trend,
                    "medium_term_trend": medium_trend,
                    "long_term_trend": long_trend,
                    "trend_alignment": self._assess_trend_alignment(short_trend, medium_trend, long_trend)
                },
                "trend_characteristics": {
                    "strength": "strong" if trend_strength > 2 else "moderate" if trend_strength > 1 else "weak",
                    "consistency": short_trend["r_squared"],
                    "momentum": "accelerating" if price_momentum > 0 else "decelerating",
                    "volume_confirmation": volume_momentum > 0 and price_momentum > 0
                },
                "reversal_signals": {
                    "trend_exhaustion": trend_strength > 3 and short_trend["r_squared"] < 0.7,
                    "momentum_divergence": (price_momentum > 0) != (volume_momentum > 0),
                    "reversal_probability": self._calculate_reversal_probability(short_trend, medium_trend, trend_strength)
                }
            })
            
        except Exception as e:
            logger.error(f"Error detecting trend patterns: {e}")
            return f"Error detecting trend patterns: {str(e)}"
    
    def _find_breakout_patterns(self, symbol: str) -> str:
        """Find potential breakout patterns"""
        try:
            symbol = symbol.strip().upper()
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if hist.empty or len(hist) < 30:
                return f"Insufficient data for breakout analysis for {symbol}"
            
            highs = hist['High']
            lows = hist['Low']
            closes = hist['Close']
            volumes = hist['Volume']
            
            current_price = float(closes.iloc[-1])
            
            # Consolidation detection
            recent_range = highs.tail(20).max() - lows.tail(20).min()
            price_volatility = closes.tail(20).std() / closes.tail(20).mean()
            
            # Volume squeeze
            avg_volume = volumes.mean()
            recent_avg_volume = volumes.tail(10).mean()
            volume_contraction = recent_avg_volume < avg_volume * 0.8
            
            # Support/resistance levels
            resistance = highs.tail(50).max()
            support = lows.tail(50).min()
            
            # Distance to key levels
            resistance_distance = (resistance - current_price) / current_price
            support_distance = (current_price - support) / current_price
            
            return json.dumps({
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "breakout_setup": {
                    "consolidation_range": round(recent_range, 2),
                    "price_volatility": round(price_volatility * 100, 2),
                    "volume_contraction": volume_contraction,
                    "consolidation_duration_days": 20  # Simplified
                },
                "key_levels": {
                    "resistance": round(resistance, 2),
                    "support": round(support, 2),
                    "distance_to_resistance": round(resistance_distance * 100, 2),
                    "distance_to_support": round(support_distance * 100, 2)
                },
                "breakout_probability": {
                    "upward_breakout": round(
                        (1 - resistance_distance) * (1 if not volume_contraction else 1.2) * 
                        (0.8 if price_volatility > 0.02 else 1.2), 2
                    ),
                    "downward_breakout": round(
                        (1 - support_distance) * (1 if not volume_contraction else 1.2) * 
                        (0.8 if price_volatility > 0.02 else 1.2), 2
                    )
                },
                "trading_strategy": {
                    "breakout_above": round(resistance * 1.01, 2),  # 1% above resistance
                    "breakdown_below": round(support * 0.99, 2),   # 1% below support
                    "stop_loss_long": round(support * 0.98, 2),
                    "stop_loss_short": round(resistance * 1.02, 2)
                }
            })
            
        except Exception as e:
            logger.error(f"Error finding breakout patterns: {e}")
            return f"Error finding breakout patterns: {str(e)}"
    
    def _detect_triangle_patterns(self, highs: np.ndarray, lows: np.ndarray) -> List[dict]:
        """Detect triangle patterns (simplified implementation)"""
        patterns = []
        
        if len(highs) < 30:
            return patterns
        
        # Look for converging trend lines (very simplified)
        recent_highs = highs[-30:]
        recent_lows = lows[-30:]
        
        # Calculate trend of highs and lows
        x = np.arange(len(recent_highs))
        
        try:
            highs_slope, _, highs_r, _, _ = stats.linregress(x, recent_highs)
            lows_slope, _, lows_r, _, _ = stats.linregress(x, recent_lows)
            
            # Ascending triangle: flat resistance, rising support
            if abs(highs_slope) < 0.1 and lows_slope > 0.1 and highs_r > 0.5 and lows_r > 0.5:
                patterns.append({
                    "name": "ascending_triangle",
                    "confidence": min(highs_r, lows_r),
                    "bias": "bullish",
                    "formation_days": 30
                })
            
            # Descending triangle: declining resistance, flat support
            elif highs_slope < -0.1 and abs(lows_slope) < 0.1 and highs_r > 0.5 and lows_r > 0.5:
                patterns.append({
                    "name": "descending_triangle",
                    "confidence": min(highs_r, lows_r),
                    "bias": "bearish",
                    "formation_days": 30
                })
            
            # Symmetrical triangle: converging lines
            elif highs_slope < -0.05 and lows_slope > 0.05 and highs_r > 0.5 and lows_r > 0.5:
                patterns.append({
                    "name": "symmetrical_triangle",
                    "confidence": min(highs_r, lows_r),
                    "bias": "neutral",
                    "formation_days": 30
                })
        except:
            pass
        
        return patterns
    
    def _detect_head_shoulders(self, highs: np.ndarray, lows: np.ndarray) -> List[dict]:
        """Detect head and shoulders patterns (simplified)"""
        patterns = []
        
        if len(highs) < 50:
            return patterns
        
        # Very simplified H&S detection
        recent_highs = highs[-50:]
        
        # Find potential shoulders and head
        peak_indices = self._find_peaks(recent_highs, prominence=0.02)
        
        if len(peak_indices) >= 3:
            # Check for H&S structure (simplified)
            left_shoulder = recent_highs[peak_indices[0]]
            head = recent_highs[peak_indices[1]]
            right_shoulder = recent_highs[peak_indices[2]]
            
            # Basic H&S criteria
            if (head > left_shoulder and head > right_shoulder and 
                abs(left_shoulder - right_shoulder) / head < 0.05):
                patterns.append({
                    "name": "head_and_shoulders",
                    "confidence": 0.6,
                    "bias": "bearish",
                    "formation_days": 50
                })
        
        return patterns
    
    def _detect_double_patterns(self, highs: np.ndarray, lows: np.ndarray) -> List[dict]:
        """Detect double top/bottom patterns"""
        patterns = []
        
        # Double top detection
        if len(highs) >= 40:
            recent_highs = highs[-40:]
            peak_indices = self._find_peaks(recent_highs, prominence=0.02)
            
            if len(peak_indices) >= 2:
                peak1 = recent_highs[peak_indices[-2]]
                peak2 = recent_highs[peak_indices[-1]]
                
                if abs(peak1 - peak2) / max(peak1, peak2) < 0.03:  # Within 3%
                    patterns.append({
                        "name": "double_top",
                        "confidence": 0.7,
                        "bias": "bearish",
                        "formation_days": 40
                    })
        
        # Double bottom detection
        if len(lows) >= 40:
            recent_lows = lows[-40:]
            trough_indices = self._find_peaks(-recent_lows, prominence=0.02)
            
            if len(trough_indices) >= 2:
                trough1 = recent_lows[trough_indices[-2]]
                trough2 = recent_lows[trough_indices[-1]]
                
                if abs(trough1 - trough2) / min(trough1, trough2) < 0.03:
                    patterns.append({
                        "name": "double_bottom",
                        "confidence": 0.7,
                        "bias": "bullish",
                        "formation_days": 40
                    })
        
        return patterns
    
    def _find_peaks(self, data: np.ndarray, prominence: float) -> List[int]:
        """Find peaks in data (simplified implementation)"""
        peaks = []
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and data[i] > data[i+1] and
                data[i] > np.mean(data) * (1 + prominence)):
                peaks.append(i)
        return peaks
    
    def _find_local_extrema(self, series: pd.Series, extrema_type: str) -> List[float]:
        """Find local extrema in price series"""
        values = []
        data = series.values
        
        for i in range(2, len(data) - 2):
            if extrema_type == "max":
                if (data[i] > data[i-1] and data[i] > data[i-2] and
                    data[i] > data[i+1] and data[i] > data[i+2]):
                    values.append(data[i])
            else:  # min
                if (data[i] < data[i-1] and data[i] < data[i-2] and
                    data[i] < data[i+1] and data[i] < data[i+2]):
                    values.append(data[i])
        
        return sorted(set(values), reverse=(extrema_type == "max"))[:5]
    
    def _calculate_trend(self, series: pd.Series) -> dict:
        """Calculate trend characteristics"""
        if len(series) < 2:
            return {"direction": "unknown", "slope_percent": 0, "r_squared": 0}
        
        x = np.arange(len(series))
        y = series.values
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            slope_percent = (slope * len(series) / y[0]) * 100 if y[0] != 0 else 0
            
            return {
                "direction": "up" if slope > 0 else "down",
                "slope_percent": round(slope_percent, 2),
                "r_squared": round(r_value ** 2, 3),
                "strength": "strong" if abs(slope_percent) > 2 else "moderate" if abs(slope_percent) > 1 else "weak"
            }
        except:
            return {"direction": "unknown", "slope_percent": 0, "r_squared": 0}
    
    def _assess_trend_alignment(self, short: dict, medium: dict, long: dict) -> str:
        """Assess alignment of trends across timeframes"""
        directions = [short["direction"], medium["direction"], long["direction"]]
        
        if all(d == "up" for d in directions):
            return "strong_bullish"
        elif all(d == "down" for d in directions):
            return "strong_bearish"
        elif directions.count("up") == 2:
            return "bullish"
        elif directions.count("down") == 2:
            return "bearish"
        else:
            return "mixed"
    
    def _calculate_reversal_probability(self, short: dict, medium: dict, strength: float) -> float:
        """Calculate probability of trend reversal"""
        base_prob = 0.1  # Base 10% chance
        
        # High strength increases reversal probability
        if strength > 3:
            base_prob += 0.2
        
        # Trend inconsistency increases reversal probability
        if short["r_squared"] < 0.6:
            base_prob += 0.15
        
        # Opposing medium-term trend
        if short["direction"] != medium["direction"]:
            base_prob += 0.25
        
        return min(base_prob, 0.8)  # Cap at 80%
    
    def _get_pattern_implications(self, patterns: List[dict]) -> dict:
        """Get trading implications from found patterns"""
        if not patterns:
            return {"signal": "neutral", "confidence": 0}
        
        bullish_patterns = [p for p in patterns if p.get("bias") == "bullish"]
        bearish_patterns = [p for p in patterns if p.get("bias") == "bearish"]
        
        if len(bullish_patterns) > len(bearish_patterns):
            return {
                "signal": "bullish",
                "confidence": np.mean([p["confidence"] for p in bullish_patterns]),
                "dominant_patterns": [p["name"] for p in bullish_patterns]
            }
        elif len(bearish_patterns) > len(bullish_patterns):
            return {
                "signal": "bearish", 
                "confidence": np.mean([p["confidence"] for p in bearish_patterns]),
                "dominant_patterns": [p["name"] for p in bearish_patterns]
            }
        else:
            return {"signal": "mixed", "confidence": 0.5}


# Global pattern tool instance
_pattern_tool: Optional[PatternTool] = None


def get_pattern_tool() -> PatternTool:
    """Get the global pattern tool instance"""
    global _pattern_tool
    
    if _pattern_tool is None:
        _pattern_tool = PatternTool()
    
    return _pattern_tool