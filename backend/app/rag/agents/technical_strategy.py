"""
Technical Strategy Plugin
Implements technical indicator-based trading strategies with StrategyPlugin interface
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from langchain.agents import Tool

from app.monitoring.metrics import PrometheusMetrics

logger = logging.getLogger(__name__)


class TechnicalStrategy:
    """Technical indicator-based trading strategy plugin"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = "technical"

        # Technical indicator configurations
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)

        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)

        self.bollinger_period = config.get("bollinger_period", 20)
        self.bollinger_std = config.get("bollinger_std", 2.0)

        self.sma_short = config.get("sma_short", 10)
        self.sma_long = config.get("sma_long", 50)

        self.confidence_threshold = config.get("confidence_threshold", 0.6)

    def get_tools(self) -> List[Tool]:
        """Return LangChain tools for technical strategy"""
        return [
            Tool(
                name="calculate_rsi",
                func=self._calculate_rsi_tool,
                description="Calculate RSI (Relative Strength Index) for momentum analysis",
            ),
            Tool(
                name="calculate_macd",
                func=self._calculate_macd_tool,
                description="Calculate MACD (Moving Average Convergence Divergence) for trend analysis",
            ),
            Tool(
                name="calculate_bollinger_bands",
                func=self._calculate_bollinger_tool,
                description="Calculate Bollinger Bands for volatility and support/resistance analysis",
            ),
            Tool(
                name="analyze_moving_averages",
                func=self._analyze_moving_averages_tool,
                description="Analyze moving average crossovers and trends",
            ),
            Tool(
                name="get_technical_summary",
                func=self._technical_summary_tool,
                description="Get comprehensive technical analysis summary with all indicators",
            ),
        ]

    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical indicator analysis"""
        start_time = datetime.now()
        metrics = PrometheusMetrics()
        symbol = market_data.get("symbol", "UNKNOWN")

        try:
            prices = market_data.get("prices", [])
            volumes = market_data.get("volumes", [])
            highs = market_data.get("highs", prices)  # Default to close prices
            market_data.get("lows", prices)

            if len(prices) < max(
                self.rsi_period, self.macd_slow, self.bollinger_period
            ):
                return {
                    "error": "insufficient_data",
                    "message": f"Need at least {max(self.rsi_period, self.macd_slow, self.bollinger_period)} periods",
                }

            # Calculate all technical indicators
            rsi_analysis = self._calculate_rsi(prices)
            macd_analysis = self._calculate_macd(prices)
            bollinger_analysis = self._calculate_bollinger_bands(prices)
            ma_analysis = self._analyze_moving_averages(prices)
            volume_analysis = (
                self._analyze_volume_indicators(volumes) if volumes else {}
            )

            # Generate composite signal
            signals = {
                "rsi": self._rsi_signal(rsi_analysis),
                "macd": self._macd_signal(macd_analysis),
                "bollinger": self._bollinger_signal(bollinger_analysis),
                "moving_average": self._ma_signal(ma_analysis),
            }

            if volume_analysis:
                signals["volume"] = self._volume_signal(volume_analysis)

            # Calculate composite confidence and direction
            composite_signal, confidence = self._calculate_composite_signal(signals)

            analysis_time = (datetime.now() - start_time).total_seconds()
            metrics.record_strategy_analysis_latency(analysis_time, "technical", symbol)

            return {
                "rsi": rsi_analysis,
                "macd": macd_analysis,
                "bollinger_bands": bollinger_analysis,
                "moving_averages": ma_analysis,
                "volume_analysis": volume_analysis,
                "individual_signals": signals,
                "composite_signal": composite_signal,
                "confidence": confidence,
                "indicator_count": len(signals),
            }

        except Exception as e:
            analysis_time = (datetime.now() - start_time).total_seconds()
            metrics.record_strategy_analysis_latency(analysis_time, "technical", symbol)
            logger.error(f"Technical analysis failed for {symbol}: {str(e)}")
            return {"error": "analysis_failed", "message": str(e)}

    def generate_signal(
        self, analysis: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate technical indicator-based trading signal"""
        if "error" in analysis:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": analysis.get("message", "Technical analysis failed"),
            }

        composite_signal = analysis.get("composite_signal", "neutral")
        confidence = analysis.get("confidence", 0.0)
        individual_signals = analysis.get("individual_signals", {})

        if confidence < self.confidence_threshold:
            return {
                "action": "HOLD",
                "confidence": confidence,
                "reasoning": f"Technical confluence too low: {confidence:.1%} (threshold: {self.confidence_threshold:.1%})",
            }

        # Convert composite signal to trading action
        action_mapping = {"bullish": "BUY", "bearish": "SELL", "neutral": "HOLD"}

        action = action_mapping.get(composite_signal, "HOLD")

        # Generate detailed reasoning
        supporting_indicators = []
        for indicator, signal in individual_signals.items():
            if signal.get("direction") == composite_signal:
                supporting_indicators.append(
                    f"{indicator.upper()}({signal.get('strength', 0.0):.2f})"
                )

        reasoning = (
            f"Technical confluence: {len(supporting_indicators)} indicators aligned"
        )
        if supporting_indicators:
            reasoning += f" - {', '.join(supporting_indicators[:3])}"

        # Add specific indicator insights
        rsi = analysis.get("rsi", {})
        if rsi.get("current_rsi"):
            reasoning += f", RSI: {rsi['current_rsi']:.1f}"

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "supporting_indicators": supporting_indicators,
        }

    def _calculate_rsi(self, prices: List[float]) -> Dict[str, Any]:
        """Calculate Relative Strength Index"""
        if len(prices) < self.rsi_period + 1:
            return {"error": "insufficient_data"}

        prices_array = np.array(prices)
        deltas = np.diff(prices_array)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Initial average gains and losses (SMA)
        avg_gain = np.mean(gains[: self.rsi_period])
        avg_loss = np.mean(losses[: self.rsi_period])

        # Exponential smoothing for subsequent values
        for i in range(self.rsi_period, len(gains)):
            avg_gain = ((avg_gain * (self.rsi_period - 1)) + gains[i]) / self.rsi_period
            avg_loss = (
                (avg_loss * (self.rsi_period - 1)) + losses[i]
            ) / self.rsi_period

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        return {
            "current_rsi": rsi,
            "oversold_threshold": self.rsi_oversold,
            "overbought_threshold": self.rsi_overbought,
            "is_oversold": rsi < self.rsi_oversold,
            "is_overbought": rsi > self.rsi_overbought,
        }

    def _calculate_macd(self, prices: List[float]) -> Dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < self.macd_slow + self.macd_signal:
            return {"error": "insufficient_data"}

        prices_array = np.array(prices)

        # Calculate EMAs
        ema_fast = self._calculate_ema(prices_array, self.macd_fast)
        ema_slow = self._calculate_ema(prices_array, self.macd_slow)

        macd_line = ema_fast[-1] - ema_slow[-1]

        # Calculate signal line (EMA of MACD)
        if len(prices) >= self.macd_slow + self.macd_signal:
            macd_history = ema_fast[-self.macd_signal :] - ema_slow[-self.macd_signal :]
            signal_line = self._calculate_ema(macd_history, self.macd_signal)[-1]
            histogram = macd_line - signal_line
        else:
            signal_line = macd_line
            histogram = 0

        return {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
            "is_bullish_crossover": macd_line > signal_line and histogram > 0,
            "is_bearish_crossover": macd_line < signal_line and histogram < 0,
        }

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _calculate_bollinger_bands(self, prices: List[float]) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        if len(prices) < self.bollinger_period:
            return {"error": "insufficient_data"}

        prices_array = np.array(prices)
        sma = np.mean(prices_array[-self.bollinger_period :])
        std = np.std(prices_array[-self.bollinger_period :])

        upper_band = sma + (self.bollinger_std * std)
        lower_band = sma - (self.bollinger_std * std)
        current_price = prices[-1]

        # Band position (0 = lower band, 0.5 = middle, 1 = upper band)
        band_position = (current_price - lower_band) / (upper_band - lower_band)

        return {
            "upper_band": upper_band,
            "middle_band": sma,
            "lower_band": lower_band,
            "current_price": current_price,
            "band_position": band_position,
            "is_near_upper": band_position > 0.8,
            "is_near_lower": band_position < 0.2,
            "band_width": (upper_band - lower_band) / sma,
        }

    def _analyze_moving_averages(self, prices: List[float]) -> Dict[str, Any]:
        """Analyze moving average trends and crossovers"""
        if len(prices) < max(self.sma_short, self.sma_long):
            return {"error": "insufficient_data"}

        prices_array = np.array(prices)

        sma_short_current = np.mean(prices_array[-self.sma_short :])
        sma_long_current = np.mean(prices_array[-self.sma_long :])

        # Previous values for crossover detection
        if len(prices) > max(self.sma_short, self.sma_long):
            sma_short_prev = np.mean(prices_array[-self.sma_short - 1 : -1])
            sma_long_prev = np.mean(prices_array[-self.sma_long - 1 : -1])

            golden_cross = (
                sma_short_current > sma_long_current and sma_short_prev <= sma_long_prev
            )
            death_cross = (
                sma_short_current < sma_long_current and sma_short_prev >= sma_long_prev
            )
        else:
            golden_cross = False
            death_cross = False

        return {
            "sma_short": sma_short_current,
            "sma_long": sma_long_current,
            "is_uptrend": sma_short_current > sma_long_current,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "trend_strength": abs(sma_short_current - sma_long_current)
            / sma_long_current,
        }

    def _analyze_volume_indicators(self, volumes: List[float]) -> Dict[str, Any]:
        """Analyze volume-based indicators"""
        if len(volumes) < 20:
            return {"error": "insufficient_data"}

        volumes_array = np.array(volumes)
        avg_volume = np.mean(volumes_array[-20:])
        current_volume = volumes[-1]

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        return {
            "current_volume": current_volume,
            "average_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "high_volume": volume_ratio > 1.5,
            "low_volume": volume_ratio < 0.5,
        }

    def _rsi_signal(self, rsi_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate RSI-based signal"""
        if "error" in rsi_analysis:
            return {"direction": "neutral", "strength": 0.0}

        rsi = rsi_analysis["current_rsi"]

        if rsi_analysis["is_oversold"]:
            return {
                "direction": "bullish",
                "strength": (self.rsi_oversold - rsi) / self.rsi_oversold,
            }
        elif rsi_analysis["is_overbought"]:
            return {
                "direction": "bearish",
                "strength": (rsi - self.rsi_overbought) / (100 - self.rsi_overbought),
            }
        else:
            return {"direction": "neutral", "strength": 0.3}

    def _macd_signal(self, macd_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate MACD-based signal"""
        if "error" in macd_analysis:
            return {"direction": "neutral", "strength": 0.0}

        if macd_analysis["is_bullish_crossover"]:
            return {"direction": "bullish", "strength": 0.8}
        elif macd_analysis["is_bearish_crossover"]:
            return {"direction": "bearish", "strength": 0.8}
        elif macd_analysis["histogram"] > 0:
            return {"direction": "bullish", "strength": 0.5}
        elif macd_analysis["histogram"] < 0:
            return {"direction": "bearish", "strength": 0.5}
        else:
            return {"direction": "neutral", "strength": 0.0}

    def _bollinger_signal(self, bollinger_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Bollinger Bands signal"""
        if "error" in bollinger_analysis:
            return {"direction": "neutral", "strength": 0.0}

        if bollinger_analysis["is_near_lower"]:
            return {"direction": "bullish", "strength": 0.6}
        elif bollinger_analysis["is_near_upper"]:
            return {"direction": "bearish", "strength": 0.6}
        else:
            return {"direction": "neutral", "strength": 0.2}

    def _ma_signal(self, ma_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate moving average signal"""
        if "error" in ma_analysis:
            return {"direction": "neutral", "strength": 0.0}

        if ma_analysis["golden_cross"]:
            return {"direction": "bullish", "strength": 0.9}
        elif ma_analysis["death_cross"]:
            return {"direction": "bearish", "strength": 0.9}
        elif ma_analysis["is_uptrend"]:
            return {"direction": "bullish", "strength": 0.6}
        else:
            return {"direction": "bearish", "strength": 0.6}

    def _volume_signal(self, volume_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate volume-based signal"""
        if "error" in volume_analysis:
            return {"direction": "neutral", "strength": 0.0}

        if volume_analysis["high_volume"]:
            return {
                "direction": "neutral",
                "strength": 0.3,
            }  # Volume confirms other signals
        else:
            return {"direction": "neutral", "strength": 0.1}

    def _calculate_composite_signal(
        self, signals: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, float]:
        """Calculate composite signal from individual indicators"""
        bullish_strength = 0.0
        bearish_strength = 0.0
        total_weight = 0.0

        # Weight individual indicators
        weights = {
            "rsi": 0.25,
            "macd": 0.30,
            "bollinger": 0.20,
            "moving_average": 0.20,
            "volume": 0.05,
        }

        for indicator, signal in signals.items():
            weight = weights.get(indicator, 0.1)
            strength = signal.get("strength", 0.0)
            direction = signal.get("direction", "neutral")

            if direction == "bullish":
                bullish_strength += weight * strength
            elif direction == "bearish":
                bearish_strength += weight * strength

            total_weight += weight

        # Normalize strengths
        if total_weight > 0:
            bullish_strength /= total_weight
            bearish_strength /= total_weight

        # Determine dominant direction and confidence
        if bullish_strength > bearish_strength:
            return "bullish", bullish_strength
        elif bearish_strength > bullish_strength:
            return "bearish", bearish_strength
        else:
            return "neutral", max(bullish_strength, bearish_strength) * 0.5

    # Tool implementations
    def _calculate_rsi_tool(self, price_data: str) -> str:
        """Tool implementation for RSI calculation"""
        try:
            prices = [float(x.strip()) for x in price_data.split(",")]
            rsi_analysis = self._calculate_rsi(prices)

            if "error" in rsi_analysis:
                return "RSI calculation failed: insufficient data"

            rsi = rsi_analysis["current_rsi"]
            status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"

            return f"RSI: {rsi:.2f} ({status})"

        except Exception as e:
            return f"RSI calculation error: {str(e)}"

    def _calculate_macd_tool(self, price_data: str) -> str:
        """Tool implementation for MACD calculation"""
        try:
            prices = [float(x.strip()) for x in price_data.split(",")]
            macd_analysis = self._calculate_macd(prices)

            if "error" in macd_analysis:
                return "MACD calculation failed: insufficient data"

            macd = macd_analysis["macd_line"]
            signal = macd_analysis["signal_line"]
            histogram = macd_analysis["histogram"]

            return f"MACD: {macd:.4f}, Signal: {signal:.4f}, Histogram: {histogram:.4f}"

        except Exception as e:
            return f"MACD calculation error: {str(e)}"

    def _calculate_bollinger_tool(self, price_data: str) -> str:
        """Tool implementation for Bollinger Bands calculation"""
        try:
            prices = [float(x.strip()) for x in price_data.split(",")]
            bb_analysis = self._calculate_bollinger_bands(prices)

            if "error" in bb_analysis:
                return "Bollinger Bands calculation failed: insufficient data"

            position = bb_analysis["band_position"]
            status = (
                "Near Upper Band"
                if position > 0.8
                else "Near Lower Band" if position < 0.2 else "Mid-Range"
            )

            return f"Bollinger Position: {position:.2f} ({status})"

        except Exception as e:
            return f"Bollinger Bands calculation error: {str(e)}"

    def _analyze_moving_averages_tool(self, price_data: str) -> str:
        """Tool implementation for moving averages analysis"""
        try:
            prices = [float(x.strip()) for x in price_data.split(",")]
            ma_analysis = self._analyze_moving_averages(prices)

            if "error" in ma_analysis:
                return "Moving averages analysis failed: insufficient data"

            trend = "Uptrend" if ma_analysis["is_uptrend"] else "Downtrend"
            crossover = ""
            if ma_analysis["golden_cross"]:
                crossover = " (Golden Cross!)"
            elif ma_analysis["death_cross"]:
                crossover = " (Death Cross!)"

            return f"MA Trend: {trend}{crossover}, SMA{self.sma_short}: {ma_analysis['sma_short']:.2f}, SMA{self.sma_long}: {ma_analysis['sma_long']:.2f}"

        except Exception as e:
            return f"Moving averages analysis error: {str(e)}"

    def _technical_summary_tool(self, price_data: str) -> str:
        """Tool implementation for comprehensive technical summary"""
        try:
            prices = [float(x.strip()) for x in price_data.split(",")]
            analysis = self.analyze_market({"prices": prices})

            if "error" in analysis:
                return f"Technical analysis failed: {analysis.get('message', 'Unknown error')}"

            composite_signal = analysis.get("composite_signal", "neutral")
            confidence = analysis.get("confidence", 0.0)

            return f"Technical Summary: {composite_signal.upper()} signal with {confidence:.1%} confidence"

        except Exception as e:
            return f"Technical summary error: {str(e)}"

    def extract_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical indicator features for pattern matching"""
        analysis = self.analyze_market(market_data)

        return {
            "strategy": self.name,
            "timestamp": datetime.now().isoformat(),
            "composite_signal": analysis.get("composite_signal", "neutral"),
            "confidence": analysis.get("confidence", 0.0),
            "rsi": analysis.get("rsi", {}).get("current_rsi", 50),
            "macd_histogram": analysis.get("macd", {}).get("histogram", 0),
            "bollinger_position": analysis.get("bollinger_bands", {}).get(
                "band_position", 0.5
            ),
            "trend_direction": (
                "up"
                if analysis.get("moving_averages", {}).get("is_uptrend")
                else "down"
            ),
        }
