"""
Technical Analysis Indicators
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import talib

from app.core.exceptions import TradingError

logger = structlog.get_logger()


class TechnicalIndicators:
    """Technical analysis indicators for trading signals"""

    def __init__(self):
        self.indicators = {}
        logger.info("Technical indicators initialized")

    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate all technical indicators for given OHLCV data

        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Dict: All calculated indicators
        """
        try:
            if len(data) < 50:  # Need minimum data for reliable indicators
                raise TradingError(
                    f"Insufficient data: need at least 50 periods, got {len(data)}"
                )

            # Convert to numpy arrays for TA-Lib
            open_prices = data["open"].values
            high_prices = data["high"].values
            low_prices = data["low"].values
            close_prices = data["close"].values
            volume = data["volume"].values

            indicators = {}

            # Trend Indicators
            indicators.update(self._calculate_trend_indicators(close_prices))

            # Momentum Indicators
            indicators.update(
                self._calculate_momentum_indicators(
                    close_prices, high_prices, low_prices
                )
            )

            # Volatility Indicators
            indicators.update(
                self._calculate_volatility_indicators(
                    high_prices, low_prices, close_prices
                )
            )

            # Volume Indicators
            indicators.update(self._calculate_volume_indicators(close_prices, volume))

            # Support and Resistance
            indicators.update(
                self._calculate_support_resistance(
                    high_prices, low_prices, close_prices
                )
            )

            # Fibonacci Levels
            indicators.update(self._calculate_fibonacci_levels(high_prices, low_prices))

            # Elliott Wave Analysis
            indicators.update(self._calculate_elliott_wave_signals(close_prices))

            # Wyckoff Analysis
            indicators.update(self._calculate_wyckoff_signals(close_prices, volume))

            # Generate composite signals
            indicators["composite_signals"] = self._generate_composite_signals(
                indicators
            )

            logger.info("Technical indicators calculated successfully")
            return indicators

        except Exception as e:
            logger.error("Error calculating technical indicators", error=str(e))
            raise TradingError(f"Technical analysis failed: {str(e)}")

    def _calculate_trend_indicators(self, close_prices: np.ndarray) -> Dict:
        """Calculate trend-following indicators"""
        indicators = {}

        # Simple Moving Averages
        indicators["sma_20"] = talib.SMA(close_prices, timeperiod=20)
        indicators["sma_50"] = talib.SMA(close_prices, timeperiod=50)
        indicators["sma_200"] = talib.SMA(close_prices, timeperiod=200)

        # Exponential Moving Averages
        indicators["ema_12"] = talib.EMA(close_prices, timeperiod=12)
        indicators["ema_26"] = talib.EMA(close_prices, timeperiod=26)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        indicators["macd"] = macd
        indicators["macd_signal"] = macd_signal
        indicators["macd_histogram"] = macd_hist

        # Parabolic SAR
        indicators["sar"] = talib.SAR(
            close_prices, close_prices, acceleration=0.02, maximum=0.2
        )

        # ADX (Average Directional Index)
        indicators["adx"] = talib.ADX(
            close_prices, close_prices, close_prices, timeperiod=14
        )

        return indicators

    def _calculate_momentum_indicators(
        self, close_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray
    ) -> Dict:
        """Calculate momentum indicators"""
        indicators = {}

        # RSI
        indicators["rsi"] = talib.RSI(close_prices, timeperiod=14)

        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            high_prices,
            low_prices,
            close_prices,
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )
        indicators["stoch_k"] = slowk
        indicators["stoch_d"] = slowd

        # Williams %R
        indicators["williams_r"] = talib.WILLR(
            high_prices, low_prices, close_prices, timeperiod=14
        )

        # CCI (Commodity Channel Index)
        indicators["cci"] = talib.CCI(
            high_prices, low_prices, close_prices, timeperiod=14
        )

        # Rate of Change
        indicators["roc"] = talib.ROC(close_prices, timeperiod=10)

        # Momentum
        indicators["momentum"] = talib.MOM(close_prices, timeperiod=10)

        return indicators

    def _calculate_volatility_indicators(
        self, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray
    ) -> Dict:
        """Calculate volatility indicators"""
        indicators = {}

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close_prices, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        indicators["bb_upper"] = bb_upper
        indicators["bb_middle"] = bb_middle
        indicators["bb_lower"] = bb_lower
        indicators["bb_width"] = (bb_upper - bb_lower) / bb_middle
        indicators["bb_position"] = (close_prices - bb_lower) / (bb_upper - bb_lower)

        # Average True Range
        indicators["atr"] = talib.ATR(
            high_prices, low_prices, close_prices, timeperiod=14
        )

        # Keltner Channels
        ema_20 = talib.EMA(close_prices, timeperiod=20)
        kc_upper = ema_20 + (2 * indicators["atr"])
        kc_lower = ema_20 - (2 * indicators["atr"])
        indicators["kc_upper"] = kc_upper
        indicators["kc_middle"] = ema_20
        indicators["kc_lower"] = kc_lower

        return indicators

    def _calculate_volume_indicators(
        self, close_prices: np.ndarray, volume: np.ndarray
    ) -> Dict:
        """Calculate volume-based indicators"""
        indicators = {}

        # On-Balance Volume
        indicators["obv"] = talib.OBV(close_prices, volume)

        # Accumulation/Distribution Line
        indicators["ad"] = talib.AD(close_prices, close_prices, close_prices, volume)

        # Chaikin Money Flow
        indicators["cmf"] = talib.ADOSC(
            close_prices,
            close_prices,
            close_prices,
            volume,
            fastperiod=3,
            slowperiod=10,
        )

        # Volume Rate of Change
        indicators["volume_roc"] = talib.ROC(volume, timeperiod=10)

        return indicators

    def _calculate_support_resistance(
        self, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray
    ) -> Dict:
        """Calculate support and resistance levels"""
        indicators = {}

        # Pivot Points
        pivot = (high_prices + low_prices + close_prices) / 3
        indicators["pivot"] = pivot
        indicators["resistance_1"] = 2 * pivot - low_prices
        indicators["support_1"] = 2 * pivot - high_prices
        indicators["resistance_2"] = pivot + (high_prices - low_prices)
        indicators["support_2"] = pivot - (high_prices - low_prices)

        # Recent highs and lows
        indicators["recent_high"] = talib.MAX(high_prices, timeperiod=20)
        indicators["recent_low"] = talib.MIN(low_prices, timeperiod=20)

        return indicators

    def _calculate_fibonacci_levels(
        self, high_prices: np.ndarray, low_prices: np.ndarray
    ) -> Dict:
        """Calculate Fibonacci retracement levels"""
        indicators = {}

        # Find recent swing high and low
        recent_high = np.max(high_prices[-50:])  # Last 50 periods
        recent_low = np.min(low_prices[-50:])

        # Calculate Fibonacci levels
        fib_range = recent_high - recent_low
        indicators["fib_0"] = recent_high
        indicators["fib_23.6"] = recent_high - (0.236 * fib_range)
        indicators["fib_38.2"] = recent_high - (0.382 * fib_range)
        indicators["fib_50"] = recent_high - (0.5 * fib_range)
        indicators["fib_61.8"] = recent_high - (0.618 * fib_range)
        indicators["fib_100"] = recent_low

        return indicators

    def _calculate_elliott_wave_signals(self, close_prices: np.ndarray) -> Dict:
        """Calculate Elliott Wave analysis signals"""
        indicators = {}

        # Simplified Elliott Wave detection using price patterns
        # This is a basic implementation - real Elliott Wave analysis is much more complex

        # Calculate price changes
        price_changes = np.diff(close_prices)

        # Identify wave patterns (simplified)
        wave_count = 0
        current_trend = 1 if price_changes[0] > 0 else -1

        for change in price_changes:
            if (change > 0 and current_trend == -1) or (
                change < 0 and current_trend == 1
            ):
                wave_count += 1
                current_trend *= -1

        indicators["elliott_wave_count"] = wave_count
        indicators["elliott_wave_phase"] = (
            "Impulse" if wave_count % 2 == 1 else "Corrective"
        )

        # Basic wave strength
        recent_volatility = np.std(price_changes[-20:])
        indicators["elliott_wave_strength"] = min(recent_volatility * 100, 1.0)

        return indicators

    def _calculate_wyckoff_signals(
        self, close_prices: np.ndarray, volume: np.ndarray
    ) -> Dict:
        """Calculate Wyckoff method signals"""
        indicators = {}

        # Wyckoff phases (simplified)
        # Accumulation, Markup, Distribution, Markdown

        # Calculate volume-price relationship
        price_change = np.diff(close_prices)
        volume_change = np.diff(volume)

        # Volume-Price Trend
        vpt = np.cumsum(volume_change * price_change)
        indicators["wyckoff_vpt"] = vpt

        # Wyckoff phase detection (simplified)
        recent_price_trend = np.mean(price_change[-10:])
        recent_volume_trend = np.mean(volume_change[-10:])

        if recent_price_trend > 0 and recent_volume_trend > 0:
            indicators["wyckoff_phase"] = "Markup"
        elif recent_price_trend < 0 and recent_volume_trend > 0:
            indicators["wyckoff_phase"] = "Distribution"
        elif recent_price_trend < 0 and recent_volume_trend < 0:
            indicators["wyckoff_phase"] = "Markdown"
        else:
            indicators["wyckoff_phase"] = "Accumulation"

        # Wyckoff strength
        indicators["wyckoff_strength"] = abs(recent_price_trend) * abs(
            recent_volume_trend
        )

        return indicators

    def _generate_composite_signals(self, indicators: Dict) -> Dict:
        """Generate composite trading signals from all indicators"""
        signals = {}

        try:
            # Trend signals
            current_price = indicators.get("sma_20", [0])[-1]
            sma_50 = indicators.get("sma_50", [0])[-1]
            sma_200 = indicators.get("sma_200", [0])[-1]

            if current_price > sma_50 > sma_200:
                signals["trend"] = "BULLISH"
            elif current_price < sma_50 < sma_200:
                signals["trend"] = "BEARISH"
            else:
                signals["trend"] = "NEUTRAL"

            # Momentum signals
            rsi = indicators.get("rsi", [50])[-1]
            if rsi > 70:
                signals["momentum"] = "OVERBOUGHT"
            elif rsi < 30:
                signals["momentum"] = "OVERSOLD"
            else:
                signals["momentum"] = "NEUTRAL"

            # Volatility signals
            bb_position = indicators.get("bb_position", [0.5])[-1]
            if bb_position > 0.8:
                signals["volatility"] = "HIGH"
            elif bb_position < 0.2:
                signals["volatility"] = "LOW"
            else:
                signals["volatility"] = "NORMAL"

            # Volume signals
            volume_roc = indicators.get("volume_roc", [0])[-1]
            if volume_roc > 20:
                signals["volume"] = "HIGH"
            elif volume_roc < -20:
                signals["volume"] = "LOW"
            else:
                signals["volume"] = "NORMAL"

            # Composite signal
            signal_score = 0
            if signals["trend"] == "BULLISH":
                signal_score += 1
            elif signals["trend"] == "BEARISH":
                signal_score -= 1

            if signals["momentum"] == "OVERSOLD":
                signal_score += 1
            elif signals["momentum"] == "OVERBOUGHT":
                signal_score -= 1

            if signals["volume"] == "HIGH":
                signal_score += 0.5

            if signal_score > 1:
                signals["composite"] = "BUY"
            elif signal_score < -1:
                signals["composite"] = "SELL"
            else:
                signals["composite"] = "HOLD"

            signals["signal_strength"] = abs(signal_score) / 2.5  # Normalize to 0-1

        except Exception as e:
            logger.error("Error generating composite signals", error=str(e))
            signals = {
                "trend": "NEUTRAL",
                "momentum": "NEUTRAL",
                "volatility": "NORMAL",
                "volume": "NORMAL",
                "composite": "HOLD",
                "signal_strength": 0.0,
            }

        return signals
