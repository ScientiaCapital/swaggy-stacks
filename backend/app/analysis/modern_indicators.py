"""
Modern Technical Analysis Indicators
Implementation of advanced indicators from Sofien Kaabar's "New Technical Indicators in Python"
"""

from typing import Dict

import numpy as np
import pandas as pd
import structlog

from app.core.exceptions import TradingError

logger = structlog.get_logger()


class ModernIndicators:
    """Modern technical analysis indicators for advanced trading signals"""

    def __init__(self):
        self.indicators = {}
        logger.info("Modern indicators initialized")

    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate all modern technical indicators for given OHLCV data

        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Dict: All calculated modern indicators
        """
        try:
            if len(data) < 50:  # Need minimum data for reliable indicators
                raise TradingError(
                    f"Insufficient data: need at least 50 periods, got {len(data)}"
                )

            # Convert to numpy arrays
            open_prices = data["open"].values
            high_prices = data["high"].values
            low_prices = data["low"].values
            close_prices = data["close"].values
            volume = data["volume"].values

            indicators = {}

            # Adaptive Indicators
            indicators.update(
                self._calculate_adaptive_indicators(
                    open_prices, high_prices, low_prices, close_prices, volume
                )
            )

            # Contrarian Indicators
            indicators.update(
                self._calculate_contrarian_indicators(
                    high_prices, low_prices, close_prices
                )
            )

            # Pattern Recognition Indicators
            indicators.update(
                self._calculate_pattern_indicators(
                    high_prices, low_prices, close_prices
                )
            )

            # Advanced Momentum Indicators
            indicators.update(
                self._calculate_advanced_momentum_indicators(close_prices, volume)
            )

            # Market Timing Indicators
            indicators.update(
                self._calculate_market_timing_indicators(
                    open_prices, high_prices, low_prices, close_prices, volume
                )
            )

            # Generate composite modern signals
            indicators["modern_composite_signals"] = (
                self._generate_modern_composite_signals(indicators)
            )

            logger.info("Modern indicators calculated successfully")
            return indicators

        except Exception as e:
            logger.error("Error calculating modern indicators", error=str(e))
            raise TradingError(f"Modern indicator analysis failed: {str(e)}")

    def _calculate_adaptive_indicators(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
    ) -> Dict:
        """Calculate adaptive indicators that adjust to market conditions"""
        indicators = {}

        # K's Envelopes - Dynamic support/resistance with volatility adjustment
        indicators.update(self._calculate_k_envelopes(close_prices))

        # KAMA - Kaufman Adaptive Moving Average
        indicators.update(self._calculate_kama(close_prices))

        # Zero-Lag EMA - Reduced lag exponential moving average
        indicators.update(self._calculate_zlema(close_prices))

        return indicators

    def _calculate_contrarian_indicators(
        self, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray
    ) -> Dict:
        """Calculate contrarian indicators for reversal signals"""
        indicators = {}

        # Polarized Fractal Efficiency - Trend efficiency measurement
        indicators.update(self._calculate_pfe(close_prices))

        return indicators

    def _calculate_pattern_indicators(
        self, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray
    ) -> Dict:
        """Calculate pattern recognition indicators"""
        indicators = {}

        # Fractal Patterns - 5-bar reversal pattern detection
        indicators.update(self._calculate_fractal_patterns(high_prices, low_prices))

        return indicators

    def _calculate_advanced_momentum_indicators(
        self, close_prices: np.ndarray, volume: np.ndarray
    ) -> Dict:
        """Calculate advanced momentum indicators"""
        indicators = {}

        # Relative Vigor Index (RVI)
        indicators.update(self._calculate_rvi(close_prices))

        # Coppock Curve
        indicators.update(self._calculate_coppock_curve(close_prices))

        # Know Sure Thing (KST) Oscillator
        indicators.update(self._calculate_kst_oscillator(close_prices))

        return indicators

    def _calculate_k_envelopes(
        self, close_prices: np.ndarray, period: int = 20, multiplier: float = 1.5
    ) -> Dict:
        """
        Calculate K's Envelopes - Dynamic support/resistance bands

        K's Envelopes adapt to volatility using rolling standard deviation
        """
        indicators = {}

        # Calculate Simple Moving Average
        sma = pd.Series(close_prices).rolling(window=period).mean().values

        # Calculate rolling standard deviation for volatility adjustment
        rolling_std = pd.Series(close_prices).rolling(window=period).std().values

        # K's Envelope bands
        k_upper = sma + (multiplier * rolling_std)
        k_lower = sma - (multiplier * rolling_std)
        k_middle = sma

        # K's Envelope position (similar to Bollinger Band position)
        k_position = np.divide(
            close_prices - k_lower,
            k_upper - k_lower,
            out=np.zeros_like(close_prices),
            where=(k_upper - k_lower) != 0,
        )

        # K's Envelope width (volatility measure)
        k_width = np.divide(
            k_upper - k_lower,
            k_middle,
            out=np.zeros_like(k_middle),
            where=k_middle != 0,
        )

        indicators["k_envelope_upper"] = k_upper
        indicators["k_envelope_middle"] = k_middle
        indicators["k_envelope_lower"] = k_lower
        indicators["k_envelope_position"] = k_position
        indicators["k_envelope_width"] = k_width

        return indicators

    def _calculate_kama(
        self,
        close_prices: np.ndarray,
        period: int = 14,
        fast_sc: float = 2,
        slow_sc: float = 30,
    ) -> Dict:
        """
        Calculate KAMA - Kaufman Adaptive Moving Average

        KAMA adjusts its smoothing constant based on market efficiency
        """
        indicators = {}

        # Calculate Efficiency Ratio (ER)
        price_changes = np.abs(np.diff(close_prices))
        direction = np.abs(close_prices[period:] - close_prices[:-period])
        volatility = (
            pd.Series(price_changes).rolling(window=period).sum().values[period - 1 :]
        )

        # Avoid division by zero
        efficiency_ratio = np.divide(
            direction, volatility, out=np.zeros_like(direction), where=volatility != 0
        )

        # Smoothing Constant (SC)
        fast_sc_adj = 2.0 / (fast_sc + 1)
        slow_sc_adj = 2.0 / (slow_sc + 1)

        smoothing_constant = np.square(
            efficiency_ratio * (fast_sc_adj - slow_sc_adj) + slow_sc_adj
        )

        # Calculate KAMA
        kama = np.zeros_like(close_prices)
        kama[:period] = close_prices[:period]  # Initialize with actual prices

        for i in range(period, len(close_prices)):
            sc_idx = i - period
            if sc_idx < len(smoothing_constant):
                kama[i] = kama[i - 1] + smoothing_constant[sc_idx] * (
                    close_prices[i] - kama[i - 1]
                )
            else:
                kama[i] = kama[i - 1]

        indicators["kama"] = kama
        indicators["kama_efficiency_ratio"] = np.concatenate(
            [np.full(period, np.nan), efficiency_ratio]
        )

        return indicators

    def _calculate_zlema(self, close_prices: np.ndarray, period: int = 21) -> Dict:
        """
        Calculate Zero-Lag EMA - Reduces lag of traditional EMA

        ZLEMA applies error correction to reduce the inherent lag of EMA
        """
        indicators = {}

        # Calculate lag (half of period)
        lag = int(period / 2)

        # Error correction: price + (price - price[lag periods ago])
        corrected_prices = np.zeros_like(close_prices)
        corrected_prices[:lag] = close_prices[:lag]
        corrected_prices[lag:] = close_prices[lag:] + (
            close_prices[lag:] - close_prices[:-lag]
        )

        # Calculate EMA on corrected prices
        alpha = 2.0 / (period + 1)
        zlema = np.zeros_like(close_prices)
        zlema[0] = corrected_prices[0]

        for i in range(1, len(corrected_prices)):
            zlema[i] = alpha * corrected_prices[i] + (1 - alpha) * zlema[i - 1]

        indicators["zlema"] = zlema

        return indicators

    def _calculate_pfe(self, close_prices: np.ndarray, period: int = 14) -> Dict:
        """
        Calculate Polarized Fractal Efficiency (PFE)

        PFE measures the efficiency of price movement (-100 to +100)
        Positive values indicate uptrend efficiency, negative values downtrend efficiency
        """
        indicators = {}

        pfe = np.zeros_like(close_prices)

        for i in range(period, len(close_prices)):
            # Calculate linear distance (direct path from start to end)
            linear_distance = abs(close_prices[i] - close_prices[i - period])

            # Calculate actual path distance (sum of all price movements)
            actual_distances = []
            for j in range(i - period + 1, i + 1):
                if j > 0:
                    actual_distances.append(abs(close_prices[j] - close_prices[j - 1]))

            actual_distance = sum(actual_distances) if actual_distances else 1

            # Calculate efficiency
            if actual_distance != 0:
                efficiency = linear_distance / actual_distance
            else:
                efficiency = 0

            # Polarize based on price direction
            if close_prices[i] > close_prices[i - period]:
                pfe[i] = efficiency * 100
            else:
                pfe[i] = -efficiency * 100

        indicators["pfe"] = pfe

        return indicators

    def _calculate_fractal_patterns(
        self, high_prices: np.ndarray, low_prices: np.ndarray
    ) -> Dict:
        """
        Calculate Fractal Patterns - 5-bar reversal pattern detection

        Identifies potential reversal points using 5-bar fractal patterns
        """
        indicators = {}

        # Bullish fractals (low points)
        bullish_fractals = np.zeros_like(low_prices)
        # Bearish fractals (high points)
        bearish_fractals = np.zeros_like(high_prices)

        # Need at least 5 bars for fractal pattern
        for i in range(2, len(low_prices) - 2):
            # Bullish fractal: current low is lower than 2 bars before and after
            if (
                low_prices[i] < low_prices[i - 2]
                and low_prices[i] < low_prices[i - 1]
                and low_prices[i] < low_prices[i + 1]
                and low_prices[i] < low_prices[i + 2]
            ):
                bullish_fractals[i] = low_prices[i]

            # Bearish fractal: current high is higher than 2 bars before and after
            if (
                high_prices[i] > high_prices[i - 2]
                and high_prices[i] > high_prices[i - 1]
                and high_prices[i] > high_prices[i + 1]
                and high_prices[i] > high_prices[i + 2]
            ):
                bearish_fractals[i] = high_prices[i]

        indicators["fractal_bullish"] = bullish_fractals
        indicators["fractal_bearish"] = bearish_fractals

        # Fractal signal strength based on recent fractal density
        recent_bullish = np.sum(bullish_fractals[-20:] > 0)
        recent_bearish = np.sum(bearish_fractals[-20:] > 0)

        if recent_bullish > recent_bearish:
            fractal_bias = "BULLISH"
            fractal_strength = (recent_bullish - recent_bearish) / 20.0
        elif recent_bearish > recent_bullish:
            fractal_bias = "BEARISH"
            fractal_strength = (recent_bearish - recent_bullish) / 20.0
        else:
            fractal_bias = "NEUTRAL"
            fractal_strength = 0.0

        indicators["fractal_bias"] = fractal_bias
        indicators["fractal_strength"] = min(abs(fractal_strength), 1.0)

        return indicators

    def _generate_modern_composite_signals(self, indicators: Dict) -> Dict:
        """Generate composite trading signals from modern indicators"""
        signals = {}

        try:
            # K's Envelope signals
            k_position = indicators.get("k_envelope_position", [0.5])[-1]
            if not np.isnan(k_position):
                if k_position > 0.8:
                    signals["k_envelope"] = "OVERBOUGHT"
                elif k_position < 0.2:
                    signals["k_envelope"] = "OVERSOLD"
                else:
                    signals["k_envelope"] = "NEUTRAL"
            else:
                signals["k_envelope"] = "NEUTRAL"

            # KAMA trend signal
            kama_values = indicators.get("kama", [])
            if len(kama_values) >= 3:
                kama_current = kama_values[-1]
                kama_prev = kama_values[-2]
                kama_prev2 = kama_values[-3]

                if kama_current > kama_prev > kama_prev2:
                    signals["kama_trend"] = "BULLISH"
                elif kama_current < kama_prev < kama_prev2:
                    signals["kama_trend"] = "BEARISH"
                else:
                    signals["kama_trend"] = "NEUTRAL"
            else:
                signals["kama_trend"] = "NEUTRAL"

            # PFE momentum signal
            pfe = indicators.get("pfe", [0])[-1]
            if not np.isnan(pfe):
                if pfe > 50:
                    signals["pfe_momentum"] = "STRONG_BULLISH"
                elif pfe > 20:
                    signals["pfe_momentum"] = "BULLISH"
                elif pfe < -50:
                    signals["pfe_momentum"] = "STRONG_BEARISH"
                elif pfe < -20:
                    signals["pfe_momentum"] = "BEARISH"
                else:
                    signals["pfe_momentum"] = "NEUTRAL"
            else:
                signals["pfe_momentum"] = "NEUTRAL"

            # Fractal pattern signal
            signals["fractal_bias"] = indicators.get("fractal_bias", "NEUTRAL")

            # Advanced momentum signals
            rvi = (
                indicators.get("rvi", [0])[-1]
                if len(indicators.get("rvi", [])) > 0
                else 0
            )
            rvi_signal = (
                indicators.get("rvi_signal", [0])[-1]
                if len(indicators.get("rvi_signal", [])) > 0
                else 0
            )

            if not np.isnan(rvi) and not np.isnan(rvi_signal):
                if rvi > rvi_signal and rvi > 0:
                    signals["rvi_momentum"] = "BULLISH"
                elif rvi < rvi_signal and rvi < 0:
                    signals["rvi_momentum"] = "BEARISH"
                else:
                    signals["rvi_momentum"] = "NEUTRAL"
            else:
                signals["rvi_momentum"] = "NEUTRAL"

            # Coppock Curve signal
            coppock = (
                indicators.get("coppock_curve", [0])[-1]
                if len(indicators.get("coppock_curve", [])) > 0
                else 0
            )
            if not np.isnan(coppock):
                if coppock > 0:
                    signals["coppock"] = "BULLISH"
                elif coppock < 0:
                    signals["coppock"] = "BEARISH"
                else:
                    signals["coppock"] = "NEUTRAL"
            else:
                signals["coppock"] = "NEUTRAL"

            # KST Oscillator signal
            kst = (
                indicators.get("kst", [0])[-1]
                if len(indicators.get("kst", [])) > 0
                else 0
            )
            kst_signal = (
                indicators.get("kst_signal", [0])[-1]
                if len(indicators.get("kst_signal", [])) > 0
                else 0
            )

            if not np.isnan(kst) and not np.isnan(kst_signal):
                if kst > kst_signal:
                    signals["kst_momentum"] = "BULLISH"
                elif kst < kst_signal:
                    signals["kst_momentum"] = "BEARISH"
                else:
                    signals["kst_momentum"] = "NEUTRAL"
            else:
                signals["kst_momentum"] = "NEUTRAL"

            # Modern composite signal (updated weights to include advanced indicators)
            signal_score = 0

            # KAMA trend weight: 25%
            if signals["kama_trend"] == "BULLISH":
                signal_score += 0.25
            elif signals["kama_trend"] == "BEARISH":
                signal_score -= 0.25

            # PFE momentum weight: 20%
            if signals["pfe_momentum"] in ["STRONG_BULLISH", "BULLISH"]:
                signal_score += (
                    0.2 if signals["pfe_momentum"] == "STRONG_BULLISH" else 0.1
                )
            elif signals["pfe_momentum"] in ["STRONG_BEARISH", "BEARISH"]:
                signal_score -= (
                    0.2 if signals["pfe_momentum"] == "STRONG_BEARISH" else 0.1
                )

            # RVI momentum weight: 15%
            if signals["rvi_momentum"] == "BULLISH":
                signal_score += 0.15
            elif signals["rvi_momentum"] == "BEARISH":
                signal_score -= 0.15

            # Coppock Curve weight: 15%
            if signals["coppock"] == "BULLISH":
                signal_score += 0.15
            elif signals["coppock"] == "BEARISH":
                signal_score -= 0.15

            # KST Oscillator weight: 15%
            if signals["kst_momentum"] == "BULLISH":
                signal_score += 0.15
            elif signals["kst_momentum"] == "BEARISH":
                signal_score -= 0.15

            # K's Envelope mean reversion weight: 5%
            if signals["k_envelope"] == "OVERSOLD":
                signal_score += 0.05
            elif signals["k_envelope"] == "OVERBOUGHT":
                signal_score -= 0.05

            # Fractal pattern weight: 5%
            if signals["fractal_bias"] == "BULLISH":
                signal_score += 0.05
            elif signals["fractal_bias"] == "BEARISH":
                signal_score -= 0.05

            # Generate final signal
            if signal_score > 0.3:
                signals["modern_composite"] = "BUY"
            elif signal_score < -0.3:
                signals["modern_composite"] = "SELL"
            else:
                signals["modern_composite"] = "HOLD"

            signals["modern_signal_strength"] = abs(signal_score)

        except Exception as e:
            logger.error("Error generating modern composite signals", error=str(e))
            signals = {
                "k_envelope": "NEUTRAL",
                "kama_trend": "NEUTRAL",
                "pfe_momentum": "NEUTRAL",
                "fractal_bias": "NEUTRAL",
                "rvi_momentum": "NEUTRAL",
                "coppock": "NEUTRAL",
                "kst_momentum": "NEUTRAL",
                "modern_composite": "HOLD",
                "modern_signal_strength": 0.0,
            }

        return signals

    def _calculate_market_timing_indicators(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
    ) -> Dict:
        """Calculate market timing indicators"""
        indicators = {}

        # Hindenburg Omen
        indicators.update(
            self._calculate_hindenburg_omen(high_prices, low_prices, close_prices)
        )

        # McClellan Oscillator (simplified single-stock version)
        indicators.update(self._calculate_mcclellan_oscillator(close_prices))

        return indicators

    def _calculate_rvi(self, close_prices: np.ndarray, period: int = 14) -> Dict:
        """
        Calculate Relative Vigor Index (RVI)

        RVI compares closing prices to opening prices to measure price momentum
        """
        indicators = {}

        # For single stock, we approximate open prices using lagged close prices
        # In real implementation, this would use actual open prices
        estimated_open = np.roll(close_prices, 1)
        estimated_open[0] = close_prices[0]  # First value approximation

        # Calculate RVI numerator (close - open)
        co_diff = close_prices - estimated_open

        # Calculate RVI denominator (high - low) - approximated as price range
        # We estimate high-low as 2% of close price as reasonable approximation
        hl_range = close_prices * 0.02  # Approximate daily range

        # Smooth both numerator and denominator
        co_smooth = pd.Series(co_diff).rolling(window=period).mean().values
        hl_smooth = pd.Series(hl_range).rolling(window=period).mean().values

        # Calculate RVI
        rvi = np.divide(
            co_smooth, hl_smooth, out=np.zeros_like(co_smooth), where=hl_smooth != 0
        )

        # Calculate RVI signal line (4-period smoothed RVI)
        rvi_signal = pd.Series(rvi).rolling(window=4).mean().values

        indicators["rvi"] = rvi
        indicators["rvi_signal"] = rvi_signal

        return indicators

    def _calculate_coppock_curve(self, close_prices: np.ndarray) -> Dict:
        """
        Calculate Coppock Curve

        Long-term momentum indicator using rate of change and weighted moving average
        """
        indicators = {}

        # Calculate rate of change for 14 and 11 periods
        roc_14 = ((close_prices / np.roll(close_prices, 14)) - 1) * 100
        roc_11 = ((close_prices / np.roll(close_prices, 11)) - 1) * 100

        # Set first values to zero to avoid invalid calculations
        roc_14[:14] = 0
        roc_11[:11] = 0

        # Sum of both ROCs
        roc_sum = roc_14 + roc_11

        # Apply 10-period weighted moving average
        # Simplified WMA calculation
        weights = np.arange(1, 11)  # Weights 1 to 10
        weights = weights / weights.sum()  # Normalize weights

        coppock = np.zeros_like(close_prices)
        for i in range(9, len(close_prices)):
            if i >= 14:  # Ensure we have enough data
                coppock[i] = np.sum(roc_sum[i - 9 : i + 1] * weights)

        indicators["coppock_curve"] = coppock

        return indicators

    def _calculate_kst_oscillator(self, close_prices: np.ndarray) -> Dict:
        """
        Calculate Know Sure Thing (KST) Oscillator

        Momentum oscillator based on multiple rate of change timeframes
        """
        indicators = {}

        # Calculate ROCs for different periods
        roc_10 = ((close_prices / np.roll(close_prices, 10)) - 1) * 100
        roc_15 = ((close_prices / np.roll(close_prices, 15)) - 1) * 100
        roc_20 = ((close_prices / np.roll(close_prices, 20)) - 1) * 100
        roc_30 = ((close_prices / np.roll(close_prices, 30)) - 1) * 100

        # Set initial values to zero
        roc_10[:10] = 0
        roc_15[:15] = 0
        roc_20[:20] = 0
        roc_30[:30] = 0

        # Smooth each ROC with different periods
        roc_10_smooth = pd.Series(roc_10).rolling(window=10).mean().values
        roc_15_smooth = pd.Series(roc_15).rolling(window=10).mean().values
        roc_20_smooth = pd.Series(roc_20).rolling(window=10).mean().values
        roc_30_smooth = pd.Series(roc_30).rolling(window=15).mean().values

        # Calculate KST with different weights
        kst = (
            (roc_10_smooth * 1)
            + (roc_15_smooth * 2)
            + (roc_20_smooth * 3)
            + (roc_30_smooth * 4)
        )

        # Calculate KST signal line
        kst_signal = pd.Series(kst).rolling(window=9).mean().values

        indicators["kst"] = kst
        indicators["kst_signal"] = kst_signal

        return indicators

    def _calculate_hindenburg_omen(
        self, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray
    ) -> Dict:
        """
        Calculate Hindenburg Omen

        Market crash predictor based on new highs vs new lows
        Note: Simplified single-stock version (normally requires market breadth data)
        """
        indicators = {}

        period = 50  # Look-back period for highs/lows
        hindenburg_signals = np.zeros_like(close_prices)

        for i in range(period, len(close_prices)):
            # Check if current price is at or near 52-period high
            recent_high = np.max(high_prices[i - period : i + 1])
            near_high = high_prices[i] >= (recent_high * 0.95)  # Within 5% of high

            # Check if current price is at or near 52-period low
            recent_low = np.min(low_prices[i - period : i + 1])
            near_low = low_prices[i] <= (recent_low * 1.05)  # Within 5% of low

            # Simplified Hindenburg condition
            # In real implementation, this would check market breadth metrics
            if near_high and near_low:  # Paradoxical condition
                hindenburg_signals[i] = 1

        # Count recent Hindenburg signals
        hindenburg_count = np.zeros_like(close_prices)
        for i in range(30, len(close_prices)):
            hindenburg_count[i] = np.sum(hindenburg_signals[i - 30 : i + 1])

        indicators["hindenburg_omen"] = hindenburg_signals
        indicators["hindenburg_count"] = hindenburg_count

        return indicators

    def _calculate_mcclellan_oscillator(self, close_prices: np.ndarray) -> Dict:
        """
        Calculate McClellan Oscillator

        Momentum oscillator for market breadth (simplified single-stock version)
        """
        indicators = {}

        # Calculate price momentum as proxy for advances/declines
        price_change = np.diff(close_prices)
        advances = np.where(price_change > 0, 1, 0)
        declines = np.where(price_change < 0, 1, 0)

        # Add first value
        advances = np.concatenate([[0], advances])
        declines = np.concatenate([[0], declines])

        # Calculate advance-decline difference
        ad_diff = advances - declines

        # Calculate EMAs (19 and 39 periods)
        alpha_19 = 2.0 / (19 + 1)
        alpha_39 = 2.0 / (39 + 1)

        ema_19 = np.zeros_like(ad_diff, dtype=float)
        ema_39 = np.zeros_like(ad_diff, dtype=float)

        ema_19[0] = ad_diff[0]
        ema_39[0] = ad_diff[0]

        for i in range(1, len(ad_diff)):
            ema_19[i] = alpha_19 * ad_diff[i] + (1 - alpha_19) * ema_19[i - 1]
            ema_39[i] = alpha_39 * ad_diff[i] + (1 - alpha_39) * ema_39[i - 1]

        # McClellan Oscillator
        mcclellan = ema_19 - ema_39

        indicators["mcclellan_oscillator"] = mcclellan
        indicators["mcclellan_ema19"] = ema_19
        indicators["mcclellan_ema39"] = ema_39

        return indicators
