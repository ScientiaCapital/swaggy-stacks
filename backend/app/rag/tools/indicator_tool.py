"""
Technical Indicator Tools for LangChain Integration

Provides comprehensive technical analysis indicators as LangChain Tools
"""

import json
import logging
from typing import List, Optional

import yfinance as yf
from langchain.agents import Tool

logger = logging.getLogger(__name__)


class IndicatorTool:
    """Technical indicator tools for market analysis"""

    def get_tools(self) -> List[Tool]:
        """Get all technical indicator tools"""
        return [
            Tool(
                name="calculate_rsi",
                description="Calculate RSI for a symbol (format: 'SYMBOL,PERIOD' e.g., 'AAPL,14')",
                func=self._calculate_rsi,
            ),
            Tool(
                name="calculate_macd",
                description="Calculate MACD for a symbol (format: 'SYMBOL' or 'SYMBOL,FAST,SLOW,SIGNAL')",
                func=self._calculate_macd,
            ),
            Tool(
                name="calculate_bollinger_bands",
                description="Calculate Bollinger Bands (format: 'SYMBOL,PERIOD,STD_DEV' e.g., 'AAPL,20,2')",
                func=self._calculate_bollinger_bands,
            ),
            Tool(
                name="calculate_fibonacci_levels",
                description="Calculate Fibonacci retracement levels for a symbol",
                func=self._calculate_fibonacci_levels,
            ),
            Tool(
                name="calculate_moving_averages",
                description="Calculate various moving averages (format: 'SYMBOL,PERIODS' e.g., 'AAPL,20,50,200')",
                func=self._calculate_moving_averages,
            ),
            Tool(
                name="calculate_stochastic",
                description="Calculate Stochastic oscillator (format: 'SYMBOL,K_PERIOD,D_PERIOD' e.g., 'AAPL,14,3')",
                func=self._calculate_stochastic,
            ),
        ]

    def _calculate_rsi(self, params: str) -> str:
        """Calculate RSI indicator"""
        try:
            parts = params.split(",")
            symbol = parts[0].strip().upper()
            period = int(parts[1]) if len(parts) > 1 else 14

            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")  # Get enough data for calculation

            if hist.empty or len(hist) < period + 10:
                return f"Insufficient data for RSI calculation for {symbol}"

            # Calculate RSI
            closes = hist["Close"]
            delta = closes.diff()

            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            current_rsi = float(rsi.iloc[-1])
            prev_rsi = float(rsi.iloc[-2]) if len(rsi) > 1 else current_rsi

            # RSI interpretation
            if current_rsi > 70:
                condition = "overbought"
            elif current_rsi < 30:
                condition = "oversold"
            else:
                condition = "neutral"

            return json.dumps(
                {
                    "symbol": symbol,
                    "indicator": "RSI",
                    "period": period,
                    "current_value": round(current_rsi, 2),
                    "previous_value": round(prev_rsi, 2),
                    "change": round(current_rsi - prev_rsi, 2),
                    "condition": condition,
                    "interpretation": {
                        "overbought_threshold": 70,
                        "oversold_threshold": 30,
                        "trend": "rising" if current_rsi > prev_rsi else "falling",
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return f"Error calculating RSI: {str(e)}"

    def _calculate_macd(self, params: str) -> str:
        """Calculate MACD indicator"""
        try:
            parts = params.split(",")
            symbol = parts[0].strip().upper()
            fast_period = int(parts[1]) if len(parts) > 1 else 12
            slow_period = int(parts[2]) if len(parts) > 2 else 26
            signal_period = int(parts[3]) if len(parts) > 3 else 9

            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")

            if hist.empty or len(hist) < slow_period + signal_period + 10:
                return f"Insufficient data for MACD calculation for {symbol}"

            # Calculate MACD
            closes = hist["Close"]

            exp1 = closes.ewm(span=fast_period).mean()
            exp2 = closes.ewm(span=slow_period).mean()

            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal_period).mean()
            histogram = macd_line - signal_line

            current_macd = float(macd_line.iloc[-1])
            current_signal = float(signal_line.iloc[-1])
            current_histogram = float(histogram.iloc[-1])

            prev_histogram = (
                float(histogram.iloc[-2]) if len(histogram) > 1 else current_histogram
            )

            # MACD interpretation
            if current_macd > current_signal:
                trend = "bullish"
            else:
                trend = "bearish"

            if current_histogram > prev_histogram:
                momentum = "strengthening"
            else:
                momentum = "weakening"

            return json.dumps(
                {
                    "symbol": symbol,
                    "indicator": "MACD",
                    "parameters": {
                        "fast_period": fast_period,
                        "slow_period": slow_period,
                        "signal_period": signal_period,
                    },
                    "values": {
                        "macd_line": round(current_macd, 4),
                        "signal_line": round(current_signal, 4),
                        "histogram": round(current_histogram, 4),
                    },
                    "signals": {
                        "trend": trend,
                        "momentum": momentum,
                        "crossover": (
                            "bullish" if current_macd > current_signal else "bearish"
                        ),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return f"Error calculating MACD: {str(e)}"

    def _calculate_bollinger_bands(self, params: str) -> str:
        """Calculate Bollinger Bands"""
        try:
            parts = params.split(",")
            symbol = parts[0].strip().upper()
            period = int(parts[1]) if len(parts) > 1 else 20
            std_dev = float(parts[2]) if len(parts) > 2 else 2.0

            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")

            if hist.empty or len(hist) < period + 10:
                return f"Insufficient data for Bollinger Bands calculation for {symbol}"

            # Calculate Bollinger Bands
            closes = hist["Close"]

            sma = closes.rolling(window=period).mean()
            std = closes.rolling(window=period).std()

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            current_price = float(closes.iloc[-1])
            current_sma = float(sma.iloc[-1])
            current_upper = float(upper_band.iloc[-1])
            current_lower = float(lower_band.iloc[-1])

            # Calculate position within bands
            band_width = current_upper - current_lower
            position = (
                (current_price - current_lower) / band_width if band_width > 0 else 0.5
            )

            # Bollinger Band interpretation
            if current_price > current_upper:
                condition = "above_upper_band"
            elif current_price < current_lower:
                condition = "below_lower_band"
            elif position > 0.8:
                condition = "near_upper_band"
            elif position < 0.2:
                condition = "near_lower_band"
            else:
                condition = "within_bands"

            return json.dumps(
                {
                    "symbol": symbol,
                    "indicator": "Bollinger_Bands",
                    "period": period,
                    "std_deviation": std_dev,
                    "current_price": round(current_price, 2),
                    "bands": {
                        "upper_band": round(current_upper, 2),
                        "middle_band": round(current_sma, 2),
                        "lower_band": round(current_lower, 2),
                    },
                    "analysis": {
                        "position_in_bands": round(position, 3),
                        "condition": condition,
                        "band_width": round(band_width, 2),
                        "squeeze": band_width
                        < (current_sma * 0.1),  # Bollinger squeeze
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return f"Error calculating Bollinger Bands: {str(e)}"

    def _calculate_fibonacci_levels(self, symbol: str) -> str:
        """Calculate Fibonacci retracement levels"""
        try:
            symbol = symbol.strip().upper()

            # Get historical data (longer period to find swing high/low)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")

            if hist.empty:
                return f"No data available for Fibonacci calculation for {symbol}"

            # Find recent swing high and low
            highs = hist["High"]
            lows = hist["Low"]

            # Use rolling max/min to find significant levels
            swing_high = float(highs.rolling(window=20).max().max())
            swing_low = float(lows.rolling(window=20).min().min())

            current_price = float(hist["Close"].iloc[-1])

            # Calculate Fibonacci levels
            price_range = swing_high - swing_low
            fib_levels = {
                "0.0": swing_high,
                "23.6": swing_high - (price_range * 0.236),
                "38.2": swing_high - (price_range * 0.382),
                "50.0": swing_high - (price_range * 0.500),
                "61.8": swing_high - (price_range * 0.618),
                "78.6": swing_high - (price_range * 0.786),
                "100.0": swing_low,
            }

            # Find closest level
            closest_level = min(
                fib_levels.items(), key=lambda x: abs(x[1] - current_price)
            )

            return json.dumps(
                {
                    "symbol": symbol,
                    "indicator": "Fibonacci_Retracement",
                    "swing_high": round(swing_high, 2),
                    "swing_low": round(swing_low, 2),
                    "current_price": round(current_price, 2),
                    "levels": {k: round(v, 2) for k, v in fib_levels.items()},
                    "analysis": {
                        "closest_level": f"{closest_level[0]}% ({round(closest_level[1], 2)})",
                        "distance_to_closest": round(
                            abs(closest_level[1] - current_price), 2
                        ),
                        "trend_direction": (
                            "retracement"
                            if swing_low < current_price < swing_high
                            else "extension"
                        ),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return f"Error calculating Fibonacci levels: {str(e)}"

    def _calculate_moving_averages(self, params: str) -> str:
        """Calculate various moving averages"""
        try:
            parts = params.split(",")
            symbol = parts[0].strip().upper()
            periods = (
                [int(p.strip()) for p in parts[1:]] if len(parts) > 1 else [20, 50, 200]
            )

            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                return f"No data available for moving averages for {symbol}"

            closes = hist["Close"]
            current_price = float(closes.iloc[-1])

            # Calculate moving averages
            mas = {}
            for period in periods:
                if len(closes) >= period:
                    ma = closes.rolling(window=period).mean()
                    mas[f"MA{period}"] = float(ma.iloc[-1])
                else:
                    mas[f"MA{period}"] = None

            # Analyze trends
            trend_analysis = {}
            for period in periods:
                ma_key = f"MA{period}"
                if mas[ma_key] is not None:
                    if current_price > mas[ma_key]:
                        trend_analysis[ma_key] = "above"
                    else:
                        trend_analysis[ma_key] = "below"
                else:
                    trend_analysis[ma_key] = "insufficient_data"

            return json.dumps(
                {
                    "symbol": symbol,
                    "indicator": "Moving_Averages",
                    "current_price": round(current_price, 2),
                    "moving_averages": {
                        k: round(v, 2) if v is not None else None
                        for k, v in mas.items()
                    },
                    "trend_analysis": trend_analysis,
                    "golden_cross": (
                        mas.get("MA50")
                        and mas.get("MA200")
                        and mas["MA50"] > mas["MA200"]
                        and current_price > mas["MA50"]
                    ),
                    "death_cross": (
                        mas.get("MA50")
                        and mas.get("MA200")
                        and mas["MA50"] < mas["MA200"]
                        and current_price < mas["MA50"]
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return f"Error calculating moving averages: {str(e)}"

    def _calculate_stochastic(self, params: str) -> str:
        """Calculate Stochastic oscillator"""
        try:
            parts = params.split(",")
            symbol = parts[0].strip().upper()
            k_period = int(parts[1]) if len(parts) > 1 else 14
            d_period = int(parts[2]) if len(parts) > 2 else 3

            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")

            if hist.empty or len(hist) < k_period + d_period + 10:
                return f"Insufficient data for Stochastic calculation for {symbol}"

            # Calculate Stochastic
            highs = hist["High"]
            lows = hist["Low"]
            closes = hist["Close"]

            # %K calculation
            lowest_lows = lows.rolling(window=k_period).min()
            highest_highs = highs.rolling(window=k_period).max()

            k_percent = 100 * (closes - lowest_lows) / (highest_highs - lowest_lows)

            # %D calculation (moving average of %K)
            d_percent = k_percent.rolling(window=d_period).mean()

            current_k = float(k_percent.iloc[-1])
            current_d = float(d_percent.iloc[-1])
            prev_k = float(k_percent.iloc[-2]) if len(k_percent) > 1 else current_k

            # Stochastic interpretation
            if current_k > 80 and current_d > 80:
                condition = "overbought"
            elif current_k < 20 and current_d < 20:
                condition = "oversold"
            else:
                condition = "neutral"

            # Signal analysis
            if current_k > current_d and prev_k <= d_percent.iloc[-2]:
                signal = "bullish_crossover"
            elif current_k < current_d and prev_k >= d_percent.iloc[-2]:
                signal = "bearish_crossover"
            else:
                signal = "no_crossover"

            return json.dumps(
                {
                    "symbol": symbol,
                    "indicator": "Stochastic",
                    "parameters": {"k_period": k_period, "d_period": d_period},
                    "values": {
                        "percent_k": round(current_k, 2),
                        "percent_d": round(current_d, 2),
                    },
                    "analysis": {
                        "condition": condition,
                        "signal": signal,
                        "momentum": "rising" if current_k > prev_k else "falling",
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return f"Error calculating Stochastic: {str(e)}"


# Global indicator tool instance
_indicator_tool: Optional[IndicatorTool] = None


def get_indicator_tool() -> IndicatorTool:
    """Get the global indicator tool instance"""
    global _indicator_tool

    if _indicator_tool is None:
        _indicator_tool = IndicatorTool()

    return _indicator_tool
