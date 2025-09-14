"""
Fibonacci retracement and extension analyzer
"""

from typing import Any, Dict, List

from .base_analyzer import BaseAnalyzer


class FibonacciAnalyzer(BaseAnalyzer):
    """Analyzer for Fibonacci retracements and extensions with Golden Zone emphasis"""

    RETRACEMENT_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]
    EXTENSION_LEVELS = [1.272, 1.414, 1.618, 2.000]
    GOLDEN_ZONE = 0.618  # The most significant Fibonacci level

    async def analyze(
        self, data: List[Dict], lookback_period: int, min_strength: float, **kwargs
    ) -> Dict[str, Any]:
        """Analyze Fibonacci patterns with Golden Zone emphasis"""
        try:
            if not self._validate_data(data):
                return {
                    "patterns": [],
                    "error": "Insufficient data for Fibonacci analysis",
                }

            df = self._prepare_dataframe(data)
            swing_points = self._find_swing_points(df["high"], lookback=5)
            swing_lows = self._find_swing_points(df["low"], lookback=5)["lows"]
            swing_highs = swing_points["highs"]

            if not swing_highs or not swing_lows:
                return {"patterns": [], "error": "No swing points found"}

            patterns_found = []
            current_price = float(df["close"].iloc[-1])
            trend_direction = self._calculate_trend(df["close"])

            # Find most recent significant swing
            recent_high = max(swing_highs, key=lambda x: x["index"])
            recent_low = min(swing_lows, key=lambda x: x["index"])

            if trend_direction == "bullish":
                patterns_found.extend(
                    self._analyze_bullish_fibonacci(
                        recent_low, recent_high, current_price, min_strength
                    )
                )
            elif trend_direction == "bearish":
                patterns_found.extend(
                    self._analyze_bearish_fibonacci(
                        recent_high, recent_low, current_price, min_strength
                    )
                )

            # Add extension levels
            patterns_found.extend(
                self._calculate_extensions(
                    recent_low, recent_high, current_price, trend_direction
                )
            )

            return {
                "patterns": patterns_found,
                "trend_direction": trend_direction,
                "current_price": current_price,
                "swing_high": recent_high["price"],
                "swing_low": recent_low["price"],
                "in_golden_zone": self._is_in_golden_zone(
                    current_price, recent_low, recent_high, trend_direction
                ),
                "golden_zone_level": self._calculate_golden_zone_level(
                    recent_low, recent_high, trend_direction
                ),
            }

        except Exception as e:
            return {"patterns": [], "error": f"Fibonacci analysis failed: {str(e)}"}

    def _analyze_bullish_fibonacci(
        self,
        swing_low: Dict,
        swing_high: Dict,
        current_price: float,
        min_strength: float,
    ) -> List[Dict]:
        """Analyze Fibonacci retracements in bullish trend"""
        patterns = []
        swing_range = swing_high["price"] - swing_low["price"]

        for level in self.RETRACEMENT_LEVELS:
            fib_price = swing_high["price"] - (swing_range * level)
            distance_from_fib = abs(current_price - fib_price) / current_price

            if distance_from_fib < 0.02:  # Within 2% of Fibonacci level
                strength = max(0.3, 1.0 - (distance_from_fib * 50))

                if strength >= min_strength:
                    pattern = {
                        "name": f"Fibonacci Retracement {level:.1%}",
                        "type": (
                            "support"
                            if level != self.GOLDEN_ZONE
                            else "golden_zone_support"
                        ),
                        "level": fib_price,
                        "ratio": level,
                        "strength": strength,
                        "signal": (
                            "strong_support" if level == self.GOLDEN_ZONE else "support"
                        ),
                        "is_golden_zone": level == self.GOLDEN_ZONE,
                        "entry_zone": fib_price,
                        "stop_loss": swing_low["price"] * 0.98,  # 2% below swing low
                        "target": (
                            swing_high["price"] * 1.05
                            if level == self.GOLDEN_ZONE
                            else swing_high["price"]
                        ),
                    }

                    if level == self.GOLDEN_ZONE:
                        pattern["confluence_note"] = (
                            "High probability reversal zone - ideal for entry"
                        )

                    patterns.append(pattern)

        return patterns

    def _analyze_bearish_fibonacci(
        self,
        swing_high: Dict,
        swing_low: Dict,
        current_price: float,
        min_strength: float,
    ) -> List[Dict]:
        """Analyze Fibonacci retracements in bearish trend"""
        patterns = []
        swing_range = swing_high["price"] - swing_low["price"]

        for level in self.RETRACEMENT_LEVELS:
            fib_price = swing_low["price"] + (swing_range * level)
            distance_from_fib = abs(current_price - fib_price) / current_price

            if distance_from_fib < 0.02:  # Within 2% of Fibonacci level
                strength = max(0.3, 1.0 - (distance_from_fib * 50))

                if strength >= min_strength:
                    pattern = {
                        "name": f"Fibonacci Retracement {level:.1%}",
                        "type": (
                            "resistance"
                            if level != self.GOLDEN_ZONE
                            else "golden_zone_resistance"
                        ),
                        "level": fib_price,
                        "ratio": level,
                        "strength": strength,
                        "signal": (
                            "strong_resistance"
                            if level == self.GOLDEN_ZONE
                            else "resistance"
                        ),
                        "is_golden_zone": level == self.GOLDEN_ZONE,
                        "entry_zone": fib_price,
                        "stop_loss": swing_high["price"] * 1.02,  # 2% above swing high
                        "target": (
                            swing_low["price"] * 0.95
                            if level == self.GOLDEN_ZONE
                            else swing_low["price"]
                        ),
                    }

                    if level == self.GOLDEN_ZONE:
                        pattern["confluence_note"] = (
                            "High probability reversal zone - ideal for entry"
                        )

                    patterns.append(pattern)

        return patterns

    def _calculate_extensions(
        self,
        swing_low: Dict,
        swing_high: Dict,
        current_price: float,
        trend_direction: str,
    ) -> List[Dict]:
        """Calculate Fibonacci extension levels"""
        patterns = []
        swing_range = swing_high["price"] - swing_low["price"]

        for level in self.EXTENSION_LEVELS:
            if trend_direction == "bullish":
                extension_price = swing_high["price"] + (swing_range * (level - 1))
                if (
                    extension_price > current_price * 1.01
                ):  # Only show extensions above current price
                    patterns.append(
                        {
                            "name": f"Fibonacci Extension {level:.1%}",
                            "type": "target",
                            "level": extension_price,
                            "ratio": level,
                            "strength": 0.7,
                            "signal": "profit_target",
                            "is_golden_zone": level == 1.618,
                            "target_type": "extension",
                        }
                    )
            elif trend_direction == "bearish":
                extension_price = swing_low["price"] - (swing_range * (level - 1))
                if (
                    extension_price < current_price * 0.99
                ):  # Only show extensions below current price
                    patterns.append(
                        {
                            "name": f"Fibonacci Extension {level:.1%}",
                            "type": "target",
                            "level": extension_price,
                            "ratio": level,
                            "strength": 0.7,
                            "signal": "profit_target",
                            "is_golden_zone": level == 1.618,
                            "target_type": "extension",
                        }
                    )

        return patterns

    def _is_in_golden_zone(
        self,
        current_price: float,
        swing_low: Dict,
        swing_high: Dict,
        trend_direction: str,
    ) -> bool:
        """Check if current price is in the Golden Zone"""
        swing_range = swing_high["price"] - swing_low["price"]

        if trend_direction == "bullish":
            golden_zone_fib = swing_high["price"] - (swing_range * self.GOLDEN_ZONE)
        else:
            golden_zone_fib = swing_low["price"] + (swing_range * self.GOLDEN_ZONE)

        golden_zone_distance = abs(current_price - golden_zone_fib) / current_price
        return golden_zone_distance < 0.015  # Within 1.5% of Golden Zone

    def _calculate_golden_zone_level(
        self, swing_low: Dict, swing_high: Dict, trend_direction: str
    ) -> float:
        """Calculate the Golden Zone price level"""
        swing_range = swing_high["price"] - swing_low["price"]

        if trend_direction == "bullish":
            return swing_high["price"] - (swing_range * self.GOLDEN_ZONE)
        else:
            return swing_low["price"] + (swing_range * self.GOLDEN_ZONE)
