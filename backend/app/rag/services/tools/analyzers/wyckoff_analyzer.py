"""
Wyckoff Method analyzer for market phases and volume analysis
"""

from typing import Any, Dict, List

import numpy as np

from .base_analyzer import BaseAnalyzer


class WyckoffAnalyzer(BaseAnalyzer):
    """Analyzer for Wyckoff Method phases and volume-price relationships"""

    def __init__(self):
        super().__init__()
        self.phases = {
            "accumulation": [
                "preliminary_support",
                "selling_climax",
                "automatic_reaction",
                "secondary_test",
                "spring",
            ],
            "markup": ["sign_of_strength", "last_point_of_support", "backup_to_edge"],
            "distribution": [
                "preliminary_supply",
                "buying_climax",
                "automatic_reaction",
                "secondary_test",
                "upthrust",
            ],
            "markdown": ["sign_of_weakness", "last_point_of_supply", "backup_to_edge"],
        }

    async def analyze(
        self, data: List[Dict], lookback_period: int, min_strength: float, **kwargs
    ) -> Dict[str, Any]:
        """Analyze Wyckoff patterns with volume-price relationships"""
        try:
            if not self._validate_data(data, min_length=30):
                return {
                    "patterns": [],
                    "error": "Insufficient data for Wyckoff analysis",
                }

            df = self._prepare_dataframe(data)

            # Check if volume data is available
            if "volume" not in df.columns:
                return {
                    "patterns": [],
                    "error": "Volume data required for Wyckoff analysis",
                }

            patterns_found = []
            current_price = float(df["close"].iloc[-1])

            # Analyze market phases
            accumulation_patterns = self._find_accumulation_phase(df, min_strength)
            distribution_patterns = self._find_distribution_phase(df, min_strength)
            markup_patterns = self._find_markup_phase(df, min_strength)
            markdown_patterns = self._find_markdown_phase(df, min_strength)

            patterns_found.extend(accumulation_patterns)
            patterns_found.extend(distribution_patterns)
            patterns_found.extend(markup_patterns)
            patterns_found.extend(markdown_patterns)

            # Analyze key Wyckoff events
            springs_upthrusts = self._find_springs_and_upthrusts(df, min_strength)
            patterns_found.extend(springs_upthrusts)

            return {
                "patterns": patterns_found,
                "current_price": current_price,
                "current_phase": self._determine_current_phase(df),
                "volume_analysis": self._analyze_volume_pattern(df),
                "key_levels": self._identify_key_levels(df),
            }

        except Exception as e:
            return {"patterns": [], "error": f"Wyckoff analysis failed: {str(e)}"}

    def _find_accumulation_phase(self, df, min_strength: float) -> List[Dict]:
        """Identify accumulation phase characteristics"""
        patterns = []

        # Look for accumulation signs: declining volume, price stability
        recent_data = df.tail(20)
        price_stability = recent_data["close"].std() / recent_data["close"].mean()
        volume_trend = self._calculate_volume_trend(recent_data)

        if (
            price_stability < 0.03 and volume_trend < -0.1
        ):  # Low volatility, declining volume
            patterns.append(
                {
                    "name": "Wyckoff Accumulation Phase",
                    "type": "accumulation",
                    "phase": "accumulation",
                    "strength": min(0.8, 1.0 - price_stability * 10),
                    "signal": "accumulation_in_progress",
                    "characteristics": {
                        "price_stability": price_stability,
                        "volume_trend": volume_trend,
                        "phase_confidence": (
                            "medium" if price_stability < 0.02 else "low"
                        ),
                    },
                    "trading_strategy": "Look for spring (false breakdown) to confirm accumulation completion",
                }
            )

        return patterns

    def _find_distribution_phase(self, df, min_strength: float) -> List[Dict]:
        """Identify distribution phase characteristics"""
        patterns = []

        recent_data = df.tail(20)
        price_stability = recent_data["close"].std() / recent_data["close"].mean()
        volume_analysis = self._analyze_distribution_volume(recent_data)

        if price_stability < 0.04 and volume_analysis["high_volume_days"] > 0.3:
            patterns.append(
                {
                    "name": "Wyckoff Distribution Phase",
                    "type": "distribution",
                    "phase": "distribution",
                    "strength": min(0.8, volume_analysis["strength"]),
                    "signal": "distribution_in_progress",
                    "characteristics": {
                        "price_stability": price_stability,
                        "high_volume_percentage": volume_analysis["high_volume_days"],
                        "phase_confidence": (
                            "high"
                            if volume_analysis["high_volume_days"] > 0.4
                            else "medium"
                        ),
                    },
                    "trading_strategy": "Look for upthrust (false breakout) to confirm distribution completion",
                }
            )

        return patterns

    def _find_markup_phase(self, df, min_strength: float) -> List[Dict]:
        """Identify markup phase characteristics"""
        patterns = []

        recent_data = df.tail(15)
        price_trend = self._calculate_trend_strength(recent_data["close"])
        volume_confirmation = self._check_markup_volume(recent_data)

        if price_trend > 0.05 and volume_confirmation:
            patterns.append(
                {
                    "name": "Wyckoff Markup Phase",
                    "type": "markup",
                    "phase": "markup",
                    "strength": min(0.9, price_trend * 10),
                    "signal": "strong_uptrend",
                    "characteristics": {
                        "trend_strength": price_trend,
                        "volume_confirmation": volume_confirmation,
                        "phase_confidence": "high",
                    },
                    "trading_strategy": "Ride the trend, look for backup to edge of creek for entries",
                }
            )

        return patterns

    def _find_markdown_phase(self, df, min_strength: float) -> List[Dict]:
        """Identify markdown phase characteristics"""
        patterns = []

        recent_data = df.tail(15)
        price_trend = self._calculate_trend_strength(recent_data["close"])
        volume_confirmation = self._check_markdown_volume(recent_data)

        if price_trend < -0.05 and volume_confirmation:
            patterns.append(
                {
                    "name": "Wyckoff Markdown Phase",
                    "type": "markdown",
                    "phase": "markdown",
                    "strength": min(0.9, abs(price_trend) * 10),
                    "signal": "strong_downtrend",
                    "characteristics": {
                        "trend_strength": price_trend,
                        "volume_confirmation": volume_confirmation,
                        "phase_confidence": "high",
                    },
                    "trading_strategy": "Short positions, look for backup to edge for entries",
                }
            )

        return patterns

    def _find_springs_and_upthrusts(self, df, min_strength: float) -> List[Dict]:
        """Find springs (false breakdowns) and upthrusts (false breakouts)"""
        patterns = []

        # Find recent swing points
        swings = self._find_swing_points(df["close"], lookback=5)

        # Look for springs (false breakdowns followed by recovery)
        springs = self._identify_springs(df, swings)
        patterns.extend(springs)

        # Look for upthrusts (false breakouts followed by decline)
        upthrusts = self._identify_upthrusts(df, swings)
        patterns.extend(upthrusts)

        return patterns

    def _identify_springs(self, df, swings: Dict) -> List[Dict]:
        """Identify spring patterns (false breakdowns)"""
        springs = []

        if not swings["lows"]:
            return springs

        recent_lows = swings["lows"][-3:]  # Last 3 lows
        current_price = df["close"].iloc[-1]

        for i, low in enumerate(recent_lows[:-1]):
            next_low = recent_lows[i + 1]

            # Spring: price breaks below previous low but recovers quickly
            if (
                next_low["price"] < low["price"] * 0.98  # Break below previous low
                and current_price > next_low["price"] * 1.02
            ):  # Recovery above the low

                springs.append(
                    {
                        "name": "Wyckoff Spring",
                        "type": "spring",
                        "event_type": "false_breakdown",
                        "strength": 0.8,
                        "signal": "accumulation_complete",
                        "spring_low": next_low["price"],
                        "recovery_level": current_price,
                        "confirmation": current_price > low["price"],
                        "trading_strategy": "Buy after spring confirmation with stop below spring low",
                    }
                )

        return springs

    def _identify_upthrusts(self, df, swings: Dict) -> List[Dict]:
        """Identify upthrust patterns (false breakouts)"""
        upthrusts = []

        if not swings["highs"]:
            return upthrusts

        recent_highs = swings["highs"][-3:]  # Last 3 highs
        current_price = df["close"].iloc[-1]

        for i, high in enumerate(recent_highs[:-1]):
            next_high = recent_highs[i + 1]

            # Upthrust: price breaks above previous high but fails quickly
            if (
                next_high["price"] > high["price"] * 1.02  # Break above previous high
                and current_price < next_high["price"] * 0.98
            ):  # Failure below the high

                upthrusts.append(
                    {
                        "name": "Wyckoff Upthrust",
                        "type": "upthrust",
                        "event_type": "false_breakout",
                        "strength": 0.8,
                        "signal": "distribution_complete",
                        "upthrust_high": next_high["price"],
                        "failure_level": current_price,
                        "confirmation": current_price < high["price"],
                        "trading_strategy": "Short after upthrust confirmation with stop above upthrust high",
                    }
                )

        return upthrusts

    def _calculate_volume_trend(self, df) -> float:
        """Calculate volume trend over the period"""
        try:
            volume_data = df["volume"].values
            x = np.arange(len(volume_data))
            slope = np.polyfit(x, volume_data, 1)[0]
            return slope / np.mean(volume_data)  # Normalize by average volume
        except Exception:
            return 0.0

    def _analyze_distribution_volume(self, df) -> Dict[str, Any]:
        """Analyze volume patterns during potential distribution"""
        try:
            avg_volume = df["volume"].mean()
            high_volume_days = len(df[df["volume"] > avg_volume * 1.5]) / len(df)

            return {
                "high_volume_days": high_volume_days,
                "strength": min(0.8, high_volume_days * 2),
            }
        except Exception:
            return {"high_volume_days": 0, "strength": 0}

    def _calculate_trend_strength(self, prices) -> float:
        """Calculate trend strength using linear regression slope"""
        try:
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            return slope / np.mean(prices)  # Normalize by average price
        except Exception:
            return 0.0

    def _check_markup_volume(self, df) -> bool:
        """Check if volume confirms markup phase"""
        try:
            # Volume should increase on up days during markup
            up_days = df[df["close"] > df["close"].shift(1)]
            if len(up_days) == 0:
                return False

            avg_up_volume = up_days["volume"].mean()
            avg_total_volume = df["volume"].mean()

            return avg_up_volume > avg_total_volume * 1.1
        except Exception:
            return False

    def _check_markdown_volume(self, df) -> bool:
        """Check if volume confirms markdown phase"""
        try:
            # Volume should increase on down days during markdown
            down_days = df[df["close"] < df["close"].shift(1)]
            if len(down_days) == 0:
                return False

            avg_down_volume = down_days["volume"].mean()
            avg_total_volume = df["volume"].mean()

            return avg_down_volume > avg_total_volume * 1.1
        except Exception:
            return False

    def _determine_current_phase(self, df) -> str:
        """Determine the current Wyckoff phase"""
        try:
            recent_data = df.tail(20)

            # Analyze price and volume characteristics
            price_volatility = recent_data["close"].std() / recent_data["close"].mean()
            volume_trend = self._calculate_volume_trend(recent_data)
            price_trend = self._calculate_trend_strength(recent_data["close"])

            if abs(price_trend) < 0.02 and price_volatility < 0.03:
                if volume_trend < -0.1:
                    return "accumulation"
                else:
                    return "distribution"
            elif price_trend > 0.03:
                return "markup"
            elif price_trend < -0.03:
                return "markdown"
            else:
                return "transitional"

        except Exception:
            return "unknown"

    def _analyze_volume_pattern(self, df) -> Dict[str, Any]:
        """Analyze overall volume patterns"""
        try:
            recent_volume = df["volume"].tail(10)
            avg_volume = df["volume"].mean()

            return {
                "current_vs_average": recent_volume.mean() / avg_volume,
                "volume_trend": self._calculate_volume_trend(df.tail(20)),
                "volume_spikes": len(recent_volume[recent_volume > avg_volume * 2]),
                "volume_classification": self._classify_volume_pattern(
                    recent_volume, avg_volume
                ),
            }
        except Exception:
            return {"error": "Volume analysis failed"}

    def _classify_volume_pattern(self, recent_volume, avg_volume) -> str:
        """Classify the current volume pattern"""
        recent_avg = recent_volume.mean()

        if recent_avg > avg_volume * 1.5:
            return "high_volume"
        elif recent_avg < avg_volume * 0.7:
            return "low_volume"
        else:
            return "normal_volume"

    def _identify_key_levels(self, df) -> Dict[str, float]:
        """Identify key Wyckoff support/resistance levels"""
        try:
            swings = self._find_swing_points(df["close"])

            levels = {}

            if swings["highs"]:
                recent_highs = [h["price"] for h in swings["highs"][-3:]]
                levels["resistance"] = max(recent_highs)
                levels["secondary_resistance"] = (
                    sorted(recent_highs, reverse=True)[1]
                    if len(recent_highs) > 1
                    else None
                )

            if swings["lows"]:
                recent_lows = [l["price"] for l in swings["lows"][-3:]]
                levels["support"] = min(recent_lows)
                levels["secondary_support"] = (
                    sorted(recent_lows)[1] if len(recent_lows) > 1 else None
                )

            return levels

        except Exception:
            return {}
