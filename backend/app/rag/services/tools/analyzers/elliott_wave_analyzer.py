"""
Elliott Wave pattern analyzer
"""

from typing import Any, Dict, List

from .base_analyzer import BaseAnalyzer


class ElliottWaveAnalyzer(BaseAnalyzer):
    """Analyzer for Elliott Wave patterns with 5-3 wave structure"""

    def __init__(self):
        super().__init__()
        self.wave_ratios = {
            "wave_2_retracement": [0.382, 0.500, 0.618],
            "wave_4_retracement": [0.236, 0.382, 0.500],
            "wave_3_extension": [1.618, 2.618, 4.236],
            "wave_5_extension": [0.618, 1.000, 1.618],
        }

    async def analyze(
        self, data: List[Dict], lookback_period: int, min_strength: float, **kwargs
    ) -> Dict[str, Any]:
        """Analyze Elliott Wave patterns"""
        try:
            if not self._validate_data(
                data, min_length=50
            ):  # Need more data for wave analysis
                return {
                    "patterns": [],
                    "error": "Insufficient data for Elliott Wave analysis",
                }

            df = self._prepare_dataframe(data)
            swing_points = self._find_detailed_swings(df)

            if len(swing_points) < 5:
                return {
                    "patterns": [],
                    "error": "Not enough swing points for wave analysis",
                }

            patterns_found = []
            current_price = float(df["close"].iloc[-1])

            # Analyze impulse patterns (5 waves)
            impulse_patterns = self._find_impulse_patterns(
                swing_points, current_price, min_strength
            )
            patterns_found.extend(impulse_patterns)

            # Analyze corrective patterns (3 waves)
            corrective_patterns = self._find_corrective_patterns(
                swing_points, current_price, min_strength
            )
            patterns_found.extend(corrective_patterns)

            return {
                "patterns": patterns_found,
                "current_price": current_price,
                "swing_points_analyzed": len(swing_points),
                "wave_count": self._get_current_wave_count(swing_points),
                "probable_wave_position": self._determine_wave_position(
                    swing_points, current_price
                ),
            }

        except Exception as e:
            return {"patterns": [], "error": f"Elliott Wave analysis failed: {str(e)}"}

    def _find_detailed_swings(self, df) -> List[Dict]:
        """Find detailed swing points for wave analysis"""
        highs_data = self._find_swing_points(df["high"], lookback=3)
        lows_data = self._find_swing_points(df["low"], lookback=3)

        # Combine and sort by index
        all_swings = []

        for high in highs_data["highs"]:
            all_swings.append(
                {"index": high["index"], "price": high["price"], "type": "high"}
            )

        for low in lows_data["lows"]:
            all_swings.append(
                {"index": low["index"], "price": low["price"], "type": "low"}
            )

        # Sort by index and filter alternating highs/lows
        all_swings.sort(key=lambda x: x["index"])
        filtered_swings = []
        last_type = None

        for swing in all_swings:
            if swing["type"] != last_type:
                filtered_swings.append(swing)
                last_type = swing["type"]

        return filtered_swings[-10:]  # Keep last 10 swings for analysis

    def _find_impulse_patterns(
        self, swings: List[Dict], current_price: float, min_strength: float
    ) -> List[Dict]:
        """Find 5-wave impulse patterns"""
        patterns = []

        if len(swings) < 9:  # Need at least 9 points for a 5-wave pattern
            return patterns

        # Look for 5-wave structures in the most recent swings
        for start_idx in range(len(swings) - 8):
            wave_candidate = swings[start_idx : start_idx + 9]  # 9 points = 5 waves

            if self._validate_impulse_structure(wave_candidate):
                pattern_strength = self._calculate_impulse_strength(wave_candidate)

                if pattern_strength >= min_strength:
                    wave_direction = (
                        "bullish"
                        if wave_candidate[-1]["price"] > wave_candidate[0]["price"]
                        else "bearish"
                    )

                    pattern = {
                        "name": f"Elliott Wave Impulse - {wave_direction.title()}",
                        "type": "impulse",
                        "wave_type": "5_wave_impulse",
                        "strength": pattern_strength,
                        "signal": (
                            "trend_continuation"
                            if self._is_trend_continuing(wave_candidate, current_price)
                            else "trend_completion"
                        ),
                        "wave_structure": self._describe_wave_structure(wave_candidate),
                        "fibonacci_ratios": self._check_wave_ratios(wave_candidate),
                        "wave_1_start": wave_candidate[0]["price"],
                        "wave_5_end": wave_candidate[-1]["price"],
                        "direction": wave_direction,
                        "completion_percentage": self._estimate_completion(
                            wave_candidate, current_price
                        ),
                    }

                    patterns.append(pattern)

        return patterns

    def _find_corrective_patterns(
        self, swings: List[Dict], current_price: float, min_strength: float
    ) -> List[Dict]:
        """Find 3-wave corrective patterns (A-B-C)"""
        patterns = []

        if len(swings) < 7:  # Need at least 7 points for a 3-wave pattern
            return patterns

        # Look for 3-wave ABC structures
        for start_idx in range(len(swings) - 6):
            wave_candidate = swings[start_idx : start_idx + 7]  # 7 points = 3 waves

            if self._validate_corrective_structure(wave_candidate):
                pattern_strength = self._calculate_corrective_strength(wave_candidate)

                if pattern_strength >= min_strength:
                    correction_type = self._identify_correction_type(wave_candidate)

                    pattern = {
                        "name": f"Elliott Wave Correction - {correction_type}",
                        "type": "corrective",
                        "wave_type": "3_wave_correction",
                        "correction_type": correction_type,
                        "strength": pattern_strength,
                        "signal": "correction_in_progress",
                        "wave_a_start": wave_candidate[0]["price"],
                        "wave_c_end": wave_candidate[-1]["price"],
                        "retracement_level": self._calculate_retracement_level(
                            wave_candidate
                        ),
                        "fibonacci_confirmation": self._check_corrective_ratios(
                            wave_candidate
                        ),
                    }

                    patterns.append(pattern)

        return patterns

    def _validate_impulse_structure(self, waves: List[Dict]) -> bool:
        """Validate 5-wave impulse structure rules"""
        if len(waves) != 9:
            return False

        try:
            # Extract wave extremes (0, 2, 4, 6, 8 are the wave endpoints)
            w1_start, w1_end = waves[0]["price"], waves[2]["price"]
            w2_end = waves[4]["price"]
            w3_end = waves[6]["price"]
            w4_end = waves[8]["price"]
            w5_end = waves[8]["price"]  # Same as w4_end in this indexing

            # Rule 1: Wave 2 never retraces more than 100% of Wave 1
            wave_1_size = abs(w1_end - w1_start)
            wave_2_retracement = abs(w2_end - w1_end)
            if wave_2_retracement >= wave_1_size:
                return False

            # Rule 2: Wave 3 is never the shortest wave
            abs(w2_end - w1_end)
            wave_3_size = abs(w3_end - w2_end)
            wave_5_size = abs(w5_end - w4_end)

            if wave_3_size <= wave_1_size and wave_3_size <= wave_5_size:
                return False

            # Rule 3: Wave 4 does not overlap Wave 1 price territory
            if waves[0]["type"] == "low":  # Bullish impulse
                if w4_end <= w1_end:
                    return False
            else:  # Bearish impulse
                if w4_end >= w1_end:
                    return False

            return True

        except (IndexError, KeyError):
            return False

    def _validate_corrective_structure(self, waves: List[Dict]) -> bool:
        """Validate 3-wave corrective structure"""
        if len(waves) != 7:
            return False

        # Basic validation - alternating high/low pattern
        for i in range(len(waves) - 1):
            if waves[i]["type"] == waves[i + 1]["type"]:
                return False

        return True

    def _calculate_impulse_strength(self, waves: List[Dict]) -> float:
        """Calculate strength of impulse pattern based on Fibonacci ratios"""
        strength = 0.5  # Base strength

        try:
            # Check common Fibonacci relationships
            w1_size = abs(waves[2]["price"] - waves[0]["price"])
            w3_size = abs(waves[6]["price"] - waves[4]["price"])
            w5_size = abs(waves[8]["price"] - waves[6]["price"])

            # Wave 3 extension ratios
            w3_ratio = w3_size / w1_size if w1_size > 0 else 0
            for target_ratio in self.wave_ratios["wave_3_extension"]:
                if abs(w3_ratio - target_ratio) < 0.1:
                    strength += 0.2
                    break

            # Wave 5 ratios
            w5_ratio = w5_size / w1_size if w1_size > 0 else 0
            for target_ratio in self.wave_ratios["wave_5_extension"]:
                if abs(w5_ratio - target_ratio) < 0.1:
                    strength += 0.15
                    break

            return min(strength, 1.0)

        except (IndexError, ZeroDivisionError):
            return 0.5

    def _calculate_corrective_strength(self, waves: List[Dict]) -> float:
        """Calculate strength of corrective pattern"""
        strength = 0.4  # Base strength for corrections

        try:
            # Check if Wave C equals Wave A (common in corrections)
            wave_a_size = abs(waves[2]["price"] - waves[0]["price"])
            wave_c_size = abs(waves[6]["price"] - waves[4]["price"])

            if wave_a_size > 0:
                c_to_a_ratio = wave_c_size / wave_a_size
                if abs(c_to_a_ratio - 1.0) < 0.1:  # C = A
                    strength += 0.3
                elif abs(c_to_a_ratio - 1.618) < 0.1:  # C = 1.618 * A
                    strength += 0.2

            return min(strength, 1.0)

        except (IndexError, ZeroDivisionError):
            return 0.4

    def _check_wave_ratios(self, waves: List[Dict]) -> Dict[str, Any]:
        """Check Fibonacci ratios between waves"""
        ratios = {}

        try:
            w1_size = abs(waves[2]["price"] - waves[0]["price"])
            w3_size = abs(waves[6]["price"] - waves[4]["price"])
            w5_size = abs(waves[8]["price"] - waves[6]["price"])

            if w1_size > 0:
                ratios["wave_3_to_1"] = round(w3_size / w1_size, 3)
                ratios["wave_5_to_1"] = round(w5_size / w1_size, 3)

            return ratios

        except (IndexError, ZeroDivisionError):
            return {}

    def _check_corrective_ratios(self, waves: List[Dict]) -> Dict[str, Any]:
        """Check Fibonacci ratios in corrective patterns"""
        ratios = {}

        try:
            wave_a_size = abs(waves[2]["price"] - waves[0]["price"])
            wave_c_size = abs(waves[6]["price"] - waves[4]["price"])

            if wave_a_size > 0:
                ratios["wave_c_to_a"] = round(wave_c_size / wave_a_size, 3)

                # Check common corrective ratios
                if abs(ratios["wave_c_to_a"] - 1.0) < 0.05:
                    ratios["pattern"] = "C = A (Equal waves)"
                elif abs(ratios["wave_c_to_a"] - 1.618) < 0.05:
                    ratios["pattern"] = "C = 1.618 * A (Extended C wave)"

            return ratios

        except (IndexError, ZeroDivisionError):
            return {}

    def _describe_wave_structure(self, waves: List[Dict]) -> str:
        """Provide human-readable wave structure description"""
        if len(waves) == 9:
            direction = "Up" if waves[-1]["price"] > waves[0]["price"] else "Down"
            return f"5-wave impulse moving {direction.lower()}"
        elif len(waves) == 7:
            return "3-wave corrective pattern"
        else:
            return "Incomplete wave structure"

    def _identify_correction_type(self, waves: List[Dict]) -> str:
        """Identify the type of corrective pattern"""
        try:
            # Simple classification based on wave relationships
            wave_b_retracement = self._calculate_retracement_level(
                waves[:5]
            )  # A-B portion

            if wave_b_retracement > 0.618:
                return "Deep correction"
            elif wave_b_retracement > 0.382:
                return "Standard correction"
            else:
                return "Shallow correction"

        except Exception:
            return "Unknown correction"

    def _calculate_retracement_level(self, waves: List[Dict]) -> float:
        """Calculate retracement level as percentage"""
        try:
            if len(waves) >= 5:
                wave_a_size = abs(waves[2]["price"] - waves[0]["price"])
                wave_b_size = abs(waves[4]["price"] - waves[2]["price"])

                if wave_a_size > 0:
                    return wave_b_size / wave_a_size

            return 0.0

        except (IndexError, ZeroDivisionError):
            return 0.0

    def _get_current_wave_count(self, swings: List[Dict]) -> int:
        """Estimate current wave count"""
        return len(swings)

    def _determine_wave_position(self, swings: List[Dict], current_price: float) -> str:
        """Determine probable current wave position"""
        if len(swings) < 3:
            return "Insufficient data"

        last_swing = swings[-1]["price"]
        trend_direction = "up" if current_price > last_swing else "down"

        wave_count = len(swings)
        if wave_count % 5 == 0:
            return f"Possibly completing Wave 5 ({trend_direction})"
        elif wave_count % 3 == 0:
            return f"Possibly in corrective Wave C ({trend_direction})"
        else:
            return f"Possibly in Wave {wave_count % 5 or 5} ({trend_direction})"

    def _is_trend_continuing(self, waves: List[Dict], current_price: float) -> bool:
        """Check if trend is likely to continue"""
        if len(waves) < 2:
            return False

        overall_direction = waves[-1]["price"] > waves[0]["price"]
        current_direction = current_price > waves[-1]["price"]

        return overall_direction == current_direction

    def _estimate_completion(self, waves: List[Dict], current_price: float) -> float:
        """Estimate wave completion percentage"""
        try:
            if len(waves) >= 5:
                total_move = abs(waves[-1]["price"] - waves[0]["price"])
                current_move = abs(current_price - waves[0]["price"])

                if total_move > 0:
                    return min(current_move / total_move, 1.0)

            return 0.5  # Default to 50% if can't calculate

        except (IndexError, ZeroDivisionError):
            return 0.5
