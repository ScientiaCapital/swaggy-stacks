"""
Integrated Alpha Pattern Detection System
Combines new alpha patterns with existing Markov, Elliott Wave, Fibonacci, Golden Zone, and Wyckoff analysis
for split-second trading decisions
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from app.analysis.enhanced_markov_system import MarkovChainAnalyzer

logger = logging.getLogger(__name__)


class AlphaConfidenceLevel(Enum):
    VERY_HIGH = "very_high"  # 90%+ confidence
    HIGH = "high"  # 75-89% confidence
    MEDIUM = "medium"  # 60-74% confidence
    LOW = "low"  # 45-59% confidence
    VERY_LOW = "very_low"  # <45% confidence


@dataclass
class IntegratedAlphaSignal:
    """Unified alpha signal combining all methodologies"""

    symbol: str
    timestamp: datetime
    signal_strength: float  # -1 to 1
    confidence_level: AlphaConfidenceLevel
    direction: str  # "long", "short", "neutral"

    # Pattern contributions
    momentum_score: float
    volatility_score: float
    cross_asset_score: float
    microstructure_score: float
    options_flow_score: float
    regime_score: float

    # Technical analysis contributions
    markov_state: str
    markov_probability: float
    elliott_wave_position: str
    fibonacci_level: float
    golden_zone_signal: bool
    wyckoff_phase: str

    # Composite scores
    technical_confluence: float
    alpha_potential: float
    risk_reward_ratio: float
    time_horizon: str  # "scalp", "intraday", "swing", "position"

    # Metadata
    detected_patterns: List[str]
    supporting_indicators: List[str]
    risk_factors: List[str]


class IntegratedAlphaDetector:
    """
    Real-time alpha pattern detector integrating all methodologies
    Designed for split-second decision making
    """

    def __init__(self):
        self.markov_analyzer = MarkovChainAnalyzer()

        # Confidence thresholds for each methodology
        self.confidence_weights = {
            "markov_chain": 0.25,
            "elliott_wave": 0.20,
            "fibonacci": 0.15,
            "golden_zone": 0.10,
            "wyckoff": 0.20,
            "alpha_patterns": 0.10,
        }

        # Pattern importance weights (higher = more reliable)
        self.pattern_weights = {
            "momentum_alpha": 0.20,
            "volatility_alpha": 0.18,
            "microstructure_alpha": 0.17,
            "regime_alpha": 0.15,
            "options_flow_alpha": 0.15,
            "cross_asset_alpha": 0.15,
        }

        logger.info("🎯 Integrated Alpha Detector initialized")

    async def detect_alpha_opportunity(
        self, symbol: str, market_data: Dict[str, Any], real_time_data: Dict[str, Any]
    ) -> Optional[IntegratedAlphaSignal]:
        """
        Main detection method - analyzes all methodologies in parallel for split-second decisions
        """
        start_time = datetime.utcnow()

        # Parallel analysis of all methodologies
        analysis_tasks = [
            self._analyze_markov_chain(symbol, market_data),
            self._analyze_elliott_wave(market_data),
            self._analyze_fibonacci_levels(market_data),
            self._analyze_golden_zone(market_data),
            self._analyze_wyckoff_method(market_data),
            self._analyze_alpha_patterns(symbol, market_data, real_time_data),
        ]

        try:
            # Execute all analyses in parallel
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            (
                markov_result,
                elliott_result,
                fib_result,
                golden_result,
                wyckoff_result,
                alpha_result,
            ) = results

            # Handle any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Analysis {i} failed: {result}")
                    results[i] = self._get_default_result(i)

            # Integrate all results into unified signal
            integrated_signal = self._integrate_analysis_results(
                symbol,
                markov_result,
                elliott_result,
                fib_result,
                golden_result,
                wyckoff_result,
                alpha_result,
            )

            # Performance logging
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.debug(f"⚡ Alpha detection completed in {processing_time:.1f}ms")

            return integrated_signal

        except Exception as e:
            logger.error(f"❌ Alpha detection failed for {symbol}: {e}")
            return None

    async def _analyze_markov_chain(
        self, symbol: str, market_data: Dict
    ) -> Dict[str, Any]:
        """Analyze current Markov state and transition probabilities"""
        try:
            # Extract price series for Markov analysis
            prices = market_data.get("prices", [])
            if not prices or len(prices) < 20:
                return {"state": "unknown", "probability": 0.0, "signal": 0.0}

            # Get current Markov state
            current_state = self.markov_analyzer.get_current_state(prices)
            transition_prob = self.markov_analyzer.get_transition_probability(prices)

            # Convert to signal strength
            signal_strength = self._markov_to_signal(current_state, transition_prob)

            return {
                "state": current_state,
                "probability": transition_prob,
                "signal": signal_strength,
                "regime": self.markov_analyzer.detect_regime(prices),
            }

        except Exception as e:
            logger.warning(f"Markov analysis failed: {e}")
            return {"state": "neutral", "probability": 0.0, "signal": 0.0}

    async def _analyze_elliott_wave(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze Elliott Wave patterns"""
        try:
            prices = market_data.get("prices", [])
            if len(prices) < 50:
                return {"position": "unknown", "signal": 0.0, "wave_count": 0}

            # Simplified Elliott Wave analysis
            # In production, this would be a sophisticated wave counting algorithm
            wave_position = self._identify_elliott_wave_position(prices)
            signal = self._elliott_wave_to_signal(wave_position)

            return {
                "position": wave_position,
                "signal": signal,
                "wave_count": self._count_waves(prices),
                "impulse_correction": self._identify_impulse_or_correction(prices),
            }

        except Exception as e:
            logger.warning(f"Elliott Wave analysis failed: {e}")
            return {"position": "unknown", "signal": 0.0}

    async def _analyze_fibonacci_levels(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze Fibonacci retracement and extension levels"""
        try:
            highs = market_data.get("highs", [])
            lows = market_data.get("lows", [])
            current_price = market_data.get("current_price", 0)

            if not highs or not lows or not current_price:
                return {"level": 0.0, "signal": 0.0, "support_resistance": "none"}

            # Calculate Fibonacci levels
            swing_high = max(highs[-20:])  # Last 20 periods
            swing_low = min(lows[-20:])

            fib_levels = self._calculate_fibonacci_levels(swing_high, swing_low)
            current_fib_level = self._get_current_fibonacci_level(
                current_price, fib_levels
            )

            # Generate signal based on Fibonacci level
            signal = self._fibonacci_to_signal(
                current_fib_level, current_price, fib_levels
            )

            return {
                "level": current_fib_level,
                "signal": signal,
                "support_resistance": self._identify_fib_support_resistance(
                    current_price, fib_levels
                ),
                "key_levels": fib_levels,
            }

        except Exception as e:
            logger.warning(f"Fibonacci analysis failed: {e}")
            return {"level": 0.0, "signal": 0.0}

    async def _analyze_golden_zone(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze Golden Zone (61.8-78.6% Fibonacci levels)"""
        try:
            current_price = market_data.get("current_price", 0)
            highs = market_data.get("highs", [])
            lows = market_data.get("lows", [])

            if not current_price or not highs or not lows:
                return {"in_zone": False, "signal": 0.0}

            # Calculate Golden Zone
            swing_high = max(highs[-20:])
            swing_low = min(lows[-20:])

            golden_zone_low = swing_low + (swing_high - swing_low) * 0.618
            golden_zone_high = swing_low + (swing_high - swing_low) * 0.786

            in_golden_zone = golden_zone_low <= current_price <= golden_zone_high

            # Generate signal if in Golden Zone
            signal = 0.0
            if in_golden_zone:
                # Determine if it's a buying or selling opportunity
                zone_position = (current_price - golden_zone_low) / (
                    golden_zone_high - golden_zone_low
                )
                signal = (
                    0.8 if zone_position < 0.5 else -0.8
                )  # Buy low in zone, sell high in zone

            return {
                "in_zone": in_golden_zone,
                "signal": signal,
                "zone_position": zone_position if in_golden_zone else 0.0,
                "zone_range": (golden_zone_low, golden_zone_high),
            }

        except Exception as e:
            logger.warning(f"Golden Zone analysis failed: {e}")
            return {"in_zone": False, "signal": 0.0}

    async def _analyze_wyckoff_method(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze Wyckoff market phases"""
        try:
            prices = market_data.get("prices", [])
            volumes = market_data.get("volumes", [])

            if len(prices) < 30 or len(volumes) < 30:
                return {"phase": "unknown", "signal": 0.0}

            # Simplified Wyckoff analysis
            phase = self._identify_wyckoff_phase(prices, volumes)
            signal = self._wyckoff_to_signal(phase, prices, volumes)

            return {
                "phase": phase,
                "signal": signal,
                "volume_analysis": self._analyze_volume_patterns(volumes),
                "price_action": self._analyze_price_action(prices),
            }

        except Exception as e:
            logger.warning(f"Wyckoff analysis failed: {e}")
            return {"phase": "unknown", "signal": 0.0}

    async def _analyze_alpha_patterns(
        self, symbol: str, market_data: Dict, real_time_data: Dict
    ) -> Dict[str, Any]:
        """Analyze new alpha patterns for signals"""
        try:
            pattern_scores = {}

            # Analyze each alpha pattern category
            for category in [
                "momentum_alpha",
                "volatility_alpha",
                "cross_asset_alpha",
                "microstructure_alpha",
                "options_flow_alpha",
                "regime_alpha",
            ]:

                score = await self._analyze_pattern_category(
                    category, market_data, real_time_data
                )
                pattern_scores[category] = score

            # Calculate weighted composite score
            composite_score = sum(
                score * self.pattern_weights.get(category, 0.1)
                for category, score in pattern_scores.items()
            )

            return {
                "pattern_scores": pattern_scores,
                "composite_score": composite_score,
                "detected_patterns": self._identify_active_patterns(pattern_scores),
                "signal": composite_score,
            }

        except Exception as e:
            logger.warning(f"Alpha pattern analysis failed: {e}")
            return {"composite_score": 0.0, "signal": 0.0}

    async def _analyze_pattern_category(
        self, category: str, market_data: Dict, real_time_data: Dict
    ) -> float:
        """Analyze specific alpha pattern category"""

        if category == "momentum_alpha":
            return self._analyze_momentum_patterns(market_data)
        elif category == "volatility_alpha":
            return self._analyze_volatility_patterns(market_data)
        elif category == "microstructure_alpha":
            return self._analyze_microstructure_patterns(real_time_data)
        elif category == "regime_alpha":
            return self._analyze_regime_patterns(market_data)
        elif category == "options_flow_alpha":
            return self._analyze_options_patterns(real_time_data)
        elif category == "cross_asset_alpha":
            return self._analyze_cross_asset_patterns(market_data)
        else:
            return 0.0

    def _analyze_momentum_patterns(self, market_data: Dict) -> float:
        """Analyze momentum-based alpha patterns"""
        prices = market_data.get("prices", [])
        if len(prices) < 20:
            return 0.0

        # Calculate momentum indicators
        recent_returns = np.diff(prices[-10:]) / prices[-11:-1]
        momentum_strength = np.mean(recent_returns)

        # Detect momentum exhaustion
        momentum_acceleration = np.diff(recent_returns)
        exhaustion_signal = (
            -0.8
            if len(momentum_acceleration) > 0 and momentum_acceleration[-1] < -0.002
            else 0.0
        )

        return max(-1.0, min(1.0, momentum_strength * 10 + exhaustion_signal))

    def _analyze_volatility_patterns(self, market_data: Dict) -> float:
        """Analyze volatility-based alpha patterns"""
        prices = market_data.get("prices", [])
        if len(prices) < 20:
            return 0.0

        # Calculate volatility metrics
        returns = np.diff(prices) / prices[:-1]
        current_vol = np.std(returns[-10:])  # 10-period volatility
        historical_vol = np.std(returns[-50:])  # 50-period volatility

        # Volatility expansion signal
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        expansion_signal = 0.7 if vol_ratio > 1.5 else 0.0

        return max(-1.0, min(1.0, expansion_signal))

    def _analyze_microstructure_patterns(self, real_time_data: Dict) -> float:
        """Analyze market microstructure patterns"""
        real_time_data.get("bid_ask_spread", 0)
        volume_imbalance = real_time_data.get("volume_imbalance", 0)

        # Order flow imbalance signal
        if volume_imbalance > 0.6:
            return 0.6  # Bullish imbalance
        elif volume_imbalance < -0.6:
            return -0.6  # Bearish imbalance
        else:
            return 0.0

    def _integrate_analysis_results(
        self,
        symbol: str,
        markov_result: Dict,
        elliott_result: Dict,
        fib_result: Dict,
        golden_result: Dict,
        wyckoff_result: Dict,
        alpha_result: Dict,
    ) -> IntegratedAlphaSignal:
        """Integrate all analysis results into unified alpha signal"""

        # Calculate weighted signal strength
        signal_components = {
            "markov": markov_result.get("signal", 0.0),
            "elliott": elliott_result.get("signal", 0.0),
            "fibonacci": fib_result.get("signal", 0.0),
            "golden": golden_result.get("signal", 0.0),
            "wyckoff": wyckoff_result.get("signal", 0.0),
            "alpha": alpha_result.get("signal", 0.0),
        }

        # Weighted composite signal
        composite_signal = sum(
            signal
            * self.confidence_weights.get(
                (
                    f"{method}_chain"
                    if method == "markov"
                    else f"{method}_wave" if method == "elliott" else method
                ),
                0.1,
            )
            for method, signal in signal_components.items()
        )

        # Determine confidence level
        signal_consensus = sum(1 for s in signal_components.values() if abs(s) > 0.3)
        if signal_consensus >= 5:
            confidence = AlphaConfidenceLevel.VERY_HIGH
        elif signal_consensus >= 4:
            confidence = AlphaConfidenceLevel.HIGH
        elif signal_consensus >= 3:
            confidence = AlphaConfidenceLevel.MEDIUM
        elif signal_consensus >= 2:
            confidence = AlphaConfidenceLevel.LOW
        else:
            confidence = AlphaConfidenceLevel.VERY_LOW

        # Determine direction
        if composite_signal > 0.2:
            direction = "long"
        elif composite_signal < -0.2:
            direction = "short"
        else:
            direction = "neutral"

        # Extract pattern scores
        pattern_scores = alpha_result.get("pattern_scores", {})

        return IntegratedAlphaSignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signal_strength=composite_signal,
            confidence_level=confidence,
            direction=direction,
            # Pattern contributions
            momentum_score=pattern_scores.get("momentum_alpha", 0.0),
            volatility_score=pattern_scores.get("volatility_alpha", 0.0),
            cross_asset_score=pattern_scores.get("cross_asset_alpha", 0.0),
            microstructure_score=pattern_scores.get("microstructure_alpha", 0.0),
            options_flow_score=pattern_scores.get("options_flow_alpha", 0.0),
            regime_score=pattern_scores.get("regime_alpha", 0.0),
            # Technical analysis contributions
            markov_state=markov_result.get("state", "unknown"),
            markov_probability=markov_result.get("probability", 0.0),
            elliott_wave_position=elliott_result.get("position", "unknown"),
            fibonacci_level=fib_result.get("level", 0.0),
            golden_zone_signal=golden_result.get("in_zone", False),
            wyckoff_phase=wyckoff_result.get("phase", "unknown"),
            # Composite scores
            technical_confluence=signal_consensus / 6.0,  # Normalize to 0-1
            alpha_potential=abs(composite_signal),
            risk_reward_ratio=self._calculate_risk_reward(composite_signal, confidence),
            time_horizon=self._determine_time_horizon(signal_components),
            # Metadata
            detected_patterns=alpha_result.get("detected_patterns", []),
            supporting_indicators=self._get_supporting_indicators(signal_components),
            risk_factors=self._identify_risk_factors(signal_components, confidence),
        )

    def _calculate_risk_reward(
        self, signal_strength: float, confidence: AlphaConfidenceLevel
    ) -> float:
        """Calculate risk-reward ratio based on signal strength and confidence"""
        confidence_multiplier = {
            AlphaConfidenceLevel.VERY_HIGH: 3.0,
            AlphaConfidenceLevel.HIGH: 2.5,
            AlphaConfidenceLevel.MEDIUM: 2.0,
            AlphaConfidenceLevel.LOW: 1.5,
            AlphaConfidenceLevel.VERY_LOW: 1.0,
        }.get(confidence, 1.0)

        return abs(signal_strength) * confidence_multiplier

    def _determine_time_horizon(self, signal_components: Dict[str, float]) -> str:
        """Determine optimal time horizon based on signal characteristics"""
        alpha_strength = abs(signal_components.get("alpha", 0.0))

        if alpha_strength > 0.8:
            return "scalp"  # Very strong short-term signal
        elif alpha_strength > 0.5:
            return "intraday"
        elif alpha_strength > 0.3:
            return "swing"
        else:
            return "position"

    # Helper methods for technical analysis (simplified implementations)
    def _markov_to_signal(self, state: str, probability: float) -> float:
        """Convert Markov state to signal strength"""
        state_signals = {
            "bullish_momentum": 0.8,
            "bearish_momentum": -0.8,
            "consolidation": 0.0,
            "reversal_up": 0.6,
            "reversal_down": -0.6,
            "neutral": 0.0,
        }
        return state_signals.get(state, 0.0) * probability

    def _elliott_wave_to_signal(self, wave_position: str) -> float:
        """Convert Elliott Wave position to signal strength"""
        wave_signals = {
            "wave_1": 0.6,  # Beginning of impulse
            "wave_2": -0.3,  # Correction
            "wave_3": 0.9,  # Strongest impulse
            "wave_4": -0.3,  # Correction
            "wave_5": 0.4,  # Final impulse (weakening)
            "wave_a": -0.6,  # Correction wave
            "wave_b": 0.3,  # Counter-trend
            "wave_c": -0.8,  # Final correction
        }
        return wave_signals.get(wave_position, 0.0)

    def _fibonacci_to_signal(
        self, fib_level: float, current_price: float, fib_levels: Dict
    ) -> float:
        """Convert Fibonacci level proximity to signal strength"""
        # Strong signals at key Fibonacci levels
        key_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618]

        for level in key_levels:
            if abs(fib_level - level) < 0.05:  # Within 5% of key level
                if level < 0.5:
                    return 0.6  # Bullish at lower retracements
                elif level > 0.618:
                    return -0.4  # Bearish at deeper retracements

        return 0.0

    def _wyckoff_to_signal(
        self, phase: str, prices: List[float], volumes: List[float]
    ) -> float:
        """Convert Wyckoff phase to signal strength"""
        phase_signals = {
            "accumulation": 0.7,
            "markup": 0.5,
            "distribution": -0.7,
            "markdown": -0.5,
            "re_accumulation": 0.4,
            "re_distribution": -0.4,
        }
        return phase_signals.get(phase, 0.0)

    # Placeholder methods for complex analysis (would be fully implemented in production)
    def _identify_elliott_wave_position(self, prices: List[float]) -> str:
        """Identify current Elliott Wave position (simplified)"""
        if len(prices) < 20:
            return "unknown"

        # Simplified wave identification
        recent_trend = prices[-1] - prices[-10]
        if recent_trend > 0:
            return "wave_3"  # Assume strong upward movement is wave 3
        else:
            return "wave_c"  # Assume downward movement is correction

    def _calculate_fibonacci_levels(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        return {
            "0%": high,
            "23.6%": high - diff * 0.236,
            "38.2%": high - diff * 0.382,
            "50%": high - diff * 0.5,
            "61.8%": high - diff * 0.618,
            "78.6%": high - diff * 0.786,
            "100%": low,
        }

    def _identify_wyckoff_phase(self, prices: List[float], volumes: List[float]) -> str:
        """Identify Wyckoff market phase (simplified)"""
        if len(prices) < 20 or len(volumes) < 20:
            return "unknown"

        price_trend = (prices[-1] - prices[-20]) / prices[-20]
        volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-20:])

        if price_trend > 0.05 and volume_trend > 1.2:
            return "markup"
        elif price_trend < -0.05 and volume_trend > 1.2:
            return "markdown"
        elif abs(price_trend) < 0.02 and volume_trend < 0.8:
            return "accumulation"
        else:
            return "distribution"


# Factory function for easy integration
def create_alpha_detector() -> IntegratedAlphaDetector:
    """Create integrated alpha detector instance"""
    return IntegratedAlphaDetector()
