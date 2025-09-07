"""
Consolidated Strategy Agent System
Eliminates redundancy across strategy agents while maintaining modularity
Uses plugin pattern for different trading strategies
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from langchain.agents import Tool
from langchain.schema import HumanMessage

from app.rag.agents.base_agent import BaseTradingAgent, TradingSignal
from app.services.market_research import (
    AnalysisComplexity,
    IntegratedAnalysis,
    MarketResearchService,
    get_market_research_service,
)

logger = logging.getLogger(__name__)


# ============================================================================
# STRATEGY PLUGIN INTERFACE
# ============================================================================


class StrategyPlugin(ABC):
    """Base class for strategy plugins"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__.replace("Strategy", "").lower()

    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """Return LangChain tools for this strategy"""
        pass

    @abstractmethod
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform strategy-specific market analysis"""
        pass

    @abstractmethod
    def generate_signal(
        self, analysis: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate trading signal from analysis"""
        pass

    def extract_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract strategy-specific features for pattern matching"""
        return {"strategy": self.name, "timestamp": datetime.now().isoformat()}


# ============================================================================
# MARKOV STRATEGY PLUGIN
# ============================================================================


class MarkovStrategy(StrategyPlugin):
    """Markov chain analysis strategy plugin"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.lookback_periods = config.get("lookback_periods", [5, 10, 20, 50])
        self.n_states = config.get("n_states", 5)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="calculate_markov_states",
                func=self._calculate_markov_states,
                description="Calculate current Markov chain states from price data",
            ),
            Tool(
                name="predict_markov_transition",
                func=self._predict_next_state,
                description="Predict the most likely next market state",
            ),
            Tool(
                name="analyze_markov_patterns",
                func=self._analyze_patterns,
                description="Find similar Markov state patterns in history",
            ),
        ]

    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Markov analysis"""
        prices = market_data.get("prices", [])

        if len(prices) < 20:
            return {"error": "insufficient_data", "states": [], "transitions": []}

        # Calculate returns and discretize into states
        returns = np.diff(np.log(prices))
        quantiles = np.linspace(0, 1, self.n_states + 1)
        state_bounds = np.quantile(returns, quantiles)
        states = np.digitize(returns, state_bounds) - 1
        states = np.clip(states, 0, self.n_states - 1)

        # Build transition matrix
        transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(len(states) - 1):
            transition_matrix[states[i], states[i + 1]] += 1

        # Normalize
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = np.divide(
            transition_matrix,
            row_sums[:, np.newaxis],
            out=np.zeros_like(transition_matrix),
            where=row_sums[:, np.newaxis] != 0,
        )

        current_state = states[-1] if len(states) > 0 else 0
        next_state_probs = (
            transition_matrix[current_state]
            if len(transition_matrix) > current_state
            else np.zeros(self.n_states)
        )

        return {
            "current_state": int(current_state),
            "transition_matrix": transition_matrix.tolist(),
            "next_state_probabilities": next_state_probs.tolist(),
            "confidence": float(np.max(next_state_probs)),
            "states_sequence": states.tolist(),
        }

    def generate_signal(
        self, analysis: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate Markov-based trading signal"""
        if "error" in analysis:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": "Insufficient data for Markov analysis",
            }

        confidence = analysis.get("confidence", 0)
        current_state = analysis.get("current_state", 2)  # Default to neutral

        if confidence < self.confidence_threshold:
            return {
                "action": "HOLD",
                "confidence": confidence,
                "reasoning": f"Low transition confidence: {confidence:.2f}",
            }

        # State-based signal logic (0-4 scale, 2 is neutral)
        if current_state >= 3:
            action = "BUY"
        elif current_state <= 1:
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": f"Markov state {current_state} with {confidence:.1%} confidence",
        }

    def _calculate_markov_states(self, price_data: str) -> str:
        try:
            prices = [float(x.strip()) for x in price_data.split(",")]
            analysis = self.analyze_market({"prices": prices})
            return f"Current state: {analysis.get('current_state')}, Confidence: {analysis.get('confidence', 0):.2f}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _predict_next_state(self, current_state: str) -> str:
        return f"Prediction for state {current_state} - use market analysis for full results"

    def _analyze_patterns(self, pattern: str) -> str:
        return f"Pattern analysis for: {pattern} - integrated with RAG system"


# ============================================================================
# WYCKOFF STRATEGY PLUGIN
# ============================================================================


class WyckoffStrategy(StrategyPlugin):
    """Wyckoff method analysis strategy plugin"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.volume_lookback = config.get("volume_lookback", 20)
        self.phase_threshold = config.get("phase_threshold", 0.7)

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="analyze_wyckoff_phase",
                func=self._analyze_phase,
                description="Identify current Wyckoff market cycle phase",
            ),
            Tool(
                name="calculate_effort_vs_result",
                func=self._effort_vs_result,
                description="Analyze volume (effort) vs price movement (result)",
            ),
            Tool(
                name="detect_supply_demand",
                func=self._supply_demand,
                description="Identify supply and demand zones",
            ),
        ]

    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Wyckoff analysis"""
        prices = market_data.get("prices", [])
        volumes = market_data.get("volumes", [])

        if len(prices) < self.volume_lookback or len(volumes) < self.volume_lookback:
            return {"error": "insufficient_data"}

        # Calculate effort vs result
        price_changes = np.diff(prices)
        volume_avg = np.mean(volumes[-self.volume_lookback :])

        # Simple phase detection
        recent_volumes = volumes[-10:]
        recent_prices = prices[-10:]

        volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]

        # Determine phase
        if volume_trend > 0 and abs(price_trend) < np.std(price_changes):
            phase = "accumulation"
        elif volume_trend > 0 and price_trend > 0:
            phase = "markup"
        elif volume_trend > 0 and price_trend < 0:
            phase = "distribution"
        elif price_trend < 0:
            phase = "markdown"
        else:
            phase = "unknown"

        return {
            "phase": phase,
            "volume_trend": volume_trend,
            "price_trend": price_trend,
            "effort_result_ratio": volume_trend / (abs(price_trend) + 1e-6),
            "confidence": min(abs(volume_trend) / volume_avg, 1.0),
        }

    def generate_signal(
        self, analysis: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate Wyckoff-based trading signal"""
        if "error" in analysis:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": "Insufficient data",
            }

        phase = analysis.get("phase", "unknown")
        confidence = analysis.get("confidence", 0)

        if phase == "accumulation" and confidence > self.phase_threshold:
            return {
                "action": "BUY",
                "confidence": confidence,
                "reasoning": f"Wyckoff accumulation phase detected",
            }
        elif phase == "distribution" and confidence > self.phase_threshold:
            return {
                "action": "SELL",
                "confidence": confidence,
                "reasoning": f"Wyckoff distribution phase detected",
            }
        else:
            return {
                "action": "HOLD",
                "confidence": confidence,
                "reasoning": f"Wyckoff phase: {phase} (confidence: {confidence:.1%})",
            }

    def _analyze_phase(self, data: str) -> str:
        return f"Phase analysis: {data}"

    def _effort_vs_result(self, data: str) -> str:
        return f"Effort vs result: {data}"

    def _supply_demand(self, data: str) -> str:
        return f"Supply/demand analysis: {data}"


# ============================================================================
# FIBONACCI STRATEGY PLUGIN
# ============================================================================


class FibonacciStrategy(StrategyPlugin):
    """Fibonacci retracement and extension strategy plugin"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.retracement_ratios = config.get(
            "retracement_ratios", [0.236, 0.382, 0.5, 0.618, 0.786]
        )
        self.golden_zone = config.get("golden_zone", [0.618, 0.786])
        self.proximity_threshold = config.get("proximity_threshold", 0.005)

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="calculate_fibonacci_levels",
                func=self._calculate_levels,
                description="Calculate Fibonacci retracement levels",
            ),
            Tool(
                name="identify_golden_zone",
                func=self._golden_zone_analysis,
                description="Check if price is in Fibonacci golden zone",
            ),
        ]

    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Fibonacci analysis"""
        prices = market_data.get("prices", [])

        if len(prices) < 20:
            return {"error": "insufficient_data"}

        # Find swing high and low
        recent_prices = np.array(prices[-50:])  # Last 50 periods
        swing_high = np.max(recent_prices)
        swing_low = np.min(recent_prices)
        current_price = prices[-1]

        # Calculate Fibonacci levels
        price_range = swing_high - swing_low
        fib_levels = {}

        for ratio in self.retracement_ratios:
            fib_levels[f"fib_{ratio}"] = swing_high - (price_range * ratio)

        # Check golden zone
        golden_low = swing_high - (price_range * self.golden_zone[1])  # 78.6%
        golden_high = swing_high - (price_range * self.golden_zone[0])  # 61.8%

        in_golden_zone = golden_low <= current_price <= golden_high

        # Find nearest level
        nearest_level = min(fib_levels.values(), key=lambda x: abs(x - current_price))
        distance_to_nearest = abs(current_price - nearest_level) / current_price

        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "current_price": current_price,
            "fib_levels": fib_levels,
            "in_golden_zone": in_golden_zone,
            "golden_zone_bounds": [golden_low, golden_high],
            "nearest_level": nearest_level,
            "distance_to_nearest": distance_to_nearest,
            "at_key_level": distance_to_nearest < self.proximity_threshold,
        }

    def generate_signal(
        self, analysis: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate Fibonacci-based trading signal"""
        if "error" in analysis:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": "Insufficient data",
            }

        in_golden_zone = analysis.get("in_golden_zone", False)
        at_key_level = analysis.get("at_key_level", False)
        distance = analysis.get("distance_to_nearest", 1.0)

        confidence = max(
            0.0, 1.0 - (distance * 100)
        )  # Higher confidence when closer to levels

        if in_golden_zone and at_key_level:
            return {
                "action": "BUY",
                "confidence": min(confidence + 0.2, 1.0),  # Golden zone bonus
                "reasoning": "Price in Fibonacci golden zone at key support level",
            }
        elif at_key_level:
            return {
                "action": "BUY",
                "confidence": confidence,
                "reasoning": f"Price near Fibonacci support ({distance*100:.1f}% away)",
            }
        else:
            return {
                "action": "HOLD",
                "confidence": confidence,
                "reasoning": f"No significant Fibonacci level nearby",
            }

    def _calculate_levels(self, data: str) -> str:
        return f"Fibonacci levels: {data}"

    def _golden_zone_analysis(self, data: str) -> str:
        return f"Golden zone: {data}"


# ============================================================================
# CONSOLIDATED STRATEGY AGENT
# ============================================================================


class ConsolidatedStrategyAgent(BaseTradingAgent):
    """
    Consolidated agent that can run multiple strategies through plugins
    Eliminates code duplication while maintaining strategy modularity
    """

    # Strategy registry
    AVAILABLE_STRATEGIES = {
        "markov": MarkovStrategy,
        "wyckoff": WyckoffStrategy,
        "fibonacci": FibonacciStrategy,
        # Can easily add more: 'elliott_wave': ElliottWaveStrategy, etc.
    }

    def __init__(
        self,
        strategies: List[str] = None,
        strategy_configs: Dict[str, Dict] = None,
        **kwargs,
    ):

        # Determine agent name based on strategies
        if strategies:
            agent_name = f"multi_strategy_{'+'.join(strategies)}"
            strategy_type = f"consolidated_{'+'.join(strategies)}"
        else:
            agent_name = "consolidated_strategy_agent"
            strategy_type = "multi_strategy"

        super().__init__(agent_name=agent_name, strategy_type=strategy_type, **kwargs)

        # Initialize strategy plugins
        self.strategies = {}
        self.strategy_configs = strategy_configs or {}

        # Initialize market research service
        self.market_research_service: Optional[MarketResearchService] = None
        self.use_market_research = kwargs.get("use_market_research", True)
        self.analysis_complexity = kwargs.get(
            "analysis_complexity", AnalysisComplexity.INTERMEDIATE
        )

        # Default to all strategies if none specified
        strategies_to_load = strategies or list(self.AVAILABLE_STRATEGIES.keys())

        for strategy_name in strategies_to_load:
            if strategy_name in self.AVAILABLE_STRATEGIES:
                config = self.strategy_configs.get(strategy_name, {})
                self.strategies[strategy_name] = self.AVAILABLE_STRATEGIES[
                    strategy_name
                ](config)
                logger.info(f"Loaded strategy plugin: {strategy_name}")

        self.consensus_method = kwargs.get("consensus_method", "weighted_average")

        logger.info(
            f"ConsolidatedStrategyAgent initialized with {len(self.strategies)} strategies and market research {'enabled' if self.use_market_research else 'disabled'}"
        )

    async def _create_tools(self) -> List[Tool]:
        """Create tools from all loaded strategy plugins"""
        all_tools = []

        for strategy_name, strategy in self.strategies.items():
            strategy_tools = strategy.get_tools()
            # Prefix tool names with strategy to avoid conflicts
            for tool in strategy_tools:
                tool.name = f"{strategy_name}_{tool.name}"
                tool.description = f"[{strategy_name.upper()}] {tool.description}"
            all_tools.extend(strategy_tools)

        return all_tools

    async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
        """
        Multi-strategy market analysis with consensus building and market research integration
        """
        try:
            symbol = market_data.get("symbol", "UNKNOWN")
            current_price = market_data.get("current_price", 0.0)

            # Initialize market research service if needed
            market_research_result: Optional[IntegratedAnalysis] = None
            if self.use_market_research:
                try:
                    if not self.market_research_service:
                        self.market_research_service = (
                            await get_market_research_service()
                        )

                    # Perform integrated market research analysis
                    market_research_result = (
                        await self.market_research_service.integrated_strategy_analysis(
                            symbol=symbol,
                            analysis_complexity=self.analysis_complexity,
                            include_sentiment=True,
                            include_technical=True,
                            include_complex_analysis=True,
                        )
                    )
                    logger.info(
                        f"Market research analysis completed for {symbol}",
                        sentiment=(
                            market_research_result.market_sentiment.sentiment.name
                            if market_research_result.market_sentiment
                            else None
                        ),
                        confidence=market_research_result.confidence_score,
                        recommendation=market_research_result.trading_recommendation.get(
                            "action"
                        ),
                    )

                except Exception as e:
                    logger.warning(
                        f"Market research analysis failed for {symbol}", error=str(e)
                    )

            # Run all strategy analyses
            strategy_results = {}
            strategy_signals = {}

            for strategy_name, strategy in self.strategies.items():
                try:
                    analysis = strategy.analyze_market(market_data)
                    signal_data = strategy.generate_signal(analysis, market_data)

                    strategy_results[strategy_name] = analysis
                    strategy_signals[strategy_name] = signal_data

                except Exception as e:
                    logger.warning(
                        f"Strategy {strategy_name} analysis failed", error=str(e)
                    )
                    strategy_signals[strategy_name] = {
                        "action": "HOLD",
                        "confidence": 0.0,
                        "reasoning": f"Analysis error: {str(e)}",
                    }

            # Integrate market research signals with strategy signals
            if market_research_result and market_research_result.trading_recommendation:
                market_research_signal = self._convert_market_research_to_signal(
                    market_research_result
                )
                strategy_signals["market_research"] = market_research_signal
                logger.info(
                    f"Added market research signal to consensus",
                    action=market_research_signal["action"],
                    confidence=market_research_signal["confidence"],
                )

            # Build consensus including market research
            consensus = self._build_consensus(strategy_signals)

            # Find similar patterns using combined features
            features = self._extract_combined_features(strategy_results, market_data)
            similar_patterns = await self.find_similar_patterns(features)
            pattern_context = await self.get_pattern_context(features)

            # Create final signal
            final_signal = TradingSignal(
                agent_type=self.agent_name,
                strategy_name=self.strategy_type,
                symbol=symbol,
                action=consensus["action"],
                confidence=consensus["confidence"],
                reasoning=consensus["reasoning"],
                entry_price=current_price if consensus["action"] != "HOLD" else None,
                metadata={
                    "strategy_results": strategy_results,
                    "strategy_signals": strategy_signals,
                    "consensus_method": self.consensus_method,
                    "similar_patterns_count": len(similar_patterns),
                    "pattern_context_available": bool(pattern_context),
                },
            )

            return final_signal

        except Exception as e:
            logger.error(f"Multi-strategy analysis failed", error=str(e))
            return TradingSignal(
                agent_type=self.agent_name,
                strategy_name=self.strategy_type,
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}",
            )

    def _convert_market_research_to_signal(
        self, market_research: "IntegratedAnalysis"
    ) -> Dict[str, Any]:
        """Convert market research results to strategy signal format"""
        try:
            # Extract sentiment and confidence from market research
            sentiment_score = 0.0
            confidence = 0.5

            # Process sentiment analysis if available
            if hasattr(market_research, "sentiment") and market_research.sentiment:
                sentiment_mapping = {
                    "bullish": 0.7,
                    "very_bullish": 0.9,
                    "bearish": -0.7,
                    "very_bearish": -0.9,
                    "neutral": 0.0,
                }
                sentiment_score = sentiment_mapping.get(
                    market_research.sentiment.overall_sentiment.lower(), 0.0
                )
                confidence = min(
                    market_research.sentiment.confidence_score / 100.0, 1.0
                )

            # Process complex analysis if available
            if (
                hasattr(market_research, "complex_analysis")
                and market_research.complex_analysis
            ):
                # Adjust signal strength based on analysis insights
                if market_research.complex_analysis.key_insights:
                    insight_keywords = " ".join(
                        market_research.complex_analysis.key_insights
                    ).lower()

                    # Boost confidence for strong technical signals
                    if any(
                        keyword in insight_keywords
                        for keyword in [
                            "breakout",
                            "momentum",
                            "trend",
                            "support",
                            "resistance",
                        ]
                    ):
                        confidence = min(confidence + 0.1, 1.0)

                    # Adjust sentiment based on risk factors
                    if any(
                        keyword in insight_keywords
                        for keyword in ["risk", "volatile", "uncertainty", "concern"]
                    ):
                        sentiment_score *= 0.8
                        confidence = max(confidence - 0.1, 0.1)

            # Convert to trading signal format
            signal_strength = abs(sentiment_score) * confidence
            action = (
                "buy"
                if sentiment_score > 0.1
                else "sell" if sentiment_score < -0.1 else "hold"
            )

            return {
                "action": action,
                "confidence": confidence,
                "signal_strength": signal_strength,
                "sentiment_score": sentiment_score,
                "source": "market_research",
                "reasoning": f"Market research indicates {action} signal with {confidence:.1%} confidence",
                "metadata": {
                    "sentiment": (
                        getattr(
                            market_research.sentiment, "overall_sentiment", "neutral"
                        )
                        if hasattr(market_research, "sentiment")
                        else "neutral"
                    ),
                    "analysis_type": (
                        getattr(
                            market_research.complex_analysis, "analysis_type", "general"
                        )
                        if hasattr(market_research, "complex_analysis")
                        else "general"
                    ),
                },
            }

        except Exception as e:
            self.logger.warning(f"Error converting market research to signal: {e}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "signal_strength": 0.0,
                "sentiment_score": 0.0,
                "source": "market_research",
                "reasoning": "Market research conversion failed",
                "metadata": {"error": str(e)},
            }

    def _build_consensus(self, strategy_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Build consensus from multiple strategy signals"""
        if not strategy_signals:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": "No strategy signals available",
            }

        actions = []
        confidences = []
        reasonings = []

        for strategy_name, signal in strategy_signals.items():
            actions.append(signal.get("action", "HOLD"))
            confidences.append(signal.get("confidence", 0.0))
            reasonings.append(
                f"{strategy_name}: {signal.get('reasoning', 'No reason')}"
            )

        if self.consensus_method == "weighted_average":
            # Weight by confidence
            total_weight = sum(confidences)
            if total_weight == 0:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "reasoning": "All strategies have zero confidence",
                }

            buy_weight = sum(
                conf for action, conf in zip(actions, confidences) if action == "BUY"
            )
            sell_weight = sum(
                conf for action, conf in zip(actions, confidences) if action == "SELL"
            )
            hold_weight = sum(
                conf for action, conf in zip(actions, confidences) if action == "HOLD"
            )

            if buy_weight > sell_weight and buy_weight > hold_weight:
                final_action = "BUY"
                final_confidence = buy_weight / total_weight
            elif sell_weight > hold_weight:
                final_action = "SELL"
                final_confidence = sell_weight / total_weight
            else:
                final_action = "HOLD"
                final_confidence = hold_weight / total_weight

        elif self.consensus_method == "majority_vote":
            # Simple majority vote
            action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
            for action in actions:
                action_counts[action] += 1

            final_action = max(action_counts, key=action_counts.get)
            final_confidence = np.mean(confidences)

        else:  # Default to average
            final_action = "HOLD"
            final_confidence = np.mean(confidences)

        return {
            "action": final_action,
            "confidence": final_confidence,
            "reasoning": f"Consensus ({self.consensus_method}): "
            + " | ".join(reasonings),
        }

    def _extract_combined_features(
        self, strategy_results: Dict, market_data: Dict
    ) -> Dict[str, Any]:
        """Extract features from all strategies for pattern matching"""
        features = {
            "symbol": market_data.get("symbol", "UNKNOWN"),
            "timestamp": datetime.now().isoformat(),
            "strategies_used": list(strategy_results.keys()),
        }

        # Add strategy-specific features
        for strategy_name, result in strategy_results.items():
            if "error" not in result:
                features[f"{strategy_name}_features"] = result

        return features


# ============================================================================
# BACKWARDS COMPATIBILITY AND CONVENIENCE FUNCTIONS
# ============================================================================


def create_markov_agent(**kwargs) -> ConsolidatedStrategyAgent:
    """Create agent with only Markov strategy"""
    return ConsolidatedStrategyAgent(strategies=["markov"], **kwargs)


def create_wyckoff_agent(**kwargs) -> ConsolidatedStrategyAgent:
    """Create agent with only Wyckoff strategy"""
    return ConsolidatedStrategyAgent(strategies=["wyckoff"], **kwargs)


def create_fibonacci_agent(**kwargs) -> ConsolidatedStrategyAgent:
    """Create agent with only Fibonacci strategy"""
    return ConsolidatedStrategyAgent(strategies=["fibonacci"], **kwargs)


def create_multi_strategy_agent(
    strategies: List[str] = None, **kwargs
) -> ConsolidatedStrategyAgent:
    """Create agent with multiple strategies"""
    return ConsolidatedStrategyAgent(strategies=strategies, **kwargs)


# Backwards compatibility aliases
MarkovTradingAgent = create_markov_agent
WyckoffAgent = create_wyckoff_agent
FibonacciAgent = create_fibonacci_agent

# Export main classes
__all__ = [
    "ConsolidatedStrategyAgent",
    "StrategyPlugin",
    "MarkovStrategy",
    "WyckoffStrategy",
    "FibonacciStrategy",
    "create_markov_agent",
    "create_wyckoff_agent",
    "create_fibonacci_agent",
    "create_multi_strategy_agent",
    "MarkovTradingAgent",
    "WyckoffAgent",
    "FibonacciAgent",  # Backwards compatibility
]
