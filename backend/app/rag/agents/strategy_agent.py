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
from app.ai.trading_advisor import AITradingAdvisor, AnalysisType

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


class StrategyAgent(BaseTradingAgent):
    """
    Multi-Strategy Trading Agent
    Consolidates Markov, Wyckoff, and Fibonacci analysis strategies
    """

    AVAILABLE_STRATEGIES = {
        "markov": "Markov Chain Analysis",
        "wyckoff": "Wyckoff Method Analysis", 
        "fibonacci": "Fibonacci Retracement Analysis"
    }

    def __init__(
        self, 
        strategies: List[str] = None,
        strategy_configs: Dict[str, Dict] = None,
        market_research_service=None,
        use_market_research: bool = False,
        analysis_complexity: str = "medium",
        ai_advisor=None,
        use_ai_advisor: bool = False,
        ai_analysis_types: List[str] = None,
        consensus_method: str = "weighted_average",
        **kwargs
    ):
        # Initialize strategy configurations
        self.strategies = strategies or ["markov"]
        self.strategy_configs = strategy_configs or {}
        
        # Market research integration
        self.market_research_service = market_research_service
        self.use_market_research = use_market_research
        self.analysis_complexity = analysis_complexity
        
        # AI advisor integration  
        self.ai_advisor = ai_advisor
        self.use_ai_advisor = use_ai_advisor
        self.ai_analysis_types = ai_analysis_types or [
            "technical_analysis", 
            "sentiment_analysis", 
            "fundamental_analysis"
        ]
        
        # Validate strategies
        invalid_strategies = [s for s in self.strategies if s not in self.AVAILABLE_STRATEGIES]
        if invalid_strategies:
            raise ValueError(f"Invalid strategies: {invalid_strategies}. Available: {list(self.AVAILABLE_STRATEGIES.keys())}")
        
        # Initialize analysis engines based on selected strategies
        self._initialize_strategy_engines()
        
        # Consensus building method
        self.consensus_method = consensus_method
        
        super().__init__(**kwargs)
        
        self.logger.info(
            f"StrategyAgent initialized with {len(self.strategies)} strategies, "
            f"market research: {self.use_market_research}, AI advisor: {self.use_ai_advisor}"
        )
    
    def _create_tools(self) -> List:
        """Create tools for the strategy agent"""
        tools = super()._create_tools()
        
        # Add strategy-specific tools if needed
        # This can be extended with custom strategy tools
        
        return tools

# ============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# ============================================================================

# Maintain backwards compatibility with existing imports
ConsolidatedStrategyAgent = StrategyAgent  # For existing code that imports ConsolidatedStrategyAgent


# ============================================================================
# BACKWARDS COMPATIBILITY AND CONVENIENCE FUNCTIONS
# ============================================================================


def create_markov_agent(**kwargs) -> StrategyAgent:
    """Create agent with only Markov strategy"""
    return StrategyAgent(strategies=["markov"], **kwargs)


def create_wyckoff_agent(**kwargs) -> StrategyAgent:
    """Create agent with only Wyckoff strategy"""
    return StrategyAgent(strategies=["wyckoff"], **kwargs)


def create_fibonacci_agent(**kwargs) -> StrategyAgent:
    """Create agent with only Fibonacci strategy"""
    return StrategyAgent(strategies=["fibonacci"], **kwargs)


def create_multi_strategy_agent(
    strategies: List[str] = None, **kwargs
) -> StrategyAgent:
    """Create agent with multiple strategies"""
    return StrategyAgent(strategies=strategies, **kwargs)


# Backwards compatibility aliases
MarkovTradingAgent = create_markov_agent
WyckoffAgent = create_wyckoff_agent
FibonacciAgent = create_fibonacci_agent

# Export main classes
__all__ = [
    "StrategyAgent",  # New primary class name
    "ConsolidatedStrategyAgent",  # Backwards compatibility alias
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
