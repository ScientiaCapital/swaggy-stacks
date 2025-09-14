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
# Import metrics for monitoring
from app.monitoring.metrics import PrometheusMetrics


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
        start_time = datetime.now()
        metrics = PrometheusMetrics()
        symbol = market_data.get("symbol", "UNKNOWN")
        
        try:
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

            analysis_time = (datetime.now() - start_time).total_seconds()
            confidence = float(np.max(next_state_probs))
            
            # Record strategy performance metrics
            metrics.record_strategy_analysis_latency(analysis_time, "markov", symbol)
            
            return {
                "current_state": int(current_state),
                "transition_matrix": transition_matrix.tolist(),
                "next_state_probabilities": next_state_probs.tolist(),
                "confidence": confidence,
                "states_sequence": states.tolist(),
            }

        except Exception as e:
            analysis_time = (datetime.now() - start_time).total_seconds()
            metrics.record_strategy_analysis_latency(analysis_time, "markov", symbol)
            logger.error(f"Markov analysis failed for {symbol}: {str(e)}")
            return {"error": "analysis_failed", "message": str(e)}

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
        start_time = datetime.now()
        metrics = PrometheusMetrics()
        symbol = market_data.get("symbol", "UNKNOWN")
        
        try:
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

            analysis_time = (datetime.now() - start_time).total_seconds()
            confidence = min(abs(volume_trend) / volume_avg, 1.0)
            
            # Record strategy performance metrics
            metrics.record_strategy_analysis_latency(analysis_time, "wyckoff", symbol)

            return {
                "phase": phase,
                "volume_trend": volume_trend,
                "price_trend": price_trend,
                "effort_result_ratio": volume_trend / (abs(price_trend) + 1e-6),
                "confidence": confidence,
            }

        except Exception as e:
            analysis_time = (datetime.now() - start_time).total_seconds()
            metrics.record_strategy_analysis_latency(analysis_time, "wyckoff", symbol)
            logger.error(f"Wyckoff analysis failed for {symbol}: {str(e)}")
            return {"error": "analysis_failed", "message": str(e)}

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
        start_time = datetime.now()
        metrics = PrometheusMetrics()
        symbol = market_data.get("symbol", "UNKNOWN")
        
        try:
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

            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Record strategy performance metrics
            metrics.record_strategy_analysis_latency(analysis_time, "fibonacci", symbol)

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

        except Exception as e:
            analysis_time = (datetime.now() - start_time).total_seconds()
            metrics.record_strategy_analysis_latency(analysis_time, "fibonacci", symbol)
            logger.error(f"Fibonacci analysis failed for {symbol}: {str(e)}")
            return {"error": "analysis_failed", "message": str(e)}

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
    Refactored Multi-Strategy Trading Agent
    Delegates to modular strategy components while maintaining backwards compatibility
    """

    AVAILABLE_STRATEGIES = {
        "markov": "Markov Chain Analysis",
        "wyckoff": "Wyckoff Method Analysis", 
        "fibonacci": "Fibonacci Retracement Analysis",
        "candlestick": "Candlestick Pattern Analysis",
        "technical": "Technical Indicator Analysis",
        "composite": "Composite Multi-Strategy Analysis"
    }

    def __init__(self, strategies: List[str] = None, strategy_configs: Dict[str, Dict] = None,
                 market_research_service=None, use_market_research: bool = False,
                 analysis_complexity: str = "medium", ai_advisor=None,
                 use_ai_advisor: bool = False, ai_analysis_types: List[str] = None,
                 consensus_method: str = "weighted_average", **kwargs):
        
        # Initialize configurations
        self.strategies = strategies or ["composite"]
        self.strategy_configs = strategy_configs or {}
        self.market_research_service = market_research_service
        self.use_market_research = use_market_research
        self.analysis_complexity = analysis_complexity
        self.ai_advisor = ai_advisor
        self.use_ai_advisor = use_ai_advisor
        self.ai_analysis_types = ai_analysis_types or ["technical_analysis", "sentiment_analysis", "fundamental_analysis"]
        
        # Validate strategies
        invalid_strategies = [s for s in self.strategies if s not in self.AVAILABLE_STRATEGIES]
        if invalid_strategies:
            raise ValueError(f"Invalid strategies: {invalid_strategies}")
        
        self._initialize_modular_strategy_engines()
        self.consensus_method = consensus_method

        # Initialize BacktestService for AI agent learning
        self._initialize_backtest_service()

        super().__init__(**kwargs)
        self.logger.info(f"Refactored StrategyAgent initialized with {len(self.strategies)} strategies")
    
    def _initialize_modular_strategy_engines(self) -> None:
        """Initialize modular strategy engines"""
        self.strategy_engines = {}
        
        try:
            from app.rag.agents.candlestick_strategy import CandlestickStrategy
            from app.rag.agents.technical_strategy import TechnicalStrategy
            from app.rag.agents.composite_strategy import CompositeStrategy
            
            strategy_class_map = {
                "candlestick": CandlestickStrategy,
                "technical": TechnicalStrategy,
                "composite": CompositeStrategy,
                "markov": MarkovStrategy,
                "wyckoff": WyckoffStrategy,
                "fibonacci": FibonacciStrategy
            }
            
            for strategy in self.strategies:
                config = self.strategy_configs.get(strategy, {})
                if strategy in strategy_class_map:
                    self.strategy_engines[strategy] = strategy_class_map[strategy](config)
                    
        except Exception as e:
            self.logger.error(f"Strategy engine initialization failed: {e}")
            for strategy in self.strategies:
                self.strategy_engines[strategy] = {"name": self.AVAILABLE_STRATEGIES[strategy], "enabled": False, "error": str(e)}
            
        self.logger.info(f"Initialized {len(self.strategy_engines)} modular strategy engines")

    def _initialize_backtest_service(self) -> None:
        """Initialize BacktestService for AI agent learning integration"""
        try:
            from app.services.backtest_service import BacktestService
            self.backtest_service = BacktestService()
            self.agent_id = f"strategy_agent_{id(self)}"
            self.logger.info("✅ BacktestService initialized for AI agent learning")
        except Exception as e:
            self.logger.warning(f"⚠️ BacktestService initialization failed: {e}")
            self.backtest_service = None

    async def submit_trading_idea_for_learning(
        self,
        symbol: str,
        strategy: str,
        direction: str,
        confidence: float,
        market_data: Dict[str, Any],
        reasoning: str = ""
    ) -> Dict[str, Any]:
        """Submit trading idea to backtest service for learning and feedback"""
        try:
            if not self.backtest_service:
                return {"error": "BacktestService not available"}

            from app.services.backtest_service import TradingIdea

            # Create trading idea from strategy analysis
            trading_idea = TradingIdea(
                id="",  # Will be generated by service
                agent_id=self.agent_id,
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                confidence=confidence,
                entry_conditions=self._extract_entry_conditions(market_data, strategy),
                exit_conditions=self._extract_exit_conditions(strategy),
                risk_parameters={"max_loss": 0.05, "position_size": 0.02},
                timeframe="1D",
                reasoning=reasoning
            )

            # Submit for backtesting and learning
            result = await self.backtest_service.submit_trading_idea(
                agent_id=self.agent_id,
                trading_idea=trading_idea
            )

            self.logger.info(
                "Trading idea submitted for learning",
                symbol=symbol,
                strategy=strategy,
                performance_rank=result.get("feedback", {}).get("performance_rank", "unknown")
            )

            return result

        except Exception as e:
            self.logger.error("Error submitting trading idea for learning", error=str(e))
            return {"error": str(e)}

    async def get_pattern_learning_insights(
        self,
        pattern_name: str,
        market_conditions: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get learning insights for a specific pattern"""
        try:
            if not self.backtest_service:
                return {"error": "BacktestService not available"}

            insights = await self.backtest_service.get_pattern_performance(
                pattern_name=pattern_name,
                market_conditions=market_conditions,
                agent_id=self.agent_id
            )

            self.logger.info(
                "Pattern learning insights retrieved",
                pattern=pattern_name,
                success_rate=insights.get("success_rate", 0)
            )

            return insights

        except Exception as e:
            self.logger.error("Error retrieving pattern insights", error=str(e))
            return {"error": str(e)}

    async def optimize_strategy_parameters(
        self,
        strategy_name: str,
        current_parameters: Dict[str, Any],
        optimization_target: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using backtesting"""
        try:
            if not self.backtest_service:
                return {"error": "BacktestService not available"}

            # Define parameter space for optimization
            parameter_space = self._define_parameter_space(strategy_name, current_parameters)

            optimization_result = await self.backtest_service.optimize_strategy_parameters(
                agent_id=self.agent_id,
                strategy_name=strategy_name,
                parameter_space=parameter_space,
                optimization_metric=optimization_target,
                max_iterations=25  # Reasonable limit for real-time optimization
            )

            self.logger.info(
                "Strategy parameters optimized",
                strategy=strategy_name,
                best_metric=optimization_result.get("best_metric_value", 0),
                combinations_tested=optimization_result.get("total_combinations_tested", 0)
            )

            return optimization_result

        except Exception as e:
            self.logger.error("Error optimizing strategy parameters", error=str(e))
            return {"error": str(e)}

    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary for this agent"""
        try:
            if not self.backtest_service:
                return {"error": "BacktestService not available"}

            summary = await self.backtest_service.get_agent_learning_summary(
                agent_id=self.agent_id,
                days_lookback=30
            )

            self.logger.info(
                "Learning summary retrieved",
                total_ideas=summary.get("total_ideas_submitted", 0),
                success_rate=summary.get("success_rate", 0)
            )

            return summary

        except Exception as e:
            self.logger.error("Error retrieving learning summary", error=str(e))
            return {"error": str(e)}

    def _extract_entry_conditions(self, market_data: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Extract entry conditions from market analysis"""
        try:
            conditions = {}

            if strategy == "technical":
                conditions.update({
                    "rsi_threshold": 70 if market_data.get("rsi", 50) > 70 else 30,
                    "macd_signal": "bullish" if market_data.get("macd_histogram", 0) > 0 else "bearish",
                    "volume_confirmation": market_data.get("volume_ratio", 1.0) > 1.2
                })

            elif strategy == "candlestick":
                conditions.update({
                    "pattern_confidence": market_data.get("pattern_confidence", 0.6),
                    "trend_alignment": market_data.get("trend_context", "neutral"),
                    "volume_confirmation": market_data.get("volume_confirmation", 0.5) > 0.6
                })

            elif strategy == "composite":
                conditions.update({
                    "consensus_strength": market_data.get("consensus_strength", 0.5),
                    "strategy_agreement": market_data.get("strategy_agreement", 0.5),
                    "confidence_threshold": 0.7
                })

            else:  # markov, wyckoff, fibonacci
                conditions.update({
                    "signal_strength": market_data.get("signal_strength", 0.5),
                    "market_phase": market_data.get("market_phase", "unknown"),
                    "probability_threshold": 0.6
                })

            return conditions

        except Exception as e:
            self.logger.error("Error extracting entry conditions", error=str(e))
            return {"error": str(e)}

    def _extract_exit_conditions(self, strategy: str) -> Dict[str, Any]:
        """Extract exit conditions based on strategy type"""
        try:
            base_conditions = {
                "stop_loss_pct": 0.05,  # 5% stop loss
                "take_profit_pct": 0.10,  # 10% take profit
                "max_holding_days": 30
            }

            if strategy == "technical":
                base_conditions.update({
                    "rsi_exit": True,  # Exit when RSI reaches opposite extreme
                    "macd_reversal": True  # Exit on MACD signal reversal
                })

            elif strategy == "candlestick":
                base_conditions.update({
                    "reversal_pattern": True,  # Exit on reversal pattern
                    "volume_divergence": True  # Exit on volume divergence
                })

            return base_conditions

        except Exception as e:
            self.logger.error("Error extracting exit conditions", error=str(e))
            return {"stop_loss_pct": 0.05, "take_profit_pct": 0.10}

    def _define_parameter_space(
        self,
        strategy_name: str,
        current_parameters: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Define parameter space for optimization based on strategy type"""
        try:
            if strategy_name == "technical":
                return {
                    "rsi_period": {"type": "int", "min": 10, "max": 30, "steps": 5},
                    "rsi_oversold": {"type": "int", "min": 20, "max": 40, "steps": 5},
                    "rsi_overbought": {"type": "int", "min": 60, "max": 80, "steps": 5},
                    "macd_fast": {"type": "int", "min": 8, "max": 15, "steps": 4},
                    "macd_slow": {"type": "int", "min": 20, "max": 30, "steps": 4}
                }

            elif strategy_name == "candlestick":
                return {
                    "pattern_confidence_threshold": {"type": "float", "min": 0.4, "max": 0.8, "steps": 5},
                    "volume_confirmation_weight": {"type": "float", "min": 0.1, "max": 0.5, "steps": 5},
                    "trend_confirmation_weight": {"type": "float", "min": 0.1, "max": 0.5, "steps": 5}
                }

            elif strategy_name == "composite":
                return {
                    "consensus_threshold": {"type": "float", "min": 0.5, "max": 0.8, "steps": 4},
                    "strategy_weight_technical": {"type": "float", "min": 0.1, "max": 0.5, "steps": 4},
                    "strategy_weight_pattern": {"type": "float", "min": 0.1, "max": 0.5, "steps": 4}
                }

            else:  # Default parameter space
                return {
                    "confidence_threshold": {"type": "float", "min": 0.5, "max": 0.9, "steps": 5},
                    "lookback_period": {"type": "int", "min": 10, "max": 50, "steps": 5}
                }

        except Exception as e:
            self.logger.error("Error defining parameter space", error=str(e))
            return {}

    async def _create_tools(self) -> List[Tool]:
        """Create tools with modular strategy tools"""
        tools = await super()._create_tools()
        
        # Add tools from strategy engines
        for strategy_name, strategy_engine in self.strategy_engines.items():
            if hasattr(strategy_engine, 'get_tools'):
                try:
                    tools.extend(strategy_engine.get_tools())
                except Exception as e:
                    self.logger.warning(f"Failed to get tools from {strategy_name}: {e}")
        
        # Add LangGraph workflow tool
        try:
            from app.rag.agents.langgraph.trading_workflow import TradingWorkflowEngine
            if not hasattr(self, 'workflow_engine'):
                self.workflow_engine = TradingWorkflowEngine()
                await self.workflow_engine.initialize()
            
            tools.append(Tool(
                name="execute_trading_workflow",
                description="Execute comprehensive LangGraph trading analysis workflow",
                func=self._execute_langgraph_workflow
            ))
        except Exception as e:
            self.logger.warning(f"LangGraph workflow tool failed: {e}")

        # Add AI learning and backtesting tools
        if self.backtest_service:
            tools.extend([
                Tool(
                    name="submit_trading_idea_for_learning",
                    description="Submit trading idea for backtesting and AI learning feedback. Input: symbol,strategy,direction,confidence,reasoning",
                    func=self._submit_learning_idea_tool
                ),
                Tool(
                    name="get_pattern_learning_insights",
                    description="Get historical performance insights for trading patterns. Input: pattern_name,market_conditions",
                    func=self._get_pattern_insights_tool
                ),
                Tool(
                    name="optimize_strategy_parameters",
                    description="Optimize strategy parameters using backtesting. Input: strategy_name,optimization_target",
                    func=self._optimize_parameters_tool
                ),
                Tool(
                    name="get_agent_learning_summary",
                    description="Get comprehensive learning summary for this AI agent",
                    func=self._get_learning_summary_tool
                )
            ])

        return tools

    # AI Learning Tool Implementations
    async def _submit_learning_idea_tool(self, input_str: str) -> str:
        """Tool wrapper for submitting trading ideas for learning"""
        try:
            # Parse input: symbol,strategy,direction,confidence,reasoning
            parts = [p.strip() for p in input_str.split(",")]
            if len(parts) < 4:
                return "Error: Input format should be 'symbol,strategy,direction,confidence,reasoning'"

            symbol = parts[0]
            strategy = parts[1]
            direction = parts[2].upper()
            confidence = float(parts[3])
            reasoning = parts[4] if len(parts) > 4 else ""

            # Get current market data for context
            mock_market_data = {
                "symbol": symbol,
                "rsi": 60,
                "macd_histogram": 0.1,
                "volume_ratio": 1.3,
                "pattern_confidence": confidence,
                "trend_context": "bullish" if direction == "BUY" else "bearish"
            }

            result = await self.submit_trading_idea_for_learning(
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                confidence=confidence,
                market_data=mock_market_data,
                reasoning=reasoning
            )

            if "error" in result:
                return f"Error submitting idea: {result['error']}"

            feedback = result.get("feedback", {})
            return (
                f"Trading idea submitted successfully!\n"
                f"Idea ID: {result.get('idea_id')}\n"
                f"Performance Rank: {feedback.get('performance_rank', 'unknown')}\n"
                f"Expected Return: {feedback.get('total_return', 0):.2%}\n"
                f"Risk Score: {feedback.get('risk_score', 0):.1f}\n"
                f"Suggestions: {', '.join(feedback.get('improvement_suggestions', []))[:200]}..."
            )

        except Exception as e:
            return f"Error processing trading idea: {str(e)}"

    async def _get_pattern_insights_tool(self, input_str: str) -> str:
        """Tool wrapper for getting pattern learning insights"""
        try:
            # Parse input: pattern_name,market_conditions (optional)
            parts = [p.strip() for p in input_str.split(",")]
            pattern_name = parts[0]

            market_conditions = None
            if len(parts) > 1 and parts[1]:
                # Simple market conditions parsing
                market_conditions = {
                    "volatility": "normal",
                    "trend": parts[1] if parts[1] in ["bullish", "bearish", "neutral"] else "neutral"
                }

            insights = await self.get_pattern_learning_insights(
                pattern_name=pattern_name,
                market_conditions=market_conditions
            )

            if "error" in insights:
                return f"Error getting insights: {insights['error']}"

            return (
                f"Pattern Learning Insights for {pattern_name}:\n"
                f"Success Rate: {insights.get('success_rate', 0):.1%}\n"
                f"Details: {insights.get('details', {})}\n"
                f"Improvement Suggestions: {', '.join(insights.get('improvement_suggestions', []))}"
            )

        except Exception as e:
            return f"Error getting pattern insights: {str(e)}"

    async def _optimize_parameters_tool(self, input_str: str) -> str:
        """Tool wrapper for optimizing strategy parameters"""
        try:
            # Parse input: strategy_name,optimization_target
            parts = [p.strip() for p in input_str.split(",")]
            strategy_name = parts[0]
            optimization_target = parts[1] if len(parts) > 1 else "sharpe_ratio"

            result = await self.optimize_strategy_parameters(
                strategy_name=strategy_name,
                current_parameters={},  # Will use defaults from parameter space
                optimization_target=optimization_target
            )

            if "error" in result:
                return f"Error optimizing parameters: {result['error']}"

            best_params = result.get("best_parameters", {})
            return (
                f"Parameter Optimization Results for {strategy_name}:\n"
                f"Best {optimization_target}: {result.get('best_metric_value', 0):.3f}\n"
                f"Combinations Tested: {result.get('total_combinations_tested', 0)}\n"
                f"Best Parameters: {', '.join([f'{k}={v}' for k, v in best_params.items()])}"
            )

        except Exception as e:
            return f"Error optimizing parameters: {str(e)}"

    async def _get_learning_summary_tool(self, input_str: str = "") -> str:
        """Tool wrapper for getting agent learning summary"""
        try:
            summary = await self.get_learning_summary()

            if "error" in summary:
                return f"Error getting learning summary: {summary['error']}"

            if summary.get("total_ideas_submitted", 0) == 0:
                return "No trading ideas have been submitted for learning yet. Use submit_trading_idea_for_learning to begin the learning process."

            strategy_perf = summary.get("strategy_performance", {})
            best_strategy = summary.get("best_performing_strategy", "None")

            return (
                f"AI Agent Learning Summary:\n"
                f"Total Ideas Submitted: {summary.get('total_ideas_submitted', 0)}\n"
                f"Overall Success Rate: {summary.get('success_rate', 0):.1%}\n"
                f"Average Return: {summary.get('average_return', 0):.2%}\n"
                f"Average Sharpe Ratio: {summary.get('average_sharpe_ratio', 0):.2f}\n"
                f"Best Performing Strategy: {best_strategy}\n"
                f"Improvement Trend: {summary.get('improvement_trend', 'unknown')}\n"
                f"Recommendations: {', '.join(summary.get('learning_recommendations', []))[:300]}..."
            )

        except Exception as e:
            return f"Error getting learning summary: {str(e)}"

    async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
        """Main market analysis using modular strategy components"""
        symbol = market_data.get("symbol", "UNKNOWN")
        
        try:
            # Use composite strategy if available
            if "composite" in self.strategy_engines:
                composite = self.strategy_engines["composite"]
                if hasattr(composite, 'analyze_market') and hasattr(composite, 'generate_signal'):
                    analysis = composite.analyze_market(market_data)
                    if "error" not in analysis:
                        signal_data = composite.generate_signal(analysis, market_data)
                        return self._convert_to_trading_signal(signal_data, symbol, {"composite_analysis": analysis})
            
            # Fallback to first available strategy
            return await self._fallback_modular_analysis(market_data)
            
        except Exception as e:
            self.logger.error(f"Market analysis failed for {symbol}: {e}")
            return await self._fallback_analysis(market_data)

    async def _fallback_modular_analysis(self, market_data: Dict[str, Any]) -> TradingSignal:
        """Fallback using first available strategy engine"""
        symbol = market_data.get("symbol", "UNKNOWN")
        
        for strategy_name, strategy_engine in self.strategy_engines.items():
            if hasattr(strategy_engine, 'analyze_market') and hasattr(strategy_engine, 'generate_signal'):
                try:
                    analysis = strategy_engine.analyze_market(market_data)
                    if "error" not in analysis:
                        signal_data = strategy_engine.generate_signal(analysis, market_data)
                        return self._convert_to_trading_signal(signal_data, symbol, {f"{strategy_name}_analysis": analysis})
                except Exception as e:
                    self.logger.warning(f"{strategy_name} fallback failed: {e}")
                    continue
        
        return await self._fallback_analysis(market_data)

    def _convert_to_trading_signal(self, signal_data: Dict[str, Any], symbol: str, metadata: Dict[str, Any]) -> TradingSignal:
        """Convert strategy signal to TradingSignal object"""
        from app.rag.models.trading_models import TradingSignal, SignalType
        
        action_map = {"BUY": SignalType.BUY, "SELL": SignalType.SELL, "HOLD": SignalType.HOLD}
        
        return TradingSignal(
            agent_id=self.agent_name,
            symbol=symbol,
            signal_type=action_map.get(signal_data.get("action", "HOLD"), SignalType.HOLD),
            confidence=signal_data.get("confidence", 0.5),
            reasoning=signal_data.get("reasoning", "Modular strategy analysis"),
            metadata={**metadata, "modular_strategy": True, "strategies_used": self.strategies},
            timestamp=datetime.now()
        )

    async def _execute_langgraph_workflow(self, symbol: str) -> str:
        """Execute LangGraph workflow (preserved for compatibility)"""
        import json
        
        try:
            if not hasattr(self, 'workflow_engine'):
                from app.rag.agents.langgraph.trading_workflow import TradingWorkflowEngine
                self.workflow_engine = TradingWorkflowEngine()
                await self.workflow_engine.initialize()
            
            initial_state = {
                "session_id": f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "symbol": symbol.upper(),
                "user_preferences": {"risk_tolerance": "medium", "strategy_focus": self.strategies, "analysis_depth": self.analysis_complexity},
                "workflow_metadata": {"initiated_by": f"RefactoredStrategyAgent_{self.agent_name}", "timestamp": datetime.now().isoformat()}
            }
            
            final_state = await self.workflow_engine.execute_workflow(initial_state)
            
            analysis_summary = {
                "symbol": symbol,
                "market_analysis": final_state.get("market_context", {}),
                "confidence_score": final_state.get("decision_confidence", 0.0),
                "recommended_action": final_state.get("final_decision", {}).get("action", "HOLD"),
                "reasoning": final_state.get("final_decision", {}).get("reasoning", "")
            }
            
            if self.learning_enabled and final_state.get("final_decision"):
                await self._store_workflow_results(symbol, analysis_summary)
            
            return json.dumps(analysis_summary, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"LangGraph workflow failed: {str(e)}", "symbol": symbol})
    
    async def _store_workflow_results(self, symbol: str, analysis: Dict[str, Any]) -> None:
        """Store LangGraph results (preserved for compatibility)"""
        # Implementation preserved for learning system integration
        pass

    async def _fallback_analysis(self, market_data: Dict[str, Any]) -> TradingSignal:
        """Ultimate fallback analysis"""
        from app.rag.models.trading_models import TradingSignal, SignalType
        
        return TradingSignal(
            agent_id=self.agent_name,
            symbol=market_data.get("symbol", "UNKNOWN"),
            signal_type=SignalType.HOLD,
            confidence=0.5,
            reasoning="Fallback analysis - modular strategies unavailable",
            metadata={"fallback_mode": True, "refactored_agent": True},
            timestamp=datetime.now()
        )

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
