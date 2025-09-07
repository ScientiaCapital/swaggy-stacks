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
    
    async def _create_tools(self) -> List[Tool]:
        """Create tools for the strategy agent"""
        tools = await super()._create_tools()
        
        # Import and initialize our LangGraph trading workflow
        from app.rag.agents.langgraph.trading_workflow import TradingWorkflowEngine
        from app.rag.services.tool_registry import get_langgraph_tool_registry
        
        # Initialize the LangGraph workflow engine
        if not hasattr(self, 'workflow_engine'):
            self.workflow_engine = TradingWorkflowEngine()
            await self.workflow_engine.initialize()
        
        # Get LangChain tools from our registry
        tool_registry = await get_langgraph_tool_registry()
        strategy_tools = await tool_registry.get_tools_for_agent(
            agent_type="strategy_agent",
            permission_level="advanced"
        )
        
        # Add LangGraph workflow execution tool
        from langchain.tools import Tool
        
        workflow_tool = Tool(
            name="execute_trading_workflow",
            description="Execute comprehensive LangGraph trading analysis workflow for a symbol",
            func=self._execute_langgraph_workflow
        )
        
        tools.extend(strategy_tools)
        tools.append(workflow_tool)
        
        return tools

    async def _execute_langgraph_workflow(self, symbol: str) -> str:
        """Execute the LangGraph trading workflow for comprehensive analysis"""
        import json
        from datetime import datetime
        
        try:
            # Initialize workflow engine if not already done
            if not hasattr(self, 'workflow_engine'):
                from app.rag.agents.langgraph.trading_workflow import TradingWorkflowEngine
                self.workflow_engine = TradingWorkflowEngine()
                await self.workflow_engine.initialize()
            
            # Create initial state for the workflow
            initial_state = {
                "session_id": f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "symbol": symbol.upper(),
                "user_preferences": {
                    "risk_tolerance": "medium",
                    "strategy_focus": self.strategies,
                    "analysis_depth": self.analysis_complexity
                },
                "workflow_metadata": {
                    "initiated_by": f"StrategyAgent_{self.agent_name}",
                    "timestamp": datetime.now().isoformat(),
                    "strategies_enabled": self.strategies
                }
            }
            
            # Execute the complete LangGraph workflow
            final_state = await self.workflow_engine.execute_workflow(initial_state)
            
            # Extract key insights for strategy decision making
            analysis_summary = {
                "symbol": symbol,
                "market_analysis": final_state.get("market_context", {}),
                "strategy_signals": final_state.get("strategy_signals", []),
                "risk_assessment": final_state.get("risk_metrics", {}),
                "execution_plan": final_state.get("execution_plan", {}),
                "confidence_score": final_state.get("decision_confidence", 0.0),
                "recommended_action": final_state.get("final_decision", {}).get("action", "HOLD"),
                "reasoning": final_state.get("final_decision", {}).get("reasoning", ""),
                "workflow_session": final_state.get("session_id", "")
            }
            
            # Store results in agent memory for learning
            if self.learning_enabled and final_state.get("final_decision"):
                await self._store_workflow_results(symbol, analysis_summary)
            
            return json.dumps(analysis_summary, indent=2)
            
        except Exception as e:
            error_msg = f"LangGraph workflow execution failed for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return json.dumps({"error": error_msg, "symbol": symbol})
    
    async def _store_workflow_results(self, symbol: str, analysis: Dict[str, Any]) -> None:
        """Store LangGraph workflow results for learning and pattern recognition"""
        try:
            # Create a learning pattern from workflow results
            workflow_features = {
                "symbol": symbol,
                "strategy_combination": "_".join(self.strategies),
                "confidence": analysis.get("confidence_score", 0.0),
                "action": analysis.get("recommended_action", "HOLD"),
                "market_regime": analysis.get("market_analysis", {}).get("regime", "unknown"),
                "risk_score": analysis.get("risk_assessment", {}).get("overall_score", 0.5),
                "signal_count": len(analysis.get("strategy_signals", [])),
                "workflow_session": analysis.get("workflow_session", "")
            }
            
            # Convert to text for embedding storage
            feature_text = self._features_to_text(workflow_features)
            
            if self.embedding_service:
                embedding_result = await self.embedding_service.embed_text(feature_text)
                
                # Store pattern for future similarity matching
                async with self._get_db_connection() as conn:
                    await conn.execute(
                        """
                        INSERT INTO agent_patterns (
                            agent_type, strategy_name, pattern_name, pattern_embedding,
                            pattern_metadata, market_data, success_rate, occurrence_count,
                            total_profit_loss, is_active, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        """,
                        self.agent_name,
                        f"langgraph_{self.strategy_type}",
                        f"workflow_{symbol}_{analysis.get('workflow_session', 'unknown')}",
                        embedding_result.embedding.tolist(),
                        json.dumps(workflow_features),
                        json.dumps(analysis),
                        0.5,  # Neutral until we get outcome feedback
                        1,
                        0.0,  # Will be updated when we get P&L feedback
                        True,
                        datetime.now()
                    )
                    
                self.logger.info(f"Stored LangGraph workflow results for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error storing workflow results for {symbol}: {e}")

    async def analyze_market_with_langgraph(self, market_data: Dict[str, Any]) -> TradingSignal:
        """Enhanced market analysis using LangGraph workflow integration"""
        symbol = market_data.get("symbol", "UNKNOWN")
        
        try:
            # Execute LangGraph workflow for comprehensive analysis
            workflow_results = await self._execute_langgraph_workflow(symbol)
            analysis = json.loads(workflow_results)
            
            if "error" in analysis:
                # Fallback to traditional strategy analysis
                return await self._fallback_analysis(market_data)
            
            # Convert LangGraph results to TradingSignal
            from app.rag.models.trading_models import TradingSignal, SignalType
            
            action_map = {
                "BUY": SignalType.BUY,
                "SELL": SignalType.SELL,
                "HOLD": SignalType.HOLD,
                "WAIT": SignalType.HOLD
            }
            
            signal = TradingSignal(
                agent_id=self.agent_name,
                symbol=symbol,
                signal_type=action_map.get(analysis.get("recommended_action", "HOLD"), SignalType.HOLD),
                confidence=analysis.get("confidence_score", 0.5),
                reasoning=analysis.get("reasoning", "LangGraph workflow analysis"),
                metadata={
                    "workflow_session": analysis.get("workflow_session", ""),
                    "risk_assessment": analysis.get("risk_assessment", {}),
                    "strategy_signals": analysis.get("strategy_signals", []),
                    "market_analysis": analysis.get("market_analysis", {}),
                    "langgraph_enhanced": True
                },
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"LangGraph analysis failed for {symbol}, falling back to traditional analysis: {e}")
            return await self._fallback_analysis(market_data)
    
    async def _fallback_analysis(self, market_data: Dict[str, Any]) -> TradingSignal:
        """Fallback to traditional strategy analysis if LangGraph fails"""
        # This would call the original strategy-specific analysis
        # For now, return a basic signal
        from app.rag.models.trading_models import TradingSignal, SignalType
        
        return TradingSignal(
            agent_id=self.agent_name,
            symbol=market_data.get("symbol", "UNKNOWN"),
            signal_type=SignalType.HOLD,
            confidence=0.5,
            reasoning="Fallback analysis - LangGraph workflow unavailable",
            metadata={"fallback_mode": True},
            timestamp=datetime.now()
        )

    async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
        """Main market analysis method - uses LangGraph workflow by default"""
        return await self.analyze_market_with_langgraph(market_data)
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract strategy-specific features from market data for pattern recognition"""
        symbol = market_data.get("symbol", "UNKNOWN")
        
        # Extract basic market features
        features = {
            "symbol": symbol,
            "strategies_used": self.strategies,
            "price": market_data.get("current_price", 0.0),
            "volume": market_data.get("volume", 0.0),
            "timestamp": market_data.get("timestamp", datetime.now().isoformat())
        }
        
        # Add strategy-specific feature extraction
        if "markov" in self.strategies:
            features.update(self._extract_markov_features(market_data))
        
        if "wyckoff" in self.strategies:
            features.update(self._extract_wyckoff_features(market_data))
            
        if "fibonacci" in self.strategies:
            features.update(self._extract_fibonacci_features(market_data))
        
        return features
    
    def _extract_markov_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Markov chain specific features"""
        return {
            "markov_state": market_data.get("markov_state", "neutral"),
            "transition_probability": market_data.get("transition_prob", 0.5),
            "regime_strength": market_data.get("regime_strength", 0.5)
        }
    
    def _extract_wyckoff_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Wyckoff method specific features"""
        return {
            "wyckoff_phase": market_data.get("wyckoff_phase", "unknown"),
            "volume_analysis": market_data.get("volume_profile", {}),
            "effort_vs_result": market_data.get("effort_result", 0.5)
        }
    
    def _extract_fibonacci_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Fibonacci analysis specific features"""
        return {
            "fib_level": market_data.get("fibonacci_level", 0.5),
            "retracement_depth": market_data.get("retracement", 0.0),
            "support_resistance": market_data.get("sr_levels", [])
        }
    
    def _initialize_strategy_engines(self) -> None:
        """Initialize individual strategy engines based on selected strategies"""
        self.strategy_engines = {}
        
        # This would initialize individual strategy engines
        # For now, we'll rely on the LangGraph workflow to handle strategy coordination
        for strategy in self.strategies:
            self.strategy_engines[strategy] = {
                "name": self.AVAILABLE_STRATEGIES[strategy],
                "enabled": True,
                "config": self.strategy_configs.get(strategy, {})
            }
            
        self.logger.info(f"Initialized {len(self.strategy_engines)} strategy engines")

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
