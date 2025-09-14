"""
Composite Strategy Plugin
Combines multiple trading strategies using ensemble methods with StrategyPlugin interface
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from langchain.agents import Tool

from app.monitoring.metrics import PrometheusMetrics

logger = logging.getLogger(__name__)


class CompositeStrategy:
    """Composite strategy that combines multiple strategy signals using ensemble methods"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = "composite"

        # Ensemble configuration
        self.ensemble_method = config.get("ensemble_method", "weighted_voting")  # or "majority_voting", "confidence_weighted"
        self.min_strategies_agreement = config.get("min_strategies_agreement", 2)
        self.confidence_threshold = config.get("confidence_threshold", 0.65)

        # Strategy weights (can be dynamically adjusted based on performance)
        self.strategy_weights = config.get("strategy_weights", {
            "markov": 0.25,
            "wyckoff": 0.20,
            "fibonacci": 0.15,
            "candlestick": 0.15,
            "technical": 0.25
        })

        # Initialize sub-strategies
        self.strategies = {}
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize all available sub-strategies"""
        try:
            # Import and initialize individual strategies
            from app.rag.agents.strategy_agent import MarkovStrategy, WyckoffStrategy, FibonacciStrategy
            from app.rag.agents.candlestick_strategy import CandlestickStrategy
            from app.rag.agents.technical_strategy import TechnicalStrategy

            self.strategies = {
                "markov": MarkovStrategy(self.config.get("markov_config", {})),
                "wyckoff": WyckoffStrategy(self.config.get("wyckoff_config", {})),
                "fibonacci": FibonacciStrategy(self.config.get("fibonacci_config", {})),
                "candlestick": CandlestickStrategy(self.config.get("candlestick_config", {})),
                "technical": TechnicalStrategy(self.config.get("technical_config", {})),
            }

            logger.info(f"✅ Composite strategy initialized with {len(self.strategies)} sub-strategies")

        except Exception as e:
            logger.error(f"⚠️ Composite strategy initialization failed: {e}")
            self.strategies = {}

    def get_tools(self) -> List[Tool]:
        """Return LangChain tools for composite strategy"""
        tools = [
            Tool(
                name="analyze_strategy_consensus",
                func=self._analyze_consensus_tool,
                description="Analyze consensus across all trading strategies",
            ),
            Tool(
                name="get_strategy_breakdown",
                func=self._strategy_breakdown_tool,
                description="Get detailed breakdown of individual strategy signals",
            ),
            Tool(
                name="adjust_strategy_weights",
                func=self._adjust_weights_tool,
                description="Adjust strategy weights based on recent performance",
            ),
            Tool(
                name="validate_signal_quality",
                func=self._validate_signal_tool,
                description="Validate the quality and reliability of composite signals",
            ),
        ]

        # Add tools from all sub-strategies
        for strategy_name, strategy in self.strategies.items():
            if hasattr(strategy, 'get_tools'):
                strategy_tools = strategy.get_tools()
                # Prefix tool names to avoid conflicts
                for tool in strategy_tools:
                    tool.name = f"{strategy_name}_{tool.name}"
                    tool.description = f"[{strategy_name.upper()}] {tool.description}"
                tools.extend(strategy_tools)

        return tools

    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform composite market analysis using all sub-strategies"""
        start_time = datetime.now()
        metrics = PrometheusMetrics()
        symbol = market_data.get("symbol", "UNKNOWN")

        try:
            individual_analyses = {}
            individual_signals = {}
            failed_strategies = []

            # Run analysis for each strategy
            for strategy_name, strategy in self.strategies.items():
                try:
                    if hasattr(strategy, 'analyze_market'):
                        analysis = strategy.analyze_market(market_data)
                        individual_analyses[strategy_name] = analysis

                        # Generate signal from analysis
                        if hasattr(strategy, 'generate_signal') and "error" not in analysis:
                            signal = strategy.generate_signal(analysis, market_data)
                            individual_signals[strategy_name] = signal
                        else:
                            individual_signals[strategy_name] = {
                                "action": "HOLD",
                                "confidence": 0.0,
                                "reasoning": "Analysis failed or insufficient data"
                            }

                except Exception as e:
                    logger.warning(f"{strategy_name} strategy analysis failed: {e}")
                    failed_strategies.append(strategy_name)
                    individual_analyses[strategy_name] = {"error": "analysis_failed", "message": str(e)}
                    individual_signals[strategy_name] = {
                        "action": "HOLD",
                        "confidence": 0.0,
                        "reasoning": f"Strategy failed: {str(e)}"
                    }

            # Calculate ensemble signal
            ensemble_result = self._calculate_ensemble_signal(individual_signals)

            # Generate strategy performance metrics
            successful_strategies = len(self.strategies) - len(failed_strategies)
            strategy_agreement = self._calculate_strategy_agreement(individual_signals)

            analysis_time = (datetime.now() - start_time).total_seconds()
            metrics.record_strategy_analysis_latency(analysis_time, "composite", symbol)

            return {
                "individual_analyses": individual_analyses,
                "individual_signals": individual_signals,
                "failed_strategies": failed_strategies,
                "successful_strategies": successful_strategies,
                "strategy_agreement": strategy_agreement,
                "ensemble_method": self.ensemble_method,
                "ensemble_signal": ensemble_result["signal"],
                "ensemble_confidence": ensemble_result["confidence"],
                "ensemble_reasoning": ensemble_result["reasoning"],
                "weights_used": self.strategy_weights,
                "analysis_time": analysis_time,
            }

        except Exception as e:
            analysis_time = (datetime.now() - start_time).total_seconds()
            metrics.record_strategy_analysis_latency(analysis_time, "composite", symbol)
            logger.error(f"Composite strategy analysis failed for {symbol}: {str(e)}")
            return {"error": "composite_analysis_failed", "message": str(e)}

    def generate_signal(self, analysis: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate composite trading signal from ensemble analysis"""
        if "error" in analysis:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": analysis.get("message", "Composite analysis failed"),
            }

        ensemble_signal = analysis.get("ensemble_signal", "HOLD")
        ensemble_confidence = analysis.get("ensemble_confidence", 0.0)
        strategy_agreement = analysis.get("strategy_agreement", 0.0)
        successful_strategies = analysis.get("successful_strategies", 0)

        # Quality checks
        if successful_strategies < 2:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Insufficient strategy data: only {successful_strategies} strategies succeeded",
            }

        if ensemble_confidence < self.confidence_threshold:
            return {
                "action": "HOLD",
                "confidence": ensemble_confidence,
                "reasoning": f"Ensemble confidence too low: {ensemble_confidence:.1%} (threshold: {self.confidence_threshold:.1%})",
            }

        if strategy_agreement < 0.5:
            return {
                "action": "HOLD",
                "confidence": ensemble_confidence * 0.7,  # Reduce confidence for low agreement
                "reasoning": f"Low strategy agreement: {strategy_agreement:.1%}, reducing confidence",
            }

        # Generate detailed reasoning
        reasoning = f"Ensemble signal from {successful_strategies} strategies "
        reasoning += f"(agreement: {strategy_agreement:.1%}, method: {self.ensemble_method})"

        return {
            "action": ensemble_signal,
            "confidence": ensemble_confidence,
            "reasoning": analysis.get("ensemble_reasoning", reasoning),
            "strategy_count": successful_strategies,
            "agreement_score": strategy_agreement,
        }

    def _calculate_ensemble_signal(self, individual_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate ensemble signal using configured method"""
        if self.ensemble_method == "weighted_voting":
            return self._weighted_voting_ensemble(individual_signals)
        elif self.ensemble_method == "majority_voting":
            return self._majority_voting_ensemble(individual_signals)
        elif self.ensemble_method == "confidence_weighted":
            return self._confidence_weighted_ensemble(individual_signals)
        else:
            # Default to weighted voting
            return self._weighted_voting_ensemble(individual_signals)

    def _weighted_voting_ensemble(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble using predefined strategy weights"""
        action_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        total_weight = 0.0
        reasoning_parts = []

        for strategy_name, signal in signals.items():
            if signal.get("confidence", 0) > 0:
                action = signal.get("action", "HOLD")
                confidence = signal.get("confidence", 0.0)
                weight = self.strategy_weights.get(strategy_name, 0.1)

                action_scores[action] += weight * confidence
                total_weight += weight

                reasoning_parts.append(f"{strategy_name}({action}, {confidence:.2f})")

        if total_weight == 0:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": "No valid strategy signals"
            }

        # Normalize scores
        for action in action_scores:
            action_scores[action] /= total_weight

        # Find dominant action
        dominant_action = max(action_scores, key=action_scores.get)
        ensemble_confidence = action_scores[dominant_action]

        reasoning = f"Weighted ensemble: {', '.join(reasoning_parts[:3])}"
        if len(reasoning_parts) > 3:
            reasoning += f" +{len(reasoning_parts) - 3} more"

        return {
            "signal": dominant_action,
            "confidence": ensemble_confidence,
            "reasoning": reasoning
        }

    def _majority_voting_ensemble(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble using simple majority voting"""
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        valid_signals = 0
        reasoning_parts = []

        for strategy_name, signal in signals.items():
            if signal.get("confidence", 0) > 0.3:  # Minimum confidence threshold
                action = signal.get("action", "HOLD")
                votes[action] += 1
                valid_signals += 1
                reasoning_parts.append(f"{strategy_name}({action})")

        if valid_signals == 0:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": "No valid strategy votes"
            }

        # Find majority
        dominant_action = max(votes, key=votes.get)
        majority_strength = votes[dominant_action] / valid_signals

        # Calculate ensemble confidence
        ensemble_confidence = majority_strength * 0.8  # Cap at 80% for majority voting

        reasoning = f"Majority vote ({votes[dominant_action]}/{valid_signals}): {', '.join(reasoning_parts[:3])}"

        return {
            "signal": dominant_action,
            "confidence": ensemble_confidence,
            "reasoning": reasoning
        }

    def _confidence_weighted_ensemble(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble weighted by individual strategy confidence"""
        action_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        total_confidence = 0.0
        reasoning_parts = []

        for strategy_name, signal in signals.items():
            confidence = signal.get("confidence", 0.0)
            if confidence > 0:
                action = signal.get("action", "HOLD")

                action_scores[action] += confidence
                total_confidence += confidence

                reasoning_parts.append(f"{strategy_name}({action}, {confidence:.2f})")

        if total_confidence == 0:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": "No confident strategy signals"
            }

        # Normalize by total confidence
        for action in action_scores:
            action_scores[action] /= total_confidence

        dominant_action = max(action_scores, key=action_scores.get)
        ensemble_confidence = action_scores[dominant_action]

        reasoning = f"Confidence-weighted: {', '.join(reasoning_parts[:3])}"

        return {
            "signal": dominant_action,
            "confidence": ensemble_confidence,
            "reasoning": reasoning
        }

    def _calculate_strategy_agreement(self, signals: Dict[str, Dict[str, Any]]) -> float:
        """Calculate agreement level between strategies"""
        actions = [signal.get("action", "HOLD") for signal in signals.values()
                  if signal.get("confidence", 0) > 0.3]

        if len(actions) < 2:
            return 0.0

        # Calculate agreement as percentage of strategies agreeing with majority
        action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for action in actions:
            action_counts[action] += 1

        majority_count = max(action_counts.values())
        return majority_count / len(actions)

    def update_strategy_weights(self, performance_metrics: Dict[str, float]):
        """Update strategy weights based on recent performance"""
        total_performance = sum(performance_metrics.values())

        if total_performance > 0:
            # Normalize performance to create new weights
            new_weights = {}
            for strategy, performance in performance_metrics.items():
                new_weights[strategy] = performance / total_performance

            # Apply smoothing to avoid dramatic weight changes
            smoothing_factor = 0.3
            for strategy in self.strategy_weights:
                if strategy in new_weights:
                    self.strategy_weights[strategy] = (
                        (1 - smoothing_factor) * self.strategy_weights[strategy] +
                        smoothing_factor * new_weights[strategy]
                    )

            logger.info(f"Updated strategy weights: {self.strategy_weights}")

    # Tool implementations
    def _analyze_consensus_tool(self, market_data: str) -> str:
        """Tool implementation for consensus analysis"""
        try:
            # Parse basic market data
            data_parts = market_data.split(",")
            if len(data_parts) < 2:
                return "Invalid market data format. Use: symbol,price,volume"

            market_dict = {
                "symbol": data_parts[0].strip(),
                "current_price": float(data_parts[1].strip()),
                "volume": float(data_parts[2].strip()) if len(data_parts) > 2 else 1000
            }

            analysis = self.analyze_market(market_dict)

            if "error" in analysis:
                return f"Consensus analysis failed: {analysis.get('message', 'Unknown error')}"

            agreement = analysis.get("strategy_agreement", 0.0)
            signal = analysis.get("ensemble_signal", "HOLD")
            confidence = analysis.get("ensemble_confidence", 0.0)

            return f"Consensus: {signal} (confidence: {confidence:.1%}, agreement: {agreement:.1%})"

        except Exception as e:
            return f"Consensus analysis error: {str(e)}"

    def _strategy_breakdown_tool(self, strategy_filter: str = "") -> str:
        """Tool implementation for strategy breakdown"""
        try:
            breakdown = []
            for strategy_name, strategy in self.strategies.items():
                if not strategy_filter or strategy_filter.lower() in strategy_name:
                    weight = self.strategy_weights.get(strategy_name, 0.0)
                    breakdown.append(f"{strategy_name}: {weight:.1%} weight")

            return "Strategy breakdown: " + ", ".join(breakdown)

        except Exception as e:
            return f"Strategy breakdown error: {str(e)}"

    def _adjust_weights_tool(self, performance_data: str) -> str:
        """Tool implementation for weight adjustment"""
        try:
            # Parse performance data: "strategy1:score1,strategy2:score2"
            performance_metrics = {}
            for item in performance_data.split(","):
                if ":" in item:
                    strategy, score = item.split(":")
                    performance_metrics[strategy.strip()] = float(score.strip())

            self.update_strategy_weights(performance_metrics)
            return f"Updated strategy weights based on performance: {self.strategy_weights}"

        except Exception as e:
            return f"Weight adjustment error: {str(e)}"

    def _validate_signal_tool(self, signal_data: str) -> str:
        """Tool implementation for signal quality validation"""
        try:
            # Parse signal data: "action,confidence,agreement"
            parts = signal_data.split(",")
            if len(parts) < 3:
                return "Invalid signal format. Use: action,confidence,agreement"

            action = parts[0].strip()
            confidence = float(parts[1].strip())
            agreement = float(parts[2].strip())

            validation_score = 0.0

            # Validate action
            if action in ["BUY", "SELL"]:
                validation_score += 0.3
            elif action == "HOLD":
                validation_score += 0.1

            # Validate confidence
            if confidence >= 0.7:
                validation_score += 0.4
            elif confidence >= 0.5:
                validation_score += 0.2

            # Validate agreement
            if agreement >= 0.8:
                validation_score += 0.3
            elif agreement >= 0.6:
                validation_score += 0.2

            quality_rating = "High" if validation_score >= 0.8 else "Medium" if validation_score >= 0.5 else "Low"

            return f"Signal quality: {quality_rating} (score: {validation_score:.2f})"

        except Exception as e:
            return f"Signal validation error: {str(e)}"

    def extract_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract composite strategy features for pattern matching"""
        analysis = self.analyze_market(market_data)

        return {
            "strategy": self.name,
            "timestamp": datetime.now().isoformat(),
            "ensemble_signal": analysis.get("ensemble_signal", "HOLD"),
            "ensemble_confidence": analysis.get("ensemble_confidence", 0.0),
            "strategy_agreement": analysis.get("strategy_agreement", 0.0),
            "successful_strategies": analysis.get("successful_strategies", 0),
            "ensemble_method": self.ensemble_method,
            "total_strategies": len(self.strategies),
        }