"""
Strategy Coordinator - Extracted from StrategyAgent

Coordinates strategy analysis, learning, and optimization
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.rag.strategies.strategy_engines import StrategyEngineManager, StrategySignal
from app.services.backtest_service import BacktestService

logger = logging.getLogger(__name__)


class StrategyCoordinator:
    """
    Coordinates strategy analysis and learning across multiple strategy engines

    This replaces the complex StrategyAgent by providing a clean interface
    for strategy analysis, backtesting integration, and parameter optimization
    """

    def __init__(
        self,
        agent_id: str = "default",
        strategy_config: Dict[str, Any] = None,
        backtest_service: BacktestService = None,
    ):
        self.agent_id = agent_id
        self.config = strategy_config or {}

        # Initialize strategy engine manager
        self.strategy_manager = StrategyEngineManager(self.config.get("strategies", {}))

        # Initialize backtest service for learning
        self.backtest_service = backtest_service or BacktestService()

        # Configuration options
        self.consensus_method = self.config.get(
            "consensus_method", "confidence_weighted"
        )
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.use_backtesting_feedback = self.config.get(
            "use_backtesting_feedback", True
        )

        logger.info(f"StrategyCoordinator initialized for agent {agent_id}")

    async def analyze_symbol(
        self,
        symbol: str,
        market_data: Dict[str, Any] = None,
        use_consensus: bool = True,
    ) -> Optional[StrategySignal]:
        """
        Analyze a symbol using available strategies

        Args:
            symbol: Trading symbol to analyze
            market_data: Market data for analysis
            use_consensus: Whether to generate consensus signal

        Returns:
            StrategySignal or None if no signal generated
        """
        try:
            # Prepare market data if not provided
            if not market_data:
                market_data = await self._fetch_market_data(symbol)

            if use_consensus:
                signal = await self.strategy_manager.get_consensus_signal(
                    symbol, market_data, self.consensus_method
                )
            else:
                # Get signals from all strategies
                signals = await self.strategy_manager.analyze_all_strategies(
                    symbol, market_data
                )
                signal = signals[0] if signals else None

            # Filter by minimum confidence
            if signal and signal.confidence >= self.min_confidence:
                logger.info(
                    f"Generated {signal.strategy} signal for {symbol}: {signal.direction} @ {signal.confidence:.2f}"
                )
                return signal

            return None

        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return None

    async def submit_trading_idea_for_learning(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        rationale: str = "",
    ) -> Dict[str, Any]:
        """Submit trading idea to backtest service for learning"""
        try:
            if not self.use_backtesting_feedback:
                return {
                    "status": "disabled",
                    "message": "Backtesting feedback disabled",
                }

            result = await self.backtest_service.submit_trading_idea(
                agent_id=self.agent_id,
                symbol=symbol,
                strategy=strategy,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                rationale=rationale,
            )

            return result

        except Exception as e:
            logger.error(f"Error submitting trading idea for learning: {e}")
            return {"status": "error", "error": str(e)}

    async def get_pattern_learning_insights(
        self, pattern_name: str = None
    ) -> Dict[str, Any]:
        """Get pattern learning insights from backtest service"""
        try:
            if not self.use_backtesting_feedback:
                return {"status": "disabled"}

            insights = await self.backtest_service.get_pattern_performance(pattern_name)
            return insights

        except Exception as e:
            logger.error(f"Error getting pattern insights: {e}")
            return {"error": str(e)}

    async def optimize_strategy_parameters(
        self,
        strategy_name: str,
        optimization_metric: str = "sharpe_ratio",
        max_iterations: int = 50,
    ) -> Dict[str, Any]:
        """Optimize parameters for a specific strategy"""
        try:
            # Get parameter space for the strategy
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if not strategy:
                return {"error": f"Strategy {strategy_name} not found"}

            parameter_space = strategy.get_parameter_space()
            if not parameter_space:
                return {"error": f"No parameter space defined for {strategy_name}"}

            # Use backtest service for optimization
            result = await self.backtest_service.optimize_strategy_parameters(
                agent_id=self.agent_id,
                strategy_name=strategy_name,
                parameter_ranges=parameter_space,
                optimization_metric=optimization_metric,
                max_iterations=max_iterations,
            )

            return result

        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {e}")
            return {"error": str(e)}

    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary for this agent"""
        try:
            summary = await self.backtest_service.get_agent_learning_summary(
                self.agent_id
            )

            # Add strategy-specific information
            summary["available_strategies"] = list(
                self.strategy_manager.strategies.keys()
            )
            summary["enabled_strategies"] = [
                name
                for name, strategy in self.strategy_manager.strategies.items()
                if strategy.enabled
            ]
            summary["consensus_method"] = self.consensus_method
            summary["min_confidence"] = self.min_confidence

            return summary

        except Exception as e:
            logger.error(f"Error getting learning summary: {e}")
            return {"error": str(e)}

    async def analyze_multiple_symbols(
        self, symbols: List[str], use_consensus: bool = True
    ) -> Dict[str, Optional[StrategySignal]]:
        """Analyze multiple symbols and return signals"""
        results = {}

        for symbol in symbols:
            try:
                signal = await self.analyze_symbol(symbol, use_consensus=use_consensus)
                results[symbol] = signal
            except Exception as e:
                logger.error(f"Error analyzing symbol {symbol}: {e}")
                results[symbol] = None

        return results

    async def get_strategy_performance_comparison(
        self, days_back: int = 30
    ) -> Dict[str, Any]:
        """Compare performance across different strategies"""
        try:
            # Get overall pattern performance
            pattern_performance = await self.backtest_service.get_pattern_performance(
                days_back=days_back
            )

            # Organize by strategy
            strategy_comparison = {}
            for strategy_name in self.strategy_manager.strategies.keys():
                if strategy_name in pattern_performance.get("patterns", {}):
                    strategy_comparison[strategy_name] = pattern_performance[
                        "patterns"
                    ][strategy_name]

            return {
                "comparison_period": days_back,
                "strategies": strategy_comparison,
                "total_strategies": len(strategy_comparison),
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting strategy performance comparison: {e}")
            return {"error": str(e)}

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.strategy_manager.strategies.keys())

    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategy names"""
        return [
            name
            for name, strategy in self.strategy_manager.strategies.items()
            if strategy.enabled
        ]

    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a specific strategy"""
        strategy = self.strategy_manager.get_strategy(strategy_name)
        if strategy:
            strategy.enabled = True
            logger.info(f"Enabled strategy: {strategy_name}")
            return True
        return False

    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a specific strategy"""
        strategy = self.strategy_manager.get_strategy(strategy_name)
        if strategy:
            strategy.enabled = False
            logger.info(f"Disabled strategy: {strategy_name}")
            return True
        return False

    def update_strategy_config(
        self, strategy_name: str, config: Dict[str, Any]
    ) -> bool:
        """Update configuration for a specific strategy"""
        strategy = self.strategy_manager.get_strategy(strategy_name)
        if strategy:
            strategy.config.update(config)
            logger.info(f"Updated config for strategy: {strategy_name}")
            return True
        return False

    def set_consensus_method(self, method: str):
        """Set consensus method for multi-strategy analysis"""
        valid_methods = ["confidence_weighted", "majority_vote", "simple_average"]
        if method in valid_methods:
            self.consensus_method = method
            logger.info(f"Consensus method set to: {method}")
        else:
            logger.warning(
                f"Invalid consensus method: {method}. Valid options: {valid_methods}"
            )

    def set_minimum_confidence(self, min_confidence: float):
        """Set minimum confidence threshold for signals"""
        if 0.0 <= min_confidence <= 1.0:
            self.min_confidence = min_confidence
            logger.info(f"Minimum confidence set to: {min_confidence}")
        else:
            logger.warning(
                f"Invalid confidence level: {min_confidence}. Must be between 0.0 and 1.0"
            )

    # Private helper methods
    async def _fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch market data for analysis (placeholder implementation)"""
        try:
            # TODO: Implement actual market data fetching
            # This is a placeholder that would connect to real market data sources

            import random

            base_price = 100.0
            return {
                "symbol": symbol,
                "current_price": base_price * random.uniform(0.95, 1.05),
                "recent_high": base_price * random.uniform(1.05, 1.15),
                "recent_low": base_price * random.uniform(0.85, 0.95),
                "volume": random.randint(500000, 2000000),
                "avg_volume": random.randint(800000, 1200000),
                "volatility": random.uniform(0.15, 0.35),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "current_price": 100.0,
                "recent_high": 105.0,
                "recent_low": 95.0,
                "volume": 1000000,
                "avg_volume": 1000000,
                "volatility": 0.2,
            }
