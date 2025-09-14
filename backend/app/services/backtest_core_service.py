"""
Core Backtest Service - Refactored from BacktestService

Handles core backtesting orchestration, trading idea execution, and pattern learning
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.backtesting.engine import BacktestConfig, RefactoredBacktestEngine as BacktestEngine
from app.core.database import get_db
from app.monitoring.metrics import PrometheusMetrics
from app.services.indicator_performance_service import IndicatorPerformanceService
from app.services.ml_model_service import MLModelService

logger = logging.getLogger(__name__)


class TradingIdea:
    """Trading idea data structure"""

    def __init__(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        rationale: str = "",
        timeframe: str = "1d",
    ):
        self.symbol = symbol
        self.strategy = strategy
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence
        self.rationale = rationale
        self.timeframe = timeframe
        self.timestamp = datetime.utcnow()


class BacktestFeedback:
    """Backtest feedback structure"""

    def __init__(
        self,
        idea_id: str,
        performance_score: float,
        win_rate: float,
        avg_return: float,
        max_drawdown: float,
        suggestions: List[str] = None,
    ):
        self.idea_id = idea_id
        self.performance_score = performance_score
        self.win_rate = win_rate
        self.avg_return = avg_return
        self.max_drawdown = max_drawdown
        self.suggestions = suggestions or []
        self.timestamp = datetime.utcnow()


class PatternLearner:
    """Simple pattern learning mechanism"""

    def __init__(self):
        self.patterns = {}
        self.performance_history = []

    def learn_from_result(self, pattern: str, result: float):
        if pattern not in self.patterns:
            self.patterns[pattern] = {"results": [], "avg_performance": 0.0}

        self.patterns[pattern]["results"].append(result)
        self.patterns[pattern]["avg_performance"] = sum(
            self.patterns[pattern]["results"]
        ) / len(self.patterns[pattern]["results"])

    def get_pattern_confidence(self, pattern: str) -> float:
        if pattern not in self.patterns:
            return 0.5  # Default confidence
        return max(0.1, min(0.9, self.patterns[pattern]["avg_performance"]))


class CoreBacktestService:
    """Core backtesting service with orchestration and pattern learning"""

    def __init__(self, db: Session = None, metrics: PrometheusMetrics = None):
        self.db = db or next(get_db())
        self.metrics = metrics or PrometheusMetrics()

        # Initialize sub-services
        self.indicator_service = IndicatorPerformanceService(self.db, self.metrics)
        self.ml_service = MLModelService(self.db, self.metrics)

        # Initialize core components
        self.backtest_engine = BacktestEngine(BacktestConfig())
        self.pattern_learner = PatternLearner()

        # In-memory storage for active trading ideas and performance
        self.trading_ideas: Dict[str, TradingIdea] = {}
        self.agent_performance: Dict[str, List[float]] = {}

        logger.info("CoreBacktestService initialized with sub-services")

    async def submit_trading_idea(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        rationale: str = "",
        timeframe: str = "1d",
        agent_id: str = "default",
    ) -> Dict[str, Any]:
        """Submit and backtest a trading idea"""
        try:
            # Create trading idea
            idea = TradingIdea(
                symbol=symbol,
                strategy=strategy,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                rationale=rationale,
                timeframe=timeframe,
            )

            idea_id = f"{symbol}_{strategy}_{int(idea.timestamp.timestamp())}"
            self.trading_ideas[idea_id] = idea

            # Prepare market data for backtesting
            market_data = await self._prepare_market_data_for_idea(symbol, timeframe)

            if not market_data:
                return {
                    "idea_id": idea_id,
                    "status": "failed",
                    "error": "Could not retrieve market data",
                }

            # Execute backtest
            backtest_results = await self._execute_idea_backtest(idea, market_data)

            # Generate agent feedback
            feedback = await self._generate_agent_feedback(idea, backtest_results)

            # Update agent performance tracking
            await self._update_agent_performance(agent_id, backtest_results)

            # Update pattern learning
            await self._update_pattern_learning(strategy, backtest_results)

            # Track indicator performance if applicable
            if "indicators_used" in backtest_results:
                for indicator_name in backtest_results["indicators_used"]:
                    await self.indicator_service.track_indicator_performance(
                        indicator_name=indicator_name,
                        symbol=symbol,
                        signal_type="buy" if entry_price > 0 else "sell",
                        confidence=confidence,
                        entry_price=entry_price,
                        current_price=entry_price,  # Initial tracking
                        market_condition="normal",
                        timeframe=timeframe,
                    )

            # Update metrics
            if self.metrics:
                self.metrics.backtest_ideas_total.labels(
                    strategy=strategy, symbol=symbol
                ).inc()

            logger.info(f"Processed trading idea {idea_id} for {symbol}")

            return {
                "idea_id": idea_id,
                "status": "completed",
                "backtest_results": backtest_results,
                "feedback": feedback.__dict__ if feedback else None,
                "confidence_adjusted": self.pattern_learner.get_pattern_confidence(
                    strategy
                ),
            }

        except Exception as e:
            logger.error(f"Error processing trading idea: {e}")
            return {
                "idea_id": idea_id if "idea_id" in locals() else "unknown",
                "status": "failed",
                "error": str(e),
            }

    async def get_pattern_performance(
        self, pattern_name: str = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Get performance analysis for trading patterns"""
        try:
            if pattern_name and pattern_name in self.pattern_learner.patterns:
                pattern_data = self.pattern_learner.patterns[pattern_name]
                recent_results = pattern_data["results"][-10:]  # Last 10 results

                return {
                    "pattern_name": pattern_name,
                    "total_trades": len(pattern_data["results"]),
                    "avg_performance": pattern_data["avg_performance"],
                    "recent_performance": (
                        sum(recent_results) / len(recent_results)
                        if recent_results
                        else 0
                    ),
                    "win_rate": (
                        len([r for r in recent_results if r > 0]) / len(recent_results)
                        if recent_results
                        else 0
                    ),
                    "confidence_score": self.pattern_learner.get_pattern_confidence(
                        pattern_name
                    ),
                }

            # Return all patterns summary
            all_patterns = {}
            for pattern, data in self.pattern_learner.patterns.items():
                recent_results = data["results"][-10:]
                all_patterns[pattern] = {
                    "total_trades": len(data["results"]),
                    "avg_performance": data["avg_performance"],
                    "recent_performance": (
                        sum(recent_results) / len(recent_results)
                        if recent_results
                        else 0
                    ),
                    "win_rate": (
                        len([r for r in recent_results if r > 0]) / len(recent_results)
                        if recent_results
                        else 0
                    ),
                    "confidence_score": self.pattern_learner.get_pattern_confidence(
                        pattern
                    ),
                }

            return {
                "patterns": all_patterns,
                "total_patterns": len(all_patterns),
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting pattern performance: {e}")
            return {"error": str(e)}

    async def optimize_strategy_parameters(
        self,
        strategy_name: str,
        symbol: str,
        parameter_ranges: Dict[str, tuple],
        optimization_metric: str = "sharpe_ratio",
        max_iterations: int = 50,
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using backtesting"""
        try:
            logger.info(
                f"Starting parameter optimization for {strategy_name} on {symbol}"
            )

            # Generate parameter combinations
            param_combinations = await self._generate_parameter_combinations(
                parameter_ranges, max_iterations
            )

            best_result = None
            best_params = None
            best_score = float("-inf")
            all_results = []

            for i, params in enumerate(param_combinations):
                try:
                    # Run backtest with these parameters
                    config = BacktestConfig()
                    config.strategy_params = params

                    # Create temporary backtest engine with these params
                    BacktestEngine(config)

                    # Simulate backtest (placeholder - would use real backtesting)
                    result = await self._simulate_parameter_backtest(
                        strategy_name, symbol, params
                    )

                    if result:
                        score = result.get(optimization_metric, 0)
                        all_results.append(
                            {"parameters": params, "score": score, "metrics": result}
                        )

                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_result = result

                    # Update progress
                    progress = (i + 1) / len(param_combinations) * 100
                    logger.info(f"Optimization progress: {progress:.1f}%")

                except Exception as e:
                    logger.warning(f"Error in parameter combination {i}: {e}")
                    continue

            if not best_result:
                return {"error": "No valid optimization results found"}

            # Calculate improvement
            improvement = await self._calculate_improvement_trend(all_results)

            optimization_result = {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "optimization_metric": optimization_metric,
                "best_parameters": best_params,
                "best_score": best_score,
                "best_result": best_result,
                "total_combinations_tested": len(all_results),
                "improvement_percentage": improvement,
                "top_5_results": sorted(
                    all_results, key=lambda x: x["score"], reverse=True
                )[:5],
                "completed_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                f"Optimization completed. Best {optimization_metric}: {best_score}"
            )
            return optimization_result

        except Exception as e:
            logger.error(f"Error in strategy optimization: {e}")
            return {"error": str(e)}

    async def get_agent_learning_summary(
        self, agent_id: str = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Get learning and performance summary for trading agents"""
        try:
            if agent_id and agent_id in self.agent_performance:
                performance_history = self.agent_performance[agent_id]
                recent_performance = (
                    performance_history[-10:]
                    if len(performance_history) >= 10
                    else performance_history
                )

                avg_performance = (
                    sum(performance_history) / len(performance_history)
                    if performance_history
                    else 0
                )
                recent_avg = (
                    sum(recent_performance) / len(recent_performance)
                    if recent_performance
                    else 0
                )

                # Calculate trend
                trend = (
                    "improving"
                    if recent_avg > avg_performance
                    else "declining" if recent_avg < avg_performance else "stable"
                )

                return {
                    "agent_id": agent_id,
                    "total_ideas": len(performance_history),
                    "avg_performance": round(avg_performance, 4),
                    "recent_performance": round(recent_avg, 4),
                    "performance_trend": trend,
                    "win_rate": (
                        len([p for p in recent_performance if p > 0])
                        / len(recent_performance)
                        if recent_performance
                        else 0
                    ),
                    "generated_at": datetime.utcnow().isoformat(),
                }

            # Return summary for all agents
            all_agents_summary = {}
            for agent_id, performance_history in self.agent_performance.items():
                recent_performance = (
                    performance_history[-10:]
                    if len(performance_history) >= 10
                    else performance_history
                )
                avg_performance = (
                    sum(performance_history) / len(performance_history)
                    if performance_history
                    else 0
                )
                recent_avg = (
                    sum(recent_performance) / len(recent_performance)
                    if recent_performance
                    else 0
                )
                trend = (
                    "improving"
                    if recent_avg > avg_performance
                    else "declining" if recent_avg < avg_performance else "stable"
                )

                all_agents_summary[agent_id] = {
                    "total_ideas": len(performance_history),
                    "avg_performance": round(avg_performance, 4),
                    "recent_performance": round(recent_avg, 4),
                    "performance_trend": trend,
                    "win_rate": (
                        len([p for p in recent_performance if p > 0])
                        / len(recent_performance)
                        if recent_performance
                        else 0
                    ),
                }

            return {
                "agents": all_agents_summary,
                "total_agents": len(all_agents_summary),
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting agent learning summary: {e}")
            return {"error": str(e)}

    # Private helper methods
    async def _prepare_market_data_for_idea(
        self, symbol: str, timeframe: str
    ) -> Optional[Dict]:
        """Prepare market data for backtesting"""
        try:
            # TODO: Implement actual market data fetching
            # This is a placeholder that would connect to real data sources
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "data_points": 100,  # Placeholder
                "price_range": {"high": 100, "low": 90},  # Placeholder
                "volume_avg": 1000000,  # Placeholder
            }
        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            return None

    async def _execute_idea_backtest(
        self, idea: TradingIdea, market_data: Dict
    ) -> Dict[str, Any]:
        """Execute backtest for a trading idea"""
        try:
            # TODO: Implement actual backtesting logic
            # This is a placeholder that would run the idea through the backtest engine

            # Simulate backtest results
            import random

            performance = random.uniform(-0.1, 0.15)  # -10% to +15%
            win_rate = random.uniform(0.4, 0.8)  # 40% to 80%
            max_drawdown = random.uniform(0.05, 0.2)  # 5% to 20%

            return {
                "total_return": performance,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": performance / max_drawdown if max_drawdown > 0 else 0,
                "total_trades": random.randint(10, 50),
                "avg_trade_duration": random.uniform(1, 10),  # days
                "indicators_used": [idea.strategy],
                "market_conditions_tested": ["normal", "volatile"],
                "backtest_period": "30 days",
            }

        except Exception as e:
            logger.error(f"Error executing backtest: {e}")
            return {"error": str(e)}

    async def _generate_agent_feedback(
        self, idea: TradingIdea, backtest_results: Dict[str, Any]
    ) -> Optional[BacktestFeedback]:
        """Generate feedback for the trading agent"""
        try:
            performance_score = backtest_results.get("total_return", 0)
            win_rate = backtest_results.get("win_rate", 0)
            avg_return = backtest_results.get("total_return", 0)
            max_drawdown = backtest_results.get("max_drawdown", 0)

            suggestions = []

            # Generate suggestions based on results
            if win_rate < 0.5:
                suggestions.append(
                    "Consider tightening entry criteria to improve win rate"
                )
            if max_drawdown > 0.15:
                suggestions.append(
                    "Risk management could be improved - consider smaller position sizes"
                )
            if performance_score > 0.1:
                suggestions.append("Good performance - consider scaling this strategy")

            return BacktestFeedback(
                idea_id=f"{idea.symbol}_{idea.strategy}_{int(idea.timestamp.timestamp())}",
                performance_score=performance_score,
                win_rate=win_rate,
                avg_return=avg_return,
                max_drawdown=max_drawdown,
                suggestions=suggestions,
            )

        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            return None

    async def _update_agent_performance(
        self, agent_id: str, backtest_results: Dict[str, Any]
    ):
        """Update agent performance tracking"""
        try:
            if agent_id not in self.agent_performance:
                self.agent_performance[agent_id] = []

            performance_score = backtest_results.get("total_return", 0)
            self.agent_performance[agent_id].append(performance_score)

            # Keep only last 100 results to manage memory
            if len(self.agent_performance[agent_id]) > 100:
                self.agent_performance[agent_id] = self.agent_performance[agent_id][
                    -100:
                ]

        except Exception as e:
            logger.error(f"Error updating agent performance: {e}")

    async def _update_pattern_learning(
        self, strategy: str, backtest_results: Dict[str, Any]
    ):
        """Update pattern learning with backtest results"""
        try:
            performance_score = backtest_results.get("total_return", 0)
            self.pattern_learner.learn_from_result(strategy, performance_score)

        except Exception as e:
            logger.error(f"Error updating pattern learning: {e}")

    async def _generate_parameter_combinations(
        self, parameter_ranges: Dict[str, tuple], max_combinations: int
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations for optimization"""
        try:
            import itertools
            import random

            # Generate all possible combinations
            param_names = list(parameter_ranges.keys())
            param_values = []

            for param_name, (min_val, max_val) in parameter_ranges.items():
                # Generate reasonable number of test values
                if isinstance(min_val, int) and isinstance(max_val, int):
                    step = max(1, (max_val - min_val) // 10)
                    values = list(range(min_val, max_val + 1, step))
                else:
                    values = [min_val + (max_val - min_val) * i / 9 for i in range(10)]

                param_values.append(values)

            # Generate combinations
            all_combinations = list(itertools.product(*param_values))

            # Randomly sample if too many combinations
            if len(all_combinations) > max_combinations:
                selected_combinations = random.sample(
                    all_combinations, max_combinations
                )
            else:
                selected_combinations = all_combinations

            # Convert to dictionaries
            param_dicts = []
            for combo in selected_combinations:
                param_dict = {}
                for i, param_name in enumerate(param_names):
                    param_dict[param_name] = combo[i]
                param_dicts.append(param_dict)

            return param_dicts

        except Exception as e:
            logger.error(f"Error generating parameter combinations: {e}")
            return []

    async def _calculate_improvement_trend(self, results: List[Dict]) -> float:
        """Calculate improvement trend from optimization results"""
        try:
            if len(results) < 2:
                return 0.0

            scores = [r["score"] for r in results]
            best_score = max(scores)
            avg_score = sum(scores) / len(scores)

            improvement = (
                ((best_score - avg_score) / abs(avg_score)) * 100
                if avg_score != 0
                else 0
            )
            return round(improvement, 2)

        except Exception as e:
            logger.error(f"Error calculating improvement trend: {e}")
            return 0.0

    async def _simulate_parameter_backtest(
        self, strategy_name: str, symbol: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Simulate backtest with specific parameters"""
        try:
            # TODO: Replace with actual backtesting logic
            # This is a placeholder simulation
            import random

            base_return = random.uniform(-0.05, 0.12)
            volatility = random.uniform(0.1, 0.3)

            # Simulate how parameters might affect performance
            param_effect = (
                sum(params.values()) * 0.001
                if all(isinstance(v, (int, float)) for v in params.values())
                else 0
            )

            return {
                "total_return": base_return + param_effect,
                "sharpe_ratio": (base_return + param_effect) / volatility,
                "max_drawdown": volatility * random.uniform(0.5, 1.2),
                "win_rate": random.uniform(0.45, 0.75),
                "total_trades": random.randint(20, 100),
                "parameters_used": params,
            }

        except Exception as e:
            logger.error(f"Error simulating parameter backtest: {e}")
            return None
