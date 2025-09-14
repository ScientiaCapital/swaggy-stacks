"""
AI Agent Backtesting Integration Service
Provides interface for AI agents to submit trading ideas, learn from results, and optimize strategies
"""

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog
from sqlalchemy.orm import Session

from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.models.indicator_performance import MLModelVersion
from app.monitoring.metrics import PrometheusMetrics
from app.services.ml_training_pipeline import MLTrainingPipeline

logger = structlog.get_logger()


@dataclass
class TradingIdea:
    """Trading idea submitted by AI agents"""

    id: str
    agent_id: str
    symbol: str
    strategy: str
    direction: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    risk_parameters: Dict[str, Any]
    timeframe: str
    expected_duration: Optional[str] = None
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    submitted_at: datetime = None

    def __post_init__(self):
        if self.submitted_at is None:
            self.submitted_at = datetime.now()


@dataclass
class BacktestFeedback:
    """Structured feedback for AI agents"""

    idea_id: str
    agent_id: str
    backtest_id: str
    success: bool
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades_count: int
    risk_score: float
    performance_rank: str  # excellent, good, average, poor
    improvement_suggestions: List[str]
    learned_patterns: List[Dict[str, Any]]
    optimized_parameters: Dict[str, Any]
    market_context: Dict[str, Any]
    execution_time: float
    feedback_timestamp: datetime = None

    def __post_init__(self):
        if self.feedback_timestamp is None:
            self.feedback_timestamp = datetime.now()


class PatternLearner:
    """ML-based pattern performance learning for AI agents"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_samples = self.config.get("min_samples", 10)
        self.performance_threshold = self.config.get("performance_threshold", 0.15)
        self.learning_rate = self.config.get("learning_rate", 0.01)

        # Pattern performance tracking
        self.pattern_performance = {}
        self.market_context_weights = {}

    def update_pattern_performance(
        self,
        pattern_name: str,
        market_conditions: Dict[str, Any],
        performance_metrics: Dict[str, float],
    ):
        """Update pattern performance based on backtest results"""
        try:
            if pattern_name not in self.pattern_performance:
                self.pattern_performance[pattern_name] = {
                    "total_occurrences": 0,
                    "successful_trades": 0,
                    "total_return": 0.0,
                    "market_contexts": [],
                    "performance_history": [],
                }

            pattern_data = self.pattern_performance[pattern_name]
            pattern_data["total_occurrences"] += 1

            # Track performance
            return_value = performance_metrics.get("return", 0.0)
            pattern_data["total_return"] += return_value

            if return_value > 0:
                pattern_data["successful_trades"] += 1

            # Store context and performance
            pattern_data["market_contexts"].append(market_conditions)
            pattern_data["performance_history"].append(
                {
                    "return": return_value,
                    "sharpe": performance_metrics.get("sharpe_ratio", 0.0),
                    "drawdown": performance_metrics.get("max_drawdown", 0.0),
                    "timestamp": datetime.now().isoformat(),
                    "market_context": market_conditions,
                }
            )

            # Keep only recent history (last 100 samples)
            if len(pattern_data["performance_history"]) > 100:
                pattern_data["performance_history"] = pattern_data[
                    "performance_history"
                ][-100:]

            logger.info(
                "Pattern performance updated",
                pattern=pattern_name,
                occurrences=pattern_data["total_occurrences"],
                success_rate=pattern_data["successful_trades"]
                / pattern_data["total_occurrences"],
            )

        except Exception as e:
            logger.error(
                "Error updating pattern performance", error=str(e), pattern=pattern_name
            )

    def get_pattern_success_rate(
        self, pattern_name: str, market_conditions: Dict[str, Any] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Get pattern success rate, optionally filtered by market conditions"""
        try:
            if pattern_name not in self.pattern_performance:
                return 0.5, {"reason": "no_historical_data"}

            pattern_data = self.pattern_performance[pattern_name]

            if pattern_data["total_occurrences"] < self.min_samples:
                return 0.5, {
                    "reason": "insufficient_samples",
                    "samples": pattern_data["total_occurrences"],
                }

            # If no market conditions specified, return overall success rate
            if not market_conditions:
                success_rate = (
                    pattern_data["successful_trades"]
                    / pattern_data["total_occurrences"]
                )
                avg_return = (
                    pattern_data["total_return"] / pattern_data["total_occurrences"]
                )

                return success_rate, {
                    "overall_success_rate": success_rate,
                    "average_return": avg_return,
                    "total_occurrences": pattern_data["total_occurrences"],
                }

            # Filter by market conditions
            filtered_history = []
            for record in pattern_data["performance_history"]:
                context_match = self._calculate_context_similarity(
                    market_conditions, record["market_context"]
                )
                if context_match > 0.7:  # 70% similarity threshold
                    filtered_history.append(record)

            if len(filtered_history) < 3:  # Need at least 3 similar contexts
                success_rate = (
                    pattern_data["successful_trades"]
                    / pattern_data["total_occurrences"]
                )
                return success_rate, {"reason": "insufficient_similar_contexts"}

            # Calculate context-specific performance
            successful = sum(1 for record in filtered_history if record["return"] > 0)
            success_rate = successful / len(filtered_history)
            avg_return = np.mean([record["return"] for record in filtered_history])

            return success_rate, {
                "context_specific_success_rate": success_rate,
                "context_specific_return": avg_return,
                "similar_contexts_count": len(filtered_history),
            }

        except Exception as e:
            logger.error("Error calculating pattern success rate", error=str(e))
            return 0.5, {"error": str(e)}

    def _calculate_context_similarity(
        self, context1: Dict[str, Any], context2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between market contexts"""
        try:
            if not context1 or not context2:
                return 0.0

            # Key context factors to compare
            factors = ["volatility", "trend", "volume", "sector_performance"]
            similarity_scores = []

            for factor in factors:
                if factor in context1 and factor in context2:
                    val1 = context1[factor]
                    val2 = context2[factor]

                    if isinstance(val1, (int, float)) and isinstance(
                        val2, (int, float)
                    ):
                        # Numerical similarity
                        max_val = max(
                            abs(val1), abs(val2), 0.001
                        )  # Avoid division by zero
                        similarity = 1 - abs(val1 - val2) / max_val
                        similarity_scores.append(max(0, similarity))
                    elif val1 == val2:
                        # Exact match for categorical values
                        similarity_scores.append(1.0)
                    else:
                        similarity_scores.append(0.0)

            return np.mean(similarity_scores) if similarity_scores else 0.0

        except Exception as e:
            logger.error("Error calculating context similarity", error=str(e))
            return 0.0

    def suggest_pattern_improvements(
        self, pattern_name: str, recent_performance: Dict[str, float]
    ) -> List[str]:
        """Suggest improvements based on pattern performance analysis"""
        suggestions = []

        try:
            if pattern_name not in self.pattern_performance:
                suggestions.append("Collect more historical data for this pattern")
                return suggestions

            pattern_data = self.pattern_performance[pattern_name]

            # Analyze recent performance vs historical
            if pattern_data["total_occurrences"] >= self.min_samples:
                historical_avg = (
                    pattern_data["total_return"] / pattern_data["total_occurrences"]
                )
                recent_return = recent_performance.get("return", 0.0)

                if recent_return < historical_avg * 0.5:
                    suggestions.append(
                        f"Pattern underperforming: consider tightening entry conditions"
                    )
                    suggestions.append(
                        f"Review market context - pattern may work better in different conditions"
                    )

                if recent_performance.get("max_drawdown", 0.0) > 0.15:
                    suggestions.append(
                        "High drawdown detected: consider smaller position sizes"
                    )
                    suggestions.append("Implement tighter stop losses for this pattern")

                if recent_performance.get("win_rate", 0.5) < 0.4:
                    suggestions.append(
                        "Low win rate: pattern recognition may need refinement"
                    )

            # General improvement suggestions
            success_rate = pattern_data["successful_trades"] / max(
                pattern_data["total_occurrences"], 1
            )
            if success_rate < 0.6:
                suggestions.append(
                    "Consider combining with other confirming indicators"
                )
                suggestions.append(
                    "Analyze failed trades to identify common characteristics"
                )

        except Exception as e:
            logger.error(
                "Error generating pattern improvement suggestions", error=str(e)
            )
            suggestions.append("Error analyzing pattern - manual review recommended")

        return suggestions[:5]  # Limit to top 5 suggestions


@dataclass
class IndicatorPerformance:
    """Performance metrics for a specific indicator"""

    indicator_name: str
    indicator_type: str  # TRADITIONAL, MODERN, LLM
    total_signals: int = 0
    correct_signals: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_signal_strength: float = 0.0
    market_condition_performance: Dict[str, Dict[str, float]] = None
    parameter_performance: Dict[str, Dict[str, float]] = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.market_condition_performance is None:
            self.market_condition_performance = {}
        if self.parameter_performance is None:
            self.parameter_performance = {}
        if self.last_updated is None:
            self.last_updated = datetime.now()


class IndicatorPerformanceTracker:
    """
    Performance tracking system for technical indicators and ML predictions
    Tracks win rates, Sharpe ratios, and market condition effectiveness
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_samples = self.config.get("min_samples", 10)
        self.performance_threshold = self.config.get("performance_threshold", 0.15)
        self.lookback_days = self.config.get("lookback_days", 90)

        # Performance tracking storage
        self.indicator_performance = {}  # {indicator_name: IndicatorPerformance}
        self.signal_history = []  # List of signal records for analysis

        # Market condition categorization
        self.market_conditions_map = {
            "high_vol": {"volatility": (0.25, float("inf"))},
            "low_vol": {"volatility": (0.0, 0.15)},
            "trending_up": {"trend": "UP", "volatility": (0.0, 0.30)},
            "trending_down": {"trend": "DOWN", "volatility": (0.0, 0.30)},
            "sideways": {"trend": "NEUTRAL", "volatility": (0.0, 0.20)},
            "volatile": {"volatility": (0.30, float("inf"))},
        }

        logger.info("IndicatorPerformanceTracker initialized")

    def record_indicator_signal(
        self,
        indicator_name: str,
        indicator_type: str,
        signal_data: Dict[str, Any],
        market_conditions: Dict[str, Any],
        actual_outcome: Dict[str, Any] = None,
    ):
        """Record an indicator signal and its performance"""
        try:
            signal_record = {
                "indicator_name": indicator_name,
                "indicator_type": indicator_type,
                "signal_time": datetime.now(),
                "signal_strength": signal_data.get("signal_strength", 0.0),
                "signal_direction": signal_data.get("signal_direction", "NEUTRAL"),
                "confidence": signal_data.get("confidence", 0.0),
                "market_conditions": market_conditions,
                "entry_price": signal_data.get("entry_price", 0.0),
                "actual_outcome": actual_outcome,
                "signal_id": str(uuid.uuid4()),
            }

            self.signal_history.append(signal_record)

            # Update indicator performance if outcome is provided
            if actual_outcome:
                self._update_indicator_performance(signal_record)

            logger.debug(
                "Indicator signal recorded",
                indicator=indicator_name,
                signal_direction=signal_record["signal_direction"],
                confidence=signal_record["confidence"],
            )

        except Exception as e:
            logger.error(f"Error recording indicator signal: {e}")

    def update_signal_outcome(self, signal_id: str, actual_outcome: Dict[str, Any]):
        """Update the actual outcome for a previously recorded signal"""
        try:
            # Find and update signal record
            for signal_record in self.signal_history:
                if signal_record.get("signal_id") == signal_id:
                    signal_record["actual_outcome"] = actual_outcome
                    self._update_indicator_performance(signal_record)
                    logger.debug(f"Signal outcome updated for {signal_id}")
                    return

            logger.warning(f"Signal ID {signal_id} not found for outcome update")

        except Exception as e:
            logger.error(f"Error updating signal outcome: {e}")

    def _update_indicator_performance(self, signal_record: Dict[str, Any]):
        """Update performance metrics for an indicator based on signal outcome"""
        try:
            indicator_name = signal_record["indicator_name"]
            indicator_type = signal_record["indicator_type"]
            actual_outcome = signal_record.get("actual_outcome", {})

            if not actual_outcome:
                return

            # Initialize performance tracking for indicator
            if indicator_name not in self.indicator_performance:
                self.indicator_performance[indicator_name] = IndicatorPerformance(
                    indicator_name=indicator_name, indicator_type=indicator_type
                )

            perf = self.indicator_performance[indicator_name]

            # Update basic metrics
            perf.total_signals += 1
            perf.avg_signal_strength = (
                perf.avg_signal_strength * (perf.total_signals - 1)
                + signal_record.get("signal_strength", 0.0)
            ) / perf.total_signals

            # Determine if signal was correct
            signal_direction = signal_record.get("signal_direction", "NEUTRAL")
            actual_return = actual_outcome.get("return", 0.0)

            signal_correct = False
            if signal_direction == "BULLISH" and actual_return > 0.01:
                signal_correct = True
            elif signal_direction == "BEARISH" and actual_return < -0.01:
                signal_correct = True
            elif signal_direction == "NEUTRAL" and abs(actual_return) <= 0.01:
                signal_correct = True

            if signal_correct:
                perf.correct_signals += 1

            # Update performance metrics
            perf.win_rate = perf.correct_signals / perf.total_signals

            # Update cumulative return
            perf.total_return += actual_return

            # Update Sharpe ratio (simplified calculation)
            if perf.total_signals >= 5:
                returns = [
                    r.get("actual_outcome", {}).get("return", 0.0)
                    for r in self.signal_history
                    if r.get("indicator_name") == indicator_name
                    and r.get("actual_outcome")
                ]

                if returns:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    perf.sharpe_ratio = (
                        mean_return / max(std_return, 0.001) * np.sqrt(252)
                    )  # Annualized

            # Update max drawdown
            drawdown = actual_outcome.get("drawdown", 0.0)
            if drawdown > perf.max_drawdown:
                perf.max_drawdown = drawdown

            # Update market condition performance
            market_condition = self._classify_market_condition(
                signal_record.get("market_conditions", {})
            )
            if market_condition not in perf.market_condition_performance:
                perf.market_condition_performance[market_condition] = {
                    "signals": 0,
                    "correct": 0,
                    "win_rate": 0.0,
                    "avg_return": 0.0,
                }

            mc_perf = perf.market_condition_performance[market_condition]
            mc_perf["signals"] += 1
            if signal_correct:
                mc_perf["correct"] += 1
            mc_perf["win_rate"] = mc_perf["correct"] / mc_perf["signals"]
            mc_perf["avg_return"] = (
                mc_perf["avg_return"] * (mc_perf["signals"] - 1) + actual_return
            ) / mc_perf["signals"]

            perf.last_updated = datetime.now()

            logger.debug(
                "Indicator performance updated",
                indicator=indicator_name,
                win_rate=perf.win_rate,
                total_signals=perf.total_signals,
                sharpe_ratio=perf.sharpe_ratio,
            )

        except Exception as e:
            logger.error(f"Error updating indicator performance: {e}")

    def _classify_market_condition(self, market_conditions: Dict[str, Any]) -> str:
        """Classify market conditions into predefined categories"""
        try:
            volatility = market_conditions.get("volatility", 0.2)
            trend = market_conditions.get("trend", "NEUTRAL")

            # Apply classification rules
            if volatility > 0.30:
                return "volatile"
            elif volatility > 0.25:
                return "high_vol"
            elif volatility < 0.15:
                return "low_vol"
            elif trend == "UP" and volatility <= 0.30:
                return "trending_up"
            elif trend == "DOWN" and volatility <= 0.30:
                return "trending_down"
            elif trend == "NEUTRAL" and volatility <= 0.20:
                return "sideways"
            else:
                return "normal"

        except Exception as e:
            logger.error(f"Error classifying market condition: {e}")
            return "unknown"

    def get_indicator_performance(
        self,
        indicator_name: str = None,
        indicator_type: str = None,
        market_condition: str = None,
    ) -> Union[Dict[str, IndicatorPerformance], IndicatorPerformance]:
        """Get performance metrics for indicators"""
        try:
            # Filter indicators based on criteria
            if indicator_name:
                # Return specific indicator performance
                if indicator_name in self.indicator_performance:
                    perf = self.indicator_performance[indicator_name]
                    if (
                        market_condition
                        and market_condition in perf.market_condition_performance
                    ):
                        # Return market-condition-specific performance
                        mc_perf = perf.market_condition_performance[market_condition]
                        return {
                            "indicator_name": indicator_name,
                            "market_condition": market_condition,
                            "performance": mc_perf,
                            "overall_performance": asdict(perf),
                        }
                    return perf
                else:
                    return IndicatorPerformance(
                        indicator_name, indicator_type or "UNKNOWN"
                    )

            # Return all indicators filtered by type
            filtered_performance = {}
            for name, perf in self.indicator_performance.items():
                if not indicator_type or perf.indicator_type == indicator_type:
                    filtered_performance[name] = perf

            return filtered_performance

        except Exception as e:
            logger.error(f"Error getting indicator performance: {e}")
            return {}

    def get_top_performing_indicators(
        self, metric: str = "sharpe_ratio", top_n: int = 10, min_signals: int = None
    ) -> List[Dict[str, Any]]:
        """Get top performing indicators ranked by specified metric"""
        try:
            min_signals = min_signals or self.min_samples

            # Filter indicators with sufficient data
            candidates = [
                (name, perf)
                for name, perf in self.indicator_performance.items()
                if perf.total_signals >= min_signals
            ]

            if not candidates:
                return []

            # Sort by metric
            metric_map = {
                "sharpe_ratio": lambda p: p.sharpe_ratio,
                "win_rate": lambda p: p.win_rate,
                "total_return": lambda p: p.total_return,
                "signal_strength": lambda p: p.avg_signal_strength,
            }

            if metric not in metric_map:
                metric = "sharpe_ratio"

            sorted_indicators = sorted(
                candidates, key=lambda x: metric_map[metric](x[1]), reverse=True
            )[:top_n]

            # Format results
            top_indicators = []
            for name, perf in sorted_indicators:
                top_indicators.append(
                    {
                        "indicator_name": name,
                        "indicator_type": perf.indicator_type,
                        "rank_metric": metric,
                        "rank_value": metric_map[metric](perf),
                        "performance_summary": {
                            "win_rate": perf.win_rate,
                            "total_signals": perf.total_signals,
                            "sharpe_ratio": perf.sharpe_ratio,
                            "total_return": perf.total_return,
                            "avg_signal_strength": perf.avg_signal_strength,
                            "max_drawdown": perf.max_drawdown,
                        },
                        "market_condition_performance": perf.market_condition_performance,
                    }
                )

            return top_indicators

        except Exception as e:
            logger.error(f"Error getting top performing indicators: {e}")
            return []

    def generate_performance_report(self, lookback_days: int = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            lookback_days = lookback_days or self.lookback_days
            cutoff_date = datetime.now() - timedelta(days=lookback_days)

            # Filter recent signals
            recent_signals = [
                s
                for s in self.signal_history
                if s.get("signal_time", datetime.min) > cutoff_date
            ]

            # Overall statistics
            total_signals = len(recent_signals)
            signals_with_outcomes = len(
                [s for s in recent_signals if s.get("actual_outcome")]
            )

            # Performance by indicator type
            type_performance = {}
            for signal in recent_signals:
                if not signal.get("actual_outcome"):
                    continue

                indicator_type = signal.get("indicator_type", "UNKNOWN")
                if indicator_type not in type_performance:
                    type_performance[indicator_type] = {
                        "signals": 0,
                        "correct": 0,
                        "returns": [],
                    }

                type_perf = type_performance[indicator_type]
                type_perf["signals"] += 1

                # Check if signal was correct
                signal_direction = signal.get("signal_direction", "NEUTRAL")
                actual_return = signal.get("actual_outcome", {}).get("return", 0.0)

                if (
                    (signal_direction == "BULLISH" and actual_return > 0.01)
                    or (signal_direction == "BEARISH" and actual_return < -0.01)
                    or (signal_direction == "NEUTRAL" and abs(actual_return) <= 0.01)
                ):
                    type_perf["correct"] += 1

                type_perf["returns"].append(actual_return)

            # Calculate type-level metrics
            for indicator_type, perf in type_performance.items():
                perf["win_rate"] = perf["correct"] / max(perf["signals"], 1)
                perf["avg_return"] = (
                    np.mean(perf["returns"]) if perf["returns"] else 0.0
                )
                perf["sharpe_ratio"] = (
                    np.mean(perf["returns"])
                    / max(np.std(perf["returns"]), 0.001)
                    * np.sqrt(252)
                    if perf["returns"]
                    else 0.0
                )

            # Top performers
            top_performers = self.get_top_performing_indicators(top_n=5)

            report = {
                "report_period": f"Last {lookback_days} days",
                "generation_time": datetime.now().isoformat(),
                "summary_statistics": {
                    "total_signals_generated": total_signals,
                    "signals_with_outcomes": signals_with_outcomes,
                    "outcome_completion_rate": signals_with_outcomes
                    / max(total_signals, 1),
                    "total_indicators_tracked": len(self.indicator_performance),
                },
                "performance_by_type": type_performance,
                "top_performing_indicators": top_performers,
                "indicator_count_by_type": {
                    indicator_type: len(
                        [
                            perf
                            for perf in self.indicator_performance.values()
                            if perf.indicator_type == indicator_type
                        ]
                    )
                    for indicator_type in ["TRADITIONAL", "MODERN", "LLM"]
                },
                "recommendations": self._generate_performance_recommendations(),
            }

            return report

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e), "generation_time": datetime.now().isoformat()}

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance-based recommendations"""
        recommendations = []

        try:
            # Analyze overall performance patterns
            if len(self.indicator_performance) == 0:
                recommendations.append(
                    "No indicator performance data available - begin tracking indicators"
                )
                return recommendations

            # Calculate average win rates by type
            type_win_rates = {}
            for perf in self.indicator_performance.values():
                if perf.total_signals >= self.min_samples:
                    if perf.indicator_type not in type_win_rates:
                        type_win_rates[perf.indicator_type] = []
                    type_win_rates[perf.indicator_type].append(perf.win_rate)

            # Generate recommendations based on patterns
            for indicator_type, win_rates in type_win_rates.items():
                avg_win_rate = np.mean(win_rates)
                if avg_win_rate < 0.45:
                    recommendations.append(
                        f"{indicator_type} indicators underperforming - review parameters"
                    )
                elif avg_win_rate > 0.65:
                    recommendations.append(
                        f"{indicator_type} indicators performing well - consider increasing allocation"
                    )

            # Market condition recommendations
            market_condition_performance = {}
            for perf in self.indicator_performance.values():
                for condition, mc_perf in perf.market_condition_performance.items():
                    if condition not in market_condition_performance:
                        market_condition_performance[condition] = []
                    if mc_perf["signals"] >= 3:  # Minimum for meaningful analysis
                        market_condition_performance[condition].append(
                            mc_perf["win_rate"]
                        )

            for condition, win_rates in market_condition_performance.items():
                if win_rates:
                    avg_wr = np.mean(win_rates)
                    if avg_wr < 0.40:
                        recommendations.append(
                            f"Poor performance in {condition} market - avoid or adjust parameters"
                        )
                    elif avg_wr > 0.70:
                        recommendations.append(
                            f"Excellent performance in {condition} market - prioritize these conditions"
                        )

            # General recommendations
            if not recommendations:
                recommendations.append(
                    "Performance tracking is healthy - continue monitoring"
                )

        except Exception as e:
            logger.error(f"Error generating performance recommendations: {e}")
            recommendations.append(
                "Error generating recommendations - manual review advised"
            )

        return recommendations[:5]  # Limit to top 5 recommendations


class BacktestService:
    """AI Agent Backtesting Integration Service"""

    def __init__(self, db: Session = None):
        self.backtest_engine = BacktestEngine()
        self.pattern_learner = PatternLearner()
        self.metrics = PrometheusMetrics()

        # Initialize indicator performance tracking
        self.indicator_tracker = IndicatorPerformanceTracker()

        # Initialize ML pipeline if database session provided
        self.db = db
        self.ml_pipeline = MLTrainingPipeline(db) if db else None

        # Track active trading ideas and results
        self.trading_ideas = {}
        self.agent_performance = {}

        logger.info("BacktestService initialized with ML prediction capabilities")

    async def submit_trading_idea(
        self, agent_id: str, trading_idea: TradingIdea, db: Session = None
    ) -> Dict[str, Any]:
        """Submit trading idea from AI agent for backtesting"""
        try:
            start_time = datetime.now()
            idea_id = str(uuid.uuid4())
            trading_idea.id = idea_id
            trading_idea.agent_id = agent_id

            # Store the trading idea
            self.trading_ideas[idea_id] = trading_idea

            # Prepare backtest configuration
            backtest_config = BacktestConfig(
                initial_capital=100000,
                max_position_size=0.15,
                transaction_cost=0.001,
                years_lookback=1,  # Focus on recent performance
            )

            # Create market data for backtesting (in production, fetch real data)
            mock_market_data = await self._prepare_market_data_for_idea(trading_idea)

            # Execute backtest
            backtest_results = await self._execute_idea_backtest(
                trading_idea, mock_market_data, backtest_config
            )

            # Generate structured feedback
            feedback = await self._generate_agent_feedback(
                trading_idea, backtest_results
            )

            # Update agent performance tracking
            self._update_agent_performance(agent_id, feedback)

            # Update pattern learning
            await self._update_pattern_learning(trading_idea, backtest_results)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Record metrics
            self.metrics.record_ai_agent_backtest_submission(
                execution_time, agent_id, trading_idea.strategy
            )

            logger.info(
                "Trading idea submitted and backtested",
                idea_id=idea_id,
                agent_id=agent_id,
                strategy=trading_idea.strategy,
                performance_rank=feedback.performance_rank,
            )

            return {
                "idea_id": idea_id,
                "backtest_id": backtest_results.get("backtest_id"),
                "status": "completed",
                "feedback": asdict(feedback),
                "execution_time": execution_time,
            }

        except Exception as e:
            logger.error(
                "Error processing trading idea submission",
                error=str(e),
                agent_id=agent_id,
            )
            return {
                "status": "error",
                "error": str(e),
                "idea_id": getattr(trading_idea, "id", None),
            }

    async def get_pattern_performance(
        self,
        pattern_name: str,
        market_conditions: Dict[str, Any] = None,
        agent_id: str = None,
    ) -> Dict[str, Any]:
        """Get historical pattern performance for AI agent learning"""
        try:
            # Get pattern success rate
            success_rate, details = self.pattern_learner.get_pattern_success_rate(
                pattern_name, market_conditions
            )

            # Get pattern-specific suggestions
            recent_performance = {"return": 0.0, "win_rate": success_rate}
            suggestions = self.pattern_learner.suggest_pattern_improvements(
                pattern_name, recent_performance
            )

            # Compile pattern performance data
            performance_data = {
                "pattern_name": pattern_name,
                "success_rate": success_rate,
                "details": details,
                "improvement_suggestions": suggestions,
                "market_conditions": market_conditions,
                "timestamp": datetime.now().isoformat(),
            }

            # Add agent-specific performance if available
            if agent_id and agent_id in self.agent_performance:
                agent_data = self.agent_performance[agent_id]
                agent_pattern_performance = [
                    feedback
                    for feedback in agent_data.get("feedback_history", [])
                    if pattern_name.lower() in feedback.get("strategy", "").lower()
                ]

                if agent_pattern_performance:
                    agent_success = sum(
                        1 for f in agent_pattern_performance if f.get("success", False)
                    )
                    performance_data["agent_specific"] = {
                        "agent_success_rate": agent_success
                        / len(agent_pattern_performance),
                        "agent_attempts": len(agent_pattern_performance),
                        "last_attempt": agent_pattern_performance[-1].get("timestamp"),
                    }

            logger.info(
                "Pattern performance retrieved",
                pattern=pattern_name,
                success_rate=success_rate,
                agent_id=agent_id,
            )

            return performance_data

        except Exception as e:
            logger.error("Error retrieving pattern performance", error=str(e))
            return {
                "pattern_name": pattern_name,
                "error": str(e),
                "success_rate": 0.5,  # Default neutral performance
            }

    async def optimize_strategy_parameters(
        self,
        agent_id: str,
        strategy_name: str,
        parameter_space: Dict[str, Dict[str, Any]],
        optimization_metric: str = "sharpe_ratio",
        max_iterations: int = 50,
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search or Bayesian optimization"""
        try:
            start_time = datetime.now()

            # Generate parameter combinations for grid search
            parameter_combinations = self._generate_parameter_combinations(
                parameter_space, max_iterations
            )

            optimization_results = []
            best_parameters = None
            best_metric_value = float("-inf")

            for i, params in enumerate(parameter_combinations):
                try:
                    # Create trading idea with these parameters
                    test_idea = TradingIdea(
                        id=f"opt_{i}",
                        agent_id=agent_id,
                        symbol="TEST",  # Use test symbol
                        strategy=strategy_name,
                        direction="BUY",
                        confidence=0.7,
                        entry_conditions=params,
                        exit_conditions={},
                        risk_parameters={},
                        timeframe="1D",
                    )

                    # Run backtest with these parameters
                    mock_data = await self._prepare_market_data_for_idea(test_idea)
                    backtest_config = BacktestConfig(
                        initial_capital=100000, years_lookback=1
                    )

                    results = await self._execute_idea_backtest(
                        test_idea, mock_data, backtest_config
                    )

                    # Extract optimization metric
                    metric_value = results.get(optimization_metric, 0.0)

                    optimization_results.append(
                        {
                            "parameters": params,
                            "metric_value": metric_value,
                            "full_results": results,
                        }
                    )

                    # Track best parameters
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_parameters = params

                except Exception as e:
                    logger.warning(f"Optimization iteration {i} failed", error=str(e))
                    continue

            # Sort results by performance
            optimization_results.sort(key=lambda x: x["metric_value"], reverse=True)

            execution_time = (datetime.now() - start_time).total_seconds()

            optimization_summary = {
                "agent_id": agent_id,
                "strategy_name": strategy_name,
                "optimization_metric": optimization_metric,
                "best_parameters": best_parameters,
                "best_metric_value": best_metric_value,
                "parameter_space": parameter_space,
                "total_combinations_tested": len(optimization_results),
                "top_results": optimization_results[:5],  # Top 5 results
                "execution_time": execution_time,
                "optimization_timestamp": datetime.now().isoformat(),
            }

            logger.info(
                "Strategy parameter optimization completed",
                agent_id=agent_id,
                strategy=strategy_name,
                best_metric=best_metric_value,
                combinations_tested=len(optimization_results),
            )

            return optimization_summary

        except Exception as e:
            logger.error("Error in strategy parameter optimization", error=str(e))
            return {
                "agent_id": agent_id,
                "strategy_name": strategy_name,
                "error": str(e),
                "optimization_timestamp": datetime.now().isoformat(),
            }

    async def get_agent_learning_summary(
        self, agent_id: str, days_lookback: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive learning summary for an AI agent"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_lookback)

            if agent_id not in self.agent_performance:
                return {
                    "agent_id": agent_id,
                    "message": "No performance data available",
                    "suggestions": ["Submit trading ideas to begin learning"],
                }

            agent_data = self.agent_performance[agent_id]
            recent_feedback = [
                feedback
                for feedback in agent_data.get("feedback_history", [])
                if datetime.fromisoformat(feedback.get("timestamp", "1970-01-01"))
                > cutoff_date
            ]

            if not recent_feedback:
                return {
                    "agent_id": agent_id,
                    "message": "No recent performance data",
                    "total_historical_ideas": len(
                        agent_data.get("feedback_history", [])
                    ),
                }

            # Calculate performance metrics
            successful_ideas = sum(
                1 for f in recent_feedback if f.get("success", False)
            )
            success_rate = successful_ideas / len(recent_feedback)

            avg_return = np.mean([f.get("total_return", 0.0) for f in recent_feedback])
            avg_sharpe = np.mean([f.get("sharpe_ratio", 0.0) for f in recent_feedback])

            # Strategy performance breakdown
            strategy_performance = {}
            for feedback in recent_feedback:
                strategy = feedback.get("strategy", "unknown")
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(feedback.get("total_return", 0.0))

            strategy_summary = {
                strategy: {
                    "attempts": len(returns),
                    "avg_return": np.mean(returns),
                    "success_rate": sum(1 for r in returns if r > 0) / len(returns),
                }
                for strategy, returns in strategy_performance.items()
            }

            # Learning recommendations
            recommendations = []
            if success_rate < 0.5:
                recommendations.append("Focus on higher-confidence trading ideas")
                recommendations.append("Review failed trades to identify patterns")

            if avg_sharpe < 1.0:
                recommendations.append("Consider risk-adjusted position sizing")
                recommendations.append("Implement tighter stop losses")

            best_strategy = (
                max(
                    strategy_summary.keys(),
                    key=lambda k: strategy_summary[k]["avg_return"],
                )
                if strategy_summary
                else None
            )

            learning_summary = {
                "agent_id": agent_id,
                "period": f"Last {days_lookback} days",
                "total_ideas_submitted": len(recent_feedback),
                "success_rate": success_rate,
                "average_return": avg_return,
                "average_sharpe_ratio": avg_sharpe,
                "strategy_performance": strategy_summary,
                "best_performing_strategy": best_strategy,
                "learning_recommendations": recommendations,
                "improvement_trend": self._calculate_improvement_trend(recent_feedback),
                "summary_timestamp": datetime.now().isoformat(),
            }

            return learning_summary

        except Exception as e:
            logger.error("Error generating agent learning summary", error=str(e))
            return {"agent_id": agent_id, "error": str(e)}

    async def track_indicator_performance(
        self,
        indicator_name: str,
        indicator_type: str,
        signal_data: Dict[str, Any],
        market_conditions: Dict[str, Any],
        actual_outcome: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Track performance of an individual indicator"""
        try:
            # Record the signal
            self.indicator_tracker.record_indicator_signal(
                indicator_name=indicator_name,
                indicator_type=indicator_type,
                signal_data=signal_data,
                market_conditions=market_conditions,
                actual_outcome=actual_outcome,
            )

            # Get updated performance metrics
            performance = self.indicator_tracker.get_indicator_performance(
                indicator_name
            )

            return {
                "status": "success",
                "indicator_name": indicator_name,
                "performance_metrics": (
                    asdict(performance)
                    if hasattr(performance, "__dict__")
                    else performance
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error tracking indicator performance: {e}")
            return {
                "status": "error",
                "error": str(e),
                "indicator_name": indicator_name,
            }

    async def get_indicator_performance_report(
        self,
        indicator_name: str = None,
        indicator_type: str = None,
        market_condition: str = None,
        lookback_days: int = None,
    ) -> Dict[str, Any]:
        """Get comprehensive indicator performance report"""
        try:
            # Get specific indicator performance if requested
            if indicator_name:
                performance = self.indicator_tracker.get_indicator_performance(
                    indicator_name=indicator_name, market_condition=market_condition
                )

                return {
                    "report_type": "individual_indicator",
                    "indicator_name": indicator_name,
                    "market_condition": market_condition,
                    "performance": (
                        asdict(performance)
                        if hasattr(performance, "__dict__")
                        else performance
                    ),
                    "generation_time": datetime.now().isoformat(),
                }

            # Get comprehensive performance report
            report = self.indicator_tracker.generate_performance_report(lookback_days)

            # Filter by indicator type if specified
            if indicator_type:
                filtered_performance = self.indicator_tracker.get_indicator_performance(
                    indicator_type=indicator_type
                )
                report["filtered_by_type"] = {
                    "indicator_type": indicator_type,
                    "indicators": {
                        name: asdict(perf)
                        for name, perf in filtered_performance.items()
                    },
                }

            return {"report_type": "comprehensive", "report_data": report}

        except Exception as e:
            logger.error(f"Error generating indicator performance report: {e}")
            return {
                "report_type": "error",
                "error": str(e),
                "generation_time": datetime.now().isoformat(),
            }

    async def get_top_performing_indicators(
        self, metric: str = "sharpe_ratio", top_n: int = 10, min_signals: int = None
    ) -> Dict[str, Any]:
        """Get top performing indicators ranked by specified metric"""
        try:
            top_indicators = self.indicator_tracker.get_top_performing_indicators(
                metric=metric, top_n=top_n, min_signals=min_signals
            )

            return {
                "status": "success",
                "ranking_metric": metric,
                "top_indicators": top_indicators,
                "total_indicators_evaluated": len(
                    self.indicator_tracker.indicator_performance
                ),
                "generation_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting top performing indicators: {e}")
            return {"status": "error", "error": str(e), "ranking_metric": metric}

    async def update_indicator_signal_outcome(
        self, signal_id: str, actual_outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update the outcome for a previously recorded indicator signal"""
        try:
            # Update the signal outcome
            self.indicator_tracker.update_signal_outcome(signal_id, actual_outcome)

            return {
                "status": "success",
                "signal_id": signal_id,
                "outcome_updated": True,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error updating indicator signal outcome: {e}")
            return {"status": "error", "error": str(e), "signal_id": signal_id}

    async def get_indicator_market_condition_analysis(
        self, market_condition: str = None
    ) -> Dict[str, Any]:
        """Get analysis of indicator performance by market conditions"""
        try:
            analysis_results = {}

            # Get all indicators and their market condition performance
            all_performance = self.indicator_tracker.get_indicator_performance()

            if market_condition:
                # Analyze specific market condition
                condition_performance = {}
                for indicator_name, perf in all_performance.items():
                    if hasattr(perf, "market_condition_performance"):
                        mc_perf = perf.market_condition_performance.get(
                            market_condition, {}
                        )
                        if (
                            mc_perf.get("signals", 0) >= 3
                        ):  # Minimum signals for analysis
                            condition_performance[indicator_name] = mc_perf

                analysis_results = {
                    "market_condition": market_condition,
                    "indicator_performance": condition_performance,
                    "top_performers_in_condition": sorted(
                        condition_performance.items(),
                        key=lambda x: x[1].get("win_rate", 0.0),
                        reverse=True,
                    )[:5],
                }
            else:
                # Analyze all market conditions
                all_conditions = {}
                for indicator_name, perf in all_performance.items():
                    if hasattr(perf, "market_condition_performance"):
                        for (
                            condition,
                            mc_perf,
                        ) in perf.market_condition_performance.items():
                            if condition not in all_conditions:
                                all_conditions[condition] = {}
                            if mc_perf.get("signals", 0) >= 3:
                                all_conditions[condition][indicator_name] = mc_perf

                analysis_results = {
                    "all_market_conditions": all_conditions,
                    "condition_summary": {
                        condition: {
                            "indicator_count": len(indicators),
                            "avg_win_rate": (
                                np.mean(
                                    [
                                        perf.get("win_rate", 0.0)
                                        for perf in indicators.values()
                                    ]
                                )
                                if indicators
                                else 0.0
                            ),
                            "total_signals": sum(
                                perf.get("signals", 0) for perf in indicators.values()
                            ),
                        }
                        for condition, indicators in all_conditions.items()
                    },
                }

            return {
                "status": "success",
                "analysis": analysis_results,
                "generation_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generating market condition analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "market_condition": market_condition,
            }

    # Helper methods
    async def _prepare_market_data_for_idea(
        self, trading_idea: TradingIdea
    ) -> Dict[str, Any]:
        """Prepare market data for backtesting a trading idea"""
        # Mock market data generation (in production, fetch real data)
        days = 252  # 1 year of data
        base_price = 100.0

        # Generate realistic price series
        returns = np.random.normal(0.0008, 0.02, days)  # Daily returns
        prices = [base_price]

        for return_val in returns:
            prices.append(prices[-1] * (1 + return_val))

        return {
            "symbol": trading_idea.symbol,
            "prices": prices,
            "volumes": np.random.randint(100000, 1000000, days + 1).tolist(),
            "dates": [
                (datetime.now() - timedelta(days=days - i)).isoformat()
                for i in range(days + 1)
            ],
            "highs": [p * 1.01 for p in prices],
            "lows": [p * 0.99 for p in prices],
        }

    async def _execute_idea_backtest(
        self,
        trading_idea: TradingIdea,
        market_data: Dict[str, Any],
        config: BacktestConfig,
    ) -> Dict[str, Any]:
        """Execute backtest for a specific trading idea"""
        try:
            # Simulate backtest results (in production, use real BacktestEngine)
            base_return = np.random.normal(0.05, 0.15)  # 5% average with 15% volatility

            # Adjust based on confidence
            confidence_multiplier = 0.5 + (trading_idea.confidence * 0.5)
            adjusted_return = base_return * confidence_multiplier

            # Calculate metrics
            volatility = abs(np.random.normal(0.15, 0.05))
            sharpe_ratio = (
                adjusted_return / max(volatility, 0.01) if volatility > 0 else 0
            )
            max_drawdown = abs(np.random.normal(0.08, 0.04))

            # Win rate based on strategy and confidence
            base_win_rate = (
                0.55 if trading_idea.strategy in ["technical", "pattern"] else 0.50
            )
            win_rate = min(0.90, base_win_rate + (trading_idea.confidence * 0.2))

            trades_count = np.random.randint(10, 50)

            return {
                "backtest_id": str(uuid.uuid4()),
                "total_return": adjusted_return,
                "annualized_return": adjusted_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "trades_count": trades_count,
                "volatility": volatility,
                "final_capital": config.initial_capital * (1 + adjusted_return),
                "execution_time": np.random.uniform(1.0, 5.0),
            }

        except Exception as e:
            logger.error("Error executing idea backtest", error=str(e))
            return {
                "backtest_id": str(uuid.uuid4()),
                "error": str(e),
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trades_count": 0,
            }

    async def _generate_agent_feedback(
        self, trading_idea: TradingIdea, backtest_results: Dict[str, Any]
    ) -> BacktestFeedback:
        """Generate structured feedback for AI agent"""
        try:
            total_return = backtest_results.get("total_return", 0.0)
            sharpe_ratio = backtest_results.get("sharpe_ratio", 0.0)
            max_drawdown = backtest_results.get("max_drawdown", 0.0)
            win_rate = backtest_results.get("win_rate", 0.0)

            # Determine performance rank
            performance_rank = "poor"
            if total_return > 0.15 and sharpe_ratio > 1.5:
                performance_rank = "excellent"
            elif total_return > 0.08 and sharpe_ratio > 1.0:
                performance_rank = "good"
            elif total_return > 0.02 and sharpe_ratio > 0.5:
                performance_rank = "average"

            # Generate improvement suggestions
            suggestions = []
            if total_return < 0:
                suggestions.append("Consider inverse correlation analysis")
                suggestions.append("Review entry timing conditions")

            if max_drawdown > 0.15:
                suggestions.append("Implement tighter risk management")
                suggestions.append("Consider smaller position sizes")

            if win_rate < 0.5:
                suggestions.append("Refine pattern recognition criteria")
                suggestions.append("Add confirmation indicators")

            # Risk score calculation
            risk_score = min(
                100,
                max(
                    0,
                    (max_drawdown * 50)
                    + ((1 - win_rate) * 30)
                    + (max(0, -total_return) * 20),
                ),
            )

            return BacktestFeedback(
                idea_id=trading_idea.id,
                agent_id=trading_idea.agent_id,
                backtest_id=backtest_results.get("backtest_id", ""),
                success=total_return > 0,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                trades_count=backtest_results.get("trades_count", 0),
                risk_score=risk_score,
                performance_rank=performance_rank,
                improvement_suggestions=suggestions,
                learned_patterns=[],  # Will be populated by pattern learning
                optimized_parameters={},
                market_context={"volatility": "normal", "trend": "neutral"},
                execution_time=backtest_results.get("execution_time", 0.0),
            )

        except Exception as e:
            logger.error("Error generating agent feedback", error=str(e))
            # Return default feedback on error
            return BacktestFeedback(
                idea_id=trading_idea.id,
                agent_id=trading_idea.agent_id,
                backtest_id="",
                success=False,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                trades_count=0,
                risk_score=100.0,
                performance_rank="error",
                improvement_suggestions=["Error in backtest execution"],
                learned_patterns=[],
                optimized_parameters={},
                market_context={},
                execution_time=0.0,
            )

    def _update_agent_performance(self, agent_id: str, feedback: BacktestFeedback):
        """Update agent performance tracking"""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                "total_ideas": 0,
                "successful_ideas": 0,
                "feedback_history": [],
                "strategy_preferences": {},
                "learning_progress": [],
            }

        agent_data = self.agent_performance[agent_id]
        agent_data["total_ideas"] += 1

        if feedback.success:
            agent_data["successful_ideas"] += 1

        # Store feedback with timestamp
        feedback_record = asdict(feedback)
        feedback_record["timestamp"] = datetime.now().isoformat()
        agent_data["feedback_history"].append(feedback_record)

        # Keep only recent feedback (last 100 records)
        if len(agent_data["feedback_history"]) > 100:
            agent_data["feedback_history"] = agent_data["feedback_history"][-100:]

    async def _update_pattern_learning(
        self, trading_idea: TradingIdea, backtest_results: Dict[str, Any]
    ):
        """Update pattern learning from backtest results"""
        try:
            # Extract patterns from trading idea
            strategy = trading_idea.strategy

            # Mock market conditions
            market_conditions = {
                "volatility": backtest_results.get("volatility", 0.2),
                "trend": "neutral",
                "volume": "normal",
            }

            # Performance metrics for learning
            performance_metrics = {
                "return": backtest_results.get("total_return", 0.0),
                "sharpe_ratio": backtest_results.get("sharpe_ratio", 0.0),
                "max_drawdown": backtest_results.get("max_drawdown", 0.0),
            }

            # Update pattern performance
            self.pattern_learner.update_pattern_performance(
                strategy, market_conditions, performance_metrics
            )

        except Exception as e:
            logger.error("Error updating pattern learning", error=str(e))

    def _generate_parameter_combinations(
        self, parameter_space: Dict[str, Dict[str, Any]], max_combinations: int = 50
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations for optimization"""
        try:
            combinations = []

            # Simple grid search implementation
            param_names = list(parameter_space.keys())
            param_values = []

            for param_name in param_names:
                param_config = parameter_space[param_name]
                param_type = param_config.get("type", "float")

                if param_type == "float":
                    min_val = param_config.get("min", 0.0)
                    max_val = param_config.get("max", 1.0)
                    steps = param_config.get("steps", 5)
                    values = np.linspace(min_val, max_val, steps).tolist()
                elif param_type == "int":
                    min_val = param_config.get("min", 1)
                    max_val = param_config.get("max", 10)
                    values = list(range(min_val, max_val + 1))
                else:  # categorical
                    values = param_config.get("values", [])

                param_values.append(values)

            # Generate combinations
            import itertools

            all_combinations = list(itertools.product(*param_values))

            # Limit combinations
            if len(all_combinations) > max_combinations:
                # Random sampling if too many combinations
                import random

                all_combinations = random.sample(all_combinations, max_combinations)

            # Convert to dictionary format
            for combination in all_combinations:
                param_dict = {}
                for i, param_name in enumerate(param_names):
                    param_dict[param_name] = combination[i]
                combinations.append(param_dict)

            return combinations

        except Exception as e:
            logger.error("Error generating parameter combinations", error=str(e))
            return [{}]  # Return empty parameters as fallback

    def _calculate_improvement_trend(
        self, feedback_history: List[Dict[str, Any]]
    ) -> str:
        """Calculate if agent is improving over time"""
        try:
            if len(feedback_history) < 5:
                return "insufficient_data"

            # Get recent and older performance
            recent_performance = np.mean(
                [f.get("total_return", 0.0) for f in feedback_history[-5:]]
            )

            older_performance = (
                np.mean([f.get("total_return", 0.0) for f in feedback_history[-10:-5]])
                if len(feedback_history) >= 10
                else recent_performance
            )

            improvement = recent_performance - older_performance

            if improvement > 0.02:
                return "improving"
            elif improvement < -0.02:
                return "declining"
            else:
                return "stable"

        except Exception as e:
            logger.error("Error calculating improvement trend", error=str(e))
            return "unknown"

    # ML Prediction Integration Methods

    async def initialize_ml_models(self) -> bool:
        """Initialize ML prediction models"""
        if not self.ml_pipeline:
            logger.warning("ML pipeline not available - no database session provided")
            return False

        try:
            await self.ml_pipeline.initialize_llm_predictor()
            return True
        except Exception as e:
            logger.error("Failed to initialize ML models", error=str(e))
            return False

    async def track_ml_prediction(
        self, symbol: str, prediction_result: Dict[str, Any], horizon_days: int
    ) -> Optional[int]:
        """Track ML prediction for later outcome verification"""
        if not self.ml_pipeline:
            return None

        try:
            # Get or create model version for ensemble
            model_version = await self._get_or_create_ensemble_model_version()

            # Record the prediction
            prediction = await self.ml_pipeline.record_prediction(
                model_version_id=model_version.id,
                symbol=symbol,
                prediction_horizon=horizon_days,
                predicted_direction=prediction_result.get("ensemble_prediction"),
                predicted_return=prediction_result.get("predicted_return", 0.0),
                confidence_score=prediction_result.get("confidence", 0.0),
                individual_predictions=prediction_result.get("model_predictions", {}),
                market_conditions=prediction_result.get("market_context", {}),
                technical_indicators=prediction_result.get("technical_context", {}),
            )

            return prediction.id if prediction else None

        except Exception as e:
            logger.error("Error tracking ML prediction", error=str(e))
            return None

    async def update_ml_prediction_outcome(
        self, prediction_id: int, actual_direction: str, actual_return: float
    ) -> bool:
        """Update the actual outcome for a tracked ML prediction"""
        if not self.ml_pipeline:
            return False

        try:
            return await self.ml_pipeline.update_prediction_outcome(
                prediction_id, actual_direction, actual_return
            )
        except Exception as e:
            logger.error("Error updating ML prediction outcome", error=str(e))
            return False

    async def get_ml_model_performance(
        self, model_name: str = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Get ML model performance report"""
        if not self.ml_pipeline:
            return {"error": "ML pipeline not available"}

        try:
            return await self.ml_pipeline.get_model_performance_report(
                model_name=model_name, days_back=days_back
            )
        except Exception as e:
            logger.error("Error getting ML model performance", error=str(e))
            return {"error": f"Failed to get performance report: {str(e)}"}

    async def _get_or_create_ensemble_model_version(self):
        """Get or create the ensemble model version"""
        try:
            # Check for existing ensemble model version
            existing_version = (
                self.db.query(MLModelVersion)
                .filter(
                    MLModelVersion.model_name == "llm_ensemble",
                    MLModelVersion.is_active == True,
                )
                .first()
                if self.db
                else None
            )

            if existing_version:
                return existing_version

            # Create new ensemble model version
            model_config = {
                "type": "ensemble",
                "models": [
                    "qwen_quant",
                    "yi_technical",
                    "glm_risk",
                    "deepseek_strategy",
                ],
                "voting_method": "weighted_confidence",
                "confidence_threshold": 0.6,
            }

            return await self.ml_pipeline.create_model_version(
                model_name="llm_ensemble",
                model_type="LLM_ENSEMBLE",
                model_config=model_config,
            )

        except Exception as e:
            logger.error("Error getting/creating ensemble model version", error=str(e))
            raise
