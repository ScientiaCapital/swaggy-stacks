"""
Refactored BacktestService - Lightweight orchestrator

This replaces the original 1720-line BacktestService by delegating to specialized services
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session

from app.core.database import get_db
from app.monitoring.metrics import PrometheusMetrics
from app.services.backtest_core_service import CoreBacktestService, TradingIdea
from app.services.indicator_performance_service import IndicatorPerformanceService
from app.services.ml_model_service import MLModelService

logger = logging.getLogger(__name__)


class BacktestService:
    """
    Refactored Backtest Service - Main interface for AI agent backtesting

    This service now acts as a lightweight orchestrator, delegating work to:
    - CoreBacktestService: Core backtesting logic and trading idea execution
    - IndicatorPerformanceService: Indicator tracking and analytics
    - MLModelService: ML model management and predictions
    """

    def __init__(self, db: Session = None):
        self.db = db or next(get_db())
        self.metrics = PrometheusMetrics()

        # Initialize specialized services
        self.core_service = CoreBacktestService(self.db, self.metrics)
        self.indicator_service = IndicatorPerformanceService(self.db, self.metrics)
        self.ml_service = MLModelService(self.db, self.metrics)

        logger.info("RefactoredBacktestService initialized with specialized sub-services")

    # Core Backtesting Methods - Delegate to CoreBacktestService
    async def submit_trading_idea(
        self,
        agent_id: str,
        symbol: str,
        strategy: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        rationale: str = "",
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """Submit trading idea from AI agent for backtesting"""
        return await self.core_service.submit_trading_idea(
            symbol=symbol,
            strategy=strategy,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            rationale=rationale,
            timeframe=timeframe,
            agent_id=agent_id
        )

    async def get_pattern_performance(
        self,
        pattern_name: str = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get performance analysis for trading patterns"""
        return await self.core_service.get_pattern_performance(pattern_name, days_back)

    async def optimize_strategy_parameters(
        self,
        agent_id: str,
        strategy_name: str,
        parameter_ranges: Dict[str, tuple],
        optimization_metric: str = "sharpe_ratio",
        max_iterations: int = 50
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using backtesting"""
        return await self.core_service.optimize_strategy_parameters(
            strategy_name=strategy_name,
            symbol="TEST",  # Use test symbol for optimization
            parameter_ranges=parameter_ranges,
            optimization_metric=optimization_metric,
            max_iterations=max_iterations
        )

    async def get_agent_learning_summary(
        self,
        agent_id: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get learning and performance summary for trading agents"""
        return await self.core_service.get_agent_learning_summary(agent_id, days_back)

    # Indicator Performance Methods - Delegate to IndicatorPerformanceService
    async def track_indicator_performance(
        self,
        indicator_name: str,
        symbol: str,
        signal_type: str,
        confidence: float,
        entry_price: float,
        current_price: float,
        market_condition: str = "normal",
        timeframe: str = "1d",
        additional_data: Dict = None
    ) -> Dict[str, Any]:
        """Track performance of an indicator signal"""
        try:
            performance = await self.indicator_service.track_indicator_performance(
                indicator_name=indicator_name,
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=entry_price,
                current_price=current_price,
                market_condition=market_condition,
                timeframe=timeframe,
                additional_data=additional_data
            )

            return {
                "status": "success",
                "performance_id": performance.id if performance else None,
                "indicator_name": indicator_name
            }
        except Exception as e:
            logger.error(f"Error tracking indicator performance: {e}")
            return {"status": "error", "error": str(e)}

    async def update_indicator_signal_outcome(
        self,
        performance_id: int,
        exit_price: float,
        exit_timestamp: datetime = None,
        outcome: str = None
    ) -> Dict[str, Any]:
        """Update the final outcome of an indicator signal"""
        success = await self.indicator_service.update_indicator_signal_outcome(
            performance_id, exit_price, exit_timestamp, outcome
        )

        return {
            "status": "success" if success else "error",
            "performance_id": performance_id,
            "outcome_updated": success
        }

    async def get_indicator_performance_report(
        self,
        indicator_name: str = None,
        symbol: str = None,
        days_back: int = 30,
        market_condition: str = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report for indicators"""
        return await self.indicator_service.get_indicator_performance_report(
            indicator_name, symbol, days_back, market_condition
        )

    async def get_top_performing_indicators(
        self,
        limit: int = 10,
        days_back: int = 30,
        min_signals: int = 5
    ) -> Dict[str, Any]:
        """Get top performing indicators based on win rate and P&L"""
        indicators = await self.indicator_service.get_top_performing_indicators(
            limit, days_back, min_signals
        )

        return {
            "status": "success",
            "top_indicators": indicators,
            "period_days": days_back
        }

    async def get_indicator_market_condition_analysis(
        self,
        indicator_name: str,
        days_back: int = 90
    ) -> Dict[str, Any]:
        """Analyze indicator performance across different market conditions"""
        return await self.indicator_service.get_indicator_market_condition_analysis(
            indicator_name, days_back
        )

    # ML Model Methods - Delegate to MLModelService
    async def initialize_ml_models(self, config: Dict = None) -> bool:
        """Initialize ML prediction models"""
        return await self.ml_service.initialize_ml_models(config)

    async def track_ml_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction_type: str,
        prediction_value: float,
        confidence: float,
        features_used: Dict = None,
        model_version: str = "1.0"
    ) -> Optional[int]:
        """Track a machine learning prediction"""
        prediction = await self.ml_service.track_ml_prediction(
            model_name, symbol, prediction_type, prediction_value,
            confidence, features_used, model_version
        )

        return prediction.id if prediction else None

    async def update_ml_prediction_outcome(
        self,
        prediction_id: int,
        actual_value: float,
        outcome_timestamp: datetime = None
    ) -> bool:
        """Update the actual outcome of an ML prediction"""
        return await self.ml_service.update_ml_prediction_outcome(
            prediction_id, actual_value, outcome_timestamp
        )

    async def get_ml_model_performance(
        self,
        model_name: str = None,
        days_back: int = 30,
        prediction_type: str = None
    ) -> Dict[str, Any]:
        """Get performance metrics for ML models"""
        return await self.ml_service.get_ml_model_performance(
            model_name, days_back, prediction_type
        )

    # Convenience methods that combine services
    async def comprehensive_performance_report(
        self,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive performance report across all services"""
        try:
            # Get reports from all services
            indicator_report = await self.indicator_service.get_indicator_performance_report(
                days_back=days_back
            )

            ml_report = await self.ml_service.get_ml_model_performance(
                days_back=days_back
            )

            pattern_report = await self.core_service.get_pattern_performance(
                days_back=days_back
            )

            return {
                "report_type": "comprehensive",
                "period_days": days_back,
                "indicator_performance": indicator_report,
                "ml_model_performance": ml_report,
                "pattern_performance": pattern_report,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {
                "report_type": "comprehensive",
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            }

    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health across all services"""
        try:
            health_summary = {
                "core_service": {"status": "healthy", "active_ideas": len(self.core_service.trading_ideas)},
                "indicator_service": {"status": "healthy"},
                "ml_service": {"status": "healthy", "models_initialized": self.ml_service.ml_pipeline is not None},
                "overall_status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            }

            return health_summary

        except Exception as e:
            logger.error(f"Error getting system health summary: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    # Legacy compatibility methods (simplified)
    def get_indicator_tracker(self):
        """Legacy compatibility - returns indicator service"""
        return self.indicator_service

    def get_pattern_learner(self):
        """Legacy compatibility - returns pattern learner from core service"""
        return self.core_service.pattern_learner

    def get_backtest_engine(self):
        """Legacy compatibility - returns backtest engine from core service"""
        return self.core_service.backtest_engine