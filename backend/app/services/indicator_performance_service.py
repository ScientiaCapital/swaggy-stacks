"""
Indicator Performance Service - Extracted from BacktestService

Handles indicator performance tracking, analytics, and reporting
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.indicator_performance import (
    IndicatorPerformance,
    IndicatorParameters,
    ParameterOptimization,
    MLModelVersion,
    MLModelPrediction
)
from app.monitoring.metrics import PrometheusMetrics

logger = logging.getLogger(__name__)


class IndicatorPerformanceService:
    """Service for tracking and analyzing indicator performance"""

    def __init__(self, db: Session = None, metrics: PrometheusMetrics = None):
        self.db = db or next(get_db())
        self.metrics = metrics or PrometheusMetrics()
        logger.info("IndicatorPerformanceService initialized")

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
    ) -> IndicatorPerformance:
        """Track performance of an indicator signal"""
        try:
            # Calculate unrealized P&L
            if signal_type.lower() == 'buy':
                unrealized_pnl = current_price - entry_price
            else:  # sell
                unrealized_pnl = entry_price - current_price

            performance = IndicatorPerformance(
                indicator_name=indicator_name,
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                market_condition=market_condition,
                timeframe=timeframe,
                signal_timestamp=datetime.utcnow(),
                metadata_json=additional_data or {}
            )

            self.db.add(performance)
            self.db.commit()

            # Update metrics
            if self.metrics:
                self.metrics.indicator_signals_total.labels(
                    indicator=indicator_name,
                    signal_type=signal_type
                ).inc()

            logger.info(f"Tracked performance for {indicator_name} signal on {symbol}")
            return performance

        except Exception as e:
            logger.error(f"Error tracking indicator performance: {e}")
            self.db.rollback()
            raise

    async def update_indicator_signal_outcome(
        self,
        performance_id: int,
        exit_price: float,
        exit_timestamp: datetime = None,
        outcome: str = None
    ) -> bool:
        """Update the final outcome of an indicator signal"""
        try:
            performance = self.db.query(IndicatorPerformance).filter(
                IndicatorPerformance.id == performance_id
            ).first()

            if not performance:
                logger.warning(f"Performance record {performance_id} not found")
                return False

            performance.exit_price = exit_price
            performance.exit_timestamp = exit_timestamp or datetime.utcnow()
            performance.outcome = outcome

            # Calculate final P&L
            if performance.signal_type.lower() == 'buy':
                performance.realized_pnl = exit_price - performance.entry_price
            else:  # sell
                performance.realized_pnl = performance.entry_price - exit_price

            self.db.commit()

            # Update metrics
            if self.metrics and outcome:
                if outcome == 'win':
                    self.metrics.indicator_win_rate.labels(
                        indicator=performance.indicator_name
                    ).inc()
                elif outcome == 'loss':
                    self.metrics.indicator_loss_rate.labels(
                        indicator=performance.indicator_name
                    ).inc()

            logger.info(f"Updated outcome for performance {performance_id}: {outcome}")
            return True

        except Exception as e:
            logger.error(f"Error updating signal outcome: {e}")
            self.db.rollback()
            return False

    async def get_indicator_performance_report(
        self,
        indicator_name: str = None,
        symbol: str = None,
        days_back: int = 30,
        market_condition: str = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report for indicators"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            query = self.db.query(IndicatorPerformance).filter(
                IndicatorPerformance.signal_timestamp >= cutoff_date
            )

            if indicator_name:
                query = query.filter(IndicatorPerformance.indicator_name == indicator_name)
            if symbol:
                query = query.filter(IndicatorPerformance.symbol == symbol)
            if market_condition:
                query = query.filter(IndicatorPerformance.market_condition == market_condition)

            performances = query.all()

            if not performances:
                return {"message": "No performance data found", "indicators": []}

            # Calculate metrics by indicator
            indicator_stats = {}
            for perf in performances:
                name = perf.indicator_name
                if name not in indicator_stats:
                    indicator_stats[name] = {
                        'total_signals': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_pnl': 0.0,
                        'avg_confidence': 0.0,
                        'signals': []
                    }

                stats = indicator_stats[name]
                stats['total_signals'] += 1
                stats['signals'].append(perf)
                stats['avg_confidence'] += perf.confidence

                if perf.realized_pnl is not None:
                    stats['total_pnl'] += perf.realized_pnl
                    if perf.realized_pnl > 0:
                        stats['wins'] += 1
                    else:
                        stats['losses'] += 1

            # Calculate final metrics
            report_data = []
            for name, stats in indicator_stats.items():
                total_closed = stats['wins'] + stats['losses']
                win_rate = (stats['wins'] / total_closed * 100) if total_closed > 0 else 0
                avg_confidence = stats['avg_confidence'] / stats['total_signals']

                report_data.append({
                    'indicator_name': name,
                    'total_signals': stats['total_signals'],
                    'closed_signals': total_closed,
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'win_rate': round(win_rate, 2),
                    'total_pnl': round(stats['total_pnl'], 2),
                    'avg_confidence': round(avg_confidence, 3),
                    'avg_pnl_per_trade': round(stats['total_pnl'] / total_closed, 2) if total_closed > 0 else 0
                })

            # Sort by total P&L descending
            report_data.sort(key=lambda x: x['total_pnl'], reverse=True)

            return {
                'period_days': days_back,
                'total_indicators': len(report_data),
                'indicators': report_data,
                'generated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}

    async def get_top_performing_indicators(
        self,
        limit: int = 10,
        days_back: int = 30,
        min_signals: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top performing indicators based on win rate and P&L"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            performances = self.db.query(IndicatorPerformance).filter(
                IndicatorPerformance.signal_timestamp >= cutoff_date,
                IndicatorPerformance.realized_pnl.isnot(None)
            ).all()

            if not performances:
                return []

            # Group by indicator and calculate performance
            indicator_metrics = {}
            for perf in performances:
                name = perf.indicator_name
                if name not in indicator_metrics:
                    indicator_metrics[name] = {
                        'wins': 0,
                        'total': 0,
                        'total_pnl': 0.0,
                        'confidence_sum': 0.0
                    }

                metrics = indicator_metrics[name]
                metrics['total'] += 1
                metrics['total_pnl'] += perf.realized_pnl
                metrics['confidence_sum'] += perf.confidence

                if perf.realized_pnl > 0:
                    metrics['wins'] += 1

            # Calculate final scores and filter
            top_performers = []
            for name, metrics in indicator_metrics.items():
                if metrics['total'] < min_signals:
                    continue

                win_rate = metrics['wins'] / metrics['total']
                avg_pnl = metrics['total_pnl'] / metrics['total']
                avg_confidence = metrics['confidence_sum'] / metrics['total']

                # Composite score: win_rate * 0.4 + normalized_pnl * 0.4 + confidence * 0.2
                score = win_rate * 0.4 + (avg_pnl / 100) * 0.4 + avg_confidence * 0.2

                top_performers.append({
                    'indicator_name': name,
                    'win_rate': round(win_rate * 100, 2),
                    'total_signals': metrics['total'],
                    'total_pnl': round(metrics['total_pnl'], 2),
                    'avg_pnl_per_trade': round(avg_pnl, 2),
                    'avg_confidence': round(avg_confidence, 3),
                    'performance_score': round(score, 4)
                })

            # Sort by performance score descending
            top_performers.sort(key=lambda x: x['performance_score'], reverse=True)

            return top_performers[:limit]

        except Exception as e:
            logger.error(f"Error getting top performers: {e}")
            return []

    async def get_indicator_market_condition_analysis(
        self,
        indicator_name: str,
        days_back: int = 90
    ) -> Dict[str, Any]:
        """Analyze indicator performance across different market conditions"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            performances = self.db.query(IndicatorPerformance).filter(
                IndicatorPerformance.indicator_name == indicator_name,
                IndicatorPerformance.signal_timestamp >= cutoff_date,
                IndicatorPerformance.realized_pnl.isnot(None)
            ).all()

            if not performances:
                return {"message": f"No data found for {indicator_name}"}

            # Group by market condition
            condition_stats = {}
            for perf in performances:
                condition = perf.market_condition or 'unknown'
                if condition not in condition_stats:
                    condition_stats[condition] = {
                        'total': 0,
                        'wins': 0,
                        'total_pnl': 0.0,
                        'confidence_sum': 0.0
                    }

                stats = condition_stats[condition]
                stats['total'] += 1
                stats['total_pnl'] += perf.realized_pnl
                stats['confidence_sum'] += perf.confidence

                if perf.realized_pnl > 0:
                    stats['wins'] += 1

            # Calculate metrics for each condition
            analysis = []
            for condition, stats in condition_stats.items():
                win_rate = stats['wins'] / stats['total'] * 100
                avg_pnl = stats['total_pnl'] / stats['total']
                avg_confidence = stats['confidence_sum'] / stats['total']

                analysis.append({
                    'market_condition': condition,
                    'total_signals': stats['total'],
                    'wins': stats['wins'],
                    'win_rate': round(win_rate, 2),
                    'total_pnl': round(stats['total_pnl'], 2),
                    'avg_pnl_per_trade': round(avg_pnl, 2),
                    'avg_confidence': round(avg_confidence, 3)
                })

            # Sort by win rate descending
            analysis.sort(key=lambda x: x['win_rate'], reverse=True)

            return {
                'indicator_name': indicator_name,
                'analysis_period_days': days_back,
                'market_conditions': analysis,
                'total_signals_analyzed': sum(perf['total_signals'] for perf in analysis),
                'generated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing market conditions for {indicator_name}: {e}")
            return {"error": str(e)}