"""
Alpha Pattern Tracker Service
Tracks pattern performance and alpha generation for Chinese LLM routing optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from sqlalchemy import func, and_, desc, asc
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.pattern_performance import (
    PatternPerformance, 
    LLMPerformanceMetrics, 
    AlphaSignal, 
    PatternLearning
)
from app.ai.deepseek_trade_orchestrator import TaskType, TradingDecisionResult


@dataclass
class PatternDetection:
    """Pattern detection data structure"""
    pattern_type: str
    pattern_subtype: Optional[str]
    symbol: str
    timeframe: str
    confidence: float
    predicted_direction: str
    predicted_magnitude: Optional[float]
    time_horizon: str
    technical_indicators: Dict[str, Any]
    market_context: Dict[str, Any]
    detected_by_llm: str


@dataclass
class AlphaMetrics:
    """Alpha generation metrics"""
    alpha_generated: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    direction_accuracy: float
    magnitude_accuracy: float


class AlphaPatternTracker:
    """
    Service for tracking pattern performance and alpha generation
    Focuses on identifying and optimizing alpha-seeking patterns
    """
    
    def __init__(self, db_session: Session = None):
        self.logger = logging.getLogger(__name__)
        self.db = db_session or next(get_db())
        
        # Alpha tracking configuration
        self.alpha_config = {
            "min_tracking_period_days": 30,
            "pattern_success_threshold": 0.6,
            "alpha_significance_threshold": 0.02,  # 2% minimum alpha
            "confidence_calibration_window": 100,  # Rolling window for calibration
        }
        
        # LLM specialization tracking
        self.llm_specializations = {
            "deepseek_r1": ["alpha_analysis", "pattern_recognition", "market_regime"],
            "qwen_quant": ["quantitative_analysis", "statistical_patterns", "momentum"],
            "yi_technical": ["technical_patterns", "chart_analysis", "breakouts"],
            "glm_risk": ["risk_assessment", "volatility_patterns", "drawdown_control"],
            "deepseek_coder": ["algorithmic_patterns", "strategy_optimization", "backtesting"]
        }

    async def record_pattern_detection(
        self, 
        detection: PatternDetection,
        market_conditions: Dict[str, Any] = None
    ) -> str:
        """
        Record a new pattern detection for alpha tracking
        
        Returns:
            Pattern performance ID for later outcome tracking
        """
        try:
            # Generate pattern signature for deduplication
            pattern_signature = self._generate_pattern_signature(detection)
            
            # Check for existing pattern in recent timeframe
            existing_pattern = self.db.query(PatternPerformance).filter(
                and_(
                    PatternPerformance.pattern_signature == pattern_signature,
                    PatternPerformance.detection_timestamp >= datetime.utcnow() - timedelta(hours=24)
                )
            ).first()
            
            if existing_pattern:
                self.logger.info(f"Pattern already detected recently: {pattern_signature}")
                return str(existing_pattern.id)
            
            # Create new pattern performance record
            pattern_record = PatternPerformance(
                pattern_type=detection.pattern_type,
                pattern_subtype=detection.pattern_subtype,
                pattern_signature=pattern_signature,
                symbol=detection.symbol,
                timeframe=detection.timeframe,
                detected_by_llm=detection.detected_by_llm,
                detection_confidence=detection.confidence,
                predicted_direction=detection.predicted_direction,
                predicted_magnitude=detection.predicted_magnitude,
                confidence_level=detection.confidence,
                time_horizon=detection.time_horizon,
                technical_indicators=detection.technical_indicators,
                market_context=detection.market_context,
                market_volatility=market_conditions.get("volatility") if market_conditions else None,
                market_regime=market_conditions.get("regime") if market_conditions else None,
                volume_profile=market_conditions.get("volume_profile") if market_conditions else None,
            )
            
            self.db.add(pattern_record)
            self.db.commit()
            self.db.refresh(pattern_record)
            
            self.logger.info(
                f"Pattern detection recorded: {detection.pattern_type} on {detection.symbol} "
                f"by {detection.detected_by_llm} with confidence {detection.confidence:.3f}"
            )
            
            # Schedule async alpha tracking
            asyncio.create_task(self._schedule_alpha_verification(str(pattern_record.id)))
            
            return str(pattern_record.id)
            
        except Exception as e:
            self.logger.error(f"Error recording pattern detection: {e}")
            self.db.rollback()
            raise

    async def update_pattern_outcome(
        self, 
        pattern_id: str, 
        actual_direction: str,
        actual_magnitude: float,
        alpha_metrics: AlphaMetrics
    ) -> bool:
        """
        Update pattern with actual outcome and alpha generation
        
        Returns:
            True if successfully updated
        """
        try:
            pattern = self.db.query(PatternPerformance).filter(
                PatternPerformance.id == pattern_id
            ).first()
            
            if not pattern:
                self.logger.warning(f"Pattern not found: {pattern_id}")
                return False
            
            # Calculate prediction accuracy
            direction_correct = (pattern.predicted_direction == actual_direction)
            magnitude_accuracy = self._calculate_magnitude_accuracy(
                pattern.predicted_magnitude, actual_magnitude
            )
            
            # Update pattern with outcomes
            pattern.outcome_verified = True
            pattern.actual_direction = actual_direction
            pattern.actual_magnitude = actual_magnitude
            pattern.verification_timestamp = datetime.utcnow()
            pattern.alpha_generated = alpha_metrics.alpha_generated
            pattern.sharpe_ratio = alpha_metrics.sharpe_ratio
            pattern.max_drawdown = alpha_metrics.max_drawdown
            pattern.win_rate = alpha_metrics.win_rate
            pattern.prediction_accuracy = 1.0 if direction_correct else 0.0
            pattern.magnitude_accuracy = magnitude_accuracy
            pattern.pattern_score = self._calculate_pattern_score(pattern, alpha_metrics)
            
            self.db.commit()
            
            # Update LLM performance metrics
            await self._update_llm_performance_metrics(
                pattern.detected_by_llm,
                self._get_task_type_from_pattern(pattern.pattern_type),
                direction_correct,
                alpha_metrics.alpha_generated,
                pattern.detection_confidence,
                alpha_metrics
            )
            
            # Update pattern learning insights
            await self._update_pattern_learning(pattern, alpha_metrics)
            
            self.logger.info(
                f"Pattern outcome updated: {pattern_id} - "
                f"Alpha: {alpha_metrics.alpha_generated:.3f}, "
                f"Direction correct: {direction_correct}, "
                f"Score: {pattern.pattern_score:.3f}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating pattern outcome: {e}")
            self.db.rollback()
            return False

    async def generate_alpha_signal(
        self,
        symbol: str,
        signal_type: str,
        llm_model: str,
        direction: str,
        confidence: float,
        expected_alpha: float,
        time_horizon_days: int,
        reasoning: str,
        technical_setup: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> str:
        """
        Generate and store an alpha signal
        
        Returns:
            Signal ID
        """
        try:
            signal_id = f"{symbol}_{signal_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate expiration date
            expires_at = datetime.utcnow() + timedelta(days=time_horizon_days)
            
            # Create alpha signal record
            alpha_signal = AlphaSignal(
                signal_id=signal_id,
                signal_type=signal_type,
                symbol=symbol,
                generated_by_llm=llm_model,
                generation_method="single",  # Will be updated for ensemble
                confidence_score=confidence,
                direction=direction,
                strength=confidence,  # Using confidence as strength for now
                expected_alpha=expected_alpha,
                time_horizon_days=time_horizon_days,
                status="active",
                reasoning=reasoning,
                technical_setup=technical_setup,
                market_conditions=market_conditions,
                expires_at=expires_at
            )
            
            self.db.add(alpha_signal)
            self.db.commit()
            self.db.refresh(alpha_signal)
            
            self.logger.info(
                f"Alpha signal generated: {signal_id} - "
                f"{direction} {symbol} with {expected_alpha:.3f} expected alpha"
            )
            
            return signal_id
            
        except Exception as e:
            self.logger.error(f"Error generating alpha signal: {e}")
            self.db.rollback()
            raise

    async def get_llm_alpha_performance(
        self, 
        llm_model: str, 
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get LLM performance metrics focused on alpha generation
        
        Returns:
            Performance metrics dictionary
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Get recent pattern performance
            patterns = self.db.query(PatternPerformance).filter(
                and_(
                    PatternPerformance.detected_by_llm == llm_model,
                    PatternPerformance.detection_timestamp >= cutoff_date,
                    PatternPerformance.outcome_verified == True
                )
            ).all()
            
            if not patterns:
                return {
                    "llm_model": llm_model,
                    "period_days": days_back,
                    "total_patterns": 0,
                    "alpha_performance": {},
                    "error": "No verified patterns found"
                }
            
            # Calculate alpha metrics
            total_alpha = sum(p.alpha_generated or 0 for p in patterns)
            successful_patterns = len([p for p in patterns if (p.alpha_generated or 0) > 0])
            direction_correct = sum(p.prediction_accuracy or 0 for p in patterns)
            
            avg_confidence = np.mean([p.detection_confidence for p in patterns])
            avg_alpha = total_alpha / len(patterns) if patterns else 0
            success_rate = successful_patterns / len(patterns) if patterns else 0
            direction_accuracy = direction_correct / len(patterns) if patterns else 0
            
            # Calculate Sharpe ratio (simplified)
            alpha_returns = [p.alpha_generated or 0 for p in patterns]
            avg_return = np.mean(alpha_returns)
            return_std = np.std(alpha_returns) if len(alpha_returns) > 1 else 0.1
            sharpe_estimate = avg_return / return_std if return_std > 0 else 0
            
            # Pattern type breakdown
            pattern_breakdown = {}
            for pattern in patterns:
                pattern_type = pattern.pattern_type
                if pattern_type not in pattern_breakdown:
                    pattern_breakdown[pattern_type] = {
                        "count": 0,
                        "total_alpha": 0,
                        "success_rate": 0
                    }
                
                pattern_breakdown[pattern_type]["count"] += 1
                pattern_breakdown[pattern_type]["total_alpha"] += pattern.alpha_generated or 0
                if (pattern.alpha_generated or 0) > 0:
                    pattern_breakdown[pattern_type]["success_rate"] += 1
            
            # Calculate success rates
            for pattern_type in pattern_breakdown:
                pb = pattern_breakdown[pattern_type]
                pb["avg_alpha"] = pb["total_alpha"] / pb["count"] if pb["count"] > 0 else 0
                pb["success_rate"] = pb["success_rate"] / pb["count"] if pb["count"] > 0 else 0
            
            return {
                "llm_model": llm_model,
                "period_days": days_back,
                "total_patterns": len(patterns),
                "alpha_performance": {
                    "total_alpha_generated": round(total_alpha, 4),
                    "avg_alpha_per_pattern": round(avg_alpha, 4),
                    "success_rate": round(success_rate, 3),
                    "direction_accuracy": round(direction_accuracy, 3),
                    "estimated_sharpe": round(sharpe_estimate, 3),
                    "avg_confidence": round(avg_confidence, 3)
                },
                "pattern_specialization": pattern_breakdown,
                "recent_trends": self._calculate_recent_trends(patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting LLM alpha performance: {e}")
            return {"error": str(e)}

    async def get_top_alpha_patterns(
        self, 
        limit: int = 10,
        min_occurrences: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top alpha-generating patterns across all LLMs
        
        Returns:
            List of top patterns with performance metrics
        """
        try:
            # Query for pattern learning records with good alpha generation
            top_patterns = self.db.query(PatternLearning).filter(
                and_(
                    PatternLearning.total_occurrences >= min_occurrences,
                    PatternLearning.avg_alpha_generated > self.alpha_config["alpha_significance_threshold"]
                )
            ).order_by(desc(PatternLearning.avg_alpha_generated)).limit(limit).all()
            
            patterns_list = []
            for pattern in top_patterns:
                patterns_list.append({
                    "pattern_family": pattern.pattern_family,
                    "pattern_variant": pattern.pattern_variant,
                    "total_occurrences": pattern.total_occurrences,
                    "success_rate": round(pattern.success_rate, 3),
                    "avg_alpha_generated": round(pattern.avg_alpha_generated, 4),
                    "alpha_consistency": round(pattern.alpha_consistency or 0, 3),
                    "best_performing_llm": pattern.best_performing_llm,
                    "optimal_conditions": {
                        "market_regime": pattern.preferred_market_regime,
                        "volume_profile": pattern.preferred_volume_profile,
                        "confidence_threshold": pattern.optimal_confidence_threshold
                    },
                    "learning_confidence": round(pattern.confidence_in_learning or 0, 3)
                })
            
            return patterns_list
            
        except Exception as e:
            self.logger.error(f"Error getting top alpha patterns: {e}")
            return []

    def _generate_pattern_signature(self, detection: PatternDetection) -> str:
        """Generate unique signature for pattern deduplication"""
        return f"{detection.pattern_type}_{detection.symbol}_{detection.timeframe}_{detection.predicted_direction}"

    def _calculate_magnitude_accuracy(
        self, 
        predicted: Optional[float], 
        actual: float
    ) -> float:
        """Calculate accuracy of magnitude prediction"""
        if predicted is None:
            return 0.0
        
        if predicted == 0:
            return 1.0 if abs(actual) < 0.01 else 0.0
        
        error = abs(predicted - actual) / abs(predicted)
        return max(0.0, 1.0 - error)

    def _calculate_pattern_score(
        self, 
        pattern: PatternPerformance, 
        alpha_metrics: AlphaMetrics
    ) -> float:
        """Calculate overall pattern score combining multiple factors"""
        # Weights for different factors
        alpha_weight = 0.4
        direction_weight = 0.3
        magnitude_weight = 0.2
        confidence_weight = 0.1
        
        # Normalize alpha (assuming max reasonable alpha is 20%)
        alpha_score = min(1.0, abs(alpha_metrics.alpha_generated) / 0.20)
        
        # Direction accuracy
        direction_score = pattern.prediction_accuracy or 0
        
        # Magnitude accuracy
        magnitude_score = pattern.magnitude_accuracy or 0
        
        # Confidence calibration (how well confidence matched outcome)
        confidence_score = pattern.detection_confidence if (alpha_metrics.alpha_generated > 0) else (1 - pattern.detection_confidence)
        
        total_score = (
            alpha_score * alpha_weight +
            direction_score * direction_weight +
            magnitude_score * magnitude_weight +
            confidence_score * confidence_weight
        )
        
        return min(1.0, max(0.0, total_score))

    def _get_task_type_from_pattern(self, pattern_type: str) -> TaskType:
        """Map pattern type to TaskType enum"""
        pattern_mapping = {
            "momentum": TaskType.PATTERN_RECOGNITION,
            "reversal": TaskType.PATTERN_RECOGNITION,
            "breakout": TaskType.PATTERN_RECOGNITION,
            "alpha_analysis": TaskType.PATTERN_RECOGNITION,
            "risk_pattern": TaskType.RISK_ASSESSMENT,
            "volatility_pattern": TaskType.RISK_ASSESSMENT,
            "strategy_pattern": TaskType.STRATEGY_OPTIMIZATION
        }
        
        return pattern_mapping.get(pattern_type, TaskType.PATTERN_RECOGNITION)

    async def _update_llm_performance_metrics(
        self,
        llm_model: str,
        task_type: TaskType,
        success: bool,
        alpha_generated: float,
        confidence: float,
        alpha_metrics: AlphaMetrics
    ):
        """Update LLM performance metrics in database"""
        try:
            # Get or create today's metrics record
            today = datetime.utcnow().date()
            
            metrics = self.db.query(LLMPerformanceMetrics).filter(
                and_(
                    LLMPerformanceMetrics.llm_model == llm_model,
                    LLMPerformanceMetrics.task_type == task_type.value,
                    func.date(LLMPerformanceMetrics.measurement_date) == today,
                    LLMPerformanceMetrics.measurement_period == "daily"
                )
            ).first()
            
            if not metrics:
                metrics = LLMPerformanceMetrics(
                    llm_model=llm_model,
                    task_type=task_type.value,
                    measurement_date=datetime.utcnow(),
                    measurement_period="daily"
                )
                self.db.add(metrics)
            
            # Update metrics
            metrics.total_predictions += 1
            if success:
                metrics.successful_predictions += 1
            
            metrics.success_rate = metrics.successful_predictions / metrics.total_predictions
            
            # Update alpha metrics with moving average
            alpha_weight = 1.0 / metrics.total_predictions
            metrics.total_alpha_generated += alpha_generated
            metrics.avg_alpha_per_prediction = metrics.total_alpha_generated / metrics.total_predictions
            
            # Update other metrics (simplified moving average)
            if metrics.avg_sharpe_ratio == 0:
                metrics.avg_sharpe_ratio = alpha_metrics.sharpe_ratio
            else:
                metrics.avg_sharpe_ratio = (
                    (1 - alpha_weight) * metrics.avg_sharpe_ratio + 
                    alpha_weight * alpha_metrics.sharpe_ratio
                )
            
            self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Error updating LLM performance metrics: {e}")

    async def _update_pattern_learning(
        self, 
        pattern: PatternPerformance, 
        alpha_metrics: AlphaMetrics
    ):
        """Update pattern learning insights"""
        try:
            # Get or create pattern learning record
            learning = self.db.query(PatternLearning).filter(
                PatternLearning.pattern_family == pattern.pattern_type
            ).first()
            
            if not learning:
                learning = PatternLearning(
                    pattern_family=pattern.pattern_type,
                    pattern_variant=pattern.pattern_subtype
                )
                self.db.add(learning)
            
            # Update learning metrics
            learning.total_occurrences += 1
            if alpha_metrics.alpha_generated > 0:
                learning.successful_occurrences += 1
            
            learning.success_rate = learning.successful_occurrences / learning.total_occurrences
            
            # Update alpha metrics with exponential moving average
            alpha_weight = 0.1
            if learning.avg_alpha_generated == 0:
                learning.avg_alpha_generated = alpha_metrics.alpha_generated
            else:
                learning.avg_alpha_generated = (
                    (1 - alpha_weight) * learning.avg_alpha_generated + 
                    alpha_weight * alpha_metrics.alpha_generated
                )
            
            # Update LLM performance ranking for this pattern
            if not learning.llm_performance_ranking:
                learning.llm_performance_ranking = {}
            
            llm_model = pattern.detected_by_llm
            if llm_model not in learning.llm_performance_ranking:
                learning.llm_performance_ranking[llm_model] = {
                    "count": 0,
                    "total_alpha": 0,
                    "avg_alpha": 0
                }
            
            llm_perf = learning.llm_performance_ranking[llm_model]
            llm_perf["count"] += 1
            llm_perf["total_alpha"] += alpha_metrics.alpha_generated
            llm_perf["avg_alpha"] = llm_perf["total_alpha"] / llm_perf["count"]
            
            # Update best performing LLM
            best_llm = max(
                learning.llm_performance_ranking.keys(),
                key=lambda llm: learning.llm_performance_ranking[llm]["avg_alpha"]
            )
            learning.best_performing_llm = best_llm
            
            learning.last_learning_update = datetime.utcnow()
            learning.learning_sample_size = learning.total_occurrences
            learning.confidence_in_learning = min(1.0, learning.total_occurrences / 50.0)  # Confidence increases with sample size
            
            self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Error updating pattern learning: {e}")

    def _calculate_recent_trends(self, patterns: List[PatternPerformance]) -> Dict[str, Any]:
        """Calculate recent performance trends"""
        if len(patterns) < 5:
            return {"insufficient_data": True}
        
        # Sort patterns by detection time
        sorted_patterns = sorted(patterns, key=lambda p: p.detection_timestamp)
        
        # Split into first and second half
        mid_point = len(sorted_patterns) // 2
        first_half = sorted_patterns[:mid_point]
        second_half = sorted_patterns[mid_point:]
        
        first_half_alpha = np.mean([p.alpha_generated or 0 for p in first_half])
        second_half_alpha = np.mean([p.alpha_generated or 0 for p in second_half])
        
        first_half_accuracy = np.mean([p.prediction_accuracy or 0 for p in first_half])
        second_half_accuracy = np.mean([p.prediction_accuracy or 0 for p in second_half])
        
        return {
            "alpha_trend": "improving" if second_half_alpha > first_half_alpha else "declining",
            "alpha_change": round(second_half_alpha - first_half_alpha, 4),
            "accuracy_trend": "improving" if second_half_accuracy > first_half_accuracy else "declining",
            "accuracy_change": round(second_half_accuracy - first_half_accuracy, 3),
            "sample_size": len(patterns)
        }

    async def _schedule_alpha_verification(self, pattern_id: str):
        """Schedule pattern verification based on time horizon"""
        # This would be implemented with a task queue (Celery/Redis)
        # For now, just log the scheduling
        self.logger.info(f"Scheduled alpha verification for pattern {pattern_id}")
        # TODO: Implement actual scheduling with background task system