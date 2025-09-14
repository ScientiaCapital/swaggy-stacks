"""
ML Model Service - Extracted from BacktestService

Handles machine learning model management, predictions, and performance tracking
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.indicator_performance import (
    MLPrediction as MLModelPrediction,
    MLModelVersion,
    ParameterOptimization,
)
from app.monitoring.metrics import PrometheusMetrics

logger = logging.getLogger(__name__)


class MLModelService:
    """Service for managing ML models, predictions, and performance tracking"""

    def __init__(self, db: Session = None, metrics: PrometheusMetrics = None):
        self.db = db or next(get_db())
        self.metrics = metrics or PrometheusMetrics()
        self.ml_pipeline = None  # Will be initialized when needed
        logger.info("MLModelService initialized")

    async def initialize_ml_models(self, config: Dict = None) -> bool:
        """Initialize ML prediction models"""
        try:
            # TODO: Initialize actual ML models based on config
            # This is a placeholder for ML pipeline initialization
            self.ml_pipeline = {
                "ensemble_model": None,
                "trend_predictor": None,
                "volatility_model": None,
                "sentiment_analyzer": None,
            }

            logger.info("ML models initialized (placeholder implementation)")
            return True

        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            return False

    async def track_ml_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction_type: str,
        prediction_value: float,
        confidence: float,
        features_used: Dict = None,
        model_version: str = "1.0",
    ) -> Optional[MLModelPrediction]:
        """Track a machine learning prediction"""
        try:
            # Get or create model version
            model_version_obj = await self._get_or_create_model_version(
                model_name, model_version
            )

            if not model_version_obj:
                logger.error(f"Could not create/find model version for {model_name}")
                return None

            prediction = MLModelPrediction(
                model_version_id=model_version_obj.id,
                symbol=symbol,
                prediction_type=prediction_type,
                prediction_value=prediction_value,
                confidence=confidence,
                features_used=features_used or {},
                prediction_timestamp=datetime.utcnow(),
            )

            self.db.add(prediction)
            self.db.commit()

            # Update metrics
            if self.metrics:
                self.metrics.ml_predictions_total.labels(
                    model=model_name, prediction_type=prediction_type
                ).inc()

            logger.info(f"Tracked ML prediction for {model_name} on {symbol}")
            return prediction

        except Exception as e:
            logger.error(f"Error tracking ML prediction: {e}")
            self.db.rollback()
            return None

    async def update_ml_prediction_outcome(
        self,
        prediction_id: int,
        actual_value: float,
        outcome_timestamp: datetime = None,
    ) -> bool:
        """Update the actual outcome of an ML prediction"""
        try:
            prediction = (
                self.db.query(MLModelPrediction)
                .filter(MLModelPrediction.id == prediction_id)
                .first()
            )

            if not prediction:
                logger.warning(f"ML prediction {prediction_id} not found")
                return False

            prediction.actual_value = actual_value
            prediction.outcome_timestamp = outcome_timestamp or datetime.utcnow()

            # Calculate accuracy/error metrics
            if prediction.prediction_value is not None:
                error = abs(actual_value - prediction.prediction_value)
                relative_error = error / abs(actual_value) if actual_value != 0 else 0
                prediction.prediction_error = error
                prediction.accuracy_score = max(0, 1 - relative_error)

            self.db.commit()

            # Update metrics
            if self.metrics and prediction.model_version:
                model_name = prediction.model_version.model_name
                self.metrics.ml_prediction_accuracy.labels(model=model_name).observe(
                    prediction.accuracy_score or 0
                )

            logger.info(f"Updated ML prediction outcome for {prediction_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating ML prediction outcome: {e}")
            self.db.rollback()
            return False

    async def get_ml_model_performance(
        self, model_name: str = None, days_back: int = 30, prediction_type: str = None
    ) -> Dict[str, Any]:
        """Get performance metrics for ML models"""
        try:
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            # Build query
            query = (
                self.db.query(MLModelPrediction)
                .join(MLModelVersion)
                .filter(
                    MLModelPrediction.prediction_timestamp >= cutoff_date,
                    MLModelPrediction.actual_value.isnot(None),
                )
            )

            if model_name:
                query = query.filter(MLModelVersion.model_name == model_name)
            if prediction_type:
                query = query.filter(
                    MLModelPrediction.prediction_type == prediction_type
                )

            predictions = query.all()

            if not predictions:
                return {"message": "No ML prediction data found", "models": []}

            # Calculate performance by model
            model_stats = {}
            for pred in predictions:
                model_name = pred.model_version.model_name
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        "total_predictions": 0,
                        "accuracy_sum": 0.0,
                        "error_sum": 0.0,
                        "confidence_sum": 0.0,
                        "predictions": [],
                    }

                stats = model_stats[model_name]
                stats["total_predictions"] += 1
                stats["predictions"].append(pred)
                stats["confidence_sum"] += pred.confidence

                if pred.accuracy_score is not None:
                    stats["accuracy_sum"] += pred.accuracy_score
                if pred.prediction_error is not None:
                    stats["error_sum"] += pred.prediction_error

            # Calculate final metrics
            performance_data = []
            for model_name, stats in model_stats.items():
                total = stats["total_predictions"]

                performance_data.append(
                    {
                        "model_name": model_name,
                        "total_predictions": total,
                        "avg_accuracy": (
                            round(stats["accuracy_sum"] / total, 4) if total > 0 else 0
                        ),
                        "avg_error": (
                            round(stats["error_sum"] / total, 4) if total > 0 else 0
                        ),
                        "avg_confidence": (
                            round(stats["confidence_sum"] / total, 4)
                            if total > 0
                            else 0
                        ),
                        "latest_prediction": max(
                            (p.prediction_timestamp for p in stats["predictions"]),
                            default=None,
                        ),
                    }
                )

            # Sort by average accuracy descending
            performance_data.sort(key=lambda x: x["avg_accuracy"], reverse=True)

            return {
                "period_days": days_back,
                "total_models": len(performance_data),
                "models": performance_data,
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting ML model performance: {e}")
            return {"error": str(e)}

    async def _get_or_create_model_version(
        self,
        model_name: str,
        version: str = "1.0",
        model_type: str = "ensemble",
        description: str = None,
    ) -> Optional[MLModelVersion]:
        """Get existing model version or create new one"""
        try:
            # Try to find existing version
            model_version = (
                self.db.query(MLModelVersion)
                .filter(
                    MLModelVersion.model_name == model_name,
                    MLModelVersion.version == version,
                )
                .first()
            )

            if model_version:
                return model_version

            # Create new version
            model_version = MLModelVersion(
                model_name=model_name,
                version=version,
                model_type=model_type,
                description=description or f"{model_name} version {version}",
                created_at=datetime.utcnow(),
                is_active=True,
            )

            self.db.add(model_version)
            self.db.commit()

            logger.info(f"Created new ML model version: {model_name} v{version}")
            return model_version

        except Exception as e:
            logger.error(f"Error creating model version: {e}")
            self.db.rollback()
            return None

    async def get_ensemble_model_performance(
        self, days_back: int = 30
    ) -> Dict[str, Any]:
        """Get performance metrics specifically for ensemble models"""
        return await self.get_ml_model_performance(
            model_name="ensemble", days_back=days_back
        )

    async def optimize_model_parameters(
        self,
        model_name: str,
        parameter_space: Dict,
        optimization_method: str = "bayesian",
        metric: str = "accuracy",
        max_iterations: int = 100,
    ) -> Optional[ParameterOptimization]:
        """Start parameter optimization for a model"""
        try:
            optimization = ParameterOptimization(
                optimization_id=f"{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                indicator_name=model_name,
                optimization_type="model",
                method=optimization_method,
                metric=metric,
                parameter_space=parameter_space,
                total_iterations=max_iterations,
                status="PENDING",
                created_at=datetime.utcnow(),
            )

            self.db.add(optimization)
            self.db.commit()

            logger.info(f"Started parameter optimization for {model_name}")
            return optimization

        except Exception as e:
            logger.error(f"Error starting parameter optimization: {e}")
            self.db.rollback()
            return None

    async def get_model_prediction_history(
        self, model_name: str, symbol: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent prediction history for a model"""
        try:
            query = (
                self.db.query(MLModelPrediction)
                .join(MLModelVersion)
                .filter(MLModelVersion.model_name == model_name)
            )

            if symbol:
                query = query.filter(MLModelPrediction.symbol == symbol)

            predictions = (
                query.order_by(MLModelPrediction.prediction_timestamp.desc())
                .limit(limit)
                .all()
            )

            history = []
            for pred in predictions:
                history.append(
                    {
                        "id": pred.id,
                        "symbol": pred.symbol,
                        "prediction_type": pred.prediction_type,
                        "prediction_value": pred.prediction_value,
                        "actual_value": pred.actual_value,
                        "confidence": pred.confidence,
                        "accuracy_score": pred.accuracy_score,
                        "prediction_timestamp": (
                            pred.prediction_timestamp.isoformat()
                            if pred.prediction_timestamp
                            else None
                        ),
                        "outcome_timestamp": (
                            pred.outcome_timestamp.isoformat()
                            if pred.outcome_timestamp
                            else None
                        ),
                    }
                )

            return history

        except Exception as e:
            logger.error(f"Error getting model prediction history: {e}")
            return []
