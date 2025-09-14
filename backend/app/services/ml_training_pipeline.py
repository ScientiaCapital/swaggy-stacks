"""
ML Training Pipeline for Chinese LLM models with feedback learning
Handles model versioning, performance tracking, and continuous improvement
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import structlog
from sqlalchemy.orm import Session

from app.analysis.llm_predictors import get_llm_predictor
from app.models.indicator_performance import (
    MLModelVersion,
    MLPrediction,
)

logger = structlog.get_logger()


class MLTrainingPipeline:
    """
    Advanced ML training pipeline for Chinese LLM models with feedback learning
    Handles model versioning, performance tracking, and continuous improvement
    """

    def __init__(self, db: Session):
        self.db = db
        self.llm_predictor = None
        self.training_config = {
            "feedback_threshold": 0.8,  # Min confidence to use for training
            "batch_size": 50,  # Predictions to process in batch
            "model_retention_days": 30,  # How long to keep old model versions
            "performance_window": 100,  # Recent predictions to evaluate
            "retraining_threshold": 0.05,  # Performance drop to trigger retraining
        }

    async def initialize_llm_predictor(self):
        """Initialize LLM predictor with current active models"""
        try:
            self.llm_predictor = await get_llm_predictor()
            logger.info(
                "ML training pipeline initialized",
                available_models=(
                    self.llm_predictor.available_models if self.llm_predictor else 0
                ),
            )
        except Exception as e:
            logger.error("Failed to initialize LLM predictor", error=str(e))

    async def create_model_version(
        self,
        model_name: str,
        model_type: str,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any] = None,
    ) -> MLModelVersion:
        """Create new model version entry"""
        try:
            # Generate version string
            existing_versions = (
                self.db.query(MLModelVersion)
                .filter(MLModelVersion.model_name == model_name)
                .count()
            )

            version = f"v{existing_versions + 1}.0"

            model_version = MLModelVersion(
                model_name=model_name,
                model_type=model_type,
                version=version,
                model_config=model_config,
                training_config=training_config or {},
                is_active=True,
                is_production=False,
            )

            # Deactivate previous versions
            self.db.query(MLModelVersion).filter(
                MLModelVersion.model_name == model_name,
                MLModelVersion.is_active == True,
            ).update({"is_active": False})

            self.db.add(model_version)
            self.db.commit()
            self.db.refresh(model_version)

            logger.info(
                "Created new model version", model_name=model_name, version=version
            )
            return model_version

        except Exception as e:
            logger.error("Error creating model version", error=str(e))
            self.db.rollback()
            raise

    async def record_prediction(
        self,
        model_version_id: int,
        symbol: str,
        prediction_horizon: int,
        predicted_direction: str,
        predicted_return: float,
        confidence_score: float,
        individual_predictions: Dict[str, Any] = None,
        market_conditions: Dict[str, Any] = None,
        technical_indicators: Dict[str, Any] = None,
    ) -> MLPrediction:
        """Record ML prediction for later outcome tracking"""
        try:
            prediction = MLPrediction(
                model_version_id=model_version_id,
                symbol=symbol,
                prediction_time=datetime.now(),
                prediction_horizon=prediction_horizon,
                predicted_direction=predicted_direction,
                predicted_return=predicted_return,
                confidence_score=confidence_score,
                individual_predictions=individual_predictions or {},
                market_conditions=market_conditions or {},
                technical_indicators=technical_indicators or {},
            )

            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)

            logger.info(
                "Recorded ML prediction",
                symbol=symbol,
                direction=predicted_direction,
                confidence=confidence_score,
            )
            return prediction

        except Exception as e:
            logger.error("Error recording prediction", error=str(e))
            self.db.rollback()
            raise

    async def update_prediction_outcome(
        self, prediction_id: int, actual_direction: str, actual_return: float
    ) -> bool:
        """Update actual outcome for a prediction"""
        try:
            prediction = (
                self.db.query(MLPrediction)
                .filter(MLPrediction.id == prediction_id)
                .first()
            )

            if not prediction:
                logger.warning("Prediction not found", prediction_id=prediction_id)
                return False

            # Calculate prediction accuracy
            direction_correct = prediction.predicted_direction == actual_direction

            # Calculate prediction error
            prediction_error = (
                abs(prediction.predicted_return - actual_return)
                if prediction.predicted_return
                else None
            )

            # Update prediction
            prediction.actual_direction = actual_direction
            prediction.actual_return = actual_return
            prediction.outcome_recorded_at = datetime.now()
            prediction.prediction_error = prediction_error
            prediction.was_correct = direction_correct

            # Set learning weight based on confidence and accuracy
            if direction_correct and prediction.confidence_score > 0.7:
                prediction.learning_weight = (
                    1.2  # Higher weight for confident correct predictions
                )
            elif direction_correct and prediction.confidence_score < 0.5:
                prediction.learning_weight = 0.8  # Lower weight for lucky guesses
            elif not direction_correct and prediction.confidence_score > 0.7:
                prediction.learning_weight = (
                    1.5  # Higher weight to learn from confident mistakes
                )
            else:
                prediction.learning_weight = 1.0

            self.db.commit()

            # Check if model needs retraining
            await self._check_retraining_trigger(prediction.model_version_id)

            logger.info(
                "Updated prediction outcome",
                prediction_id=prediction_id,
                was_correct=direction_correct,
            )
            return True

        except Exception as e:
            logger.error("Error updating prediction outcome", error=str(e))
            self.db.rollback()
            return False

    async def get_model_performance_report(
        self, model_name: str = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive model performance report"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Query predictions
            query = (
                self.db.query(MLPrediction)
                .join(MLModelVersion)
                .filter(MLPrediction.prediction_time >= cutoff_date)
            )

            if model_name:
                query = query.filter(MLModelVersion.model_name == model_name)

            predictions = query.all()

            if not predictions:
                return {"error": "No predictions found for specified criteria"}

            # Group by model and calculate performance
            model_performance = {}
            for prediction in predictions:
                model_key = f"{prediction.model_version.model_name}_v{prediction.model_version.version}"

                if model_key not in model_performance:
                    model_performance[model_key] = {
                        "predictions": [],
                        "model_info": {
                            "name": prediction.model_version.model_name,
                            "version": prediction.model_version.version,
                            "type": prediction.model_version.model_type,
                            "is_active": prediction.model_version.is_active,
                        },
                    }

                model_performance[model_key]["predictions"].append(prediction)

            # Calculate performance metrics for each model
            performance_report = {}
            for model_key, data in model_performance.items():
                predictions_list = data["predictions"]
                completed_predictions = [
                    p for p in predictions_list if p.was_correct is not None
                ]

                if completed_predictions:
                    accuracy = sum(
                        1 for p in completed_predictions if p.was_correct
                    ) / len(completed_predictions)
                    avg_confidence = np.mean(
                        [float(p.confidence_score) for p in completed_predictions]
                    )
                    returns = [
                        float(p.actual_return)
                        for p in completed_predictions
                        if p.actual_return
                    ]
                    avg_return = np.mean(returns) if returns else 0.0

                    performance_report[model_key] = {
                        "model_info": data["model_info"],
                        "total_predictions": len(predictions_list),
                        "completed_predictions": len(completed_predictions),
                        "accuracy": accuracy,
                        "avg_confidence": avg_confidence,
                        "avg_return": avg_return,
                        "prediction_frequency": len(predictions_list) / days_back,
                    }

            return {
                "report_period_days": days_back,
                "total_models": len(performance_report),
                "model_performance": performance_report,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("Error generating model performance report", error=str(e))
            return {"error": f"Failed to generate report: {str(e)}"}

    async def _check_retraining_trigger(self, model_version_id: int):
        """Check if model performance has degraded and needs retraining"""
        try:
            # Get recent predictions for this model
            recent_predictions = (
                self.db.query(MLPrediction)
                .filter(
                    MLPrediction.model_version_id == model_version_id,
                    MLPrediction.was_correct.isnot(None),
                )
                .order_by(MLPrediction.prediction_time.desc())
                .limit(self.training_config["performance_window"])
                .all()
            )

            if len(recent_predictions) < 20:  # Need minimum samples
                return

            # Calculate recent accuracy
            correct_predictions = sum(1 for p in recent_predictions if p.was_correct)
            accuracy = correct_predictions / len(recent_predictions)

            # Get model version baseline
            model_version = (
                self.db.query(MLModelVersion)
                .filter(MLModelVersion.id == model_version_id)
                .first()
            )

            if not model_version:
                return

            baseline_accuracy = (
                model_version.production_metrics.get("accuracy", 0.6)
                if model_version.production_metrics
                else 0.6
            )

            # Check if performance dropped significantly
            performance_drop = baseline_accuracy - accuracy
            if performance_drop > self.training_config["retraining_threshold"]:
                logger.warning(
                    "Model performance degraded - scheduling retraining",
                    model_version_id=model_version_id,
                    current_accuracy=accuracy,
                    baseline_accuracy=baseline_accuracy,
                    performance_drop=performance_drop,
                )

                await self._schedule_model_retraining(model_version_id)

        except Exception as e:
            logger.error("Error checking retraining trigger", error=str(e))

    async def _schedule_model_retraining(self, model_version_id: int):
        """Schedule model retraining based on feedback data"""
        try:
            # Get feedback data for retraining
            feedback_predictions = (
                self.db.query(MLPrediction)
                .filter(
                    MLPrediction.model_version_id == model_version_id,
                    MLPrediction.feedback_processed == False,
                    MLPrediction.was_correct.isnot(None),
                    MLPrediction.learning_weight
                    > 0.5,  # Only use high-quality feedback
                )
                .limit(500)
                .all()
            )  # Limit to prevent memory issues

            if len(feedback_predictions) < 50:
                logger.info(
                    "Insufficient feedback data for retraining",
                    available_samples=len(feedback_predictions),
                )
                return

            # Mark predictions as processed
            for prediction in feedback_predictions:
                prediction.feedback_processed = True

            self.db.commit()

            # Extract training data
            training_data = []
            for prediction in feedback_predictions:
                training_example = {
                    "symbol": prediction.symbol,
                    "market_conditions": prediction.market_conditions,
                    "technical_indicators": prediction.technical_indicators,
                    "actual_direction": prediction.actual_direction,
                    "actual_return": prediction.actual_return,
                    "learning_weight": float(prediction.learning_weight),
                }
                training_data.append(training_example)

            # This would integrate with actual LLM fine-tuning process
            logger.info(
                "Prepared training data for model retraining",
                model_version_id=model_version_id,
                training_samples=len(training_data),
            )

            # Update model metrics
            await self._update_model_metrics(model_version_id, feedback_predictions)

        except Exception as e:
            logger.error("Error scheduling model retraining", error=str(e))
            self.db.rollback()

    async def _update_model_metrics(
        self, model_version_id: int, predictions: List[MLPrediction]
    ):
        """Update production metrics for model version"""
        try:
            model_version = (
                self.db.query(MLModelVersion)
                .filter(MLModelVersion.id == model_version_id)
                .first()
            )

            if not model_version:
                return

            # Calculate metrics
            correct_predictions = [p for p in predictions if p.was_correct]
            accuracy = (
                len(correct_predictions) / len(predictions) if predictions else 0.0
            )
            avg_confidence = np.mean([float(p.confidence_score) for p in predictions])

            # Update metrics
            production_metrics = model_version.production_metrics or {}
            production_metrics.update(
                {
                    "accuracy": accuracy,
                    "avg_confidence": avg_confidence,
                    "total_predictions": len(predictions),
                    "last_updated": datetime.now().isoformat(),
                }
            )

            model_version.production_metrics = production_metrics
            self.db.commit()

            logger.info(
                "Updated model production metrics",
                model_version_id=model_version_id,
                accuracy=accuracy,
            )

        except Exception as e:
            logger.error("Error updating model metrics", error=str(e))

    async def cleanup_old_model_versions(self):
        """Clean up old model versions based on retention policy"""
        try:
            retention_date = datetime.now() - timedelta(
                days=self.training_config["model_retention_days"]
            )

            # Find old model versions (keep production and active models)
            old_versions = (
                self.db.query(MLModelVersion)
                .filter(
                    MLModelVersion.created_at < retention_date,
                    MLModelVersion.is_active == False,
                    MLModelVersion.is_production == False,
                )
                .all()
            )

            for version in old_versions:
                # Check if there are any references to this version
                prediction_count = (
                    self.db.query(MLPrediction)
                    .filter(MLPrediction.model_version_id == version.id)
                    .count()
                )

                if prediction_count == 0:  # Safe to delete
                    self.db.delete(version)
                    logger.info(
                        "Deleted old model version",
                        model_name=version.model_name,
                        version=version.version,
                    )

            self.db.commit()

        except Exception as e:
            logger.error("Error cleaning up old model versions", error=str(e))
            self.db.rollback()
