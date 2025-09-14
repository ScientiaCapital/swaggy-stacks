"""
Base classes and interfaces for unsupervised learning in trading system.

Provides abstract base classes and common functionality for all unsupervised learning
implementations, following existing project patterns from MarkovSystem.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

from app.core.exceptions import TradingError

warnings.filterwarnings("ignore")
logger = structlog.get_logger()


class UnsupervisedModelError(TradingError):
    """Custom exception for unsupervised learning models"""
    pass


class BaseUnsupervisedModel(ABC):
    """
    Abstract base class for all unsupervised learning models in the trading system.

    Provides common interface and functionality following MarkovSystem patterns.
    All models must implement fit(), predict(), and update() methods for real-time trading.
    """

    def __init__(self, lookback_period: int = 100, enable_scaling: bool = True):
        """
        Initialize base unsupervised model.

        Args:
            lookback_period: Number of periods to look back for analysis
            enable_scaling: Whether to apply StandardScaler preprocessing
        """
        self.lookback_period = lookback_period
        self.enable_scaling = enable_scaling
        self.scaler = StandardScaler() if enable_scaling else None
        self.is_fitted = False
        self.feature_names = None
        self.model_params = {}

        logger.info(
            "BaseUnsupervisedModel initialized",
            lookback_period=lookback_period,
            enable_scaling=enable_scaling,
            model_type=self.__class__.__name__
        )

    @abstractmethod
    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'BaseUnsupervisedModel':
        """
        Fit the unsupervised model to training data.

        Args:
            data: Training data as DataFrame or numpy array

        Returns:
            Self for method chaining

        Raises:
            UnsupervisedModelError: If fitting fails
        """
        pass

    @abstractmethod
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make predictions using the fitted model.

        Args:
            data: Input data for prediction

        Returns:
            Predictions (format depends on specific model)

        Raises:
            UnsupervisedModelError: If model not fitted or prediction fails
        """
        pass

    @abstractmethod
    def update(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Incrementally update the model with new data for real-time learning.

        Args:
            data: New data to incorporate into the model

        Raises:
            UnsupervisedModelError: If update fails
        """
        pass

    def _preprocess_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Preprocess data using StandardScaler if enabled.

        Args:
            data: Input data to preprocess

        Returns:
            Preprocessed numpy array
        """
        if isinstance(data, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = list(data.columns)
            data_array = data.values
        else:
            data_array = np.asarray(data)

        if self.enable_scaling and self.scaler is not None:
            if hasattr(self.scaler, 'n_features_in_'):
                return self.scaler.transform(data_array)
            else:
                return self.scaler.fit_transform(data_array)

        return data_array

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'lookback_period': self.lookback_period,
            'enable_scaling': self.enable_scaling,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            **self.model_params
        }

    def _validate_fitted(self) -> None:
        """Validate that model is fitted"""
        if not self.is_fitted:
            raise UnsupervisedModelError(f"{self.__class__.__name__} model must be fitted before prediction")


class IncrementalLearner(ABC):
    """
    Interface for models supporting incremental/online learning.

    Essential for real-time trading where models must adapt to new market data
    without full retraining.
    """

    @abstractmethod
    def partial_fit(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Incrementally fit model with a batch of new data.

        Args:
            data: New batch of data to learn from
        """
        pass

    @abstractmethod
    def get_learning_rate(self) -> float:
        """
        Get current learning rate for incremental updates.

        Returns:
            Current learning rate
        """
        pass

    @abstractmethod
    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Set learning rate for incremental updates.

        Args:
            learning_rate: New learning rate to use
        """
        pass


class ClusteringMetrics:
    """
    Comprehensive clustering evaluation metrics for model validation.

    Provides standard clustering metrics used across all clustering algorithms
    to ensure consistent evaluation and comparison.
    """

    @staticmethod
    def silhouette_score_safe(data: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette score with error handling.

        Args:
            data: Input data used for clustering
            labels: Cluster labels assigned to each data point

        Returns:
            Silhouette score between -1 and 1 (higher is better)
        """
        try:
            if len(np.unique(labels)) < 2:
                return 0.0  # Cannot calculate silhouette with single cluster
            return silhouette_score(data, labels)
        except Exception as e:
            logger.warning("Failed to calculate silhouette score", error=str(e))
            return 0.0

    @staticmethod
    def davies_bouldin_score_safe(data: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Davies-Bouldin score with error handling.

        Args:
            data: Input data used for clustering
            labels: Cluster labels assigned to each data point

        Returns:
            Davies-Bouldin score (lower is better)
        """
        try:
            if len(np.unique(labels)) < 2:
                return float('inf')  # Worst possible score for single cluster
            return davies_bouldin_score(data, labels)
        except Exception as e:
            logger.warning("Failed to calculate Davies-Bouldin score", error=str(e))
            return float('inf')

    @staticmethod
    def inertia_score(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Calculate within-cluster sum of squares (inertia).

        Args:
            data: Input data used for clustering
            labels: Cluster labels assigned to each data point
            centroids: Cluster centroids

        Returns:
            Inertia score (lower is better)
        """
        try:
            total_inertia = 0.0
            for i, centroid in enumerate(centroids):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    distances = np.sum((cluster_points - centroid) ** 2, axis=1)
                    total_inertia += np.sum(distances)
            return total_inertia
        except Exception as e:
            logger.warning("Failed to calculate inertia score", error=str(e))
            return float('inf')

    @classmethod
    def comprehensive_evaluation(
        cls,
        data: np.ndarray,
        labels: np.ndarray,
        centroids: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive clustering evaluation metrics.

        Args:
            data: Input data used for clustering
            labels: Cluster labels assigned to each data point
            centroids: Optional cluster centroids for inertia calculation

        Returns:
            Dictionary containing all available metrics
        """
        metrics = {
            'silhouette_score': cls.silhouette_score_safe(data, labels),
            'davies_bouldin_score': cls.davies_bouldin_score_safe(data, labels),
            'n_clusters': len(np.unique(labels)),
            'n_samples': len(data)
        }

        if centroids is not None:
            metrics['inertia'] = cls.inertia_score(data, labels, centroids)

        logger.info("Clustering evaluation completed", **metrics)
        return metrics