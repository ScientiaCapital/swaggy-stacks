"""
Core clustering algorithms optimized for real-time trading.

Implements k-means, hierarchical clustering, and DBSCAN with production-ready
optimizations including incremental learning and sub-second performance requirements.
"""

import warnings
from typing import Dict, Any, Optional, Union, List, Tuple
import time

import numpy as np
import pandas as pd
import structlog
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from .base import BaseUnsupervisedModel, IncrementalLearner, ClusteringMetrics, UnsupervisedModelError

warnings.filterwarnings("ignore")
logger = structlog.get_logger()


class EnhancedKMeans(BaseUnsupervisedModel, IncrementalLearner):
    """
    Enhanced k-means clustering using MiniBatchKMeans for speed.

    Optimized for real-time trading with adaptive cluster selection and
    incremental learning capabilities.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        max_clusters: int = 10,
        batch_size: int = 100,
        max_iter: int = 100,
        random_state: int = 42,
        lookback_period: int = 100,
        enable_scaling: bool = True
    ):
        """
        Initialize Enhanced K-Means clusterer.

        Args:
            n_clusters: Number of clusters (auto-selected if None)
            max_clusters: Maximum clusters for adaptive selection
            batch_size: Batch size for MiniBatch algorithm
            max_iter: Maximum iterations
            random_state: Random seed for reproducibility
            lookback_period: Number of periods for analysis
            enable_scaling: Whether to scale data
        """
        super().__init__(lookback_period, enable_scaling)

        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.learning_rate = 0.1

        self.model = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.optimal_k = None

        self.model_params.update({
            'n_clusters': n_clusters,
            'max_clusters': max_clusters,
            'batch_size': batch_size,
            'max_iter': max_iter,
            'random_state': random_state
        })

        logger.info(
            "EnhancedKMeans initialized",
            n_clusters=n_clusters,
            max_clusters=max_clusters,
            batch_size=batch_size
        )

    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """
        Find optimal number of clusters using elbow method.

        Args:
            data: Input data for clustering

        Returns:
            Optimal number of clusters
        """
        if self.n_clusters is not None:
            return self.n_clusters

        start_time = time.time()
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(self.max_clusters + 1, len(data) // 2))

        for k in k_range:
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_init=3  # Reduce for speed
            )
            labels = kmeans.fit_predict(data)
            inertias.append(kmeans.inertia_)

            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(data, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(-1)

        # Use elbow method with silhouette score validation
        if len(inertias) < 2:
            optimal_k = 2
        else:
            # Calculate elbow using percentage decrease
            decreases = [
                (inertias[i] - inertias[i + 1]) / inertias[i] * 100
                for i in range(len(inertias) - 1)
            ]

            # Find elbow point (where decrease rate drops significantly)
            elbow_idx = 0
            for i in range(1, len(decreases)):
                if decreases[i] < decreases[i - 1] * 0.3:  # 70% drop in improvement
                    elbow_idx = i
                    break

            # Validate with silhouette score
            candidate_k = k_range[elbow_idx]
            if silhouette_scores[elbow_idx] > 0.3:  # Good silhouette score
                optimal_k = candidate_k
            else:
                # Fall back to best silhouette score
                best_sil_idx = np.argmax(silhouette_scores)
                optimal_k = k_range[best_sil_idx]

        duration = time.time() - start_time
        logger.info(
            "Optimal clusters found",
            optimal_k=optimal_k,
            duration_ms=duration * 1000,
            tested_k_values=list(k_range)
        )

        return optimal_k

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'EnhancedKMeans':
        """
        Fit k-means model to training data.

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        start_time = time.time()

        try:
            # Preprocess data
            processed_data = self._preprocess_data(data)

            if len(processed_data) < 2:
                raise UnsupervisedModelError("Need at least 2 data points for clustering")

            # Find optimal number of clusters
            self.optimal_k = self._find_optimal_clusters(processed_data)

            # Fit model
            self.model = MiniBatchKMeans(
                n_clusters=self.optimal_k,
                batch_size=min(self.batch_size, len(processed_data)),
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_init=10
            )

            self.labels_ = self.model.fit_predict(processed_data)
            self.cluster_centers_ = self.model.cluster_centers_
            self.inertia_ = self.model.inertia_
            self.is_fitted = True

            duration = time.time() - start_time
            logger.info(
                "EnhancedKMeans fitted successfully",
                n_clusters=self.optimal_k,
                inertia=self.inertia_,
                duration_ms=duration * 1000,
                n_samples=len(processed_data)
            )

            return self

        except Exception as e:
            raise UnsupervisedModelError(f"K-means fitting failed: {str(e)}")

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            data: Input data for prediction

        Returns:
            Cluster labels
        """
        self._validate_fitted()

        try:
            processed_data = self._preprocess_data(data)
            labels = self.model.predict(processed_data)

            logger.debug(
                "K-means prediction completed",
                n_samples=len(processed_data),
                unique_labels=len(np.unique(labels))
            )

            return labels

        except Exception as e:
            raise UnsupervisedModelError(f"K-means prediction failed: {str(e)}")

    def update(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Incrementally update model with new data.

        Args:
            data: New data to incorporate
        """
        self._validate_fitted()

        try:
            processed_data = self._preprocess_data(data)
            self.model.partial_fit(processed_data)
            self.cluster_centers_ = self.model.cluster_centers_
            self.inertia_ = self.model.inertia_

            logger.debug(
                "K-means model updated",
                n_new_samples=len(processed_data),
                new_inertia=self.inertia_
            )

        except Exception as e:
            raise UnsupervisedModelError(f"K-means update failed: {str(e)}")

    def partial_fit(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """Alias for update method to match IncrementalLearner interface"""
        self.update(data)

    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.learning_rate

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set learning rate"""
        self.learning_rate = learning_rate


class HierarchicalClusterer(BaseUnsupervisedModel):
    """
    Hierarchical clustering with multiple linkage options.

    Optimized for discovering market regime hierarchies and nested patterns.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        linkage_method: str = 'ward',
        distance_threshold: Optional[float] = None,
        max_clusters: int = 10,
        lookback_period: int = 100,
        enable_scaling: bool = True
    ):
        """
        Initialize Hierarchical Clusterer.

        Args:
            n_clusters: Number of clusters (auto-selected if None)
            linkage_method: Linkage criteria ('ward', 'complete', 'average', 'single')
            distance_threshold: Distance threshold for automatic cluster detection
            max_clusters: Maximum clusters for adaptive selection
            lookback_period: Number of periods for analysis
            enable_scaling: Whether to scale data
        """
        super().__init__(lookback_period, enable_scaling)

        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.distance_threshold = distance_threshold
        self.max_clusters = max_clusters

        self.model = None
        self.labels_ = None
        self.linkage_matrix = None
        self.optimal_k = None

        self.model_params.update({
            'n_clusters': n_clusters,
            'linkage_method': linkage_method,
            'distance_threshold': distance_threshold,
            'max_clusters': max_clusters
        })

        logger.info(
            "HierarchicalClusterer initialized",
            linkage_method=linkage_method,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold
        )

    def _find_optimal_clusters_hierarchical(self, data: np.ndarray) -> int:
        """
        Find optimal number of clusters using dendrogram analysis.

        Args:
            data: Input data for clustering

        Returns:
            Optimal number of clusters
        """
        if self.n_clusters is not None:
            return self.n_clusters

        # Use distance threshold if provided
        if self.distance_threshold is not None:
            model = AgglomerativeClustering(
                n_clusters=None,
                linkage=self.linkage_method,
                distance_threshold=self.distance_threshold
            )
            labels = model.fit_predict(data)
            return len(np.unique(labels))

        # Otherwise use silhouette analysis
        best_k = 2
        best_score = -1

        k_range = range(2, min(self.max_clusters + 1, len(data) // 2))

        for k in k_range:
            model = AgglomerativeClustering(n_clusters=k, linkage=self.linkage_method)
            labels = model.fit_predict(data)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        logger.info(
            "Optimal hierarchical clusters found",
            optimal_k=best_k,
            best_silhouette=best_score
        )

        return best_k

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'HierarchicalClusterer':
        """
        Fit hierarchical clustering model.

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        start_time = time.time()

        try:
            processed_data = self._preprocess_data(data)

            if len(processed_data) < 2:
                raise UnsupervisedModelError("Need at least 2 data points for clustering")

            # Find optimal clusters
            self.optimal_k = self._find_optimal_clusters_hierarchical(processed_data)

            # Fit model
            self.model = AgglomerativeClustering(
                n_clusters=self.optimal_k,
                linkage=self.linkage_method
            )

            self.labels_ = self.model.fit_predict(processed_data)

            # Generate linkage matrix for analysis
            if len(processed_data) <= 1000:  # Only for smaller datasets due to memory
                distances = pdist(processed_data)
                self.linkage_matrix = linkage(distances, method=self.linkage_method)

            self.is_fitted = True

            duration = time.time() - start_time
            logger.info(
                "HierarchicalClusterer fitted successfully",
                n_clusters=self.optimal_k,
                linkage_method=self.linkage_method,
                duration_ms=duration * 1000,
                n_samples=len(processed_data)
            )

            return self

        except Exception as e:
            raise UnsupervisedModelError(f"Hierarchical clustering fitting failed: {str(e)}")

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Note: Hierarchical clustering doesn't naturally support prediction on new data.
        This implementation fits a new model with combined data.

        Args:
            data: Input data for prediction

        Returns:
            Cluster labels for new data
        """
        self._validate_fitted()

        try:
            processed_data = self._preprocess_data(data)

            # For hierarchical clustering, we need to refit with new data
            # This is a limitation of hierarchical clustering
            model = AgglomerativeClustering(
                n_clusters=self.optimal_k,
                linkage=self.linkage_method
            )

            labels = model.fit_predict(processed_data)

            logger.debug(
                "Hierarchical clustering prediction completed",
                n_samples=len(processed_data),
                unique_labels=len(np.unique(labels))
            )

            return labels

        except Exception as e:
            raise UnsupervisedModelError(f"Hierarchical clustering prediction failed: {str(e)}")

    def update(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Update model with new data (requires refitting).

        Args:
            data: New data to incorporate
        """
        logger.warning(
            "Hierarchical clustering requires full refitting for updates",
            recommendation="Consider using k-means for incremental updates"
        )
        # For hierarchical clustering, update means refit
        self.fit(data)


class DensityBasedDetector(BaseUnsupervisedModel):
    """
    DBSCAN-based density clustering for anomaly detection.

    Optimized for identifying market anomalies and outlier patterns
    with KD-tree optimization for performance.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean',
        algorithm: str = 'kd_tree',
        leaf_size: int = 30,
        lookback_period: int = 100,
        enable_scaling: bool = True
    ):
        """
        Initialize DBSCAN detector.

        Args:
            eps: Maximum distance between samples in a cluster
            min_samples: Minimum samples in a neighborhood for core point
            metric: Distance metric
            algorithm: Algorithm for nearest neighbors computation
            leaf_size: Leaf size for tree algorithms
            lookback_period: Number of periods for analysis
            enable_scaling: Whether to scale data
        """
        super().__init__(lookback_period, enable_scaling)

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size

        self.model = None
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None
        self.n_noise_ = None

        self.model_params.update({
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'algorithm': algorithm,
            'leaf_size': leaf_size
        })

        logger.info(
            "DensityBasedDetector initialized",
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm
        )

    def _auto_tune_eps(self, data: np.ndarray) -> float:
        """
        Auto-tune eps parameter using k-distance graph.

        Args:
            data: Input data

        Returns:
            Optimal eps value
        """
        if len(data) < self.min_samples:
            return self.eps

        # Use k-nearest neighbors to estimate optimal eps
        k = min(self.min_samples, len(data) - 1)

        nn = NearestNeighbors(
            n_neighbors=k,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric
        )
        nn.fit(data)

        distances, _ = nn.kneighbors(data)
        k_distances = distances[:, k-1]  # Distance to k-th nearest neighbor
        k_distances = np.sort(k_distances)

        # Find elbow in k-distance graph
        # Use simple gradient approach for efficiency
        gradients = np.gradient(k_distances)
        elbow_idx = np.argmax(gradients)

        optimal_eps = k_distances[elbow_idx]

        logger.info(
            "Auto-tuned eps parameter",
            original_eps=self.eps,
            optimal_eps=optimal_eps,
            k_value=k
        )

        return optimal_eps

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'DensityBasedDetector':
        """
        Fit DBSCAN model to training data.

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        start_time = time.time()

        try:
            processed_data = self._preprocess_data(data)

            if len(processed_data) < self.min_samples:
                raise UnsupervisedModelError(f"Need at least {self.min_samples} data points for DBSCAN")

            # Auto-tune eps if dataset is not too large
            eps_to_use = self.eps
            if len(processed_data) <= 1000:  # Only for smaller datasets
                eps_to_use = self._auto_tune_eps(processed_data)

            # Fit DBSCAN
            self.model = DBSCAN(
                eps=eps_to_use,
                min_samples=self.min_samples,
                metric=self.metric,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size
            )

            self.labels_ = self.model.fit_predict(processed_data)
            self.core_sample_indices_ = self.model.core_sample_indices_

            # Calculate cluster statistics
            unique_labels = np.unique(self.labels_)
            self.n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
            self.n_noise_ = list(self.labels_).count(-1)

            self.is_fitted = True

            duration = time.time() - start_time
            logger.info(
                "DensityBasedDetector fitted successfully",
                n_clusters=self.n_clusters_,
                n_noise_points=self.n_noise_,
                eps_used=eps_to_use,
                duration_ms=duration * 1000,
                n_samples=len(processed_data)
            )

            return self

        except Exception as e:
            raise UnsupervisedModelError(f"DBSCAN fitting failed: {str(e)}")

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Note: DBSCAN doesn't naturally support prediction. This implementation
        assigns new points to nearest cluster or marks as outliers.

        Args:
            data: Input data for prediction

        Returns:
            Cluster labels (-1 for outliers)
        """
        self._validate_fitted()

        try:
            processed_data = self._preprocess_data(data)

            # For new points, use nearest neighbor approach to existing clusters
            if hasattr(self.model, 'components_'):
                # Use core samples for nearest neighbor prediction
                nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
                nn.fit(self.model.components_)

                distances, indices = nn.kneighbors(processed_data)

                # Assign labels based on nearest core sample
                core_labels = self.labels_[self.core_sample_indices_]
                predicted_labels = core_labels[indices.flatten()]

                # Mark as outliers if too far from any cluster
                outlier_mask = distances.flatten() > self.eps * 2
                predicted_labels[outlier_mask] = -1

                logger.debug(
                    "DBSCAN prediction completed",
                    n_samples=len(processed_data),
                    n_outliers=np.sum(outlier_mask)
                )

                return predicted_labels
            else:
                # No core samples found, mark all as outliers
                return np.full(len(processed_data), -1)

        except Exception as e:
            raise UnsupervisedModelError(f"DBSCAN prediction failed: {str(e)}")

    def update(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Update model with new data (requires refitting).

        Args:
            data: New data to incorporate
        """
        logger.warning(
            "DBSCAN requires full refitting for updates",
            recommendation="Consider using incremental density estimation"
        )
        # For DBSCAN, update means refit
        self.fit(data)

    def get_anomaly_scores(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get anomaly scores for data points.

        Args:
            data: Input data

        Returns:
            Anomaly scores (higher means more anomalous)
        """
        self._validate_fitted()

        try:
            processed_data = self._preprocess_data(data)
            labels = self.predict(processed_data)

            # Calculate anomaly scores based on distance to nearest cluster
            scores = np.zeros(len(processed_data))

            if hasattr(self.model, 'components_'):
                nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
                nn.fit(self.model.components_)
                distances, _ = nn.kneighbors(processed_data)

                # Normalize distances to [0, 1] range
                max_distance = np.max(distances) if np.max(distances) > 0 else 1
                scores = distances.flatten() / max_distance

                # Outliers get maximum score
                scores[labels == -1] = 1.0
            else:
                # No clusters found, all points are anomalous
                scores.fill(1.0)

            return scores

        except Exception as e:
            raise UnsupervisedModelError(f"Anomaly score calculation failed: {str(e)}")